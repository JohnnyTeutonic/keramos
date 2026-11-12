"""A trainable interpreter of keramos documents in PyTorch (CPU), for the ops the search uses.

It follows tools/ir_numpy.py op by op on batches ([B, T, width] values) and is checked against it in the
tests (same weights, logits within 1e-4). Ops: embedding, pos_embedding, linear, scale, rmsnorm (group,
weight_offset), layernorm, rope (interleaved, neox, n_rot, rope_heads all or first, kv_heads), attention
(causal, window, sinks, grouped kv, softcap), conv1d_causal, gelu, silu, relu, sigmoid, add, sub, mul,
affine, split; per_layer attributes, weight_index and "{i}" names. Anything else raises Unsupported.
"""

import math

import numpy as np
import torch

from . import fingerprint as fp


class Unsupported(ValueError):
    pass


SUPPORTED = {"embedding", "pos_embedding", "linear", "scale", "rmsnorm", "layernorm", "rope", "attention",
             "conv1d_causal", "gelu", "silu", "relu", "sigmoid", "add", "sub", "mul", "affine", "split"}


def supported(doc):
    ops = {n.get("op") for s in ("embed", "block", "head") for n in (doc.get(s) or {}).get("nodes", [])}
    return ops <= SUPPORTED and not any("apply" in n for n in doc["block"]["nodes"])


def init_params(doc, given=None, seed=0):
    """Parameters initialised as slip's `run --random` (name-keyed streams), as torch leaf tensors."""
    W = fp.random_weights(doc, given, seed=seed, keyed="name")
    return {k: torch.tensor(np.asarray(v), dtype=torch.float32, requires_grad=True) for k, v in W.items()}


def _per_layer(nd, i):
    if not any(isinstance(v, dict) and "per_layer" in v for v in nd.values()):
        return nd
    return {k: v["per_layer"][i] if isinstance(v, dict) and "per_layer" in v else v for k, v in nd.items()}


def _rope(x, pos, heads, hd, n_rot, theta, pairing, freq_scale=1.0):
    B, T, _ = x.shape
    t = x.reshape(B, T, heads, hd)
    half = n_rot // 2
    j = np.arange(half)
    ang = (pos[:, None].astype(np.float64) * float(freq_scale)) * (float(theta) ** (-2.0 * j / n_rot))[None, :]
    c = torch.tensor(np.cos(ang).astype(np.float32))[None, :, None, :]
    s = torch.tensor(np.sin(ang).astype(np.float32))[None, :, None, :]
    if pairing == "neox":
        a, b = t[..., :half], t[..., half:n_rot]
        ra, rb = a * c - b * s, a * s + b * c
        out = torch.cat([ra, rb, t[..., n_rot:]], dim=-1)
    elif pairing == "interleaved":
        a, b = t[..., 0:n_rot:2], t[..., 1:n_rot:2]
        ra, rb = a * c - b * s, a * s + b * c
        rot = torch.stack([ra, rb], dim=-1).reshape(B, T, heads, n_rot)
        out = torch.cat([rot, t[..., n_rot:]], dim=-1)
    else:
        raise Unsupported(f"rope pairing {pairing}")
    return out.reshape(B, T, heads * hd)


def run_nodes(nodes, env, P, i=None, wi=None):
    ti = i if wi is None else wi

    def name(key):
        return key.replace("{i}", str(ti)) if ti is not None else key

    def w(key):
        return P[name(key)]

    for nd in nodes:
        nd = _per_layer(nd, i)
        op, ins = nd["op"], nd["in"]
        x = env[ins[0]]
        if op == "embedding":
            y = w(nd["weight"])[x]
        elif op == "pos_embedding":
            pos = env["pos"]
            y = x + w(nd["weight"])[torch.as_tensor(pos)][None]
        elif op == "linear":
            wt = P.get(name(nd["weight"]))
            if wt is None:
                wt = P[nd["tied_to"]]
            y = x @ wt.T
            if nd.get("bias"):
                y = y + w(nd["bias"])
        elif op == "scale":
            y = x * w(nd["weight"]).reshape(-1)
            if nd.get("bias"):
                y = y + w(nd["bias"]).reshape(-1)
        elif op == "layernorm":
            mu = x.mean(dim=-1, keepdim=True)
            var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
            y = (x - mu) / torch.sqrt(var + nd["eps"]) * w(nd["weight"]) + w(nd["bias"])
        elif op == "rmsnorm":
            g = nd.get("group") or x.shape[-1]
            Bn, T, Wd = x.shape
            t = x.reshape(Bn, T, Wd // g, g)
            wt = w(nd["weight"])
            if nd.get("weight_offset"):
                wt = wt + nd["weight_offset"]
            t = t / torch.sqrt((t * t).mean(dim=-1, keepdim=True) + nd["eps"]) * wt.reshape(-1)
            y = t.reshape(Bn, T, Wd)
        elif op == "conv1d_causal":
            K = nd["kernel"]
            Bn, T, C = x.shape
            wt = w(nd["weight"]).reshape(C, K)
            xx = torch.cat([x.new_zeros(Bn, K - 1, C), x], dim=1)
            y = sum(wt[:, k] * xx[:, k:k + T] for k in range(K))
            if nd.get("bias"):
                y = y + w(nd["bias"]).reshape(-1)
        elif op == "gelu":
            if nd["approx"] == "tanh":
                y = torch.nn.functional.gelu(x, approximate="tanh")
            else:
                y = torch.nn.functional.gelu(x)
        elif op == "silu":
            y = x * torch.sigmoid(x)
        elif op == "relu":
            y = torch.relu(x)
        elif op == "sigmoid":
            y = torch.sigmoid(x)
        elif op in ("add", "sub", "mul"):
            o = env[ins[1]]
            y = x + o if op == "add" else x - o if op == "sub" else x * o
        elif op == "affine":
            y = x * nd["scale"] + nd["shift"]
        elif op == "split":
            outs, at = [], 0
            for s in nd["sizes"]:
                outs.append(x[..., at:at + s])
                at += s
            for o, v in zip(nd["out"], outs):
                env[o] = v
            continue
        elif op == "rope":
            if nd.get("freq_factors") is not None or nd.get("attn_factor", 1.0) != 1.0:
                raise Unsupported("rope with freq_factors or attn_factor")
            H, hd, pos = nd["heads"], nd["head_dim"], env["pos"]
            n_rot = nd.get("n_rot") or hd
            for o, src, heads in zip(nd["out"], ins[:2], (H, nd.get("kv_heads") or H)):
                v = env[src]
                if nd["rope_heads"] == "first":
                    r = _rope(v[..., :hd], pos, 1, hd, n_rot, nd["theta"], nd["pairing"], nd.get("freq_scale", 1.0))
                    env[o] = torch.cat([r, v[..., hd:]], dim=-1)
                else:
                    env[o] = _rope(v, pos, heads, hd, n_rot, nd["theta"], nd["pairing"], nd.get("freq_scale", 1.0))
            continue
        elif op == "attention":
            if nd.get("mask", "causal") != "causal" or nd.get("position_bias") or nd.get("relative_kv"):
                raise Unsupported("attention with mask, position_bias or relative_kv")
            q, k, v = (env[n] for n in ins)
            H, hd, KV = nd["heads"], nd["head_dim"], nd["kv_heads"]
            Bn, T, _ = q.shape
            pos = torch.as_tensor(env["pos"])
            p, j = pos[:, None], pos[None, :]
            mask = j > p
            if nd["window"]:
                mask = mask | ((j >= nd["sinks"]) & (j < p - nd["window"] + 1))
            qh = q.reshape(Bn, T, H, hd).transpose(1, 2)
            kh = k.reshape(Bn, T, KV, hd).transpose(1, 2)
            vh = v.reshape(Bn, T, KV, hd).transpose(1, 2)
            if KV != H:
                kh = kh.repeat_interleave(H // KV, dim=1)
                vh = vh.repeat_interleave(H // KV, dim=1)
            s = (qh @ kh.transpose(-1, -2)) * nd["scale"]
            if nd.get("softcap") is not None:
                c = nd["softcap"]
                s = torch.tanh(s / c) * c
            s = s.masked_fill(mask[None, None], float("-inf"))
            a = torch.softmax(s, dim=-1)
            y = (a @ vh).transpose(1, 2).reshape(Bn, T, H * hd)
        else:
            raise Unsupported(f"no torch kernel for op {op}")
        env[nd["out"]] = y


def forward(doc, P, tokens):
    """tokens: LongTensor [B, T] -> logits [B, T, vocab]."""
    if not supported(doc):
        raise Unsupported("document uses ops outside torch_exec.SUPPORTED")
    T = tokens.shape[1]
    env = {"tokens": tokens, "pos": np.arange(T)}
    run_nodes(doc["embed"]["nodes"], env, P)
    blk = doc["block"]
    wmap = blk.get("weight_index")
    for i in range(blk["repeat"]):
        wi = wmap["per_layer"][i] if wmap else i
        run_nodes(blk["nodes"], env, P, i, wi)
    run_nodes(doc["head"]["nodes"], env, P)
    return env["logits"]


def n_params(P):
    return int(sum(p.numel() for p in P.values()))


def seconds_per_step(doc, P, B, T):
    import time
    toks = torch.zeros(B, T, dtype=torch.long)
    t0 = time.time()
    forward(doc, P, toks).sum().backward()
    return time.time() - t0


__all__ = ["Unsupported", "SUPPORTED", "supported", "init_params", "forward", "n_params", "math"]
