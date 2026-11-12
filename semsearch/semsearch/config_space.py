"""A paperkiln-style configuration space and its lowering to keramos documents.

The keys mirror paperkiln's `LlamaConfig` (microtorch/include/microtorch/llama.hpp: n_heads, n_kv_heads with
0 meaning n_heads, d_ff, rope_theta, rope_pairing, rope_dims with 0 meaning head_dim, qk_norm,
tie_embeddings, the recursion block with `repeats` and `inject`), its sliding-window attention (window,
sinks), and one implementation-only key it really has (`checkpoint_blocks`, activation checkpointing,
which changes memory and not the function). The lowering writes the document the way paperkiln's
exporter writes pk_llama (same node order and value names; checked in the tests against the conformance
model's embedded document), so a config search over these dicts is a search over documents of the kind
paperkiln emits.

Fixed for the experiments: vocab 32, d 32, ctx 32.
"""

import itertools
import math
import random

VOCAB, D, CTX = 32, 32, 32
EPS = 1e-05

SPACE = {
    "n_layers": [1, 2, 3],
    "n_heads": [1, 2, 4],
    "n_kv_heads": [0, 1, 2, 4],
    "d_ff": [64, 128],
    "rope_theta": [10000.0, 100000.0],
    "rope_pairing": ["interleaved", "neox"],
    "rope_dims": [0, 4, 8],
    "qk_norm": [False, True],
    "tie_embeddings": [True, False],
    "window": [0, 8, 16, 32],
    "sinks": [0, 2],
    "repeats": [1, 2],
    "inject": ["none", "add"],
    "checkpoint_blocks": [False, True],
}

# The space the search demonstration uses: wider value lists, the same keys.
SEARCH_SPACE = {
    "n_layers": [1, 2, 3, 4],
    "n_heads": [1, 2, 4, 8],
    "n_kv_heads": [0, 1, 2, 4, 8],
    "d_ff": [32, 64, 96, 128],
    "rope_theta": [10000.0, 100000.0],
    "rope_pairing": ["interleaved", "neox"],
    "rope_dims": [0, 4, 8, 16],
    "qk_norm": [False, True],
    "tie_embeddings": [True, False],
    "window": [0, 4, 8, 16, 24, 32, 64],
    "sinks": [0, 2],
    "repeats": [1, 2],
    "inject": ["none", "add"],
    "checkpoint_blocks": [False, True],
}

ARCH_ONLY = [k for k in SPACE if k != "checkpoint_blocks"]


class InvalidConfig(ValueError):
    pass


def default_config():
    return {"n_layers": 2, "n_heads": 2, "n_kv_heads": 0, "d_ff": 64, "rope_theta": 10000.0,
            "rope_pairing": "interleaved", "rope_dims": 0, "qk_norm": False, "tie_embeddings": True,
            "window": 0, "sinks": 0, "repeats": 1, "inject": "none", "checkpoint_blocks": False}


def check(cfg, d=D):
    H = cfg["n_heads"]
    if d % H:
        raise InvalidConfig(f"n_heads {H} does not divide d {d}")
    hd = d // H
    kv = cfg["n_kv_heads"] or H
    if H % kv:
        raise InvalidConfig(f"n_kv_heads {kv} does not divide n_heads {H}")
    rd = cfg["rope_dims"]
    if rd and (rd > hd or rd % 2):
        raise InvalidConfig(f"rope_dims {rd} is not an even number up to head_dim {hd}")
    if cfg["window"] and cfg["sinks"] >= cfg["window"]:
        raise InvalidConfig("sinks must be fewer than the window")
    return hd, kv


def lower(cfg, vocab=VOCAB, d=D, ctx=CTX):
    """(document, given shapes): the keramos document paperkiln would export for this config, and the
    shapes the graph leaves free (the FFN's hidden width), keyed by tensor name."""
    hd, kv = check(cfg, d)
    H, L, R = cfg["n_heads"], cfg["n_layers"], cfg["repeats"]
    B = "blk.{i}."
    win, sinks = cfg["window"], cfg["sinks"] if cfg["window"] else 0
    recursion = R > 1 or cfg["inject"] != "none"
    apps = L * R
    embed = [{"op": "embedding", "in": ["tokens"], "weight": "token_embd.weight", "out": "x"}]
    nodes = []
    if recursion and cfg["inject"] == "add":
        embed.append({"op": "embedding", "in": ["tokens"], "weight": "token_embd.weight", "out": "e"})
        # paperkiln: the embedding is added before each repeat after the first
        scale = [1.0 if (a % L == 0 and a >= L) else 0.0 for a in range(apps)]
        nodes += [{"op": "affine", "in": ["e"], "scale": {"per_layer": scale}, "shift": 0.0, "out": "inj"},
                  {"op": "add", "in": ["x", "inj"], "out": "x"}]
    nodes += [
        {"op": "rmsnorm", "in": ["x"], "weight": B + "attn_norm.weight", "eps": EPS, "out": "h"},
        {"op": "linear", "in": ["h"], "weight": B + "attn_q.weight", "bias": None, "out": "q"},
        {"op": "linear", "in": ["h"], "weight": B + "attn_k.weight", "bias": None, "out": "k"},
        {"op": "linear", "in": ["h"], "weight": B + "attn_v.weight", "bias": None, "out": "v"},
    ]
    if cfg["qk_norm"]:
        nodes += [{"op": "rmsnorm", "in": ["q"], "weight": B + "attn_q_norm.weight", "eps": EPS, "group": hd,
                   "out": "q"},
                  {"op": "rmsnorm", "in": ["k"], "weight": B + "attn_k_norm.weight", "eps": EPS, "group": hd,
                   "out": "k"}]
    rope = {"op": "rope", "in": ["q", "k", "pos"], "heads": H, "kv_heads": kv, "head_dim": hd,
            "theta": float(cfg["rope_theta"]), "rope_heads": "all", "pairing": cfg["rope_pairing"],
            "out": ["q", "k"]}
    if cfg["rope_dims"]:
        rope["n_rot"] = cfg["rope_dims"]
    nodes.append(rope)
    nodes += [
        {"op": "attention", "in": ["q", "k", "v"], "state": "kv", "heads": H, "kv_heads": kv, "head_dim": hd,
         "causal": True, "scale": float(f"{1.0 / math.sqrt(hd):.8g}"), "window": win, "sinks": sinks,
         "softmax": "max-shifted, masked entries exactly zero", "out": "a"},
        {"op": "linear", "in": ["a"], "weight": B + "attn_output.weight", "bias": None, "out": "o"},
        {"op": "add", "in": ["x", "o"], "out": "x"},
        {"op": "rmsnorm", "in": ["x"], "weight": B + "ffn_norm.weight", "eps": EPS, "out": "h2"},
        {"op": "linear", "in": ["h2"], "weight": B + "ffn_gate.weight", "bias": None, "out": "g"},
        {"op": "linear", "in": ["h2"], "weight": B + "ffn_up.weight", "bias": None, "out": "u"},
        {"op": "silu", "in": ["g"], "out": "g"},
        {"op": "mul", "in": ["g", "u"], "out": "gu"},
        {"op": "linear", "in": ["gu"], "weight": B + "ffn_down.weight", "bias": None, "out": "f"},
        {"op": "add", "in": ["x", "f"], "out": "x"},
    ]
    head_lin = {"op": "linear", "in": ["h"], "weight": "output.weight", "bias": None, "out": "logits"}
    if cfg["tie_embeddings"]:
        head_lin["tied_to"] = "token_embd.weight"
    block = {"repeat": apps,
             "state": [{"name": "kv", "kind": "kv", "heads": H, "kv_heads": kv, "head_dim": hd, "max_len": ctx,
                        "window": win, "sinks": sinks, "dtype": "f32"}],
             "nodes": nodes}
    if recursion:
        block["weight_index"] = {"per_layer": [a % L for a in range(apps)]}
    doc = {
        "_comment": "lowered by semsearch.config_space from a paperkiln-style config",
        "ir_version": 1,
        "family_hint": "llama_recursive" if recursion else "llama",
        "model": {"vocab": vocab, "d": d, "n_layers": apps, "ctx": ctx, "dtype": "f32"},
        "embed": {"nodes": embed},
        "block": block,
        "head": {"nodes": [{"op": "rmsnorm", "in": ["x"], "weight": "output_norm.weight", "eps": EPS, "out": "h"},
                           head_lin]},
    }
    ff = cfg["d_ff"]
    given = {B + "ffn_gate.weight": [ff, d], B + "ffn_up.weight": [ff, d], B + "ffn_down.weight": [d, ff]}
    return doc, given


def key(cfg):
    """The config's own identity: the dict, as a search over configs caches it."""
    return tuple(sorted(cfg.items()))


def grid(space=SPACE):
    keys = list(space)
    for vals in itertools.product(*(space[k] for k in keys)):
        yield dict(zip(keys, vals))


def grid_size(space=SPACE):
    n = 1
    for v in space.values():
        n *= len(v)
    return n


def random_config(rng, space=SEARCH_SPACE, valid_only=True):
    while True:
        cfg = {k: rng.choice(v) for k, v in space.items()}
        if not valid_only:
            return cfg
        try:
            check(cfg)
            return cfg
        except InvalidConfig:
            continue


def mutate(cfg, rng, space=SEARCH_SPACE):
    """One config-space move: one key, set to a different value of its list. The result may be invalid."""
    c = dict(cfg)
    k = rng.choice(list(space))
    c[k] = rng.choice([v for v in space[k] if v != cfg[k]])
    return c, k


def crossover(a, b, rng, space=SEARCH_SPACE):
    """Uniform crossover of two configs, key by key."""
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in space}


__all__ = ["SPACE", "SEARCH_SPACE", "InvalidConfig", "default_config", "check", "lower", "key", "grid",
           "grid_size", "random_config", "mutate", "crossover", "random"]
