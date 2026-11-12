"""Typed edits of keramos documents, on an SSA form of the graph.

A genome is (document, given): a keramos document and the shapes of the tensors whose sizes the graph
leaves free (an FFN's hidden width), keyed by tensor name, as slip's width solver takes them.

Edits work on `SSA`, a graph whose every value has one producer (built from canon.Graph), and are written
back as a document by `SSA.emit`, which orders each section topologically and names the values. The edit
types, each a graph difference rather than a config key:

  set_attr     change one semantic attribute of one node: attention window and sinks (with the slot),
               attention softcap, rope theta and n_rot, gelu approx
  heads        change an attention's heads, kv_heads and head_dim together (rope, slot, q/k norm group,
               scale 1/sqrt(head_dim)), the projection widths following through slip's solver
  swap_op      replace an op by another of its type: an activation (silu, gelu, relu) or a norm
               (rmsnorm, layernorm)
  qk_norm      insert or delete a per-head rmsnorm on q and k before RoPE
  insert       insert a pre-norm residual sublayer from the library (attention, SwiGLU FFN, GELU MLP,
               causal depthwise convolution) at a point of the residual stream
  delete       delete a residual sublayer (its add, and every node only it read)
  rewire       make two consecutive sublayers parallel (both read the stream before the first) or
               sequential (the second reads the first's output)
  repeat       one more or one fewer block application
  tie          tie or untie the output projection to the token embedding
  width        set the hidden width of one FFN-like sublayer (a free size, so a `given` shape)

and `crossover(a, b)`: the child is a's block up to a cut followed by b's sublayers from a cut, b's
tensors renamed so that none is shared by accident.

Every result is checked by keramos's validator and slip's width solver (`check`); an edit that cannot
apply returns None. `propose` draws edits until one gives a valid document whose canonical hash (at the
given level) differs from the parent's, so a semantic search never spends an evaluation on a no-op.
"""

import copy
import math
import random

from . import bridge, canon, torch_exec

WINDOWS = (0, 2, 4, 8, 16, 24)
THETAS = (1000.0, 10000.0, 100000.0, 1000000.0)
WIDTHS = (32, 64, 96, 128)
ACTS = ("silu", "gelu", "relu")
LIBRARY = ("attn", "swiglu", "mlp", "conv")
EDIT_TYPES = ("set_attr", "heads", "swap_op", "qk_norm", "insert", "delete", "rewire", "repeat", "tie", "width")


class SSA:
    """nodes: list of {"sect", "raw" (attributes, no in/out), "ins": [vid], "outs": [vid], "multi": bool}.
    vids: "tokens", "pos", ("carry", name) for the block's read of a carried name, ints for node outputs.
    carry_out[name]: the vid the block's last write of a carried name; embed_out[name]: the embed's.
    slots: {slot name: slot dict}."""

    def __init__(self):
        self.nodes = []
        self.carry_out = {}
        self.embed_out = {}
        self.slots = {}
        self.top = {}
        self.block_attrs = {}
        self.logits = None
        self._next = 0

    def fresh(self):
        self._next += 1
        return self._next

    @classmethod
    def from_doc(cls, doc):
        A = canon.normalise(doc)
        g = canon.Graph(A, hash_tensors=False)
        s = cls()
        s.top = {k: copy.deepcopy(v) for k, v in A.items() if k not in ("embed", "block", "head")}
        s.block_attrs = {k: copy.deepcopy(v) for k, v in A["block"].items() if k not in ("nodes", "state")}
        s.slots = {k: copy.deepcopy(v) for k, v in g.slots.items()}
        vmap = {}

        def vid(v):
            if v in ("tokens", "pos"):
                return v
            if v[0] == "carry":
                return v
            if v[0] == "final":
                return vid(g.block_final[v[1]])
            if v[0] == "n":
                if v not in vmap:
                    vmap[v] = s.fresh()
                return vmap[v]
            raise ValueError(f"cannot edit a graph that reads {v}")

        for i in sorted(g.reach_nodes):
            nd = g.nodes[i]
            if nd.phi or nd.op in canon.CONTROL_OPS:
                raise ValueError("per-layer apply and depth_route are not edited")
            raw = {k: copy.deepcopy(v) for k, v in nd.raw.items() if k not in ("in", "out")}
            s.nodes.append({"sect": nd.sect, "raw": raw, "ins": [vid(v) for v in nd.ins],
                            "outs": [vid(v) for v in nd.outs], "multi": isinstance(nd.raw.get("out"), list)})
        for name in g.carried:
            s.carry_out[name] = vid(g.block_final[name])
            s.embed_out[name] = vid(g.embed_env[name])
        s.logits = vid(g.logits)
        return s

    def copy(self):
        return copy.deepcopy(self)

    # ---- queries
    def producer(self, v):
        for i, n in enumerate(self.nodes):
            if v in n["outs"]:
                return i
        return None

    def users(self, v):
        return [i for i, n in enumerate(self.nodes) if v in n["ins"]]

    def block(self):
        return [i for i, n in enumerate(self.nodes) if n["sect"] == "block"]

    def tensor_names(self):
        out = set()
        for n in self.nodes:
            for _, name in bridge.tools()._tensor_names(n["raw"]):
                out.add(name)
        return out

    def stream(self, name="x"):
        """The residual stream's versions [x0, x1, ..., xn] (x0 = the block's read of x, xn its last write)
        and the adds [a1, ..., an] that make them, with the branch value each adds."""
        start = ("carry", name)
        end = self.carry_out.get(name)
        if end is None:
            return [start], [], []
        memo = {}

        def reaches(v):
            if v == start:
                return True
            if v in memo:
                return memo[v]
            p = self.producer(v) if isinstance(v, int) else None
            r = p is not None and self.nodes[p]["raw"]["op"] == "add" and self.nodes[p]["sect"] == "block" and \
                any(reaches(u) for u in self.nodes[p]["ins"])
            memo[v] = r
            return r

        versions, adds, branches = [end], [], []
        v = end
        while v != start:
            p = self.producer(v)
            if p is None or not reaches(v):
                return None
            ins = self.nodes[p]["ins"]
            si = 0 if reaches(ins[0]) else 1
            adds.append(p)
            branches.append(ins[1 - si])
            v = ins[si]
            versions.append(v)
        return versions[::-1], adds[::-1], branches[::-1]

    def cone(self, v, stop):
        """Block nodes v depends on, not crossing the values in stop (stream versions) or leaving the block."""
        out, stack = set(), [v]
        while stack:
            u = stack.pop()
            if u in stop or not isinstance(u, int):
                continue
            p = self.producer(u)
            if p is None or p in out or self.nodes[p]["sect"] != "block":
                continue
            out.add(p)
            stack.extend(self.nodes[p]["ins"])
        return out

    def sublayers(self):
        """[(add index, branch vid, cone node indices, stream version read)] in stream order."""
        st = self.stream()
        if st is None:
            return []
        versions, adds, branches = st
        stop = set(versions)
        out = []
        for k, (a, b) in enumerate(zip(adds, branches)):
            c = self.cone(b, stop)
            reads = sorted({u for i in c for u in self.nodes[i]["ins"] if u in stop}, key=versions.index)
            out.append((a, b, c, reads))
        return out

    # ---- edits
    def redirect(self, old, new, skip=()):
        for i, n in enumerate(self.nodes):
            if i in skip:
                continue
            n["ins"] = [new if u == old else u for u in n["ins"]]
        for k, v in self.carry_out.items():
            if v == old:
                self.carry_out[k] = new
        if self.logits == old:
            self.logits = new

    def add_node(self, sect, raw, ins, n_out=1, multi=False):
        outs = [self.fresh() for _ in range(n_out)]
        self.nodes.append({"sect": sect, "raw": raw, "ins": list(ins), "outs": outs, "multi": multi})
        return outs if multi else outs[0]

    def prune(self):
        """Drop nodes nothing reachable reads (from logits through the carried values)."""
        need, stack = set(), [self.logits]
        prod = {}
        for i, n in enumerate(self.nodes):
            for v in n["outs"]:
                prod[v] = i
        while stack:
            v = stack.pop()
            if isinstance(v, tuple) and v[0] == "carry":
                stack.append(self.carry_out.get(v[1]))
                stack.append(self.embed_out.get(v[1]))
                continue
            p = prod.get(v)
            if p is None or p in need:
                continue
            need.add(p)
            stack.extend(self.nodes[p]["ins"])
        self.nodes = [n for i, n in enumerate(self.nodes) if i in need]
        used = {n["raw"].get("state") for n in self.nodes}
        self.slots = {k: v for k, v in self.slots.items() if k in used}
        for k in list(self.carry_out):
            if ("carry", k) not in {u for n in self.nodes for u in n["ins"]}:
                # the block no longer reads the carried name before writing it: still carried by the head
                pass

    # ---- writing back
    def emit(self):
        self.prune()
        names = {"tokens": "tokens", "pos": "pos"}
        for k, v in self.carry_out.items():
            names[v] = k
        for k, v in self.embed_out.items():
            names[v] = k
        names[self.logits] = "logits"
        counter = [0]
        reserved = set(self.carry_out) | {"logits", "tokens", "pos"}

        def fresh():
            while True:
                counter[0] += 1
                nm = f"v{counter[0]}"
                if nm not in reserved:
                    return nm

        prod = {}
        for i, n in enumerate(self.nodes):
            for v in n["outs"]:
                prod[v] = i
        sections = {}
        for sect in ("embed", "block", "head"):
            idx = [i for i, n in enumerate(self.nodes) if n["sect"] == sect]
            deps = {i: {prod[u] for u in self.nodes[i]["ins"] if u in prod and prod[u] in idx} for i in idx}
            if sect == "block":
                for k, v in self.carry_out.items():
                    if v in prod and prod[v] in deps:
                        for i in idx:
                            if ("carry", k) in self.nodes[i]["ins"] and i != prod[v]:
                                deps[prod[v]].add(i)
            order, done = [], set()
            while len(order) < len(idx):
                ready = [i for i in idx if i not in done and deps[i] <= done]
                if not ready:
                    raise ValueError(f"{sect}: the edit made a cycle")
                i = ready[0]
                order.append(i)
                done.add(i)
            sections[sect] = order
        out = copy.deepcopy(self.top)
        slot_names = {}
        docsec = {}
        for sect in ("embed", "block", "head"):
            nodes = []
            for i in sections[sect]:
                n = self.nodes[i]
                raw = copy.deepcopy(n["raw"])
                ins = []
                for u in n["ins"]:
                    if isinstance(u, tuple):
                        ins.append(u[1])
                    else:
                        ins.append(names[u])
                outs = []
                for v in n["outs"]:
                    if v not in names:
                        names[v] = fresh()
                    outs.append(names[v])
                node = {"op": raw.pop("op"), "in": ins}
                node.update(raw)
                node["out"] = outs if n["multi"] else outs[0]
                if "state" in node:
                    old = node["state"]
                    if old not in slot_names:
                        slot_names[old] = f"s{len(slot_names)}"
                    node["state"] = slot_names[old]
                nodes.append(node)
            docsec[sect] = nodes
        out["embed"] = {"nodes": docsec["embed"]}
        blk = copy.deepcopy(self.block_attrs)
        state = []
        for old, new in slot_names.items():
            sl = copy.deepcopy(self.slots[old])
            sl["name"] = new
            state.append(sl)
        blk["state"] = state
        blk["nodes"] = docsec["block"]
        out["block"] = blk
        out["head"] = {"nodes": docsec["head"]}
        return out


# ------------------------------------------------------------------ helpers

def _model(s):
    return s.top["model"]


def _fresh_tag(s, base):
    names = s.tensor_names()
    k = 0
    while any(f".{base}{k}_" in t or t.startswith(f"{base}{k}_") for t in names):
        k += 1
    return f"{base}{k}"


def _attn_dims(s):
    for n in s.nodes:
        if n["raw"]["op"] == "attention":
            return n["raw"]["heads"], n["raw"]["kv_heads"], n["raw"]["head_dim"]
    d = _model(s)["d"]
    return 2, 2, d // 2


def _new_slot(s, raw):
    k = 0
    while f"n{k}" in s.slots:
        k += 1
    raw["name"] = f"n{k}"
    s.slots[raw["name"]] = raw
    return raw["name"]


def _library(s, kind, v, rng, given):
    """Nodes of one pre-norm sublayer reading stream value v; returns the branch value."""
    d, ctx = _model(s)["d"], _model(s)["ctx"]
    B = "blk.{i}."
    tag = _fresh_tag(s, kind)
    h = s.add_node("block", {"op": "rmsnorm", "weight": B + tag + "_norm.weight", "eps": 1e-05}, [v])
    if kind == "attn":
        H, KV, hd = _attn_dims(s)
        q = s.add_node("block", {"op": "linear", "weight": B + tag + "_q.weight"}, [h])
        k = s.add_node("block", {"op": "linear", "weight": B + tag + "_k.weight"}, [h])
        vv = s.add_node("block", {"op": "linear", "weight": B + tag + "_v.weight"}, [h])
        rope = {"op": "rope", "heads": H, "head_dim": hd, "theta": 10000.0, "rope_heads": "all",
                "pairing": "interleaved"}
        if KV != H:
            rope["kv_heads"] = KV
        q2, k2 = s.add_node("block", rope, [q, k, "pos"], n_out=2, multi=True)
        slot = _new_slot(s, {"kind": "kv", "heads": H, "kv_heads": KV, "head_dim": hd, "max_len": ctx,
                             "window": 0, "sinks": 0})
        a = s.add_node("block", {"op": "attention", "state": slot, "heads": H, "kv_heads": KV, "head_dim": hd,
                                 "causal": True, "scale": float(f"{1 / math.sqrt(hd):.8g}"), "window": 0,
                                 "sinks": 0}, [q2, k2, vv])
        return s.add_node("block", {"op": "linear", "weight": B + tag + "_o.weight"}, [a])
    if kind in ("swiglu", "mlp"):
        ff = rng.choice(WIDTHS)
        up = s.add_node("block", {"op": "linear", "weight": B + tag + "_up.weight"}, [h])
        given[B + tag + "_up.weight"] = [ff, d]
        given[B + tag + "_down.weight"] = [d, ff]
        if kind == "swiglu":
            g = s.add_node("block", {"op": "linear", "weight": B + tag + "_gate.weight"}, [h])
            given[B + tag + "_gate.weight"] = [ff, d]
            g = s.add_node("block", {"op": "silu"}, [g])
            x = s.add_node("block", {"op": "mul"}, [g, up])
        else:
            x = s.add_node("block", {"op": "gelu", "approx": "tanh"}, [up])
        return s.add_node("block", {"op": "linear", "weight": B + tag + "_down.weight"}, [x])
    if kind == "conv":
        K = rng.choice((2, 3, 4))
        slot = _new_slot(s, {"kind": "recurrent", "op": "conv1d_causal", "dtype": "f32",
                             "tensors": [{"name": "x", "shape": [K - 1, d], "init": 0}]})
        c = s.add_node("block", {"op": "conv1d_causal", "state": slot, "weight": B + tag + "_conv.weight",
                                 "channels": d, "kernel": K}, [h])
        return s.add_node("block", {"op": "linear", "weight": B + tag + "_o.weight"}, [c])
    raise ValueError(kind)


def _clean_given(doc, given):
    """given restricted to the tensors the document names."""
    names = set()
    for sect in ("embed", "block", "head"):
        for n in doc[sect]["nodes"]:
            for _, t in bridge.tools()._tensor_names(n):
                names.add(t)
    return {k: v for k, v in given.items() if k in names}


def check(doc, given, trainable=True):
    """None when keramos accepts the document, slip solves every width with nothing left free, and (when
    trainable) the torch interpreter runs it; otherwise the first problem."""
    errs = bridge.doc_errors(doc)
    if errs:
        return "keramos: " + errs[0]
    werrs = bridge.width_errors(doc, given)
    if werrs:
        return "widths: " + str(werrs[0][2])
    shapes = canon.solved_shapes(doc, given)
    free = [t for t, s in shapes.items() if None in s]
    if free:
        return f"free sizes: {free[:3]}"
    if trainable and not torch_exec.supported(doc):
        return "torch: unsupported op"
    return None


# ------------------------------------------------------------------ the edits

def _attention_nodes(s):
    return [i for i, n in enumerate(s.nodes) if n["raw"]["op"] == "attention"]


def edit_set_attr(s, given, rng):
    ctx = _model(s)["ctx"]
    cands = []
    for i, n in enumerate(s.nodes):
        op = n["raw"]["op"]
        if op == "attention":
            cands += [(i, "window"), (i, "softcap")]
        elif op == "rope":
            cands += [(i, "theta"), (i, "n_rot")]
        elif op == "gelu":
            cands.append((i, "approx"))
    if not cands:
        return None
    i, key = rng.choice(cands)
    raw = s.nodes[i]["raw"]
    if key == "window":
        cur = raw.get("window", 0)
        w = rng.choice([x for x in WINDOWS if x != cur and x < ctx])
        sinks = rng.choice((0, 2)) if w > 2 else 0
        raw["window"], raw["sinks"] = w, sinks
        sl = s.slots[raw["state"]]
        sl["window"], sl["sinks"] = w, sinks
    elif key == "softcap":
        if "softcap" in raw:
            del raw["softcap"]
        else:
            raw["softcap"] = rng.choice((5.0, 20.0))
    elif key == "theta":
        raw["theta"] = rng.choice([t for t in THETAS if t != raw["theta"]])
    elif key == "n_rot":
        hd = raw["head_dim"]
        cur = raw.get("n_rot", hd)
        opts = sorted(({r for r in (2, 4, 8, 16, 32, 64) if r < hd} | {hd}) - {cur})
        if not opts:
            return None
        r = rng.choice(opts)
        if r == hd:
            raw.pop("n_rot", None)
        else:
            raw["n_rot"] = r
    elif key == "approx":
        raw["approx"] = "none" if raw["approx"] == "tanh" else "tanh"
    return f"set_attr {raw['op']}.{key}"


def edit_heads(s, given, rng):
    att = _attention_nodes(s)
    if not att:
        return None
    i = rng.choice(att)
    a = s.nodes[i]["raw"]
    width = a["heads"] * a["head_dim"]
    kv_width = a["kv_heads"] * a["head_dim"]
    if kv_width != width:   # keep it simple: GQA documents get heads changed with kv following ratio 1
        pass
    opts = []
    for H in (1, 2, 4, 8):
        if width % H:
            continue
        hd = width // H
        if hd % 2:
            continue
        for KV in (1, 2, 4, 8):
            if KV <= H and H % KV == 0 and (H, KV) != (a["heads"], a["kv_heads"]):
                opts.append((H, KV, hd))
    if not opts:
        return None
    H, KV, hd = rng.choice(opts)
    q, k, v = s.nodes[i]["ins"]
    # the rope that writes q and k, and per-head norms before it
    rp = s.producer(q)
    if rp is None or s.nodes[rp]["raw"]["op"] != "rope":
        return None
    r = s.nodes[rp]["raw"]
    r["heads"], r["head_dim"] = H, hd
    if KV != H:
        r["kv_heads"] = KV
    else:
        r.pop("kv_heads", None)
    if r.get("n_rot", hd) > hd or r.get("n_rot") == hd:
        r.pop("n_rot", None)
    for u in s.nodes[rp]["ins"][:2]:
        p = s.producer(u)
        if p is not None and s.nodes[p]["raw"]["op"] == "rmsnorm" and "group" in s.nodes[p]["raw"]:
            s.nodes[p]["raw"]["group"] = hd
    a["heads"], a["kv_heads"], a["head_dim"] = H, KV, hd
    a["scale"] = float(f"{1 / math.sqrt(hd):.8g}")
    sl = s.slots[a["state"]]
    sl["heads"], sl["kv_heads"], sl["head_dim"] = H, KV, hd
    return f"heads {H}/{KV}x{hd}"


def edit_swap_op(s, given, rng):
    cands = [i for i, n in enumerate(s.nodes) if n["raw"]["op"] in ACTS or
             (n["raw"]["op"] in ("rmsnorm", "layernorm") and "group" not in n["raw"])]
    if not cands:
        return None
    i = rng.choice(cands)
    raw = s.nodes[i]["raw"]
    op = raw["op"]
    if op in ACTS:
        new = rng.choice([a for a in ACTS if a != op])
        s.nodes[i]["raw"] = {"op": new, "approx": "tanh"} if new == "gelu" else {"op": new}
        return f"swap {op}->{new}"
    if op == "rmsnorm":
        w = raw["weight"]
        if not w.endswith(".weight"):
            return None
        s.nodes[i]["raw"] = {"op": "layernorm", "weight": w, "bias": w[:-len(".weight")] + ".bias", "eps": raw["eps"]}
        return "swap rmsnorm->layernorm"
    s.nodes[i]["raw"] = {"op": "rmsnorm", "weight": raw["weight"], "eps": raw["eps"]}
    return "swap layernorm->rmsnorm"


def edit_qk_norm(s, given, rng):
    ropes = [i for i, n in enumerate(s.nodes) if n["raw"]["op"] == "rope"]
    if not ropes:
        return None
    i = rng.choice(ropes)
    r = s.nodes[i]["raw"]
    ins = s.nodes[i]["ins"]
    prods = [s.producer(u) for u in ins[:2]]
    normed = [p is not None and s.nodes[p]["raw"]["op"] == "rmsnorm" and "group" in s.nodes[p]["raw"] for p in prods]
    if all(normed):
        for p, u in zip(prods, ins[:2]):
            s.redirect(u, s.nodes[p]["ins"][0])
        return "qk_norm delete"
    if any(normed):
        return None
    tag = _fresh_tag(s, "qkn")
    B = "blk.{i}." if s.nodes[i]["sect"] == "block" else ""
    new = []
    for which, u in zip(("q", "k"), ins[:2]):
        n = s.add_node(s.nodes[i]["sect"], {"op": "rmsnorm", "weight": f"{B}{tag}_{which}.weight", "eps": 1e-05,
                                            "group": r["head_dim"]}, [u])
        new.append(n)
    s.nodes[i]["ins"] = [new[0], new[1]] + ins[2:]
    return "qk_norm insert"


def edit_insert(s, given, rng):
    st = s.stream()
    if st is None:
        return None
    versions = st[0]
    k = rng.randrange(len(versions))
    v = versions[k]
    kind = rng.choice(LIBRARY)
    if k == len(versions) - 1:
        # after the last write of x: that write takes a fresh value, the new add writes x
        new_prev = s.fresh()
        p = s.producer(v)
        if p is None:
            return None
        s.nodes[p]["outs"] = [new_prev if o == v else o for o in s.nodes[p]["outs"]]
        b = _library(s, kind, new_prev, rng, given)
        s.add_node("block", {"op": "add"}, [new_prev, b])
        s.nodes[-1]["outs"] = [v]
        return f"insert {kind} at end"
    before = set(range(len(s.nodes)))
    b = _library(s, kind, v, rng, given)
    newnodes = set(range(len(s.nodes))) - before
    out = s.add_node("block", {"op": "add"}, [v, b])
    newnodes.add(len(s.nodes) - 1)
    s.redirect(v, out, skip=newnodes)
    return f"insert {kind} at {k}"


def edit_delete(s, given, rng):
    subs = s.sublayers()
    if len(subs) < 2:
        return None
    a, b, cone, reads = rng.choice(subs)
    add = s.nodes[a]
    stream_in = [u for u in add["ins"] if u != b][0]
    out = add["outs"][0]
    op = s.nodes[a]["raw"]["op"]
    s.nodes.pop(a)
    s.redirect(out, stream_in)
    return f"delete sublayer ({op})"


def edit_rewire(s, given, rng):
    st = s.stream()
    subs = s.sublayers()
    if st is None or len(subs) < 2:
        return None
    versions = st[0]
    k = rng.randrange(len(subs) - 1)
    a1, b1, c1, r1 = subs[k]
    a2, b2, c2, r2 = subs[k + 1]
    v_before, v_after = versions[k], versions[k + 1]
    if r2 == [v_after]:
        # sequential -> parallel: the second sublayer's nodes read the stream before the first
        for i in c2:
            s.nodes[i]["ins"] = [v_before if u == v_after else u for u in s.nodes[i]["ins"]]
        return "rewire parallel"
    if r2 == [v_before]:
        for i in c2:
            s.nodes[i]["ins"] = [v_after if u == v_before else u for u in s.nodes[i]["ins"]]
        return "rewire sequential"
    return None


def _resize_per_layer(obj, L):
    for k, v in list(obj.items()):
        if isinstance(v, dict) and "per_layer" in v:
            pl = v["per_layer"]
            obj[k] = {"per_layer": (pl * L)[:L] if len(pl) < L else pl[:L]}


def edit_repeat(s, given, rng):
    L = s.block_attrs["repeat"]
    new = rng.choice([x for x in (L - 1, L + 1) if 1 <= x <= 6])
    s.block_attrs["repeat"] = new
    s.top["model"]["n_layers"] = new
    wi = s.block_attrs.get("weight_index")
    if wi:
        pl = wi["per_layer"]
        s.block_attrs["weight_index"] = {"per_layer": (pl * new)[:new]}
    for n in s.nodes:
        if n["sect"] == "block":
            _resize_per_layer(n["raw"], new)
    for sl in s.slots.values():
        _resize_per_layer(sl, new)
    return f"repeat {L}->{new}"


def edit_tie(s, given, rng):
    if s.logits is None:
        return None
    p = s.producer(s.logits)
    if p is None or s.nodes[p]["raw"]["op"] != "linear":
        return None
    raw = s.nodes[p]["raw"]
    emb = [n["raw"]["weight"] for n in s.nodes if n["raw"]["op"] == "embedding"]
    if not emb:
        return None
    if raw["weight"] == emb[0]:
        raw["weight"] = "output.weight"
        return "untie"
    raw["weight"] = emb[0]
    return "tie"


def edit_width(s, given, rng):
    groups = {}
    for t, shp in given.items():
        stem = t.rsplit("_", 1)[0] if "_" in t.split(".")[-2] else t
        groups.setdefault(stem, []).append(t)
    if not groups:
        return None
    stem = rng.choice(sorted(groups))
    ts = groups[stem]
    d = _model(s)["d"]

    def is_down(t):
        return t.endswith("down.weight")

    cur = None
    for t in ts:
        cur = given[t][1] if is_down(t) else given[t][0]
    ff = rng.choice([w for w in WIDTHS if w != cur])
    for t in ts:
        given[t] = [d, ff] if is_down(t) else [ff, d]
    return f"width {cur}->{ff}"


EDITS = {"set_attr": edit_set_attr, "heads": edit_heads, "swap_op": edit_swap_op, "qk_norm": edit_qk_norm,
         "insert": edit_insert, "delete": edit_delete, "rewire": edit_rewire, "repeat": edit_repeat,
         "tie": edit_tie, "width": edit_width}


def apply(doc, given, kind, rng, trainable=True):
    """(document, given, description) after one edit of the kind, or None when it does not apply or the
    result fails `check` (never expected; the tests count it)."""
    s = SSA.from_doc(doc)
    g = dict(given or {})
    desc = EDITS[kind](s, g, rng)
    if desc is None:
        return None
    try:
        out = s.emit()
    except ValueError:
        return None
    g = _clean_given(out, g)
    if check(out, g, trainable) is not None:
        return None
    return out, g, desc


def crossover(a, ga, b, gb, rng):
    """a's block up to a cut, then b's sublayers from a cut (b's tensors and slots renamed apart). The rest
    of the document (embed, head, repeat) is a's."""
    sa, sb = SSA.from_doc(a), SSA.from_doc(b)
    subs_a, subs_b = sa.sublayers(), sb.sublayers()
    if not subs_a or not subs_b:
        return None
    ka = rng.randrange(1, len(subs_a) + 1)
    kb = rng.randrange(0, len(subs_b))
    st_a = sa.stream()
    versions_a = st_a[0]
    # cut a after its ka-th sublayer: the stream version there becomes the input of b's tail
    g = dict(ga or {})
    tail = subs_b[kb:]
    stop_b = set(sb.stream()[0])
    v = versions_a[ka]
    # a's sublayers after the cut are dropped: the final x becomes the tail's end
    final_name = "x"
    tag = _fresh_tag(sa, "xo")
    tensor_map = {}

    def rename_tensors(raw):
        for path, t in list(bridge.tools()._tensor_names(raw)):
            if t not in tensor_map:
                pre = "blk.{i}." if t.startswith("blk.{i}.") else ""
                tensor_map[t] = pre + tag + "_" + t[len(pre):]
            canon._set_path(raw, path, tensor_map[t])

    # make v a plain value (if it is a's final x, give the write a fresh value)
    end = sa.carry_out.get(final_name)
    if v == end:
        nv = sa.fresh()
        p = sa.producer(v)
        sa.nodes[p]["outs"] = [nv if o == v else o for o in sa.nodes[p]["outs"]]
        v = nv
    cur = v
    vmap = {}
    for (add_i, branch, cone, reads) in tail:
        if len(reads) != 1:
            return None
        order = sorted(cone)
        vmap = {reads[0]: cur}
        for i in order:
            n = copy.deepcopy(sb.nodes[i])
            ins = []
            for u in n["ins"]:
                if u in vmap:
                    ins.append(vmap[u])
                elif u == "pos":
                    ins.append(u)
                else:
                    return None   # reads something outside its cone and the stream
            raw = n["raw"]
            rename_tensors(raw)
            if "state" in raw:
                raw["state"] = _new_slot(sa, copy.deepcopy(sb.slots[raw["state"]]))
            outs = sa.add_node("block", raw, ins, n_out=len(n["outs"]), multi=n["multi"])
            outs = outs if n["multi"] else [outs]
            for o_old, o_new in zip(n["outs"], outs):
                vmap[o_old] = o_new
        if branch not in vmap:
            return None
        cur = sa.add_node("block", {"op": "add"}, [cur, vmap[branch]])
    for t_old, t_new in tensor_map.items():
        if t_old in (gb or {}):
            g[t_new] = list(gb[t_old])
    # cur is the new final x: the head's reads of the old end, and the carried value, become cur
    for n in sa.nodes:
        if n["sect"] == "head":
            n["ins"] = [cur if u == end else u for u in n["ins"]]
    if sa.logits == end:
        sa.logits = cur
    sa.carry_out[final_name] = cur
    try:
        out = sa.emit()
    except ValueError:
        return None
    g = _clean_given(out, g)
    if check(out, g) is not None:
        return None
    return out, g, f"crossover a[:{ka}] + b[{kb}:]"


def propose(doc, given, rng, level="class", parent_hash=None, tries=50, kinds=EDIT_TYPES, stats=None,
            trainable=True):
    """One edit that gives a valid document of a different canonical hash: (doc, given, desc, hash).
    stats (a dict) counts the draws: inapplicable, noop (filtered before any evaluation)."""
    if parent_hash is None:
        parent_hash = canon.canonical_hash(doc, level, given)
    for _ in range(tries):
        kind = rng.choice(kinds)
        r = apply(doc, given, kind, rng, trainable)
        if r is None:
            if stats is not None:
                stats["inapplicable"] = stats.get("inapplicable", 0) + 1
            continue
        out, g, desc = r
        h = canon.canonical_hash(out, level, g)
        if h == parent_hash:
            if stats is not None:
                stats["noop_filtered"] = stats.get("noop_filtered", 0) + 1
            continue
        return out, g, desc, h
    return None


__all__ = ["SSA", "EDIT_TYPES", "EDITS", "apply", "crossover", "propose", "check", "random"]
