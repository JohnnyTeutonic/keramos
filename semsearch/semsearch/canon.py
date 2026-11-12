"""Canonical form and canonical hash of a keramos document.

keramos's own identity of graphs is rule 38 (`structure` in tools/ir_validate.py): value and slot names,
add/mul operand order, informative keys, defaults and number spelling aside, and nothing else. It keeps
node order ("two independent nodes written the other way round differ"), keeps dead nodes, and never
equates a `per_layer` list with a number. That is the right rule for a file's label; for search it is too
fine, because many documents that differ under it compute the same function on the same weights.

This module adds, on top of rule 38 (each step is a rewrite that keeps the function exactly, by the
IR_SPEC section named):

  order      the graph is a dataflow DAG (SSA over value writes), hashed Merkle-style, so node order
             counts only through data dependencies (3.5: "Nodes run in order. A later write to a name
             replaces the earlier value" is exactly SSA);
  dead code  only what `logits` depends on is hashed (a node nobody reads changes no output);
  per_layer  a per_layer list whose entries are all equal is that value (3.8);
  window     attention window >= model.ctx is window 0 (5.15: "window >= p + 1 with sinks 0 is plain
             causal attention", and p < ctx, 3.3); with window 0 the sinks are 0 (the mask applies only
             with a window, 5.15);
  identity   weight_index [0, 1, ..., L-1] is no weight_index (7.1); apply true is no apply (3.8);
  tying      a linear with tied_to reads the tied tensor (5.3 reads it only when the file lacks the
             weight; a search genome carries no weights, so it always does. A document read against a
             file that holds the weight is outside this module's claim);
  tensors    tensor names do not count; which nodes share a tensor (each reader identified by what it
             computes from and where its result goes), and its shape, do. A name with "{i}" and a literal
             per-layer name for the same tensor (blk.1.w) are treated as different tensors: a possible
             false split, never a false merge.

The claim: two documents of one hash compute the same function, on weights that correspond through a
renaming of tensors. The rewrites are each licensed by the spec; the hash is a Merkle hash plus a
refinement of tensor identities, so a merge of different graphs needs a hash collision or a sharing
pattern that forward and backward contexts cannot tell apart (none known; the review of 13 Nov 2026
found the one-round version's counterexample, now a test).

and, at level "class", one rewrite that keeps the *function class* and the training distribution but
not the function on fixed weights:

  rope pairing  neox and interleaved RoPE differ by a fixed permutation of each head's first n_rot
             dimensions; when q and k come from linear projections (optionally through a per-head
             rmsnorm) and are read only by the attention, the permutation folds into the projection rows
             (and the norm gains), and i.i.d. initialisation and Adam are permutation-equivariant.

The result is a 32-hex-digit canonical hash, and, for documents without per-layer `apply` or
`depth_route`, a canonical document: the reachable nodes in a canonical topological order with canonical
value and slot names, which keramos accepts and which computes bit-identical logits on the same weights.
"""

import copy
import hashlib
import json

from . import bridge

LEVELS = ("exact", "class")
COMMUTATIVE = ("add", "mul")
CONTROL_OPS = ("depth_route",)       # ops whose effect is control flow, not a value: always kept
ATTENTION_OPS = ("attention", "mla_attention")


def _h(*parts):
    s = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:32]


def _layers(v, L):
    if isinstance(v, dict) and "per_layer" in v:
        return list(v["per_layer"])
    return [v] * L


def _collapse(vals):
    if all(json.dumps(x, sort_keys=True) == json.dumps(vals[0], sort_keys=True) for x in vals):
        return vals[0]
    return {"per_layer": list(vals)}


# ------------------------------------------------------------------ normalisation (attribute aliases)

def normalise(doc, level="exact"):
    """The document in one spelling: slip diff's (no prose, informative keys or defaults; numbers the
    shortest decimal of their f32), with the aliases of this module's docstring applied. Tensor names
    are kept. The result is still a document keramos accepts (ir_version and family_hint restored)."""
    if level not in LEVELS:
        raise ValueError(f"level {level!r} is not one of {LEVELS}")
    A = bridge.slip_diff.architecture(doc, tensor_names=True)
    A["ir_version"] = 1
    A["family_hint"] = "semsearch"
    if isinstance(doc.get("tokenizer"), dict):
        A["tokenizer"] = copy.deepcopy({k: v for k, v in doc["tokenizer"].items() if not k.startswith("_")})
    ctx = A["model"]["ctx"]
    blk = A.get("block") or {}
    L = blk.get("repeat", 1)

    # per_layer lists of equal entries, in block nodes and slots
    for n in blk.get("nodes", []):
        for k, v in list(n.items()):
            if isinstance(v, dict) and "per_layer" in v and isinstance(v["per_layer"], list) and v["per_layer"]:
                n[k] = _collapse(v["per_layer"])
        if n.get("apply") is True:
            del n["apply"]
    for sl in blk.get("state", []):
        for k, v in list(sl.items()):
            if isinstance(v, dict) and "per_layer" in v and isinstance(v["per_layer"], list) and v["per_layer"]:
                sl[k] = _collapse(v["per_layer"])

    wi = blk.get("weight_index")
    if isinstance(wi, dict) and wi.get("per_layer") == list(range(L)) and \
            not any("tensor_index" in n for n in blk.get("nodes", [])):
        del blk["weight_index"]   # (tensor_index needs a weight_index, rule 19: then it stays)

    # attention windows: >= ctx is unbounded; unbounded has no sinks. Node and slot alike.
    def fix_window(obj):
        if "window" not in obj:
            return
        ws = _layers(obj["window"], L)
        ss = _layers(obj.get("sinks", 0), L)
        ws = [0 if isinstance(w, int) and w >= ctx else w for w in ws]
        ss = [0 if w == 0 else s for w, s in zip(ws, ss)]
        obj["window"] = _collapse(ws)
        if "sinks" in obj:
            obj["sinks"] = _collapse(ss)

    # (not under the turn_state mask, whose rule needs window 0 as written: 6.8.2)
    turn_state_slots = {n.get("state") for n in blk.get("nodes", []) if n.get("mask") == "turn_state"}
    for n in blk.get("nodes", []):
        if n.get("op") in ATTENTION_OPS and n.get("mask") != "turn_state":
            fix_window(n)
    for sl in blk.get("state", []):
        if sl.get("kind") == "kv" and sl.get("name") not in turn_state_slots:
            fix_window(sl)

    for sect in ("embed", "block", "head"):
        for n in (A.get(sect) or {}).get("nodes", []):
            if n.get("op") == "linear" and n.get("tied_to"):
                n["weight"] = n.pop("tied_to")

    if level == "class":
        _rope_pairing_class(A)
    return A


def _rope_pairing_class(A):
    """neox RoPE -> interleaved, where the permutation between them folds into trainable weights."""
    g = Graph(A, hash_tensors=False)
    tensor_reads = {}
    for i in g.reach_nodes:
        for _, t in _tensor_refs(g.nodes[i].raw):
            tensor_reads[t] = tensor_reads.get(t, 0) + 1
    finals = {}
    for name, v in g.block_final.items():
        finals.setdefault(v, []).append(name)

    def private(nd):
        """Every tensor of the node is read by this node alone (so a permutation of it moves nothing else)."""
        return all(tensor_reads.get(t, 0) == 1 for _, t in _tensor_refs(nd.raw))

    for node in g.nodes:
        if node.op != "rope" or node.raw.get("pairing") != "neox" or node.raw.get("rope_heads", "all") != "all":
            continue
        if node.raw.get("freq_factors") is not None:
            continue
        ok = True
        # each rotated output is read only by an attention node, as its q (output 0) or k (output 1), and is not
        # read as a carried value by the next application or by the head
        for k, vid in enumerate(node.outs):
            for (ri, pos) in g.uses.get(vid, []):
                r = g.nodes[ri]
                if ri in g.reach_nodes and (r.op not in ATTENTION_OPS or pos != k):
                    ok = False
            for name in finals.get(vid, []):
                for key in (("carry", name), ("final", name)):
                    if any(c in g.reach_nodes for c, _ in g.uses.get(key, [])):
                        ok = False
        # each input (q, k) comes from a linear, optionally through an rmsnorm whose group is head_dim, and
        # every value on that chain is read only along it
        for k in (0, 1):
            vid = node.ins[k]
            p = g.producer.get(vid)
            if p is None or len(g.uses.get(vid, [])) != 1:
                ok = False
                continue
            pn = g.nodes[p[0]]
            if pn.op == "rmsnorm" and pn.raw.get("group") == node.raw.get("head_dim") and \
                    not pn.raw.get("weight_offset"):
                if not private(pn):
                    ok = False
                vid2 = pn.ins[0]
                p2 = g.producer.get(vid2)
                if p2 is None or len(g.uses.get(vid2, [])) != 1:
                    ok = False
                    continue
                pn = g.nodes[p2[0]]
            if pn.op != "linear" or not private(pn):
                ok = False
            for vv in (vid,):
                for name in finals.get(vv, []):
                    if any(c in g.reach_nodes for c, _ in g.uses.get(("carry", name), []) + g.uses.get(("final", name), [])):
                        ok = False
        if ok:
            node.raw["pairing"] = "interleaved"   # node.raw is the node of A itself


# ------------------------------------------------------------------ the dataflow graph

class Node:
    __slots__ = ("sect", "idx", "op", "raw", "ins", "outs", "h1", "h2", "phi")

    def __init__(self, sect, idx, raw):
        self.sect, self.idx, self.raw, self.op = sect, idx, raw, raw.get("op")
        self.ins, self.outs, self.phi = [], [], []
        self.h1 = self.h2 = None


def _tensor_refs(n):
    return list(bridge.tools()._tensor_names(n))


def _set_path(n, path, value):
    parts = path.split(".")
    obj = n
    for p in parts[:-1]:
        obj = obj[p]
    obj[parts[-1]] = value


class Graph:
    """The document as SSA: every write is a new value. Values are ids: 'tokens', 'pos', ('n', node, k)
    for output k of a node, ('carry', name) for the value a block application reads before writing it
    (the embed's value in application 0, the previous application's after), and ('final', name) for the
    block's last write of a name as later sections read it."""

    def __init__(self, A, hash_tensors=True, shapes=None):
        self.A = A
        self.nodes = []
        self.producer = {}      # vid -> (node index, out k)
        self.uses = {}          # vid -> [(node index, input position)]
        self.slots = {s.get("name"): s for s in (A.get("block") or {}).get("state", [])}
        env = {"tokens": "tokens", "pos": "pos"}
        self._section("embed", env)
        self.embed_env = dict(env)
        blk_nodes = (A.get("block") or {}).get("nodes", [])
        written = []
        for n in blk_nodes:
            outs = n.get("out")
            for o in (outs if isinstance(outs, list) else [outs]):
                if o not in written:
                    written.append(o)
        benv = dict(env)
        self.carried = []
        # every name the block writes is read, before the block writes it, as the carried value: the embed's
        # in application 0 (if it wrote one), the previous application's after. Without per-layer apply a
        # read before any write of a name the embed did not write is refused (rule 36); with it, such a read
        # in a later application sees the previous application's write (3.8), which this links.
        for name in written:
            benv[name] = ("carry", name)
        self._section("block", benv)
        self.block_final = {name: benv[name] for name in written}
        # carried: names the block reads before writing (their carry placeholder is read)
        self.carried = [name for name in written if ("carry", name) in self.uses]
        henv = dict(env)
        for name in written:
            henv[name] = ("final", name)
        self._section("head", henv)
        self.head_env = henv
        self.logits = henv.get("logits")
        self._reach()
        if hash_tensors:
            self._hash(shapes or {})

    def _section(self, sect, env):
        for j, raw in enumerate((self.A.get(sect) or {}).get("nodes", [])):
            nd = Node(sect, j, raw)
            i = len(self.nodes)
            self.nodes.append(nd)
            for pos, name in enumerate(raw.get("in") or []):
                vid = env.get(name, ("unwritten", name))
                nd.ins.append(vid)
                self.uses.setdefault(vid, []).append((i, pos))
            outs = raw.get("out")
            outs = outs if isinstance(outs, list) else [outs]
            conditional = "apply" in raw   # per-layer apply: an output keeps the earlier value where not run
            for k, o in enumerate(outs):
                if conditional:
                    nd.phi.append(env.get(o))
                vid = ("n", i, k)
                nd.outs.append(vid)
                self.producer[vid] = (i, k)
                env[o] = vid

    # reachability from logits (and control ops)
    def _reach(self):
        seen, stack = set(), []
        if self.logits is not None:
            stack.append(self.logits)
        for i, nd in enumerate(self.nodes):
            if nd.op in CONTROL_OPS:
                stack.extend(nd.ins)
                seen.add(("node", i))
        self.reach_nodes = set()
        while stack:
            vid = stack.pop()
            if vid in seen:
                continue
            seen.add(vid)
            if isinstance(vid, tuple) and vid[0] == "n":
                i = vid[1]
                self.reach_nodes.add(i)
                nd = self.nodes[i]
                stack.extend(nd.ins)
                stack.extend(v for v in nd.phi if v is not None)
            elif isinstance(vid, tuple) and vid[0] == "carry":
                stack.append(self.embed_env.get(vid[1]))
                stack.append(self.block_final.get(vid[1]))
            elif isinstance(vid, tuple) and vid[0] == "final":
                stack.append(self.block_final.get(vid[1]))
        for i, nd in enumerate(self.nodes):
            if nd.op in CONTROL_OPS:
                self.reach_nodes.add(i)

    # ---- hashing
    def _local(self, nd, tensor_ids=None):
        """The node's own label: op, attributes, slot, tensor marks; no names of values or slots."""
        n = copy.deepcopy(nd.raw)
        for k in ("in", "out", "state"):
            n.pop(k, None)
        refs = _tensor_refs(nd.raw)
        for path, name in refs:
            mark = ["T", "{i}" in name]
            if tensor_ids is not None:
                mark.append(tensor_ids[name])
            _set_path(n, path, mark)
        slot = None
        if "state" in nd.raw:
            s = copy.deepcopy(self.slots.get(nd.raw["state"], {"missing": nd.raw["state"]}))
            s.pop("name", None)
            slot = s
        outs = nd.raw.get("out")
        return [nd.sect, n, slot, len(outs) if isinstance(outs, list) else 1]

    def _merkle(self, label_of):
        """Value hashes, in a fixed number of rounds: carry placeholders first take the hash of the embed's
        value, then also of the block's final write (a fixpoint pair, broken after one refinement)."""
        memo = {}
        carry = {}

        def vh(vid):
            if vid is None:
                return "none"
            if vid in ("tokens", "pos"):
                return vid
            if vid in memo:
                return memo[vid]
            if vid[0] == "carry":
                return carry.get(vid[1]) or _h("carry", vh(self.embed_env.get(vid[1])))
            if vid[0] == "final":
                r = _h("final", vh(self.block_final.get(vid[1])))
            elif vid[0] == "unwritten":
                r = _h("unwritten", vid[1])
            else:
                i, k = vid[1], vid[2]
                r = _h(nh(i), k)
            memo[vid] = r
            return r

        nmemo = {}

        def nh(i):
            if i in nmemo:
                return nmemo[i]
            nd = self.nodes[i]
            ins = [vh(v) for v in nd.ins]
            if nd.op in COMMUTATIVE:
                ins = sorted(ins)
            r = _h(label_of(nd), ins, [vh(v) for v in nd.phi])
            nmemo[i] = r
            return r

        import sys
        old = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old, 20000))
        try:
            # round 1: placeholders from the embed side only
            for name in list(self.block_final):
                vh(self.block_final[name])
            for name in self.block_final:
                carry[name] = _h("carry", vh(self.embed_env.get(name)), vh(self.block_final[name]))
            memo.clear()
            nmemo.clear()
            hashes = {i: nh(i) for i in range(len(self.nodes))}
            logits = vh(self.logits)
        finally:
            sys.setrecursionlimit(old)
        return hashes, logits

    def _backward(self):
        """{node: hash of its local label and, recursively, of every reader of its outputs with the input
        position read (unordered for add and mul)}. The loop edges (a block's last write read by the next
        application or the head) end in markers, so the recursion is over a DAG."""
        finals = {}
        for name, v in self.block_final.items():
            finals.setdefault(v, []).append(name)
        memo = {}

        def bh(i):
            if i in memo:
                return memo[i]
            nd = self.nodes[i]
            cons = []
            live = self.reach_nodes
            for k, v in enumerate(nd.outs):
                for (c, pos) in self.uses.get(v, []):
                    if c in live:
                        cons.append([k, bh(c), "c" if self.nodes[c].op in COMMUTATIVE else pos])
                for name in finals.get(v, []):
                    read_next = any(c in live for c, _ in self.uses.get(("carry", name), []))
                    read_head = any(c in live for c, _ in self.uses.get(("final", name), []))
                    if read_next or read_head:    # a last write nobody reads is no edge (names aside)
                        cons.append([k, "loop", read_next, read_head])
                if v == self.logits:
                    cons.append([k, "logits"])
            r = _h(self._local(nd), sorted(cons, key=json.dumps))
            memo[i] = r
            return r

        import sys
        old = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old, 20000))
        try:
            return {i: bh(i) for i in range(len(self.nodes))}
        finally:
            sys.setrecursionlimit(old)

    def _hash(self, shapes):
        h1, _ = self._merkle(lambda nd: self._local(nd))
        for i, nd in enumerate(self.nodes):
            nd.h1 = h1[i]
        bwd = self._backward()
        # tensor classes: which reachable nodes read a tensor, each by its tensor-blind hash (what it computes
        # from) and its backward hash (where its result goes, input position included), and the shape
        consumers = {}
        for i in sorted(self.reach_nodes):
            nd = self.nodes[i]
            for path, name in _tensor_refs(nd.raw):
                consumers.setdefault(name, []).append([nd.h1, bwd[i], path])
        self.tensor_ids = {}
        for name, cs in consumers.items():
            shp = shapes.get(name)
            self.tensor_ids[name] = _h("tensor", sorted(cs), "{i}" in name, shp)
        allnames = {name for nd in self.nodes for _, name in _tensor_refs(nd.raw)}
        for name in allnames - set(self.tensor_ids):
            self.tensor_ids[name] = _h("unread-tensor", name)
        h2, logits = self._merkle(lambda nd: self._local(nd, self.tensor_ids))
        for i, nd in enumerate(self.nodes):
            nd.h2 = h2[i]
        self.logits_hash = logits
        blk = self.A.get("block") or {}
        m = self.A.get("model") or {}
        control = sorted(self.nodes[i].h2 for i in self.reach_nodes if self.nodes[i].op in CONTROL_OPS)
        self.hash = _h("semsearch/2", {k: m.get(k) for k in ("vocab", "d", "ctx", "n_layers")},
                       blk.get("repeat"), blk.get("weight_index"), logits, control,
                       sorted(self.tensor_ids[n] for n in consumers))


# ------------------------------------------------------------------ public API

def solved_shapes(doc, given=None):
    """{tensor name: shape list with None for a free size}, from slip's width solver."""
    errs, w = bridge.widths.check(doc, given=given)
    out = {}
    for t in w.names():
        if t in w.tied:
            continue
        out[t] = [None if x is None else int(x) for x in w.shape(t)]
    return out


def canonical_hash(doc, level="exact", given=None, with_shapes=True):
    """The canonical hash of a document (and of its tensor shapes, solved by slip from the graph and the
    given free sizes, when with_shapes)."""
    A = normalise(doc, level)
    shapes = {}
    if with_shapes:
        g0 = solved_shapes(doc, given)
        # names after tying: a tied linear reads its tied tensor, whose shape slip solved
        shapes = g0
    return Graph(A, shapes=shapes).hash


def rule38_key(doc):
    """keramos's own identity (rule 38), tensor names masked as slip diff masks them."""
    return _h(bridge.tools().structure(bridge.slip_diff.architecture(doc, tensor_names=False)))


class NotEmittable(ValueError):
    pass


def canonical_document(doc, level="exact"):
    """The canonical document: reachable nodes only, each section in the topological order that breaks
    ties by node hash, value names v1, v2, ... (names the sections hand to each other kept: x, logits and
    any block value the head reads), slots s0, s1, ... in order of first use. Tensor names are kept, so the
    document runs on the original's weights."""
    A = normalise(doc, level)
    g = Graph(A)
    if any(nd.op in CONTROL_OPS or nd.phi for nd in g.nodes):
        raise NotEmittable("per-layer apply or depth_route: no canonical node order is emitted")
    keep = set(g.carried) | {"x", "logits"}
    head_reads = {v[1] for nd_i in g.reach_nodes for v in g.nodes[nd_i].ins
                  if g.nodes[nd_i].sect == "head" and isinstance(v, tuple) and v[0] == "final"}
    keep |= head_reads
    names = {"tokens": "tokens", "pos": "pos"}
    counter = [0]

    def fresh():
        while True:
            counter[0] += 1
            nm = f"v{counter[0]}"
            if nm not in keep:
                return nm

    out = {k: copy.deepcopy(v) for k, v in A.items() if k not in ("embed", "block", "head")}
    slot_names = {}
    new_state = []
    sections = {}
    for sect in ("embed", "block", "head"):
        idx = [i for i in sorted(g.reach_nodes) if g.nodes[i].sect == sect]
        # final writes of kept names in this section
        final_of = {}
        if sect == "embed":
            for nm in keep:
                v = g.embed_env.get(nm)
                if isinstance(v, tuple) and v[0] == "n" and g.nodes[v[1]].sect == "embed":
                    final_of[v] = nm
        elif sect == "block":
            for nm in keep:
                v = g.block_final.get(nm)
                if isinstance(v, tuple) and v[0] == "n":
                    final_of[v] = nm
        else:
            v = g.head_env.get("logits")
            if isinstance(v, tuple):
                final_of[v] = "logits"
        # dependencies: data, plus "readers of a carried name's placeholder before its final write"
        deps = {i: set() for i in idx}
        for i in idx:
            for v in g.nodes[i].ins:
                if isinstance(v, tuple) and v[0] == "n" and v[1] in deps:
                    deps[i].add(v[1])
        if sect == "block":
            for nm in g.carried:
                fv = g.block_final.get(nm)
                if not (isinstance(fv, tuple) and fv[0] == "n" and fv[1] in deps):
                    continue
                for (ri, _) in g.uses.get(("carry", nm), []):
                    if ri in deps and ri != fv[1]:
                        deps[fv[1]].add(ri)
        # embed and head: a kept name's final write follows every reader of an earlier write of it is
        # unnecessary (earlier writes get fresh names)
        order = []
        done = set()
        ready = sorted([i for i in idx if not deps[i]], key=lambda i: (g.nodes[i].h2, i))
        pending = {i: set(d) for i, d in deps.items()}
        while ready:
            i = ready.pop(0)
            order.append(i)
            done.add(i)
            newly = []
            for j in idx:
                if j not in done and j not in order and i in pending[j]:
                    pending[j].discard(i)
                    if not pending[j] and j not in ready:
                        newly.append(j)
            ready = sorted(ready + newly, key=lambda i: (g.nodes[i].h2, i))
        if len(order) != len(idx):
            raise NotEmittable(f"{sect}: no topological order (a cycle through the carried names)")
        nodes = []
        for i in order:
            nd = g.nodes[i]
            n = copy.deepcopy(nd.raw)
            ins = []
            for v in nd.ins:
                if v in ("tokens", "pos"):
                    ins.append(v)
                elif v[0] == "carry":
                    ins.append(v[1])
                elif v[0] == "final":
                    ins.append(names.get(g.block_final[v[1]], v[1]))
                else:
                    ins.append(names[v])
            n["in"] = ins
            outs = []
            for k, v in enumerate(nd.outs):
                nm = final_of.get(v) or fresh()
                names[v] = nm
                outs.append(nm)
            n["out"] = outs if isinstance(nd.raw.get("out"), list) else outs[0]
            if "state" in n:
                old = n["state"]
                if old not in slot_names:
                    slot_names[old] = f"s{len(slot_names)}"
                    s = copy.deepcopy(g.slots[old])
                    s["name"] = slot_names[old]
                    new_state.append({k: s[k] for k in ["name", "kind"] + sorted(set(s) - {"name", "kind"})
                                      if k in s})
                n["state"] = slot_names[old]
            # one key order: op, in, the attributes sorted, out
            n = {k: n[k] for k in ["op", "in"] + sorted(set(n) - {"op", "in", "out"}) + ["out"] if k in n}
            nodes.append(n)
        sections[sect] = nodes
    out["embed"] = {"nodes": sections["embed"]}
    blk = {k: copy.deepcopy(v) for k, v in (A.get("block") or {}).items() if k not in ("nodes", "state")}
    blk["state"] = new_state
    blk["nodes"] = sections["block"]
    out["block"] = blk
    out["head"] = {"nodes": sections["head"]}
    return out
