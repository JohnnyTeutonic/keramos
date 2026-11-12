"""The semantic difference between two keramos documents: a typed edit script and a distance.

Both documents are first put in canonical form (canon.canonical_document: dead code gone, aliases
resolved, a canonical topological order and SSA value names), so whatever canon.py equates contributes
nothing. Each section's node lists are then aligned (Needleman-Wunsch) on node labels, and the alignment
is read as typed edits:

  insert / delete   a node present on one side only                       cost 1
  replace           aligned nodes of different ops                         cost 1
  attr              aligned nodes of one op whose attributes differ        cost 0.5 (the attributes listed)
  rewire            aligned nodes one of whose inputs comes from a node
                    that is not the aligned counterpart of the other's     cost 0.5 per input
  shape             aligned nodes whose tensors differ in solved shape     cost 0.25
  tie               aligned nodes whose tensor is shared on one side only  cost 0.5
  repeat            block applications differ                              cost 1 (also weight_index)

The distance is the sum. It is 0 for documents of one canonical hash and an upper bound on a graph edit
distance with these costs. The node correspondence is heuristic (anchors by unique forward or backward
structural hash, then greedy propagation along edges, then alignment in canonical order), not a search
over every correspondence, which is NP-hard in general, followed by a local search that swaps or moves
partners while the cost drops. One direction's script is therefore not guaranteed to cost what the
reverse costs; `distance` takes the smaller, so it is symmetric. The text form is slip's diff of the two
canonical documents, tensor names masked, so it reads as slip.
"""

import json

from . import bridge, canon

COST = {"insert": 1.0, "delete": 1.0, "replace": 1.0, "attr": 0.5, "rewire": 0.5, "shape": 0.25, "tie": 0.5,
        "repeat": 1.0}


def _prepare(doc, given, level):
    c = canon.canonical_document(doc, level)
    shapes = canon.solved_shapes(doc, given)
    # tensor use counts (sharing)
    uses = {}
    for sect in ("embed", "block", "head"):
        for n in c[sect]["nodes"]:
            for _, t in bridge.tools()._tensor_names(n):
                uses[t] = uses.get(t, 0) + 1
    slots = {s["name"]: {k: v for k, v in s.items() if k != "name"} for s in c["block"].get("state", [])}
    secs = {}
    for sect in ("embed", "block", "head"):
        rows = []
        prod = {}
        for j, n in enumerate(c[sect]["nodes"]):
            refs = list(bridge.tools()._tensor_names(n))
            attrs = {k: v for k, v in n.items() if k not in ("op", "in", "out", "state")}
            for path, t in refs:
                canon._set_path(attrs, path, "T")
            if "state" in n:
                attrs["slot"] = slots.get(n["state"])
            # each input's source as the node reads it: (node, out) in this section, or the name read from
            # outside it (x before the block writes it, pos, an embed value)
            row = {"op": n["op"], "attrs": attrs, "ins": list(n["in"]),
                   "src": [prod.get(u, ("ext", u)) for u in n["in"]],
                   "shapes": [shapes.get(t) for _, t in refs],
                   "shared": [uses.get(t, 0) > 1 for _, t in refs]}
            outs = n["out"] if isinstance(n["out"], list) else [n["out"]]
            row["final"] = [o for o in outs if o in ("x", "logits")]
            rows.append(row)
            for k, o in enumerate(outs):
                prod[o] = (j, k)
        secs[sect] = (rows, prod)
    return c, secs


def _sub_cost(a, b):
    if a["op"] != b["op"]:
        return COST["replace"]
    if json.dumps(a["attrs"], sort_keys=True) != json.dumps(b["attrs"], sort_keys=True):
        return COST["attr"]
    return 0.0


def _align(A, B):
    n, m = len(A), len(B)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i * COST["delete"]
    for j in range(1, m + 1):
        D[0][j] = j * COST["insert"]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + COST["delete"], D[i][j - 1] + COST["insert"],
                          D[i - 1][j - 1] + _sub_cost(A[i - 1], B[j - 1]))
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(D[i][j] - (D[i - 1][j - 1] + _sub_cost(A[i - 1], B[j - 1]))) < 1e-9:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and abs(D[i][j] - (D[i - 1][j] + COST["delete"])) < 1e-9:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return pairs[::-1]


def _label(r):
    return canon._h(r["op"], r["attrs"])


def _directional(rows):
    """Forward hashes (a node and everything it reads) and backward hashes (a node and everything that
    reads it), within the section. A local edit changes the forward hashes downstream of it and the
    backward hashes upstream of it, never both on one side: unchanged nodes keep one of the two."""
    n = len(rows)
    fwd = [None] * n
    for i, r in enumerate(rows):
        ins = [fwd[s[0]] + f"/{s[1]}" if s[0] != "ext" else "ext:" + s[1] for s in r["src"]]
        if r["op"] in canon.COMMUTATIVE:
            ins = sorted(ins)
        fwd[i] = canon._h(_label(r), ins)
    cons = [[] for _ in range(n)]
    for i, r in enumerate(rows):
        for k, s in enumerate(r["src"]):
            if s[0] != "ext":
                cons[s[0]].append((i, k, s[1]))
    bwd = [None] * n
    for i in range(n - 1, -1, -1):
        r = rows[i]
        cs = []
        for (c, k, o) in cons[i]:
            kk = "c" if rows[c]["op"] in canon.COMMUTATIVE else k
            cs.append(f"{bwd[c]}/{kk}/{o}")
        bwd[i] = canon._h(_label(r), sorted(cs), r.get("final"))
    return fwd, bwd, cons


def _match(A, B):
    """{i: j}: a correspondence of A's nodes to B's (anchors by unique forward or backward hash, then
    propagation along edges between nodes of one op, then the rest by alignment in canonical order)."""
    fa, ba, ca = _directional(A)
    fb, bb, cb = _directional(B)
    m, used = {}, set()

    def anchor(ha, hb):
        from collections import defaultdict
        ia, ib = defaultdict(list), defaultdict(list)
        for i, h in enumerate(ha):
            if i not in m:
                ia[h].append(i)
        for j, h in enumerate(hb):
            if j not in used:
                ib[h].append(j)
        for h, lst in ia.items():
            if h in ib and len(lst) == 1 and len(ib[h]) == 1:   # unique on both sides; the rest by edges
                m[lst[0]] = ib[h][0]
                used.add(ib[h][0])

    anchor(fa, fb)
    anchor(ba, bb)
    changed = True
    while changed:
        changed = False
        for i, j in list(m.items()):
            for k, (sa_, sb_) in enumerate(zip(A[i]["src"], B[j]["src"])):
                if sa_[0] != "ext" and sb_[0] != "ext" and sa_[0] not in m and sb_[0] not in used and \
                        A[sa_[0]]["op"] == B[sb_[0]]["op"]:
                    m[sa_[0]] = sb_[0]
                    used.add(sb_[0])
                    changed = True
            # consumers: only where one unmatched reader of this op at this input position exists on each side
            for (c, k, o) in ca[i]:
                if c in m:
                    continue
                ca_ = [x for (x, k1, o1) in ca[i] if x not in m and k1 == k and o1 == o and A[x]["op"] == A[c]["op"]]
                cb_ = [d for (d, k2, o2) in cb[j] if d not in used and k2 == k and o2 == o and B[d]["op"] == A[c]["op"]]
                if len(ca_) == 1 and len(cb_) == 1:
                    m[c] = cb_[0]
                    used.add(cb_[0])
                    changed = True
    ra = [i for i in range(len(A)) if i not in m]
    rb = [j for j in range(len(B)) if j not in used]
    for i, j in _align([A[i] for i in ra], [B[j] for j in rb]):
        if i is not None and j is not None:
            m[ra[i]] = rb[j]
    m = _repair(A, B, m)
    # a second start: alignment of the whole canonical sequences; keep the cheaper local optimum
    m2 = {i: j for i, j in _align(A, B) if i is not None and j is not None}
    m2 = _repair(A, B, m2)
    return m if _cost(A, B, m) <= _cost(A, B, m2) else m2


def _broken_edges(A, B, m):
    n = 0
    for i, j in m.items():
        for sa_, sb_ in zip(A[i]["src"], B[j]["src"]):
            if sa_[0] == "ext" or sb_[0] == "ext":
                n += sa_ != sb_
            else:
                n += m.get(sa_[0]) != sb_[0] or sa_[1] != sb_[1]
    return n


def _cost(A, B, m):
    """The distance of a correspondence: substitutions, broken edges, and nodes left unmatched."""
    return (sum(_sub_cost(A[i], B[j]) for i, j in m.items()) + COST["rewire"] * _broken_edges(A, B, m)
            + COST["delete"] * (len(A) - len(m)) + COST["insert"] * (len(B) - len(m)))


def _repair(A, B, m, rounds=6):
    """Local search on the correspondence: swap the partners of two matched nodes, or move a node's partner
    to an unmatched node of the same op, whenever the cost drops (interchangeable nodes, such as the q, k
    and v projections or two attention sublayers, matched crosswise)."""
    best = _cost(A, B, m)
    for _ in range(rounds):
        improved = False
        keys = sorted(m)
        for x in range(len(keys)):
            for y in range(x + 1, len(keys)):
                i1, i2 = keys[x], keys[y]
                if A[i1]["op"] != A[i2]["op"]:
                    continue
                m[i1], m[i2] = m[i2], m[i1]
                c = _cost(A, B, m)
                if c < best - 1e-9:
                    best, improved = c, True
                else:
                    m[i1], m[i2] = m[i2], m[i1]
        free = [j for j in range(len(B)) if j not in set(m.values())]
        for i in sorted(m):
            for j in free:
                if B[j]["op"] != A[i]["op"]:
                    continue
                old = m[i]
                m[i] = j
                c = _cost(A, B, m)
                if c < best - 1e-9:
                    best, improved = c, True
                    free = [jj for jj in range(len(B)) if jj not in set(m.values())]
                    break
                m[i] = old
        if not improved:
            break
    return m


def edits(a, b, given_a=None, given_b=None, level="class"):
    """(distance, [edit records]) from a to b."""
    ca, sa = _prepare(a, given_a, level)
    cb, sb = _prepare(b, given_b, level)
    out = []
    ra, rb = ca["block"].get("repeat"), cb["block"].get("repeat")
    if ra != rb:
        out.append({"type": "repeat", "detail": f"{ra} -> {rb}"})
    if ca["block"].get("weight_index") != cb["block"].get("weight_index"):
        out.append({"type": "repeat", "detail": "weight_index"})
    for k in ("vocab", "d", "ctx"):
        if ca["model"].get(k) != cb["model"].get(k):
            out.append({"type": "replace", "detail": f"model.{k} {ca['model'].get(k)} -> {cb['model'].get(k)}"})
    for sect in ("embed", "block", "head"):
        A, pa = sa[sect]
        B, pb = sb[sect]
        amap = _match(A, B)
        inv = {j: i for i, j in amap.items()}
        pairs = [(i, amap.get(i)) for i in range(len(A))] + [(None, j) for j in range(len(B)) if j not in inv]
        for i, j in pairs:
            if j is None:
                out.append({"type": "delete", "section": sect, "op": A[i]["op"]})
            elif i is None:
                out.append({"type": "insert", "section": sect, "op": B[j]["op"]})
            else:
                x, y = A[i], B[j]
                if x["op"] != y["op"]:
                    out.append({"type": "replace", "section": sect, "detail": f"{x['op']} -> {y['op']}"})
                    continue
                if json.dumps(x["attrs"], sort_keys=True) != json.dumps(y["attrs"], sort_keys=True):
                    keys = sorted(k for k in set(x["attrs"]) | set(y["attrs"])
                                  if json.dumps(x["attrs"].get(k), sort_keys=True) !=
                                  json.dumps(y["attrs"].get(k), sort_keys=True))
                    out.append({"type": "attr", "section": sect, "op": x["op"],
                                "detail": ", ".join(f"{k}: {x['attrs'].get(k)} -> {y['attrs'].get(k)}" for k in keys)})
                for k, (src_a, src_b) in enumerate(zip(x["src"], y["src"])):
                    if src_a[0] == "ext" or src_b[0] == "ext":
                        same = src_a[0] == src_b[0] == "ext" and (src_a[1] == src_b[1] or sect == "block" and
                                                                  src_a[1] not in ("x", "pos") and
                                                                  src_b[1] not in ("x", "pos"))
                    else:
                        same = amap.get(src_a[0]) == src_b[0] and src_a[1] == src_b[1]
                    if not same:
                        out.append({"type": "rewire", "section": sect, "op": x["op"], "detail": f"input {k}"})
                if x["shapes"] != y["shapes"]:
                    out.append({"type": "shape", "section": sect, "op": x["op"],
                                "detail": f"{x['shapes']} -> {y['shapes']}"})
                if x["shared"] != y["shared"]:
                    out.append({"type": "tie", "section": sect, "op": x["op"]})
    dist = sum(COST[e["type"]] for e in out)
    return dist, out


def distance(a, b, given_a=None, given_b=None, level="class"):
    """The smaller of the two directions' edit-script costs: each is an upper bound on the graph edit
    distance with these costs, so the smaller is the tighter, and it is symmetric by construction."""
    return min(edits(a, b, given_a, given_b, level)[0], edits(b, a, given_b, given_a, level)[0])


def text(a, b, level="class", name_a="a", name_b="b"):
    """slip's unified diff of the two canonical documents (tensor names masked)."""
    ca, cb = canon.canonical_document(a, level), canon.canonical_document(b, level)
    return bridge.slip_diff.diff(ca, cb, name_a, name_b)[1]
