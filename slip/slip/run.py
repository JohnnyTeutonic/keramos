"""Running a compiled document in keramos's reference interpreter (tools/ir_numpy.py)."""

import numpy as np

from . import keramos

_NUMPY = {0: np.float32, 1: np.float16}  # GGUF types slip reads itself: F32 and F16


def load_weights(path):
    """Every tensor of a GGUF as f32, shaped as ir_numpy expects (numpy order = reversed GGUF dims).
    F32 and F16 are read here; other types go to ir_numpy's reader, which needs the gguf package."""
    V = keramos.tools()
    meta, tensors, start, raw, _ = V.read_gguf(path)
    if any(t not in _NUMPY for _, _, t, _ in tensors):
        import ir_numpy
        return ir_numpy.load_weights(str(path))
    W = {}
    for name, dims, t, off in tensors:
        n = int(np.prod(dims)) if dims else 1
        a = np.frombuffer(raw, dtype=_NUMPY[t], count=n, offset=start + off).astype(np.float32)
        W[name] = a.reshape([int(x) for x in reversed(dims)])
    return W


def _spec(name, w):
    """A --shape name: a tensor name as the document writes it, or slip's @path (@ffn_up is
    blk.{i}.ffn_up.weight in the block, or ffn_up.weight outside it)."""
    if name in w.tensors:
        return name
    if name.startswith("@"):
        p = name[1:]
        cands = [pre + p + ("" if "." in p else ".weight") for pre in ("blk.{i}.", "")]
        found = [c for c in cands if c in w.tensors]
        if found:
            return found[0]
    raise SystemExit(f"--shape {name}: the model names no such tensor")


def solve_shapes(doc, shapes=()):
    """The solver with every --shape applied: 'NAME=AxB' gives the whole shape, 'NAME=H' (or 'NAME=AxB'
    with fewer numbers than the tensor's dimensions) fills its free dimensions in order. Where the spec
    allows several sizes and nothing chose one (relative_kv's width, position_bias's bucket count), the
    least is taken."""
    from . import widths
    errs, w = widths.check(doc)
    if errs:
        raise SystemExit("\n".join(m for _, _, m in errs))
    given = {}
    for s in shapes:
        name, _, val = s.partition("=")
        try:
            nums = [int(x) for x in val.lower().split("x")]
        except ValueError:
            raise SystemExit(f"--shape {s}: write NAME=AxB or NAME=H")
        t = _spec(name.strip(), w)
        cur = w.shape(t)
        if len(nums) == len(cur):
            given[t] = nums
        else:
            free = [k for k, v in enumerate(cur) if v is None]
            if len(nums) != len(free):
                raise SystemExit(f"--shape {s}: {t} is {widths.fmt(cur)}, with {len(free)} free dimension(s)")
            full = list(cur)
            for k, v in zip(free, nums):
                full[k] = v
            given[t] = full
    errs, w = widths.check(doc, given=given)
    for t, k, v in w.defaults:
        if t in w.tensors and w.shape(t)[k] is None and v is not None:
            given[t] = [v if j == k else x for j, x in enumerate(w.shape(t))]
    errs, w = widths.check(doc, given=given)
    if errs:
        raise SystemExit("\n".join(m for _, _, m in errs))
    return w


def _at(t):
    """A tensor name as slip's @path where there is one."""
    p = t[len("blk.{i}."):] if t.startswith("blk.{i}.") else t
    p = p[:-len(".weight")] if p.endswith(".weight") and "." not in p[:-len(".weight")] else p
    return "@" + p if "{i}" not in p else t


def free_sizes(w):
    """One --shape per free size: the free dimensions are unknowns shared between tensors (an FFN's hidden
    width is the rows of up and gate and the columns of down), so each is named once, by the first tensor
    that holds it alone."""
    from .widths import find
    seen, out = set(), []

    def first_free(t):  # a free out-dimension (the rows: up's, not down's) reads most naturally
        s = w.shape(t)
        return (s.index(None) if None in s else len(s), t)
    for t in sorted((t for t in w.names() if t not in w.tied and t in w.tensors), key=first_free):
        free = list({id(r): r for r in (find(d) for d in w.tensors[t]) if r.value is None}.values())
        if not free or any(id(r) in seen for r in free):
            continue
        out.append(f"--shape {_at(t)}=" + "x".join(f"N{len(seen) + k + 1}" for k in range(len(free))))
        seen.update(id(r) for r in free)
    return out


def random_weights(doc, shapes=(), seed=0):
    """Weights for every tensor the document names, from the solved shapes: matrices N(0, 1/in), gains
    1, biases 0, attention-residual pseudo-queries 0, a state-space transition 0.5 I plus small noise,
    selective_scan's A -exp(N(0, 1)) (so negative), and other small tensors N(0, 0.1^2). Each tensor
    draws from its own stream (seed, name), so one tensor's values do not depend on the others."""
    import zlib

    from . import widths
    w = solve_shapes(doc, shapes)
    free = [(t, w.shape(t)) for t in w.names() if None in w.shape(t) and t not in w.tied]
    if free:
        raise SystemExit("random weights need every tensor's shape; still free:\n" + "\n".join(
            f"  {t} {widths.fmt(s)}" for t, s in free) + "\ngive the free sizes: " + " ".join(free_sizes(w)))
    W = {}
    for t in w.names():
        if t in w.tied:
            continue  # a linear weight with tied_to reads the tensor it is tied to (IR_SPEC 5.3)
        shape, role = w.shape(t), w.roles.get(t, "matrix")
        for inst in w.instances(t):
            rng = np.random.default_rng([seed, zlib.crc32(inst.encode())])
            if role == "gain":
                a = np.ones(shape)
            elif role in ("bias", "zero"):
                a = np.zeros(shape)
            elif role == "transition":
                a = 0.5 * np.eye(shape[0]) + rng.normal(0, 0.05 / np.sqrt(shape[0]), shape)
            elif role == "negative":
                a = -np.exp(rng.normal(0, 1, shape))
            elif role == "small":
                a = rng.normal(0, 0.1, shape)
            elif role == "embedding":
                a = rng.normal(0, 1, shape)
            else:  # a matrix [out, in] or a stack [E, out, in]: unit-variance outputs
                a = rng.normal(0, 1 / np.sqrt(shape[-1]), shape)
            W[inst] = np.asarray(a, np.float32)
    return W, w


def save_gguf(path, doc, W, name=None):
    """The document and its weights as a GGUF that keramos's checks accept, written by the conformance
    suite's own writer (tools/ir_conformance.py): architecture "paperkiln" (IR_SPEC 7.1), F32 tensors,
    paperkiln.ir embedded, a word tokenizer of model.vocab placeholder tokens."""
    keramos.tools()
    import ir_conformance
    m = doc["model"]
    wi = (doc.get("block") or {}).get("weight_index")
    stored = max(wi["per_layer"]) + 1 if isinstance(wi, dict) else doc["block"]["repeat"]
    meta = {
        "general.architecture": (8, "paperkiln"),
        "general.name": (8, name or doc.get("family_hint") or "slip"),
        "general.alignment": (4, 32),
        "paperkiln.context_length": (4, m["ctx"]),
        "paperkiln.embedding_length": (4, m["d"]),
        "paperkiln.block_count": (4, stored),
        "paperkiln.vocab_size": (4, m["vocab"]),
    }
    tokens = ["<unk>"] + [f"w{i}" for i in range(1, m["vocab"])]
    sws = [n["state_weight"] for n in (doc.get("embed") or {}).get("nodes", []) if n.get("state_weight")]
    if sws and sws[0] in W:  # per-turn state tokens (IR_SPEC 6.8.1, rule 28): k, and the template's markers
        meta["paperkiln.state_tokens.k"] = (4, int(W[sws[0]].shape[0]))
        specials = (doc.get("tokenizer") or {}).get("specials") or {}
        for role, marker in keramos.tools().MARKERS.items():
            if isinstance(specials.get(role), int) and 0 <= specials[role] < len(tokens):
                tokens[specials[role]] = marker
    meta["tokenizer.ggml.model"] = (8, "word")
    meta["tokenizer.ggml.tokens"] = (9, tokens)
    ir_conformance.write_gguf(path, meta, dict(sorted(W.items())), doc)


def forward(doc, W, ids):
    keramos.tools()
    import ir_numpy
    return np.asarray(ir_numpy.forward(doc, W, list(ids)), dtype=np.float32)


def greedy(doc, W, ids, n):
    """n greedy tokens after ids, each by a full prefill (the reference's simplest exact path)."""
    ids = list(ids)
    for _ in range(n):
        ids.append(int(np.argmax(forward(doc, W, ids)[-1])))
    return ids
