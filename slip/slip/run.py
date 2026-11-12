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
