"""A black-box fixture for full-session LongRoPE under gathered depth routing."""

import copy
from pathlib import Path

import numpy as np


def routed_longrope_traces():
    """Keep the existing trained MoR tensors; add only explicit factor tables.

    The four-token session routes to depths [2, 1, 3, 1]. At recursion step 1,
    positions [0, 2] are active: their largest position implies length 3, but
    their actual session length is 4. With original_ctx=3 these choose opposite
    tables. This is deliberately one full forward, not an assertion of chunk
    invariance for the non-causal two-set denotation.
    """
    import ir_conformance as runner
    import ir_conformance_gen as gen

    doc, weights, meta, _ = runner.load_model(gen.SUITE / "models" / "pk_mor")
    doc, weights = copy.deepcopy(doc), dict(weights)
    doc["_coverage"] = (
        "Derived from pk_mor's trained tensors, with explicit short=1 and long=2 "
        "RoPE factors. Full session length must survive depth-route gathers."
    )
    weights["regression.rope.short"] = np.ones(4, dtype=np.float32)
    weights["regression.rope.long"] = np.full(4, 2, dtype=np.float32)
    for node in doc["block"]["nodes"]:
        if node["op"] == "rope":
            node["freq_factors"] = {
                "short": "regression.rope.short",
                "long": "regression.rope.long",
                "original_ctx": 3,
            }
    relative = Path("regressions") / "routed_longrope"
    directory = gen.SUITE / relative
    directory.mkdir(parents=True, exist_ok=True)
    runner.write_gguf(directory / "routed_longrope.gguf", meta, weights, doc)
    case = gen.trace_case(
        "pk_mor", doc, weights, [{"method": "forward", "ids": [7, 9, 5, 6]}]
    )
    case["model"] = relative.as_posix()
    case["_coverage"] = doc["_coverage"]
    return [("routed_longrope_session_length", case)]
