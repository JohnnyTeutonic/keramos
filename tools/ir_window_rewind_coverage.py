"""Deep window-rewind traces; cache retention is legal, discard is not assumed.

The existing Gemma3 rewind already uses 48 tokens and window 4. These cases add
incremental calls and an independent fresh-prefix continuation. Actual discard
and unsafe acceptance are demonstrated by the bounded KV test adapter, not by a
claim that an arbitrary external engine evicts its windowed rows.
"""

from __future__ import annotations

import numpy as np

from ir_conformance import SUITE, load_model


MODEL = "tp_mistral_hd16"


def discarded_window_traces():
    """Return deterministic (name, case) pairs without changing existing cases."""
    from ir_conformance_gen import trace_case

    doc, weights, _, _ = load_model(SUITE / "models" / MODEL)
    rng = np.random.default_rng(631024)
    ids = [int(x) for x in rng.integers(1, doc["model"]["vocab"], 53)]
    prefix, continuation = ids[:48], ids[48:]
    steps = [{"method": "forward", "ids": prefix[i : i + 4]} for i in range(0, 48, 4)]
    steps += [
        {"method": "truncate", "n": 8},
        {"method": "forward", "ids": continuation},
    ]
    fresh = [
        {"method": "forward", "ids": prefix[:8]},
        {"method": "forward", "ids": continuation},
    ]
    return [
        (MODEL + "_discarded_window_rewind", trace_case(MODEL, doc, weights, steps)),
        (MODEL + "_discarded_window_fresh", trace_case(MODEL, doc, weights, fresh)),
    ]
