"""A real bounded KV test double, not an external-engine eviction assertion."""

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ir_conformance as conf  # noqa: E402
import ir_numpy  # noqa: E402
from ir_window_rewind_coverage import discarded_window_traces  # noqa: E402


class BoundedKV(conf.Adapter):
    """Incremental Mistral subset with actual per-layer KV arrays.

    Only attention is implemented here; rowwise operators use the reference
    kernels. No full-prefix forward recomputation is used during normal decode.
    Token history is retained solely to implement the explicit safe re-prefill.
    """

    def __init__(self, mode):
        self.mode = mode
        self.evidence = []
        self.s = None

    def open(self, gguf, ir=None, evict=None):
        doc, weights, _, _ = conf.load_model(Path(gguf).parent)
        assert doc["family_hint"] == "mistral"
        self.s = {
            "doc": doc,
            "W": weights,
            "ids": [],
            "kv": [{} for _ in range(doc["block"]["repeat"])],
        }
        return {"session": "one"}

    def _token(self, token, position):
        s = self.s
        env = {
            "tokens": np.array([token]),
            "pos": np.array([position]),
            "_seq_len": position + 1,
        }
        ir_numpy.run_nodes(s["doc"]["embed"]["nodes"], env, s["W"])
        for layer in range(s["doc"]["block"]["repeat"]):
            for node in s["doc"]["block"]["nodes"]:
                if node["op"] != "attention":
                    ir_numpy.run_nodes([node], env, s["W"], layer)
                    continue
                assert node["window"] == 4 and node["sinks"] == 0
                assert not any(
                    node.get(k) for k in ("softcap", "position_bias", "relative_kv")
                )
                q, k, v = (env[n] for n in node["in"])
                cache = s["kv"][layer]
                cache[position] = (k.copy(), v.copy())
                positions = sorted(p for p in cache if position - 3 <= p <= position)
                keys = np.concatenate([cache[p][0] for p in positions])
                values = np.concatenate([cache[p][1] for p in positions])
                heads, kv_heads, hd = node["heads"], node["kv_heads"], node["head_dim"]
                outs = []
                for h in range(heads):
                    kh = h // (heads // kv_heads)
                    scores = (
                        q[:, h * hd : (h + 1) * hd] @ keys[:, kh * hd : (kh + 1) * hd].T
                    ) * np.float32(node["scale"])
                    probs = np.exp(scores - scores.max(axis=1, keepdims=True))
                    probs /= probs.sum(axis=1, keepdims=True)
                    outs.append(probs @ values[:, kh * hd : (kh + 1) * hd])
                env[node["out"]] = np.concatenate(outs, axis=1).astype(np.float32)
                if self.mode != "retain":
                    for p in list(cache):
                        if p < position - 3:
                            del cache[p]
        ir_numpy.run_nodes(s["doc"]["head"]["nodes"], env, s["W"])
        return env["logits"]

    def forward(self, session, ids, report):
        rows = []
        for token in ids:
            rows.append(self._token(token, len(self.s["ids"])))
            self.s["ids"].append(token)
        result = {"len": len(self.s["ids"])}
        if "logits" in report:
            result["logits"] = np.concatenate(rows)
        return result

    def truncate(self, session, n, report):
        old = list(self.s["ids"])
        held = [sorted(c) for c in self.s["kv"]]
        needed = list(range(max(0, n - 3), n))
        missing = [sorted(set(needed) - set(p)) for p in held]
        self.evidence.append(
            {
                "mode": self.mode,
                "before_len": len(old),
                "cut": n,
                "held_positions": held,
                "needed_for_next": needed,
                "missing_needed": missing,
            }
        )
        if self.mode == "refuse" and any(missing):
            raise NotImplementedError("discarded window rows require re-prefill")
        if self.mode == "safe":
            self.s["ids"] = []
            self.s["kv"] = [{} for _ in self.s["kv"]]
            self.forward(session, old[:n], [])
        else:
            self.s["ids"] = old[:n]
            for cache in self.s["kv"]:
                for p in list(cache):
                    if p >= n:
                        del cache[p]
        return {"len": n}

    def close(self, session):
        pass


def test_trace_has_long_incremental_prefill_deep_cut_and_fresh_reference():
    cases = dict(discarded_window_traces())
    rewind, fresh = cases.values()
    assert len(rewind["steps"]) == 14
    assert sum(len(s["ids"]) for s in rewind["steps"][:12]) == 48
    assert all(len(s["ids"]) == 4 for s in rewind["steps"][:12])
    assert rewind["steps"][12] == {"method": "truncate", "n": 8}
    assert (
        rewind["steps"][0]["ids"] + rewind["steps"][1]["ids"]
        == fresh["steps"][0]["ids"]
    )
    assert rewind["steps"][-1] == fresh["steps"][-1]
    np.testing.assert_array_equal(
        conf.dec(rewind["expect"][-1]["logits"]),
        conf.dec(fresh["expect"][-1]["logits"]),
    )


@pytest.mark.parametrize("mode", ["retain", "safe"])
def test_legal_retention_or_actual_discard_and_reprefill_pass(mode):
    adapter = BoundedKV(mode)
    case = discarded_window_traces()[0][1]
    ok, message = conf.run_trace(adapter, case, conf.SUITE / case["model"], 1e-5)
    assert ok, message
    evidence = adapter.evidence[0]
    if mode == "safe":
        assert evidence["held_positions"] == [[44, 45, 46, 47]] * 2
        assert evidence["missing_needed"] == [[5, 6, 7]] * 2
    else:
        assert evidence["held_positions"] == [list(range(48))] * 2
        assert evidence["missing_needed"] == [[], []]


def test_unsafe_accepting_truncate_after_real_discard_fails_numeric_continuation():
    adapter = BoundedKV("unsafe")
    case = discarded_window_traces()[0][1]
    ok, message = conf.run_trace(adapter, case, conf.SUITE / case["model"], 1e-5)
    assert not ok
    assert "step 13 logits" in message
    assert message.count("logits") == 1
    assert "step 0 logits" not in message
    assert adapter.evidence[0]["missing_needed"] == [[5, 6, 7]] * 2


def test_explicit_refusal_is_not_falsely_reported_as_numeric_pass():
    adapter = BoundedKV("refuse")
    case = discarded_window_traces()[0][1]
    with pytest.raises(NotImplementedError, match="discarded window"):
        conf.run_trace(adapter, case, conf.SUITE / case["model"], 1e-5)


@pytest.mark.parametrize("mode", ["retain", "safe", "unsafe"])
def test_fresh_prefix_continuation_passes_with_same_incremental_kernels(mode):
    adapter = BoundedKV(mode)
    case = discarded_window_traces()[1][1]
    ok, message = conf.run_trace(adapter, case, conf.SUITE / case["model"], 1e-5)
    assert ok, message
    assert adapter.evidence == []


def test_committed_new_fixtures_are_reproducible():
    import json

    for name, case in discarded_window_traces():
        assert json.loads((conf.SUITE / "state" / (name + ".json")).read_text()) == case
