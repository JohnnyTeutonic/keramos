"""The IR conformance suite (conformance/, IR_SPEC.md section 11) and its runner
(tools/ir_conformance.py): the tensor hex format round-trips bit for bit; the GGUF writer's files read back
through tools/ir_validate.py; the eviction rule's legal row sets (6.8.3) on a worked sequence; every case
passes against the numpy reference (a FAIL is a regression of the reference or of the suite)."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ir_conformance as conf  # noqa: E402
import ir_validate  # noqa: E402

HAVE_SUITE = (conf.SUITE / "manifest.json").is_file()


def test_hex_round_trip_keeps_every_bit():
    a = np.array([[1.0, -0.0, np.inf, -np.inf], [np.float32(1e-45), 3.4e38, np.nan, 0.1]], np.float32)
    t = conf.enc(a)
    assert t["f32"][:8] == "3f800000" and t["f32"][8:16] == "80000000"
    assert np.array_equal(conf.dec(t).view(np.uint32), a.view(np.uint32))


def test_exact_compares_bits_but_any_nan_matches():
    a = np.array([0.0, np.nan], np.float32)
    assert conf.close(a, np.array([0.0, -np.nan], np.float32), "EXACT", 0)[0]
    assert not conf.close(np.array([-0.0], np.float32), np.array([0.0], np.float32), "EXACT", 0)[0]


def test_gguf_writer_reads_back(tmp_path):
    doc = json.loads((ROOT / "examples" / "chat_sftfinal.ir.json").read_text(encoding="utf-8"))
    W = {"a.weight": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.ones(5, np.float32)}
    p = tmp_path / "m.gguf"
    conf.write_gguf(p, {"general.architecture": (8, "paperkiln"), "x.n": (4, 7),
                        "tokenizer.ggml.tokens": (9, ["<unk>", "a"])}, W, doc)
    meta, tensors, data_start, raw, _ = ir_validate.read_gguf(p)
    assert meta["x.n"] == (4, 7) and meta["tokenizer.ggml.tokens"][1] == ["<unk>", "a"]
    assert json.loads(meta["paperkiln.ir"][1]) == doc and meta["paperkiln.ir_version"] == (4, 0)
    assert list(meta)[-2:] == ["paperkiln.ir", "paperkiln.ir_version"]
    dims = {t[0]: t[1] for t in tensors}
    assert dims == {"a.weight": [3, 2], "b": [5]}


def test_eviction_rows_eager_and_lazy():
    V, k = 10, 2
    S0, S1 = V, V + 1
    ids = [2, 5, 3, 6, 1, S0, S1]  # one exchange and its block: the call ends where block 0 completes
    allowed = conf.evict_allowed(ids, V, k)
    assert allowed == [[5, 6], [0, 1, 2, 3, 4, 5, 6]]  # eager: the block; lazy: segment 0's text too
    allowed = conf.evict_allowed(ids + [2], V, k)  # the next text position: eager only
    assert allowed == [[5, 6, 7]]
    allowed = conf.evict_allowed([2, 5, 3, 6, 1, S0], V, k)  # a block left open: nothing completes
    assert allowed == [list(range(6))]


@pytest.mark.skipif(not HAVE_SUITE, reason="conformance/ has no manifest yet")
def test_suite_passes_against_the_reference():
    counts = conf.run_suite(conf.NumpyAdapter())
    assert counts["FAIL"] == 0 and counts["PASS"] > 0, counts


@pytest.mark.skipif(not HAVE_SUITE, reason="conformance/ has no manifest yet")
def test_versions_0_and_1_read_alike(tmp_path):
    """IR_SPEC 2.1: a version 0 document is a version 1 document. A suite model written with each version
    validates and gives bit-identical reference logits."""
    import ir_numpy
    doc, W, meta, _ = conf.load_model(conf.SUITE / "models" / "pk_llama")
    ids = [1, 4, 2, 7, 3]
    out = {}
    for ver in (0, 1):
        d = dict(doc, ir_version=ver)
        p = tmp_path / f"v{ver}" / f"v{ver}.gguf"
        p.parent.mkdir()
        conf.write_gguf(p, meta, W, d)
        assert ir_validate.file_errors(p) == []
        d2, W2, _, _ = conf.load_model(p.parent)
        assert d2["ir_version"] == ver
        out[ver] = ir_numpy.forward(d2, W2, ids)
    assert np.array_equal(out[0].view(np.uint32), out[1].view(np.uint32))


@pytest.mark.parametrize("got, ref", [(np.nan, 1.0), (1.0, np.nan), (np.inf, np.nan),
                                      (np.nan, -np.inf)])
def test_rounding_rejects_one_sided_nan_even_beside_equal_finite_values(got, ref):
    with np.errstate(all="raise"):
        assert not conf.close([got, 2.0], [ref, 2.0], "ROUNDING", 1e-5)[0]


@pytest.mark.parametrize("got", [np.nan, np.inf, -np.inf, 1.0])
@pytest.mark.parametrize("ref", [np.nan, np.inf, -np.inf, 1.0])
def test_nonfinite_comparison_matrix(got, ref):
    with np.errstate(all="raise"):
        rounding, message = conf.close([got, 0.0], [ref, 0.0], "ROUNDING", 1e-5)
        exact, _ = conf.close([got, 0.0], [ref, 0.0], "EXACT", 0)
    same_nan = np.isnan(got) and np.isnan(ref)
    same_finite = np.isfinite(got) and np.isfinite(ref) and got == ref
    assert rounding == (same_nan or same_finite)
    assert exact == (same_nan or got == ref)
    if not (np.isnan(got) or np.isnan(ref)) and (np.isinf(got) or np.isinf(ref)):
        assert "infinity" in message


def test_rounding_matching_nan_does_not_disable_finite_tolerance():
    ref = [np.nan, 1000.0]
    with np.errstate(all="raise"):
        assert conf.close([np.nan, 1000.005], ref, "ROUNDING", 1e-5)[0]
        assert not conf.close([np.nan, 1000.02], ref, "ROUNDING", 1e-5)[0]
        assert conf.close([np.nan, np.nan], [np.nan, np.nan], "ROUNDING", 1e-5)[0]
        assert conf.close([], [], "ROUNDING", 1e-5)[0]


def test_matching_infinity_cannot_hide_bad_finite_values_under_rounding():
    assert not conf.close([np.inf, 999.0], [np.inf, 0.0], "ROUNDING", 1e-5)[0]


class ReportedStateAdapter:
    exact = True

    def __init__(self, state):
        self.state = state

    def run_node(self, node, inputs, weights, state):
        result = {"outputs": {"y": np.array([[2.0]], np.float32)}}
        if self.state is not ...:
            result["state"] = self.state
        return result


def recurrent_op_case():
    # Independent adapter fixture: correct output cannot stand in for required state.
    return {"node": {"op": "conv1d_causal", "in": ["x"], "out": "y", "state": "history",
                     "weight": "w", "channels": 1, "kernel": 2},
            "inputs": {"x": conf.enc(np.array([[1.0]], np.float32))},
            "weights": {"w": conf.enc(np.array([[0.0, 2.0]], np.float32))},
            "expect": {"y": conf.enc(np.array([[2.0]], np.float32))},
            "expect_state": {"history": {"x": conf.enc(np.array([[1.0]], np.float32))}},
            "class": "ROUNDING"}


@pytest.mark.parametrize("state", [..., None, {}, {"other": {"x": np.array([[1.0]], np.float32)}},
                                  {"history": {}}])
def test_op_case_fails_when_expected_state_is_not_reported(state):
    ok, detail = conf.run_op_case(ReportedStateAdapter(state), recurrent_op_case(), 1e-5)
    assert not ok and "not reported" in detail


@pytest.mark.parametrize("value, expected", [(1.0, True), (2.0, False)])
def test_op_case_checks_required_state_but_allows_extra_reports(value, expected):
    state = {"history": {"x": np.array([[value]], np.float32), "extra": np.array([7.0], np.float32)},
             "other": {"extra": np.array([9.0], np.float32)}}
    ok, _ = conf.run_op_case(ReportedStateAdapter(state), recurrent_op_case(), 1e-5)
    assert ok == expected


def test_op_case_does_not_require_state_without_an_expectation():
    case = recurrent_op_case()
    del case["expect_state"]
    assert conf.run_op_case(ReportedStateAdapter(...), case, 1e-5)[0]


def test_independent_recurrent_fixture_passes_the_reference():
    assert conf.run_op_case(conf.NumpyAdapter(), recurrent_op_case(), 1e-5)[0]


class ScoredModelAdapter:
    def __init__(self, lp):
        self.lp = lp

    def score(self, model_dir, ids, continuation):
        return {"logprobs": self.lp, "argmax": [2]}


@pytest.mark.parametrize("lp, expected", [
    ([np.nan], False), ([np.inf], False), ([-np.inf], False),
    ([-1.000005], True), ([-1.00002], False), ([[-1.0]], False),
])
def test_model_case_never_credits_nonfinite_or_wrong_shape_scores(tmp_path, lp, expected):
    model = tmp_path / "stub"
    model.mkdir()
    (model / "stub.gguf").touch()  # score-only stub does not open the file
    case = {"prompts": [{"ids": [1], "continuation": [2], "logprobs": [-1.0], "argmax": [2]}]}
    ok, detail = conf.run_model_case(ScoredModelAdapter(lp), case, model, 1e-5)
    assert ok == expected, detail


def test_model_score_keeps_float64_comparison(tmp_path):
    # The f64 score is outside tolerance but rounds to the reference in f32.
    model = tmp_path / "stub"
    model.mkdir()
    (model / "stub.gguf").touch()
    case = {"prompts": [{"ids": [1], "continuation": [2], "logprobs": [-1.0], "argmax": [2]}]}
    ok, _ = conf.run_model_case(ScoredModelAdapter([-1.0 - 1e-8]), case, model, 1e-9)
    assert not ok
