"""The implementer's guide's example adapter runs against the suite as written (conformance/IMPLEMENTING.md),
and the runner's version declaration, verdict and robustness rules hold."""
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ir_conformance as conf  # noqa: E402

MINIMAL = ROOT / "adapters" / "minimal" / "minimal_adapter.py"
EXPECTED_PASS = {"op/add/ab", "op/sub/ab", "op/mul/ab", "dequant/blocks/f32", "dequant/blocks/f16",
                 "dequant/blocks/bf16"}


def minimal(*args):
    # Forward slashes: the runner splits the command with shlex, which drops Windows backslashes.
    return conf.CmdAdapter(" ".join([Path(sys.executable).as_posix(), MINIMAL.as_posix(), *args]))


def test_suite_revision_matches_the_specification_header():
    head = (ROOT / "IR_SPEC.md").read_text(encoding="utf-8")[:2000]
    assert re.search(r"specification revision\s+(\d+\.\d+)", head).group(1) == conf.SUITE_REVISION


def test_minimal_adapter_passes_what_it_implements_and_skips_the_rest(tmp_path):
    out = tmp_path / "minimal.json"
    counts = conf.run_suite(minimal(), out_json=out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert {r["id"] for r in data["results"] if r["verdict"] == "PASS"} == EXPECTED_PASS
    assert counts["FAIL"] == counts["XFAIL"] == counts["XPASS"] == 0
    assert counts["SKIP"] == len(data["results"]) - len(EXPECTED_PASS)
    assert data["adapter"]["version"] == "0.1 (example)"  # the engine's own hello is recorded
    assert data["spec"]["suite_revision"] == conf.SUITE_REVISION and data["spec"]["match"] is True
    assert set(data["provenance"]) == {"runner_sha256", "manifest_sha256", "expected_fail_sha256"}
    assert sum(sum(v.values()) for v in data["by_profile"].values()) == len(data["results"])


def test_declared_revision_mismatch_is_recorded_and_changes_no_verdict(tmp_path, capsys):
    out = tmp_path / "mismatch.json"
    counts = conf.run_suite(minimal("--spec", "1.4"), only="dequant", out_json=out)
    assert counts["PASS"] == 3 and counts["SKIP"] == 7
    assert json.loads(out.read_text(encoding="utf-8"))["spec"]["match"] is False
    assert "run the suite at tag v1.4.x" in capsys.readouterr().out
    assert conf.spec_check(None)["record"]["match"] is None
    assert conf.spec_check("v1.6.2")["record"]["match"] is True
    assert conf.spec_check("one point six")["record"]["match"] is False


class _Scripted(conf.Adapter):
    """Replies from a script: a value is returned, an exception raised."""
    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)

    def _next(self):
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def open(self, gguf, ir=None, evict=None):
        return {"session": "s"}

    def forward(self, session, ids, report):
        return self._next()

    def run_node(self, node, inputs, weights, state):
        return self._next()


def _model_dir():
    return next(d for d in sorted((conf.SUITE / "models").iterdir()) if (d / f"{d.name}.gguf").is_file())


def test_a_later_skip_does_not_hide_an_earlier_mismatch_in_a_trace():
    want = conf.enc(np.zeros((1, 4), np.float32))
    case = {"steps": [{"method": "forward", "ids": [1]}, {"method": "forward", "ids": [2]}],
            "expect": [{"len": 1, "logits": want}, {"len": 2, "logits": want}]}
    wrong = {"len": 1, "logits": np.ones((1, 4), np.float32)}
    ok, msg = conf.run_trace(_Scripted([wrong, NotImplementedError("no step 2")]), case, _model_dir(), 1e-5)
    assert not ok and "then a skip" in msg
    right = {"len": 1, "logits": np.zeros((1, 4), np.float32)}
    try:
        conf.run_trace(_Scripted([right, NotImplementedError("no step 2")]), case, _model_dir(), 1e-5)
    except NotImplementedError:
        pass  # nothing wrong was seen: the case is a SKIP
    else:
        raise AssertionError("a skip after matching steps must stay a skip")


def test_a_later_skip_does_not_hide_an_earlier_mismatch_in_a_model_case():
    logits = np.zeros((2, 4), np.float32)
    logits[:, 3] = 5.0  # argmax 3 at both rows, not the expected 1
    case = {"prompts": [{"ids": [1], "continuation": [1, 2], "logprobs": [-1.0, -1.0], "argmax": [1, 1]},
                        {"ids": [2], "continuation": [1, 2], "logprobs": [-1.0, -1.0], "argmax": [1, 1]}]}
    ok, msg = conf.run_model_case(_Scripted([{"logits": logits}, NotImplementedError("later")]), case,
                                  _model_dir(), 1e-5)
    assert not ok and "then a skip" in msg


def test_a_malformed_result_fails_the_case_without_aborting_the_run():
    counts = conf.run_suite(_Scripted([{"outputs": "not an object"}]), only="op/add")
    assert counts["FAIL"] == 1 and counts["PASS"] == 0
