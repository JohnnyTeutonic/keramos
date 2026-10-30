"""The command adapter enforces IR_SPEC 11.3 before interpreting an outcome."""
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ir_conformance as conf  # noqa: E402


def adapter_with_reply(reply):
    adapter = object.__new__(conf.CmdAdapter)
    adapter.n = 0
    adapter.p = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(json.dumps(reply) + "\n"))
    return adapter


@pytest.mark.parametrize("reply, message", [
    ([], "expected an object"),
    ({"result": {}}, "expected id 1"),
    ({"id": 0, "result": {}}, "expected id 1"),
    ({"id": True, "result": {}}, "expected id 1"),
    ({"id": 1.0, "result": {}}, "expected id 1"),
    ({"id": 1}, "exactly one"),
    ({"id": 1, "error": "failure", "skip": "unsupported"}, "exactly one"),
    ({"id": 1, "result": {}, "error": "failure"}, "exactly one"),
    ({"id": 1, "result": []}, "result has the wrong type"),
    ({"id": 1, "error": {}}, "error has the wrong type"),
    ({"id": 1, "skip": None}, "skip has the wrong type"),
])
def test_malformed_reply_fails_instead_of_passing_or_skipping(reply, message):
    with pytest.raises(RuntimeError, match=message):
        adapter_with_reply(reply)._call("forward", {})


def test_result_decodes_tensors_and_request_id_advances():
    tensor = np.array([1.0, -0.0], np.float32)
    adapter = adapter_with_reply({"id": 1, "result": {"logits": conf.enc(tensor)}})
    result = adapter._call("forward", {"session": "s", "ids": [7]})
    np.testing.assert_array_equal(result["logits"].view(np.uint32), tensor.view(np.uint32))
    assert json.loads(adapter.p.stdin.getvalue()) == {
        "id": 1, "method": "forward", "params": {"session": "s", "ids": [7]}}
    adapter.p.stdout = io.StringIO('{"id": 1, "result": {}}\n')
    with pytest.raises(RuntimeError, match="expected id 2"):
        adapter._call("forward", {})


@pytest.mark.parametrize("outcome, exception", [("error", RuntimeError), ("skip", NotImplementedError)])
def test_valid_error_and_skip_keep_distinct_meanings(outcome, exception):
    with pytest.raises(exception, match="engine message"):
        adapter_with_reply({"id": 1, outcome: "engine message"})._call("open", {})
