"""The new black-box case must expose the historical active-subset defect."""

import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ir_conformance as runner
import ir_conformance_gen as gen
import ir_numpy
import ir_validate
from ir_routed_longrope_coverage import routed_longrope_traces


def fixture(tmp_path, monkeypatch):
    original = gen.SUITE
    monkeypatch.setattr(gen, "SUITE", tmp_path)
    # Source lookup is deliberately the existing immutable model, while the new
    # derived GGUF is written only into pytest's temporary suite.
    load = runner.load_model
    monkeypatch.setattr(
        runner, "load_model", lambda path: load(original / "models" / "pk_mor")
    )
    name, case = routed_longrope_traces()[0]
    monkeypatch.setattr(runner, "load_model", load)
    monkeypatch.setattr(runner, "SUITE", tmp_path)
    return name, case


def test_full_session_fixture_passes_and_is_valid(tmp_path, monkeypatch):
    _, case = fixture(tmp_path, monkeypatch)
    model = tmp_path / case["model"]
    assert ir_validate.file_errors(model / "routed_longrope.gguf") == []
    assert case["expect"][0]["depth"] == [2, 1, 3, 1]
    assert runner.run_trace(runner.NumpyAdapter(), case, model, case["tol"])[0]


def test_active_subset_length_mutant_fails_new_case(tmp_path, monkeypatch):
    _, case = fixture(tmp_path, monkeypatch)
    execute = ir_numpy.run_nodes
    seen = []

    def active_subset_length(nodes, env, weights, *args, **kwargs):
        saved = env.get("_seq_len")
        if any(node["op"] == "rope" for node in nodes):
            subset = int(np.max(env["pos"])) + 1 if len(env["pos"]) else 0
            seen.append((saved, subset))
            env["_seq_len"] = subset
        try:
            return execute(nodes, env, weights, *args, **kwargs)
        finally:
            if saved is not None:
                env["_seq_len"] = saved

    monkeypatch.setattr(ir_numpy, "run_nodes", active_subset_length)
    ok, detail = runner.run_trace(
        runner.NumpyAdapter(), copy.deepcopy(case), tmp_path / case["model"], case["tol"]
    )
    assert (4, 3) in seen
    assert not ok
    assert "logits" in detail


def test_committed_case_and_derived_gguf_are_reproducible(tmp_path, monkeypatch):
    suite = runner.SUITE
    name, case = fixture(tmp_path, monkeypatch)
    assert json.loads((suite / "state" / (name + ".json")).read_text()) == case
    relative = Path(case["model"]) / "routed_longrope.gguf"
    assert (tmp_path / relative).read_bytes() == (suite / relative).read_bytes()
