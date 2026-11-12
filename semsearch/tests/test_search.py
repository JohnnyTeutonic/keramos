"""The task, the proxy and the search loop, at toy sizes."""

import numpy as np

from semsearch import config_space as cs, search, task


def test_task_format():
    toks, mask = task.batch(np.random.default_rng(0), 4)
    assert toks.shape == (4, 32) and mask.sum() == 4 * 8
    for b in range(4):
        pairs = {int(toks[b, 2 * i]): int(toks[b, 2 * i + 1]) for i in range(8)}
        for t in np.nonzero(mask[b])[0]:
            assert toks[b, t + 1] == pairs[int(toks[b, t])]


def test_evaluation_is_deterministic():
    doc, given = cs.lower(cs.default_config())
    a = task.evaluate(doc, given, steps=5, seed=3)
    b = task.evaluate(doc, given, steps=5, seed=3)
    assert a["fitness"] == b["fitness"]


def test_every_arm_runs_and_a_journal_resumes(tmp_path, monkeypatch):
    for arm in search.ARMS:
        j = tmp_path / f"{arm}.jsonl"
        r1 = search.run(arm, seed=0, budget=6, pop_size=3, sample=2, steps=3, journal=j)
        assert r1["evaluations"] == 6
        calls = []
        real = task.evaluate

        def counting(*a, **k):
            calls.append(1)
            return real(*a, **k)

        monkeypatch.setattr(task, "evaluate", counting)
        r2 = search.run(arm, seed=0, budget=6, pop_size=3, sample=2, steps=3, journal=j)
        monkeypatch.setattr(task, "evaluate", real)
        assert calls == []                                  # every training run came from the journal
        assert [h["fitness"] for h in r1["history"]] == [h["fitness"] for h in r2["history"]]


def test_semantic_arm_wastes_nothing_it_can_see():
    r = search.run("semantic", seed=1, budget=8, pop_size=4, sample=2, steps=2)
    assert r["counters"]["wasted_class"] == 0 and r["counters"]["wasted_fingerprint"] == 0
