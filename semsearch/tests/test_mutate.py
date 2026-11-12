"""Typed edits: every result is a keramos document with every width solved, each edit type applies, and
the edits work on documents written by other exporters than the config lowering."""

import random

import pytest

from semsearch import bridge, canon, config_space as cs, mutate as mu, semdiff

K = bridge.KERAMOS


def _starts():
    out = [cs.lower(cs.default_config()), cs.lower(dict(cs.default_config(), qk_norm=True, n_heads=4, n_kv_heads=2)),
           cs.lower(dict(cs.default_config(), repeats=2, inject="add", window=8, sinks=2))]
    return out


def test_random_edit_chains_stay_valid():
    rng = random.Random(0)
    counts = {k: 0 for k in mu.EDIT_TYPES}
    for doc, given in _starts():
        cur = (doc, given)
        for _ in range(120):
            kind = rng.choice(mu.EDIT_TYPES)
            s = mu.SSA.from_doc(cur[0])
            g = dict(cur[1])
            if mu.EDITS[kind](s, g, rng) is None:
                continue
            out = s.emit()
            g = mu._clean_given(out, g)
            assert mu.check(out, g) is None, kind
            counts[kind] += 1
            if rng.random() < 0.7:
                cur = (out, g)
    assert all(v > 0 for v in counts.values()), counts


@pytest.mark.parametrize("kind", mu.EDIT_TYPES)
def test_each_edit_applies_to_the_default_model(kind):
    doc, given = cs.lower(cs.default_config())
    rng = random.Random(1)
    if kind in ("delete", "rewire", "width"):
        doc, given = mu.apply(doc, given, "insert", random.Random(2))[:2]
    r = None
    for _ in range(20):
        r = mu.apply(doc, given, kind, rng)
        if r:
            break
    assert r is not None
    out, g, desc = r
    assert mu.check(out, g) is None


def test_propose_never_returns_a_no_op():
    doc, given = cs.lower(cs.default_config())
    rng = random.Random(3)
    h = canon.canonical_hash(doc, "class", given)
    stats = {}
    for _ in range(40):
        out, g, desc, h2 = mu.propose(doc, given, rng, stats=stats)
        assert h2 != h and h2 == canon.canonical_hash(out, "class", g)


@pytest.mark.parametrize("model", ["pk_gpt2", "tp_mistral", "tp_qwen3", "pk_llama", "tp_gptneox"])
def test_edits_apply_to_other_exporters_documents(model):
    path = next(K.glob(f"conformance/models/{model}/*.gguf"))
    doc = bridge.load_document(path)
    dims = bridge.widths.gguf_dims(path, bridge.tools())
    shapes = canon.solved_shapes(doc)
    given = {t: list(dims[t.replace("{i}", "0")]) for t, s in shapes.items()
             if None in s and t.replace("{i}", "0") in dims}
    assert mu.check(doc, given) is None
    rng = random.Random(4)
    ok = 0
    for _ in range(40):
        r = mu.apply(doc, given, rng.choice(mu.EDIT_TYPES), rng)
        if r:
            assert mu.check(r[0], r[1]) is None
            ok += 1
    assert ok >= 20


def test_crossover_children_are_valid():
    rng = random.Random(5)
    a, ga = cs.lower(cs.default_config())
    b, gb = mu.apply(*cs.lower(dict(cs.default_config(), n_heads=4)), "insert", random.Random(6))[:2]
    ok = 0
    for _ in range(20):
        r = mu.crossover(a, ga, b, gb, rng)
        if r:
            assert mu.check(r[0], r[1]) is None
            ok += 1
    assert ok >= 15


def test_rewire_there_and_back():
    doc, given = cs.lower(cs.default_config())
    par = mu.apply(doc, given, "rewire", random.Random(0))
    assert par and par[2] == "rewire parallel"
    back = mu.apply(par[0], par[1], "rewire", random.Random(0))
    assert back and back[2] == "rewire sequential"
    assert canon.canonical_hash(back[0], given=back[1]) == canon.canonical_hash(doc, given=given)
    assert semdiff.distance(doc, par[0], given, par[1]) == 0.5


def test_insert_then_delete_returns_home():
    doc, given = cs.lower(cs.default_config())
    h = canon.canonical_hash(doc, given=given)
    rng = random.Random(7)
    for _ in range(10):
        ins = mu.apply(doc, given, "insert", rng)
        hits = 0
        for k in range(40):
            dele = mu.apply(ins[0], ins[1], "delete", random.Random(k))
            if dele and canon.canonical_hash(dele[0], given=dele[1]) == h:
                hits += 1
        assert hits > 0
