"""The semantic diff: zero across what canon.py equates, symmetric, and one typed record per typed edit."""

import random

import pytest

from semsearch import bridge, config_space as cs, mutate as mu, semdiff

K = bridge.KERAMOS


def test_zero_for_one_graph_in_two_spellings():
    hand, _ = bridge.compile_source((K / "slip/examples/llama.slip").read_text())
    pk = bridge.load_document(K / "conformance/models/pk_llama/pk_llama.gguf")
    assert semdiff.edits(hand, pk) == (0, [])
    assert semdiff.text(hand, pk) == ""


@pytest.mark.parametrize("change", [{"window": 32}, {"rope_pairing": "neox"}, {"checkpoint_blocks": True},
                                    {"n_kv_heads": 2}])
def test_zero_for_aliases(change):
    a, ga = cs.lower(cs.default_config())
    b, gb = cs.lower(dict(cs.default_config(), **change))
    assert semdiff.distance(a, b, ga, gb) == 0


@pytest.mark.parametrize("change,types", [
    ({"window": 8}, {"attr"}), ({"tie_embeddings": False}, {"tie"}), ({"d_ff": 128}, {"shape"}),
    ({"n_layers": 3}, {"repeat"}), ({"qk_norm": True}, {"insert", "rewire"})])
def test_config_changes_read_as_typed_edits(change, types):
    a, ga = cs.lower(cs.default_config())
    b, gb = cs.lower(dict(cs.default_config(), **change))
    d, e = semdiff.edits(a, b, ga, gb)
    assert d > 0 and {x["type"] for x in e} == types


def test_symmetric_and_small_for_single_edits():
    a, ga = cs.lower(cs.default_config())
    rng = random.Random(0)
    for _ in range(40):
        r = mu.apply(a, ga, rng.choice(mu.EDIT_TYPES), rng)
        if not r:
            continue
        b, gb, desc = r
        d1 = semdiff.distance(a, b, ga, gb)
        d2 = semdiff.distance(b, a, gb, ga)
        assert d1 == d2
        # an inserted attention sublayer is the largest single edit: 8 nodes, an add, a rewire
        assert 0 < d1 <= 10, desc
        one_way = semdiff.edits(a, b, ga, gb)[0]
        assert d1 <= one_way <= d1 + 1.5, desc
