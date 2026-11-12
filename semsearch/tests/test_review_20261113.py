"""Counterexamples from the independent review of 13 Nov 2026 (Explore agent), kept as tests."""

import copy
import random

import pytest

from semsearch import bridge, canon, config_space as cs, fingerprint as fp, mutate as mu


def _doc(block_nodes, head_nodes=None, repeat=2, state=None, d=32):
    return {
        "ir_version": 1, "family_hint": "test",
        "model": {"vocab": 32, "d": d, "n_layers": repeat, "ctx": 32, "dtype": "f32"},
        "embed": {"nodes": [{"op": "embedding", "in": ["tokens"], "weight": "token_embd.weight", "out": "x"}]},
        "block": {"repeat": repeat, "state": state or [], "nodes": block_nodes},
        "head": {"nodes": head_nodes or [
            {"op": "rmsnorm", "in": ["x"], "weight": "output_norm.weight", "eps": 1e-05, "out": "hh"},
            {"op": "linear", "in": ["hh"], "weight": "output.weight", "out": "logits"}]},
    }


def test_per_layer_apply_reads_the_previous_applications_write():
    """Application 1 reads h from application 0's last node C: C must count."""
    A = {"op": "linear", "in": ["x"], "weight": "blk.{i}.w1.weight", "apply": {"per_layer": [True, False]},
         "out": "h"}
    B = {"op": "add", "in": ["x", "h"], "out": "x"}
    C = {"op": "linear", "in": ["x"], "weight": "blk.{i}.w2.weight", "out": "h"}
    d1 = _doc([A, B, C])
    d2 = _doc([A, B, {"op": "relu", "in": ["x"], "out": "h"}])
    d3 = _doc([A, B])
    for d in (d1, d2, d3):
        if bridge.doc_errors(d):
            pytest.skip("keramos refuses the construction: " + bridge.doc_errors(d)[0])
    hs = {canon.canonical_hash(d) for d in (d1, d2, d3)}
    assert len(hs) == 3


def _sharing(pattern):
    """sub(a1, a2) + sub(b1, b2), a = linear of x, b = linear of y, tensors X and Y read per `pattern`."""
    B = "blk.{i}."
    t = {k: B + v + ".weight" for k, v in pattern.items()}
    nodes = [
        {"op": "rmsnorm", "in": ["x"], "weight": B + "n1.weight", "eps": 1e-05, "out": "hx"},
        {"op": "rmsnorm", "in": ["x"], "weight": B + "n2.weight", "eps": 1e-05, "out": "hy"},
        {"op": "linear", "in": ["hx"], "weight": t["a1"], "out": "a1"},
        {"op": "linear", "in": ["hx"], "weight": t["a2"], "out": "a2"},
        {"op": "linear", "in": ["hy"], "weight": t["b1"], "out": "b1"},
        {"op": "linear", "in": ["hy"], "weight": t["b2"], "out": "b2"},
        {"op": "sub", "in": ["a1", "a2"], "out": "da"},
        {"op": "sub", "in": ["b1", "b2"], "out": "db"},
        {"op": "add", "in": ["da", "db"], "out": "s"},
        {"op": "add", "in": ["x", "s"], "out": "x"},
    ]
    return _doc(nodes, repeat=1)


def test_which_readers_share_a_tensor_counts_beyond_one_round():
    a = _sharing({"a1": "X", "b1": "X", "a2": "Y", "b2": "Y"})   # (X - Y) hx + (X - Y) hy
    b = _sharing({"a1": "X", "b2": "X", "a2": "Y", "b1": "Y"})   # (X - Y) hx + (Y - X) hy
    assert bridge.valid(a) and bridge.valid(b)
    assert not fp.match(fp.fingerprint(a), fp.fingerprint(b))
    assert canon.canonical_hash(a) != canon.canonical_hash(b)
    # and a renaming of one pattern still hashes alike
    c = _sharing({"a1": "P", "b1": "P", "a2": "Q", "b2": "Q"})
    assert canon.canonical_hash(a) == canon.canonical_hash(c)


def test_rope_pairing_not_merged_when_the_head_reads_the_rotated_q():
    a, ga = cs.lower(cs.default_config())
    docs = []
    for pairing in ("interleaved", "neox"):
        d = copy.deepcopy(a)
        next(n for n in d["block"]["nodes"] if n["op"] == "rope")["pairing"] = pairing
        d["head"]["nodes"].insert(0, {"op": "add", "in": ["x", "q"], "out": "x"})
        docs.append(d)
    assert all(bridge.valid(d, ga) for d in docs)
    assert canon.canonical_hash(docs[0], "class", ga) != canon.canonical_hash(docs[1], "class", ga)


def test_rope_pairing_not_merged_when_the_q_projection_is_shared():
    a, ga = cs.lower(cs.default_config())
    docs = []
    for pairing in ("interleaved", "neox"):
        d = copy.deepcopy(a)
        next(n for n in d["block"]["nodes"] if n["op"] == "rope")["pairing"] = pairing
        next(n for n in d["block"]["nodes"] if n.get("weight") == "blk.{i}.attn_v.weight")["weight"] = \
            "blk.{i}.attn_q.weight"
        docs.append(d)
    assert all(bridge.valid(d, ga) for d in docs)
    assert canon.canonical_hash(docs[0], "class", ga) != canon.canonical_hash(docs[1], "class", ga)


def test_repeat_edit_on_eight_applications():
    doc, given = cs.lower(dict(cs.default_config(), n_layers=4, repeats=2))
    assert doc["block"]["repeat"] == 8
    for k in range(10):
        r = mu.apply(doc, given, "repeat", random.Random(k))
        assert r is not None and r[0]["block"]["repeat"] == 7


def test_identity_weight_index_kept_under_tensor_index():
    for name, doc in bridge.corpus():
        if any("tensor_index" in n for n in doc["block"]["nodes"]):
            try:
                c = canon.canonical_document(doc)
            except canon.NotEmittable:
                continue
            assert bridge.doc_errors(c) == []
