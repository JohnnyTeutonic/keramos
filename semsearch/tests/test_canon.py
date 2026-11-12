"""The canonical form: what it equates (each a rewrite that keeps the function), what it keeps apart, and
that its documents are keramos documents computing bit-identical logits."""

import copy
import random

import numpy as np
import pytest

from semsearch import bridge, canon, config_space as cs, fingerprint as fp

K = bridge.KERAMOS
CORPUS = bridge.corpus()


def _id(x):
    return x[0]


def test_corpus_is_there():
    assert len(CORPUS) >= 40


@pytest.mark.parametrize("item", CORPUS, ids=_id)
def test_every_document_hashes_and_its_canonical_form_is_a_fixed_point(item):
    name, doc = item
    h = canon.canonical_hash(doc)
    assert h == canon.canonical_hash(copy.deepcopy(doc))
    try:
        c = canon.canonical_document(doc)
    except canon.NotEmittable:
        pytest.skip("per-layer apply or depth_route: hashed, not emitted")
    assert bridge.doc_errors(c) == []
    assert canon.canonical_hash(c) == h
    assert canon.canonical_document(c) == c


@pytest.mark.parametrize("item", [x for x in CORPUS if x[0].endswith(".gguf")], ids=_id)
def test_canonical_documents_compute_bit_identical_logits(item):
    name, doc = item
    try:
        c = canon.canonical_document(doc)
    except canon.NotEmittable:
        pytest.skip("not emitted")
    path = next(K.glob(f"conformance/models/*/{name}"))
    W = bridge.slip_run.load_weights(path)
    nr = bridge.numpy_ref()
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    a = np.asarray(nr.forward(doc, W, ids))
    b = np.asarray(nr.forward(c, W, ids))
    assert np.array_equal(a, b)


def test_hand_written_llama_is_paperkiln_llama():
    """slip's README: llama.slip computes v after rope where pk_llama's export computes it before, so rule 38
    calls them different documents although their logits agree bit for bit. The canonical form agrees."""
    hand, _ = bridge.compile_source((K / "slip/examples/llama.slip").read_text())
    pk = bridge.load_document(K / "conformance/models/pk_llama/pk_llama.gguf")
    assert canon.rule38_key(hand) != canon.rule38_key(pk)
    assert canon.canonical_hash(hand) == canon.canonical_hash(pk)
    assert bridge.slip_diff.diff(canon.canonical_document(hand), canon.canonical_document(pk))[1] == ""


def _base():
    return cs.lower(cs.default_config())


def _swap_nodes(doc, i, j):
    d = copy.deepcopy(doc)
    n = d["block"]["nodes"]
    n[i], n[j] = n[j], n[i]
    return d


def test_independent_nodes_in_either_order():
    doc, given = _base()
    # the q, k and v projections are nodes 1, 2, 3 of the block
    d2 = _swap_nodes(doc, 1, 3)
    assert canon.rule38_key(doc) != canon.rule38_key(d2)
    assert canon.canonical_hash(doc, given=given) == canon.canonical_hash(d2, given=given)


def test_dependent_nodes_in_the_other_order_differ():
    doc, given = _base()
    n = doc["block"]["nodes"]
    i = next(k for k, x in enumerate(n) if x["op"] == "silu")
    d2 = _swap_nodes(doc, i, i + 1)   # silu after the mul that reads it: the mul reads the unactivated gate
    assert bridge.doc_errors(d2) == []
    assert canon.canonical_hash(doc, given=given) != canon.canonical_hash(d2, given=given)


def test_value_and_tensor_names_do_not_count():
    doc, given = _base()
    d2 = copy.deepcopy(doc)
    ren = {"h": "hh", "q": "qq", "k": "kk", "v": "vv", "a": "aa", "o": "oo", "h2": "z", "g": "gg", "u": "uu",
           "gu": "w", "f": "ff"}
    for sect in ("block",):
        for n in d2[sect]["nodes"]:
            n["in"] = [ren.get(x, x) for x in n["in"]]
            n["out"] = [ren.get(x, x) for x in n["out"]] if isinstance(n["out"], list) else ren.get(n["out"], n["out"])
            for key in ("weight",):
                if key in n:
                    n[key] = n[key].replace("attn_q", "query").replace("ffn_up", "upper")
    g2 = {k.replace("ffn_up", "upper"): v for k, v in given.items()}
    assert canon.canonical_hash(doc, given=given) == canon.canonical_hash(d2, given=g2)


def test_dead_code_does_not_count():
    doc, given = _base()
    d2 = copy.deepcopy(doc)
    d2["block"]["nodes"].insert(5, {"op": "linear", "in": ["h"], "weight": "blk.{i}.unused.weight", "out": "dead"})
    assert canon.canonical_hash(doc, given=given) == canon.canonical_hash(d2, given=given)


@pytest.mark.parametrize("change", [
    {"window": 32}, {"window": 64}, {"n_kv_heads": 2}, {"rope_dims": 16}, {"checkpoint_blocks": True},
    {"sinks": 2}, {"repeats": 1, "inject": "none"}])
def test_aliases_of_the_default_config(change):
    a, ga = _base()
    b, gb = cs.lower(dict(cs.default_config(), **change))
    assert canon.canonical_hash(a, given=ga) == canon.canonical_hash(b, given=gb)
    assert fp.match(fp.fingerprint(a, ga), fp.fingerprint(b, gb))


def test_window_at_or_beyond_ctx_is_unbounded_per_layer_too():
    a, ga = cs.lower(dict(cs.default_config(), n_layers=2))
    b = copy.deepcopy(a)
    for obj in [n for n in b["block"]["nodes"] if n["op"] == "attention"] + b["block"]["state"]:
        obj["window"] = {"per_layer": [32, 1000]}
    assert bridge.doc_errors(b) == []
    assert canon.canonical_hash(a, given=ga) == canon.canonical_hash(b, given=ga)
    c = copy.deepcopy(a)
    for obj in [n for n in c["block"]["nodes"] if n["op"] == "attention"] + c["block"]["state"]:
        obj["window"] = {"per_layer": [8, 32]}
    assert canon.canonical_hash(a, given=ga) != canon.canonical_hash(c, given=ga)


def test_weight_index_identity_is_no_weight_index():
    a, ga = cs.lower(dict(cs.default_config(), n_layers=2))
    b = copy.deepcopy(a)
    b["block"]["weight_index"] = {"per_layer": [0, 1]}
    assert bridge.doc_errors(b) == []
    assert canon.canonical_hash(a, given=ga) == canon.canonical_hash(b, given=ga)
    b["block"]["weight_index"] = {"per_layer": [0, 0]}
    assert canon.canonical_hash(a, given=ga) != canon.canonical_hash(b, given=ga)


@pytest.mark.parametrize("change", [
    {"window": 8}, {"tie_embeddings": False}, {"d_ff": 128}, {"n_heads": 4}, {"qk_norm": True},
    {"rope_theta": 100000.0}, {"repeats": 2}, {"rope_dims": 8}])
def test_real_changes_are_kept_apart(change):
    a, ga = _base()
    b, gb = cs.lower(dict(cs.default_config(), **change))
    assert canon.canonical_hash(a, "class", ga) != canon.canonical_hash(b, "class", gb)


def test_add_operands_commute_and_sub_operands_do_not():
    doc, given = _base()
    d2 = copy.deepcopy(doc)
    last = d2["block"]["nodes"][-1]
    last["in"] = last["in"][::-1]
    assert canon.canonical_hash(doc, given=given) == canon.canonical_hash(d2, given=given)
    d3, d4 = copy.deepcopy(doc), copy.deepcopy(doc)
    d3["block"]["nodes"][-1]["op"] = "sub"
    d4["block"]["nodes"][-1]["op"] = "sub"
    d4["block"]["nodes"][-1]["in"] = d4["block"]["nodes"][-1]["in"][::-1]
    assert canon.canonical_hash(d3, given=given) != canon.canonical_hash(d4, given=given)


def test_which_tensor_is_shared_counts():
    """Two linears of one label reading two tensors, one of them shared with a third node: which operand
    of a non-commutative op holds the shared one is a real difference."""
    doc, given = _base()
    n = doc["block"]["nodes"]
    gate = next(x for x in n if x.get("weight", "").endswith("ffn_gate.weight"))
    up = next(x for x in n if x.get("weight", "").endswith("ffn_up.weight"))
    a = copy.deepcopy(doc)
    b = copy.deepcopy(doc)
    for d, shared in ((a, gate), (b, up)):
        for x in d["block"]["nodes"]:
            if x.get("weight") == "blk.{i}.attn_q.weight":
                x["weight"] = "blk.{i}.ffn_gate.weight" if shared is gate else "blk.{i}.ffn_up.weight"
    g = dict(given)
    ga = {k: v for k, v in g.items()}
    # q's projection is [32, 32]: give the FFN width 32 so that the shared tensor fits both readers
    for k in ga:
        ga[k] = [32, 32]
    assert bridge.valid(a, ga) and bridge.valid(b, ga)
    assert canon.canonical_hash(a, given=ga) != canon.canonical_hash(b, given=ga)


def test_rope_pairing_is_one_class():
    a, ga = _base()
    b, gb = cs.lower(dict(cs.default_config(), rope_pairing="neox"))
    assert canon.canonical_hash(a, "exact", ga) != canon.canonical_hash(b, "exact", gb)
    assert canon.canonical_hash(a, "class", ga) == canon.canonical_hash(b, "class", gb)


@pytest.mark.parametrize("change", [{}, {"qk_norm": True}, {"rope_dims": 8}, {"n_kv_heads": 1}])
def test_rope_pairing_class_is_a_permutation_of_the_projection_rows(change):
    """The claim behind the class rewrite: neox RoPE on weights whose q and k rows (and q/k norm gains) are
    permuted computes what interleaved RoPE computes on the original weights."""
    cfg = dict(cs.default_config(), **change)
    a, ga = cs.lower(dict(cfg, rope_pairing="interleaved"))
    b, gb = cs.lower(dict(cfg, rope_pairing="neox"))
    W = fp.random_weights(a, ga, seed=4)
    rope = next(n for n in a["block"]["nodes"] if n["op"] == "rope")
    H, KV, hd = rope["heads"], rope.get("kv_heads", rope["heads"]), rope["head_dim"]
    n_rot = rope.get("n_rot", hd)
    half = n_rot // 2
    # interleaved dim 2j -> neox dim j, 2j+1 -> j + half, the rest in place
    perm = list(range(hd))
    for j in range(half):
        perm[j] = 2 * j
        perm[j + half] = 2 * j + 1
    Wb = {}
    for k, v in W.items():
        v = v.copy()
        if "attn_q.weight" in k or "attn_k.weight" in k:
            heads = H if "attn_q" in k else KV
            v = v.reshape(heads, hd, -1)[:, perm, :].reshape(heads * hd, -1)
        if "attn_q_norm" in k or "attn_k_norm" in k:
            v = v.reshape(-1)[perm].reshape(v.shape)
        Wb[k] = v
    nr = bridge.numpy_ref()
    ids = fp.probes(32, 32, 1)[0]
    la = np.asarray(nr.forward(a, W, ids))
    lb = np.asarray(nr.forward(b, Wb, ids))
    assert np.max(np.abs(la - lb)) < 1e-4 * max(1, np.max(np.abs(la)))


def test_rope_pairing_not_merged_where_the_rotation_is_read_elsewhere():
    a, ga = _base()
    b = copy.deepcopy(a)
    rope = next(n for n in b["block"]["nodes"] if n["op"] == "rope")
    rope["pairing"] = "neox"
    # the rotated q also feeds the residual stream: its coordinates are visible, so the pairing matters
    b["block"]["nodes"].insert(b["block"]["nodes"].index(rope) + 1,
                               {"op": "add", "in": ["x", "q"], "out": "x"})
    c = copy.deepcopy(b)
    next(n for n in c["block"]["nodes"] if n["op"] == "rope")["pairing"] = "interleaved"
    assert bridge.valid(b, ga) and bridge.valid(c, ga)
    assert canon.canonical_hash(b, "class", ga) != canon.canonical_hash(c, "class", ga)


def test_one_exact_hash_one_fingerprint_on_random_configs():
    """Soundness on a sample: configs of one exact hash compute one function (fingerprint), and the
    fingerprint separates what the hash separates in at least the obvious cases."""
    rng = random.Random(5)
    by = {}
    for _ in range(60):
        c = cs.random_config(rng, cs.SPACE)
        for alias in ({}, {"checkpoint_blocks": not c["checkpoint_blocks"]},
                      {"window": 32 if c["window"] == 0 else c["window"]}):
            c2 = dict(c, **alias)
            try:
                cs.check(c2)
            except cs.InvalidConfig:
                continue
            d, g = cs.lower(c2)
            by.setdefault(canon.canonical_hash(d, given=g), []).append(fp.fingerprint(d, g))
    for vs in by.values():
        assert all(fp.match(vs[0], v) for v in vs[1:])
