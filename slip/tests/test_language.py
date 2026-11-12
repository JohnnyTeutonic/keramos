"""The language: what each piece of sugar compiles to, the errors, and a hand-written model run on real weights."""

from pathlib import Path

import numpy as np
import pytest

from slip import SlipError, compile_source, decompile
from slip import keramos

ROOT = Path(__file__).resolve().parents[1]
HEAD = "ir_version 1\nfamily_hint t\nmodel {vocab: 4, d: 4, n_layers: 3, ctx: 8}\n"


def nodes(src, section="block"):
    doc, _ = compile_source(src)
    return doc[section]["nodes"]


def block(body, repeat=3):
    return HEAD + "embed {\n  x = embedding(tokens) @token_embd\n}\n" + \
        f"block(repeat={repeat}) {{\n{body}\n}}\nhead {{\n  logits = linear(x) @output\n}}\n"


def test_operators_and_updates():
    ns = nodes(block("  y = x * x\n  x += y\n  x -= y\n  z = y - x"))
    assert [(n["op"], n["in"], n["out"]) for n in ns] == [
        ("mul", ["x", "x"], "y"), ("add", ["x", "y"], "x"), ("sub", ["x", "y"], "x"), ("sub", ["y", "x"], "z")]


def test_precedence_and_parentheses():
    ns = nodes(block("  y = x + x * x\n  z = (x + x) * x"))
    assert [(n["op"], n["in"], n["out"]) for n in ns] == [
        ("mul", ["x", "x"], "t1"), ("add", ["x", "t1"], "y"),
        ("add", ["x", "x"], "t2"), ("mul", ["t2", "x"], "z")]


def test_nested_calls_get_fresh_names_in_order():
    ns = nodes(block("  t1 = silu(x)\n  x += linear(silu(linear(t1) @a) * linear(t1) @b) @c"))
    assert [(n["op"], n["in"], n["out"]) for n in ns] == [
        ("silu", ["x"], "t1"),
        ("linear", ["t1"], "t2"), ("silu", ["t2"], "t3"), ("linear", ["t1"], "t4"),
        ("mul", ["t3", "t4"], "t5"), ("linear", ["t5"], "t6"), ("add", ["x", "t6"], "x")]
    assert ns[1]["weight"] == "blk.{i}.a.weight" and ns[5]["weight"] == "blk.{i}.c.weight"


def test_temporaries_avoid_names_used_in_any_section():
    src = HEAD + "embed {\n  x = embedding(tokens) @token_embd\n}\n" + \
        "block(repeat=3) {\n  x += silu(x)\n}\nhead {\n  t1 = silu(x)\n  logits = linear(t1) @output\n}\n"
    assert nodes(src)[0]["out"] == "t2"


def test_tensor_paths():
    ns = nodes(block("  y = layernorm(x, bias=@n.bias, eps=1e-5) @n\n  z = linear(y, weight=\"exact.name\")"))
    assert ns[0]["weight"] == "blk.{i}.n.weight" and ns[0]["bias"] == "blk.{i}.n.bias"
    assert ns[1]["weight"] == "exact.name"
    assert nodes(block(""), "head")[0]["weight"] == "output.weight"


def test_per_layer_and_cycle():
    ns = nodes(block("  y = affine(x, scale=cycle[1, 2], shift=per_layer[0, 0, 1])", repeat=5)
               .replace("per_layer[0, 0, 1]", "per_layer[0, 0, 1, 0, 0]"))
    assert ns[0]["scale"] == {"per_layer": [1, 2, 1, 2, 1]}
    assert ns[0]["shift"] == {"per_layer": [0, 0, 1, 0, 0]}


def test_literals_are_json_values():
    ns = nodes(block('  y = expr(x, expr="a * 2", note2={k: [1, -2.5, true, null, "s p"]})'))
    assert ns[0]["note2"] == {"k": [1, -2.5, True, None, "s p"]}


def test_state_slots():
    doc, _ = compile_source(block("  state kv = kv(heads=1, kv_heads=1, head_dim=4, max_len=8, window=cycle[2, 0], sinks=0)"))
    assert doc["block"]["state"] == [{"name": "kv", "kind": "kv", "heads": 1, "kv_heads": 1, "head_dim": 4,
                                      "max_len": 8, "window": {"per_layer": [2, 0, 2]}, "sinks": 0}]


@pytest.mark.parametrize("src, msg, line, col", [
    (block("  y = silu(x"), "expected ',' or ')'", 9, 1),
    (block("  y = x"), "no copy op", 8, 7),
    (block("  a, b = x + x"), "one op call", 8, 3),
    (block("  y = silu(x, k=1, k=2)"), "given twice", 8, 20),
    (block("  y = silu(k=1, x)"), "inputs come before attributes", 8, 17),
    (block("  y = linear(x, weight=@a) @b"), "weight is given twice", 8, 28),
    (HEAD + "embed {\n  x = affine(x, scale=cycle[1], shift=0)\n}\n", "inside the block", 5, 23),
    (HEAD + "block {\n}\n", "'(' after block", 4, 7),
    (block('  y = "x"'), "expected a value or an op call", 8, 7),
])
def test_errors_point_at_the_source(src, msg, line, col):
    with pytest.raises(SlipError) as e:
        compile_source(src)
    assert msg in e.value.msg and (e.value.line, e.value.col) == (line, col)


def test_keramos_refusals_map_to_lines():
    doc, comp = compile_source(block("  y = rope(x, x, pos, heads=1, head_dim=4, rope_heads=all, pairing=neox)"))
    errs = keramos.errors(doc, comp)
    assert any(line == 8 and "theta" in msg for line, _, msg in errs)


def test_decompile_uses_the_sugar():
    doc, _ = compile_source(block("  y = rmsnorm(x, eps=1e-5) @n\n  x += y\n  x = y * x", repeat=4)
                            .replace("block(repeat=4) {", "block(repeat=4) {\n  state kv = kv(heads=1, kv_heads=1, "
                                     "head_dim=4, max_len=8, window=cycle[2, 2, 0], sinks=0)"))
    src = decompile(doc)
    assert "y = rmsnorm(x, eps=1e-05) @n" in src and "x += y" in src and "x = y * x" in src
    assert "window=per_layer[2, 2, 0, 2]" in src  # no period of at most half the repeat count
    doc2, _ = compile_source(src)
    assert keramos.digest(doc2) == keramos.digest(doc)


def test_hand_written_llama_matches_the_exported_document():
    from slip import run
    gguf = keramos.keramos_dir() / "conformance" / "models" / "pk_llama" / "pk_llama.gguf"
    W = run.load_weights(gguf)
    mine, _ = compile_source((ROOT / "examples" / "llama.slip").read_text(encoding="utf-8"))
    assert keramos.tools().doc_errors(mine) == []
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    a, b = run.forward(mine, W, ids), run.forward(keramos.load_document(gguf), W, ids)
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
