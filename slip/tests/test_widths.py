"""Width checking: no false alarm on any keramos document, with or without its weights; a mismatch is
reported at the source line of the node that meets it, with where each number came from."""

import copy
import glob
from pathlib import Path

import pytest

from slip import compile_source, keramos, widths
from slip.__main__ import main, problems

K = keramos.keramos_dir()
MODELS = sorted(glob.glob(str(K / "conformance" / "models" / "*" / "*.gguf")))
PATHS = sorted(glob.glob(str(K / "examples" / "*.ir.json"))) + MODELS
LLAMA = (Path(__file__).resolve().parents[1] / "examples" / "llama.slip").read_text(encoding="utf-8")
PK_LLAMA = K / "conformance" / "models" / "pk_llama" / "pk_llama.gguf"


def _id(p):
    return Path(p).name


@pytest.mark.parametrize("path", PATHS, ids=_id)
def test_no_false_alarms(path):
    doc = keramos.load_document(path)
    if keramos.tools().doc_errors(doc):
        pytest.skip("keramos refuses the original")
    assert widths.check(doc)[0] == []
    if path.endswith(".gguf"):
        errs, w = widths.check(doc, widths.gguf_dims(path, keramos.tools()))
        assert errs == []


def test_swapped_weights_are_caught_in_every_model():
    """In each model, two block linear weights of different shapes trade places: always a mismatch."""
    t = keramos.tools()
    tried = 0
    for path in MODELS:
        doc = keramos.load_document(path)
        if t.doc_errors(doc):
            continue
        dims = widths.gguf_dims(path, t)
        lins = [n for n in doc["block"]["nodes"] if n["op"] == "linear" and isinstance(n.get("weight"), str)]
        shape = [dims.get(n["weight"].replace("{i}", "0")) for n in lins]
        pair = next(((i, j) for i in range(len(lins)) for j in range(i + 1, len(lins))
                     if shape[i] and shape[j] and shape[i] != shape[j] and shape[i][::-1] != shape[j]), None)
        if pair is None:
            continue
        m = copy.deepcopy(doc)
        ml = [n for n in m["block"]["nodes"] if n["op"] == "linear" and isinstance(n.get("weight"), str)]
        a, b = pair
        ml[a]["weight"], ml[b]["weight"] = ml[b]["weight"], ml[a]["weight"]
        assert widths.check(m, dims)[0], path
        tried += 1
    assert tried >= 25


def test_shapes_are_solved_from_the_attributes_alone():
    _, w = widths.check(compile_source(LLAMA)[0])
    assert w.shape("blk.{i}.attn_q.weight") == [16, 16]
    assert w.shape("output.weight") == [15, 16]
    assert w.shape("blk.{i}.ffn_up.weight") == [None, 16]  # the hidden width: only the weights say
    _, w = widths.check(compile_source(LLAMA)[0], widths.gguf_dims(PK_LLAMA, keramos.tools()))
    assert w.shape("blk.{i}.ffn_up.weight") == [48, 16]


def _problems(src, dims=None):
    doc, comp = compile_source(src)
    return problems(doc, comp, dims)


def test_mismatch_without_weights_points_at_the_node():
    src = LLAMA.replace("heads=2, head_dim=8, theta", "heads=4, head_dim=8, theta")
    (line, col, msg), = _problems(src)
    assert line == 25 and "width of q: 32" in msg and "heads 4 x head_dim 8 of rope" in msg \
        and "heads 2 x head_dim 8 of attention" in msg


def test_mismatch_against_the_weights_points_at_the_node():
    src = LLAMA.replace("* linear(h) @ffn_up", "* h")
    errs = _problems(src, widths.gguf_dims(PK_LLAMA, keramos.tools()))
    assert errs and errs[0][0] == 30
    assert "48" in errs[0][2] and "16" in errs[0][2] and "ffn_gate.weight is [48, 16] in the weights" in errs[0][2]


def test_residual_and_logits_widths():
    src = LLAMA.replace("x += linear(a) @attn_output", "x += linear(a) @attn_output\n  x = linear(x) @widen")
    dims = dict(widths.gguf_dims(PK_LLAMA, keramos.tools()))
    dims.update({f"blk.{i}.widen.weight": [32, 16] for i in range(2)})
    msgs = [m for _, _, m in _problems(src, dims)]
    assert any("model.d" in m or "as one application writes it" in m for m in msgs)


def test_check_command(capsys):
    assert main(["check", str(PK_LLAMA)]) == 0
    assert main(["check", str(Path(__file__).resolve().parents[1] / "examples" / "llama.slip"),
                 "--weights", str(PK_LLAMA), "--shapes"]) == 0
    assert "blk.{i}.ffn_down.weight [16, 48]" in capsys.readouterr().out
