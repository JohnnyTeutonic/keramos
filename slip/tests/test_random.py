"""Prototyping without weights: every op's widths are solved (recurrence, depth_route and list values
included), the sizes the graph leaves free are named once, and `run --random` runs any model."""

import glob
from pathlib import Path

import numpy as np
import pytest

from slip import compile_source, keramos, widths
from slip import run as slip_run
from slip.__main__ import main, problems

K = keramos.keramos_dir()
EX = Path(__file__).resolve().parents[1] / "examples"
MODELS = sorted(glob.glob(str(K / "conformance" / "models" / "*" / "*.gguf")))
RETENTION = (EX / "retention.slip").read_text(encoding="utf-8")


def _id(p):
    return Path(p).name


@pytest.mark.parametrize("path", MODELS, ids=_id)
def test_free_sizes_alone_fix_every_shape(path, tmp_path):
    """Given only the sizes the graph leaves free (taken from the model's file), the solved shapes are the
    file's, tensor for tensor, the model runs on random weights of them, and `--save`'s GGUF of them is one
    keramos accepts."""
    t = keramos.tools()
    doc = keramos.load_document(path)
    if t.doc_errors(doc):
        pytest.skip("keramos refuses the original")
    dims = widths.gguf_dims(path, t)
    _, w0 = widths.check(doc)
    shapes = []
    for name in w0.names():
        if None in w0.shape(name) and name not in w0.tied:
            inst = [i for i in w0.instances(name) if i in dims]
            shapes.append(name + "=" + "x".join(map(str, dims[inst[0]])))
    W, w = slip_run.random_weights(doc, shapes, seed=0)
    for name, a in W.items():
        if name in dims:
            if name in w.flat:  # a recurrence weight: its element count is fixed, its stored shape free
                assert a.size == int(np.prod(dims[name])), name
            else:
                assert list(a.shape) == dims[name], name
    assert [n for n in dims if n not in W and n not in w.tied] == []
    V = doc["model"]["vocab"]
    ids = [j % V for j in range(1, 6)]
    logits = slip_run.forward(doc, W, ids)
    assert logits.shape == (5, V) and np.all(np.isfinite(logits))
    # written out, keramos's file checks accept it, and it reads back to the same logits
    out = Path(tmp_path) / "m.gguf"
    slip_run.save_gguf(out, doc, W)
    assert t.file_errors(out) == []
    back = slip_run.forward(keramos.load_document(out), slip_run.load_weights(out), ids)
    assert np.array_equal(back.view(np.uint32), logits.view(np.uint32))


def test_the_free_sizes_are_named_once():
    _, w = widths.check(keramos.load_document(K / "conformance" / "models" / "pk_llama" / "pk_llama.gguf"))
    assert slip_run.free_sizes(w) == ["--shape @ffn_gate=N1"]  # gate, up and down share the hidden width
    with pytest.raises(SystemExit, match="give the free sizes: --shape @ffn_gate=N1"):
        slip_run.random_weights(compile_source(RETENTION)[0])


def test_a_new_architecture_runs_without_weights(capsys):
    """examples/retention.slip: a token mixer no GGUF holds, written as a recurrence and run on random
    weights; every shape but the FFN's hidden width comes from the graph."""
    assert main(["check", str(EX / "retention.slip")]) == 0
    capsys.readouterr()
    assert main(["run", str(EX / "retention.slip"), "--random", "--shape", "@ffn_up=64", "--generate", "3"]) == 0
    out = capsys.readouterr().out
    assert "random weights, seed 0: 33 tensors, 33,004 parameters" in out and "greedy" in out
    _, w = widths.check(compile_source(RETENTION)[0])
    assert w.shape("blk.{i}.retention_decay.weight") == [4]
    assert w.shape("blk.{i}.attn_v.weight") == [32, 32]  # v[h,f] with h 4, f 8


def test_the_seed_decides_the_weights():
    doc = compile_source(RETENTION)[0]
    a, _ = slip_run.random_weights(doc, ["@ffn_up=8"], seed=1)
    b, _ = slip_run.random_weights(doc, ["@ffn_up=8"], seed=1)
    c, _ = slip_run.random_weights(doc, ["@ffn_up=8"], seed=2)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["blk.0.attn_q.weight"], c["blk.0.attn_q.weight"])
    assert np.all(a["blk.1.ffn_norm.weight"] == 1)  # a gain starts at 1


def test_recurrence_widths_are_its_declarations():
    src = RETENTION.replace('"v[h,f]"]', '"v[h,e]"]').replace("index={h: 4, e: 8, f: 8}", "index={h: 4, e: 8, f: 4}") \
        .replace("shape: [4, 8, 8]", "shape: [4, 8, 4]")
    msgs = [m for _, _, m in problems(*compile_source(src))]
    assert msgs == []  # v is 4 x 8 = 32 either way; the readout y[h,f] is now 16 wide
    _, w = widths.check(compile_source(src)[0])
    assert w.shape("blk.{i}.attn_output.weight") == [32, 16]
    # k read straight from the 32-wide norm output, against a declaration k[h,e] of 4 x 4
    src = RETENTION.replace("linear(h) @attn_k,", "h,").replace("index={h: 4, e: 8, f: 8}", "index={h: 4, e: 4, f: 8}") \
        .replace("shape: [4, 8, 8]", "shape: [4, 4, 8]")
    (line, col, msg), = problems(*compile_source(src))
    assert src.splitlines()[line - 1].strip().startswith("a = recurrence(")
    assert "the width of h: 32" in msg and "16 (k[h,e] with index" in msg


def test_depth_route_router_is_solved():
    doc = keramos.load_document(K / "conformance" / "models" / "pk_mor" / "pk_mor.gguf")
    _, w = widths.check(doc)
    route = next(n for n in doc["block"]["nodes"] if n["op"] == "depth_route")
    assert w.shape(route["weight"]) == [route["n_rec"], doc["model"]["d"]]


def test_list_values_carry_widths():
    """Attention residuals: every source depth_attend reads (the list's tensors, then its own inputs) has
    one width, so the pseudo-query is solved."""
    doc = keramos.load_document(K / "conformance" / "models" / "pk_llama_attnres" / "pk_llama_attnres.gguf")
    _, w = widths.check(doc)
    da = [n for s in ("block", "head") for n in doc[s]["nodes"] if n["op"] == "depth_attend"]
    assert da and all(w.shape(n["weight"]) == [doc["model"]["d"]] for n in da)


def test_a_tensor_path_ends_at_a_brace():
    doc = compile_source(RETENTION)[0]
    rec = next(n for n in doc["block"]["nodes"] if n["op"] == "recurrence")
    assert rec["weights"] == {"g[h]": "blk.{i}.retention_decay.weight"}
