"""slip diff, and decompiling a run's export folder."""

import json
import re
import shutil
from pathlib import Path

import pytest

from slip import compile_source, decompile, keramos
from slip.__main__ import main, resolve
from slip.diff import diff

K = keramos.keramos_dir()
M = K / "conformance" / "models"


def doc(name):
    return keramos.load_document(next((M / name).glob("*.gguf")))


def renamed(d, pairs):
    """The document with value names changed, by way of its slip."""
    src = decompile(d)
    for a, b in pairs:
        src = re.sub(rf"\b{a}\b", b, src)
    return compile_source(src)[0]


def test_identical():
    a = doc("pk_llama")
    same, text = diff(a, a)
    assert same and text == ""


def test_value_names_are_spelling_only():
    a = doc("pk_llama")
    b = renamed(a, [("gu", "gated"), ("h2", "hn")])
    assert keramos.digest(a) != keramos.digest(b)
    same, text = diff(a, b)
    assert same and "-  gu = g * u" in text and "+  gated = g * u" in text


def test_informative_keys_defaults_and_number_spelling_are_ignored():
    a = doc("pk_llama")
    b = json.loads(json.dumps(a))
    for k in ("tokenizer", "inputs", "outputs", "modes", "family_hint"):
        b.pop(k, None)
    for n in b["block"]["nodes"]:
        n.pop("bias", None) if n.get("bias", 0) is None else None
        n.pop("softmax", None)
        if n["op"] == "rope":
            n["theta"] = 10000
    for s in b["block"]["state"]:
        s.pop("dtype", None)
    same, text = diff(a, b)
    assert same and text == ""


def test_tensor_names_aside_unless_asked():
    a = doc("pk_llama")
    b = json.loads(json.dumps(a).replace("attn_q.weight", "wq.weight"))
    assert diff(a, b) == (True, "")
    same, text = diff(a, b, tensor_names=True)
    assert not same and "@wq" in text


def test_a_change_of_size_is_shown():
    same, text = diff(doc("tp_qwen3"), doc("tp_qwen3_hd16"))
    assert not same
    assert "-model {vocab: 24, d: 16," in text and "+model {vocab: 24, d: 32," in text
    assert "+    head_dim=16," in text


def test_diff_command(capsys):
    assert main(["diff", str(M / "pk_llama"), str(M / "pk_llama")]) == 0
    assert "the same document" in capsys.readouterr().out
    assert main(["diff", str(M / "pk_llama"), str(M / "pk_llama_attnres")]) == 1
    out = capsys.readouterr().out
    assert "+  x = depth_attend(bank) @_" in out and out.rstrip().endswith("different graphs, tensor names aside")


def test_decompile_a_run_folder(tmp_path, capsys):
    """A run's export folder (its .ir.json beside the GGUF) decompiles in one command, digest stamped."""
    run = tmp_path / "run"
    run.mkdir()
    shutil.copy(M / "pk_llama" / "pk_llama.gguf", run / "pk_llama.gguf")
    d = doc("pk_llama")
    (run / "pk_llama.ir.json").write_text(json.dumps(d), encoding="utf-8")
    assert resolve(run) == run / "pk_llama.ir.json"
    out = tmp_path / "pk_llama.slip"
    assert main(["decompile", str(run), "-o", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith(f"# decompiled by slip from pk_llama.ir.json; keramos digest {keramos.digest(d)}")
    assert keramos.digest(compile_source(text)[0]) == keramos.digest(d)
    (run / "pk_llama.ir.json").unlink()
    assert resolve(run) == run / "pk_llama.gguf"  # no .ir.json: the GGUF that carries one


def test_a_folder_of_several_documents_is_refused(tmp_path):
    for n in ("a", "b"):
        (tmp_path / f"{n}.ir.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="holds 2 documents"):
        resolve(tmp_path)


def test_diff_reads_slip_files(capsys):
    slip = Path(__file__).resolve().parents[1] / "examples" / "llama.slip"
    assert main(["diff", str(slip), str(slip)]) == 0
