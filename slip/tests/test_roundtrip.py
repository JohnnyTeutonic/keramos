"""Every keramos document in the keramos checkout (its examples and the documents embedded in the
conformance suite's models) decompiles to slip that compiles back to the same document: the same
canonical form and so the same digest (IR_SPEC 3.10). Decompiling the result again gives the same text."""

import glob
from pathlib import Path

import pytest

from slip import compile_source, decompile
from slip import keramos

K = keramos.keramos_dir()
PATHS = sorted(glob.glob(str(K / "examples" / "*.ir.json"))) + sorted(
    glob.glob(str(K / "conformance" / "models" / "*" / "*.gguf")))


def _id(p):
    return Path(p).name


def test_corpus_is_there():
    assert len(PATHS) >= 40


@pytest.mark.parametrize("path", PATHS, ids=_id)
def test_roundtrip(path):
    doc = keramos.load_document(path)
    src = decompile(doc)
    back, _ = compile_source(src)
    assert keramos.canonical(back) == keramos.canonical(doc)
    assert keramos.digest(back) == keramos.digest(doc)
    assert decompile(back) == decompile(_strip_prose(doc))  # prose aside, a fixed point


def _strip_prose(v):
    if isinstance(v, dict):
        return {k: _strip_prose(x) for k, x in v.items() if not k.startswith("_")}
    if isinstance(v, list):
        return [_strip_prose(x) for x in v]
    return v


@pytest.mark.parametrize("path", PATHS, ids=_id)
def test_compiled_documents_keep_their_verdict(path):
    """A document keramos accepts compiles, from its slip, to a document keramos accepts."""
    doc = keramos.load_document(path)
    if keramos.tools().doc_errors(doc):
        pytest.skip("keramos refuses the original (a version 0 or refusal fixture)")
    back, _ = compile_source(decompile(doc))
    assert keramos.tools().doc_errors(back) == []
