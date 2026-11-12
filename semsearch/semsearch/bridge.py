"""The bridge to keramos (validator, reference interpreter) and to slip (widths, diff, decompiler).

semsearch lives in the keramos repository (keramos/semsearch/), beside slip (keramos/slip/). It imports
slip from there and uses slip's own bridge to keramos, so both see one keramos checkout ($SLIP_KERAMOS
names another, as for slip).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # keramos/semsearch
KERAMOS = ROOT.parent                               # keramos
SLIP = KERAMOS / "slip"

if str(SLIP) not in sys.path:
    sys.path.insert(0, str(SLIP))

from slip import keramos as _k  # noqa: E402
from slip import widths as _widths  # noqa: E402
from slip import diff as _diff  # noqa: E402
from slip import run as _run  # noqa: E402
from slip.compiler import compile_source as _compile_source  # noqa: E402
from slip.decompiler import decompile as _decompile  # noqa: E402


def tools():
    """keramos's tools/ir_validate.py, as a module."""
    return _k.tools()


def numpy_ref():
    """keramos's reference interpreter, tools/ir_numpy.py."""
    tools()
    import ir_numpy
    return ir_numpy


widths = _widths
slip_diff = _diff
slip_run = _run
compile_source = _compile_source
decompile = _decompile
load_document = _k.load_document
digest = _k.digest


def doc_errors(doc):
    return tools().doc_errors(doc)


def width_errors(doc, given=None):
    """slip's width check: a list of (section, index, message); [] when every width agrees."""
    errs, _ = _widths.check(doc, given=given)
    return errs


def valid(doc, given=None):
    """keramos accepts the document and slip finds no width mismatch."""
    return not doc_errors(doc) and not width_errors(doc, given)


def corpus():
    """Every document in the keramos checkout that keramos accepts: examples and conformance models."""
    import glob
    paths = sorted(glob.glob(str(KERAMOS / "examples" / "*.ir.json"))) + sorted(
        glob.glob(str(KERAMOS / "conformance" / "models" / "*" / "*.gguf")))
    out = []
    for p in paths:
        try:
            d = load_document(p)
        except SystemExit:
            continue
        if not doc_errors(d):
            out.append((Path(p).name, d))
    return out
