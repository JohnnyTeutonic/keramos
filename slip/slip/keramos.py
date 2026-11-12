"""The bridge to keramos: its validator (digest, rules, GGUF reading) and its reference interpreter.

slip lives in the keramos repository (keramos/slip/), so keramos is the folder above slip's; $SLIP_KERAMOS
names another checkout.
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def keramos_dir():
    d = Path(os.environ.get("SLIP_KERAMOS") or ROOT.parent)
    if not (d / "tools" / "ir_validate.py").exists():
        raise SystemExit(f"keramos not found at {d}: set SLIP_KERAMOS to a keramos checkout")
    return d


def tools():
    p = str(keramos_dir() / "tools")
    if p not in sys.path:
        sys.path.insert(0, p)
    import ir_validate
    return ir_validate


def load_document(path):
    """A document from a .json file or from the paperkiln.ir key of a GGUF."""
    path = Path(path)
    if path.suffix == ".gguf":
        meta = tools().read_gguf(path)[0]
        if "paperkiln.ir" not in meta:
            raise SystemExit(f"{path}: no embedded keramos document (paperkiln.ir)")
        return json.loads(meta["paperkiln.ir"][1])
    return json.loads(path.read_text(encoding="utf-8"))


def digest(doc):
    return tools().digest(doc)


def canonical(doc):
    return tools().canonical(doc)


_WHERE = re.compile(r"\b(embed|block|head)(?:\.nodes)?\[(\d+)\]")
_SLOT = re.compile(r"\bblock\.state\[(\d+)\]")


def errors(doc, compiler=None):
    """keramos's document rules, each as (line, col, message); line is None where no node is named."""
    out = []
    for msg in tools().doc_errors(doc):
        line = col = None
        if compiler is not None:
            m = _SLOT.search(msg)
            if m and int(m.group(1)) in compiler.slot_where:
                line, col = compiler.slot_where[int(m.group(1))]
            else:
                m = _WHERE.search(msg)
                if m and (m.group(1), int(m.group(2))) in compiler.where:
                    line, col = compiler.where[(m.group(1), int(m.group(2)))]
        m = _UNKNOWN_OP.search(msg)
        if m:
            import difflib
            ops = re.findall(r"'([a-z_0-9]+)'", m.group(2))
            near = difflib.get_close_matches(m.group(1), ops, n=3, cutoff=0.5)
            msg = msg[:m.start()] + f'op "{m.group(1)}" is not a keramos op' + (
                f"; did you mean {' or '.join(near)}?" if near else "")
        out.append((line, col, msg))
    return out


_UNKNOWN_OP = re.compile(r'op "([^"]*)" is not one of (\[.*\])')
