"""The architectural difference between two keramos documents, as a diff of their slip.

Both documents are written in one spelling before they are decompiled and compared:
- prose (`_` members) and the informative keys (family_hint, tokenizer, inputs, outputs, modes,
  ir_version, attention's softmax, moe_ffn's routing) are left out;
- an optional attribute at its default (IR_SPEC 3.5: linear's bias null, a kv slot's dtype f32,
  rope's kv_heads = heads, ...) is left out;
- a number is written as the shortest decimal of the f32 it holds (IR_SPEC 3.5), so 1e-05 and 0.00001,
  10000 and 10000.0 read the same;
- every tensor name is `@_`, unless the tensor names are asked for.
The verdict is keramos's own: two documents are the same graph when their `structure` agrees (IR_SPEC
rule 38: value and slot names, operand order of add and mul, and defaults aside). Value names are kept in
the text, so two graphs that are the same up to names show as the same graph with a diff of spellings.
"""

import copy
import difflib
import struct

from . import keramos
from .decompiler import TENSOR_KEYS, _same, decompile

NESTED = ("position_bias", "relative_kv", "freq_factors")
INFORMATIVE_TOP = ("family_hint", "tokenizer", "inputs", "outputs", "modes", "ir_version")


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def _number(x):
    """The shortest decimal of the f32 that x holds."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return x
    if isinstance(x, int) and abs(x) < 2 ** 24:
        return x
    v = _f32(x)
    if v == int(v) and abs(v) < 2 ** 53:
        return int(v)
    for digits in range(1, 18):
        s = float(f"{v:.{digits}g}")
        if _f32(s) == v:
            return s
    return v


def _clean(v):
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items() if not k.startswith("_")}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return _number(v)


def _mask(v, section):
    if isinstance(v, str):
        return "blk.{i}._.weight" if section == "block" else "_.weight"
    if isinstance(v, dict):
        return {k: _mask(x, section) for k, x in v.items()}
    return v


def _drop_defaults(n, defaults, informative):
    op = n.get("op")
    for k in informative.get(op, ()):
        n.pop(k, None)
    for k, dv in defaults.get(op, {}).items():
        if k in n and _same(n[k], dv):
            del n[k]
    if op == "rope":
        if "kv_heads" in n and n["kv_heads"] == n.get("heads"):
            del n["kv_heads"]
        if "n_rot" in n and n["n_rot"] == n.get("head_dim"):
            del n["n_rot"]
    if n.get("apply") is True:
        del n["apply"]


def architecture(doc, tensor_names=False):
    """The document in the spelling the diff compares."""
    t = keramos.tools()
    defaults = {op: {k: _number(v) for k, v in d.items()} for op, d in t.NODE_DEFAULTS.items()}
    d = copy.deepcopy(doc)
    for k in INFORMATIVE_TOP:
        d.pop(k, None)
    d = _clean(d)
    for slot in (d.get("block") or {}).get("state") or []:
        if isinstance(slot, dict) and slot.get("kind") == "kv" and slot.get("dtype") == "f32":
            del slot["dtype"]
    for section in ("embed", "block", "head"):
        for n in (d.get(section) or {}).get("nodes", []):
            if not isinstance(n, dict):
                continue
            _drop_defaults(n, defaults, t.INFORMATIVE_ATTRS)
            if not tensor_names:
                for k in list(n):
                    if k in TENSOR_KEYS or k in NESTED:
                        n[k] = _mask(n[k], section)
    return d


def same_graph(a, b, tensor_names=False):
    """keramos's rule 38 on the two documents (tensor names masked unless asked for)."""
    s = keramos.tools().structure
    return s(architecture(a, tensor_names)) == s(architecture(b, tensor_names))


def diff(a, b, name_a="a", name_b="b", tensor_names=False, context=3):
    """(same graph, text): text is a unified diff of the two documents' slip, empty when it is the same."""
    ta = decompile(architecture(a, tensor_names)).splitlines()
    tb = decompile(architecture(b, tensor_names)).splitlines()
    text = "" if ta == tb else "\n".join(difflib.unified_diff(ta, tb, name_a, name_b, lineterm="", n=context)) + "\n"
    return same_graph(a, b, tensor_names), text
