"""The slip decompiler: a keramos document to slip source that compiles back to the same document.

"The same" is the digest's sense (keramos IR_SPEC 3.10): equal after the '_' prose members are
removed, member order aside. The prose comes back as '#' comments, which the compiler drops. Every
node is written as one statement with the names it has in the document, so nothing is renamed; the
sugar is used only where compiling it gives back exactly what is there (a + b, x += o, @tensor
names, cycle[...]).
"""

import json
import math
import re

from .compiler import tensor_name
from .parser import RESERVED

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TENSOR_PATH = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.{}-]*$")
TENSOR_KEYS = {"weight", "bias", "router", "experts", "state_weight", "tied_to", "a", "b", "c", "d",
               "k_b", "v_b", "selection_bias"}
BINOPS = {"add": "+", "sub": "-", "mul": "*"}
SECTIONS = ("embed", "block", "head")
WIDTH = 100


class Unsupported(ValueError):
    pass


def _same(a, b):
    """Equality as JSON values: a boolean is never a number; 1 and 1.0 are the same number."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def tensor_sugar(s, section):
    """The @path that compiles to s in this section, or None."""
    pre = "blk.{i}." if section == "block" else ""
    if not isinstance(s, str) or not s.startswith(pre):
        return None
    rest = s[len(pre):]
    for cand in ([rest[:-len(".weight")]] if rest.endswith(".weight") else []) + [rest]:
        if cand and TENSOR_PATH.match(cand) and tensor_name(cand, section) == s:
            return "@" + cand
    return None


class Writer:
    def __init__(self, doc):
        self.doc = doc
        self.repeat = doc.get("block", {}).get("repeat") if isinstance(doc.get("block"), dict) else None

    # literals
    def scalar(self, v):
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            if not math.isfinite(v):
                raise Unsupported(f"{v} is not a JSON number")
            return repr(v)
        if isinstance(v, str):
            if IDENT.match(v) and v not in RESERVED:
                return v
            return json.dumps(v, ensure_ascii=False)
        raise Unsupported(f"unexpected value {v!r}")

    def key(self, k):
        return k if IDENT.match(k) else json.dumps(k, ensure_ascii=False)

    def lit(self, v, section, key=None, indent=0, col=None):
        """A literal starting at column col on a line indented by indent: on one line if it fits,
        else one entry per line at indent + 2, closed at indent."""
        one = self.lit1(v, section, key)
        if len(one) + (indent if col is None else col) <= WIDTH or not isinstance(v, (list, dict)):
            return one
        inner = indent + 2
        pad = " " * inner
        if isinstance(v, dict):
            pl = self.per_layer(v, section)
            if pl is not None:
                head, items = pl
                body = ",\n".join(pad + self.lit(x, section, None, inner) for x in items)
                return f"{head}[\n{body},\n{' ' * indent}]"
            body = ",\n".join(f"{pad}{self.key(k)}: {self.lit(x, section, k, inner, inner + len(self.key(k)) + 2)}"
                              for k, x in v.items())
            return "{\n" + body + ",\n" + " " * indent + "}"
        sub = key if key == "tensors" else None
        body = ",\n".join(pad + self.lit(x, section, sub, inner) for x in v)
        return "[\n" + body + ",\n" + " " * indent + "]"

    def per_layer(self, v, section):
        """('per_layer' | 'cycle', entries) when v is exactly {"per_layer": [...]}."""
        if list(v.keys()) != ["per_layer"] or not isinstance(v["per_layer"], list):
            return None
        xs = v["per_layer"]
        if section == "block" and self.repeat and len(xs) == self.repeat:
            for p in range(1, len(xs) // 2 + 1):
                if all(_same(xs[j], xs[j % p]) for j in range(len(xs))):
                    return "cycle", xs[:p]
        return "per_layer", xs

    def lit1(self, v, section, key=None):
        if isinstance(v, str) and (key in TENSOR_KEYS or key == "tensors"):
            s = tensor_sugar(v, section)
            if s:
                return s
        if isinstance(v, list):
            sub = key if key == "tensors" else None
            return "[" + ", ".join(self.lit1(x, section, sub) for x in v) + "]"
        if isinstance(v, dict):
            pl = self.per_layer(v, section)
            if pl is not None:
                head, items = pl
                return head + "[" + ", ".join(self.lit1(x, section) for x in items) + "]"
            return "{" + ", ".join(f"{self.key(k)}: {self.lit1(x, section, k)}" for k, x in v.items()) + "}"
        return self.scalar(v)

    # prose
    def comments(self, obj, indent):
        out = []
        for k, v in obj.items():
            if k.startswith("_"):
                text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                for ln in text.splitlines() or [""]:
                    out.append(" " * indent + "# " + ln if ln else " " * indent + "#")
        return out

    # nodes
    def name(self, n):
        if not isinstance(n, str) or not IDENT.match(n) or n in RESERVED:
            raise Unsupported(f"value name {n!r} has no slip spelling")
        return n

    def node(self, n, section, indent):
        lines = self.comments(n, indent)
        pad = " " * indent
        out = n.get("out")
        ins = n.get("in")
        if not isinstance(ins, list):
            raise Unsupported("a node without an in list")
        if isinstance(out, list):
            if len(out) < 2:
                raise Unsupported("an out list of fewer than two names")
            target = ", ".join(self.name(o) for o in out)
        else:
            target = self.name(out)
        attrs = {k: v for k, v in n.items() if k not in ("op", "in", "out") and not k.startswith("_")}
        if n["op"] in BINOPS and not attrs and len(ins) == 2 and isinstance(out, str):
            a, b = (self.name(x) for x in ins)
            sym = BINOPS[n["op"]]
            lines.append(f"{pad}{target} {sym}= {b}" if a == out else f"{pad}{target} = {a} {sym} {b}")
            return lines
        at = ""
        if "weight" in attrs:
            s = tensor_sugar(attrs["weight"], section)
            if s:
                at = " " + s
                del attrs["weight"]
        args = [self.name(x) for x in ins]
        head = f"{pad}{target} = {n['op']}("
        flat = [*args, *(f"{k}={self.lit1(v, section, k)}" for k, v in attrs.items())]
        one = head + ", ".join(flat) + ")" + at
        if len(one) <= WIDTH:
            lines.append(one)
            return lines
        inner = indent + 2
        parts = []
        if args:
            parts.append(" " * inner + ", ".join(args))
        for k, v in attrs.items():
            parts.append(f"{' ' * inner}{k}={self.lit(v, section, k, inner, inner + len(k) + 1)}")
        lines.append(head + "\n" + ",\n".join(parts) + ",\n" + pad + ")" + at)
        return lines

    def slot(self, s, indent):
        lines = self.comments(s, indent)
        attrs = {k: v for k, v in s.items() if k not in ("name", "kind") and not k.startswith("_")}
        head = f"{' ' * indent}state {self.name(s['name'])} = {self.name(s['kind'])}("
        one = head + ", ".join(f"{k}={self.lit1(v, 'block', k)}" for k, v in attrs.items()) + ")"
        if len(one) <= WIDTH:
            return lines + [one]
        inner = indent + 2
        parts = [f"{' ' * inner}{k}={self.lit(v, 'block', k, inner, inner + len(k) + 1)}" for k, v in attrs.items()]
        return lines + [head + "\n" + ",\n".join(parts) + ",\n" + " " * indent + ")"]

    # document
    def write(self):
        d = self.doc
        lines = self.comments(d, 0)
        if lines:
            lines.append("")
        for k, v in d.items():
            if k.startswith("_"):
                continue
            if k in SECTIONS:
                if lines and lines[-1] != "":
                    lines.append("")
                lines.extend(self.section(k, v))
                lines.append("")
            else:
                lines.append(f"{k} {self.lit(v, None, k, 0, len(k) + 1)}")
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines) + "\n"

    def section(self, name, s):
        lines = self.comments(s, 0)
        if name == "block":
            kw = [f"repeat={self.scalar(s['repeat'])}"]
            for k, v in s.items():
                if k in ("repeat", "state", "nodes") or k.startswith("_"):
                    continue
                kw.append(f"{k}={self.lit1(v, 'block', k)}")
            if "state" in s and not s["state"]:
                kw.append("state=[]")
            lines.append(f"block({', '.join(kw)}) {{")
            for slot in s.get("state", []):
                lines.extend(self.slot(slot, 2))
            if s.get("state"):
                lines.append("")
        else:
            lines.append(f"{name} {{")
        for n in s.get("nodes", []):
            lines.extend(self.node(n, name, 2))
        lines.append("}")
        return lines


def decompile(doc):
    return Writer(doc).write()
