"""The slip compiler: a syntax tree to a keramos document.

What the sugar means:
- `a + b`, `a - b`, `a * b` are the ops add, sub and mul, with inputs in the order written.
- `x += e` is `x = x + e` (so the inputs are [x, e]); likewise -= and *=.
- A call or operator nested inside another becomes its own node, written to a fresh name t1, t2, ...
  that the document does not otherwise use, emitted just before the node that reads it.
- `@path` names a tensor: in the block it is "blk.{i}." + path, elsewhere just path; ".weight" is
  appended when the path holds no dot (@attn_norm is "blk.{i}.attn_norm.weight", @attn_norm.bias is
  "blk.{i}.attn_norm.bias").
- `op(...) @path` sets the node's weight.
- `per_layer[a, b, ...]` is {"per_layer": [a, b, ...]}; `cycle[a, b]` repeats its entries to the
  block's repeat count (the block only).
- `state kv = kv(heads=4, ...)` is a block state slot {"name": "kv", "kind": "kv", "heads": 4, ...}.
Everything else maps one to one: a node's inputs are its positional arguments, its attributes its
name=value arguments, its out the assigned names (a list when there are several).
"""

from .errors import SlipError
from .parser import BinOp, Call, Name, Section, State, parse

BINOPS = {"+": "add", "-": "sub", "*": "mul"}
SECTIONS = ("embed", "block", "head")
PREFIX = {"block": "blk.{i}.", "embed": "", "head": "", None: ""}


def tensor_name(path, section):
    return PREFIX[section] + path + ("" if "." in path else ".weight")


class Compiler:
    def __init__(self, tree):
        self.tree = tree
        self.used = set()
        self.counter = 0
        self.where = {}      # (section, node index) -> (line, col)
        self.slot_where = {}  # slot index -> (line, col)
        self.top_where = {}   # key -> (line, col)
        self.repeat = None

    # names
    def collect(self, e):
        if isinstance(e, Name):
            self.used.add(e.id)
        elif isinstance(e, Call):
            for a in e.args:
                self.collect(a)
        elif isinstance(e, BinOp):
            self.collect(e.left)
            self.collect(e.right)

    def fresh(self):
        while True:
            self.counter += 1
            n = f"t{self.counter}"
            if n not in self.used:
                self.used.add(n)
                return n

    # literals
    def lit(self, v, section, key=None):
        k = v.kind
        if k in ("num", "str", "bool", "null"):
            return v.value
        if k == "tensor":
            return tensor_name(v.value, section)
        if k == "list":
            return [self.lit(x, section) for x in v.value]
        if k == "obj":
            return {m: self.lit(x, section, m) for m, x in v.value}
        if k == "per_layer":
            return {"per_layer": [self.lit(x, section) for x in v.value]}
        if k == "cycle":
            if section != "block" or self.repeat is None:
                raise SlipError("cycle[...] needs the block's repeat count; use it inside the block", v.line, v.col)
            items = [self.lit(x, section) for x in v.value]
            if not items:
                raise SlipError("cycle[] needs at least one entry", v.line, v.col)
            return {"per_layer": [items[j % len(items)] for j in range(self.repeat)]}
        raise AssertionError(k)

    # document
    def compile(self):
        doc = {}
        for it in self.tree.items:  # every name the source writes or reads, so temporaries never collide
            if isinstance(it, Section):
                for st in it.body:
                    if not isinstance(st, State):
                        self.used.update(t.id for t in st.targets)
                        self.collect(st.expr)
        for it in self.tree.items:
            if isinstance(it, Section):
                if it.name in doc:
                    raise SlipError(f"the {it.name} section is given twice", it.line, it.col)
                doc[it.name] = self.section(it)
            else:
                if it.key in SECTIONS:
                    raise SlipError(f"{it.key} is a section: write {it.key} {{ ... }}", it.line, it.col)
                if it.key in doc:
                    raise SlipError(f"{it.key!r} is given twice", it.line, it.col)
                doc[it.key] = self.lit(it.value, None)
                self.top_where[it.key] = (it.line, it.col)
        return doc

    def section(self, s):
        out = {}
        if s.name == "block":
            kw = {}
            has_slots = any(isinstance(st, State) for st in s.body)
            for key, v, line, col in s.kwargs:
                if key == "nodes" or (key == "state" and (has_slots or v.kind != "list" or v.value)):
                    raise SlipError(f"the block's {key} is written in its body (only an empty state=[] "
                                    "may be given here)", line, col)
                kw[key] = (v, line, col)
            if "repeat" not in kw:
                raise SlipError("block needs repeat=N", s.line, s.col)
            rv = kw["repeat"][0]
            if rv.kind != "num" or not isinstance(rv.value, int) or isinstance(rv.value, bool) or rv.value < 1:
                raise SlipError("repeat must be a positive integer", rv.line, rv.col)
            self.repeat = rv.value
            out["repeat"] = rv.value
            slots = []
            for st in s.body:
                if isinstance(st, State):
                    slot = {"name": st.name.id, "kind": st.kind}
                    for key, v, line, col in st.kwargs:
                        if key in ("name", "kind"):
                            raise SlipError(f"a slot's {key} is written as state NAME = KIND(...)", line, col)
                        slot[key] = self.lit(v, "block", key)
                    self.slot_where[len(slots)] = (st.line, st.col)
                    slots.append(slot)
            if slots:
                out["state"] = slots
            for key, (v, _, _) in kw.items():
                if key != "repeat":
                    out[key] = self.lit(v, "block", key)
        nodes = []
        self.section_name = s.name
        for st in s.body:
            if isinstance(st, State):
                continue
            self.assign(st, nodes)
        out["nodes"] = nodes
        if s.name == "block":
            self.repeat = None
        return out

    # statements
    def emit(self, nodes, node, line, col):
        self.where[(self.section_name, len(nodes))] = (line, col)
        nodes.append(node)

    def assign(self, st, nodes):
        names = [t.id for t in st.targets]
        e = st.expr
        if st.op != "=":
            e = BinOp(st.op[0], st.targets[0], e, st.line, st.col)
        if isinstance(e, Name):
            raise SlipError("keramos has no copy op: write the producing op's output to this name instead",
                            e.line, e.col)
        if len(names) > 1 and not isinstance(e, Call):
            raise SlipError("several names can only be written by one op call, as in q, k = rope(...)",
                            st.line, st.col)
        self.node(e, names[0] if len(names) == 1 else names, nodes)

    def value(self, e, nodes):
        if isinstance(e, Name):
            return e.id
        return self.node(e, None, nodes)

    def node(self, e, out, nodes):
        """Emit e's inputs, then e itself, writing out (a fresh name when None); return the name."""
        if isinstance(e, BinOp):
            ins = [self.value(e.left, nodes), self.value(e.right, nodes)]
            out = out or self.fresh()
            self.emit(nodes, {"op": BINOPS[e.op], "in": ins, "out": out}, e.line, e.col)
            return out
        ins = [self.value(a, nodes) for a in e.args]
        out = out or self.fresh()
        n = {"op": e.op, "in": ins}
        for key, v, line, col in e.kwargs:
            if key in ("op", "in", "out"):
                raise SlipError(f"{key!r} is written by the call's shape, not as an attribute", line, col)
            n[key] = self.lit(v, self.section_name, key)
        if e.at is not None:
            if "weight" in n:
                raise SlipError("the weight is given twice (as weight= and as @)", e.at.line, e.at.col)
            n["weight"] = tensor_name(e.at.value, self.section_name)
        n["out"] = out
        self.emit(nodes, n, e.line, e.col)
        return out


def compile_source(src):
    """(document, compiler); the compiler carries the source position of every node."""
    c = Compiler(parse(src))
    return c.compile(), c
