"""The slip parser: tokens to a syntax tree.

    document  := { NEWLINE | top }
    top       := SECTION body                         # embed, head
               | "block" "(" kwargs ")" body
               | IDENT literal                        # ir_version 1, model {...}, ...
    body      := "{" { NEWLINE | stmt } "}"
    stmt      := "state" IDENT "=" IDENT "(" kwargs ")"
               | targets ("=" | "+=" | "-=" | "*=") expr
    targets   := IDENT { "," IDENT }
    expr      := term { ("+" | "-") term }
    term      := factor { "*" factor }
    factor    := IDENT "(" args ")" [ TENSOR ] | IDENT | "(" expr ")"
    args      := [ arg { "," arg } [","] ]          # positional inputs and name=literal attributes
    literal   := NUMBER | "-" NUMBER | STRING | TENSOR | "true" | "false" | "null" | IDENT
               | "[" ... "]" | "{" key ":" literal ... "}" | ("per_layer" | "cycle") "[" ... "]"

Newlines end statements; inside parentheses, brackets and literals they are ignored.
"""

from dataclasses import dataclass, field

from .errors import SlipError
from .lexer import tokenize

RESERVED = {"true", "false", "null", "per_layer", "cycle"}


# ---------------------------------------------------------------- the tree

@dataclass
class Lit:
    kind: str          # num str bool null list obj per_layer cycle tensor
    value: object      # python scalar; list of Lit; list of (key, Lit); tensor path
    line: int
    col: int


@dataclass
class Name:
    id: str
    line: int
    col: int


@dataclass
class Call:
    op: str
    args: list         # expressions, in order
    kwargs: list       # (key, Lit, line, col)
    at: object         # Lit of kind tensor, or None
    line: int
    col: int


@dataclass
class BinOp:
    op: str            # + - *
    left: object
    right: object
    line: int
    col: int


@dataclass
class Assign:
    targets: list      # Name
    op: str            # = += -= *=
    expr: object
    line: int
    col: int


@dataclass
class State:
    name: Name
    kind: str
    kwargs: list
    line: int
    col: int


@dataclass
class Section:
    name: str
    kwargs: list
    body: list
    line: int
    col: int


@dataclass
class TopLit:
    key: str
    value: Lit
    line: int
    col: int


@dataclass
class Document:
    items: list = field(default_factory=list)


# ---------------------------------------------------------------- the parser

class Parser:
    def __init__(self, src):
        self.toks = tokenize(src)
        self.i = 0

    # token helpers
    def peek(self, k=0):
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def at(self, kind, value=None):
        t = self.peek()
        return t.kind == kind and (value is None or t.value == value)

    def expect(self, kind, what=None):
        t = self.peek()
        if t.kind != kind:
            found = "end of line" if t.kind == "NEWLINE" else "end of file" if t.kind == "EOF" else repr(t.text)
            raise SlipError(f"expected {what or kind}, found {found}", t.line, t.col)
        return self.next()

    def skip_nl(self):
        while self.at("NEWLINE"):
            self.next()

    def end_stmt(self):
        t = self.peek()
        if t.kind == "}":
            return
        if t.kind not in ("NEWLINE", "EOF"):
            raise SlipError(f"expected the end of the statement, found {t.text!r}", t.line, t.col)
        self.skip_nl()

    # document
    def document(self):
        doc = Document()
        self.skip_nl()
        while not self.at("EOF"):
            doc.items.append(self.top())
            self.end_stmt()
        return doc

    def top(self):
        t = self.expect("IDENT", "a top-level key or section (embed, block, head)")
        if t.value in ("embed", "head"):
            return Section(t.value, [], self.body(t.value), t.line, t.col)
        if t.value == "block":
            self.expect("(", "'(' after block, as in block(repeat=4)")
            kwargs = self.kwargs_only(")")
            return Section("block", kwargs, self.body("block"), t.line, t.col)
        return TopLit(t.value, self.literal(), t.line, t.col)

    def body(self, section):
        self.expect("{", f"'{{' to open the {section} section")
        stmts = []
        self.skip_nl()
        while not self.at("}"):
            if self.at("EOF"):
                t = self.peek()
                raise SlipError(f"the {section} section is not closed with '}}'", t.line, t.col)
            stmts.append(self.stmt(section))
            self.end_stmt()
        self.next()
        return stmts

    def stmt(self, section):
        t = self.peek()
        if t.kind == "IDENT" and t.value == "state" and self.peek(1).kind == "IDENT":
            self.next()
            name = self.expect("IDENT", "the slot's name")
            self.expect("=", "'=' after the slot's name")
            kind = self.expect("IDENT", "the slot's kind, as in kv(...)")
            self.expect("(", "'(' after the slot's kind")
            kwargs = self.kwargs_only(")")
            if section != "block":
                raise SlipError("state slots belong in the block section", t.line, t.col)
            return State(Name(name.value, name.line, name.col), kind.value, kwargs, t.line, t.col)
        targets = [self.name_tok()]
        while self.at(","):
            self.next()
            targets.append(self.name_tok())
        op = self.peek()
        if op.kind not in ("=", "+=", "-=", "*="):
            raise SlipError(f"expected '=' or an update such as '+=', found {op.text!r}", op.line, op.col)
        self.next()
        if op.kind != "=" and len(targets) != 1:
            raise SlipError(f"'{op.kind}' updates one value", op.line, op.col)
        return Assign(targets, op.kind, self.expr(), t.line, t.col)

    def name_tok(self):
        t = self.expect("IDENT", "a value name")
        if t.value in RESERVED:
            raise SlipError(f"{t.value!r} is reserved", t.line, t.col)
        return Name(t.value, t.line, t.col)

    # expressions
    def expr(self):
        left = self.term()
        while self.at("+") or self.at("-"):
            t = self.next()
            left = BinOp(t.kind, left, self.term(), t.line, t.col)
        return left

    def term(self):
        left = self.factor()
        while self.at("*"):
            t = self.next()
            left = BinOp("*", left, self.factor(), t.line, t.col)
        return left

    def factor(self):
        t = self.peek()
        if t.kind == "(":
            self.next()
            self.skip_nl()
            e = self.expr()
            self.skip_nl()
            self.expect(")", "')'")
            return e
        if t.kind != "IDENT":
            raise SlipError(f"expected a value or an op call, found {t.text!r}", t.line, t.col)
        self.next()
        if not self.at("("):
            if t.value in RESERVED:
                raise SlipError(f"{t.value!r} is reserved", t.line, t.col)
            return Name(t.value, t.line, t.col)
        self.next()
        args, kwargs = self.args(")")
        at = None
        if self.at("TENSOR"):
            a = self.next()
            at = Lit("tensor", a.value, a.line, a.col)
        return Call(t.value, args, kwargs, at, t.line, t.col)

    def args(self, close):
        args, kwargs, seen = [], [], set()
        self.skip_nl()
        while not self.at(close):
            t = self.peek()
            if t.kind == "IDENT" and self.peek(1).kind == "=":
                self.next()
                self.next()
                self.skip_nl()
                if t.value in seen:
                    raise SlipError(f"attribute {t.value!r} is given twice", t.line, t.col)
                seen.add(t.value)
                kwargs.append((t.value, self.literal(), t.line, t.col))
            else:
                if kwargs:
                    raise SlipError("inputs come before attributes", t.line, t.col)
                args.append(self.expr())
            self.skip_nl()
            if not self.at(close):
                self.expect(",", f"',' or '{close}'")
                self.skip_nl()
        self.next()
        return args, kwargs

    def kwargs_only(self, close):
        args, kwargs = self.args(close)
        if args:
            a = args[0]
            raise SlipError("only name=value attributes are allowed here", a.line, a.col)
        return kwargs

    # literals
    def literal(self):
        self.skip_nl()
        t = self.next()
        k = t.kind
        if k == "NUMBER":
            return Lit("num", t.value, t.line, t.col)
        if k == "-" and self.at("NUMBER"):
            n = self.next()
            return Lit("num", -n.value, t.line, t.col)
        if k == "STRING":
            return Lit("str", t.value, t.line, t.col)
        if k == "TENSOR":
            return Lit("tensor", t.value, t.line, t.col)
        if k == "[":
            return Lit("list", self.items("]"), t.line, t.col)
        if k == "{":
            return Lit("obj", self.members(), t.line, t.col)
        if k == "IDENT":
            if t.value == "true":
                return Lit("bool", True, t.line, t.col)
            if t.value == "false":
                return Lit("bool", False, t.line, t.col)
            if t.value == "null":
                return Lit("null", None, t.line, t.col)
            if t.value in ("per_layer", "cycle"):
                self.expect("[", f"'[' after {t.value}")
                return Lit(t.value, self.items("]"), t.line, t.col)
            return Lit("str", t.value, t.line, t.col)
        found = "end of line" if k == "NEWLINE" else repr(t.text)
        raise SlipError(f"expected a value (a number, string, list, object or name), found {found}", t.line, t.col)

    def items(self, close):
        out = []
        self.skip_nl()
        while not self.at(close):
            out.append(self.literal())
            self.skip_nl()
            if not self.at(close):
                self.expect(",", f"',' or '{close}'")
                self.skip_nl()
        self.next()
        return out

    def members(self):
        out, seen = [], set()
        self.skip_nl()
        while not self.at("}"):
            t = self.next()
            if t.kind not in ("IDENT", "STRING"):
                raise SlipError(f"expected a member name, found {t.text!r}", t.line, t.col)
            if t.value in seen:
                raise SlipError(f"member {t.value!r} is given twice", t.line, t.col)
            seen.add(t.value)
            self.expect(":", "':' after the member name")
            out.append((t.value, self.literal()))
            self.skip_nl()
            if not self.at("}"):
                self.expect(",", "',' or '}'")
                self.skip_nl()
        self.next()
        return out


def parse(src):
    return Parser(src).document()
