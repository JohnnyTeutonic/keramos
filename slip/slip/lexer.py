"""The slip lexer: source text to tokens.

Tokens are NEWLINE, IDENT, NUMBER, STRING, TENSOR (an @path), the punctuation below and EOF. Numbers
and strings follow JSON's grammar, so every JSON scalar a keramos document holds has a spelling.
A '#' starts a comment that runs to the end of the line.
"""

import json
import re
from dataclasses import dataclass

from .errors import SlipError

PUNCT = ["+=", "-=", "*=", "(", ")", "[", "]", "{", "}", ",", "=", ":", "+", "-", "*"]

_NUMBER = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TENSOR = re.compile(r"@(?:[A-Za-z0-9_]|\{i\})(?:[A-Za-z0-9_.-]|\{i\})*")  # braces only as {i}
_STRING = re.compile(r'"(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"')


@dataclass
class Token:
    kind: str
    value: object
    line: int
    col: int
    text: str = ""

    def __repr__(self):
        return f"{self.kind}({self.text!r})@{self.line}:{self.col}"


def tokenize(src):
    toks = []
    i, line, col = 0, 1, 1
    n = len(src)
    while i < n:
        c = src[i]
        if c == "\n":
            toks.append(Token("NEWLINE", None, line, col, "\\n"))
            i += 1
            line += 1
            col = 1
            continue
        if c in " \t\r":
            i += 1
            col += 1
            continue
        if c == "#":
            while i < n and src[i] != "\n":
                i += 1
            continue
        m = _STRING.match(src, i) if c == '"' else None
        if c == '"':
            if not m:
                raise SlipError("unterminated or malformed string (slip strings are JSON strings)", line, col)
            text = m.group(0)
            toks.append(Token("STRING", json.loads(text), line, col, text))
        elif c.isdigit():
            m = _NUMBER.match(src, i)
            text = m.group(0)
            if i + len(text) < n and (src[i + len(text)].isalnum() or src[i + len(text)] == "_"):
                raise SlipError(f"malformed number {src[i:i + len(text) + 1]!r}", line, col)
            is_int = not any(ch in text for ch in ".eE")
            toks.append(Token("NUMBER", int(text) if is_int else float(text), line, col, text))
        elif c == "@":
            m = _TENSOR.match(src, i)
            if not m:
                raise SlipError("'@' must be followed by a tensor path, as in @attn_norm", line, col)
            text = m.group(0)
            toks.append(Token("TENSOR", text[1:], line, col, text))
        elif c.isalpha() or c == "_":
            m = _IDENT.match(src, i)
            text = m.group(0)
            toks.append(Token("IDENT", text, line, col, text))
        else:
            for p in PUNCT:
                if src.startswith(p, i):
                    text = p
                    toks.append(Token(p, p, line, col, p))
                    break
            else:
                raise SlipError(f"unexpected character {c!r}", line, col)
        i += len(text)
        col += len(text)
    toks.append(Token("NEWLINE", None, line, col, "\\n"))
    toks.append(Token("EOF", None, line, col, ""))
    return toks
