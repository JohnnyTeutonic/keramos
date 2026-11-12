"""slip: a language for keramos model graphs. Source compiles to a keramos IR document."""

from .compiler import compile_source
from .decompiler import decompile
from .errors import SlipError
from .parser import parse

__all__ = ["compile_source", "decompile", "parse", "SlipError"]
