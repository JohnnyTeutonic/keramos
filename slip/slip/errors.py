"""Errors that point at the source."""


class SlipError(Exception):
    """An error at a place in a slip source: line and column count from 1."""

    def __init__(self, msg, line=None, col=None):
        super().__init__(msg)
        self.msg = msg
        self.line = line
        self.col = col

    def render(self, source=None, path="<slip>"):
        if self.line is None:
            return f"{path}: {self.msg}"
        head = f"{path}:{self.line}:{self.col or 1}: {self.msg}"
        if source is None:
            return head
        lines = source.splitlines()
        if not 1 <= self.line <= len(lines):
            return head
        text = lines[self.line - 1]
        caret = " " * ((self.col or 1) - 1) + "^"
        return f"{head}\n    {text}\n    {caret}"
