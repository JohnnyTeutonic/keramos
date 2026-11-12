"""slip: a language for keramos model graphs.

    python -m slip compile MODEL.slip [-o MODEL.ir.json]   compile (and check) to a keramos document
    python -m slip check MODEL.slip                        compile and report keramos's rules, by line
    python -m slip decompile DOC.ir.json|MODEL.gguf [-o MODEL.slip]
    python -m slip digest FILE                              the keramos digest of a .slip, .json or .gguf
"""

import argparse
import json
import sys
from pathlib import Path

from . import keramos
from .compiler import compile_source
from .decompiler import decompile
from .errors import SlipError


def _compile(path):
    src = Path(path).read_text(encoding="utf-8")
    try:
        doc, comp = compile_source(src)
    except SlipError as e:
        print(e.render(src, path), file=sys.stderr)
        raise SystemExit(1)
    return src, doc, comp


def _report(path, src, doc, comp):
    errs = keramos.errors(doc, comp)
    lines = src.splitlines()
    for line, col, msg in errs:
        if line is None:
            print(f"{path}: {msg}", file=sys.stderr)
        else:
            print(f"{path}:{line}:{col}: {msg}\n    {lines[line - 1]}\n    {' ' * (col - 1)}^", file=sys.stderr)
    return len(errs)


def _doc(path):
    if Path(path).suffix == ".slip":
        return _compile(path)[1]
    return keramos.load_document(path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="slip", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile")
    c.add_argument("src")
    c.add_argument("-o", "--out")
    c.add_argument("--no-check", action="store_true", help="write the document even if keramos refuses it")
    k = sub.add_parser("check")
    k.add_argument("src")
    d = sub.add_parser("decompile")
    d.add_argument("doc")
    d.add_argument("-o", "--out")
    g = sub.add_parser("digest")
    g.add_argument("file")
    r = sub.add_parser("run", help="run a model in keramos's reference interpreter on a GGUF's weights")
    r.add_argument("model", help=".slip, .json or .gguf")
    r.add_argument("--weights", required=True, help="the GGUF whose tensors the graph names")
    r.add_argument("--tokens", help="comma-separated token ids (default 1, 2, ... up to 8)")
    r.add_argument("--generate", type=int, default=0, help="then this many greedy tokens")
    r.add_argument("--against", help="another model (.slip, .json or .gguf) to compare logits with, bit for bit")
    a = ap.parse_args(argv)

    if a.cmd in ("compile", "check"):
        src, doc, comp = _compile(a.src)
        n = _report(a.src, src, doc, comp)
        if a.cmd == "check":
            print(f"{a.src}: {'ok' if not n else f'{n} refusal(s)'}; digest {keramos.digest(doc)}")
            return 1 if n else 0
        if n and not a.no_check:
            return 1
        text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
        if a.out:
            Path(a.out).write_text(text, encoding="utf-8", newline="\n")
            print(f"{a.out}: digest {keramos.digest(doc)}", file=sys.stderr)
        else:
            sys.stdout.write(text)
        return 0
    if a.cmd == "decompile":
        text = decompile(keramos.load_document(a.doc))
        if a.out:
            Path(a.out).write_text(text, encoding="utf-8", newline="\n")
        else:
            sys.stdout.write(text)
        return 0
    if a.cmd == "digest":
        print(keramos.digest(_doc(a.file)))
        return 0
    if a.cmd == "run":
        import numpy as np

        from . import run
        doc = _doc(a.model)
        errs = keramos.tools().doc_errors(doc)
        if errs:
            print("\n".join(errs), file=sys.stderr)
            return 1
        W = run.load_weights(a.weights)
        V, ctx = doc["model"]["vocab"], doc["model"]["ctx"]
        ids = [int(t) for t in a.tokens.split(",")] if a.tokens else [j % V for j in range(1, min(8, ctx - 1) + 1)]
        logits = run.forward(doc, W, ids)
        print(f"{a.model}: digest {keramos.digest(doc)}")
        print(f"tokens  {ids}\nlogits  {list(logits.shape)}; argmax per position {logits.argmax(axis=1).tolist()}")
        if a.generate:
            print(f"greedy  {run.greedy(doc, W, ids, a.generate)[len(ids):]}")
        if a.against:
            other = _doc(a.against)
            ref = run.forward(other, W, ids)
            same = logits.shape == ref.shape and np.array_equal(logits.view(np.uint32), ref.view(np.uint32))
            diff = float(np.max(np.abs(logits - ref))) if logits.shape == ref.shape else float("nan")
            print(f"against {a.against}: digest {keramos.digest(other)}")
            print(f"        {'bit-identical logits' if same else f'logits differ (max |diff| {diff:.3g})'}")
            return 0 if same else 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
