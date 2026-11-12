"""slip: a language for keramos model graphs.

    python -m slip check MODEL [--weights W.gguf]     keramos's rules and the widths, by source line
    python -m slip compile MODEL.slip [-o MODEL.ir.json]
    python -m slip decompile DOC [-o MODEL.slip]       a .json, a .gguf, or a run folder holding one
    python -m slip diff A B [--tensor-names]          the architectural difference, as a diff of slip
    python -m slip digest FILE                         the keramos digest of a .slip, .json or .gguf
    python -m slip run MODEL --weights W.gguf [--tokens 1,2,3] [--generate N] [--against OTHER]

MODEL, A and B are each a .slip, a keramos .json, a GGUF carrying one, or a folder holding one (a
paperkiln run's export folder: its .ir.json, else its GGUF).
"""

import argparse
import json
import sys
from pathlib import Path

from . import keramos, widths
from .compiler import compile_source
from .decompiler import decompile
from .diff import diff
from .errors import SlipError


def resolve(path):
    """A file, or the one keramos document a folder holds (its .ir.json files, else its GGUFs)."""
    p = Path(path)
    if not p.is_dir():
        return p
    for pattern in ("*.ir.json", "*.gguf"):
        found = sorted(p.glob(pattern))
        if pattern == "*.gguf":
            found = [f for f in found if "paperkiln.ir" in keramos.tools().read_gguf(f)[0]]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise SystemExit(f"{p}: holds {len(found)} documents ({', '.join(f.name for f in found)}); "
                             "name the one you mean")
    raise SystemExit(f"{p}: holds no keramos document (no .ir.json, no GGUF with paperkiln.ir)")


def _compile(path):
    src = Path(path).read_text(encoding="utf-8")
    try:
        doc, comp = compile_source(src)
    except SlipError as e:
        print(e.render(src, path), file=sys.stderr)
        raise SystemExit(1)
    return src, doc, comp


def load(path):
    """(document, source text or None, compiler or None, file)."""
    p = resolve(path)
    if p.suffix == ".slip":
        src, doc, comp = _compile(p)
        return doc, src, comp, p
    return keramos.load_document(p), None, None, p


def problems(doc, comp=None, dims=None):
    """keramos's refusals, then the width mismatches, each (line, col, message)."""
    out = keramos.errors(doc, comp)
    if not out:  # widths are solved only on a document keramos accepts
        errs, _ = widths.check(doc, dims)
        for sect, idx, msg in errs:
            line, col = (comp.where.get((sect, idx), (None, None)) if comp is not None and idx is not None
                         else (None, None))
            where = f"{sect}[{idx}]: " if line is None and idx is not None else ""
            out.append((line, col, f"width: {where}{msg}"))
    return out


def report(path, src, errs):
    lines = src.splitlines() if src else []
    for line, col, msg in errs:
        if line is None:
            print(f"{path}: {msg}", file=sys.stderr)
        else:
            print(f"{path}:{line}:{col}: {msg}\n    {lines[line - 1]}\n    {' ' * (col - 1)}^", file=sys.stderr)
    return len(errs)


def weight_dims(weights, model_file):
    """Tensor shapes to check against: --weights, else the model's own GGUF."""
    if weights:
        return widths.gguf_dims(resolve(weights), keramos.tools())
    if model_file.suffix == ".gguf":
        return widths.gguf_dims(model_file, keramos.tools())
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="slip", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile", help="compile a .slip to a keramos document, refusing what keramos refuses")
    c.add_argument("src")
    c.add_argument("-o", "--out")
    c.add_argument("--weights", help="also check the widths against this GGUF's tensors")
    c.add_argument("--no-check", action="store_true", help="write the document even if it has problems")
    k = sub.add_parser("check", help="keramos's rules and the widths, reported at the source line")
    k.add_argument("src")
    k.add_argument("--weights", help="check the tensor shapes against this GGUF (a GGUF model is checked "
                                     "against its own tensors)")
    k.add_argument("--shapes", action="store_true", help="print every tensor's solved shape")
    d = sub.add_parser("decompile", help="any keramos document (or a run folder holding one) to slip")
    d.add_argument("doc")
    d.add_argument("-o", "--out")
    f = sub.add_parser("diff", help="the architectural difference between two models, tensor names aside")
    f.add_argument("a")
    f.add_argument("b")
    f.add_argument("--tensor-names", action="store_true", help="compare tensor names too")
    f.add_argument("-U", "--context", type=int, default=3)
    g = sub.add_parser("digest")
    g.add_argument("file")
    r = sub.add_parser("run", help="run a model in keramos's reference interpreter on a GGUF's weights")
    r.add_argument("model")
    r.add_argument("--weights", required=True, help="the GGUF whose tensors the graph names")
    r.add_argument("--tokens", help="comma-separated token ids (default 1, 2, ... up to 8)")
    r.add_argument("--generate", type=int, default=0, help="then this many greedy tokens")
    r.add_argument("--against", help="another model to compare logits with, bit for bit")
    a = ap.parse_args(argv)

    if a.cmd == "check":
        doc, src, comp, p = load(a.src)
        errs = problems(doc, comp, weight_dims(a.weights, p))
        n = report(p, src, errs)
        if a.shapes and not n:
            _, w = widths.check(doc, weight_dims(a.weights, p))
            for t in sorted(w.tensors):
                print(f"  {t} {widths.fmt(w.shape(t))}")
        print(f"{p}: {'ok' if not n else f'{n} problem(s)'}; digest {keramos.digest(doc)}")
        return 1 if n else 0
    if a.cmd == "compile":
        src, doc, comp = _compile(a.src)
        dims = widths.gguf_dims(resolve(a.weights), keramos.tools()) if a.weights else None
        if report(a.src, src, problems(doc, comp, dims)) and not a.no_check:
            return 1
        text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
        if a.out:
            Path(a.out).write_text(text, encoding="utf-8", newline="\n")
            print(f"{a.out}: digest {keramos.digest(doc)}", file=sys.stderr)
        else:
            sys.stdout.write(text)
        return 0
    if a.cmd == "decompile":
        p = resolve(a.doc)
        doc = keramos.load_document(p)
        text = f"# decompiled by slip from {p.name}; keramos digest {keramos.digest(doc)}\n\n" + decompile(doc)
        if a.out:
            Path(a.out).write_text(text, encoding="utf-8", newline="\n")
            print(f"{a.out}: from {p}", file=sys.stderr)
        else:
            sys.stdout.write(text)
        return 0
    if a.cmd == "diff":
        da, _, _, pa = load(a.a)
        db, _, _, pb = load(a.b)
        same, text = diff(da, db, f"{pa} ({keramos.digest(da)})", f"{pb} ({keramos.digest(db)})",
                          a.tensor_names, a.context)
        sys.stdout.write(text)
        aside = "" if a.tensor_names else ", tensor names aside"
        if keramos.digest(da) == keramos.digest(db):
            print("the same document (one digest)")
        elif same and not text:
            print(f"the same graph, in the same spelling (informative keys, defaults and prose{aside})")
        elif same:
            print(f"the same graph by keramos's rule 38 (value names, add and mul operand order{aside}): "
                  "the diff above is spelling only")
        else:
            print(f"different graphs{aside}")
        return 0 if same else 1
    if a.cmd == "digest":
        print(keramos.digest(load(a.file)[0]))
        return 0
    if a.cmd == "run":
        import numpy as np

        from . import run
        doc = load(a.model)[0]
        errs = keramos.tools().doc_errors(doc)
        if errs:
            print("\n".join(errs), file=sys.stderr)
            return 1
        W = run.load_weights(resolve(a.weights))
        V, ctx = doc["model"]["vocab"], doc["model"]["ctx"]
        ids = [int(t) for t in a.tokens.split(",")] if a.tokens else [j % V for j in range(1, min(8, ctx - 1) + 1)]
        logits = run.forward(doc, W, ids)
        print(f"{a.model}: digest {keramos.digest(doc)}")
        print(f"tokens  {ids}\nlogits  {list(logits.shape)}; argmax per position {logits.argmax(axis=1).tolist()}")
        if a.generate:
            print(f"greedy  {run.greedy(doc, W, ids, a.generate)[len(ids):]}")
        if a.against:
            other = load(a.against)[0]
            ref = run.forward(other, W, ids)
            same = logits.shape == ref.shape and np.array_equal(logits.view(np.uint32), ref.view(np.uint32))
            dif = float(np.max(np.abs(logits - ref))) if logits.shape == ref.shape else float("nan")
            print(f"against {a.against}: digest {keramos.digest(other)}")
            print(f"        {'bit-identical logits' if same else f'logits differ (max |diff| {dif:.3g})'}")
            return 0 if same else 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
