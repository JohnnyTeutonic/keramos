"""python -m semsearch COMMAND ...

    hash MODEL [--level exact|class] [--shape NAME=AxB ...]   the canonical hash
    canon MODEL [-o OUT.slip] [--level exact|class]          the canonical document, as slip
    diff A B [--level ...] [--text]                          typed edits and distance (and slip's diff)
    same A B                                                 rule 38, exact, class and fingerprint verdicts
    mutate MODEL [--kind KIND] [--seed S] [-o OUT.slip]      one typed edit that changes the graph

A MODEL is anything slip reads: a .slip file, a keramos .json, a GGUF carrying one, or a run folder.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from . import bridge, canon, fingerprint as fp, mutate as mu, semdiff


def _load(path):
    sys.path.insert(0, str(bridge.SLIP))
    from slip.__main__ import load
    return load(path)[0]


def _given(doc, shapes):
    """--shape as slip's run takes it (NAME=AxB or NAME=H, NAME a tensor name or slip's @path), as full
    shapes of the tensors the graph leaves free."""
    if not shapes:
        return {}
    free = {t for t, s in canon.solved_shapes(doc).items() if None in s}
    w = bridge.slip_run.solve_shapes(doc, shapes)
    return {t: [int(x) for x in w.shape(t)] for t in free if None not in w.shape(t)}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m semsearch")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("hash", "canon", "mutate"):
        s = sub.add_parser(name)
        s.add_argument("model")
        s.add_argument("--level", default="class", choices=canon.LEVELS)
        s.add_argument("--shape", nargs="*")
        if name in ("canon", "mutate"):
            s.add_argument("-o")
        if name == "mutate":
            s.add_argument("--kind", choices=mu.EDIT_TYPES)
            s.add_argument("--seed", type=int, default=0)
    for name in ("diff", "same"):
        s = sub.add_parser(name)
        s.add_argument("a")
        s.add_argument("b")
        s.add_argument("--level", default="class", choices=canon.LEVELS)
        s.add_argument("--text", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "hash":
        doc = _load(a.model)
        print(canon.canonical_hash(doc, a.level, _given(doc, a.shape)))
    elif a.cmd == "canon":
        src = bridge.decompile(canon.canonical_document(_load(a.model), a.level))
        Path(a.o).write_text(src) if a.o else print(src, end="")
    elif a.cmd == "mutate":
        doc = _load(a.model)
        given = _given(doc, a.shape)
        rng = random.Random(a.seed)
        r = mu.propose(doc, given, rng, level=a.level, kinds=(a.kind,) if a.kind else mu.EDIT_TYPES,
                       trainable=False)
        if r is None:
            raise SystemExit("no edit of that kind changes this graph")
        out, g, desc, h = r
        src = f"# semsearch mutate: {desc}; class hash {h}\n" + bridge.decompile(canon.canonical_document(out, a.level))
        Path(a.o).write_text(src) if a.o else print(src, end="")
        if g:
            print("given shapes: " + " ".join(fp.shape_args(g)), file=sys.stderr)
    elif a.cmd == "diff":
        A, B = _load(a.a), _load(a.b)
        d, e = semdiff.edits(A, B, level=a.level)
        for x in e:
            print(json.dumps(x))
        print(f"distance {d}")
        if a.text:
            print(semdiff.text(A, B, a.level, a.a, a.b), end="")
    elif a.cmd == "same":
        A, B = _load(a.a), _load(a.b)
        print("rule 38:", canon.rule38_key(A) == canon.rule38_key(B))
        for lv in canon.LEVELS:
            print(f"{lv}:", canon.canonical_hash(A, lv) == canon.canonical_hash(B, lv))


if __name__ == "__main__":
    main()
