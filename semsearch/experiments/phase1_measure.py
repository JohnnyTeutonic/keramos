"""Phase 1: how much of a paperkiln-style config space is semantically distinct, and how many config-space
moves are no-ops.

    python experiments/phase1_measure.py grid  --procs 6 --ckpt ~/semsearch_scratch/grid.jsonl
    python experiments/phase1_measure.py sample --n 3000
    python experiments/phase1_measure.py walks --walks 200 --steps 100

`grid` enumerates config_space.SPACE completely and records, for every config, its identity at six levels:
the config dict, keramos's digest, keramos's rule 38, semsearch's exact canonical hash, its class hash.
It checkpoints finished chunks to an append-only JSONL and resumes from it. `sample` adds behavioural
fingerprints on a random sample. `walks` runs random config-space walks (one key changed per move) and
counts invalid, no-op and revisiting moves. Each writes results/phase1_<name>.json.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from semsearch import bridge, canon, config_space as cs, fingerprint as fp  # noqa: E402

RESULTS = HERE.parent / "results"


def identities(cfg):
    try:
        cs.check(cfg)
    except cs.InvalidConfig:
        return None
    doc, given = cs.lower(cfg)
    return {
        "digest": bridge.digest(doc)[:24],
        "rule38": canon.rule38_key(doc)[:24],
        # rule 38 does not see the FFN width (a weight shape); add it so that level compares like with like
        "rule38_shapes": canon._h(canon.rule38_key(doc), cfg["d_ff"])[:24],
        "exact": canon.canonical_hash(doc, "exact", given)[:24],
        "class": canon.canonical_hash(doc, "class", given)[:24],
    }


def _chunk(args):
    k, cfgs = args
    out = []
    for c in cfgs:
        out.append([cs.key(c), identities(c)])
    return k, out


def cmd_grid(a):
    cfgs = list(cs.grid(cs.SPACE))
    chunk = 2000
    chunks = [(k, cfgs[k * chunk:(k + 1) * chunk]) for k in range((len(cfgs) + chunk - 1) // chunk)]
    ckpt = Path(os.path.expanduser(a.ckpt))
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["chunk"]] = r["rows"]
    todo = [c for c in chunks if c[0] not in done]
    print(f"{len(cfgs)} configs in {len(chunks)} chunks; {len(done)} chunks already done", flush=True)
    t0 = time.time()
    with Pool(a.procs) as pool, open(ckpt, "a") as f:
        for k, rows in pool.imap_unordered(_chunk, todo):
            f.write(json.dumps({"chunk": k, "rows": rows}) + "\n")
            f.flush()
            os.fsync(f.fileno())
            done[k] = rows
            print(f"chunk {k} done ({len(done)}/{len(chunks)}, {time.time() - t0:.0f}s)", flush=True)
    rows = [r for k in sorted(done) for r in done[k]]
    valid = [r for r in rows if r[1] is not None]
    levels = ["digest", "rule38", "rule38_shapes", "exact", "class"]
    summary = {
        "space": {k: v for k, v in cs.SPACE.items()},
        "configs": len(rows),
        "valid_configs": len(valid),
        "invalid_configs": len(rows) - len(valid),
        "distinct": {"config": len(valid)},
    }
    for lv in levels:
        summary["distinct"][lv] = len({r[1][lv] for r in valid})
    # how the duplicates spread: the largest classes at the class level
    sizes = Counter(r[1]["class"] for r in valid)
    summary["class_size_histogram"] = dict(sorted(Counter(sizes.values()).items()))
    summary["note"] = ("distinct counts over every valid config of the grid; 'config' is the number of distinct "
                       "valid config dicts. rule38 ignores the FFN width (a weight shape, not a graph attribute); "
                       "rule38_shapes adds it.")
    out = RESULTS / "phase1_grid.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    ckpt.unlink()   # the summary replaces the checkpoint


def cmd_sample(a):
    rng = random.Random(a.seed)
    cfgs = []
    seen = set()
    while len(cfgs) < a.n:
        c = cs.random_config(rng, cs.SPACE)
        if cs.key(c) not in seen:
            seen.add(cs.key(c))
            cfgs.append(c)
    arch = fp.Archive()
    rows = []
    t0 = time.time()
    for i, c in enumerate(cfgs):
        doc, given = cs.lower(c)
        ids = identities(c)
        v = fp.fingerprint(doc, given)
        j = arch.find(v)
        if j is None:
            j = arch.add(v)
        ids["fingerprint"] = j
        rows.append(ids)
        if i % 500 == 0:
            print(f"{i}/{len(cfgs)} {time.time() - t0:.0f}s", flush=True)
    levels = ["digest", "rule38_shapes", "exact", "class", "fingerprint"]
    distinct = {"config": len(rows)}
    for lv in levels:
        distinct[lv] = len({r[lv] for r in rows})
    # soundness: configs with one exact hash must share a fingerprint
    by_exact = defaultdict(set)
    for r in rows:
        by_exact[r["exact"]].add(r["fingerprint"])
    unsound = sum(1 for s in by_exact.values() if len(s) > 1)
    # completeness: fingerprint classes that hold several exact hashes (equivalences the rewrites miss)
    by_fp = defaultdict(set)
    for r in rows:
        by_fp[r["fingerprint"]].add(r["exact"])
    merged = sum(len(s) - 1 for s in by_fp.values() if len(s) > 1)
    by_fp_class = defaultdict(set)
    for r in rows:
        by_fp_class[r["fingerprint"]].add(r["class"])
    summary = {"n": len(rows), "seed": a.seed, "distinct": distinct,
               "exact_classes_with_two_fingerprints": unsound,
               "exact_hashes_merged_by_fingerprint": merged,
               "fingerprint_classes_holding_several_class_hashes": sum(1 for s in by_fp_class.values() if len(s) > 1),
               "note": "fingerprint: logits of ir_numpy on 2 fixed probes of 32 tokens, weights by tensor name "
                       "(seed 0), match within 1e-5 of max(1, max |logit|)"}
    (RESULTS / "phase1_sample.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def cmd_walks(a):
    rng = random.Random(a.seed)
    tally = Counter()
    by_key = defaultdict(Counter)
    t0 = time.time()
    for w in range(a.walks):
        cur = cs.random_config(rng, cs.SPACE)
        cur_ids = identities(cur)
        cur_fp = fp.fingerprint(*cs.lower(cur)) if a.fingerprint else None
        seen_cfg, seen_exact, seen_class = {cs.key(cur)}, {cur_ids["exact"]}, {cur_ids["class"]}
        for s in range(a.steps):
            child, k = cs.mutate(cur, rng, cs.SPACE)
            tally["moves"] += 1
            by_key[k]["moves"] += 1
            ids = identities(child)
            if ids is None:
                tally["invalid"] += 1
                by_key[k]["invalid"] += 1
                continue
            new_cfg = cs.key(child) not in seen_cfg
            if ids["exact"] == cur_ids["exact"]:
                tally["noop_exact"] += 1
                by_key[k]["noop_exact"] += 1
            if ids["class"] == cur_ids["class"]:
                tally["noop_class"] += 1
                by_key[k]["noop_class"] += 1
            if a.fingerprint:
                cfp = fp.fingerprint(*cs.lower(child))
                if fp.match(cfp, cur_fp):
                    tally["noop_fingerprint"] += 1
                    by_key[k]["noop_fingerprint"] += 1
                cur_fp = cfp
            if new_cfg and ids["class"] in seen_class:
                tally["new_config_seen_graph"] += 1
            if new_cfg:
                tally["new_config"] += 1
            seen_cfg.add(cs.key(child))
            seen_exact.add(ids["exact"])
            seen_class.add(ids["class"])
            cur, cur_ids = child, ids
        tally["distinct_configs"] += len(seen_cfg)
        tally["distinct_class_graphs"] += len(seen_class)
        if w % 20 == 0:
            print(f"walk {w} {time.time() - t0:.0f}s", flush=True)
    summary = {"walks": a.walks, "steps": a.steps, "seed": a.seed, "space": "config_space.SPACE",
               "totals": dict(tally), "by_key": {k: dict(v) for k, v in sorted(by_key.items())},
               "note": "a move changes one key to another value of its list; an invalid move is rejected and the "
                       "walk stays; a no-op is a valid move whose child has the parent's canonical hash"}
    (RESULTS / "phase1_walks.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--procs", type=int, default=4)
    g.add_argument("--ckpt", default="~/semsearch_scratch/grid.jsonl")
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=3000)
    s.add_argument("--seed", type=int, default=0)
    w = sub.add_parser("walks")
    w.add_argument("--walks", type=int, default=200)
    w.add_argument("--steps", type=int, default=100)
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--fingerprint", action="store_true")
    a = p.parse_args()
    {"grid": cmd_grid, "sample": cmd_sample, "walks": cmd_walks}[a.cmd](a)


if __name__ == "__main__":
    main()
