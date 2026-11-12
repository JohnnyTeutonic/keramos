"""Phase 2: the four arms of semsearch.search at a matched budget of training runs, several seeds.

    python experiments/phase2_search.py --seeds 0 1 2 3 4 --budget 60 --steps 300 --procs 4 \
        --start window=4 --journals ~/semsearch_scratch/journals

Each (arm, seed) runs in its own process (torch on one thread) and journals every training run, so a killed
job resumes by running the same command. Finished runs are appended to results/phase2_runs.jsonl; the
summary is results/phase2_summary.json. After its search, each run's best document is retrained with three
fresh training seeds (100, 101, 102), since the search's own fitness of its best is the maximum of noisy
draws.
"""

import argparse
import json
import os
import statistics
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from semsearch import bridge, canon, canon_v1, search, task  # noqa: E402

RESULTS = HERE.parent / "results"
RESEEDS = (100, 101, 102)


def _parse_start(items):
    out = {}
    for s in items or []:
        k, _, v = s.partition("=")
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                v = {"true": True, "false": False}.get(v, v)
        out[k] = v
    return out


def one(args):
    arm, seed, a = args
    import torch
    torch.set_num_threads(1)
    jdir = Path(os.path.expanduser(a.journals))
    jdir.mkdir(parents=True, exist_ok=True)
    tag = f"{arm.replace('+', '_')}_s{seed}_b{a.budget}_t{a.steps}"
    logf = open(jdir / f"{tag}.log", "a")
    r = search.run(arm, seed, budget=a.budget, pop_size=a.pop, sample=a.sample, steps=a.steps,
                   journal=jdir / f"{tag}.jsonl", start=_parse_start(a.start), log=logf,
                   strict_check=a.strict_check)
    J = search.Journal(jdir / f"{tag}.reeval.jsonl")
    held = task.heldout()
    re = []
    doc, given = r["best_document"], r["best_given"]
    h = canon_v1.canonical_hash(doc, "exact", given)
    for s in RESEEDS:
        key = f"{h}/{s}/{a.steps}"
        res = J.get(key)
        if res is None:
            res = task.evaluate(doc, given, steps=a.steps, seed=s, held=held)
            J.put(key, res)
        re.append(res)
    r["best_reeval"] = {"seeds": list(RESEEDS), "fitness": [x["fitness"] for x in re], "acc": [x["acc"] for x in re],
                        "mean_fitness": statistics.mean(x["fitness"] for x in re),
                        "mean_acc": statistics.mean(x["acc"] for x in re)}
    r["best_slip"] = bridge.decompile(canon.canonical_document(doc))
    logf.close()
    return r


def summarise(runs):
    by = {}
    for r in runs:
        by.setdefault(r["arm"], []).append(r)
    out = {}

    def ms(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return None
        return {"mean": round(statistics.mean(xs), 4), "sd": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0,
                "values": [round(x, 4) for x in xs]}

    for arm, rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        out[arm] = {
            "seeds": [r["seed"] for r in rs],
            "evaluations": ms([r["evaluations"] for r in rs]),
            "unique_class_graphs": ms([r["unique_class_graphs"] for r in rs]),
            "unique_exact_graphs": ms([r["unique_exact_graphs"] for r in rs]),
            "wasted_class": ms([r["counters"]["wasted_class"] for r in rs]),
            "wasted_exact": ms([r["counters"]["wasted_exact"] for r in rs]),
            "wasted_fingerprint": ms([r["counters"]["wasted_fingerprint"] for r in rs]),
            "invalid_proposals": ms([r["counters"]["invalid_proposals"] for r in rs]),
            "noop_filtered": ms([r["counters"]["noop_filtered"] for r in rs]),
            "cache_hits": ms([r["counters"]["cache_hits"] for r in rs]),
            "fingerprint_hits": ms([r["counters"]["fingerprint_hits"] for r in rs]),
            "fingerprint_match_strict_agree": sum(r["counters"].get("fingerprint_match_strict_agree", 0) for r in rs),
            "fingerprint_match_strict_disagree": sum(r["counters"].get("fingerprint_match_strict_disagree", 0)
                                                     for r in rs),
            "trained_now": sum(r["counters"].get("trained_now", 0) for r in rs),
            "best_search_fitness": ms([r["best"]["fitness"] for r in rs]),
            "best_search_acc": ms([r["best"]["acc"] for r in rs]),
            "best_reeval_fitness": ms([r["best_reeval"]["mean_fitness"] for r in rs]),
            "best_reeval_acc": ms([r["best_reeval"]["mean_acc"] for r in rs]),
            "first_unbounded_window_eval": [r["first_unbounded_window_eval"] for r in rs],
            "best_has_unbounded_window": [r["best"]["window"] == 0 for r in rs],
            "first_any_unbounded_attention_eval": [r.get("first_any_unbounded_attention_eval") for r in rs],
            "best_has_any_unbounded_attention": [r["best"].get("any_unbounded") for r in rs],
            "seconds_per_eval": ms([sum(h["seconds"] for h in r["history"]) / max(1, len(r["history"]))
                                    for r in rs]),
            "best_params": ms([r["best"]["params"] for r in rs]),
            "best_curve_mean": [round(statistics.mean(c), 4) for c in zip(*[r["best_curve"] for r in rs])],
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="*", default=list(search.ARMS))
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--budget", type=int, default=60)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--pop", type=int, default=12)
    p.add_argument("--sample", type=int, default=4)
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--start", nargs="*", default=["window=4"])
    p.add_argument("--journals", default="~/semsearch_scratch/journals")
    p.add_argument("--tag", default="phase2")
    p.add_argument("--strict-check", action="store_true",
                   help="check every fingerprint match with perturbed weights (recorded, never acted on)")
    a = p.parse_args()
    runs_path = RESULTS / f"{a.tag}_runs.jsonl"
    done = {}
    if runs_path.exists():
        for line in runs_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["arm"], r["seed"])] = r
    todo = [(arm, s, a) for s in a.seeds for arm in a.arms if (arm, s) not in done]
    print(f"{len(done)} runs done, {len(todo)} to go", flush=True)
    t0 = time.time()
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(one, todo):
            slim = {k: v for k, v in r.items() if k not in ("best_document", "best_given")}
            slim["best_given"] = r["best_given"]
            with open(runs_path, "a") as f:
                f.write(json.dumps(slim) + "\n")
            done[(r["arm"], r["seed"])] = slim
            b = r["best"]
            print(f"{r['arm']} s{r['seed']}: best {b['fitness']:.3f} (acc {b['acc']:.2f}), reeval "
                  f"{r['best_reeval']['mean_fitness']:.3f}, unique {r['unique_class_graphs']}, wasted "
                  f"{r['counters']['wasted_class']} [{time.time() - t0:.0f}s]", flush=True)
    summary = {"config": {k: v for k, v in vars(a).items()}, "reseeds": list(RESEEDS),
               "arms": summarise(list(done.values()))}
    (RESULTS / f"{a.tag}_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["arms"], indent=1)[:6000])


if __name__ == "__main__":
    main()
