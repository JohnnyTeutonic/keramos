"""Further reductions of the Phase 2 runs: python experiments/phase2_analysis.py results/phase2r_runs.jsonl

- wasted runs as a union (a run counted once whatever detector flags it);
- paired per-seed comparisons of the retrained best (the arms share each seed's initial population);
- the edits that produced each new best-so-far in each arm;
- whether each run's best has an attention that sees the whole context.
Writes results/phase2_analysis.json.
"""

import json
import sys
from collections import Counter
from math import comb
from pathlib import Path


def sign_p(wins, n):
    """Two-sided exact sign-test p-value for `wins` of `n` non-tied pairs."""
    k = max(wins, n - wins)
    return min(1.0, 2 * sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n)


def main():
    path = Path(sys.argv[1])
    runs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    by = {}
    for r in runs:
        by.setdefault(r["arm"], {})[r["seed"]] = r
    out = {"source": path.name, "arms": {}, "paired": {}}
    for arm, rs in sorted(by.items()):
        wasted_union = [sum(1 for h in r["history"] if h["wasted"]) for _, r in sorted(rs.items())]
        improvers = Counter()
        for r in rs.values():
            b = -1e9
            for h in r["history"]:
                if h["fitness"] > b + 1e-9:
                    if h["desc"] != "initial":
                        kind = h["desc"].split("=")[0].split(" ")[0]
                        if kind == "set_attr":
                            kind = h["desc"].split(" ")[1]
                        if kind == "insert":
                            kind = "insert " + h["desc"].split(" ")[1]
                        improvers[kind] += 1
                    b = h["fitness"]
        out["arms"][arm] = {
            "wasted_union_per_seed": wasted_union,
            "wasted_union_share": round(sum(wasted_union) / sum(r["evaluations"] for r in rs.values()), 3),
            "best_has_any_unbounded_attention": [r["best"].get("any_unbounded") for _, r in sorted(rs.items())],
            "best_min_window": [r["best"]["window"] for _, r in sorted(rs.items())],
            "edits_that_set_a_new_best": dict(improvers.most_common()),
            "reeval_acc_per_seed": [round(r["best_reeval"]["mean_acc"], 3) for _, r in sorted(rs.items())],
            "reeval_fitness_per_seed": [round(r["best_reeval"]["mean_fitness"], 3) for _, r in sorted(rs.items())],
        }
    arms = sorted(by)
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            seeds = sorted(set(by[a]) & set(by[b]))
            wins = sum(by[a][s]["best_reeval"]["mean_fitness"] > by[b][s]["best_reeval"]["mean_fitness"] for s in seeds)
            ties = sum(by[a][s]["best_reeval"]["mean_fitness"] == by[b][s]["best_reeval"]["mean_fitness"]
                       for s in seeds)
            n = len(seeds) - ties
            out["paired"][f"{a} vs {b}"] = {"seeds": len(seeds), f"{a} better": wins, f"{b} better": n - wins,
                                            "sign_test_p_two_sided": round(sign_p(wins, n), 4) if n else None}
    (path.parent / "phase2_analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
