"""Markdown tables of a Phase 2 summary: python experiments/phase2_table.py results/phase2r_summary.json"""

import json
import sys


def f(x, nd=2):
    if x is None:
        return "-"
    return f"{x['mean']:.{nd}f} ± {x['sd']:.{nd}f}"


def main():
    s = json.load(open(sys.argv[1]))
    arms = s["arms"]
    order = [a for a in ("config", "config+canon", "semantic-cfg", "semantic") if a in arms]
    print("| | " + " | ".join(order) + " |")
    print("|---|" + "---|" * len(order))
    rows = [
        ("training runs (budget)", lambda a: f(a["evaluations"], 0)),
        ("unique class graphs trained", lambda a: f(a["unique_class_graphs"], 1)),
        ("wasted: class duplicate", lambda a: f(a["wasted_class"], 1)),
        ("wasted: exact duplicate", lambda a: f(a["wasted_exact"], 1)),
        ("wasted: fingerprint duplicate", lambda a: f(a["wasted_fingerprint"], 1)),
        ("invalid proposals redrawn", lambda a: f(a["invalid_proposals"], 1)),
        ("no-op proposals redrawn", lambda a: f(a["noop_filtered"], 1)),
        ("duplicates served from cache", lambda a: f(a["cache_hits"], 1)),
        ("fingerprint matches served", lambda a: f(a["fingerprint_hits"], 1)),
        ("best fitness in search (-loss)", lambda a: f(a["best_search_fitness"], 3)),
        ("best, retrained on 3 seeds (-loss)", lambda a: f(a["best_reeval_fitness"], 3)),
        ("best, retrained: recall accuracy", lambda a: f(a["best_reeval_acc"], 2)),
        ("best's parameters", lambda a: f(a.get("best_params"), 0)),
        ("seconds per training run", lambda a: f(a.get("seconds_per_eval"), 1)),
    ]
    for name, fn in rows:
        print(f"| {name} | " + " | ".join(fn(arms[a]) for a in order) + " |")
    print()
    print("| per seed | " + " | ".join(order) + " |")
    print("|---|" + "---|" * len(order))
    seeds = arms[order[0]]["seeds"]
    for k, sd in enumerate(seeds):
        cells = []
        for a in order:
            A = arms[a]
            re = A["best_reeval_fitness"]["values"][k]
            ra = A["best_reeval_acc"]["values"][k]
            fu = A["first_unbounded_window_eval"][k]
            fa = A.get("first_any_unbounded_attention_eval", [None] * len(seeds))[k]
            cells.append(f"{re:.3f} / {ra:.2f} / {fu if fu else '-'} / {fa if fa else '-'}")
        print(f"| seed {sd} | " + " | ".join(cells) + " |")
    print()
    for a in order:
        A = arms[a]
        print(f"{a}: strict fingerprint check agree {A.get('fingerprint_match_strict_agree')} disagree "
              f"{A.get('fingerprint_match_strict_disagree')}; trained during this pass {A.get('trained_now')}")


if __name__ == "__main__":
    main()
