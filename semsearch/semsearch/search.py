"""Regularised (aging) evolution, Real et al. 2019, over four representations at a matched budget of
training runs.

  config        paperkiln-style config dicts; one key changed per move; an invalid config is redrawn
                (free); a config already evaluated is not evaluated again (the cache a config search keeps).
  config+canon  the same moves, but the cache is keyed by the canonical class hash of the lowered document,
                and a move whose child has its parent's hash is redrawn before any evaluation.
  semantic      keramos documents; typed graph edits (mutate.EDIT_TYPES: attributes, heads, op swaps, q/k
                norm, sublayer insert and delete from a library, residual rewiring, repeat, tying, widths);
                no-op edits are redrawn (mutate.propose), and the cache is keyed by canonical class hash and,
                behind it, by behavioural fingerprint.
  semantic-cfg  the semantic arm restricted to the edits the config keys can also make (window, theta,
                n_rot, heads, q/k norm, repeat, tying, widths): it separates the representation from the
                larger space the full edit set reaches.

All four start from one initial population (configs, lowered for the semantic arms) and use one selection
rule; only representation, moves and duplicate detection differ. Every training run is one evaluation of
the budget; a duplicate the arm does not detect costs a run and is counted as wasted (its class hash, or
its fingerprint, was evaluated before).

Each run journals its evaluations to an append-only JSONL; a rerun replays them from the journal (the
search is deterministic given its seed), so a killed run resumes where it stopped.
"""

import json
import os
import random
import time
from pathlib import Path

from . import canon, canon_v1, config_space as cs, fingerprint as fp, mutate as mu, task

ARMS = ("config", "config+canon", "semantic", "semantic-cfg")
CFG_EDITS = ("set_attr", "heads", "qk_norm", "repeat", "tie", "width")


class Journal:
    def __init__(self, path):
        self.path = Path(path) if path else None
        self.rows = {}
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.rows[r["key"]] = r["result"]

    def get(self, key):
        return self.rows.get(key)

    def put(self, key, result):
        self.rows[key] = result
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps({"key": key, "result": result}) + "\n")
                f.flush()
                os.fsync(f.fileno())


def initial_population(seed, n, start=None):
    rng = random.Random(1000 + seed)
    pop = []
    while len(pop) < n:
        c = cs.random_config(rng, cs.SEARCH_SPACE)
        if start:
            c.update(start)
            try:
                cs.check(c)
            except cs.InvalidConfig:
                continue
        pop.append(c)
    return pop


def _window_reach(doc):
    """The smallest attention window in the document (0 = unbounded), the rediscovery check's statistic."""
    ws = []
    for n in doc["block"]["nodes"]:
        if n.get("op") == "attention":
            w = n.get("window", 0)
            w = w["per_layer"] if isinstance(w, dict) else [w]
            ws += [x if x < doc["model"]["ctx"] else 0 for x in w]
    if not ws:
        return None
    return 0 if all(x == 0 for x in ws) else min(x for x in ws if x)


def _any_unbounded(doc):
    """Whether some attention node sees the whole context in every application (window 0 after aliases)."""
    for n in doc["block"]["nodes"]:
        if n.get("op") == "attention":
            w = n.get("window", 0)
            w = w["per_layer"] if isinstance(w, dict) else [w]
            if all(x == 0 or x >= doc["model"]["ctx"] for x in w):
                return True
    return False


def run(arm, seed, budget=60, pop_size=12, sample=4, steps=300, journal=None, start=None, log=None,
        strict_check=False):
    """One search. Returns a summary dict with the history of every charged evaluation."""
    rng = random.Random(seed)
    J = Journal(journal)
    held = task.heldout()
    fps = fp.Archive()
    fp_fit = []                  # fitness per fingerprint index
    cache = {}                   # the arm's own duplicate cache: key -> fitness
    seen_class, seen_exact = {}, {}
    history = []
    counters = {"invalid_proposals": 0, "noop_filtered": 0, "inapplicable": 0, "cache_hits": 0,
                "fingerprint_hits": 0, "wasted_class": 0, "wasted_exact": 0, "wasted_fingerprint": 0,
                "fingerprint_match_strict_agree": 0, "fingerprint_match_strict_disagree": 0}
    train_seed = seed
    best_doc = []

    def genome_doc(g):
        if arm.startswith("config"):
            return cs.lower(g)
        return g

    def evaluate(g, desc):
        doc, given = genome_doc(g)
        h_exact = canon.canonical_hash(doc, "exact", given)
        h_class = canon.canonical_hash(doc, "class", given)
        if arm == "config":
            key = json.dumps(cs.key(g))
        else:
            key = h_class
        if key in cache:
            counters["cache_hits"] += 1
            return cache[key], False
        v = fp.fingerprint(doc, given)
        j = fps.find(v)
        if j is not None and strict_check:
            # measured, never acted on: does the match survive perturbed gains and biases (a second weight seed)?
            md, mg = fps.keys[j]
            agree = fp.match(fp.fingerprint(doc, given, seeds=(1,), perturb=0.2),
                             fp.fingerprint(md, mg, seeds=(1,), perturb=0.2))
            counters["fingerprint_match_strict_agree" if agree else "fingerprint_match_strict_disagree"] += 1
        if arm.startswith("semantic") and j is not None:
            counters["fingerprint_hits"] += 1
            cache[key] = fp_fit[j]
            return fp_fit[j], False
        # charged: a training run (journalled by the hash the runs were recorded under: canon_v1)
        jkey = f"{canon_v1.canonical_hash(doc, 'exact', given)}/{train_seed}/{steps}"
        res = J.get(jkey)
        if res is None:
            res = task.evaluate(doc, given, steps=steps, seed=train_seed, held=held)
            J.put(jkey, res)
            counters["trained_now"] = counters.get("trained_now", 0) + 1
        wasted = []
        if h_class in seen_class:
            counters["wasted_class"] += 1
            wasted.append("class")
        if h_exact in seen_exact:
            counters["wasted_exact"] += 1
            wasted.append("exact")
        if j is not None:
            counters["wasted_fingerprint"] += 1
            wasted.append("fingerprint")
        seen_class[h_class] = res["fitness"]
        seen_exact[h_exact] = res["fitness"]
        if j is None:
            fps.add(v, key=(doc, given))
            fp_fit.append(res["fitness"])
        cache[key] = res["fitness"]
        if not best_doc or res["fitness"] > best_doc[0]:
            best_doc[:] = [res["fitness"], doc, given]
        history.append({"eval": len(history) + 1, "fitness": res["fitness"], "acc": res["acc"], "loss": res["loss"],
                        "params": res["params"], "class": h_class[:16], "exact": h_exact[:16], "wasted": wasted,
                        "desc": desc, "window": _window_reach(doc), "any_unbounded": _any_unbounded(doc),
                        "seconds": res["seconds"]})
        if log:
            print(f"[{arm} s{seed}] eval {len(history)} fit {res['fitness']:.3f} acc {res['acc']:.2f} "
                  f"win {_window_reach(doc)} {desc} {'WASTED ' + ','.join(wasted) if wasted else ''}",
                  file=log, flush=True)
        return res["fitness"], True

    def child_of(parent):
        if arm.startswith("config"):
            if arm == "config+canon":
                pd, pg = cs.lower(parent)
                parent_cls = canon.canonical_hash(pd, "class", pg)
            for _ in range(200):
                c, k = cs.mutate(parent, rng, cs.SEARCH_SPACE)
                try:
                    cs.check(c)
                except cs.InvalidConfig:
                    counters["invalid_proposals"] += 1
                    continue
                if arm == "config+canon":
                    d, g = cs.lower(c)
                    if canon.canonical_hash(d, "class", g) == parent_cls:
                        counters["noop_filtered"] += 1
                        continue
                return c, f"{k}={c[k]}"
            return None
        kinds = mu.EDIT_TYPES if arm == "semantic" else CFG_EDITS
        doc, given = parent
        stats = {}
        r = mu.propose(doc, given, rng, level="class", kinds=kinds, stats=stats)
        counters["noop_filtered"] += stats.get("noop_filtered", 0)
        counters["inapplicable"] += stats.get("inapplicable", 0)
        if r is None:
            return None
        out, g, desc, _ = r
        return (out, g), desc

    init = initial_population(seed, pop_size, start)
    population = []   # (genome, fitness)
    for c in init:
        g = c if arm.startswith("config") else cs.lower(c)
        f, charged = evaluate(g, "initial")
        population.append((g, f))
        if len(history) >= budget:
            break
    t0 = time.time()
    proposals = 0
    while len(history) < budget and proposals < budget * 50:
        proposals += 1
        cand = rng.sample(population, min(sample, len(population)))
        parent = max(cand, key=lambda x: x[1])[0]
        r = child_of(parent)
        if r is None:
            continue
        child, desc = r
        f, charged = evaluate(child, desc)
        population.append((child, f))
        population.pop(0)
    best = max(history, key=lambda h: h["fitness"]) if history else None
    curve, b = [], -1e9
    for h in history:
        b = max(b, h["fitness"])
        curve.append(b)
    first_unbounded = next((h["eval"] for h in history if h["window"] == 0), None)
    first_any_unbounded = next((h["eval"] for h in history if h.get("any_unbounded")), None)
    return {"arm": arm, "seed": seed, "budget": budget, "evaluations": len(history), "proposals": proposals,
            "unique_class_graphs": len(seen_class), "unique_exact_graphs": len(seen_exact),
            "best": best, "best_curve": curve, "first_unbounded_window_eval": first_unbounded,
            "first_any_unbounded_attention_eval": first_any_unbounded,
            "best_document": best_doc[1] if best_doc else None, "best_given": best_doc[2] if best_doc else None,
            "counters": counters, "history": history, "seconds": round(time.time() - t0, 1)}
