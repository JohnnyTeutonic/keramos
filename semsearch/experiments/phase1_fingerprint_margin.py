"""How far apart fingerprints are for one-key config changes, against the match tolerance (1e-5 relative).

    python experiments/phase1_fingerprint_margin.py --n 60

For random configs of config_space.SEARCH_SPACE and each key, the key is changed to another value; the
relative max difference of the two fingerprints is recorded, split by whether the class hashes agree.
Writes results/phase1_fingerprint_margin.json.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from semsearch import canon, config_space as cs, fingerprint as fp  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = random.Random(a.seed)
    rows = []
    for _ in range(a.n):
        c = cs.random_config(rng, cs.SEARCH_SPACE)
        va = None
        for k in cs.SEARCH_SPACE:
            c2 = dict(c)
            c2[k] = rng.choice([v for v in cs.SEARCH_SPACE[k] if v != c[k]])
            try:
                cs.check(c2)
            except cs.InvalidConfig:
                continue
            A, ga = cs.lower(c)
            B, gb = cs.lower(c2)
            same = canon.canonical_hash(A, "class", ga) == canon.canonical_hash(B, "class", gb)
            if va is None:
                va = fp.fingerprint(A, ga)
            vb = fp.fingerprint(B, gb)
            rel = float(np.max(np.abs(va - vb)) / max(1, np.max(np.abs(va))))
            rows.append((k, same, rel))
    out = {"n_configs": a.n, "seed": a.seed, "tolerance": 1e-5, "by_key": {}}
    for k in cs.SEARCH_SPACE:
        rs = [r for r in rows if r[0] == k]
        diff = sorted(r[2] for r in rs if not r[1])
        same = sorted(r[2] for r in rs if r[1])
        out["by_key"][k] = {
            "different_class": len(diff), "min_rel_diff_when_different": diff[0] if diff else None,
            "below_tolerance_when_different": sum(1 for x in diff if x <= 1e-5),
            "same_class": len(same), "max_rel_diff_when_same": same[-1] if same else None}
    (HERE.parent / "results" / "phase1_fingerprint_margin.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
