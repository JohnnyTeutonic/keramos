"""Create an immutable source-bound inventory and verdict comparison."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.root / "keramos/conformance/manifest.json").read_text())
    affected = {}
    for item in manifest["cases"]:
        if item["kind"] != "trace":
            continue
        path = args.root / "keramos/conformance" / item["file"]
        case = json.loads(path.read_text())
        if "score_form" in case and "ir" not in case:
            affected[item["id"]] = {"file": item["file"], "sha256": sha(path),
                                    "expected_shape": [len(case["score_form"]["logprobs"])]}
    inventory = []
    for relative in ("keramos/conformance/receipts", "keramos/adapters/receipts",
                     "AI_ML/keramos_paper/analysis"):
        for path in sorted((args.root / relative).rglob("*.json")):
            if "trace_score_impact_20261031" in str(path):
                continue
            doc = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(doc, dict) or not isinstance(doc.get("results"), list):
                continue
            rows = [r for r in doc["results"] if r["id"] in affected]
            fallback = [r for r in rows if "adapter's score" in r.get("detail", "")]
            inventory.append({"path": str(path.relative_to(args.root)), "sha256": sha(path),
                              "affected_path_rows": fallback, "same_case_other_path_rows": rows
                              if not fallback else []})
    baseline_path = args.root / "keramos/conformance/receipts/mtstudio.json"
    baseline = json.loads(baseline_path.read_text())
    old = {r["id"]: r for r in baseline["results"]}
    comparisons = {}
    cached_calls = {}
    for label in ("historical_runner", "parent_runner", "fixed_runner"):
        path = args.runs / (label + ".json")
        doc = json.loads(path.read_text())
        raw = json.loads(path.with_suffix(".raw.json").read_text())
        cached_calls[label] = [c for c in raw["score_calls"] if c["kwargs"].get("kv_cache")]
        rows = {r["id"]: r for r in doc["results"]}
        comparisons[label] = {
            "receipt_sha256": sha(path), "raw_sha256": sha(path.with_suffix(".raw.json")),
            "counts": doc["counts"], "case_count": len(rows),
            "missing_historical_ids": sorted(set(old) - set(rows)),
            "new_ids": sorted(set(rows) - set(old)),
            "historical_verdict_movements": [{"id": k, "before": old[k]["verdict"],
                "after": rows[k]["verdict"], "detail": rows[k]["detail"]}
                for k in sorted(set(old) & set(rows)) if old[k]["verdict"] != rows[k]["verdict"]],
            "historical_detail_changes": [{"id": k, "before": old[k]["detail"],
                "after": rows[k]["detail"]} for k in sorted(set(old) & set(rows))
                if old[k]["detail"] != rows[k]["detail"]],
            "six_affected_results": [rows[k] for k in sorted(affected)],
            "raw_score_shapes": [{"model": c["model"], "shape": c["logprobs_shape"],
                                  "kwargs": c["kwargs"]} for c in raw["score_calls"]
                                 if c["kwargs"].get("kv_cache")],
        }
    report = {"affected_cases": affected, "receipt_inventory": inventory,
              "historical_baseline_sha256": sha(baseline_path),
              "comparisons": comparisons,
              "cached_score_calls_identical_across_runners":
                  cached_calls["historical_runner"] == cached_calls["parent_runner"] == cached_calls["fixed_runner"],
              "limitation": "Source-pinned reconstruction. Original executable and raw arrays unavailable; "
                            "new arrays cannot prove the shapes returned in the historical run."}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1)
        stream.write("\n")


if __name__ == "__main__":
    main()
