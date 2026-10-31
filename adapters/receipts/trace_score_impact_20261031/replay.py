"""Replay the historical trainer suite with explicit runner/fixture separation.

Uses a source-pinned rebuilt trainer, not the deleted historical executable.
Outputs are exclusive-create: never overwrite an earlier audit.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("immutable output already exists")
    runner = load("ir_conformance", args.runner)
    runner.SUITE = args.fixtures
    adapter_module = load("historical_mtstudio_adapter", args.adapter)
    adapter = adapter_module.MtstudioAdapter(str(args.binary))
    calls = []
    original_score = adapter.score

    def score(*pos, **kw):
        result = original_score(*pos, **kw)
        calls.append({
            "model": str(pos[0]), "prompt": pos[1], "continuation": pos[2],
            "kwargs": kw, "result": result,
            "logprobs_shape": list(np.asarray(result["logprobs"]).shape),
        })
        return result

    adapter.score = score
    # run_suite writes the ordinary receipt; retain raw adapter arrays separately.
    counts = runner.run_suite(adapter, only=args.only, out_json=args.output)
    source = {
        "audit_harness_sha256": sha(Path(__file__)),
        "python": sys.version, "numpy": np.__version__,
        "runner": {"path": str(args.runner), "sha256": sha(args.runner)},
        "runner_modules": {p.name: sha(p) for p in sorted(args.runner.parent.glob("*.py"))},
        "adapter": {"path": str(args.adapter), "sha256": sha(args.adapter)},
        "binary": {"path": str(args.binary), "sha256": sha(args.binary)},
        "helper_binaries": {name: sha(args.binary.with_name(name))
                            for name in ("ir_eval", "expr_eval")},
        "fixtures": {str(p.relative_to(args.fixtures)): sha(p)
                     for p in sorted(args.fixtures.rglob("*")) if p.is_file()},
        "counts": counts, "score_calls": calls,
    }
    with args.output.with_suffix(".raw.json").open("x", encoding="utf-8") as stream:
        json.dump(source, stream, indent=1)
        stream.write("\n")
    return int(bool(counts["FAIL"] or counts["XPASS"]))


if __name__ == "__main__":
    raise SystemExit(main())
