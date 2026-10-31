# Conformance correctness guards, 31 October 2026

This is a tooling correction against publication main at the baseline in validation.json.
No normative statement, schema, reference arithmetic, suite fixture, expected value or historical
engine receipt changes.

The runner previously used NaN-ignoring reductions for ROUNDING tensors and a running maximum
that ignored NaN model-score errors. A correct finite coordinate or a reported expected argmax
could therefore hide a one-sided NaN. It also logged an omitted required recurrent state without
failing the op case.

The fix checks NaN masks before reduction, keeps matched NaNs out of the finite reference scale,
uses the same guard for scored log-probabilities without reducing their float64 precision, checks
score shapes, and requires each fixture-specified state slot and tensor. Extra or unrequested
state remains ignored. The independent small recurrent fixture is also checked against the
actual NumPy interpreter.

Section 4.3 has no matching-infinity convention. The tool explicitly cannot establish a
ROUNDING pass when either tensor contains infinity, rather than treating infinity arithmetic
as numerical evidence. EXACT infinity comparisons and the section 4.2 NaN convention remain
unchanged. This does not declare an infinity-bearing document invalid or add a refusal rule.

## Evidence

- negative_before.log: the final focused tests against the baseline runner: 26 failed, 12 passed.
- negative_after.log: the same focused tests and source with the fix: 38 passed.
- pytest_after.log: full Python suite: 252 passed, 2 optional skips.
- reference_after.json and cleanroom_after.json: each 587 PASS, 0 FAIL, 11 gradient-only SKIP.
  All 598 verdicts agree with their preserved historical receipts.
- ruff.log: checks pass. Source-only git diff whitespace checks pass.
- validation.json: source, test, baseline-runner and actual Rust binary hashes.
- proof_executed.py: the exact local proof script that generated the final before/after logs and
  compared all verdicts. Its paths and main resolution describe this execution; validation.json
  records the actual immutable baseline. It is not a general-purpose launcher.
- negative_initial_fixture.log preserves an earlier exploratory fixture run; it is not the
  final before/after evidence. rust_build_initial_path_failure.log preserves a PATH-only
  invocation failure; rust_build.log is the subsequent successful one-core build.

Checks used one CPU budget slot. No GPU, network operation, new training, release tag or paid
compute was used. These checks establish harness correctness and preserved supported conformance,
not model quality or hardware-backend conformance.
