# Cached-score trace shape validation

The cached-score fallback in `tools/ir_conformance.py:run_trace` subtracted an
adapter's log-probability array directly from the expected vector. NumPy could
broadcast a shortened vector or a column matrix, producing zero error and PASS
without one score per continuation token.

On parent `f7cc616a76807ed22b71a04e2fa5d07a82c10011`, the new eight-case regression
has two failures: `[-1]` and `[[-1], [-1]]` are both accepted against `[-1, -1]`
with matching reported greedy IDs. The other six controls pass.

The repair uses the runner's existing shape-aware ROUNDING comparison, retaining
float64 score arithmetic and the existing bound. It changes no fixture, numerical
expected value, engine implementation, schema or specification. Nonfinite scores
also follow the same comparison rules as the model-score path.

Validation on Windows, Python 3.13.14 / NumPy 2.3.5:

- `python -m pytest tests/test_ir_conformance.py tests/test_ir_adapter_protocol.py -q`:
  66 passed, including the complete NumPy conformance suite regression.
- `python -m pytest tests -q`, through the root shared CPU-budget wrapper with
  two cores: 256 passed, 10 optional skips.
- `git diff --check`: clean.

The regression uses a dependency-free score adapter and a two-token independent
fixture. These tests do not rerun the pinned external engines or change their
historical receipts. WSL was unavailable during this repair.
