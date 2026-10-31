# Adapter chat signature repair

The runner passes `tools` to `render_chat`. Both external adapters previously
raised TypeError before they could signal that IR-template rendering is unsupported.
Four regressions reproduced the crash before the fix; all four pass afterward.
The full Python test suite passes 248 tests with 10 optional skips.

Actual pinned adapters now complete all 598 cases, including 86 chat SKIPs each:

| Adapter | PASS | FAIL | SKIP |
|---|---:|---:|---:|
| transformers | 38 | 3 | 557 |
| ONNX Runtime GenAI | 8 | 8 | 582 |

Both exit 1 because their historical numerical/model failures remain. Every
historical verdict is unchanged; the additional cases are SKIP. See the raw logs,
complete reports, source commit in summary.json and historical_comparison.json.
No engine implementation, tolerance, fixture or expected value was changed.

IR_SPEC sections 3.9 and 4.5 also name the existing non-causal two-set LongRoPE
semantics and the limitation on comparing earlier outputs across a factor crossing.
Factor selection and recomputation rules are unchanged.
