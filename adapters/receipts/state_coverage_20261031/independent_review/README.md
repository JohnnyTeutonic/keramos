# Window rewind and targeted regeneration review

The existing Gemma3 fixture already exercises a 48-token session with window four. The new Mistral fixture adds twelve incremental four-token calls, a cut to eight tokens, and a five-token continuation compared with a fresh prefix run.

The test adapter uses real per-layer key/value arrays. Before the cut, its bounded mode holds positions 44–47, while the next token after the cut needs positions 5–7. The arrays for those required rows are absent. Re-prefilling the preserved token prefix passes the numerical conformance check; accepting the cut without restoring those rows fails only the continuation logits. Keeping the full cache passes. Explicit refusal remains distinct from a numerical pass.

These are bounded test-adapter controls, not a claim that every external engine evicts windowed rows. No external engine was built or tested in this review. No model training, paid job, provider access, or watchdog change occurred.

Targeted regeneration used Windows Python 3.13.14 and NumPy 2.3.5 in a temporary suite. The routed GGUF and all three new JSON artifacts were compared byte for byte; the actual outcomes and source hashes are in review.json. The first attempt found only an indentation mismatch in the two new window files; that negative attempt is retained separately. It does not indicate a numerical difference.
