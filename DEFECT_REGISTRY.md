# Defect registry

This is the public record of defects and ambiguities investigated using Keramos.
Each entry connects the execution contract, a reproduction, the evidence and the
upstream disposition. These external engines do not claim to implement Keramos;
the contract supplies an explicit oracle for investigating their native behaviour.

Last published upstream observation: **31 October 2026**.
[Machine-readable issue observations](defects/upstream_status.json).
There are eight filed upstream issues; one contains three distinct mapping findings.
Contributor activity is recorded below. No merged or released fix is established
by the observations currently linked here.

## Reading the status

- **Reported:** a public issue and reproduction exist.
- **Independently reproduced:** someone outside this project reports reproducing
  the same behaviour; this does not mean a maintainer has confirmed it.
- **Proposed fix:** a contributor describes a change; assignment, a submitted PR,
  merge, and release are separate events.
- **Merged / released:** recorded only with a verified fix commit or PR / release.
- **Retested:** the original reproduction was rerun on the identified fixed version;
  a closed issue alone does not establish this.

The registry tracks reported findings, not every invariant in the specification.
It is not a count of all failures in the suite. Known/by-design behaviours and
unattributed discrepancies remain in the adapter findings, outside the issue count.

## Stateful execution

### HF-MAMBA-CHUNK — a continuation must read the primed recurrent state

- Contract: [prefill/decode](IR_SPEC.md#39-prefill-and-decode),
  [recurrent slots](IR_SPEC.md#63-recurrent) and
  [prefix reuse](IR_SPEC.md#69-prefix-reuse-rewind-and-snapshots).
- Finding: a multi-token Mamba-1 continuation starts its selective scan from zero.
- Reproduction: [suite-side script](adapters/hf/repro/mamba_chunks.py);
  [native mamba-130m reproduction and versions](https://github.com/huggingface/transformers/issues/50195).
  [Adapter diagnosis](adapters/hf/FINDINGS.txt).
- Scope: transformers 5.20.2 CPU reference path; the native public-checkpoint
  reproduction distinguishes multi-token continuation from token-by-token continuation.
- Upstream: [transformers #50195](https://github.com/huggingface/transformers/issues/50195).
- Status: **independently reproduced; contributor volunteering a fix**.
  [ps-abhi's reproduction](https://github.com/huggingface/transformers/issues/50195#issuecomment-6540811484);
  [author handoff](https://github.com/huggingface/transformers/issues/50195#issuecomment-6542210716).
  The related Mamba-2 PR is background, not a verified fix for this entry.
- Resolution / retest: no verified fix or retest recorded yet.

### LC-WINDOW-REWIND — a successful rewind must leave a usable prefix state

- Contract: [prefix reuse, rewind and snapshots](IR_SPEC.md#69-prefix-reuse-rewind-and-snapshots).
- Finding: iSWA `seq_rm` reports success after required window rows were overwritten.
- Reproduction: [native API probe](adapters/llamacpp/repro/swa_rm.cpp),
  [setup](adapters/llamacpp/repro/setup2.sh),
  [diagnosis and controls](adapters/llamacpp/FINDINGS.txt).
- Scope: llama.cpp b6000, CPU, synthetic Gemma 3, `swa_full=false`; fresh-prefill,
  retained-window and `swa_full=true` controls distinguish the lost-history case.
- Upstream: [llama.cpp #31457](https://github.com/ggml-org/llama.cpp/issues/31457).
- Status: **reported, open**. No maintainer confirmation or verified fix recorded.
- Resolution / retest: none recorded; the server's existing guard is a control,
  not evidence that the library API was fixed.

## Graph construction and numerical semantics

| ID / finding | Contract used for diagnosis | Reproduction and evidence | Upstream / current status |
|---|---|---|---|
| HF-GEMMA3-CONFIG: GGUF loading loses global RoPE scaling and the attention scalar | [RoPE §5.14](IR_SPEC.md#514-rope), [attention §5.15](IR_SPEC.md#515-attention), metadata-derived graph (Appendix D) | [GGUF probe](adapters/hf/repro/gguf_gemma3.py), [findings](adapters/hf/FINDINGS.txt); transformers 5.20.2, synthetic Gemma 3, same-tensor config substitution controls | [#50196](https://github.com/huggingface/transformers/issues/50196), **proposed fix; contributor volunteering**. [Proposal](https://github.com/huggingface/transformers/issues/50196#issuecomment-6541734739). No independent reproduction or merged fix established by that comment. |
| HF-MIXTRAL-GRAPH: llama-with-experts GGUF becomes a dense Llama with newly initialised MLPs | Metadata-derived graph and [closed-set refusal](IR_SPEC.md#10-validation): preserve expert/router semantics or explicitly refuse unsupported input | [Mixtral probe](adapters/hf/repro/gguf_mixtral.py), [findings](adapters/hf/FINDINGS.txt); synthetic GGUF, direct Mixtral construction as control | [#50197](https://github.com/huggingface/transformers/issues/50197), **reported, open**, part 1 |
| HF-QWEN3-HEAD: GGUF key length does not determine the loaded head dimension | Metadata-derived projection/head geometry and [attention](IR_SPEC.md#515-attention) | [mapping-gap probe](adapters/hf/repro/gguf_gaps.py), [findings](adapters/hf/FINDINGS.txt); non-default synthetic head size, first-forward shape error | [#50197](https://github.com/huggingface/transformers/issues/50197), **reported, open**, part 2 |
| HF-MISTRAL-DETECTION: model-name detection loses Mistral's window | [Attention window](IR_SPEC.md#515-attention), metadata-derived graph | [mapping-gap probe](adapters/hf/repro/gguf_gaps.py), [findings](adapters/hf/FINDINGS.txt); GGUF reader and name/detection controls | [#50197](https://github.com/huggingface/transformers/issues/50197), **reported, open**, part 3 |
| LC-NEOX-GELU: GPT-NeoX uses tanh GELU where the checkpoint declares erf GELU | [GELU §5.8](IR_SPEC.md#58-gelu), GPT-NeoX graph in Appendix D | [Pythia comparison](adapters/llamacpp/repro/pythia_compare.py), [b6000 probe](adapters/llamacpp/probe_b6000.json), [findings](adapters/llamacpp/FINDINGS.txt); synthetic model and activation/table controls | [llama.cpp #31456](https://github.com/ggml-org/llama.cpp/issues/31456), **reported, open** |
| ORT-GEMMA3-LOCAL-ROPE: local rotary caches inherit global scaling | [RoPE §5.14](IR_SPEC.md#514-rope), Gemma 3 per-layer graph | [reproduction](adapters/ortgenai/repro/gemma3.py), [local-only diagnostic correction](adapters/ortgenai/repro/builder_local_rope_fix.py), [findings](adapters/ortgenai/FINDINGS.txt); synthetic model, documented config/head-size workarounds | [ORT GenAI #2831](https://github.com/microsoft/onnxruntime-genai/issues/2831), **reported, open** |
| ORT-MISTRAL-WINDOW: builder omits sliding attention when layer_types is absent | [Attention window §5.15](IR_SPEC.md#515-attention) | [reproduction](adapters/ortgenai/repro/mistral.py), [findings](adapters/ortgenai/FINDINGS.txt); builder 0.18.0, synthetic model, no-window reference control | [ORT GenAI #2832](https://github.com/microsoft/onnxruntime-genai/issues/2832), **reported, open** |
| ORT-GQA-SHAPE: emitted CPU fused-rotary GQA cannot execute the configured head size | [Explicit capability refusal §10.3](IR_SPEC.md#103-engine-capability-refusals) and attention geometry; a native builder/kernel compatibility finding | [head-size matrix](adapters/ortgenai/repro/head_size.py), [unfused control](adapters/ortgenai/repro/builder_unfused_rope.py), [findings](adapters/ortgenai/FINDINGS.txt); CPU kernel constraints differ with fused rotary | [ORT GenAI #2833](https://github.com/microsoft/onnxruntime-genai/issues/2833), **reported, open** |

For the table's open reports, **no verified fix, fixed release or post-fix retest
is recorded yet**. This describes the linked observations, not a claim that no
unlinked work exists. The Gemma 3 proposal is recorded as a proposal; its attention
scalar expression is not endorsed as an implemented or validated correction.

## Keramos-side repairs and specification clarifications

| Finding | Resolution | Validation |
|---|---|---|
| External adapters lacked the runner's `tools` argument, aborting chat cases before SKIP | [197b6ac](https://github.com/JohnnyTeutonic/keramos/commit/197b6ac); both adapter signatures repaired | [Full pinned reruns](adapters/receipts/chat_signature_20261031/): all 598 cases complete per engine, all 437 historical verdicts preserved, 86 chat SKIPs each; four crash regressions pass |
| Two-set LongRoPE's length-dependent denotation was not explicitly named non-causal | [197b6ac](https://github.com/JohnnyTeutonic/keramos/commit/197b6ac), informative clarification in §§3.9/4.5 | Existing factor-selection and recomputation rules preserved; earlier rows across a factor crossing are not a chunk-invariance guarantee |
| Cached-score trace comparison broadcast shortened or wrongly shaped score arrays, allowing a false PASS | Use the existing shape-aware ROUNDING comparison before crediting a trace | [Regression and validation](adapters/receipts/trace_score_shape_20261031/): two old false passes reproduced; eight new controls and the complete test suite pass; external engine receipts not rerun |

These are internal tooling/wording repairs. They are separate from upstream engine
fixes and are not counted as additional external defects.

## Upkeep

Check the linked issues and comments, linked fix PRs and their merge state, and
release notes when a fixed release is claimed. Record the exact URLs and version.
Update this registry and `defects/upstream_status.json` together, with meaningful
changes in Git history. A scheduled daily check maintains the record; its last
published observation remains visible above. Unchanged checks do not create a new
public claim or notification. If access fails, preserve the last verified status.

Retesting uses the original reproduction, recorded configuration and numerical
bound. Record FAIL or operational ERROR as such. Never infer a fix from closure,
silently weaken a bound, or overwrite historical receipts. Runtime work and paid
compute remain separate from status upkeep. No comments or messages are posted to
other projects as part of upkeep.
