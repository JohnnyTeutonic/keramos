# Defect registry

This is the public record of defects and ambiguities investigated using Keramos.
Each entry connects the execution contract, a reproduction, the evidence and the
upstream disposition. These external engines do not claim to implement Keramos;
the contract supplies an explicit oracle for investigating their native behaviour.

Last published upstream observation: **12 November 2026 (Sydney)**.
[Machine-readable issue observations](defects/upstream_status.json).
There are twelve filed upstream issue numbers, including three duplicate ORT reports;
the paper groups the findings as nine defects (G1-G9), with three mapping subfindings.
The earlier count of eight omitted four ORT filings from 27 October.
Contributor activity is recorded below. The GenAI sliding-window fix has merged; no fixed release or original-reproduction retest is verified for it.

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
- Status: **independently reproduced; candidate PR submitted and CPU retested**.
  [ps-abhi's reproduction](https://github.com/huggingface/transformers/issues/50195#issuecomment-6540811484);
  [author handoff](https://github.com/huggingface/transformers/issues/50195#issuecomment-6542210716).
  The related Mamba-2 PR is background, not a verified fix for this entry.
- Resolution / retest: [PR #50250](https://github.com/huggingface/transformers/pull/50250), head `be493dda56a8418dd9d2ebf4ce63d33dd93392af`, remains open and unmerged. The [reporter CPU retest](https://github.com/huggingface/transformers/pull/50250#issuecomment-6577288502) reduced the mamba-130m chunk error from about 116 to 0.0001 and passed chunk, truncate and snapshot traces. GPU kernels and the other three model families were not tested by this retest. No merged or released fix is established.

### LC-WINDOW-REWIND — a successful rewind must leave a usable prefix state

- Contract: [prefix reuse, rewind and snapshots](IR_SPEC.md#69-prefix-reuse-rewind-and-snapshots).
- Finding: iSWA `seq_rm` reports success after required window rows were overwritten.
- Reproduction: [native API probe](adapters/llamacpp/repro/swa_rm.cpp),
  [setup](adapters/llamacpp/repro/setup2.sh),
  [diagnosis and controls](adapters/llamacpp/FINDINGS.txt).
- Scope: llama.cpp b6000, CPU, synthetic Gemma 3, `swa_full=false`; fresh-prefill,
  retained-window and `swa_full=true` controls distinguish the lost-history case.
- Upstream: [llama.cpp #31457](https://github.com/ggml-org/llama.cpp/issues/31457).
- Status: **independently reproduced; candidate PR submitted, open**.
  [kukirin-ui's Gemma 3 reproduction and proposed guard](https://github.com/ggml-org/llama.cpp/issues/31457#issuecomment-6544365998)
  and [author handoff](https://github.com/ggml-org/llama.cpp/issues/31457#issuecomment-6544470364)
  supersede the earlier reported-only observation.
  [PR #31483](https://github.com/ggml-org/llama.cpp/pull/31483), head
  `35e2854b272498ef79d503206782ba8aa7cc4450`, proposes refusing the cut before
  touching either cache. Its author reports a before/after regression and existing
  context tests passing; the [reporter retest](https://github.com/ggml-org/llama.cpp/pull/31483#issuecomment-6577358662) subsequently confirmed refusal at exactly the broken cut points. A bot flags the new-contributor
  open-PR limit. A requested reviewer is not assignment or maintainer confirmation.
- Resolution / retest: the candidate head has a successful original-reproduction retest; it remains open and unmerged. The server's existing guard is a control,
  not evidence that the library API was fixed.

## Graph construction and numerical semantics

| ID / finding | Contract used for diagnosis | Reproduction and evidence | Upstream / current status |
|---|---|---|---|
| HF-GEMMA3-CONFIG: GGUF loading loses global RoPE scaling and the attention scalar | [RoPE §5.14](IR_SPEC.md#514-rope), [attention §5.15](IR_SPEC.md#515-attention), metadata-derived graph (Appendix D) | [GGUF probe](adapters/hf/repro/gguf_gemma3.py), [findings](adapters/hf/FINDINGS.txt); transformers 5.20.2, synthetic Gemma 3, same-tensor config substitution controls | [#50196](https://github.com/huggingface/transformers/issues/50196), **candidate PR submitted, open and unmerged** ([#50274](https://github.com/huggingface/transformers/pull/50274), head `d07db5dc5a28e27412da5c825b4ac2df9059588c`). Maintainer requested the new loader path; reporter-supplied CPU checks do not establish an independent original-reproduction retest. Earlier [proposal](https://github.com/huggingface/transformers/issues/50196#issuecomment-6541734739). No independent reproduction or merged fix established by that comment. |
| HF-MIXTRAL-GRAPH: llama-with-experts GGUF becomes a dense Llama with newly initialised MLPs | Metadata-derived graph and [closed-set refusal](IR_SPEC.md#10-validation): preserve expert/router semantics or explicitly refuse unsupported input | [Mixtral probe](adapters/hf/repro/gguf_mixtral.py), [findings](adapters/hf/FINDINGS.txt); synthetic GGUF, direct Mixtral construction as control | [#50197](https://github.com/huggingface/transformers/issues/50197), **candidate PR closed unmerged; new-loader implementation requested**, part 1: [#50206](https://github.com/huggingface/transformers/pull/50206) |
| HF-QWEN3-HEAD: GGUF key length does not determine the loaded head dimension | Metadata-derived projection/head geometry and [attention](IR_SPEC.md#515-attention) | [mapping-gap probe](adapters/hf/repro/gguf_gaps.py), [findings](adapters/hf/FINDINGS.txt); non-default synthetic head size, first-forward shape error | [#50197](https://github.com/huggingface/transformers/issues/50197), **candidate PR closed unmerged; new-loader implementation requested**, part 2: [#50206](https://github.com/huggingface/transformers/pull/50206) |
| HF-MISTRAL-DETECTION: model-name detection loses Mistral's window | [Attention window](IR_SPEC.md#515-attention), metadata-derived graph | [mapping-gap probe](adapters/hf/repro/gguf_gaps.py), [findings](adapters/hf/FINDINGS.txt); GGUF reader and name/detection controls | [#50197](https://github.com/huggingface/transformers/issues/50197), **candidate PR closed unmerged; new-loader implementation requested**, part 3: [#50206](https://github.com/huggingface/transformers/pull/50206) |
| LC-NEOX-GELU: GPT-NeoX uses tanh GELU where the checkpoint declares erf GELU | [GELU §5.8](IR_SPEC.md#58-gelu), GPT-NeoX graph in Appendix D | [Pythia comparison](adapters/llamacpp/repro/pythia_compare.py), [b6000 probe](adapters/llamacpp/probe_b6000.json), [findings](adapters/llamacpp/FINDINGS.txt); synthetic model and activation/table controls | [llama.cpp #31456](https://github.com/ggml-org/llama.cpp/issues/31456), **contributor proposal; activation scope clarified**, open; details below |
| ORT-GEMMA3-LOCAL-ROPE: local rotary caches inherit global scaling | [RoPE §5.14](IR_SPEC.md#514-rope), Gemma 3 per-layer graph | [reproduction](adapters/ortgenai/repro/gemma3.py), [local-only diagnostic correction](adapters/ortgenai/repro/builder_local_rope_fix.py), [findings](adapters/ortgenai/FINDINGS.txt); synthetic model, documented config/head-size workarounds | [ORT GenAI #2831](https://github.com/microsoft/onnxruntime-genai/issues/2831), **reported, open** |
| ORT-MISTRAL-WINDOW: builder omits sliding attention when layer_types is absent | [Attention window §5.15](IR_SPEC.md#515-attention) | [reproduction](adapters/ortgenai/repro/mistral.py), [findings](adapters/ortgenai/FINDINGS.txt); builder 0.18.0, synthetic model, no-window reference control | [ORT GenAI #2832](https://github.com/microsoft/onnxruntime-genai/issues/2832), **closed after candidate fix merged; release and retest unverified** |
| ORT-GQA-SHAPE: emitted CPU fused-rotary GQA cannot execute the configured head size | [Explicit capability refusal §10.3](IR_SPEC.md#103-engine-capability-refusals) and attention geometry; a native builder/kernel compatibility finding | [head-size matrix](adapters/ortgenai/repro/head_size.py), [unfused control](adapters/ortgenai/repro/builder_unfused_rope.py), [findings](adapters/ortgenai/FINDINGS.txt); CPU kernel constraints differ with fused rotary | [ORT GenAI #2833](https://github.com/microsoft/onnxruntime-genai/issues/2833), **candidate PR submitted, open and unmerged**: [#2850](https://github.com/microsoft/onnxruntime-genai/pull/2850), head `aebe897e18f5caa06bab2108649d375ef3af71b5`; details below |

For the table's open reports, **no verified fix, fixed release or post-fix retest
is recorded for the remaining table entries**. This describes the linked observations, not a claim that no
unlinked work exists. The earlier Gemma 3 proposal remains historical; its attention scalar expression
is not endorsed by recording the separately submitted new-loader PR.

For #50197, RohanQ17's [PR #50206](https://github.com/huggingface/transformers/pull/50206)
at head `632906fb2a33eb1045f2ce0e3585e0ad32ecf14f` proposes all three mapping
corrections and reports six CPU tests passing. It was [closed unmerged by SunMarc](https://github.com/huggingface/transformers/pull/50206#issuecomment-6579335609): support should be added through the new GGUF loader rather than repairing the legacy path. The [reporter retest](https://github.com/huggingface/transformers/pull/50206#issuecomment-6577321632) found the Mixtral/Qwen3 fixes but also an unintended default Mistral window and Llama 4 acceptance as Mixtral. Closure does not resolve those findings. A separate contributor-fork
[test commit](https://github.com/anishpentyala/transformers/commit/67eb3ee8b236d2ec6264aa5d5f5671503a2ba05a)
references the issue; it is test work, not an upstream merged fix.

For #31456, [kukirin-ui points out](https://github.com/ggml-org/llama.cpp/issues/31456#issuecomment-6544370718)
that GPT-NeoX-20B uses `gelu_fast` (tanh), so replacing every GPT-NeoX activation
with erf would be wrong. The proposal carries `hidden_act` through conversion and
selects the graph activation from it; the contributor reports improved Pythia-14m
logits. [The reporter agrees with this scope correction](https://github.com/ggml-org/llama.cpp/issues/31456#issuecomment-6544470630).
This preserves the original erf-checkpoint finding, rather than generalising it
to every GPT-NeoX checkpoint. No corresponding submitted PR or Keramos retest was
verified in this check.

New discussion on [Mamba #50195](https://github.com/huggingface/transformers/issues/50195#issuecomment-6602374065) questions the candidate's per-token-loop performance and treats it as low priority. The [contributor reports](https://github.com/huggingface/transformers/issues/50195#issuecomment-6613545704) overhead up to 3x without kernels and 109x for a 1024-token chunk with kernels on mamba-130m/L4, and offers a parallel alternative. These are contributor measurements, not independently reproduced here. The candidate head and historical CPU correctness retest are unchanged; no merge or release is established. A [subsequent maintainer comment](https://github.com/huggingface/transformers/issues/50195#issuecomment-6629369869) expresses uncertainty about the fix direction and suggests documenting the limitation for users, noting its dependence on the kernel implementation. This is documentation direction, not a submitted documentation PR or a verified fix.

For ORT #2833, [PR #2850](https://github.com/microsoft/onnxruntime-genai/pull/2850) selects standalone rotary or packed Attention according to CPU head size. An [automated review](https://github.com/microsoft/onnxruntime-genai/pull/2850#discussion_r4450332017) flagged QKV packing with standalone rotary in an earlier revision. At the current head, the [contributor reports](https://github.com/microsoft/onnxruntime-genai/pull/2850#issuecomment-6599012332) 121 focused tests passing, two full-builder-suite failures and one of 19 native CTest entries failing. Those failures also occurred on earlier PR heads; that does not prove they predate the entire PR. No maintainer confirmation, successful original-reproduction runtime retest, merge or release is established by these reports.

## Earlier ORT filings and current candidates (5 November 2026)

The 31 October triage omitted the 27 October filings: [#2788](https://github.com/microsoft/onnxruntime-genai/issues/2788) is the separate Gemma 3 nested-RoPE build failure (`KeyError: rope_type`); [#2789](https://github.com/microsoft/onnxruntime-genai/issues/2789)/#2831 are duplicate local-RoPE reports, [#2790](https://github.com/microsoft/onnxruntime-genai/issues/2790)/#2832 duplicate window reports, and [#2791](https://github.com/microsoft/onnxruntime-genai/issues/2791)/#2833 duplicate CPU-GQA reports. Duplicate reports add no defects. The paper groups nine defects under twelve issue numbers; the mapping finding contains three subfindings.

Outside contributor 14H034160212 [traces both Gemma defects to source](https://github.com/microsoft/onnxruntime-genai/issues/2789#issuecomment-6643580433) and [confirms the window path](https://github.com/microsoft/onnxruntime-genai/issues/2832#issuecomment-6643569702). These are source confirmations, not maintainer confirmation or independently verified runtime retests. The contributor attributes the window regression to #2386/0.16.0 and names further model configurations; those extensions were not measured here.

At the earlier observation, local-RoPE [PR #2869](https://github.com/microsoft/onnxruntime-genai/pull/2869), head `e6d558e652de62a9273b0d39b5819d58fbef3742`, and window [PR #2866](https://github.com/microsoft/onnxruntime-genai/pull/2866), head `eacfe86684910c38f3a10a264940ab92a2e6d19b`, are open/unmerged. Their author test claims are not independent original-reproduction retests. #2869 explicitly does not fix #2788; no candidate addressing that build failure was verified.

GQA [PR #2810](https://github.com/microsoft/onnxruntime-genai/pull/2810), head `804084ee9c104c4689f56fdb775c71c15d1b2600`, is open/unmerged alongside #2850, now head `86e8d05d6a7fccb6b28ef193efa24dc72d9c3525`. The historical #2850 tests above are of `aebe897e`, not its new head. [Review guidance](https://github.com/microsoft/onnxruntime-genai/pull/2810#pullrequestreview-5707714278) requests a kernel change instead. Linked [ORT PR #33768](https://github.com/microsoft/onnxruntime/pull/33768), head `11f2e9166c0c5306f9edefa1dd9d42517b206f44`, is open/unmerged and explicitly awaits native build and numerical verification. The earlier summary of no reviewer response omitted this #2810 guidance. An anticipated release is not a verified fixed release.

No new merge, fixed release or successful original-reproduction retest is established. Historical numerical bounds and retests remain unchanged. Original ORT diagnoses and reproductions are linked from [the findings](adapters/ortgenai/FINDINGS.txt).

## Keramos-side repairs and specification clarifications

| Finding | Resolution | Validation |
|---|---|---|
| External adapters lacked the runner's `tools` argument, aborting chat cases before SKIP | [197b6ac](https://github.com/JohnnyTeutonic/keramos/commit/197b6ac); both adapter signatures repaired | [Full pinned reruns](adapters/receipts/chat_signature_20261031/): all 598 cases complete per engine, all 437 historical verdicts preserved, 86 chat SKIPs each; four crash regressions pass |
| Two-set LongRoPE's length-dependent denotation was not explicitly named non-causal | [197b6ac](https://github.com/JohnnyTeutonic/keramos/commit/197b6ac), informative clarification in §§3.9/4.5 | Existing factor-selection and recomputation rules preserved; earlier rows across a factor crossing are not a chunk-invariance guarantee |
| Cached-score trace comparison broadcast shortened or wrongly shaped score arrays, allowing a false PASS | Use the existing shape-aware ROUNDING comparison before crediting a trace | [Regression and validation](adapters/receipts/trace_score_shape_20261031/): two old false passes reproduced; eight new controls pass. [Trainer impact audit](adapters/receipts/trace_score_impact_20261031/): all 524 historical verdicts and six cached-score trace passes reproduced unchanged under old/fixed runners; missing historical arrays/executable explicitly noted. |
| Routed LongRoPE's factor choice used active-subset length; original suite missed this combination | Engine repair `1fd77ed`; promoted to trace `routed_longrope_session_length` | [Actual pre-fix/fixed Rust](adapters/receipts/routed_actual_rust_parent_20261031/): old fails at 2.03e-4 against unchanged 1e-5, corrected passes identical inputs. Historical study fixtures remain unchanged. |
| Window-rewind coverage did not demonstrate that needed cache rows were physically discarded | Incremental 48-token Mistral rewind plus fresh-prefix control | [Coverage evidence](adapters/receipts/state_coverage_20261031/): bounded per-layer cache actually drops needed rows; safe re-prefill and legal retention pass, unsafe acceptance fails numerically. This is not a claim that every external engine evicts. |

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

On Gemma 3 #50196, [SunMarc requests migration to the new loader](https://github.com/huggingface/transformers/issues/50196#issuecomment-6581298370). This is maintainer direction, not assignment, a validated correction or a merged fix. The original contributor proposal remains historical evidence.

## CPU GQA candidate follow-up (6 November 2026)

PR #2850 remains open/unmerged and now has head `80ac7a53399221e16a8e2182bde94b18ccb4f616`; the `86e8d05` head above is historical. [Human reviewer tianleiwu](https://github.com/microsoft/onnxruntime-genai/pull/2850#pullrequestreview-5777161077) reports that older head `aebe897e` passed 121 focused tests and tiny Llama head sizes 8/16/24 within 3e-7 of PyTorch, while head sizes 4/12 exported unloadable graphs. The base loaded before encountering its original kernel restriction, identifying an additional fallback regression. That review is currently **dismissed**, not an approval of the new head. These are the reviewer's runtime measurements, not a fresh independent Keramos retest.

[Automated review](https://github.com/microsoft/onnxruntime-genai/pull/2850#pullrequestreview-5781170817) additionally flags per-layer Gemma4 geometry, SmolLM3 fusion, Qwen3.5 cache selection, lost attention features, test defects and missing end-to-end coverage. These are automated findings, not independently verified defects. #2869's automated approval recommendation likewise is not maintainer approval or a merged fix. The original bounds, older test reports and unresolved current-head status remain preserved; no merged or released fix is established.

## ORT kernel candidate follow-up (8 November 2026)

Linked [ORT PR #33768](https://github.com/microsoft/onnxruntime/pull/33768) remains open and unmerged at head `866a22a42dc24500a76f2e88da74fe560b3763db`; the earlier `11f2e916` observation above remains historical. The [contributor update](https://github.com/microsoft/onnxruntime/pull/33768#issuecomment-6703869862) reports an upstream sync and a FlashAttention fallback assertion, and says the stale-review policy invalidated the previous approval. Both recorded human reviews are now [dismissed](https://github.com/microsoft/onnxruntime/pull/33768#pullrequestreview-5809443079), [including the later review](https://github.com/microsoft/onnxruntime/pull/33768#pullrequestreview-5811497533); neither establishes current-head approval.

The current PR body reports earlier numerical CI success and describes accepting the MATH fallback under ORT_QUICK_BUILD. These remain contributor claims. The read-only current-head workflow listing (first page only) instead returns `action_required`, including [Linux CUDA CI](https://github.com/microsoft/onnxruntime/actions/runs/42368988690) and [Linux CI](https://github.com/microsoft/onnxruntime/actions/runs/42368988687). This does not establish current-head native or numerical success. No merged fix, fixed release or successful original-reproduction retest is established; the twelve issue states and other eight candidate heads remain unchanged.


## Current-head approval and candidate update (9 November 2026)

ORT kernel [PR #33768](https://github.com/microsoft/onnxruntime/pull/33768) now has an active [human approval by kunal-vaishnavi](https://github.com/microsoft/onnxruntime/pull/33768#pullrequestreview-5817385314), submitted 8 November at 09:36 UTC and explicitly tied to current head 866a22a42dc24500a76f2e88da74fe560b3763db. This supersedes the earlier observation of no current approval; the two dismissed reviews remain historical. The PR is still open and unmerged.

A complete current-head workflow listing returns 64 runs: 26 failure, 32 cancelled and 6 success. [Linux CUDA CI](https://github.com/microsoft/onnxruntime/actions/runs/42368988690) reports failure and [Linux CI](https://github.com/microsoft/onnxruntime/actions/runs/42368988687) cancellation, superseding their earlier action-required states. These workflow conclusions do not identify failure causes or establish a successful original-reproduction retest. Approval, CI success, merge and release remain separate.

GenAI [PR #2850](https://github.com/microsoft/onnxruntime-genai/pull/2850) is now at head ea73ac5c7f261f128a289781d8b80ab18c4b1b37, still open/unmerged; 80ac7a53 above is historical. Its current body reports 88 focused tests and a broader builder run with 2650 passed, 14 skipped, 346 subtests and 6 failures. These are author reports, not independently verified current-head runtime results; earlier 109/121-test reports and dismissed reviewer measurements retain their original scope. All twelve issue states and the other candidate heads are unchanged. No merged fix, fixed release or new successful original-reproduction retest is established.

### 12 November 2026 upstream observation

[GenAI PR #2866](https://github.com/microsoft/onnxruntime-genai/pull/2866) merged to main on 11 November at 01:48:13 UTC as [5e4a263](https://github.com/microsoft/onnxruntime-genai/commit/5e4a2630421a0221ff1c2e3c74193fe1b2c0b3dc). The change applies a configured sliding window when layer types are absent, unless sliding windows are explicitly disabled. #2832 closed as completed; duplicate #2790 remains open. The latest listed [release, v0.19.0](https://github.com/microsoft/onnxruntime-genai/releases/tag/v0.19.0), predates the merge. No fixed release or successful Keramos original-reproduction retest was verified. The PR author’s builder tests are separate evidence.
