# semsearch

Architecture search over keramos documents by their graph, not by a trainer's configuration.

Jonathan's question (13 Nov 2026): can slip and keramos make architecture search operate on semantically
meaningful graph differences rather than framework-specific configuration differences? This folder is the
answer: a measured verdict (Phase 1), and a tested search layer with a small CPU demonstration (Phase 2).
Everything here was written and run by Claude Opus 5.5 on 13 Nov 2026; the readings below are its own.

semsearch sits beside slip and changes none of slip's or keramos's files. It uses keramos's validator,
digest, rule 38 and reference interpreter, and slip's decompiler, diff and width solver.

## Verdict

**Partly yes** (Claude Opus 5.5's reading, 13 Nov 2026). keramos documents, with slip's width solver,
are a framework-neutral object on which "the same architecture" and "the difference between two
architectures" can be defined, measured and searched; and doing so removes waste that a config search
cannot see. What the evidence does not yet show is that searching graphs beats searching configs once
both deduplicate and both move through the same space.

- **Yes, identity.** A paperkiln-style config grid of 165,888 valid configs holds 14,400 distinct
  graph classes: 11.5 configs per class. keramos as it stands (rule 38) already collapses the 165,888 to
  40,320 (4.1x); semsearch's canonical form (dataflow order, dead code, spec-licensed aliases, the RoPE
  pairing class) collapses those to 14,400 (a further 2.8x), and a behavioural fingerprint catches
  equivalences no rewrite covers. About a quarter
  of valid one-key config moves change nothing a model computes or can learn.
- **Yes, budget.** At 60 training runs, a config search spent 25.7% of its runs retraining graphs it had
  already trained (5 seeds). With the canonical hash as its cache key the same search spent 9.3% (the
  functional duplicates only the fingerprint sees), and its retrained best was better on 5 of 5 seeds.
- **Yes, difference and moves.** Typed graph edits (sublayer insertion, residual rewiring, op swaps) are
  expressible on any keramos document, from paperkiln's exporter, the third-party builder or slip, and
  always validate. The full graph-edit search found the best graphs on 5 of 5 seeds, rediscovering the
  planted fix by a route no config key spells (a second, global attention).
- **Not shown.** Restricted to the config's own kinds of change, the graph-edit search did no better than
  the config search with canonical dedup (3 seeds to 2). Its gain came with a larger space.
- **Limits.** One synthetic task, small models, 5 seeds, a noisy proxy; the canonical form decides only
  the equivalences it has rules for (general equivalence of graphs is not decidable this way), the class
  level assumes permutation-equivariant initialisation and optimiser, and the FFN width enters only
  through slip's solver and given shapes, since a keramos graph does not fix it.

## Prior art

Checked on 13 Nov 2026 by web search (entries marked * were confirmed in this session; the rest are
well-known papers cited from memory with their arXiv numbers or venues).

| Work | What it does | Taken or open |
|---|---|---|
| AutoML-Zero, Real et al. 2020, arXiv 2003.03384 * | evolves ML programs from primitives; functional equivalence checking: hash the predictions on fixed examples, reuse the fitness of a duplicate (about 4x speed-up) | Taken: behavioural fingerprints to skip duplicate evaluations. semsearch's fingerprint is this idea at initialisation. |
| Unified functional hashing, Gillard et al. 2023, arXiv 2302.05433 * | one functional hash across representations (graphs, code), a cache of results, applied to NAS and algorithm discovery | Taken: representation-independent dedup by behaviour. |
| NAS-Bench-101, Ying et al. 2019, arXiv 1902.09635 * | iso-invariant hashing of cell DAGs: about 510M encodings, 423k unique graphs | Taken: canonical hashing of computation graphs to quotient a search space. |
| NAS-Bench-201, Dong and Yang 2020, arXiv 2001.00326 * | 15,625 encodings, 6,466 unique after isomorphisms; every encoding trained anyway | Taken: the measurement that encodings over-count graphs. |
| NASBOT / OTMANN, Kandasamy et al. 2018, arXiv 1802.07191 * | an optimal-transport distance between network graphs, for Bayesian optimisation | Taken: graph distances for search. |
| BANANAS, White et al. 2021, arXiv 1910.11858 | path encodings of cell DAGs for performance predictors | Taken: graph features for search. |
| PyGlove, Peng et al. 2020, arXiv 2101.08809 * | programs as mutable symbolic trees; search spaces as hyper-values; NAS-Bench-101 case study | Taken: search over a symbolic program, but bound to Python objects of one framework. |
| Archai, NNI | search-space DSLs over PyTorch modules | Framework-bound configuration spaces: what semsearch compares against. |
| NAS-Bench-Graph, Qin et al. 2022, arXiv 2206.09166 | a benchmark of graph neural network architectures | Unrelated despite the name. |
| TASO (Jia et al., SOSP 2019), PET (Wang et al., OSDI 2021), Tensat (Yang et al., MLSys 2021, arXiv 2101.01332 *) | equivalence-preserving rewriting and equality saturation of tensor graphs, to speed one model up | Taken: function-preserving rewrite rules. Used there to optimise a fixed model, not to quotient a search space. |
| NNSmith, Liu et al. ASPLOS 2023, arXiv 2207.13066 * | generates valid DNN graphs from operator specifications with type and shape constraints, to fuzz compilers | Taken: valid-by-construction graph generation over a neutral IR (for fuzzing, not search). |
| GEVO-ML, Liou et al. 2023, arXiv 2310.10211 * | evolutionary mutation of whole models in MLIR | Taken: evolution on a framework-neutral IR (low level, for performance). |
| einspace, Ericsson et al. NeurIPS 2024, arXiv 2405.20838 *; hierarchical CFG NAS, Schrodi et al. NeurIPS 2023 | search spaces as grammars over fundamental operations | Taken: expressive program-like search spaces. |
| STAR, Thomas et al. ICLR 2025, arXiv 2411.17800 * | hierarchical numeric genomes over linear input-varying operators, evolved for LM architectures | Taken: operator-level LM architecture evolution. |
| Primer, So et al. 2021, arXiv 2109.08668 | search over TensorFlow programs for transformer primitives | Taken: LM search over computation programs. |
| Genesys, Cheng et al. NeurIPS 2025, arXiv 2506.20249 * | LLM agents propose LM designs as code units, verified by training | Code as the representation. |
| GraphIR, Liu et al. Aug 2026, arXiv 2608.01633 * | an architecture-level IR beside executable code: computation skeleton, mutation surface, validity envelope, for LLM-guided NAS | Closest in spirit: an IR as a mutation-safe search state. Extracted from code; the abstract says nothing of canonical forms or deduplication. |
| Zero-cost proxies (Abdelfattah et al. 2021), NASWOT (Mellor et al. 2021) | statistics at initialisation to rank architectures | Scoring, not identity. |

Taken: every ingredient. Canonical graph hashing, functional hashing with a fitness cache, graph
distances, function-preserving rewrites, and valid-by-construction mutation over a neutral IR are all
published.

Open, as far as this search found:
- an architecture search whose state is a normative, executable, cross-engine model IR (a spec, a
  validator, a reference interpreter, two engines consuming it), so one identity, one diff and one edit
  set apply to graphs from a trainer's exporter, a third-party builder and a hand-written DSL alike;
- a canonical form built on the IR's own equivalence (rule 38) and widened only by spec-licensed rewrites,
  whose canonical documents the validator accepts and which compute bit-identical logits;
- a measurement of how much a real trainer's configuration space over-counts its graphs, and what a
  search loses to it.

These are engineering contributions on top of known ideas, not new ideas.

## What "semantically meaningful difference" means here

Operationally, on keramos documents (`semsearch/canon.py`, `fingerprint.py`, `semdiff.py`):

1. **Identity** has four levels, each coarser than the last:
   - keramos's **digest** (the document as text, member order and prose aside);
   - keramos's **rule 38** (`structure`: value and slot names, add/mul operand order, defaults, number
     spelling aside);
   - semsearch's **exact** canonical hash: rule 38, plus node order counting only through data
     dependencies (SSA, Merkle-hashed), dead code removed, `per_layer` lists of equal entries, windows at or
     past `model.ctx` (IR_SPEC 5.15: "window >= p + 1 with sinks 0 is plain causal attention"), sinks under an
     unbounded window, an identity `weight_index`, tensor names (only sharing and shapes count), and the FFN
     widths slip solves (a weight shape, which keramos's graph does not fix). Equal exact hashes mean the
     same function on correspondingly named weights.
   - the **class** hash: exact, plus neox RoPE equals interleaved RoPE where the pairing permutation folds
     into private q/k projection rows (and q/k norm gains). This keeps the function class and, since
     slip's i.i.d. initialisation and Adam are permutation-equivariant, the distribution of trained
     models; it does not keep the function on fixed weights. The tests check the permutation numerically.
2. **Behavioural fingerprint**: logits of keramos's reference interpreter on two fixed 32-token probes,
   with slip's random weights keyed by tensor name; a match is agreement within 1e-5 of the largest logit.
   It catches equivalences no rewrite covers (an injection `affine` of scale 0, when a recursion repeats
   once), and checks the hash: on 3,000 sampled configs no exact hash class held two fingerprints.
3. **Semantic difference**: a typed edit script between the canonical documents (insert, delete, replace
   an op; change attributes; rewire an input; change a shape; tie or untie; change the repeat count), with
   a distance (`semdiff.distance`), and slip's diff of the canonical documents as text.
4. **Semantic moves** (`mutate.py`): typed edits on an SSA form of the graph (attribute, heads, op swap,
   q/k norm, sublayer insert and delete from a library of attention, SwiGLU, GELU MLP and causal
   convolution, residual rewiring between parallel and sequential, repeat, tying, width), plus a sublayer
   crossover. Every result passes keramos's validator and slip's width solver; a proposal whose class hash
   equals its parent's is redrawn before any evaluation.

## Phase 1: measurements

### The config space over-counts its graphs

`config_space.SPACE` is a paperkiln-style configuration: the keys of paperkiln's `LlamaConfig` (n_layers,
n_heads, n_kv_heads with 0 meaning n_heads, d_ff, rope_theta, rope_pairing, rope_dims with 0 meaning
head_dim, qk_norm, tie_embeddings, the recursion's repeats and inject), its sliding window (window, sinks),
and one implementation-only key it really has (checkpoint_blocks). d = 32, ctx = 32, vocab = 32. The
lowering writes documents the way paperkiln's exporter does: the default config lowered at pk_llama's
sizes equals pk_llama's embedded document under rule 38 (test). Every one of the 221,184 grid points was
lowered and hashed (`experiments/phase1_measure.py grid`; `results/phase1_grid.json`):

| level | distinct among the 165,888 valid configs | configs per graph |
|---|---|---|
| config dicts | 165,888 | 1 |
| keramos digest, with the FFN width | 48,384 | 3.4 |
| keramos rule 38, with the FFN width | 40,320 | 4.1 |
| semsearch exact | 28,800 | 5.8 |
| semsearch class | 14,400 | 11.5 |

(55,296 grid points are invalid configs: kv heads not dividing heads, rope_dims past head_dim.) Every
class holds at least 4 configs (`class_size_histogram`). What does the merging: checkpoint_blocks (x2, an
implementation key), n_kv_heads 0 against n_heads and sinks under window 0 (the exporter writes one
document: the digest sees it), rope_dims 0 against head_dim (rule 38), window 32 = ctx (exact), and RoPE
pairing (class). On a random sample of 3,000 configs (`results/phase1_sample.json`) the fingerprint
merged a further 116 exact hashes (the injection of scale 0 under one repeat), and split none.

Most of the over-count is already visible to keramos as it stands (digest and rule 38: 4.1x). The
canonical form adds a further 2.8x. Part of the count is a property of the value lists chosen (a window
equal to ctx, an implementation key): a careful author could prune those by hand, if they knew them;
window >= T being plain attention is pinned in paperkiln's own `test_swa.cpp`, and the RoPE pairing
equivalence is not written down anywhere in the trainer.

### Config-space moves are often no-ops

Random walks in the config space, one key changed per move (`phase1_measure.py walks --walks 100 --steps
100 --fingerprint`; `results/phase1_walks.json`): of 10,000 moves, 354 (3.5%) gave an invalid config;
of the 9,646 valid moves 1,401 (14.5%) left the exact hash unchanged, 2,126 (22.0%) the class hash, and
340 more (3.5%) changed the class hash but not the fingerprint (inject under one repeat). So about a
quarter of valid one-key moves change nothing a model computes or could learn. By key, as a share of
its valid moves: checkpoint_blocks 100% no-op, rope_pairing 100% (class), inject 49% (fingerprint), sinks
41%, n_kv_heads 29%, rope_dims and window 15%. 24% of new configs reached in the walks had a class graph the walk had already visited.

Fingerprint margin (`experiments/phase1_fingerprint_margin.py`, `results/phase1_fingerprint_margin.json`):
for 60 random configs and every key changed once, real changes move the fingerprint by at least 0.0046
of the largest logit (rope_theta, the smallest; the others 0.047 or more), against the 1e-5 tolerance;
aliases move it by exactly 0.

## Phase 2: the search layer and a demonstration

### Design

One search algorithm, regularised (aging) evolution (Real et al. 2019): population 12, tournament of 4,
the oldest removed. Four arms differ only in representation, moves and duplicate detection
(`semsearch/search.py`):

- **config**: paperkiln-style config dicts (`config_space.SEARCH_SPACE`, wider value lists than Phase 1:
  4 layers, 8 heads, windows up to 64), one key changed per move, invalid configs redrawn for free, a
  config already trained not trained again (the cache any config search keeps);
- **config+canon**: the same moves, with the cache keyed by the class hash of the lowered document and
  no-op moves redrawn before training: what keramos and semsearch add to an existing config search;
- **semantic-cfg**: documents and typed edits, restricted to the edit kinds the config keys can also
  make (attributes, heads, q/k norm, repeat, tying, width), cache by class hash and fingerprint;
- **semantic**: documents and all ten edit kinds.

Matched budget: 60 training runs per search, every arm. Each seed's initial population is the same 12
configs in all four arms, lowered for the semantic arms. Rediscovery check: every initial member has
`window = 4` (sinks drawn at random), the planted defect. The task needs a reach of 2 to 30 tokens, so
the known-good edit is to give some attention the whole context: config window 0 (or 32, 64), the
attribute edit, or a new global attention sublayer.

Proxy (`semsearch/task.py`): associative recall, 8 key-value pairs then the 8 keys in a new order, loss
read at the 8 value positions; vocab 32, d 32, ctx 32; 300 Adam steps (lr 3e-3, batch 32, clip 1.0) from
slip's initialisation; fitness = minus the held-out loss on 256 fixed sequences. Training runs in
`torch_exec`, which agrees with keramos's reference interpreter to relative error 1e-5 or better. Seeds 0
to 4; each run's training seed is the search seed, so equal graphs get equal fitness. Each search's best
is then retrained on training seeds 100, 101, 102, because its search fitness is the maximum of noisy
draws. Calibration before the search: at 300 steps the base 2-layer model reached 52% and 99% recall on
two seeds, windows 4 and 8 stayed at 6% to 11%, one layer at 15% to 18%: the task separates the planted
defect clearly, and single training runs are noisy (a seed lottery, as in paperkiln's own Atlas).

Run: `python experiments/phase2_search.py --seeds 0 1 2 3 4 --budget 60 --steps 300 --procs 3 --start
window=4` (about an hour on 3 cores; 1,260 training runs including retraining, about 3.5 CPU-hours). Records:
`results/phase2_runs.jsonl` and `phase2_summary.json`; the training journals, which let anyone replay
the searches without training, are in `results/journals/`.

**Replay under the reviewed code.** After the review's fixes to canon.py, the 20 searches were replayed
from their journals (`--tag phase2r --strict-check`; `results/phase2r_*`): every search made the same
decisions and needed 0 new training runs, so the fixes do not touch these experiments. The replay also
checked every fingerprint match again with perturbed weights (gains and biases moved off 1 and 0, a
second weight seed): 92 of 92 matches held.

### Results (5 seeds, mean ± sd; `results/phase2r_summary.json`, `results/phase2_analysis.json`)

| | config | config+canon | semantic-cfg | semantic |
|---|---|---|---|---|
| training runs (the budget) | 60 | 60 | 60 | 60 |
| unique class graphs trained | 47.0 ± 4.8 | 60.0 ± 0.0 | 60.0 ± 0.0 | 60.0 ± 0.0 |
| runs spent on a graph already trained (any detector) | 15.4 ± 6.9 (25.7%) | 5.6 ± 2.7 (9.3%) | 0 | 0 |
| of which: same class hash | 13.0 ± 4.8 | 0 | 0 | 0 |
| of which: same fingerprint | 11.2 ± 6.1 | 5.6 ± 2.7 | 0 | 0 |
| invalid proposals redrawn (free) | 3.6 ± 1.3 | 6.2 ± 3.1 | 0 | 0 |
| no-op proposals redrawn (free) | 0 | 19.4 ± 6.8 | 0 | 0 |
| best fitness found in the search | -1.266 ± 0.791 | -0.505 ± 0.792 | -0.894 ± 0.963 | -0.343 ± 0.763 |
| best, retrained on 3 seeds | -1.392 ± 0.863 | -0.835 ± 0.993 | -1.143 ± 0.987 | -0.352 ± 0.772 |
| best, retrained: recall accuracy | 0.47 ± 0.35 | 0.68 ± 0.39 | 0.61 ± 0.33 | 0.86 ± 0.31 |
| best's parameters | 38,522 | 26,445 | 60,218 | 52,378 |
| seconds per training run | 10.8 | 8.6 | 9.2 | 11.5 |

The semantic arms' waste is zero by construction: their cache is the detector. The column says what the
detectors cost the config arms, not that the semantic arms are free of every duplicate a better detector
might find.

Retrained best per seed (minus loss / recall accuracy), and the training run at which the search first
trained a graph whose every attention sees the whole context / any attention does:

| seed | config | config+canon | semantic-cfg | semantic |
|---|---|---|---|---|
| 0 | -1.909 / 0.22 / 30 / 30 | -0.039 / 0.99 / 33 / 33 | -0.490 / 0.86 / - / - | -0.000 / 1.00 / 30 / 23 |
| 1 | -0.176 / 0.92 / 28 / 28 | -0.041 / 0.99 / 21 / 21 | -2.624 / 0.17 / - / - | -0.000 / 1.00 / 37 / 37 |
| 2 | -0.784 / 0.78 / - / - | -0.258 / 0.92 / 14 / 14 | -0.039 / 0.99 / 34 / 34 | -0.010 / 1.00 / 50 / 50 |
| 3 | -2.115 / 0.21 / 24 / 24 | -1.928 / 0.29 / 22 / 22 | -1.198 / 0.53 / 23 / 23 | -1.733 / 0.30 / 38 / 23 |
| 4 | -1.977 / 0.22 / - / - | -1.909 / 0.22 / 33 / 33 | -1.366 / 0.47 / - / - | -0.018 / 1.00 / 41 / 16 |

Paired by seed (the arms share each seed's initial population), retrained best: config+canon beat config
on 5 of 5 seeds, semantic beat config on 5 of 5 and config+canon on 5 of 5 (two-sided sign test p = 0.0625
each, the smallest five pairs allow); semantic-cfg against config+canon 3 to 2 (p = 1.0).

Rediscovery (the planted window): the best graph of the run gives some attention the whole context in
config 3 of 5 seeds, config+canon 5 of 5, semantic-cfg 2 of 5, semantic 5 of 5. The semantic arm got
there first by inserting a global attention sublayer beside the windowed one (training runs 23, 37, 50,
23, 16), the planted attribute edit later. Its best graph for seed 4 (`best_slip` in the runs file) is a
global attention sublayer with q/k norm and theta 1e6, then the original window-4 attention with an FFN in
parallel with it: a local/global hybrid, found from a window-4 start in 60 training runs.

What set each new best-so-far (`edits_that_set_a_new_best`): in the config arm, a rope_pairing change did
so 5 times. That key is a class no-op, so those "improvements" are a retrained graph's seed luck being
selected: duplicates are not only wasted runs, they feed the winner's curse. In the semantic arm,
residual rewiring (10), inserted sublayers (13: attention 4, convolution 3, MLP 3, SwiGLU 3) and op swaps
(4) did most of the improving: moves no config key spells.

### Reading (Claude Opus 5.5, 13 Nov 2026)

- Deduplication by graph identity is the measured gain that holds up. At an equal budget the config
  search trained a graph it had already trained in a quarter of its runs; with the canonical hash in
  its cache it trained 60 distinct graphs in 60 runs, and its retrained best was better on every seed.
- The full semantic arm found the best graphs, but it also searched a larger space (sublayer insertion,
  rewiring, op swaps). The arm restricted to the config's own kinds of change was no better than the
  config search with canonical dedup. So these runs do not show that the graph representation as such
  makes a search better; they show that dedup helps, and that the moves a graph representation makes
  natural (structure, not settings) were the ones that paid on this task.
- Aliases also distort move probabilities, not only budgets: the config arm proposes "window 0" in three
  spellings (0, 32, 64), which made the planted fix about four times likelier per move there (1 in 28)
  than in semantic-cfg (1 in 120: one of six edit kinds, one of four attributes, one of five windows).
  semantic-cfg's failure to keep the fix in 3 of 5 seeds fits this move distribution rather than a defect
  of graph edits (a reading, not tested separately); a search on graphs has to
  choose its edit distribution deliberately, where a config search inherits one from its schema.
- Limits: 5 seeds, one task, one proxy whose training outcome is bimodal (retrained bests fall from 0.84
  to 0.29 recall on one seed); models of tens of thousands of parameters; my edit library and its probabilities;
  semantic-cfg is close to but not exactly the config space (it can also change softcap and reach theta
  1e3 and 1e6; it lacks inject and sinks as separate moves). The sign tests are at the smallest p five
  pairs allow, and four of the six comparisons are not significant. Treat all of it as exploratory.

## What was built

```
semsearch/
  semsearch/bridge.py        keramos and slip, imported from this checkout
  semsearch/canon.py         canonical hash (exact, class) and canonical document
  semsearch/canon_v1.py      FROZEN: canon.py as it ran Phase 2, used only for the training journals' keys
  semsearch/fingerprint.py   behavioural fingerprints in ir_numpy; name- or canonically-keyed weights
  semsearch/semdiff.py       typed edit script, distance, slip-rendered diff
  semsearch/mutate.py        typed edits on an SSA form, crossover, validity check, no-op-free proposals
  semsearch/config_space.py  paperkiln-style configs and their lowering
  semsearch/torch_exec.py    trainable interpreter of keramos documents (checked against ir_numpy)
  semsearch/task.py          the proxy: associative recall, 300 Adam steps on CPU
  semsearch/search.py        regularised evolution over four representations, journalled
  semsearch/__main__.py      the command line
  experiments/               phase1_measure.py, phase1_fingerprint_margin.py, phase2_search.py
  results/                   every number in this README
  tests/                     160 tests
```

Command line (from this folder; CPU only):

    python -m semsearch hash MODEL [--level exact|class] [--shape @ffn_up=64]
    python -m semsearch canon MODEL [-o OUT.slip]
    python -m semsearch same A B          # rule 38, exact and class verdicts
    python -m semsearch diff A B [--text] # typed edits, distance, slip's diff of the canonical forms
    python -m semsearch mutate MODEL [--kind insert|delete|rewire|...] [--seed S] [--shape ...]

For example, `python -m semsearch same ../slip/examples/llama.slip
../conformance/models/pk_llama/pk_llama.gguf` prints `rule 38: False`, `exact: True`, `class: True`: the
hand-written Llama computes v after RoPE, paperkiln's export before, and slip's README notes their logits
agree bit for bit.

Tests: `python -m pytest tests -q` (about 45 seconds): 156 passed, 4 skipped (documents with per-layer
`apply` or `depth_route`, which are hashed but not emitted as canonical documents). They cover: every
document of the keramos checkout (9 examples, 33 conformance models) hashes, its canonical document
validates, is a fixed point, and computes bit-identical logits on the model's own weights; each alias and
each non-alias; the RoPE class as a weight permutation; edit chains from three starting models and on
five other exporters' documents (pk_gpt2, tp_mistral, tp_qwen3, pk_llama, tp_gptneox) always valid;
torch_exec against ir_numpy on random configs, edited documents and nine conformance models (relative
error under 1e-5); the search's journal replaying without training; and the review's counterexamples.

Independent review: an Explore agent read the code on 13 Nov 2026, given the code and not these
conclusions. It found two high-severity problems (a per-layer `apply` read the hash missed; a crash of the
repeat edit on eight applications) and several medium ones (one-round tensor-sharing identities with a
concrete collision; gaps in the RoPE class guards; that "wasted" counts are zero by construction for the
arms whose cache is the detector; that the restricted semantic arm is not exactly the config space). All
code findings are fixed with tests in `tests/test_review_20261113.py`; the design findings are stated in
this README where they bear on a number. None of the fixed holes can arise in the documents the search
builds, and the replay below confirms it for the runs themselves.

## For slip's owner (proposals, not changes)

- `slip diff` and keramos's rule 38 compare node order; a `--dataflow` mode of `slip diff` could use
  `canon.canonical_document` (or its ordering rule) so that llama.slip and pk_llama read as the same graph.
- slip's width solver leaves an FFN's hidden width free; a search needs it in the identity. A `slip check
  --shapes` mode that prints the free sizes as `--shape` arguments, given a document and a GGUF, would let
  any tool carry them with a document.
- The fingerprint uses `run.random_weights` with name-keyed streams; a stable, documented promise that the
  streams depend only on (seed, tensor instance name) would make fingerprints portable between tools.

## A stronger follow-up

- **A fair representation test.** Give the config search and the graph search exactly one space (the
  config space's graphs, or a config schema extended until it spells every edit), the same move
  distribution over semantic changes, and canonical dedup in both; then any remaining difference is the
  representation's. Pre-register the comparison and the number of seeds before running.
- **A better proxy.** Several MAD-style tasks rather than one; three or more training seeds per evaluation
  (the bimodal outcomes here make single runs a lottery); a ladder of two or three sizes, scoring the
  slope as Chimera's design asked. Then 20+ seeds per arm, so a paired test can reach a real p-value.
- **A rule-driven canonicaliser.** An e-graph (equality saturation, Tensat-style) over rewrite rules each
  derived from an IR_SPEC sentence and each checked numerically against ir_numpy, in place of hand-coded
  aliases: algebraic identities (an affine of scale 0, a scale of gain 1 where it cannot train away),
  grouped-query attention equal to multi-head with tied heads, an FFN of one expert as a dense FFN.
- **Train-time fingerprints.** Fingerprints after a few training steps (AutoML-Zero's functional
  equivalence checking), so that graphs equal at initialisation but not as functions of their weights
  are told apart without the strict check's extra forward pass.
- **Search that uses the distance.** Novelty or MAP-Elites selection on `semdiff` distance and
  fingerprints; a Gaussian-process kernel on the distance (NASBOT-style); crossover in the evolution.
- **Cost from the engines.** keramos documents run in ember, so latency and memory per token on the real
  engine can be a second objective, measured rather than estimated.
- **The edit library from the IR.** Edits generated from the IR's op list and slip's width rules rather
  than hand-written templates: recurrences, linear attention, mixture-of-experts, depth routing.
