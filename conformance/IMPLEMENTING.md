# Implementing keramos: a guide for engine authors

For an engineer who knows neither paperkiln nor ember and wants to run keramos documents in an engine,
or to measure an existing engine against the suite. The normative text is [`IR_SPEC.md`](../IR_SPEC.md);
this page is informative. It says where the contract is, how to join the suite, what each verdict means,
how to declare a version and how to report a run.

## 1. Where the contract is

- **Normative:** `IR_SPEC.md` sections 2 to 8, 10 and 11, and Appendix D where section 3.1 cites it
  (section 1.2). `ir.schema.json` is the document's structure. A useful order: 3 (the document),
  4 (numerics and classes), 5 (core ops), 10.1 (what a consumer refuses), 11 (the suite), then 6 to 8
  for the profiles you implement. Section 12 records every question an implementer has raised and its
  answer; read it when the text seems to leave a choice.
- **Not normative:** `tools/` is the reference implementation, not the contract. The clean-room Rust
  engine (`cleanroom/`) was written without opening `tools/` (`cleanroom/READ_LOG.md`) and passes every
  case except the gradient ones, so the text and the fixtures suffice. Where the reference and the text
  differ, the text holds; report the difference.
- **References you can skip.** The text names paperkiln's files (`src/`, `include/microtorch/ir.hpp`),
  ember.cpp's, and the workspace documents `LOCAL_LLM_STACK.md` and `LOCAL_LLM_WORK.md`. These record how
  the two in-house engines were gated; the documents are not in this repository, and nothing an engine
  computes depends on them: each statement you need is in the text itself (for example section 4.3's
  greedy parity and section 3.3's context limit). Fixture comments name files by their paths in paperkiln,
  which vendors this repository at `spec/` (`spec/IR_SPEC.md` is `IR_SPEC.md`).

## 2. Choose the revision you implement

A revision R.S is a tag `vR.S.P` of this repository (section 2.6). Implement one revision and run the
suite of its tag:

    git checkout v1.6.2          # revision 1.6
    git describe --tags          # v1.6.2: the suite you run is exactly that tag's

The main branch can hold text and cases not yet in a tag (on 13 November 2026 it was `v1.6.2-29`, with
three more traces than the tag); a report against it names the commit. Declare the revision in `hello`
(section 6 below).

## 3. Write an adapter

An adapter is a program the runner starts once and talks to in JSON Lines (section 11.3):

    python tools/ir_conformance.py --adapter "cmd:my_engine --conformance" --json results.json

- One request per line on stdin, `{"id", "method", "params"}`; one reply per line on stdout, in order,
  echoing the request's integer `id` with exactly one of `result` (an object), `error` (a string) or
  `skip` (a string). Write UTF-8 and flush after every line. Logs go to stderr: anything else on stdout
  breaks the protocol.
- Never exit while stdin is open. Catch your engine's failures and reply `error`: if the process dies,
  the current case and every later one fail.
- Paths in `gguf` and `ir` are absolute paths of the runner's platform, so the runner and the engine run
  on one platform (a Linux engine with a runner on Linux or WSL).
- The runner splits the `cmd:` text by POSIX shell rules, which drop backslashes: on Windows write the
  program's path with forward slashes (`cmd:C:/engines/my_engine.exe --conformance`).
- The runner removes every `EMBER_*`, `MICROTORCH_*` and `MTSTUDIO_*` variable from the engine's
  environment unless named with `--keep-env NAME`.
- A Python engine can instead subclass `ir_conformance.Adapter` (tensors as float32 arrays) and run
  through `ir_conformance.main(argv, adapters={"name": factory})`; any method it does not override
  raises `NotImplementedError`, which is a skip.

**`hello` is required** (the runner stops if it fails). Its members:

| Member | Meaning, and what the runner does with it |
|---|---|
| `engine` | the engine's name. `conformance/expected_fail.json` entries apply to every engine whose name *starts with* their key, so choose a name no existing key prefixes |
| `version` | the engine's own version and build, free text; recorded in the results file |
| `backend` | `"cpu"`: ROUNDING tolerance 1e-5, and each op case's own `tol`. Any other string (`"cuda"`, `"wasm"`, ...): 5e-5 for every case |
| `exact` | `true`: EXACT op cases are compared bit for bit. `false`: they are judged at the ROUNDING tolerance (a tiled or contracted path, section 4.2). Dequantisation is bit for bit either way (section 4.7) |
| `profiles` | the profiles of section 2.4 the engine implements; recorded and summarised, never used to choose cases |
| `methods` | the methods built; recorded, never used to choose cases |
| `spec` | optional: the revision implemented, `"R.S"` (a tag such as `"v1.6.2"` is read as its R.S); compared with the suite's (section 6 below) |

**Build in this order** (as `conformance/README.md` lists): `check`, `open`, `forward` with the `logits`
report and `close` give the model cases, the refusals and the acceptance cases; `truncate`, `snapshot`,
`restore` and the `kv_rows`, `kv_positions`, `depth` and `recurrent` reports give the state traces;
`run_node` the op cases; `digest`, `dequantize`, `render_chat` and `parse_reply` their own kinds;
`run_node_grad` the gradient cases (engines that train). Reply `skip` to everything not built yet.

**Skip or error.** A method or report not built, or a model or op the engine declines (an architecture it
has no graph for, a kernel's shape precondition, a capability refusal of section 10.3), is a `skip`,
decided before anything is computed. A failure of a path the engine offers, or a refusal after some of the
case's values are computed, is an `error`. In `check`, a capability refusal is a `skip`, and
`{"accepted": false}` is kept for a document the validation rules of section 10.1 refuse (section 11.3,
from v1.6.3). Otherwise a refusal case can pass for the wrong reason (a document refused because
the engine lacks its op, not because of the rule the case breaks), and an acceptance case fails where it
should be skipped.

## 4. A minimal adapter that runs

[`adapters/minimal/minimal_adapter.py`](../adapters/minimal/minimal_adapter.py) is the smallest adapter
that passes something: standard library only, nothing imported from keramos. It answers `hello`, runs the
EXACT ops `add`, `sub` and `mul` through `run_node` (bit for bit, signed zeros included), dequantises F32,
F16 and BF16, and skips everything else. From the repository's root:

    python3 tools/ir_conformance.py --adapter "cmd:python3 adapters/minimal/minimal_adapter.py" --json results.json

At the main branch of 13 November 2026 (601 verdicts from the manifest's 301 entries; under a second):

    summary minimal-example: 6 pass, 0 fail, 595 skip, 0 xfail, 0 xpass; by kind model 0/0/33, trace 0/0/102,
    op 3/0/81, refuse 0/0/129, digest 0/0/96, dequant 3/0/7, grad 0/0/11, accept 0/0/50, chat 0/0/86

Exit status 0. `tests/test_minimal_adapter.py` keeps this true. To grow it into a real adapter, replace
each handler's body with calls into your engine and add handlers in the order above.

`--only SUBSTRING` keeps the manifest entries whose id contains it (`op/attention`, `trace/`, `dequant`);
it does not match sub-case ids (`dequant/blocks/f32` matches nothing). `--list` prints every entry.

## 5. Verdicts

Each case gets one verdict:

| Verdict | When |
|---|---|
| PASS | every compared value is within its class and tolerance (EXACT: the same bits, any NaN matching any NaN; ROUNDING: abs(a - b) <= tol * max(1, max abs(ref)) per tensor), greedy ids and depth traces are identical, kv rows held are among the allowed values, and a refusal or acceptance case is refused or accepted as it requires |
| FAIL | a compared value is outside; the engine replied `error`; the reply was malformed (wrong `id`, not exactly one of `result`, `error` and `skip`, a member of the wrong type, invalid JSON or UTF-8); a result lacks a member the case checks or has the wrong shape; the engine's process ended; a refusal case was accepted or an acceptance case refused; or a call was skipped after an earlier step or prompt of the same case had already mismatched |
| SKIP | the engine replied `skip` to a call of the case before any mismatch was seen. The whole case is skipped, whatever its earlier calls returned. A SKIP is never a pass |
| XFAIL | a FAIL of a case `conformance/expected_fail.json` lists for this engine (by name prefix), with its reason. Not a pass |
| XPASS | a PASS of such a case: the entry should be removed |

A refusal is a result, not an error: `check` gives `{"accepted": false, "message"}`, `digest`
`{"refused"}`, `render_chat` `{"refused"}`, `parse_reply` `{"parsed": false}`. Where the case expects a
refusal it passes, with any message (only the reference must also name the case's rule); elsewhere it
fails. Because a refusal for another reason (a missing tensor, an unsupported op) also passes, the results
file records for each refusal case whether the message cites the case's rule (`"cites": "rule"`, matched as
`rule N`), only its section (`"section"`) or neither, and counts them in `refusal_cites`. No verdict depends
on it. Name the rule in your refusal messages in the reference's form, `IR_SPEC <section> (rule <n>): ...`,
so a reader can see that each refusal was for the case's reason.

The runner exits 1 on any FAIL and 0 otherwise. SKIPs and XFAILs do not change the exit status, so exit
status 0 alone does not mean an engine conforms.

## 6. Versions and capabilities

- **The document's version.** An engine reads `ir_version` 0 and 1 as the same format and refuses any
  other at load, naming it and the versions it reads (sections 2.1 and 2.3).
- **The revision.** Revisions add ops, attributes and values without changing meaning; an older engine
  refuses a document that uses an addition, through the closed-set rules (section 2.2). Run against a
  later revision's suite, such an engine fails the acceptance cases of the additions, so a run reports
  against the revision of its tag only. Declare yours in `hello` as `"spec": "R.S"`. The runner compares
  it with the revision its suite checks (`SUITE_REVISION` in `tools/ir_conformance.py`, kept equal to
  `IR_SPEC.md`'s header by the tests):
  - equal: recorded as `"match": true`;
  - different, or not a revision: a warning at the start and the end of the run, recorded as
    `"match": false`. Verdicts are unchanged, and the run is a report against neither revision: run the
    suite of your revision's tag;
  - absent: a note, recorded as `"match": null`; the run is against the suite's revision.
  Runners older than this guide ignore the member, so declaring it is safe at every tag. Section 11.3
  defines it from v1.6.3; it is optional.
- **Profiles.** Declare the profiles of section 2.4 you implement. The runner runs every case whatever you
  declare: a case that needs a profile you do not implement must `skip` (a capability refusal). The
  results file counts verdicts by the manifest's profile label, which is per manifest entry (an op file, a
  model): a case labelled `core` can use another profile's attributes, and refusal cases carry none.

## 7. Reporting a run

`--json results.json` writes the record of a run:

- `adapter`: the engine's own `hello` reply (for an in-process adapter, the members it sets);
- `spec`: the suite's revision; the checkout's tag (when the commit is tagged), `git describe`, full
  commit, the commit's tree and whether tracked files were modified; the revision the engine declares,
  and `match`;
- `provenance`: sha256 of the runner, the manifest and `expected_fail.json`;
- `counts`, `by_kind`, `by_profile`, `refusal_cites` and every case's verdict and detail.

A report of conformance names: the keramos tag, full commit and tree (and that no tracked file was
modified); the engine, its version and build, the backend and `exact`; the platform; the counts overall
and by kind; every FAIL and XFAIL with its reason; and the SKIPs grouped by reason. For example:
*"my_engine 0.3, CPU, exact: keramos v1.6.2 (commit ..., tree ...), 587 pass, 0 fail, 11 skip (the
gradient cases: no backward)."*

Record the tree as well as the commit. A rewritten history changes commit ids but not the tree of the
same files, so the tree still identifies the suite after a rewrite. This repository's history was
rewritten on 2 and 3 November 2026, and the commits named in receipts written before then are not
reachable from any current ref: match those by their tag and, where recorded, their provenance hashes.
Section 11.5 (from v1.6.3) defines when such a report is a claim of conformance: no FAIL and no XFAIL,
and every SKIP an unclaimed method or a named capability refusal. Otherwise the counts and the reasons
for the skips are a measurement; do not shorten them to "conformant".

`conformance/receipts/` holds the project's records of runs it made or reproduced. An outside engine
publishes its own results file and report; a run with a modified `expected_fail.json` or fixture shows it
in `provenance` and `spec.modified`.
