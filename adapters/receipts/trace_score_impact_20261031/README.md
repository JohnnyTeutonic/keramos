# Cached-score trace receipt impact audit — 31 October 2026

The fix in `7a9264879095de3bc5a3d185ced7a47d5bb438f4` produces **no observed
verdict movement** in the affected published trainer receipt. A source-pinned
reconstruction reproduces all 524 historical case verdicts under both the original
v1.4.2 runner and the fixed runner: **158 PASS, 0 FAIL, 366 SKIP**. All six
cached-score trace passes survive exact shape validation.

This is a reconstruction, not an exact replay of the historical executable or raw
arrays. The original receipt records only short executable hash prefixes and
scaled error details. Its `/home/jonat/e7/pk_build` tree no longer exists; none of
the extant executables in the bounded home-directory inventory matches the
recorded `eeba7b5b` prefix. The historical adapter writes its score response in a
TemporaryDirectory, and the committed receipt/evidence paths contain no raw score
arrays. Newly captured full vectors cannot establish what shape the original
executable returned in October's original run. The original receipts are untouched.

## Exact affected path and inventory

`run_trace` first attempts the session interface. The changed code is reached only
after that interface raises NotImplementedError, when the fixture supplies
`score_form` and does not supply a replacement `ir`. It calls the adapter's native
cached scorer; the old subtraction allowed NumPy broadcasting, whereas the fixed
runner uses `_rounding_error` to check shape and nonfinite values first.

The manifest at `7a92648` contains exactly these six eligible cases. The recorded
historical score-fallback path occurs only in `conformance/receipts/mtstudio.json`
among the 19 result-bearing receipts inventoried in Keramos's conformance/adapter
receipts and the paper's analysis directory. Other engines' instances of these
same case IDs use sessions or skip; they are not executions of the changed path.
`impact_v2.json` preserves each receipt's hash and relevant rows.

| Case | Captured expected/actual shape | Historical / parent / fixed verdict |
| --- | --- | --- |
| trace/pk_llama_decode | (8,) / (8,) | PASS / PASS / PASS |
| trace/pk_moe_decode | (8,) / (8,) | PASS / PASS / PASS |
| trace/pk_mor_decode | (8,) / (8,) | PASS / PASS / PASS |
| trace/pk_recursive_decode | (8,) / (8,) | PASS / PASS / PASS |
| trace/pk_state_evict_on | (5,) / (5,) | PASS / PASS / PASS |
| trace/pk_state_evict_off | (5,) / (5,) | PASS / PASS / PASS |

The captured native cached-score responses, including greedy IDs, depth and KV-row
fields where present, are identical across all three runner controls. All six
rounded error detail strings also match the historical receipt.

## Controls and evidence

The trainer and adapter are unmodified Git archives of paperkiln
`968b49261281ec01379265a8fb1d1c48da8bbd25`. The fixtures are the original receipt's
Keramos v1.4.2 commit `6bbfe01cc7459abc2bd47c42e1f0376438705e6b`.
The historical runner comes from that same archive; the immediate-parent control
uses `f7cc616a76807ed22b71a04e2fa5d07a82c10011`; the fixed runner uses `7a92648`.
Each runner executes its own Python helper modules, while all runs consume the
same original fixture/model files and rebuilt native binaries.

The Release build uses GCC 13.3 and `-ffp-contract=off`, matching the historical
receipt's stated settings, with one budgeted CPU core. A 1.5 GiB address-space cap
was applied to the active build process tree after the early small translation
units; subsequent compiler processes inherited it. The full command/source
identities and exact limit timing are in `provenance.json`; build/configure logs
are retained. No training, GPU, network model calls or paid experiments ran.

| Control | Cases | PASS | FAIL | SKIP | Historical verdict movements |
| --- | ---: | ---: | ---: | ---: | ---: |
| historical_runner | 524 | 158 | 0 | 366 | 0 |
| parent_runner, trace/pk_ filter | 59 | 6 | 0 | 53 | 0 among replayed IDs |
| fixed_runner | 524 | 158 | 0 | 366 | 0 |

Each `*.raw.json` records all native score return arrays, exact fixture-file hashes,
runner/helper/adapter hashes and complete binary SHA-256s. The replay script loads
the selected runner explicitly before loading the historical adapter and binds its
suite to the pinned historical fixtures. The only adapter wrapper captures native
score returns without modifying them. Reports are exclusive-create; rerunning
requires a fresh output directory. `replay.sh` records the replay commands and
`summarize.py` creates the inventory/comparison; the final expanded summary is
`impact_v2.json`.

One unrelated numerical detail differs from the archived receipt:
`model/pk_recurrence` reports worst error `1.34e-07` rather than `1.26e-07` under
**both** the historical and fixed runners. Its verdict remains PASS, and this is
not movement caused by the shape fix. No historical numeric distribution or receipt
has been regenerated from this reconstruction.

## Paper consequence

There is no observed reason to change C4's 158/0/366 counts, the six trainer trace
passes, or the trainer's eleven gradient passes. The paper's E2 replay previously
omitted the trainer; this audit supplies a source-pinned reconstruction under the
fixed runner, with the exact historical-artifact limitation above. It should not
be described as recovery or validation of the missing original raw score arrays.
