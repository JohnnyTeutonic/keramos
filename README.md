# keramos

keramos is the specification of the keramos IR: a versioned JSON graph of a language model that a
trainer exports and an inference engine executes. The engine interprets the graph, so it runs a new
architecture without code written for that architecture. The document travels inside the model's
GGUF file (the string key `paperkiln.ir`) beside the weights.

Every op states its numerics, not only its shape. An op is either **EXACT** (every engine and
backend gives the same f32 bits, because the order of every operation and every rounding is fixed)
or **ROUNDING** (the result is within a stated tolerance of the reference). A conformance suite
checks an engine against those statements, op by op, model by model and session by session.

Two engines implement it: [paperkiln](https://github.com/JohnnyTeutonic/paperkiln), the C++/CUDA
trainer that exports documents, and [ember.cpp](https://github.com/JohnnyTeutonic/ember.cpp), the
inference engine (CPU, CUDA and WebAssembly). Execution still requires the engine's implemented
operations and profiles; a specification-level feature is not proof of support on every backend.

## Contents

[Defect registry](DEFECT_REGISTRY.md): execution invariants, reproductions, upstream
issues and verified resolution status, maintained as the reports progress.

| Path | What it is |
|---|---|
| `IR_SPEC.md` | the normative specification: the document, numerics and classes, ops, state, recursion and chat profiles, refusal rules, the conformance adapter contract |
| `ir.schema.json` | the JSON Schema (2020-12) of a document's structure |
| `tools/ir_validate.py` | the validator: the schema and every refusal rule; the digest |
| `tools/ir_numpy.py` | the reference interpreter (numpy), with `ir_expr.py` (the `expr` op), `ir_recurrence.py` (the `recurrence` op) and `ir_quant.py` (stored tensor types) |
| `tools/ir_chat.py` | renders chat templates (the chat profile, IR_SPEC section 8.1.3) |
| `tools/ir_check.py` | checks a GGUF's embedded document and its container |
| `tools/gguf_ir.py` | builds the document of a third-party GGUF (Qwen, Gemma 3, GPT-2, Pythia, Mistral, Mamba-1) from its llama.cpp metadata |
| `tools/ir_conformance.py` | the conformance runner (and, with `ir_conformance_gen.py`, the suite's generator) |
| `conformance/` | the suite, its manifest, expected failures per engine, and engines' receipts |
| `examples/` | worked documents |
| `tests/` | the tests of the tools and the suite |
| `IR_PRIOR_ART.md` | the IR against ONNX, StableHLO, TOSA, GGUF and others |

## Requirements

Python 3.10 or later. The validator is standard library only. The reference interpreter, the runner
and the tests need numpy; `pip install -r requirements.txt` adds pytest and the optional packages
(gguf-py for reading quantised GGUFs and for the dequantisation test, scipy for `erf`, jsonschema
for a second evaluation of the schema). `requirements-reference.txt` names the environment that generated the conformance
suite's expected values, the only one to regenerate them in (IR_SPEC section 4.6).

## Quick start

From a checkout, with Python 3.10 or later:

```sh
python -m pip install -r requirements.txt
python tools/ir_validate.py examples/gpt2_tiny.ir.json
python tools/ir_conformance.py --adapter ir_numpy
python -m pytest tests -q
```

The bundled models are tiny conformance fixtures, not ready-made chat models. Validation checks
the document's contract; running the reference suite checks the reference implementation. Test
counts depend on optional dependencies: the 31 October tooling check passed 252 tests with
2 optional skips. Both NumPy and the independent Rust implementation retained 587 conformance
passes and 11 gradient skips after stricter nonfinite-value and required-state checks.
[Source-bound tooling evidence](conformance/receipts/tooling_guardrails_20261031/README.md).
The specification and historical expected values are unchanged; the original release environment
passed 206 Python tests with 10 optional skips.

## Validating a document

    python tools/ir_validate.py model.ir.json            # a document
    python tools/ir_validate.py model.gguf               # a GGUF with an embedded document
    python tools/ir_validate.py digest model.gguf        # the document's digest (IR_SPEC 3.10)
    python tools/ir_validate.py --corpus                 # every stored document of this repository
    python tools/ir_validate.py --corpus --corpus-glob "my_models/**/*.gguf"

Each refusal names its section and the rule of IR_SPEC section 10.1 it breaks (`IR_SPEC 3.7 (rule 25): ...`); the exit status is 1 on
any refusal. From Python, `ir_validate.file_errors(path)` and `ir_validate.doc_errors(doc)` return
the messages.

## Running the conformance suite

    python tools/ir_conformance.py --adapter ir_numpy                       # the reference itself
    python tools/ir_conformance.py --adapter cmd:"my_engine --conformance"  # your engine
    python tools/ir_conformance.py --adapter cmd:"..." --only trace/ --json results.json
    python tools/ir_conformance.py --list

An engine joins through an adapter: a program that reads one JSON request per line on stdin and
writes one JSON reply per line on stdout (IR_SPEC section 11.3). `conformance/README.md` lists the
methods in the order worth building them; a method not built yet replies `{"id": n, "skip": "..."}`,
and the case is reported SKIP, never PASS. A case passes when EXACT values are bit-identical and
ROUNDING values are within tolerance (|a - b| <= tol * max(1, max|ref|)). The runner exits 1 on
any FAIL. `conformance/expected_fail.json` lists, per engine, the cases it is known to fail and why.

A Python engine can instead subclass `ir_conformance.Adapter` and call
`ir_conformance.main(argv, adapters={"name": factory})`.

## Independent Rust implementation

The clean-room Rust interpreter in `cleanroom/rust/` implements revision 1.6 from the specification.
Its 30 October receipt passes **587 cases**, with **zero failures or expected failures**; the remaining
11 cases require gradients, which this inference engine does not implement. The source/read log and
findings remain under `cleanroom/`. The current full receipt is
[`conformance/receipts/cleanroom.json`](conformance/receipts/cleanroom.json); `cleanroom/receipt.json`
is the preserved historical first receipt, not the current result. This closes the revision catch-up;
it does not claim hardware/backend conformance for another engine.

To build and run the Rust adapter on Linux/WSL, with Cargo installed (build outputs stay outside
the checkout):

```sh
cargo build --release --manifest-path cleanroom/rust/Cargo.toml --target-dir /tmp/keramos-rust-build
python tools/ir_conformance.py --adapter "cmd:/tmp/keramos-rust-build/release/kcr --conformance" --json rust-results.json
```

This is a specification-isolated implementation with a recorded read log, rather than evidence
of adoption by an unrelated engine team. Its CPU result does not establish CUDA or browser
conformance.

## Versions

A document carries `ir_version` (now 1). The specification has revisions within a version (1.0,
1.1, 1.2), which add without changing meaning (IR_SPEC 2.2). Each revision is a tag `vR.S.P` of this
repository: `v1.1.0` is revision 1.1, and P counts later commits that change no normative statement,
no part of the schema and no expected value (wording, paths, tool interfaces, tests, receipts;
IR_SPEC 2.6). An engine pins a tag, records the tag and its full commit, and names both in its
receipt in `conformance/receipts/`.

The **v1.6.2 tooling patch (30 October 2026)** closes the independent Rust implementation's
revision-1.6 catch-up and validates command-adapter reply envelopes. It changes no normative
statement, schema or numerical expected value. The current Rust receipt passes all 587 supported
cases with 11 gradient-only skips; the tool tests pass 206 with 10 optional skips. CUDA/backend
conformance of other engines remains separate from this specification release.

## Tests

    python -m pytest tests -q

`EXPR_EVAL_BIN` set to an engine's `expr` evaluator adds the tests that compare it with the
reference on random inputs.

## Regenerating the suite

    python tools/ir_conformance.py --regenerate

Expected values come from the reference only, deterministically; change them only by regenerating,
in the commit that changes the reference or the specification. The `pk_` models in
`conformance/models/` are trained by paperkiln, which supplies them.

## License

Original Keramos project material is available under the [MIT License](LICENSE).
Third-party components retain their own terms; see [Third-party notices](THIRD_PARTY_NOTICES.md),
including the pinned llama.cpp/ggml reference used for GGUF dequantization layouts and numerics.
