# slip

slip is a small language for writing language-model graphs. A `.slip` file compiles to a
[keramos](../) IR document, which paperkiln trains and ember serves; slip adds a way to
write one, not a second platform. (Slip is the liquid clay poured into a mould before firing.)

```
block(repeat=2) {
  state kv = kv(heads=2, kv_heads=2, head_dim=8, max_len=16, window=0, sinks=0)

  h = rmsnorm(x, eps=1e-5) @attn_norm
  q, k = rope(linear(h) @attn_q, linear(h) @attn_k, pos,
              heads=2, head_dim=8, theta=10000, rope_heads=all, pairing=interleaved)
  a = attention(q, k, linear(h) @attn_v, state=kv,
                heads=2, kv_heads=2, head_dim=8, scale=0.35355338, causal=true, window=0, sinks=0)
  x += linear(a) @attn_output

  h = rmsnorm(x, eps=1e-5) @ffn_norm
  x += linear(silu(linear(h) @ffn_gate) * linear(h) @ffn_up) @ffn_down
}
```

That is the block of [`examples/llama.slip`](examples/llama.slip). Run on the weights of keramos's
`pk_llama` conformance model, it gives logits bit-identical to the document paperkiln exported.

## Commands

From this folder, with Python 3.10 or later and numpy (slip uses the keramos checkout it sits in, or the one at
`$SLIP_KERAMOS`):

    python -m slip check MODEL [--weights W.gguf] [--shapes]  # keramos's rules and the widths, by source line
    python -m slip compile MODEL.slip -o MODEL.ir.json [--weights W.gguf]
    python -m slip decompile DOC [-o MODEL.slip]              # any keramos document back to slip
    python -m slip diff A B [--tensor-names]                  # the architectural difference
    python -m slip digest FILE                                # the keramos digest
    python -m slip run MODEL --weights W.gguf [--tokens 1,2,3] [--generate N] [--against OTHER]

A model (MODEL, DOC, A, B) is a `.slip`, a keramos `.json`, a GGUF carrying one, or a folder holding
one: a paperkiln run's export folder gives its `.ir.json`, or else the GGUF that carries the document.
So `python -m slip decompile RUN_FOLDER -o arm.slip` turns a run's export into slip for a review, with
the source file and its digest on the first line.

`run` executes the model in keramos's reference interpreter (`tools/ir_numpy.py`) on a GGUF's
tensors; `--against` runs a second model on the same weights and compares the logits bit for bit.
`compile` refuses to write a document with a problem (`--no-check` writes it anyway).

### Widths

`check` solves every value's width and every tensor's shape, and reports a mismatch at the line and
column of the node where two numbers meet, saying where each came from:

    llama.slip:25:7: width: attention writing a: the width of q: 32 (heads 4 x head_dim 8 of rope) against 16 (heads 2 x head_dim 8 of attention)
          a = attention(q, k, linear(h) @attn_v, state=kv,
              ^

Each width is an unknown, and each op says which unknowns are equal, or equal to a number, by its
IR_SPEC section: model.d for the residual stream, model.vocab for the logits, heads x head_dim for
attention and rope, a split's sizes, a linear's [out, in] weight, and, given weights (`--weights`, or the
model's own GGUF), every tensor's shape in the file. Nothing is annotated. Without weights the
attributes fix most shapes (`--shapes` prints them); what is left free is what only a weight can say,
as an FFN's hidden width. Ops whose widths it does not model (`recurrence`, `depth_route`) write values
of unknown width, and an attribute that differs between layers is left out. The check runs only on a
document keramos accepts.

### diff

`diff A B` prints a unified diff of the two models' slip, written in one spelling: no prose or
informative keys (`family_hint`, `tokenizer`, attention's `softmax`, ...), no optional attribute at
its IR_SPEC 3.5 default (linear's `bias=null`, a kv slot's `dtype=f32`), numbers as the shortest
decimal of their f32, and every tensor named `@_` (`--tensor-names` keeps them). The verdict on the
last line is keramos's own rule 38 (`structure` in `tools/ir_validate.py`): the same graph, value and
slot names and add/mul operand order aside, or different graphs (exit status 1). Node order counts:
a graph that computes v after rope is a different document from one that computes it before, even
where the logits agree bit for bit (`examples/llama.slip` against pk_llama's export).

## The language

A file is a keramos document. The top-level keys are written `key value` (`ir_version 1`,
`model {vocab: 15, d: 16, n_layers: 2, ctx: 16}`), and the three sections as blocks:
`embed { ... }`, `block(repeat=N) { ... }` and `head { ... }`. In a section, each statement is a node:

| slip | keramos |
|---|---|
| `h = rmsnorm(x, eps=1e-5)` | `{"op": "rmsnorm", "in": ["x"], "eps": 1e-05, "out": "h"}` |
| `q, k = rope(q, k, pos, ...)` | `"out": ["q", "k"]` |
| `y = a + b`, `a - b`, `a * b` | the ops `add`, `sub`, `mul`, inputs in the order written |
| `x += e` (and `-=`, `*=`) | `x = x + e`: inputs `[x, e]` |
| `linear(silu(g) * u)` | each nested call or operator is its own node, written to a fresh name `t1`, `t2`, ... that the document does not otherwise use, just before the node that reads it |
| `linear(h) @attn_q` | `"weight": "blk.{i}.attn_q.weight"` in the block; `@output` is `"output.weight"` outside it |
| `bias=@attn_qkv.bias` | a path with a dot gets no `.weight`: `"blk.{i}.attn_qkv.bias"` |
| `weight="any.name"` | a tensor name written out |
| `window=per_layer[512, 0]` | `{"per_layer": [512, 0]}` |
| `window=cycle[512, 512, 0]` | the entries repeated to the block's repeat count |
| `state kv = kv(heads=2, ...)` | the block slot `{"name": "kv", "kind": "kv", "heads": 2, ...}` |

Positional arguments are the node's inputs, `name=value` arguments its attributes. Values are
JSON values in a lighter spelling: a bare word is a string (`approx=tanh`, `state=kv`), and `true`,
`false` and `null` are themselves; numbers and quoted strings are JSON's; lists `[...]` and objects
`{key: value}` nest. `#` starts a comment. Newlines end statements, except inside brackets.

slip adds no semantics: every op, attribute and rule is keramos's ([`IR_SPEC.md`](../IR_SPEC.md)), and
`check` reports keramos's refusals at the line and column of the call that made the node.

## Exactness

Any keramos document decompiles to slip that compiles back to the same document: the same
canonical form, so the same digest (IR_SPEC 3.10). The prose members (`_comment`, `_note`) come
back as comments, which the digest ignores. The tests check this on every document in the keramos
checkout: the nine examples and the 33 documents embedded in the conformance suite's models,
paperkiln's and the third-party graphs (Gemma 2 and 3, DeepSeek V2 and V3, Qwen 3 MoE, Mamba,
Phi-3 and others). On the same documents, the width check finds no mismatch with or without each
model's weights, and finds one in every model whose two block weights of different shapes are swapped.
(Outside the tests, the 33 documents under paperkiln's `experiments/` pass both checks as well.)

    python -m pytest tests -q

## Not yet

- Random weights for `run`, from the solved shapes, so a new architecture runs before a GGUF exists. The
  shapes the attributes leave free (an FFN's hidden width) would need a width given on the command line.
- A keramos document whose block has an `out` list of one name has no slip spelling (none exists
  in the corpus); the decompiler says so.
