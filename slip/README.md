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
    python -m slip run MODEL --random [--seed S] [--shape @ffn_up=64 ...] [--save OUT.gguf]  # no weights needed

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
IR_SPEC section: model.d for the residual stream, model.vocab for the logits and the embedding's rows,
model.ctx for a position table's rows, heads x head_dim for attention and rope, a split's sizes, a
linear's [out, in] weight, a recurrence's declarations (an input's width is the product of its index
sizes, as is the readout's; a weight holds as many elements as its declaration), depth_route's router
[n_rec, d], the widths of a list's tensors (attention residuals), and, given weights (`--weights`, or the
model's own GGUF), every tensor's shape in the file. Every op of IR_SPEC sections 5 to 7 is modelled,
and nothing is annotated. Without weights the graph fixes most shapes (`--shapes` prints them); what is
left free is what only a weight can say: an FFN's hidden width, a low-rank projection's rank, the number
of state tokens. An attribute whose value differs between layers fixes no width (one block stands for
every application). The check runs only on a document keramos accepts.

### Prototyping without weights

`run --random` runs a model that has no weights anywhere, on random weights of the solved shapes. The
sizes the graph leaves free are named once each, and given with `--shape`:

    $ python -m slip run examples/retention.slip --random
    random weights need every tensor's shape; still free:
      blk.{i}.ffn_down.weight [32, ?]
      blk.{i}.ffn_gate.weight [?, 32]
      blk.{i}.ffn_up.weight [?, 32]
    give the free sizes: --shape @ffn_gate=N1

    $ python -m slip run examples/retention.slip --random --shape @ffn_up=64 --generate 4
    random weights, seed 0: 33 tensors, 33,004 parameters
    tokens  [1, 2, 3, 4, 5, 6, 7, 8]
    logits  [8, 32]; argmax per position [13, 8, 30, 27, 16, 29, 30, 4]
    greedy  [4, 27, 11, 1]

[`examples/retention.slip`](examples/retention.slip) is a token mixer no GGUF holds (a decayed matrix
state per head, written as a keramos `recurrence`); it runs in keramos's reference interpreter straight
from its source. `--shape NAME=AxB` gives a whole shape, `NAME=H` a tensor's free sizes in order; NAME is
the document's tensor name or slip's `@path`. Where the spec allows several sizes and nothing chose one
(relative_kv's width, position_bias's bucket count) the least is taken. Matrices are N(0, 1/in), gains 1,
biases and attention-residual pseudo-queries 0, a state-space transition 0.5 I plus noise, selective_scan's
A negative; each tensor draws from its own stream (seed, name). A linear tied to another tensor reads it,
as it would in a file without the weight.

`--save OUT.gguf` writes the model and its random weights with the conformance suite's own writer
(`tools/ir_conformance.py`): architecture `paperkiln`, F32 tensors, the document embedded, a placeholder
word tokenizer (with the chat markers where the model has per-turn state tokens). `run` refuses to
report success unless keramos's file checks (`ir_check`) accept the file, so ember and paperkiln can load
the prototype as it stands. The tests run every conformance model this way: given only its free sizes,
the solved shapes are its file's, tensor for tensor; the logits are finite; and the saved GGUF passes
keramos's checks and reads back to bit-identical logits.

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

- A keramos document whose block has an `out` list of one name has no slip spelling (none exists
  in the corpus); the decompiler says so.
