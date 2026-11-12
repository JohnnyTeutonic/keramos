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

    python -m slip check MODEL.slip                  # compile; report keramos's rules by source line
    python -m slip compile MODEL.slip -o MODEL.ir.json
    python -m slip decompile DOC.ir.json|MODEL.gguf  # any keramos document back to slip
    python -m slip digest FILE                       # the keramos digest of a .slip, .json or .gguf
    python -m slip run MODEL.slip --weights MODEL.gguf [--tokens 1,2,3] [--generate N] [--against OTHER]

`run` executes the model in keramos's reference interpreter (`tools/ir_numpy.py`) on a GGUF's
tensors; `--against` runs a second model on the same weights and compares the logits bit for bit.
`compile` refuses to write a document keramos refuses (`--no-check` writes it anyway).

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
Phi-3 and others).

    python -m pytest tests -q

## Not yet

- Widths. slip checks names, not shapes; a width mismatch shows when the model runs.
- A keramos document whose block has an `out` list of one name has no slip spelling (none exists
  in the corpus); the decompiler says so.
