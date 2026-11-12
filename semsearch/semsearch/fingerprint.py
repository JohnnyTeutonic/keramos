"""Behavioural fingerprints: logits of the reference interpreter on fixed probes and random weights.

A fingerprint needs a correspondence between the weights of two documents. Two are offered:
- by tensor name (slip's `run --random`: each tensor draws from its own stream (seed, name)). Edits that
  keep the names of the tensors they do not touch (config lowering, semsearch's typed edits) keep the
  correspondence, so two documents that compute the same function under it get the same fingerprint even
  when no rewrite of canon.py relates them (an `affine` of scale 0 added to the stream, say).
- by canonical tensor identity (canon.Graph.tensor_ids), which survives renaming but changes with any
  edit upstream of a tensor; it adds nothing to the canonical hash and is kept for the tests.

Two fingerprints match when every logit agrees within `tol` of max(1, max |logit|): ROUNDING-class ops
may reorder sums (IR_SPEC 4.3), so exact equality is not required.
"""

import zlib

import numpy as np

from . import bridge, canon

PROBE_SEED = 20261113


def probes(vocab, ctx, n=2, seed=PROBE_SEED):
    rng = np.random.default_rng(seed)
    return [list(map(int, rng.integers(0, vocab, size=ctx))) for _ in range(n)]


def shape_args(given):
    return [f"{t}=" + "x".join(str(int(x)) for x in s) for t, s in (given or {}).items()]


def random_weights(doc, given=None, seed=0, keyed="name"):
    """Weights for every tensor of the document, initialised as slip's `run --random` does (role by role),
    each instance drawing from its own stream: (seed, instance name) or (seed, canonical id, layer)."""
    if keyed == "name":
        W, _ = bridge.slip_run.random_weights(doc, shape_args(given), seed=seed)
        return W
    W, w = bridge.slip_run.random_weights(doc, shape_args(given), seed=seed)
    g = canon.Graph(canon.normalise(doc), shapes=canon.solved_shapes(doc, given))
    out = {}
    for t in w.names():
        if t in w.tied:
            continue
        tid = g.tensor_ids.get(t, "unread:" + t)
        for inst in w.instances(t):
            layer = inst[len(t.split("{i}")[0]):].split(".")[0] if "{i}" in t else ""
            rng = np.random.default_rng([seed, zlib.crc32((tid + "/" + layer).encode())])
            a = W[inst]
            role = w.roles.get(t, "matrix")
            if role == "gain":
                out[inst] = np.ones_like(a)
            elif role in ("bias", "zero"):
                out[inst] = np.zeros_like(a)
            elif role == "embedding":
                out[inst] = rng.normal(0, 1, a.shape).astype(np.float32)
            else:
                out[inst] = rng.normal(0, 1 / np.sqrt(a.shape[-1]), a.shape).astype(np.float32)
    return out


def fingerprint(doc, given=None, seeds=(0,), n_probes=2, keyed="name"):
    """The logits on the probes, flattened, for each weight seed: one float32 vector."""
    nr = bridge.numpy_ref()
    m = doc["model"]
    ps = probes(m["vocab"], m["ctx"], n_probes)
    parts = []
    for s in seeds:
        W = random_weights(doc, given, seed=s, keyed=keyed)
        for p in ps:
            parts.append(np.asarray(nr.forward(doc, W, p), np.float32).reshape(-1))
    return np.concatenate(parts)


def match(a, b, tol=1e-5):
    if a.shape != b.shape:
        return False
    scale = max(1.0, float(np.max(np.abs(a))))
    return bool(np.max(np.abs(a - b)) <= tol * scale)


class Archive:
    """Fingerprints seen so far; `find` returns the index of a matching one or None."""

    def __init__(self, tol=1e-5):
        self.tol = tol
        self.vecs = []
        self.keys = []

    def find(self, v):
        for i, u in enumerate(self.vecs):
            if match(u, v, self.tol):
                return i
        return None

    def add(self, v, key=None):
        self.vecs.append(v)
        self.keys.append(key)
        return len(self.vecs) - 1
