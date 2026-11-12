"""The cheap proxy: a tiny model trained briefly on a synthetic associative-recall task (CPU).

A sequence of 32 tokens over a vocabulary of 32: eight key-value pairs (keys 0..15, values 16..31, keys
distinct), then the same eight keys in a random order, each followed by its value. The loss and the
accuracy are read at the eight value positions of the second half, which only a model that finds the
earlier occurrence of the key and copies the token after it can predict: the distance back is 2 to 30
tokens, so a short attention window cannot reach most of them (associative recall, as in the MAD and
Zoology synthetic suites, reduced to one fixed format).

Fitness is minus the mean cross-entropy at those positions on a fixed held-out set after a fixed number
of Adam steps from slip's initialisation (seeded), so equal documents with equal seeds get equal
fitness. One evaluation is one training run.
"""

import time

import numpy as np
import torch

from . import torch_exec as te

VOCAB, T, PAIRS = 32, 32, 8


def batch(rng, n):
    toks = np.zeros((n, T), np.int64)
    mask = np.zeros((n, T), bool)
    for b in range(n):
        keys = rng.choice(16, PAIRS, replace=False)
        vals = 16 + rng.integers(0, 16, PAIRS)
        seq = []
        for k, v in zip(keys, vals):
            seq += [k, v]
        order = rng.permutation(PAIRS)
        for j in order:
            seq += [keys[j], vals[j]]
        toks[b] = seq
        # predict value at position 2*PAIRS + 2*m + 1 from position 2*PAIRS + 2*m
        for m in range(PAIRS):
            mask[b, 2 * PAIRS + 2 * m] = True   # the input position whose next-token target is the value
    return toks, mask


def heldout(n=256, seed=12345):
    return batch(np.random.default_rng(seed), n)


def _loss(doc, P, toks, mask):
    x = torch.from_numpy(toks)
    logits = te.forward(doc, P, x[:, :-1])
    tgt = x[:, 1:]
    m = torch.from_numpy(mask[:, :-1])
    lp = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none")
    lp = lp.reshape(tgt.shape)
    loss = (lp * m).sum() / m.sum()
    acc = ((logits.argmax(-1) == tgt) & m).sum().item() / m.sum().item()
    return loss, acc


def evaluate(doc, given, steps=300, seed=0, batch_size=32, lr=3e-3, held=None):
    """Train `steps` Adam steps from slip's initialisation (seed) on fresh batches (seeded), then measure
    on the held-out set. Returns {fitness, loss, acc, params, seconds, steps}."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    t0 = time.time()
    P = te.init_params(doc, given, seed=seed)
    opt = torch.optim.Adam(list(P.values()), lr=lr)
    rng = np.random.default_rng([seed, 7])
    for s in range(steps):
        toks, mask = batch(rng, batch_size)
        opt.zero_grad()
        loss, _ = _loss(doc, P, toks, mask)
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(P.values()), 1.0)
        opt.step()
    held = held or heldout()
    with torch.no_grad():
        loss, acc = _loss(doc, P, *held)
    loss = float(loss)
    if not np.isfinite(loss):
        loss = 10.0
    return {"fitness": -loss, "loss": loss, "acc": acc, "params": te.n_params(P),
            "seconds": round(time.time() - t0, 2), "steps": steps}
