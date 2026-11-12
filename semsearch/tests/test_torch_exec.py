"""The trainable interpreter computes what keramos's reference interpreter computes."""

import random

import numpy as np
import pytest
import torch

from semsearch import bridge, config_space as cs, fingerprint as fp, mutate as mu, torch_exec as te

K = bridge.KERAMOS


def _close(doc, P):
    W = {k: v.detach().numpy() for k, v in P.items()}
    ids = fp.probes(doc["model"]["vocab"], min(doc["model"]["ctx"], 32), 1)[0]
    a = np.asarray(bridge.numpy_ref().forward(doc, W, ids))
    b = te.forward(doc, P, torch.tensor([ids])).detach().numpy()[0]
    return np.max(np.abs(a - b)) / max(1, np.max(np.abs(a)))


def test_random_configs():
    rng = random.Random(0)
    for _ in range(20):
        doc, given = cs.lower(cs.random_config(rng, cs.SEARCH_SPACE))
        assert _close(doc, te.init_params(doc, given, seed=1)) < 1e-5


def test_edited_documents():
    rng = random.Random(1)
    doc, given = cs.lower(cs.default_config())
    n = 0
    for _ in range(80):
        r = mu.apply(doc, given, rng.choice(mu.EDIT_TYPES), rng)
        if not r:
            continue
        doc, given = r[0], r[1]
        assert _close(doc, te.init_params(doc, given, seed=2)) < 1e-4, r[2]
        n += 1
    assert n > 40


@pytest.mark.parametrize("model", ["pk_llama", "pk_gpt2", "pk_gpt2_swa", "tp_mistral", "tp_qwen3", "tp_gemma3",
                                   "tp_gptneox", "pk_recursive", "pk_flex_highway"])
def test_conformance_models(model):
    path = next(K.glob(f"conformance/models/{model}/*.gguf"))
    doc = bridge.load_document(path)
    assert te.supported(doc)
    W = bridge.slip_run.load_weights(path)
    P = {k: torch.tensor(v) for k, v in W.items()}
    assert _close(doc, P) < 1e-5


def test_gradients_reach_every_parameter():
    doc, given = cs.lower(dict(cs.default_config(), tie_embeddings=False, qk_norm=True))
    P = te.init_params(doc, given)
    te.forward(doc, P, torch.randint(0, 32, (2, 16))).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in P.values())
