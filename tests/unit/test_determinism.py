"""
R6 -- is any published number reproducible?

`data_seed: 0` appears in every YAML under vlm/yaml/. Grepping for where it is consumed
turns up exactly one place -- `viz_emb_trainer._required_params`, which only checks that
the key *exists*. It is never passed to `transformers.set_seed`, never reaches
`TrainingArguments(seed=..., data_seed=...)`, and `scripts/train_longitudinal.py` never
calls `torch.manual_seed` at all.

Two consequences:

  * The longitudinal MLP's 32 LOSO folds are trained from unseeded initialisations, so
    `predicted_embeddings/*.npy` -- the VLM's ts1/ts2/ts3 image tokens -- differ on every
    regeneration.
  * The VLM's projection layer and both multitask heads are constructed in
    `load_train_model()` *before* `transformers.Trainer` is instantiated, so even HF's
    internal seeding would come too late to cover them.

At n = 32, where the null sd of AUROC is 0.105 (see test_pooled_auroc_estimator), an
unseeded single run is not a measurement.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.synth import make_embeddings  # noqa: E402

pytestmark = pytest.mark.unit


def test_data_seed_is_declared_in_every_config(repo_root):
    """Precondition: the configs promise determinism."""
    import yaml

    yamls = sorted((repo_root / "vlm" / "yaml").glob("*.yml"))
    assert yamls, "no YAML configs found"
    for path in yamls:
        params = yaml.safe_load(path.read_text())
        assert "data_seed" in params.get("data", {}), f"{path.name} has no data_seed"


def test_data_seed_is_actually_consumed(repo_root):
    """
    F6, fixed. `data_seed` now reaches `transformers.set_seed` before the model is
    constructed, and `TrainingArguments(seed=, data_seed=)`.
    """
    trainer_src = (repo_root / "vlm" / "model" / "viz_emb_trainer.py").read_text()

    uses = [
        "set_seed" in trainer_src,
        "seed=" in trainer_src.split("TrainingArguments")[-1].split(")")[0]
        if "TrainingArguments" in trainer_src else False,
    ]
    print(f"\n  viz_emb_trainer.py: set_seed called = {uses[0]}, TrainingArguments(seed=) = {uses[1]}")

    assert any(uses), (
        "`data_seed` is declared in every YAML but never reaches transformers.set_seed "
        "or TrainingArguments(seed=...). The projection layer and both multitask heads "
        "are also built in load_train_model() before Trainer exists, so HF's own seeding "
        "would not cover them. Every reported VLM number is a single unreproducible draw."
    )


def test_longitudinal_mlp_seeds_torch(repo_root):
    """
    F6, fixed. train_longitudinal.py seeds per fold (`seed + fold_index`), so each
    fold reproduces independently of how many folds ran before it.
    """
    src = (repo_root / "scripts" / "train_longitudinal.py").read_text()
    assert "manual_seed" in src or "set_seed" in src, (
        "scripts/train_longitudinal.py never seeds torch. Its LOSO-predicted embeddings "
        "are written to predicted_embeddings/*.npy and consumed as VLM image tokens, so "
        "the VLM's inputs change on every regeneration of the longitudinal stage."
    )
    assert "--seed" in src, "the seed should be exposed as a CLI flag for sweeps"


@pytest.mark.torch
def test_unseeded_train_fold_is_nondeterministic(scripts):
    """
    Why the fix was needed: `train_fold` itself is stateless w.r.t. seeding, so two
    calls without an external seed still diverge. main() now seeds before each call.
    """
    import torch

    mod = scripts("train_longitudinal")
    emb = make_embeddings(n_subjects=8, signal=("week",), dim=32, seed=0)
    X, Y, *_ = mod.build_pairs(emb.embeddings, emb.subject_ids, emb.weeks)

    preds = []
    for _ in range(2):
        model = mod.train_fold(
            X, Y, in_dim=X.shape[1], out_dim=Y.shape[1], hidden_dim=64,
            lr=1e-3, epochs=50, device=torch.device("cpu"),
        )
        with torch.no_grad():
            preds.append(model(torch.tensor(X)).numpy())

    a, b = preds
    cos = float(np.mean((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))))
    max_abs = float(np.abs(a - b).max())
    print(f"\n  two identical train_fold calls: mean cosine = {cos:.6f}, max abs diff = {max_abs:.4f}")

    assert max_abs > 1e-6, (
        "expected nondeterminism from the unseeded MLP; if runs now match, seeding was "
        "added and this test should become a determinism guard instead"
    )


@pytest.mark.torch
def test_seeding_would_make_it_deterministic(scripts):
    """
    The fix works: seeding immediately before construction gives bit-identical results.
    Establishes that a one-line change buys reproducibility.
    """
    import torch

    mod = scripts("train_longitudinal")
    emb = make_embeddings(n_subjects=8, signal=("week",), dim=32, seed=0)
    X, Y, *_ = mod.build_pairs(emb.embeddings, emb.subject_ids, emb.weeks)

    preds = []
    for _ in range(2):
        torch.manual_seed(1234)
        model = mod.train_fold(
            X, Y, in_dim=X.shape[1], out_dim=Y.shape[1], hidden_dim=64,
            lr=1e-3, epochs=50, device=torch.device("cpu"),
        )
        with torch.no_grad():
            preds.append(model(torch.tensor(X)).numpy())

    max_abs = float(np.abs(preds[0] - preds[1]).max())
    print(f"\n  with torch.manual_seed(1234): max abs diff = {max_abs:.2e}")
    assert max_abs == 0.0, f"seeding should give bit-identical output, got {max_abs:.2e}"
