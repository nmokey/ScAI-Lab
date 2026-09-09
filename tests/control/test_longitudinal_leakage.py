"""
R2 -- does the genotype label leak into the longitudinal MLP's predicted embeddings?

The mechanism
-------------
`scripts/train_longitudinal.py::build_pairs` concatenates a 2-d one-hot *genotype*
vector into the MLP's input (positions D..D+2 of the conditioning block, alongside
cohort and step). The MLP's outputs are written to
`predicted_embeddings/<sid>_ts{1,2,3}.npy`, and `vlm/data/vqa_dataset.py::_load_embedding`
feeds them to the VLM as image tokens 1-3.

So the VLM's genotype task consumes inputs computed *from the genotype label*. LOSO on
the MLP does not help: the held-out subject's own genotype is an input feature when its
predictions are generated. `scripts/evaluate_encoder_vs_vlm.py` already warns about this
for linear-probe condition B, and docs/results.md calls B "an inflated ceiling" -- but
the same inputs feed the VLM, where they underwrite the README's central claim.

What the probes below actually found
------------------------------------
The mechanism is real but its magnitude is *not* automatic, and the first version of
this file was wrong to assume it was. Two findings:

 1. When genotype carries no information about the target embedding, the MLP simply
    learns to ignore the bit: flipping it moves the predicted embedding by a cosine of
    ~1e-4. An unused input does not leak.

 2. The leak's size is governed by the ratio of the conditioning vector's magnitude
    (a unit one-hot) to the embedding's magnitude. At ||emb|| ~ 28 (isotropic unit-noise
    in 768-d) the bit is negligible. Shrink the embedding 50x and cos(pred, pred_flipped)
    falls to 0.92 -- a substantial dependence.

So the question "is the headline VLM genotype result inflated by this leak?" reduces to
one empirical quantity nobody has measured: the norm of the real RAD-DINO embeddings
relative to the unit conditioning vector. `test_genotype_bit_influence_realdata` is that
measurement. Until it runs, the leak is neither confirmed nor excluded -- and the cheap
insurance is to re-run `train_longitudinal.py --no-conditioning` and compare.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.synth import loso_auroc, make_embeddings, permutation_null  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.torch]

# Real defaults from train_longitudinal.parse_args.
EPOCHS, HIDDEN, LR = 300, 512, 1e-3
FUTURE_TAGS = ("Week 15", "Week 18", "Week 20")

# Layout of the conditioning block appended by build_pairs:
#   [ embedding (D) | genotype (2) | cohort (2) | step (3) ]
GENO_SLICE = slice(0, 2)  # relative to the start of the conditioning block


def _fit_mlp(mod, X, Y, seed=0):
    import torch

    torch.manual_seed(seed)  # train_longitudinal does not seed itself -- see R6
    return mod.train_fold(
        X, Y, in_dim=X.shape[1], out_dim=Y.shape[1],
        hidden_dim=HIDDEN, lr=LR, epochs=EPOCHS, device=torch.device("cpu"),
    )


def _predict(model, X):
    import torch

    with torch.no_grad():
        return model(torch.tensor(X)).numpy()


def _loso_predict(mod, emb, use_conditioning, seed=0):
    """
    Mirror of train_longitudinal.main()'s LOSO loop. main() is not factored into a
    callable, so the loop is replicated -- but build_pairs and train_fold, the parts
    under test, are the real ones.
    """
    from sklearn.model_selection import LeaveOneGroupOut

    X, Y, subjects, _g, meta = mod.build_pairs(
        emb.embeddings, emb.subject_ids, emb.weeks, use_conditioning=use_conditioning
    )
    Y_pred = np.zeros_like(Y)
    for tr, te in LeaveOneGroupOut().split(X, Y, subjects):
        Y_pred[te] = _predict(_fit_mlp(mod, X[tr], Y[tr], seed=seed), X[te])
    return X, Y_pred, meta


def _four_token_features(emb, Y_pred, meta):
    """
    The VLM's actual image-token input: observed Week 12 embedding concatenated with
    the three MLP-predicted future embeddings, zero-padded where unavailable. Mirrors
    vqa_dataset._load_embedding and evaluate_encoder_vs_vlm.build_predicted_4token_features.
    """
    dim = emb.embeddings.shape[1]
    observed, predicted = {}, {}
    for i in range(len(emb)):
        observed.setdefault(str(emb.subject_ids[i]), {})[str(emb.weeks[i])] = emb.embeddings[i]
    for i, m in enumerate(meta):
        if m["week_to"] in FUTURE_TAGS:
            predicted.setdefault(m["sid"], {})[m["week_to"]] = Y_pred[i]

    X, y, sids = [], [], []
    for sid in sorted(observed):
        if "Week 12" not in observed[sid]:
            continue
        tokens = [observed[sid]["Week 12"]]
        tokens += [predicted.get(sid, {}).get(w, np.zeros(dim, np.float32)) for w in FUTURE_TAGS]
        X.append(np.concatenate(tokens))
        y.append(1 if "_KO_" in sid else 0)
        sids.append(sid)
    return np.array(X), np.array(y), np.array(sids)


def _flip_genotype(X, dim):
    """Return a copy of the MLP input with the genotype one-hot swapped."""
    Xf = X.copy()
    block = Xf[:, dim:]
    block[:, GENO_SLICE] = block[:, GENO_SLICE][:, ::-1]
    Xf[:, dim:] = block
    return Xf


def _flip_sensitivity(mod, emb, seed=0):
    """
    Counterfactual: train on the real inputs, then re-predict with the genotype bit
    flipped. cos == 1.0 means the bit is ignored (no leak); lower means the predicted
    embedding encodes the label.
    """
    dim = emb.embeddings.shape[1]
    X, Y, _s, _g, _m = mod.build_pairs(
        emb.embeddings, emb.subject_ids, emb.weeks, use_conditioning=True
    )
    model = _fit_mlp(mod, X, Y, seed=seed)
    p0, p1 = _predict(model, X), _predict(model, _flip_genotype(X, dim))
    cos = (p0 * p1).sum(1) / (np.linalg.norm(p0, axis=1) * np.linalg.norm(p1, axis=1))
    return float(cos.mean())


# ---------------------------------------------------------------------------
# Structural: the mechanism exists and is where we think it is
# ---------------------------------------------------------------------------

def test_genotype_is_an_input_feature_to_the_longitudinal_mlp(scripts):
    """
    Contract test for the layout the rest of this file depends on: build_pairs really
    does append [genotype(2) | cohort(2) | step(3)], with KO encoded as [1, 0].
    """
    mod = scripts("train_longitudinal")
    emb = make_embeddings(n_subjects=4, signal=(), dim=16, seed=0)
    X, Y, subjects, genotypes, meta = mod.build_pairs(
        emb.embeddings, emb.subject_ids, emb.weeks, use_conditioning=True
    )
    assert X.shape[1] == 16 + 7, "conditioning block should be 7-d (geno 2 + cohort 2 + step 3)"

    for row, sid, scalar in zip(X, subjects, genotypes):
        onehot = row[16:18]
        expected = [1.0, 0.0] if "_KO_" in sid else [0.0, 1.0]
        assert list(onehot) == expected, f"genotype one-hot wrong for {sid}"
        assert scalar == (1 if "_KO_" in sid else 0)

    # And without conditioning the label is absent entirely.
    X_off, *_ = mod.build_pairs(
        emb.embeddings, emb.subject_ids, emb.weeks, use_conditioning=False
    )
    assert X_off.shape[1] == 16


# ---------------------------------------------------------------------------
# Magnitude: does the bit actually influence the output?
# ---------------------------------------------------------------------------

def test_unused_genotype_bit_is_ignored(scripts):
    """
    When genotype tells the MLP nothing about the target, it learns to ignore the bit.
    This is why the leak is not automatic, and it bounds the concern: the conditioning
    vector only leaks to the extent the model finds it useful.
    """
    mod = scripts("train_longitudinal")
    emb = make_embeddings(n_subjects=32, signal=("week",), strength=3.0, seed=0)
    cos = _flip_sensitivity(mod, emb)
    print(f"\n  [genotype uninformative] cos(pred, pred_flipped) = {cos:.6f}")
    assert cos > 0.999, (
        f"MLP output depends on the genotype bit (cos={cos:.6f}) even though genotype "
        f"is uninformative here -- unexpected, investigate before trusting the sweep below."
    )


def test_genotype_bit_influence_scales_with_embedding_norm(scripts):
    """
    The quantity that decides whether R2 matters. The conditioning vector is a unit
    one-hot; its influence relative to the embedding is set by ||embedding||.

    Reported, not asserted at a threshold -- the real RAD-DINO norm is the unknown,
    and `test_genotype_bit_influence_realdata` is where it gets measured.
    """
    mod = scripts("train_longitudinal")
    base = make_embeddings(n_subjects=32, signal=("week", "genotype"), strength=3.0, seed=0)

    print("\n  ||emb||   cos(pred, pred_genotype_flipped)   [1.0 = bit ignored]")
    results = []
    for scale in (1.0, 0.1, 0.02):
        scaled = make_embeddings(n_subjects=32, signal=("week", "genotype"), strength=3.0, seed=0)
        scaled.embeddings = (base.embeddings * scale).astype(np.float32)
        norm = float(np.linalg.norm(scaled.embeddings, axis=1).mean())
        cos = _flip_sensitivity(mod, scaled)
        results.append((norm, cos))
        print(f"  {norm:7.2f}   {cos:.6f}")

    # Smaller embeddings -> relatively larger conditioning vector -> more influence.
    assert results[-1][1] < results[0][1], (
        "genotype-bit influence should grow as the embedding norm shrinks; if it does "
        "not, this sweep is not measuring what it claims"
    )


@pytest.mark.probe
def test_conditioning_does_not_amplify_genotype_recovery(scripts):
    """
    End-to-end paired comparison on the VLM's actual 4-token input: is genotype more
    recoverable when the MLP was conditioned on it than when it was not?

    Marked `probe` because it is the direct analogue of the reported result. On the
    synthetic fixture the two arms should agree; a gap on real data would mean the
    extra genotype accuracy attributed to longitudinal tokens is the label coming back
    out of the conditioning vector.
    """
    mod = scripts("train_longitudinal")
    emb = make_embeddings(n_subjects=32, signal=("week", "genotype"), strength=3.0, seed=0)

    aucs = {}
    for label, cond in (("ON", True), ("OFF", False)):
        _X, Y_pred, meta = _loso_predict(mod, emb, use_conditioning=cond)
        Xf, y, sids = _four_token_features(emb, Y_pred, meta)
        aucs[label] = loso_auroc(Xf, y, sids)

    Xraw, yraw, sraw = emb.subject_level("Week 12")
    baseline = loso_auroc(Xraw, yraw, sraw)
    gap = aucs["ON"] - aucs["OFF"]
    print(
        f"\n  genotype AUROC -- raw ts0: {baseline:.3f} | 4-token cond OFF: {aucs['OFF']:.3f} "
        f"| cond ON: {aucs['ON']:.3f} | gap: {gap:+.3f}"
    )

    assert gap < 0.10, (
        f"Conditioning the longitudinal MLP on genotype raises downstream genotype "
        f"recoverability by {gap:+.3f} AUROC ({aucs['OFF']:.3f} -> {aucs['ON']:.3f}). "
        f"That increment is the label leaking through the predicted image tokens, not "
        f"imaging signal. Re-run train_longitudinal.py --no-conditioning and re-evaluate."
    )


# ---------------------------------------------------------------------------
# The measurement that settles it
# ---------------------------------------------------------------------------

@pytest.mark.realdata
def test_genotype_bit_influence_realdata(data_root, scripts):
    """
    Run the counterfactual flip on the real RAD-DINO embeddings. This is the number
    that decides whether R2 affects the published result.
    """
    mod = scripts("train_longitudinal")
    npz = data_root / "embeddings" / "raddino" / "raddino_embeddings.npz"
    if not npz.exists():
        pytest.skip(f"embeddings not found: {npz}")

    embs, sids, weeks = mod.load_embeddings(str(npz), cohort_filter="NaF")

    class _Wrap:  # minimal shim so the helpers above accept real arrays
        pass

    emb = _Wrap()
    emb.embeddings, emb.subject_ids, emb.weeks = embs, sids, weeks

    norm = float(np.linalg.norm(embs, axis=1).mean())
    cos = _flip_sensitivity(mod, emb)
    print(f"\n  REAL DATA  mean ||RAD-DINO emb|| = {norm:.2f}   cos(pred, pred_flipped) = {cos:.6f}")

    assert cos > 0.99, (
        f"The genotype conditioning bit materially steers the predicted embeddings on "
        f"real data (cos={cos:.6f} with mean ||emb||={norm:.2f}). Every VLM genotype "
        f"number that uses predicted_emb_dir is inflated by this. Re-run with "
        f"--no-conditioning."
    )
