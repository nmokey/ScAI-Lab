"""
R5 — is "genotype" separable because of biology, or because of the acquisition session?

Background
----------
From docs/DATA_MANIFEST.md, every scan session in this study contains mice of a
single genotype (WT 1-4, KO 5-8, ...), the *same* group of mice is scanned together
at every week, and `build_nifti_dataset.stage3_crop_and_write` assigns one
session-level `genotype` to every mouse cropped out of that session. Sessions also
differ in post-injection timepoint (1h vs 3h).

Genotype is therefore perfectly confounded with acquisition group. A classifier can
score well on "WT vs KO" by recognising scanner drift, bed position, reconstruction
settings or injection timing -- no biology required. This affects RAD-DINO's reported
T2c AUC = 0.869 and every downstream VLM genotype number.

Leave-one-subject-out does not control for this: the held-out mouse's session-mates
stay in the training set, carrying the same acquisition signature *and* the same
genotype label.

Calibration
-----------
The strength of the planted acquisition signature is a free parameter, so
`test_confound_strength_sweep` reports the whole curve rather than one number. The
default (STRENGTH = 4.0) is chosen so leave-one-subject-out lands at AUROC ~= 0.75,
i.e. the same magnitude as the genotype results actually reported for this dataset
(RAD-DINO T2c 0.869, VLM 0.762). The claim is not "the real effect is exactly this
big" -- it is "an effect of the reported size is fully reproducible from acquisition
structure alone, so the protocol cannot tell the two apart."

Acting on this finding requires adding `session_id` / `group_id` to the .npz contract
emitted by the encoder scripts, which do not currently record either.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.synth import loso_auroc, make_embeddings, permutation_null  # noqa: E402

pytestmark = pytest.mark.slow

N_PERM = 300
STRENGTH = 4.0


def _report(name, X, y, groups, n_perm=N_PERM):
    observed = loso_auroc(X, y, groups)
    null = permutation_null(
        lambda y_perm: loso_auroc(X, y_perm, groups), y, groups, n_perm=n_perm, seed=7
    )
    p = float((null >= observed).mean())
    print(f"\n  [{name}] AUROC = {observed:.3f}   null mean = {null.mean():.3f}   p = {p:.4f}")
    return observed, p


def _week12(emb):
    m = emb.weeks == "Week 12"
    return emb.embeddings[m], emb.genotype_binary[m], emb.subject_ids[m], emb.session_ids[m]


@pytest.fixture(scope="module")
def confounded():
    """
    Acquisition signature only -- no genotype direction is planted at all. Groups are
    genotype-pure and 4 mice wide and persist across weeks, mirroring the real design.
    """
    return make_embeddings(
        n_subjects=32,
        signal=("week", "group", "session"),
        strength=STRENGTH,
        session_size=4,
        seed=3,
    )


def test_sessions_are_genotype_pure(confounded):
    """Precondition: the fixture reproduces the real acquisition design."""
    by_session = {}
    for sess, geno in zip(confounded.session_ids, confounded.genotypes):
        by_session.setdefault(sess, set()).add(geno)
    mixed = {s: g for s, g in by_session.items() if len(g) > 1}
    assert not mixed, f"fixture sessions should be genotype-pure, found mixed: {mixed}"


def test_fixture_contains_no_genotype_direction(confounded):
    """
    Precondition: nothing genotype-specific was planted. Any genotype signal the
    probes recover is therefore attributable to acquisition structure alone.
    """
    assert "genotype" not in confounded.planted


@pytest.mark.probe
def test_subject_loso_is_confounded_by_session(confounded):
    """
    EXPECTED TO FAIL on the current protocol.

    Genotype is recoverable under leave-one-subject-out even though the embeddings
    contain zero genotype information -- because the held-out mouse's session-mates
    remain in training with the same acquisition signature and the same label.
    """
    X, y, sids, _ = _week12(confounded)
    auc, p = _report("LOSO by subject", X, y, sids)
    assert p > 0.05, (
        f"Acquisition signature alone yields genotype AUROC={auc:.3f} (permutation "
        f"p={p:.4f}) under leave-one-subject-out, with no genotype signal present. "
        f"Genotype results of the reported magnitude (RAD-DINO T2c=0.869; VLM 0.762) "
        f"cannot be attributed to biology under this protocol. Re-run grouping folds "
        f"by session, and record session_id in the embedding .npz."
    )


def test_session_loso_is_clean(confounded):
    """
    The control arm: grouping folds by session removes the confound and genotype
    drops to chance, as it must. This is the protocol the genotype results need.
    """
    X, y, _, sessions = _week12(confounded)
    auc, p = _report("LOSO by session", X, y, sessions)
    assert p > 0.05, (
        f"Leave-one-session-out should be clean but gave AUROC={auc:.3f} (p={p:.4f}); "
        f"the fixture or the probe is at fault."
    )


def test_confound_strength_sweep():
    """
    The magnitude of the confound as a function of how strongly the acquisition
    signature is imprinted. Reported rather than asserted at a single point, because
    the real imprint strength is unknown -- that is precisely what needs measuring on
    the real embeddings (see test_session_confound_realdata).
    """
    print("\n  acquisition-signature strength -> genotype AUROC (no genotype signal present)")
    print("     strength   LOSO-by-subject   LOSO-by-session")
    rows = []
    for strength in (2.0, 4.0, 8.0, 16.0):
        emb = make_embeddings(
            n_subjects=32, signal=("week", "group", "session"),
            strength=strength, session_size=4, seed=3,
        )
        X, y, sids, sessions = _week12(emb)
        a_subj = loso_auroc(X, y, sids)
        a_sess = loso_auroc(X, y, sessions)
        rows.append((strength, a_subj, a_sess))
        print(f"     {strength:6.1f}       {a_subj:.3f}             {a_sess:.3f}")

    # Monotone in the imprint strength, and session-grouping never inflates.
    assert rows[-1][1] > rows[0][1], "stronger acquisition signature should inflate subject-LOSO"
    assert all(a_sess <= 0.65 for _, _, a_sess in rows), (
        "leave-one-session-out must stay near chance at every strength"
    )


@pytest.mark.realdata
def test_session_confound_realdata(data_root):
    """
    The measurement that actually settles this, once session_id is recorded.

    Skipped today for a second reason beyond missing data: the .npz contract has no
    session field, so the grouping cannot be reconstructed from the embeddings alone.
    Wiring it through `build_nifti_dataset` -> mouse_manifest.csv -> the encoder
    scripts is the prerequisite.
    """
    npz = data_root / "embeddings" / "raddino" / "raddino_embeddings.npz"
    if not npz.exists():
        pytest.skip(f"embeddings not found: {npz}")

    data = np.load(npz, allow_pickle=True)
    if "session_ids" not in data:
        pytest.skip(
            "raddino_embeddings.npz has no `session_ids` field -- the session-confound "
            "control cannot be run until build_nifti_dataset records the source session "
            "per mouse and the encoder scripts propagate it into the .npz."
        )

    embs = data["embeddings"].astype(np.float32)
    sids = data["subject_ids"].astype(str)
    weeks = data["weeks"].astype(str)
    sessions = data["session_ids"].astype(str)
    geno = np.array([1 if "_KO_" in s else 0 for s in sids])

    m = weeks == "Week 12"
    by_subject = loso_auroc(embs[m], geno[m], sids[m])
    by_session = loso_auroc(embs[m], geno[m], sessions[m])
    print(f"\n  REAL DATA  genotype AUROC: by-subject={by_subject:.3f}  by-session={by_session:.3f}")

    assert by_session >= by_subject - 0.15, (
        f"Genotype AUROC collapses from {by_subject:.3f} (leave-one-subject-out) to "
        f"{by_session:.3f} (leave-one-session-out). The reported genotype signal is "
        f"substantially an acquisition artifact, not biology."
    )
