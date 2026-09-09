"""
Are the reported genotype AUROCs distinguishable from noise?

The setup
---------
`run_mouse_vlm_loso.py` trains one model per fold and appends each fold's predictions to
a single aggregate file; `vlm/data/eval.py::calculate_mouse_metrics` then collects every
record's raw `genotype_logit` into one list and computes a single rank-based AUROC over
all of them, reported against an implicit "chance = 0.5".

Two structural facts:

 1. **Every fold is single-class.** A fold holds out one subject, which contributes
    exactly one genotype record (eval.py routes questions containing "status" and not
    "TBR"). One subject has one genotype, so AUROC is undefined within a fold and only
    exists after pooling.
 2. **Every logit comes from a different model** -- 32 separately-trained LoRA adapters
    and 32 separately-initialised heads, none standardised before being ranked together.

A hypothesis that did NOT survive
---------------------------------
The original suspicion was that per-fold calibration offsets would widen the null and
manufacture apparent signal. `test_per_fold_offsets_do_not_distort_the_estimator` shows
that is **false**: offsets independent of the label add symmetric noise, and AUROC is
rank-based, so the null is unchanged (sd = 0.103 at every offset magnitude tested).
Recorded because a plausible-sounding mechanism being ruled out is worth keeping.

What the tests did find
-----------------------
Something simpler and harder to dismiss: **at n = 32, the null distribution of AUROC is
enormous** -- sd = 0.105, 95% interval [0.293, 0.699] -- with no per-fold pathology
required. Comparing a reported AUROC against 0.5 without that interval is not a test.
Two of the four reported genotype AUROCs sit inside the no-signal interval, and the
headline 0.762 survives as a single test (p = 0.005) but not as the maximum of the 2x5
epoch sweep it was actually selected from (p = 0.045, before any correction for R1).

These tests are pure numpy plus the project's own `_auroc`. No model is involved: the
question is about the estimator and the sample size, not the VLM.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.unit

N_SUBJECTS = 32   # the real LOSO fold count
N_SIM = 20_000    # null draws
SWEEP_N = 10      # docs/experiments.md: 2 input arms x 5 epoch settings

# docs/experiments.md, summary table.
REPORTED = {
    "VLM longitudinal 20ep (headline)": 0.762,
    "VLM longitudinal 100ep": 0.714,
    "VLM longitudinal 10ep": 0.560,
    "TinyLlama 10ep": 0.397,
    "VLM baseline ts0 10ep": 0.163,
}


def _pooled_auroc(labels, scores):
    """The project's own estimator, so this measures what is actually reported."""
    from data.eval import _auroc

    return _auroc(list(labels), list(scores))


def _null_draw(seed, offset_sd=0.0, n_subjects=N_SUBJECTS):
    """
    One LOSO run with NO genotype signal: each fold contributes a single logit made of
    that fold model's calibration offset plus within-model noise. Nothing depends on
    the label, so any departure from 0.5 is sampling variation.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([0, 1] * (n_subjects // 2))
    rng.shuffle(labels)
    logits = rng.normal(0, offset_sd, n_subjects) + rng.normal(0, 1.0, n_subjects)
    return _pooled_auroc(labels, logits)


@pytest.fixture(scope="module")
def null():
    """The no-signal distribution of the reported statistic at n = 32."""
    return np.array([_null_draw(s) for s in range(N_SIM)])


def test_folds_are_single_class_by_construction():
    """
    The premise: one held-out subject per fold, one genotype record per subject, so no
    fold contains both classes. Verified against eval.py's actual routing rule.
    """
    question = "What will be the eventual mouse status for atherosclerosis?"
    assert "status" in question.lower() and "TBR" not in question, (
        "genotype routing rule changed -- re-derive this test"
    )
    assert np.isnan(_pooled_auroc([1], [0.3])), (
        "AUROC within a single-class fold must be undefined; it is only computable "
        "after pooling across folds"
    )


def test_per_fold_offsets_do_not_distort_the_estimator():
    """
    A refuted hypothesis, kept as a result.

    Differently-calibrated fold models were the suspected culprit. They are not: because
    the offsets carry no label information and AUROC is rank-based, the null is
    unchanged across two orders of magnitude of offset. Pooling across fold models is
    not, by itself, the problem here.
    """
    print("\n  per-fold offset SD    null mean    null sd")
    sds = []
    for offset_sd in (0.0, 0.5, 1.0, 2.0, 4.0):
        vals = np.array([_null_draw(s, offset_sd=offset_sd) for s in range(2000)])
        sds.append(vals.std())
        print(f"     {offset_sd:6.1f}            {vals.mean():.3f}       {vals.std():.3f}")

    assert max(sds) - min(sds) < 0.02, (
        f"offsets were expected to leave the null unchanged, but sd ranged "
        f"{min(sds):.3f}-{max(sds):.3f}"
    )


def test_null_distribution_at_n32_is_very_wide(null):
    """
    The actual problem. Establishes the interval every reported AUROC has to clear.
    """
    lo, hi = np.percentile(null, [2.5, 97.5])
    print(
        f"\n  n={N_SUBJECTS}, no signal: mean={null.mean():.3f}  sd={null.std():.3f}"
        f"\n  95% interval = [{lo:.3f}, {hi:.3f}]"
    )
    assert abs(null.mean() - 0.5) < 0.01, "estimator should be unbiased"
    assert hi - lo > 0.35, (
        f"null interval is {hi - lo:.3f} wide at n={N_SUBJECTS}; any comparison against "
        f"a bare 'chance = 0.5' ignores this"
    )


@pytest.mark.probe
def test_reported_aurocs_clear_the_no_signal_interval(null):
    """
    EXPECTED TO FAIL.

    Sampling noise alone at n = 32 produces AUROCs across [0.29, 0.70] with no signal
    present. Any reported value inside that interval is not evidence of genotype signal.
    """
    lo, hi = np.percentile(null, [2.5, 97.5])
    print(f"\n  no-signal 95% interval at n={N_SUBJECTS}: [{lo:.3f}, {hi:.3f}]")
    inside = {}
    for name, val in REPORTED.items():
        p_hi = float((null >= val).mean())
        p_lo = float((null <= val).mean())
        within = lo <= val <= hi
        if within:
            inside[name] = val
        print(
            f"    {name:34s} {val:.3f}  "
            f"{'INSIDE null interval' if within else 'outside'}   "
            f"P(null>=x)={p_hi:.4f}  P(null<=x)={p_lo:.4f}"
        )

    assert not inside, (
        f"{len(inside)} of {len(REPORTED)} reported genotype AUROCs fall inside the 95% "
        f"interval produced with NO signal at all ([{lo:.3f}, {hi:.3f}]): {inside}. At "
        f"n=32 the null sd is {null.std():.3f}, so differences of this size are not "
        f"interpretable without a confidence interval. Report bootstrap CIs and a "
        f"permutation p-value, not a comparison against 0.5."
    )


@pytest.mark.probe
def test_headline_auroc_survives_the_sweep_multiplicity(null):
    """
    EXPECTED TO FAIL -- the sharpest version of the point.

    0.762 was not a pre-registered single test. docs/experiments.md reports it as the
    best cell of a 2x5 grid (baseline vs longitudinal, 15/20/25/30/40 epochs), and
    README.md calls 20 epochs "the sweet spot". The relevant null is therefore the
    distribution of the *maximum* over 10 configurations, not of one draw.

    Note this is still generous to the result: it ignores R1 entirely. Checkpoint
    selection on the held-out subject was measured to add +0.14 to +0.20 AUROC on
    signal-free data, which would have to be subtracted before this comparison is fair.
    """
    headline = REPORTED["VLM longitudinal 20ep (headline)"]
    single_p = float((null >= headline).mean())

    usable = (len(null) // SWEEP_N) * SWEEP_N
    sweep_max = null[:usable].reshape(-1, SWEEP_N).max(axis=1)
    sweep_p = float((sweep_max >= headline).mean())
    lo, hi = np.percentile(sweep_max, [2.5, 97.5])

    print(
        f"\n  headline AUROC = {headline:.3f}"
        f"\n    as a single pre-registered test : p = {single_p:.4f}"
        f"\n    as the max of a {SWEEP_N}-config sweep : p = {sweep_p:.4f}"
        f"\n    max-of-{SWEEP_N} null: median={np.median(sweep_max):.3f}  "
        f"95% interval=[{lo:.3f}, {hi:.3f}]"
    )

    assert sweep_p < 0.01, (
        f"Selecting the best of {SWEEP_N} sweep configurations, the headline genotype "
        f"AUROC of {headline:.3f} has p = {sweep_p:.4f} against a no-signal null -- "
        f"marginal, where the uncorrected single-test p is {single_p:.4f}. The median "
        f"maximum over {SWEEP_N} signal-free configurations is {np.median(sweep_max):.3f}. "
        f"And this ignores R1: checkpoint selection on the held-out subject adds a "
        f"measured +0.14 to +0.20 on noise, which would place the headline comfortably "
        f"inside this null. 'Longitudinal tokens carry the genotype signal' and "
        f"'20 epochs is the sweet spot' are not supported as stated."
    )


def test_below_chance_baseline_is_anomalous_not_noise(null):
    """
    The ts0-only baseline AUROC of 0.163 is genuinely extreme -- P(null <= 0.163) is
    about 4 in 10,000 -- so unlike the values above it is NOT explainable as sampling
    noise. Something systematic drives it.

    README.md reads it as "its head ranks genotype anti-correlated, i.e. there is no
    stable single-scan genotype signal to learn." That interpretation does not follow:
    no signal produces AUROC ~0.5, not 0.163. A systematically inverted ranking needs a
    mechanism, and two candidates are already on the table -- R1 (checkpoint selected by
    held-out loss, which need not align with head ranking) and R3 (the head's decision
    context differs between training and inference). This test pins the anomaly; it does
    not resolve it.
    """
    val = REPORTED["VLM baseline ts0 10ep"]
    p_lo = float((null <= val).mean())
    print(f"\n  baseline AUROC {val:.3f}: P(null <= x) = {p_lo:.4f} -- too extreme for noise")
    assert p_lo < 0.01, (
        f"expected 0.163 to be extreme under the null, got P = {p_lo:.4f}"
    )


@pytest.mark.realdata
def test_measure_per_fold_offsets_from_loso_predictions(data_root):
    """
    Even though offsets turned out not to distort the estimator, their magnitude is
    worth recording: large between-fold spread relative to within-fold spread means the
    32 fold models are not comparable objects, which bears on whether pooling their
    predictions is meaningful at all.
    """
    import json

    pred = data_root / "embeddings" / "vlm" / "runs" / "mouse_vlm_loso" / "vqa_loso.json"
    if not pred.exists():
        pytest.skip(f"LOSO predictions not found: {pred}")

    by_fold = {}
    for r in json.loads(pred.read_text()):
        if "genotype_logit" in r and r.get("pid"):
            by_fold.setdefault(r["pid"], []).append(float(r["genotype_logit"]))

    fold_means = np.array([np.mean(v) for v in by_fold.values()])
    within = np.array([np.std(v) for v in by_fold.values() if len(v) > 1])
    between_sd = float(fold_means.std())
    within_sd = float(within.mean()) if len(within) else float("nan")

    print(f"\n  REAL DATA  {len(by_fold)} folds")
    print(f"    between-fold logit SD : {between_sd:.3f}")
    print(f"    within-fold logit SD  : {within_sd:.3f}")
    assert between_sd < within_sd, (
        f"Between-fold logit spread ({between_sd:.3f}) exceeds within-fold spread "
        f"({within_sd:.3f}): the 32 fold models are calibrated too differently for their "
        f"raw logits to be pooled and ranked as one set."
    )
