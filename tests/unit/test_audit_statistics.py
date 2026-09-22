"""
F16 -- oracle-equivalence tests for the statistics that produced every CI and
permutation p-value fixed under F1.

Before Audit 2, `vlm/data/eval.py::_bootstrap_ci`/`_perm_p_auroc` and
`scripts/eval_stats.py::bootstrap_auc_ci`/`permutation_auc_p`/`_permute_labels`
had zero test coverage -- every "inside the null interval" claim in README.md and
results.md runs through one of these functions, and `test_pooled_auroc_estimator.py`
(the file F1's summary-table row points to) only exercises `_auroc` via its own
from-scratch simulation, never this code.

Also covers F18: the exact-enumeration path for small label-pure group counts
(e.g. F5's 10 mouse-groups), checked against an independent brute-force oracle.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# F18 -- exact enumeration, hand-checked against an independent brute force
# ---------------------------------------------------------------------------

def test_exact_enumeration_matches_hand_computed_p(scripts):
    """
    4 groups, 2 positive: C(4,2) = 6 distinct arrangements. Brute-force them here
    with a separate loop (not the library's own generator) and check
    permutation_auc_p's exact branch agrees exactly -- an oracle check, not a
    statistical approximation.
    """
    eval_stats = scripts("eval_stats")
    from itertools import combinations
    from sklearn.metrics import roc_auc_score

    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # groups a, b positive; c, d negative
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.6, 0.3, 0.4])

    group_names = ["a", "b", "c", "d"]
    idx = {g: np.where(groups == g)[0] for g in group_names}
    brute_null = []
    for combo in combinations(range(4), 2):
        y_p = np.zeros(8, dtype=int)
        for gi in combo:
            y_p[idx[group_names[gi]]] = 1
        brute_null.append(roc_auc_score(y_p, scores))
    obs_brute = roc_auc_score(y, scores)
    p_brute = sum(1 for v in brute_null if v >= obs_brute) / len(brute_null)

    obs, p, ci, meta = eval_stats.permutation_auc_p(y, scores, groups, exact_max=20000)
    assert meta["method"] == "exact"
    assert meta["n_arrangements"] == 6
    assert meta["resolution"] == pytest.approx(1 / 6)
    assert obs == pytest.approx(obs_brute)
    assert p == pytest.approx(p_brute)


def test_exact_enumeration_falls_back_to_monte_carlo_above_exact_max(scripts):
    """10 groups, 3 positive: C(10,3) = 120. Force exact_max below that and
    confirm it takes the Monte Carlo branch instead of silently going exact."""
    eval_stats = scripts("eval_stats")
    groups = np.array([f"g{i}" for i in range(10) for _ in range(2)])
    y = np.array(([1] * 2) * 3 + ([0] * 2) * 7)
    rng = np.random.default_rng(0)
    scores = rng.normal(size=20)
    obs, p, ci, meta = eval_stats.permutation_auc_p(
        y, scores, groups, exact_max=10, n_perm=500, seed=0)
    assert meta["method"] == "monte_carlo"
    assert meta["n_perm"] == 500


def test_n_exact_arrangements_matches_math_comb(scripts):
    """F5's exact shape: 10 groups, 3 WT / 7 KO -> C(10,3) = 120."""
    eval_stats = scripts("eval_stats")
    from math import comb
    y = np.array([1] * 3 + [0] * 7)
    groups = np.array([f"g{i}" for i in range(10)])
    g_lab, _inv = eval_stats._pure_group_labels(y, groups)
    assert eval_stats._n_exact_arrangements(g_lab) == comb(10, 3) == 120


# ---------------------------------------------------------------------------
# _permute_labels -- pure vs. mixed groups
# ---------------------------------------------------------------------------

def test_permute_labels_pure_groups_preserves_group_consistency(scripts):
    """
    Label-pure groups: every draw must keep each group internally consistent
    (one label per group, never split) and preserve the overall class counts --
    only WHICH groups get which label may change.
    """
    eval_stats = scripts("eval_stats")
    groups = np.array(["a", "a", "a", "b", "b", "c", "c", "c", "c"])
    y = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1])  # a, c positive; b negative
    group_order = ["a", "b", "c"]
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        y_p = eval_stats._permute_labels(y, groups, rng)
        group_labels = []
        for g in group_order:
            vals = set(y_p[groups == g].tolist())
            assert len(vals) == 1, f"group {g} split across labels: {vals}"
            group_labels.append(vals.pop())
        # Row counts differ by group size (a=3, b=2, c=4), so only the GROUP-level
        # label multiset -- not the row-level sum -- is invariant under permutation.
        assert sorted(group_labels) == [0, 1, 1]
        seen.add(tuple(group_labels))
    # C(3, 2) = 3 distinct group-level arrangements; 200 draws should see all of them.
    assert len(seen) == 3


def test_permute_labels_mixed_groups_preserves_within_group_composition(scripts):
    """
    Mixed-label groups (e.g. early-vs-late week, each mouse has both labels):
    permutation happens WITHIN each group, so each group's own label multiset is
    unchanged even though which row gets which label moves around.
    """
    eval_stats = scripts("eval_stats")
    groups = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])  # a: 2/2 split; b: 1/3 split
    rng = np.random.default_rng(0)
    for _ in range(50):
        y_p = eval_stats._permute_labels(y, groups, rng)
        assert y_p[groups == "a"].sum() == 2
        assert y_p[groups == "b"].sum() == 1
        assert y_p.sum() == y.sum()


# ---------------------------------------------------------------------------
# bootstrap_auc_ci -- sanity on separable vs. pure-noise data
# ---------------------------------------------------------------------------

def test_bootstrap_ci_tight_and_high_for_a_perfect_separator(scripts):
    eval_stats = scripts("eval_stats")
    groups = np.array([f"s{i}" for i in range(20)])
    y = np.array([1] * 10 + [0] * 10)
    scores = np.array([10.0] * 10 + [0.0] * 10)  # perfectly separable
    lo, hi = eval_stats.bootstrap_auc_ci(y, scores, groups, n_boot=2000, seed=0)
    assert lo > 0.9
    assert hi == pytest.approx(1.0, abs=1e-6)


def test_bootstrap_ci_wide_and_centred_for_pure_noise(scripts):
    eval_stats = scripts("eval_stats")
    rng = np.random.default_rng(0)
    groups = np.array([f"s{i}" for i in range(32)])
    y = np.array([1] * 16 + [0] * 16)
    scores = rng.normal(size=32)  # independent of y
    lo, hi = eval_stats.bootstrap_auc_ci(y, scores, groups, n_boot=3000, seed=0)
    assert lo < 0.5 < hi
    assert (hi - lo) > 0.25  # F1: the n=32 null is wide (sd ~0.105)


# ---------------------------------------------------------------------------
# Cross-agreement: eval.py's inline stats vs. eval_stats.py's, same input
# ---------------------------------------------------------------------------

def test_eval_py_and_eval_stats_py_agree_on_the_same_synthetic_case(scripts):
    """
    `vlm/data/eval.py`'s `_bootstrap_ci`/`_perm_p_auroc` and `scripts/eval_stats.py`'s
    `bootstrap_auc_ci`/`permutation_auc_p` are two independent implementations of the
    same idea (F16). Different RNGs and slightly different APIs mean they won't agree
    bit-for-bit, but on the same input they must land in the same neighbourhood --
    if they drift apart, one of them has a bug.
    """
    from data.eval import _auroc, _bootstrap_ci, _perm_p_auroc
    eval_stats = scripts("eval_stats")

    rng = np.random.default_rng(1)
    n = 32
    y = np.array([1] * 16 + [0] * 16)
    scores = rng.normal(size=n) + y * 0.6  # weak real signal, not pure noise

    # eval.py's _bootstrap_ci takes {group: [(label, score), ...]}; one row/subject.
    pairs_by_group = {i: [(int(y[i]), float(scores[i]))] for i in range(n)}
    lo_a, hi_a = _bootstrap_ci(pairs_by_group, _auroc, n_boot=3000, seed=0)
    p_a, _null_lo, _null_hi = _perm_p_auroc(y.tolist(), scores.tolist(), n_perm=3000, seed=0)

    groups = np.arange(n)
    lo_b, hi_b = eval_stats.bootstrap_auc_ci(y, scores, groups, n_boot=3000, seed=0)
    obs_b, p_b, _ci_b, meta_b = eval_stats.permutation_auc_p(y, scores, groups, n_perm=3000, seed=0)
    assert meta_b["method"] == "monte_carlo"  # C(32,16) is astronomically above exact_max

    assert abs(lo_a - lo_b) < 0.15
    assert abs(hi_a - hi_b) < 0.15
    assert abs(p_a - p_b) < 0.15
