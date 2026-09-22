"""
Shared statistics and metadata helpers for the evaluation scripts.

F1  -- subject-level bootstrap confidence intervals and permutation p-values, so
       no AUROC is ever reported against a bare "chance = 0.5" at n=32.
F5  -- session_id join from mouse_manifest.csv, so leave-one-session-out can be
       run on the existing .npz files without re-extracting any embeddings.
       (subject_id, week) is a unique key in the manifest: 229/229 rows join.

Importable from scripts/ (standalone) and from the test suite via conftest's
sys.path shim.
"""

import csv
import os
from collections import defaultdict
from itertools import combinations
from math import comb

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# F5 -- session join
# ---------------------------------------------------------------------------

def load_session_map(manifest_path):
    """(subject_id, 'Week N') -> session_id, from mouse_manifest.csv."""
    m = {}
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["mouse_id"], f"Week {row['week']}")
            if key in m and m[key] != row["session_id"]:
                raise ValueError(f"Ambiguous session for {key}: {m[key]} vs {row['session_id']}")
            m[key] = row["session_id"]
    return m


def join_session_ids(subject_ids, weeks, manifest_path):
    """
    Return an array of session_id aligned with subject_ids/weeks.

    Raises if any row fails to join -- a silent fallback to subject grouping is
    exactly the failure mode this exists to prevent.
    """
    sess = load_session_map(manifest_path)
    out, missing = [], []
    for s, w in zip(subject_ids, weeks):
        key = (str(s), str(w))
        if key not in sess:
            missing.append(key)
        out.append(sess.get(key, ""))
    if missing:
        raise KeyError(f"{len(missing)} embedding rows have no session in {manifest_path}: "
                       f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")
    return np.array(out, dtype=str)


def join_mouse_group_ids(subject_ids, weeks, manifest_path):
    """
    Return an array of mouse-GROUP ids aligned with subject_ids/weeks.

    A mouse-group is a connected component of the mouse--session bipartite graph:
    the set of mice that are ever scanned together, closed over all their sessions.
    On this dataset there are 10 such groups and EVERY ONE is genotype-pure, so
    genotype is confounded with group at the study-design level.

    Leave-one-SESSION-out is not a clean control for that: the held-out session's
    group-mates stay in training with the same label, and so do the held-out mice
    themselves at their other weeks. Leave-one-GROUP-out is the only split where
    no mouse and no session of the held-out block appears in training. If a
    genotype probe still generalises across groups, the signal is at least not
    group-specific; if it collapses, it was.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            a, b = find("M:" + row["mouse_id"]), find("S:" + row["session_id"])
            if a != b:
                parent[a] = b

    roots = {}
    out, missing = [], []
    for s in subject_ids:
        key = "M:" + str(s)
        if key not in parent:
            missing.append(str(s))
            out.append("")
            continue
        r = find(key)
        out.append(roots.setdefault(r, f"group{len(roots):02d}"))
    if missing:
        raise KeyError(f"{len(set(missing))} subjects absent from {manifest_path}: "
                       f"{sorted(set(missing))[:5]}")
    return np.array(out, dtype=str)


def default_manifest_path():
    """The manifest next to the pipeline outputs, per config.yaml, or None."""
    try:
        import yaml
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        out = cfg.get("paths", {}).get("output_dir")
        if out:
            p = os.path.join(out, "mouse_manifest.csv")
            if os.path.exists(p):
                return p
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# F1 -- uncertainty for pooled AUROC
# ---------------------------------------------------------------------------

def _safe_auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(set(y.tolist())) < 2:
        return np.nan
    return roc_auc_score(y, s)


def bootstrap_auc_ci(y_true, y_score, groups, n_boot=10000, seed=0, alpha=0.05):
    """
    Percentile bootstrap CI for a pooled AUROC, resampling GROUPS (subjects)
    with replacement rather than rows. Rows from the same subject are not
    independent, and F14 showed the pipeline had already been double-counting
    them; a row-level bootstrap on such data is too narrow by about sqrt(2).
    """
    y_true = np.asarray(y_true); y_score = np.asarray(y_score); groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        a = _safe_auc(y_true[idx], y_score[idx])
        if not np.isnan(a):
            aucs.append(a)
    if not aucs:
        return (np.nan, np.nan)
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _pure_group_labels(y, groups):
    """
    (group_label, group_index) if every group is label-pure (all its rows share one
    label), else (None, None). `group_index` is `inv` from `np.unique(..., return_
    inverse=True)`: `group_label[group_index]` broadcasts group labels back to rows.
    """
    uniq, inv = np.unique(groups, return_inverse=True)
    m = len(uniq)
    if not all(len(set(y[inv == k].tolist())) == 1 for k in range(m)):
        return None, None
    g_lab = np.array([int(y[inv == k][0]) for k in range(m)])
    return g_lab, inv


def _n_exact_arrangements(g_lab):
    """C(m, k): the number of distinct group-level label vectors (F18)."""
    k = int(g_lab.sum())
    return comb(len(g_lab), k)


def _exact_pure_arrangements(g_lab, inv):
    """
    Yield every distinct row-level label vector for a label-pure group structure.

    A group-level binary vector with m groups and k positives has exactly
    C(m, k) distinct arrangements -- e.g. 120 for F5's 10 groups split 3 WT / 7 KO.
    Enumerating them is both exact and (at this size) cheaper than Monte Carlo
    sampling from the same finite space.
    """
    m = len(g_lab)
    k = int(g_lab.sum())
    for combo in combinations(range(m), k):
        perm_g = np.zeros(m, dtype=int)
        perm_g[list(combo)] = 1
        yield perm_g[inv]


def _permute_labels(y, groups, rng):
    """
    One draw from the label-exchangeable null that respects the group structure.

    * Label-pure groups (e.g. genotype: every scan of a mouse shares its label):
      the groups are the exchangeable units, so permute labels ACROSS groups and
      broadcast -- each group keeps one consistent label.
    * Mixed-label groups (e.g. early-vs-late week: each mouse has both): the
      exchangeable units are rows WITHIN a group, so permute labels within each
      group and leave the group's label composition intact.

    Taking group[0]'s label for a mixed group -- the previous behaviour -- assigns
    one label to scans that in truth carry both, which is not a valid null.
    """
    g_lab, inv = _pure_group_labels(y, groups)
    if g_lab is not None:
        perm = rng.permutation(g_lab)
        return perm[inv]
    uniq, inv = np.unique(groups, return_inverse=True)
    y_p = y.copy()
    for k in range(len(uniq)):
        idx = np.where(inv == k)[0]
        y_p[idx] = rng.permutation(y[idx])
    return y_p


def permutation_auc_p(y_true, y_score, groups, n_perm=10000, seed=0, exact_max=20000):
    """
    One-sided permutation p-value for a pooled AUROC with FIXED scores: how often
    does a random relabelling score at least as high? Labels are permuted with
    `_permute_labels`, which respects whether groups are label-pure or mixed.

    This is the estimator null from F1 -- it does not refit anything, so it is
    the right test for VLM predictions (where refitting is infeasible). For a
    linear probe, prefer `permutation_refit_p`, which refits under each
    permutation and is the stricter test.

    Returns (obs, p, null_ci95, meta). F18: when the groups are label-pure and small
    enough that the exact null has at most `exact_max` distinct arrangements
    (C(n_groups, n_positive_groups) -- 120 for F5's 10 groups split 3 WT / 7 KO),
    every arrangement is enumerated exactly instead of Monte Carlo sampled from that
    same finite space, and `meta` records it (`method="exact"`, `n_arrangements`,
    `resolution` = 1/n_arrangements). The exact p-value is quantised in steps of
    `resolution` regardless of how it's computed -- callers should report that
    alongside the value rather than implying more precision than a small group count
    can support. Otherwise `meta` is `{"method": "monte_carlo", "n_perm": n_perm}`.
    """
    y_true = np.asarray(y_true); y_score = np.asarray(y_score); groups = np.asarray(groups)
    obs = _safe_auc(y_true, y_score)
    if np.isnan(obs):
        return np.nan, np.nan, (np.nan, np.nan), {"method": "none"}

    g_lab, inv = _pure_group_labels(y_true, groups)
    if g_lab is not None:
        n_exact = _n_exact_arrangements(g_lab)
        if n_exact <= exact_max:
            null = np.array([_safe_auc(y_p, y_score) for y_p in _exact_pure_arrangements(g_lab, inv)])
            null = null[~np.isnan(null)]
            # Exact: the observed arrangement is already one of the n_exact enumerated
            # ones, so no "+1" pseudo-count (that correction is only for Monte Carlo,
            # which doesn't otherwise include the observed draw).
            p = float(np.sum(null >= obs) / len(null))
            meta = {"method": "exact", "n_arrangements": int(n_exact), "resolution": 1.0 / n_exact}
            return (float(obs), p,
                    (float(np.nanpercentile(null, 2.5)), float(np.nanpercentile(null, 97.5))), meta)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = _safe_auc(_permute_labels(y_true, groups, rng), y_score)
    null = null[~np.isnan(null)]
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    meta = {"method": "monte_carlo", "n_perm": n_perm}
    return float(obs), p, (float(np.nanpercentile(null, 2.5)), float(np.nanpercentile(null, 97.5))), meta


def permutation_refit_p(fit_predict_fn, X, y, groups, n_perm=200, seed=0, exact_max=2000):
    """
    Permutation p-value that REFITS the model under each relabelling.

    fit_predict_fn(X, y, groups) -> (y_true_pooled, y_score_pooled), e.g. a LOSO
    linear probe. Labels are permuted at the group level. Slower than
    `permutation_auc_p` -- one full CV per permutation -- so the default n_perm
    is modest; resolution is ~1/n_perm.

    Returns (obs, p, null_ci95, meta), same F18 exact-enumeration contract as
    `permutation_auc_p`, but with a smaller default `exact_max` since each
    arrangement here costs a full refit, not just an AUROC.
    """
    X = np.asarray(X); y = np.asarray(y); groups = np.asarray(groups)
    yt, ys = fit_predict_fn(X, y, groups)
    obs = _safe_auc(yt, ys)
    if np.isnan(obs):
        return np.nan, np.nan, (np.nan, np.nan), {"method": "none"}

    g_lab, inv = _pure_group_labels(y, groups)
    if g_lab is not None:
        n_exact = _n_exact_arrangements(g_lab)
        if n_exact <= exact_max:
            null = []
            for y_p in _exact_pure_arrangements(g_lab, inv):
                yt_p, ys_p = fit_predict_fn(X, y_p, groups)
                a = _safe_auc(yt_p, ys_p)
                if not np.isnan(a):
                    null.append(a)
            null = np.array(null)
            p = float(np.sum(null >= obs) / len(null))
            meta = {"method": "exact", "n_arrangements": int(n_exact), "resolution": 1.0 / n_exact}
            return (float(obs), p,
                    (float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))), meta)

    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        y_p = _permute_labels(y, groups, rng)
        yt_p, ys_p = fit_predict_fn(X, y_p, groups)
        a = _safe_auc(yt_p, ys_p)
        if not np.isnan(a):
            null.append(a)
    null = np.array(null)
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    meta = {"method": "monte_carlo", "n_perm": n_perm}
    return float(obs), p, (float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))), meta
