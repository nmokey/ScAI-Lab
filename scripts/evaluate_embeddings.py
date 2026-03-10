"""
evaluate_embeddings.py

Encoder-agnostic evaluation suite for vision encoder embeddings.

Runs a fixed battery of tasks on any .npz file that conforms to the standard
embedding interface, then saves metrics.csv, report.txt, and dimensionality-
reduction plots. Designed so that any encoder (COLIPRI, RadDINO, custom MAE,
etc.) can be compared under identical conditions by swapping the .npz input.

Standard .npz interface (all encoder scripts must conform):
    embeddings   : float32  (N, D)   — one vector per volume
    subject_ids  : str      (N,)     — mouse ID, e.g. "m54253"
    weeks        : str      (N,)     — "Week 12" | "Week 15" | "Week 18" | "Week 20"
    modalities   : str      (N,)     — "CT_HiRes" | "CT_LowRes" | "PET_FDG" | "PET_NaF"
    paths        : str      (N,)     — absolute path to source NIfTI

Tasks
-----
Tier 1 — Unsupervised (no labels required):
  T1a  t-SNE and UMAP visualisations (colored by week and by subject)
  T1b  k-means cluster alignment vs. week labels (ARI, NMI)
  T1c  Silhouette score by week (cosine distance)
  T1d  Within-subject cosine similarity vs. cross-subject (Δ = intra − inter)

Tier 2 — Linear probe (Leave-One-Subject-Out cross-validation):
  T2a  4-class week classification (accuracy, macro-F1, OvR AUC)
  T2b  Binary early (Week 12) vs. late (Week 20) classification (accuracy, AUC-ROC)
  T2c  Binary WT vs. KO genotype classification (accuracy, AUC-ROC)
  T2d  Binary NaF vs. FDG cohort classification (accuracy, AUC-ROC)
  T2e  5-class WT vs. (KO + stage) disease-staging (accuracy, macro-F1, OvR AUC)
  A5   Per-cohort conditioned analysis: T2a and T2b run separately for NaF and FDG

Tier 3 — Longitudinal / relational:
  T3a  Pairwise temporal ordering — can embeddings order two scans of the same
       mouse by disease stage? (LOSO, chance = 50%)
  T3b  Same-subject retrieval — Recall@1, Recall@3, MRR
  T3c  Same-week retrieval — mAP@5

Usage
-----
    python scripts/evaluate_embeddings.py \\
        --embeddings /path/to/colipri_embeddings.npz \\
        --output-dir /path/to/eval_output/colipri/
"""

import argparse
import csv
import os
import warnings
from itertools import combinations

import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder, normalize

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("[!] umap-learn not installed — UMAP plots will be skipped. Install with: pip install umap-learn")

# Canonical week ordering (earliest → latest disease stage)
WEEK_ORDER = ["Week 12", "Week 15", "Week 18", "Week 20"]
WEEK_COLORS = {
    "Week 12": "#2196F3",  # blue  — early / healthy
    "Week 15": "#4CAF50",  # green
    "Week 18": "#FF9800",  # orange
    "Week 20": "#F44336",  # red   — late / diseased
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(val, spec=".4f"):
    """Format a numeric value for the report; returns 'n/a' for NaN/None."""
    if val is None:
        return "n/a"
    try:
        if isinstance(val, float) and np.isnan(val):
            return "n/a"
        return format(val, spec)
    except (TypeError, ValueError):
        return str(val)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings(path):
    data = np.load(path, allow_pickle=True)
    embs        = data["embeddings"].astype(np.float32)
    subject_ids = data["subject_ids"].astype(str)
    weeks       = data["weeks"].astype(str)
    modalities  = data["modalities"].astype(str)
    paths       = data["paths"].astype(str)

    print(f"[i] Loaded {len(embs)} embeddings  (dim={embs.shape[1]})")
    print(f"    Subjects  : {sorted(set(subject_ids))}")
    print(f"    Weeks     : {sorted(set(weeks))}")
    print(f"    Modalities: {sorted(set(modalities))}")
    return embs, subject_ids, weeks, modalities, paths


# ---------------------------------------------------------------------------
# T1a — Dimensionality reduction plots
# ---------------------------------------------------------------------------

def _scatter_2d(coords, labels, color_by, out_path, title):
    """Save a 2-D scatter plot; color_by is 'week' or 'subject'."""
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = sorted(set(labels))

    if color_by == "week":
        for ul in unique_labels:
            mask = np.array(labels) == ul
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       label=ul, s=45, alpha=0.85,
                       color=WEEK_COLORS.get(ul, "#999999"))
        ax.legend(title="Week", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    else:
        # Too many subjects for a manual legend — use a colormap and label centroids
        cmap = plt.cm.get_cmap("tab20", len(unique_labels))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        colors = [cmap(label_to_idx[l]) for l in labels]
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=45, alpha=0.85)
        # Annotate each subject at its centroid
        for ul in unique_labels:
            mask = np.array(labels) == ul
            cx, cy = coords[mask].mean(axis=0)
            ax.annotate(ul, (cx, cy), fontsize=6, ha="center",
                        color=cmap(label_to_idx[ul]))

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [+] {os.path.basename(out_path)}")


def run_t1a_plots(embs, weeks, subject_ids, plots_dir):
    print("\n[T1a] Dimensionality reduction plots...")
    n = len(embs)
    perplexity = min(30, max(5, n // 4))

    print(f"    t-SNE (perplexity={perplexity}, n={n})...")
    tsne_coords = TSNE(n_components=2, perplexity=perplexity,
                       random_state=42, max_iter=1000).fit_transform(embs)
    _scatter_2d(tsne_coords, list(weeks), "week",
                os.path.join(plots_dir, "tsne_by_week.png"),
                "t-SNE — colored by Week")
    _scatter_2d(tsne_coords, list(subject_ids), "subject",
                os.path.join(plots_dir, "tsne_by_subject.png"),
                "t-SNE — colored by Subject")

    if UMAP_AVAILABLE:
        n_neighbors = min(15, max(2, n - 1))
        print(f"    UMAP (n_neighbors={n_neighbors}, n={n})...")
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        umap_coords = reducer.fit_transform(embs)
        _scatter_2d(umap_coords, list(weeks), "week",
                    os.path.join(plots_dir, "umap_by_week.png"),
                    "UMAP — colored by Week")
        _scatter_2d(umap_coords, list(subject_ids), "subject",
                    os.path.join(plots_dir, "umap_by_subject.png"),
                    "UMAP — colored by Subject")

    return {}  # no numeric metrics for this task


# ---------------------------------------------------------------------------
# T1b — Unsupervised cluster alignment
# ---------------------------------------------------------------------------

def run_t1b_cluster_alignment(embs, weeks):
    print("\n[T1b] Unsupervised cluster alignment (k-means, k=4)...")
    if len(set(weeks)) < 2:
        print("    [!] Only one week label — skipping.")
        return {"T1b_ARI": float("nan"), "T1b_NMI": float("nan")}

    week_labels = LabelEncoder().fit_transform(weeks)
    cluster_labels = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(embs)

    ari = adjusted_rand_score(week_labels, cluster_labels)
    nmi = normalized_mutual_info_score(week_labels, cluster_labels)
    print(f"    ARI = {ari:.4f}  (0 = random, 1 = perfect)")
    print(f"    NMI = {nmi:.4f}  (0 = random, 1 = perfect)")
    return {"T1b_ARI": ari, "T1b_NMI": nmi}


# ---------------------------------------------------------------------------
# T1c — Silhouette score by week
# ---------------------------------------------------------------------------

def run_t1c_silhouette(embs, weeks):
    print("\n[T1c] Silhouette score by week (cosine distance)...")
    if len(set(weeks)) < 2:
        print("    [!] Only one week label — skipping.")
        return {"T1c_silhouette_cosine": float("nan")}

    week_labels = LabelEncoder().fit_transform(weeks)
    sil = silhouette_score(normalize(embs), week_labels, metric="cosine")
    print(f"    Silhouette = {sil:.4f}  (>0 means week clusters are more cohesive than random)")
    return {"T1c_silhouette_cosine": sil}


# ---------------------------------------------------------------------------
# T1d — Within-subject vs. cross-subject cosine similarity
# ---------------------------------------------------------------------------

def run_t1d_subject_consistency(embs, subject_ids):
    print("\n[T1d] Within-subject consistency...")
    cos_sim = normalize(embs) @ normalize(embs).T  # (N, N)

    intra, inter = [], []
    n = len(subject_ids)
    for i in range(n):
        for j in range(i + 1, n):
            (intra if subject_ids[i] == subject_ids[j] else inter).append(cos_sim[i, j])

    if not intra:
        print("    [!] No subject has multiple timepoints — skipping.")
        return {"T1d_intra_subject_sim": float("nan"),
                "T1d_inter_subject_sim": float("nan"),
                "T1d_delta": float("nan")}

    mean_intra = float(np.mean(intra))
    mean_inter = float(np.mean(inter))
    delta = mean_intra - mean_inter
    print(f"    Intra-subject cosine sim : {mean_intra:.4f}")
    print(f"    Inter-subject cosine sim : {mean_inter:.4f}")
    print(f"    Δ = {delta:.4f}  (>0 means the encoder preserves mouse identity across timepoints)")
    return {"T1d_intra_subject_sim": mean_intra,
            "T1d_inter_subject_sim": mean_inter,
            "T1d_delta": delta}


# ---------------------------------------------------------------------------
# Shared LOSO logistic regression helper
# ---------------------------------------------------------------------------

def _loso_logistic(X, y, groups, task_name, *, return_proba=False, multiclass=False):
    """
    Leave-One-Subject-Out logistic regression.

    Returns (accuracy, macro-F1) and optionally (y_true, y_prob) for AUC.
    When multiclass=True, y_prob is a 2-D array (n_samples, n_classes) for OvR AUC.
    When multiclass=False (binary), y_prob is a 1-D array of positive-class probabilities.
    """
    loso = LeaveOneGroupOut()
    y_true_all, y_pred_all, y_prob_all = [], [], []

    for train_idx, test_idx in loso.split(X, y, groups=groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if len(set(y_tr)) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42,
                                 multi_class="multinomial", solver="lbfgs")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)
        y_pred_all.extend(clf.predict(X_te))
        y_true_all.extend(y_te)
        if return_proba:
            proba = clf.predict_proba(X_te)
            if multiclass:
                y_prob_all.extend(proba.tolist())
            else:
                y_prob_all.extend(proba[:, 1])

    if not y_true_all:
        print(f"    [!] {task_name}: no valid LOSO folds — not enough subjects?")
        return (float("nan"), float("nan")) if not return_proba else (float("nan"), float("nan"), [], [])

    acc = accuracy_score(y_true_all, y_pred_all)
    f1  = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    print(f"    {task_name}: accuracy={acc:.4f}  macro-F1={f1:.4f}")
    if return_proba:
        return acc, f1, y_true_all, y_prob_all
    return acc, f1


# ---------------------------------------------------------------------------
# T2a — 4-class week classification (LOSO)
# ---------------------------------------------------------------------------

def run_t2a_week_classification(embs, weeks, subject_ids):
    print("\n[T2a] Week classification (4-class, LOSO)...")
    n_weeks = len(set(weeks))
    if n_weeks < 2:
        print("    [!] Only one week label — skipping.")
        return {"T2a_accuracy": float("nan"), "T2a_macro_f1": float("nan"),
                "T2a_ovr_auc": float("nan"), "T2a_chance": float("nan")}

    le = LabelEncoder()
    y  = le.fit_transform(weeks)
    acc, f1, y_true, y_prob = _loso_logistic(embs, y, subject_ids, "Week (4-class)",
                                              return_proba=True, multiclass=True)
    try:
        ovr_auc = roc_auc_score(y_true, np.array(y_prob), multi_class="ovr", average="macro")
    except ValueError:
        ovr_auc = float("nan")
    chance = 1.0 / n_weeks
    print(f"    OvR AUC = {ovr_auc:.4f}  |  Chance baseline = {chance:.4f}")
    return {"T2a_accuracy": acc, "T2a_macro_f1": f1, "T2a_ovr_auc": ovr_auc, "T2a_chance": chance}


# ---------------------------------------------------------------------------
# T2b — Binary early vs. late (Week 12 vs. Week 20, LOSO)
# ---------------------------------------------------------------------------

def run_t2b_early_vs_late(embs, weeks, subject_ids):
    print("\n[T2b] Binary early vs. late (Week 12 vs. Week 20, LOSO)...")
    mask = np.isin(weeks, ["Week 12", "Week 20"])
    if mask.sum() < 4:
        print("    [!] Fewer than 4 Week-12/20 samples — skipping.")
        return {"T2b_accuracy": float("nan"), "T2b_auc": float("nan")}

    X_sub  = embs[mask]
    y_sub  = (weeks[mask] == "Week 20").astype(int)  # 1 = late, 0 = early
    grp    = subject_ids[mask]

    acc, _, y_true, y_prob = _loso_logistic(X_sub, y_sub, grp,
                                             "Early vs. Late (binary)",
                                             return_proba=True)
    try:
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")

    print(f"    AUC-ROC = {auc:.4f}  (chance = 0.50)")
    return {"T2b_accuracy": acc, "T2b_auc": auc}


# ---------------------------------------------------------------------------
# T2c — WT vs KO genotype classification (binary LOSO)
# ---------------------------------------------------------------------------

def run_t2c_wt_vs_ko(embs, subject_ids, genotypes):
    print("\n[T2c] WT vs KO genotype classification (binary LOSO)...")
    y = (genotypes == "KO").astype(int)
    n_wt = int((y == 0).sum()); n_ko = int((y == 1).sum())
    print(f"    Class balance: WT={n_wt}, KO={n_ko}")
    if n_wt == 0 or n_ko == 0:
        print("    [!] Only one genotype present — skipping.")
        return {"T2c_accuracy": float("nan"), "T2c_auc": float("nan")}

    acc, _, y_true, y_prob = _loso_logistic(embs, y, subject_ids, "WT vs KO", return_proba=True)
    try:
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    print(f"    AUC-ROC = {auc:.4f}  (chance = 0.50)")
    return {"T2c_accuracy": acc, "T2c_auc": auc}


# ---------------------------------------------------------------------------
# T2d — NaF vs FDG cohort classification (binary LOSO)
# ---------------------------------------------------------------------------

def run_t2d_naf_vs_fdg(embs, subject_ids, cohorts):
    print("\n[T2d] NaF vs FDG cohort classification (binary LOSO)...")
    y = (cohorts == "FDG").astype(int)
    n_naf = int((y == 0).sum()); n_fdg = int((y == 1).sum())
    print(f"    Class balance: NaF={n_naf}, FDG={n_fdg}")
    if n_naf == 0 or n_fdg == 0:
        print("    [!] Only one cohort present — skipping.")
        return {"T2d_accuracy": float("nan"), "T2d_auc": float("nan")}

    acc, _, y_true, y_prob = _loso_logistic(embs, y, subject_ids, "NaF vs FDG", return_proba=True)
    try:
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    print(f"    AUC-ROC = {auc:.4f}  (chance = 0.50)")
    return {"T2d_accuracy": acc, "T2d_auc": auc}


# ---------------------------------------------------------------------------
# T2e — WT vs (KO + stage): 5-class disease-staging task
# ---------------------------------------------------------------------------

def run_t2e_staging(embs, subject_ids, genotypes, weeks):
    print("\n[T2e] WT vs (KO + stage) disease-staging (5-class, LOSO)...")
    stage_labels = np.array([
        "WT" if g == "WT" else f"KO_{w}"
        for g, w in zip(genotypes, weeks)
    ])
    for cls in sorted(set(stage_labels)):
        print(f"    {cls}: {(stage_labels == cls).sum()} samples")

    le = LabelEncoder()
    y  = le.fit_transform(stage_labels)
    acc, f1, y_true, y_prob = _loso_logistic(embs, y, subject_ids, "WT vs KO+stage",
                                              return_proba=True, multiclass=True)
    try:
        ovr_auc = roc_auc_score(y_true, np.array(y_prob), multi_class="ovr", average="macro")
    except ValueError:
        ovr_auc = float("nan")
    print(f"    OvR AUC = {ovr_auc:.4f}  (chance ≈ 0.20)")
    return {"T2e_accuracy": acc, "T2e_macro_f1": f1, "T2e_ovr_auc": ovr_auc}


# ---------------------------------------------------------------------------
# A5 — Per-cohort conditioned analysis
# ---------------------------------------------------------------------------

def run_conditioned_analysis(embs, subject_ids, weeks, cohorts):
    """Run T2a and T2b separately within each cohort to disentangle biological
    signal from scanner/tracer confounds."""
    print("\n[A5] Per-cohort conditioned analysis...")
    results = {}
    loso = LeaveOneGroupOut()

    for cohort in ["NaF", "FDG"]:
        mask_c  = cohorts == cohort
        X_c     = embs[mask_c]
        weeks_c = weeks[mask_c]
        sids_c  = subject_ids[mask_c]
        n_subj  = len(set(sids_c))
        print(f"\n    [{cohort}]  subjects={n_subj}, scans={mask_c.sum()}")

        # T2a conditioned
        le_c = LabelEncoder()
        y_c  = le_c.fit_transform(weeks_c)
        acc_a, f1_a, yt_a, yp_a = _loso_logistic(X_c, y_c, sids_c, f"T2a [{cohort}]",
                                                   return_proba=True, multiclass=True)
        try:
            auc_a = roc_auc_score(yt_a, np.array(yp_a), multi_class="ovr", average="macro")
        except ValueError:
            auc_a = float("nan")
        print(f"      T2a  acc={acc_a:.4f}  F1={f1_a:.4f}  OvR AUC={auc_a:.4f}")

        # T2b conditioned
        mask_el = np.isin(weeks_c, ["Week 12", "Week 20"])
        acc_b, auc_b = float("nan"), float("nan")
        if mask_el.sum() >= 4:
            X_el = X_c[mask_el]
            y_el = (weeks_c[mask_el] == "Week 20").astype(int)
            g_el = sids_c[mask_el]
            acc_b, _, yt_b, yp_b = _loso_logistic(X_el, y_el, g_el, f"T2b [{cohort}]",
                                                    return_proba=True)
            try:
                auc_b = roc_auc_score(yt_b, yp_b) if len(set(yt_b)) > 1 else float("nan")
            except ValueError:
                auc_b = float("nan")
        print(f"      T2b  acc={acc_b:.4f}  AUC={auc_b:.4f}")

        results[cohort] = {
            f"A5_{cohort}_T2a_acc": acc_a, f"A5_{cohort}_T2a_f1": f1_a,
            f"A5_{cohort}_T2a_auc": auc_a, f"A5_{cohort}_T2b_acc": acc_b,
            f"A5_{cohort}_T2b_auc": auc_b,
        }

    merged = {}
    for v in results.values():
        merged.update(v)
    return merged


# ---------------------------------------------------------------------------
# T3a — Pairwise temporal ordering (LOSO)
# ---------------------------------------------------------------------------

def run_t3a_temporal_ordering(embs, weeks, subject_ids):
    """
    For every same-mouse pair at different timepoints, form the signed
    difference vector (emb_later − emb_earlier).  A linear classifier
    trained on these difference vectors should be able to predict direction
    if the embeddings encode a consistent disease-progression direction.
    Both the forward and reversed difference are included so chance = 50%.
    """
    print("\n[T3a] Pairwise temporal ordering (LOSO)...")
    week_to_int = {w: i for i, w in enumerate(WEEK_ORDER)}

    pair_diffs, pair_labels, pair_groups = [], [], []
    for subj in sorted(set(subject_ids)):
        idx = np.where(subject_ids == subj)[0]
        # (week_index, array_index) for this subject, ordered by week
        tps = sorted(
            [(week_to_int[weeks[i]], i) for i in idx if weeks[i] in week_to_int],
            key=lambda x: x[0],
        )
        for (w1, i1), (w2, i2) in combinations(tps, 2):
            if w1 == w2:
                continue
            # (emb_later − emb_earlier) → label 1
            pair_diffs.append(embs[i2] - embs[i1])
            pair_labels.append(1)
            pair_groups.append(subj)
            # reversed → label 0
            pair_diffs.append(embs[i1] - embs[i2])
            pair_labels.append(0)
            pair_groups.append(subj)

    n_pairs = len(pair_diffs) // 2
    if n_pairs < 2:
        print("    [!] Too few longitudinal pairs — skipping (need ≥2 pairs from ≥2 subjects).")
        return {"T3a_pairwise_ordering_acc": float("nan")}

    X      = np.array(pair_diffs, dtype=np.float32)
    y      = np.array(pair_labels)
    groups = np.array(pair_groups)

    loso = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in loso.split(X, y, groups=groups):
        if len(set(y[train_idx])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver="lbfgs")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X[train_idx], y[train_idx])
        y_pred_all.extend(clf.predict(X[test_idx]))
        y_true_all.extend(y[test_idx])

    if not y_true_all:
        print("    [!] No valid LOSO folds.")
        return {"T3a_pairwise_ordering_acc": float("nan")}

    acc = accuracy_score(y_true_all, y_pred_all)
    print(f"    Pairwise ordering accuracy = {acc:.4f}  ({n_pairs} pairs, chance = 0.50)")
    return {"T3a_pairwise_ordering_acc": acc}


# ---------------------------------------------------------------------------
# T3b — Same-subject retrieval
# ---------------------------------------------------------------------------

def run_t3b_subject_retrieval(embs, subject_ids):
    print("\n[T3b] Same-subject retrieval...")
    cos_sim = normalize(embs) @ normalize(embs).T  # (N, N)
    n = len(subject_ids)

    r1_hits, r3_hits, rr_list = [], [], []
    for i in range(n):
        sims = cos_sim[i].copy()
        sims[i] = -np.inf  # exclude self
        ranked      = np.argsort(-sims)
        same_subj   = subject_ids[ranked] == subject_ids[i]

        r1_hits.append(int(same_subj[0]))
        r3_hits.append(int(same_subj[:3].any()))
        first_hit = int(np.argmax(same_subj)) + 1 if same_subj.any() else n
        rr_list.append(1.0 / first_hit)

    r1  = float(np.mean(r1_hits))
    r3  = float(np.mean(r3_hits))
    mrr = float(np.mean(rr_list))
    print(f"    Recall@1 = {r1:.4f}")
    print(f"    Recall@3 = {r3:.4f}")
    print(f"    MRR      = {mrr:.4f}")
    return {"T3b_recall_at_1": r1, "T3b_recall_at_3": r3, "T3b_mrr": mrr}


# ---------------------------------------------------------------------------
# T3c — Same-week retrieval (mAP@5)
# ---------------------------------------------------------------------------

def _ap_at_k(relevant_mask, k):
    """Average Precision@k for one query (relevant_mask is boolean, ordered by rank)."""
    rel = relevant_mask[:k].astype(float)
    if rel.sum() == 0:
        return 0.0
    cum_hits = np.cumsum(rel)
    precision_at_hits = cum_hits / np.arange(1, len(rel) + 1)
    return float((precision_at_hits * rel).sum() / rel.sum())


def run_t3c_week_retrieval(embs, weeks, k=5):
    print(f"\n[T3c] Same-week retrieval (mAP@{k})...")
    if len(set(weeks)) < 2:
        print("    [!] Only one week label — skipping.")
        return {f"T3c_map_at_{k}": float("nan")}

    cos_sim = normalize(embs) @ normalize(embs).T
    n = len(weeks)

    ap_list = []
    for i in range(n):
        sims = cos_sim[i].copy()
        sims[i] = -np.inf
        ranked  = np.argsort(-sims)
        rel     = weeks[ranked] == weeks[i]
        ap_list.append(_ap_at_k(rel, k))

    map_k = float(np.mean(ap_list))
    print(f"    mAP@{k} = {map_k:.4f}  (fraction of top-{k} neighbors sharing the same week)")
    return {f"T3c_map_at_{k}": map_k}


# ---------------------------------------------------------------------------
# Output: metrics.csv and report.txt
# ---------------------------------------------------------------------------

def save_metrics(metrics, output_dir, npz_path):
    row = {"source_npz": npz_path, **metrics}
    out = os.path.join(output_dir, "metrics.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"\n[+] Saved metrics  → {out}")


def save_report(metrics, output_dir, npz_path, n_samples, emb_dim):
    m = metrics  # shorthand

    lines = [
        "=" * 62,
        "  EMBEDDING EVALUATION REPORT",
        "=" * 62,
        f"  Source : {npz_path}",
        f"  Samples: {n_samples}   Embedding dim: {emb_dim}",
        "",
        "  TIER 1 — UNSUPERVISED",
        "-" * 62,
        f"  T1b  ARI  (k-means vs week)         : {_fmt(m.get('T1b_ARI'))}  [0=random, 1=perfect]",
        f"  T1b  NMI  (k-means vs week)         : {_fmt(m.get('T1b_NMI'))}  [0=random, 1=perfect]",
        f"  T1c  Silhouette (cosine, by week)   : {_fmt(m.get('T1c_silhouette_cosine'))}  [>0 = weeks cluster]",
        f"  T1d  Intra-subject cosine sim       : {_fmt(m.get('T1d_intra_subject_sim'))}",
        f"  T1d  Inter-subject cosine sim       : {_fmt(m.get('T1d_inter_subject_sim'))}",
        f"  T1d  Δ (intra − inter)              : {_fmt(m.get('T1d_delta'))}  [>0 = identity preserved]",
        "",
        "  TIER 2 — LINEAR PROBE  (Leave-One-Subject-Out CV)",
        "-" * 62,
        f"  T2a  Week (4-class) accuracy        : {_fmt(m.get('T2a_accuracy'))}  (chance={_fmt(m.get('T2a_chance'), '.2f')})",
        f"  T2a  Week (4-class) macro-F1        : {_fmt(m.get('T2a_macro_f1'))}",
        f"  T2a  Week (4-class) OvR AUC         : {_fmt(m.get('T2a_ovr_auc'))}",
        f"  T2b  Early vs Late accuracy         : {_fmt(m.get('T2b_accuracy'))}  (chance=0.50)",
        f"  T2b  Early vs Late AUC-ROC          : {_fmt(m.get('T2b_auc'))}",
        f"  T2c  WT vs KO accuracy              : {_fmt(m.get('T2c_accuracy'))}  (chance=0.50)",
        f"  T2c  WT vs KO AUC-ROC               : {_fmt(m.get('T2c_auc'))}",
        f"  T2d  NaF vs FDG accuracy            : {_fmt(m.get('T2d_accuracy'))}  (chance=0.50)",
        f"  T2d  NaF vs FDG AUC-ROC             : {_fmt(m.get('T2d_auc'))}",
        f"  T2e  WT vs KO+stage accuracy        : {_fmt(m.get('T2e_accuracy'))}  (chance≈0.20)",
        f"  T2e  WT vs KO+stage macro-F1        : {_fmt(m.get('T2e_macro_f1'))}",
        f"  T2e  WT vs KO+stage OvR AUC         : {_fmt(m.get('T2e_ovr_auc'))}",
        "",
        "  CONDITIONED ANALYSIS (per cohort)",
        "-" * 62,
    ] + [
        f"  [NaF]  T2a acc={_fmt(m.get('A5_NaF_T2a_acc'))}  F1={_fmt(m.get('A5_NaF_T2a_f1'))}  AUC={_fmt(m.get('A5_NaF_T2a_auc'))}",
        f"         T2b acc={_fmt(m.get('A5_NaF_T2b_acc'))}  AUC={_fmt(m.get('A5_NaF_T2b_auc'))}",
        f"  [FDG]  T2a acc={_fmt(m.get('A5_FDG_T2a_acc'))}  F1={_fmt(m.get('A5_FDG_T2a_f1'))}  AUC={_fmt(m.get('A5_FDG_T2a_auc'))}",
        f"         T2b acc={_fmt(m.get('A5_FDG_T2b_acc'))}  AUC={_fmt(m.get('A5_FDG_T2b_auc'))}",
        "",
        "  TIER 3 — LONGITUDINAL / RELATIONAL",
        "-" * 62,
        f"  T3a  Pairwise temporal ordering     : {_fmt(m.get('T3a_pairwise_ordering_acc'))}  (chance=0.50)",
        f"  T3b  Subject retrieval Recall@1     : {_fmt(m.get('T3b_recall_at_1'))}",
        f"  T3b  Subject retrieval Recall@3     : {_fmt(m.get('T3b_recall_at_3'))}",
        f"  T3b  Subject retrieval MRR          : {_fmt(m.get('T3b_mrr'))}",
        f"  T3c  Week retrieval mAP@5           : {_fmt(m.get('T3c_map_at_5'))}",
        "",
        "  PLOTS: see plots/ subdirectory",
        "=" * 62,
    ]

    out = os.path.join(output_dir, "report.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] Saved report   → {out}")
    print()
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Encoder-agnostic embedding evaluation suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--embeddings", required=True,
                        help="Path to .npz embeddings file (standard interface).")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write metrics.csv, report.txt, and plots/.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    embs, subject_ids, weeks, modalities, paths = load_embeddings(args.embeddings)

    # Parse cohort / genotype from subject_id strings (e.g. "NaF_WT_03")
    cohorts   = np.array([s.split("_")[0] for s in subject_ids])
    genotypes = np.array([s.split("_")[1] for s in subject_ids])

    all_metrics = {}

    # Tier 1 — unsupervised
    all_metrics.update(run_t1a_plots(embs, weeks, subject_ids, plots_dir))
    all_metrics.update(run_t1b_cluster_alignment(embs, weeks))
    all_metrics.update(run_t1c_silhouette(embs, weeks))
    all_metrics.update(run_t1d_subject_consistency(embs, subject_ids))

    # Tier 2 — linear probe
    all_metrics.update(run_t2a_week_classification(embs, weeks, subject_ids))
    all_metrics.update(run_t2b_early_vs_late(embs, weeks, subject_ids))
    all_metrics.update(run_t2c_wt_vs_ko(embs, subject_ids, genotypes))
    all_metrics.update(run_t2d_naf_vs_fdg(embs, subject_ids, cohorts))
    all_metrics.update(run_t2e_staging(embs, subject_ids, genotypes, weeks))
    all_metrics.update(run_conditioned_analysis(embs, subject_ids, weeks, cohorts))

    # Tier 3 — longitudinal
    all_metrics.update(run_t3a_temporal_ordering(embs, weeks, subject_ids))
    all_metrics.update(run_t3b_subject_retrieval(embs, subject_ids))
    all_metrics.update(run_t3c_week_retrieval(embs, weeks, k=5))

    save_metrics(all_metrics, args.output_dir, args.embeddings)
    save_report(all_metrics, args.output_dir, args.embeddings, len(embs), embs.shape[1])


if __name__ == "__main__":
    main()
