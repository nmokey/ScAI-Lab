"""
train_longitudinal.py

One-step longitudinal prediction using frozen RAD-DINO embeddings.

Trains a model to predict the T_{k+1} embedding from T_k, evaluated under
Leave-One-Subject-Out (LOSO) CV. All consecutive timepoint pairs are used
(Week 12→15, 15→18, 18→20), giving 129 pairs across 78 subjects.

The key evaluation is nearest-neighbor retrieval: for each predicted T_{k+1},
rank it against all actual T_{k+1} embeddings by cosine distance. If the
prediction is subject-specific, the correct subject should rank near #1.

Tasks
-----
  T4a  Prediction quality          — cosine similarity (predicted vs. actual T_{k+1})
  T4b  Nearest-neighbor retrieval  — Recall@1, Recall@3, MRR against actual T_{k+1} pool
  T4c  Trajectory direction        — is predicted T_{k+1} closer to actual T_{k+1}
                                     than to T_k? (directional improvement rate)

Usage
-----
    python scripts/train_longitudinal.py --dry-run
    python scripts/train_longitudinal.py
    python scripts/train_longitudinal.py --linear --no-conditioning
"""

import argparse
import csv
import os
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import yaml
except ImportError:
    yaml = None

WEEK_ORDER = ["Week 12", "Week 15", "Week 18", "Week 20"]
WEEK_COLORS = {
    "Week 12": "#2196F3",
    "Week 15": "#4CAF50",
    "Week 18": "#FF9800",
    "Week 20": "#F44336",
}


# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    if yaml is None or not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args():
    cfg = load_config()
    output_dir  = cfg.get("paths", {}).get("output_dir", ".")
    default_emb = os.path.join(output_dir, "embeddings", "raddino", "raddino_embeddings.npz")
    default_out = os.path.join(output_dir, "embeddings", "longitudinal")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings", default=default_emb)
    p.add_argument("--output-dir", default=default_out)
    p.add_argument("--dry-run", action="store_true",
                   help="Print dataset stats and exit without training")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--hidden", type=int, default=512,
                   help="Hidden dim of MLP (ignored with --linear)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--linear", action="store_true",
                   help="Single linear layer instead of MLP")
    p.add_argument("--no-conditioning", action="store_true",
                   help="Disable genotype+cohort+step one-hot conditioning")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings(path, cohort_filter=None):
    """Load embeddings. cohort_filter: if set (e.g. "NaF"), keep only that cohort."""
    data        = np.load(path, allow_pickle=True)
    embs        = data["embeddings"].astype(np.float32)
    subject_ids = data["subject_ids"].astype(str)
    weeks       = data["weeks"].astype(str)
    if cohort_filter is not None:
        mask = np.array([sid.startswith(cohort_filter) for sid in subject_ids])
        embs, subject_ids, weeks = embs[mask], subject_ids[mask], weeks[mask]
        print(f"[i] Cohort filter '{cohort_filter}': kept {mask.sum()}/{len(mask)} embeddings")
    print(f"[i] Loaded {len(embs)} embeddings  (dim={embs.shape[1]})")
    print(f"    Subjects  : {len(set(subject_ids))} unique")
    print(f"    Weeks     : {sorted(set(weeks))}")
    return embs, subject_ids, weeks


def parse_subject_meta(subject_id):
    """'NaF_WT_03' → (cohort='NaF', genotype='WT')"""
    parts    = subject_id.split("_")
    cohort   = parts[0] if len(parts) >= 1 else "unknown"
    genotype = parts[1] if len(parts) >= 2 else "unknown"
    return cohort, genotype


def build_pairs(embs, subject_ids, weeks, use_conditioning=True):
    """
    Build all consecutive (T_k, T_{k+1}) pairs across all subjects.

    Returns
    -------
    X        : (N, D[+C]) float32  — T_k embedding [+ conditioning]
    Y        : (N, D)     float32  — T_{k+1} embedding (target)
    subjects : (N,)       str      — subject ID (LOSO group key)
    genotypes: (N,)       int      — 0=WT, 1=KO
    meta     : list of dicts       — sid, week_from, week_to, cohort, genotype per pair
    """
    # Index: subject → week → embedding
    idx = {}
    for i, (sid, wk) in enumerate(zip(subject_ids, weeks)):
        idx.setdefault(str(sid), {})[str(wk)] = embs[i]

    # One-hot step encoding: which of the 3 possible transitions is this?
    step_to_onehot = {
        ("Week 12", "Week 15"): np.array([1, 0, 0], dtype=np.float32),
        ("Week 15", "Week 18"): np.array([0, 1, 0], dtype=np.float32),
        ("Week 18", "Week 20"): np.array([0, 0, 1], dtype=np.float32),
    }

    X_list, Y_list, sids, geno_list, meta = [], [], [], [], []

    for sid in sorted(idx):
        week_map = idx[sid]
        cohort, genotype = parse_subject_meta(sid)

        # One-hot encoding: [KO, WT] and [NaF, FDG] — index 0 is the positive class.
        # Note: scalar genotype elsewhere uses 1=KO; this one-hot is consistent (KO→[1,0]).
        geno_vec   = np.array([1, 0], dtype=np.float32) if genotype == "KO" else np.array([0, 1], dtype=np.float32)
        cohort_vec = np.array([1, 0], dtype=np.float32) if cohort   == "NaF" else np.array([0, 1], dtype=np.float32)

        for w0, w1 in zip(WEEK_ORDER, WEEK_ORDER[1:]):
            if w0 not in week_map or w1 not in week_map:
                continue

            tk  = week_map[w0]
            tk1 = week_map[w1]

            if use_conditioning:
                step_vec = step_to_onehot[(w0, w1)]
                x = np.concatenate([tk, geno_vec, cohort_vec, step_vec])
            else:
                x = tk

            X_list.append(x)
            Y_list.append(tk1)
            sids.append(sid)
            geno_list.append(1 if genotype == "KO" else 0)
            meta.append({"sid": sid, "week_from": w0, "week_to": w1,
                         "cohort": cohort, "genotype": genotype})

    X         = np.stack(X_list).astype(np.float32)
    Y         = np.stack(Y_list).astype(np.float32)
    subjects  = np.array(sids, dtype=str)
    genotypes = np.array(geno_list, dtype=int)
    return X, Y, subjects, genotypes, meta


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class LinearPredictor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.net(x)


def cosine_loss(pred, target):
    pred_n   = nn.functional.normalize(pred, dim=-1)
    target_n = nn.functional.normalize(target, dim=-1)
    return 1.0 - (pred_n * target_n).sum(dim=-1).mean()


def train_fold(X_train, Y_train, in_dim, out_dim, hidden_dim, lr, epochs, device, linear=False):
    model = (LinearPredictor(in_dim, out_dim) if linear
             else MLPPredictor(in_dim, hidden_dim, out_dim)).to(device)
    opt = Adam(model.parameters(), lr=lr)

    Xt = torch.tensor(X_train, device=device)
    Yt = torch.tensor(Y_train, device=device)
    dl = DataLoader(TensorDataset(Xt, Yt), batch_size=min(32, len(Xt)), shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            cosine_loss(model(xb), yb).backward()
            opt.step()

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def run_t4a(Y_true, Y_pred, meta):
    """
    T4a — Prediction quality: mean cosine similarity (predicted vs. actual T_{k+1}).
    Also broken down by step type.
    """
    print("\n[T4a] One-step prediction quality (cosine similarity)...")
    sims = [cosine_sim(Y_true[i], Y_pred[i]) for i in range(len(Y_true))]
    overall = float(np.mean(sims))
    print(f"    Overall mean cosine sim = {overall:.4f}")

    by_step = {}
    for step in [("Week 12", "Week 15"), ("Week 15", "Week 18"), ("Week 18", "Week 20")]:
        mask = [i for i, m in enumerate(meta) if (m["week_from"], m["week_to"]) == step]
        if mask:
            s = float(np.mean([sims[i] for i in mask]))
            label = f"{step[0].replace('Week ', 'W')}→{step[1].replace('Week ', 'W')}"
            by_step[label] = s
            print(f"    {label}: {s:.4f}  (n={len(mask)})")

    result = {"T4a_cosine_sim": overall}
    for label, val in by_step.items():
        result[f"T4a_{label.replace('→', '_')}"] = val
    return result


def run_t4b(Y_true, Y_pred, subjects, meta):
    """
    T4b — Nearest-neighbor retrieval.

    For each predicted T_{k+1}, rank it against all *actual* T_{k+1} embeddings
    from the same step type (e.g., all actual Week 15 embeddings when predicting
    Week 15). The correct target is the held-out subject's actual embedding.

    Metrics: Recall@1, Recall@3, MRR — same as T3b in evaluate_embeddings.py.
    This is immune to the genotype-separation leakage issue: we never train a
    classifier on the predicted embeddings.
    """
    print("\n[T4b] Nearest-neighbor retrieval (predicted → actual T_{k+1} pool)...")

    # Index actual T_{k+1} embeddings by (subject, week_to)
    actual_pool = {}  # (sid, week_to) → embedding
    for i, m in enumerate(meta):
        actual_pool[(m["sid"], m["week_to"])] = Y_true[i]

    r1_list, r3_list, mrr_list = [], [], []

    for i, m in enumerate(meta):
        query_pred  = Y_pred[i]
        correct_sid = m["sid"]
        week_to     = m["week_to"]

        # Pool: all actual T_{k+1} embeddings for this step (including the correct one)
        pool_sids = [sid for (sid, wt) in actual_pool if wt == week_to]
        pool_embs = np.stack([actual_pool[(sid, week_to)] for sid in pool_sids])

        # Cosine similarities between predicted and all pool embeddings
        norms      = np.linalg.norm(pool_embs, axis=1, keepdims=True) + 1e-8
        pool_normd = pool_embs / norms
        pred_normd = query_pred / (np.linalg.norm(query_pred) + 1e-8)
        sims       = pool_normd @ pred_normd  # (pool_size,)

        # Rank descending (highest sim = rank 1)
        order        = np.argsort(-sims)
        ranked_sids  = [pool_sids[j] for j in order]
        rank         = ranked_sids.index(correct_sid) + 1  # 1-indexed

        r1_list.append(1 if rank == 1 else 0)
        r3_list.append(1 if rank <= 3 else 0)
        mrr_list.append(1.0 / rank)

    r1  = float(np.mean(r1_list))
    r3  = float(np.mean(r3_list))
    mrr = float(np.mean(mrr_list))
    n   = len(r1_list)
    # Chance varies by step because pool size differs per week_to.
    pool_sizes = {wt: len(set(sid for (sid, wt2) in actual_pool if wt2 == wt))
                  for wt in set(m["week_to"] for m in meta)}
    avg_pool_size = sum(pool_sizes.values()) / len(pool_sizes)
    chance_r1 = 1.0 / avg_pool_size

    print(f"    Recall@1 = {r1:.4f}  (chance ≈ {chance_r1:.3f}, avg pool size={avg_pool_size:.0f})")
    print(f"    Recall@3 = {r3:.4f}")
    print(f"    MRR      = {mrr:.4f}")
    print(f"    n pairs  = {n}")
    return {"T4b_recall_at_1": r1, "T4b_recall_at_3": r3, "T4b_mrr": mrr}


def run_t4c(Y_true, Y_pred, X, meta, use_conditioning):
    """
    T4c — Directional improvement rate.

    For each pair, is the predicted T_{k+1} closer (cosine) to the actual T_{k+1}
    than the raw T_k is? If yes, the model is adding useful signal beyond simply
    returning the input. Rate > 0.5 means the predictor is net-positive.
    """
    print("\n[T4c] Directional improvement rate (pred closer to target than T_k?)...")

    improved = []
    for i, m in enumerate(meta):
        # Extract the raw T_k embedding (first 768 dims of X, regardless of conditioning)
        tk      = X[i, :768]
        target  = Y_true[i]
        pred    = Y_pred[i]

        sim_tk   = cosine_sim(tk, target)
        sim_pred = cosine_sim(pred, target)
        improved.append(1 if sim_pred > sim_tk else 0)

    rate = float(np.mean(improved))
    print(f"    Improvement rate = {rate:.4f}  (0.5 = no better than returning T_k)")
    return {"T4c_improvement_rate": rate}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_retrieval_rank_hist(Y_true, Y_pred, subjects, meta, plots_dir):
    """Histogram of retrieval ranks for each predicted T_{k+1}."""
    actual_pool = {}
    for i, m in enumerate(meta):
        actual_pool[(m["sid"], m["week_to"])] = Y_true[i]

    ranks = []
    for i, m in enumerate(meta):
        week_to    = m["week_to"]
        pool_sids  = [sid for (sid, wt) in actual_pool if wt == week_to]
        pool_embs  = np.stack([actual_pool[(sid, week_to)] for sid in pool_sids])
        norms      = np.linalg.norm(pool_embs, axis=1, keepdims=True) + 1e-8
        pool_normd = pool_embs / norms
        pred_normd = Y_pred[i] / (np.linalg.norm(Y_pred[i]) + 1e-8)
        sims       = pool_normd @ pred_normd
        order      = np.argsort(-sims)
        rank       = [pool_sids[j] for j in order].index(m["sid"]) + 1
        ranks.append(rank)

    fig, ax = plt.subplots(figsize=(7, 4))
    max_rank = max(ranks)
    ax.hist(ranks, bins=range(1, max_rank + 2), align="left",
            color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(1, color="red", linestyle="--", alpha=0.6, label="Rank 1")
    r1 = sum(1 for r in ranks if r == 1) / len(ranks)
    ax.set_title(f"T4b — Retrieval Rank Distribution  (Recall@1 = {r1:.3f})", fontsize=11)
    ax.set_xlabel("Rank of correct subject in T_{k+1} pool")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(plots_dir, "t4b_retrieval_rank_hist.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [+] {os.path.basename(out)}")


def plot_cosine_improvement(Y_true, Y_pred, X, meta, plots_dir):
    """Scatter: cosine(T_k, T_{k+1}) vs cosine(pred, T_{k+1}) per pair."""
    tk_sims, pred_sims, labels = [], [], []
    step_colors = {
        "Week 12→Week 15": "#2196F3",
        "Week 15→Week 18": "#4CAF50",
        "Week 18→Week 20": "#F44336",
    }
    for i, m in enumerate(meta):
        tk = X[i, :768]
        tk_sims.append(cosine_sim(tk, Y_true[i]))
        pred_sims.append(cosine_sim(Y_pred[i], Y_true[i]))
        labels.append(f"{m['week_from']}→{m['week_to']}")

    fig, ax = plt.subplots(figsize=(6, 6))
    for step, color in step_colors.items():
        mask = [i for i, l in enumerate(labels) if l == step]
        if mask:
            ax.scatter([tk_sims[i] for i in mask], [pred_sims[i] for i in mask],
                       label=step.replace("Week ", "W"), color=color, s=40, alpha=0.75)
    lims = [min(tk_sims + pred_sims) - 0.005, max(tk_sims + pred_sims) + 0.005]
    ax.plot(lims, lims, "k--", alpha=0.4, label="y = x (no change)")
    ax.set_xlabel("cosine(T_k,  actual T_{k+1})")
    ax.set_ylabel("cosine(pred, actual T_{k+1})")
    ax.set_title("T4c — Does the predictor improve on returning T_k?", fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(plots_dir, "t4c_cosine_improvement.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [+] {os.path.basename(out)}")


def plot_umap(embs, weeks, subject_ids, Y_pred, meta, plots_dir):
    """UMAP of all actual embeddings with predicted T_{k+1} overlaid."""
    if not UMAP_AVAILABLE:
        print("    [!] umap-learn not available — skipping UMAP.")
        return

    pred_arr = np.stack(Y_pred)
    all_vecs = np.concatenate([embs, pred_arr], axis=0)
    n_neighbors = min(15, max(2, len(all_vecs) - 1))
    coords = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42).fit_transform(all_vecs)

    actual_coords = coords[:len(embs)]
    pred_coords   = coords[len(embs):]

    fig, ax = plt.subplots(figsize=(9, 7))
    for wk in WEEK_ORDER:
        mask = np.array(list(weeks)) == wk
        if mask.any():
            ax.scatter(actual_coords[mask, 0], actual_coords[mask, 1],
                       label=wk, s=40, alpha=0.65, color=WEEK_COLORS[wk])
    ax.scatter(pred_coords[:, 0], pred_coords[:, 1],
               marker="*", s=120, color="black", zorder=5, alpha=0.7,
               label="Predicted T_{k+1}")
    ax.legend(title="Week / Predicted", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_title("UMAP — Actual embeddings + Predicted T_{k+1} (★)")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    out = os.path.join(plots_dir, "umap_predicted_step.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [+] {os.path.basename(out)}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_predicted_embeddings(Y_pred, meta, output_dir):
    """
    Save LOSO-predicted T_{k+1} embeddings as .npy files, one per (subject, week_to).

    Files are written to output_dir/predicted_embeddings/<sid>_<week_tag>.npy.
    These are used by the VLM dataset to provide the longitudinal encoder's
    predictions as additional image tokens alongside the observed ts0 embedding.
    """
    emb_dir = os.path.join(output_dir, "predicted_embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    week_tag = {"Week 12": "ts0", "Week 15": "ts1", "Week 18": "ts2", "Week 20": "ts3"}
    saved = 0
    for i, m in enumerate(meta):
        tag  = week_tag.get(m["week_to"], m["week_to"].replace(" ", "_").lower())
        path = os.path.join(emb_dir, f"{m['sid']}_{tag}.npy")
        np.save(path, Y_pred[i])
        saved += 1

    print(f"[+] Saved {saved} predicted embeddings → {emb_dir}/")


def save_metrics(metrics, output_dir):
    path = os.path.join(output_dir, "longitudinal_metrics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
        w.writeheader()
        w.writerow(metrics)
    print(f"[+] Saved metrics → {path}")


def save_report(metrics, output_dir, n_pairs, n_subjects, model_desc, conditioning):
    m    = metrics
    path = os.path.join(output_dir, "longitudinal_report.txt")
    lines = [
        "=" * 60,
        "Longitudinal One-Step Prediction Report",
        "=" * 60,
        f"Task              : T_k → T_{{k+1}} (one-step prediction)",
        f"Pairs             : {n_pairs}  (across {n_subjects} subjects)",
        f"Model             : {model_desc}",
        f"Conditioning      : {conditioning}",
        f"CV                : Leave-One-Subject-Out (LOSO)",
        "",
        "[T4a] Prediction Quality (cosine similarity to actual T_{{k+1}})",
        f"  Overall          : {m.get('T4a_cosine_sim', float('nan')):.4f}",
        f"  W12→W15          : {m.get('T4a_W12_W15', float('nan')):.4f}",
        f"  W15→W18          : {m.get('T4a_W15_W18', float('nan')):.4f}",
        f"  W18→W20          : {m.get('T4a_W18_W20', float('nan')):.4f}",
        "",
        "[T4b] Nearest-Neighbor Retrieval (predicted → actual T_{{k+1}} pool)",
        f"  Recall@1         : {m.get('T4b_recall_at_1', float('nan')):.4f}",
        f"  Recall@3         : {m.get('T4b_recall_at_3', float('nan')):.4f}",
        f"  MRR              : {m.get('T4b_mrr', float('nan')):.4f}",
        "",
        "[T4c] Directional Improvement Rate",
        f"  Rate             : {m.get('T4c_improvement_rate', float('nan')):.4f}  (>0.5 = net positive)",
        "=" * 60,
    ]
    text = "\n".join(lines)
    print("\n" + text)
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"[+] Saved report → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.exists(args.embeddings):
        raise FileNotFoundError(
            f"Embeddings not found: {args.embeddings}\n"
            "Run scripts/get_raddino_embeddings.py first."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    use_conditioning = not args.no_conditioning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Device: {device}")

    embs, subject_ids, weeks = load_embeddings(args.embeddings, cohort_filter="NaF")

    X, Y, subjects, genotypes, meta = build_pairs(
        embs, subject_ids, weeks, use_conditioning=use_conditioning
    )

    n_pairs    = len(X)
    n_subjects = len(set(subjects))
    print(f"\n[i] Consecutive pairs: {n_pairs}  ({n_subjects} subjects)")
    from collections import Counter
    for step, cnt in sorted(Counter((m["week_from"], m["week_to"]) for m in meta).items()):
        print(f"    {step[0]} → {step[1]}: {cnt}")

    if args.dry_run:
        print("\n[dry-run] Exiting without training.")
        return

    in_dim  = X.shape[1]
    out_dim = Y.shape[1]
    model_desc   = "linear" if args.linear else f"MLP (hidden={args.hidden})"
    cond_desc    = "genotype + cohort + step" if use_conditioning else "none (imaging only)"

    # ---- LOSO CV ----
    print(f"\n[i] LOSO CV — {n_subjects} subjects, {args.epochs} epochs per fold...")
    logo   = LeaveOneGroupOut()
    Y_pred = np.zeros_like(Y)

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, Y, subjects)):
        model = train_fold(
            X[train_idx], Y[train_idx],
            in_dim=in_dim, out_dim=out_dim,
            hidden_dim=args.hidden, lr=args.lr,
            epochs=args.epochs, device=device,
            linear=args.linear,
        )
        with torch.no_grad():
            pred = model(torch.tensor(X[test_idx], device=device)).cpu().numpy()
        Y_pred[test_idx] = pred

        if (fold + 1) % 10 == 0 or fold == n_subjects - 1:
            print(f"    Fold {fold+1:3d}/{n_subjects}")

    # ---- Evaluation ----
    metrics = {}
    metrics.update(run_t4a(Y, Y_pred, meta))
    metrics.update(run_t4b(Y, Y_pred, subjects, meta))
    metrics.update(run_t4c(Y, Y_pred, X, meta, use_conditioning))

    # ---- Plots ----
    print("\n[Plots]")
    plot_retrieval_rank_hist(Y, Y_pred, subjects, meta, plots_dir)
    plot_cosine_improvement(Y, Y_pred, X, meta, plots_dir)
    plot_umap(embs, weeks, subject_ids, list(Y_pred), meta, plots_dir)

    save_predicted_embeddings(Y_pred, meta, args.output_dir)
    save_metrics(metrics, args.output_dir)
    save_report(metrics, args.output_dir, n_pairs, n_subjects, model_desc, cond_desc)


if __name__ == "__main__":
    main()
