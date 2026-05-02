"""
extract_tbr_features.py

Extract aortic target-to-background ratio (TBR) approximations from
PET NIfTI volumes, and evaluate whether RAD-DINO embeddings predict
longitudinal TBR signal.

Background
----------
The source paper (Tamboline et al. 2025) computed TBR from manually drawn
aortic ROIs, achieving r²=0.83 with histology for NaF PET. Replicating
this programmatically is limited by:

  1. PET resolution: 0.54mm/voxel — the mouse aorta (~1mm diameter) spans
     only ~2 voxels. Partial-volume effects dominate.
  2. No registered atlas or trained segmentation model for this scanner/protocol.
  3. Dominant hotspots (bone, bladder) overwhelm whole-crop statistics.

This script implements THREE TBR approximation strategies of increasing
anatomical specificity, computes them for all available PET NIfTIs, and
evaluates how well RAD-DINO CT embeddings correlate with / predict each.

Strategies
----------
  TBR-1  Whole-crop: P99 / median (nonzero voxels) — simplest baseline
  TBR-2  Mid-Z band: restrict to Z 33–67% (abdominal region), exclude
          top-5% hotspots (bone/bladder leakage), P95/median
  TBR-3  Spine-anchored ROI: locate the vertebral column via CT (largest
          high-HU connected component in mid-Z), place a 3mm-radius cylinder
          3mm anterior to spine centroid in each axial slice, average PET
          signal in that cylinder vs. background. Most anatomically targeted
          but most sensitive to spine detection failures.

Evaluation
----------
For each TBR strategy:
  E1  KO vs WT separation — Mann-Whitney U, AUC-ROC (per week)
  E2  Longitudinal trend — Spearman ρ between week rank and TBR (per subject)
  E3  Embedding correlation — Spearman ρ between RAD-DINO embedding PC1
      and TBR across all (subject, week) pairs
  E4  Embedding prediction — can a LOSO linear regression from the embedding
      predict TBR? R², MAE reported.

Usage
-----
    python scripts/extract_tbr_features.py
    python scripts/extract_tbr_features.py --strategy 2
    python scripts/extract_tbr_features.py --no-eval   # features only
"""

import argparse
import csv
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel required: pip install nibabel")

try:
    import yaml
except ImportError:
    yaml = None

WEEK_ORDER  = ["week_12", "week_15", "week_18", "week_20"]
WEEK_LABELS = ["Week 12", "Week 15", "Week 18", "Week 20"]
WEEK_COLORS = {"Week 12": "#2196F3", "Week 15": "#4CAF50",
               "Week 18": "#FF9800", "Week 20": "#F44336"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    if yaml is None or not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args():
    cfg        = load_config()
    output_dir = cfg.get("paths", {}).get("output_dir", ".")
    mice_dir   = str(Path(output_dir) / "mice")
    emb_path   = str(Path(output_dir) / "embeddings" / "raddino" / "raddino_embeddings.npz")
    out_dir    = str(Path(output_dir) / "embeddings" / "longitudinal")

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mice-dir",    default=mice_dir,  help="Path to processed mice/ directory")
    p.add_argument("--embeddings",  default=emb_path,  help="Path to raddino_embeddings.npz")
    p.add_argument("--output-dir",  default=out_dir,   help="Output directory for features and plots")
    p.add_argument("--strategy",    type=int, default=0,
                   help="TBR strategy to use: 0=all, 1=whole-crop, 2=mid-Z band, 3=spine-anchored")
    p.add_argument("--no-eval",     action="store_true", help="Extract features only, skip evaluation")
    p.add_argument("--cohort",      default="NaF", choices=["NaF", "FDG"],
                   help="Which PET cohort to process (default: NaF)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# TBR extraction strategies
# ---------------------------------------------------------------------------

def tbr_strategy_1(pet_data):
    """
    TBR-1: Whole-crop P99 / median.
    Simplest possible. Dominated by bone/bladder hotspots.
    Included as a baseline to show why it fails.
    """
    nz = pet_data[pet_data > 0].flatten()
    if len(nz) < 50:
        return {"tbr1_p99_median": np.nan, "tbr1_mean": np.nan}
    return {
        "tbr1_p99_median": float(np.percentile(nz, 99) / np.median(nz)),
        "tbr1_mean":       float(nz.mean()),
    }


def tbr_strategy_2(pet_data):
    """
    TBR-2: Mid-Z band (Z 33–67%), hotspot-excluded P95/median.
    Restricts to abdominal region; excludes skull (top) and bladder (bottom).
    Hotspot exclusion (top 5%) removes residual bone uptake.
    """
    nz  = pet_data.shape[2]
    z0  = nz // 3
    z1  = 2 * nz // 3
    mid = pet_data[:, :, z0:z1].flatten()
    mid = mid[mid > 0]
    if len(mid) < 50:
        return {"tbr2_p95_median": np.nan, "tbr2_clean_mean": np.nan, "tbr2_p95": np.nan}
    thresh     = np.percentile(mid, 95)
    mid_clean  = mid[mid < thresh]
    median_val = float(np.median(mid_clean)) if len(mid_clean) > 0 else np.nan
    p95_val    = float(np.percentile(mid, 95))
    return {
        "tbr2_p95_median":  p95_val / median_val if median_val and median_val > 0 else np.nan,
        "tbr2_clean_mean":  float(mid_clean.mean()) if len(mid_clean) > 0 else np.nan,
        "tbr2_p95":         p95_val,
    }


def tbr_strategy_3(ct_data, pet_data):
    """
    TBR-3: Spine-anchored aortic ROI.
    For each axial slice in the mid-Z band:
      1. Find the vertebral column: largest high-HU (>400 HU) connected
         component after excluding the image border (ribs/skin).
      2. Place a 3mm-radius cylinder 3mm anterior (higher Y in RAS) to
         the spine centroid — this is where the aorta runs in mice.
      3. Average PET signal in that cylinder vs. median of remaining voxels.

    Returns NaN if spine detection fails in >50% of slices.

    Limitations: at 0.54mm PET resolution, the aortic ROI covers ~30 PET
    voxels. Signal is noisy; treat as a population-level trend, not
    single-subject ground truth.
    """
    ct_nz, ct_ny, ct_nz_slices = ct_data.shape
    pet_nx, pet_ny, pet_nz      = pet_data.shape

    # Scale factors between CT and PET voxel grids
    sx = pet_nx / ct_nz        # ~0.187 for 401→75
    sy = pet_ny / ct_ny        # ~0.187
    sz = pet_nz / ct_nz_slices # ~0.159 for 1200→191

    # Mid-Z band in CT voxels
    ct_z0 = int(ct_nz_slices * 0.33)
    ct_z1 = int(ct_nz_slices * 0.67)

    # Border exclusion: ignore outer 15% in XY (ribs/skin)
    bx0 = int(ct_nz * 0.15);   bx1 = int(ct_nz * 0.85)
    by0 = int(ct_ny * 0.15);   by1 = int(ct_ny * 0.85)

    # Radius of aortic ROI and offset from spine (in CT voxels)
    aorta_offset_vox = 30   # ~3mm anterior to spine centroid
    aorta_radius_vox = 15   # ~1.5mm radius around expected aorta center

    aortic_vals = []
    n_good_slices = 0

    for ct_z in range(ct_z0, ct_z1, 5):  # every 5 CT slices (~0.5mm steps)
        sl = ct_data[:, :, ct_z]

        # Find spine: largest bone component inside the body interior
        interior_bone = np.zeros_like(sl, dtype=bool)
        interior_bone[bx0:bx1, by0:by1] = sl[bx0:bx1, by0:by1] > 400

        labeled, n = ndimage.label(interior_bone)
        if n == 0:
            continue

        sizes    = [(i+1, (labeled == i+1).sum()) for i in range(n)]
        spine_id = max(sizes, key=lambda x: x[1])[0]
        ys, xs   = np.where(labeled == spine_id)

        if len(xs) < 5:
            continue

        spine_cx = float(xs.mean())
        spine_cy = float(ys.mean())

        # Aortic ROI: 1.5mm radius cylinder, 3mm anterior (higher Y) to spine
        aorta_cy = spine_cy + aorta_offset_vox

        # Bounds check — aorta must be inside the image
        if aorta_cy + aorta_radius_vox >= ct_ny or aorta_cy < 0:
            continue

        # Sample PET at the corresponding Z position
        pet_z = int(ct_z * sz)
        if pet_z >= pet_nz:
            continue
        pet_sl = pet_data[:, :, pet_z]

        # Map aortic ROI center to PET grid
        aorta_cx_pet = spine_cx * sx
        aorta_cy_pet = aorta_cy * sy
        r_pet        = max(2, int(aorta_radius_vox * sx))

        yy, xx = np.ogrid[:pet_ny, :pet_nx]
        roi_mask = ((xx - aorta_cx_pet)**2 + (yy - aorta_cy_pet)**2) <= r_pet**2

        roi_vals = pet_sl[roi_mask.T]
        roi_vals = roi_vals[roi_vals > 0]
        if len(roi_vals) < 3:
            continue

        aortic_vals.extend(roi_vals.tolist())
        n_good_slices += 1

    total_slices = (ct_z1 - ct_z0) // 5
    if n_good_slices < total_slices * 0.5:
        return {"tbr3_aortic_mean": np.nan, "tbr3_tbr": np.nan,
                "tbr3_n_slices": n_good_slices}

    aortic_mean = float(np.mean(aortic_vals))

    # Background: whole mid-Z band, nonzero, excluding top 5%
    mid_band = pet_data[:, :, int(pet_nz*0.33):int(pet_nz*0.67)].flatten()
    mid_band = mid_band[mid_band > 0]
    bg_thresh = np.percentile(mid_band, 95)
    bg_vals   = mid_band[mid_band < bg_thresh]
    bg_median = float(np.median(bg_vals)) if len(bg_vals) > 0 else np.nan

    tbr = aortic_mean / bg_median if bg_median and bg_median > 0 else np.nan

    return {
        "tbr3_aortic_mean": aortic_mean,
        "tbr3_tbr":         tbr,
        "tbr3_n_slices":    n_good_slices,
    }


# ---------------------------------------------------------------------------
# Feature extraction over all mice
# ---------------------------------------------------------------------------

def extract_all_features(mice_dir, cohort, strategies):
    mice_dir = Path(mice_dir)
    pet_key  = f"pet_{cohort.lower()}.nii.gz"
    rows     = []

    all_mice = sorted(mice_dir.glob(f"{cohort}_*"))
    print(f"[i] Found {len(all_mice)} {cohort} mice in {mice_dir}")

    for mouse_dir in all_mice:
        sid  = mouse_dir.name
        parts = sid.split("_")
        cohort_id = parts[0]
        genotype  = parts[1] if len(parts) > 1 else "unknown"

        for wk_dir_name, wk_label in zip(WEEK_ORDER, WEEK_LABELS):
            wk_dir  = mouse_dir / wk_dir_name
            pet_path = wk_dir / pet_key
            ct_path  = wk_dir / "ct_hi.nii.gz"

            if not pet_path.exists():
                continue

            row = {
                "subject_id": sid,
                "cohort":     cohort_id,
                "genotype":   genotype,
                "week":       wk_label,
                "week_rank":  WEEK_LABELS.index(wk_label),
            }

            try:
                pet_data = nib.load(str(pet_path)).get_fdata()

                if 1 in strategies:
                    row.update(tbr_strategy_1(pet_data))
                if 2 in strategies:
                    row.update(tbr_strategy_2(pet_data))
                if 3 in strategies:
                    if ct_path.exists():
                        ct_data = nib.load(str(ct_path)).get_fdata()
                        row.update(tbr_strategy_3(ct_data, pet_data))
                    else:
                        row.update({"tbr3_aortic_mean": np.nan, "tbr3_tbr": np.nan,
                                    "tbr3_n_slices": 0})
            except Exception as e:
                print(f"    [!] {sid}/{wk_dir_name}: {e}")
                continue

            rows.append(row)
            print(f"  {sid}  {wk_label}  "
                  + "  ".join(f"{k}={v:.2f}" for k, v in row.items()
                               if k.startswith("tbr") and isinstance(v, float)
                               and not np.isnan(v)))

    return rows


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_ko_vs_wt_separation(rows, tbr_col, out_dir):
    """E1: KO vs WT Mann-Whitney U and AUC-ROC per week."""
    print(f"\n[E1] KO vs WT separation — {tbr_col}")
    results = {}
    for wk in WEEK_LABELS:
        wk_rows = [r for r in rows if r["week"] == wk and not np.isnan(r.get(tbr_col, np.nan))]
        ko_vals = [r[tbr_col] for r in wk_rows if r["genotype"] == "KO"]
        wt_vals = [r[tbr_col] for r in wk_rows if r["genotype"] == "WT"]
        if len(ko_vals) < 3 or len(wt_vals) < 3:
            continue
        u, p = stats.mannwhitneyu(ko_vals, wt_vals, alternative="two-sided")
        y     = [1]*len(ko_vals) + [0]*len(wt_vals)
        score = ko_vals + wt_vals
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                auc = roc_auc_score(y, score)
            except Exception:
                auc = np.nan
        print(f"  {wk}: KO mean={np.mean(ko_vals):.2f}  WT mean={np.mean(wt_vals):.2f}"
              f"  U={u:.0f}  p={p:.3f}  AUC={auc:.3f}")
        results[wk] = {"u": u, "p": p, "auc": auc,
                       "ko_mean": np.mean(ko_vals), "wt_mean": np.mean(wt_vals)}
    return results


def eval_longitudinal_trend(rows, tbr_col):
    """E2: Per-subject Spearman ρ between week rank and TBR."""
    print(f"\n[E2] Longitudinal trend (Spearman ρ per subject) — {tbr_col}")
    by_subject = {}
    for r in rows:
        if np.isnan(r.get(tbr_col, np.nan)):
            continue
        by_subject.setdefault(r["subject_id"], []).append((r["week_rank"], r[tbr_col]))

    rhos = []
    for sid, pairs in by_subject.items():
        if len(pairs) < 3:
            continue
        ranks, vals = zip(*sorted(pairs))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, _ = stats.spearmanr(ranks, vals)
        if not np.isnan(rho):
            rhos.append(rho)

    if rhos:
        print(f"  Mean ρ = {np.mean(rhos):.3f}  (n={len(rhos)} subjects with ≥3 timepoints)")
        print(f"  Positive ρ (increasing trend): {sum(r > 0 for r in rhos)}/{len(rhos)}")
    else:
        print("  Insufficient data")
    return rhos


def eval_embedding_correlation(rows, tbr_col, emb_path):
    """E3: Spearman ρ between embedding PC1 and TBR."""
    print(f"\n[E3] Embedding–TBR correlation — {tbr_col}")

    data        = np.load(emb_path, allow_pickle=True)
    embs        = data["embeddings"].astype(np.float32)
    subject_ids = data["subject_ids"].astype(str)
    weeks       = data["weeks"].astype(str)

    # Build lookup: (subject_id, week_label) → embedding
    emb_lookup = {}
    for i, (sid, wk) in enumerate(zip(subject_ids, weeks)):
        emb_lookup[(str(sid), str(wk))] = embs[i]

    # Match rows to embeddings
    matched_embs, matched_tbr = [], []
    for r in rows:
        key = (r["subject_id"], r["week"])
        if key not in emb_lookup:
            continue
        tbr_val = r.get(tbr_col, np.nan)
        if np.isnan(tbr_val):
            continue
        matched_embs.append(emb_lookup[key])
        matched_tbr.append(tbr_val)

    if len(matched_embs) < 10:
        print(f"  Too few matched pairs ({len(matched_embs)}) — skipping")
        return np.nan

    E = np.stack(matched_embs)
    T = np.array(matched_tbr)

    # PC1 of embeddings
    pca    = PCA(n_components=5)
    E_pca  = pca.fit_transform(E)
    print(f"  Embedding PCA variance explained: {pca.explained_variance_ratio_[:5].round(3)}")

    for pc_idx in range(min(3, E_pca.shape[1])):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, p = stats.spearmanr(E_pca[:, pc_idx], T)
        print(f"  PC{pc_idx+1} ρ = {rho:.3f}  p={p:.4f}")

    # Also check raw embedding cosine similarity to week-20 centroid
    return E_pca


def eval_embedding_prediction(rows, tbr_col, emb_path):
    """
    E4: LOSO Ridge regression — predict TBR from embedding.
    Reports R², MAE, Spearman ρ between predicted and actual TBR.
    """
    print(f"\n[E4] Embedding → TBR prediction (LOSO Ridge) — {tbr_col}")

    data        = np.load(emb_path, allow_pickle=True)
    embs        = data["embeddings"].astype(np.float32)
    subject_ids = data["subject_ids"].astype(str)
    weeks       = data["weeks"].astype(str)
    emb_lookup  = {(str(s), str(w)): embs[i]
                   for i, (s, w) in enumerate(zip(subject_ids, weeks))}

    X_list, y_list, groups = [], [], []
    for r in rows:
        key     = (r["subject_id"], r["week"])
        tbr_val = r.get(tbr_col, np.nan)
        if key not in emb_lookup or np.isnan(tbr_val):
            continue
        X_list.append(emb_lookup[key])
        y_list.append(tbr_val)
        groups.append(r["subject_id"])

    if len(X_list) < 10:
        print(f"  Too few matched pairs ({len(X_list)}) — skipping")
        return

    X      = np.stack(X_list)
    y      = np.array(y_list)
    groups = np.array(groups)

    logo   = LeaveOneGroupOut()
    y_pred = np.zeros_like(y)

    for train_idx, test_idx in logo.split(X, y, groups):
        clf = Ridge(alpha=10.0)
        clf.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = clf.predict(X[test_idx])

    ss_res  = np.sum((y - y_pred)**2)
    ss_tot  = np.sum((y - y.mean())**2)
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae     = float(np.mean(np.abs(y - y_pred)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p = stats.spearmanr(y, y_pred)

    print(f"  R²  = {r2:.3f}")
    print(f"  MAE = {mae:.2f}")
    print(f"  Spearman ρ = {rho:.3f}  p={p:.4f}  (n={len(y)})")
    return {"r2": r2, "mae": mae, "rho": rho, "p": p, "n": len(y),
            "y_true": y, "y_pred": y_pred, "groups": groups}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_tbr_trajectories(rows, tbr_col, out_dir, title_suffix=""):
    """Longitudinal TBR trajectories per subject, colored by genotype."""
    ko_rows = {r["subject_id"]: [] for r in rows if r["genotype"] == "KO"}
    wt_rows = {r["subject_id"]: [] for r in rows if r["genotype"] == "WT"}

    for r in rows:
        val = r.get(tbr_col, np.nan)
        if np.isnan(val):
            continue
        d = ko_rows if r["genotype"] == "KO" else wt_rows
        if r["subject_id"] in d:
            d[r["subject_id"]].append((r["week_rank"], val))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (label, subj_dict, color) in zip(
            axes, [("KO (disease)", ko_rows, "#F44336"),
                   ("WT (control)", wt_rows, "#2196F3")]):
        for sid, pairs in subj_dict.items():
            if len(pairs) < 2:
                continue
            pairs = sorted(pairs)
            xs = [WEEK_LABELS[p[0]] for p in pairs]
            ys = [p[1] for p in pairs]
            ax.plot(xs, ys, "-o", color=color, alpha=0.4, linewidth=1, markersize=4)
        # Mean trajectory
        by_week = {}
        for r in rows:
            if r["genotype"] != label[:2] or np.isnan(r.get(tbr_col, np.nan)):
                continue
            by_week.setdefault(r["week_rank"], []).append(r[tbr_col])
        mean_xs = sorted(by_week)
        mean_ys = [np.mean(by_week[k]) for k in mean_xs]
        ax.plot([WEEK_LABELS[k] for k in mean_xs], mean_ys,
                "-o", color="black", linewidth=2.5, markersize=7, label="Mean")
        ax.set_title(f"{label}", fontsize=11)
        ax.set_xlabel("Week"); ax.tick_params(axis='x', rotation=20)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(tbr_col)
    plt.suptitle(f"Longitudinal {tbr_col} trajectories{title_suffix}", fontsize=12)
    plt.tight_layout()
    out = Path(out_dir) / f"tbr_trajectories_{tbr_col}.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [+] {out.name}")


def plot_embedding_vs_tbr(rows, tbr_col, emb_pca, out_dir):
    """Scatter: embedding PC1 vs TBR, colored by genotype and week."""
    data        = []
    ko_color    = "#F44336"
    wt_color    = "#2196F3"

    fig, ax = plt.subplots(figsize=(7, 5))
    for row, pc1 in zip(rows, emb_pca[:, 0] if hasattr(emb_pca, '__len__') else []):
        tbr_val = row.get(tbr_col, np.nan)
        if np.isnan(tbr_val):
            continue
        color = ko_color if row["genotype"] == "KO" else wt_color
        ax.scatter(pc1, tbr_val, c=color, s=30, alpha=0.6)

    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=ko_color,
                      markersize=8, label='KO'),
               Line2D([0],[0], marker='o', color='w', markerfacecolor=wt_color,
                      markersize=8, label='WT')]
    ax.legend(handles=handles)
    ax.set_xlabel("Embedding PC1"); ax.set_ylabel(tbr_col)
    ax.set_title(f"RAD-DINO embedding PC1 vs {tbr_col}")
    plt.tight_layout()
    out = Path(out_dir) / f"embedding_vs_{tbr_col}.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [+] {out.name}")


def plot_prediction_scatter(eval_result, tbr_col, out_dir):
    """Scatter: predicted vs actual TBR (LOSO)."""
    y, yp = eval_result["y_true"], eval_result["y_pred"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, yp, s=30, alpha=0.7, color="#2196F3")
    mn, mx = min(y.min(), yp.min()), max(y.max(), yp.max())
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.4, label="y=x")
    ax.set_xlabel(f"Actual {tbr_col}"); ax.set_ylabel(f"Predicted {tbr_col}")
    ax.set_title(f"E4 — LOSO prediction  R²={eval_result['r2']:.3f}  ρ={eval_result['rho']:.3f}")
    ax.legend()
    plt.tight_layout()
    out = Path(out_dir) / f"tbr_prediction_scatter_{tbr_col}.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [+] {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = Path(args.output_dir) / "plots"
    os.makedirs(plots_dir, exist_ok=True)

    strategies = [1, 2, 3] if args.strategy == 0 else [args.strategy]
    print(f"[i] Running TBR strategies: {strategies}")
    print(f"[i] Cohort: {args.cohort}")
    print(f"[i] Mice dir: {args.mice_dir}")

    # ---- Extract features ----
    rows = extract_all_features(args.mice_dir, args.cohort, strategies)

    if not rows:
        print("[!] No rows extracted — check --mice-dir path")
        return

    # Save feature CSV
    tbr_cols = [k for k in rows[0] if k.startswith("tbr")]
    fieldnames = ["subject_id", "cohort", "genotype", "week", "week_rank"] + tbr_cols
    csv_path = Path(args.output_dir) / f"tbr_features_{args.cohort}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[+] Saved features → {csv_path}  ({len(rows)} rows)")

    if args.no_eval:
        return

    if not os.path.exists(args.embeddings):
        print(f"[!] Embeddings not found: {args.embeddings} — skipping E3/E4")
        args.embeddings = None

    # ---- Evaluate each TBR strategy ----
    summary = {}
    for tbr_col in tbr_cols:
        if all(np.isnan(r.get(tbr_col, np.nan)) for r in rows):
            print(f"\n[!] {tbr_col}: all NaN — skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating: {tbr_col}")
        print(f"{'='*60}")

        e1 = eval_ko_vs_wt_separation(rows, tbr_col, args.output_dir)
        e2 = eval_longitudinal_trend(rows, tbr_col)

        emb_pca    = None
        e3_result  = None
        e4_result  = None
        if args.embeddings:
            emb_pca   = eval_embedding_correlation(rows, tbr_col, args.embeddings)
            e4_result = eval_embedding_prediction(rows, tbr_col, args.embeddings)

        # Plots
        print("\n[Plots]")
        plot_tbr_trajectories(rows, tbr_col, plots_dir,
                              title_suffix=f" ({args.cohort})")
        if emb_pca is not None and hasattr(emb_pca, '__len__'):
            plot_embedding_vs_tbr(rows, tbr_col, emb_pca, plots_dir)
        if e4_result:
            plot_prediction_scatter(e4_result, tbr_col, plots_dir)

        summary[tbr_col] = {
            "e1_week12_auc": e1.get("Week 12", {}).get("auc", np.nan),
            "e1_week20_auc": e1.get("Week 20", {}).get("auc", np.nan),
            "e2_mean_rho":   float(np.mean(e2)) if e2 else np.nan,
            "e4_r2":         e4_result["r2"] if e4_result else np.nan,
            "e4_rho":        e4_result["rho"] if e4_result else np.nan,
        }

    # ---- Summary table ----
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'TBR col':<25} {'E1 W12 AUC':>12} {'E1 W20 AUC':>12} {'E2 ρ':>8} {'E4 R²':>8} {'E4 ρ':>8}")
    print("-" * 78)
    for col, vals in summary.items():
        print(f"{col:<25} {vals['e1_week12_auc']:>12.3f} {vals['e1_week20_auc']:>12.3f} "
              f"{vals['e2_mean_rho']:>8.3f} {vals['e4_r2']:>8.3f} {vals['e4_rho']:>8.3f}")

    # Save summary
    summary_path = Path(args.output_dir) / f"tbr_eval_summary_{args.cohort}.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tbr_col"] + list(next(iter(summary.values())).keys()))
        w.writeheader()
        for col, vals in summary.items():
            w.writerow({"tbr_col": col, **vals})
    print(f"\n[+] Saved summary → {summary_path}")


if __name__ == "__main__":
    main()
