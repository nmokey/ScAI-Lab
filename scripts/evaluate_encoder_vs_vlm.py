"""
evaluate_encoder_vs_vlm.py

Encoder-level evaluation comparable to the VLM LOSO tasks, restricted to
the 32 NaF subjects with a Week 12 baseline and ≥1 future scan (matching
the VLM training population exactly).

Two evaluation conditions:
  A) RAD-DINO real embeddings only
     - Genotype (WT vs KO): LOSO logistic probe on ts0 (Week 12) alone
     - TBR regression: LOSO Ridge from ts0 alone, predicting TBR at each future week
     - TBR regression (multi-token): LOSO Ridge from [ts0, ts1, ts2, ts3] concatenated
       where ts1/ts2/ts3 are the actual observed RAD-DINO embeddings (when available)

  B) Longitudinal (ts0 real + ts1/ts2/ts3 MLP-predicted)
     - Genotype: LOSO logistic probe on [ts0, ts1, ts2, ts3] concatenated
       where ts1/ts2/ts3 come from the longitudinal MLP predicted_embeddings/
     
     - TBR regression: LOSO Ridge from the same 4-token input

LOSO is subject-level (same protocol as the VLM): train on 31, test on 1.
TBR target: tbr2_p95_median from tbr_features_NaF.csv, one value per (subject, week).
TBR slots match VLM convention: slot 0=Week 15 (Δ3wk), slot 1=Week 18 (Δ6wk),
slot 2=Week 20 (Δ8wk).

Usage:
    cd ~/ScAI-Lab
    python scripts/evaluate_encoder_vs_vlm.py
"""

import csv
import json
import os
import warnings
from collections import defaultdict
from math import sqrt

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RADDINO_NPZ  = "/data1/Processed_NIfTI_Test/embeddings/raddino/raddino_embeddings.npz"
TBR_CSV      = "/data1/Processed_NIfTI_Test/embeddings/longitudinal/tbr_features_NaF.csv"
PRED_DIR     = "/data1/Processed_NIfTI_Test/embeddings/longitudinal/predicted_embeddings"
VLM_JSON     = "/data1/Processed_NIfTI_Test/embeddings/vlm/mouse_all_vqa_traj.json"
OUT_JSON     = "/data1/Processed_NIfTI_Test/embeddings/vlm/encoder_vs_vlm_eval.json"

FUTURE_WEEKS = ["Week 15", "Week 18", "Week 20"]
SLOT_LABEL   = {0: "delta_3wk (Week 15)", 1: "delta_6wk (Week 18)", 2: "delta_8wk (Week 20)"}
EMB_DIM      = 768

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sqrt(sum((x - mx)**2 for x in xs)) * sqrt(sum((y - my)**2 for y in ys))
    return num / den if den > 0 else float("nan")


def r2(ys_true, ys_pred):
    n = len(ys_true)
    if n < 2:
        return float("nan")
    mean_t = sum(ys_true) / n
    ss_tot = sum((y - mean_t)**2 for y in ys_true)
    ss_res = sum((t - p)**2 for t, p in zip(ys_true, ys_pred))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def tbr_slot_metrics(gt_by_slot, pred_by_slot):
    results = {}
    all_gt, all_pred = [], []
    for slot in range(3):
        gt   = gt_by_slot[slot]
        pred = pred_by_slot[slot]
        if len(gt) < 2:
            results[SLOT_LABEL[slot]] = {"n": len(gt), "mae": None, "pearson_r": None, "r2": None}
            continue
        mae = float(np.mean(np.abs(np.array(gt) - np.array(pred))))
        r   = pearson(gt, pred)
        r2_ = r2(gt, pred)
        results[SLOT_LABEL[slot]] = {"n": len(gt), "mae": round(mae, 3),
                                      "pearson_r": round(r, 3), "r2": round(r2_, 3)}
        all_gt.extend(gt); all_pred.extend(pred)
    overall = {}
    if len(all_gt) >= 2:
        overall = {
            "overall_n":        len(all_gt),
            "overall_mae":      round(float(np.mean(np.abs(np.array(all_gt) - np.array(all_pred)))), 3),
            "overall_pearson_r": round(pearson(all_gt, all_pred), 3),
            "overall_r2":       round(r2(all_gt, all_pred), 3),
        }
    return results, overall


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_raddino():
    """Returns dict: sid -> {week -> 768-d np.array}. NaF only."""
    d    = np.load(RADDINO_NPZ, allow_pickle=True)
    embs = d["embeddings"].astype(np.float32)
    sids = d["subject_ids"].astype(str)
    wks  = d["weeks"].astype(str)
    out  = defaultdict(dict)
    for e, s, w in zip(embs, sids, wks):
        if s.startswith("NaF"):
            out[s][w] = e
    return dict(out)


def load_tbr():
    """Returns dict: sid -> {week -> tbr2_p95_median float}. NaN excluded."""
    out = defaultdict(dict)
    with open(TBR_CSV) as f:
        for row in csv.DictReader(f):
            val = row.get("tbr2_p95_median", "")
            try:
                v = float(val)
            except (ValueError, TypeError):
                continue
            if not np.isnan(v):
                out[row["subject_id"]][row["week"]] = v
    return dict(out)


def load_predicted(sid):
    """Load MLP-predicted future embeddings for one NaF subject. Returns {tag: array}."""
    out = {}
    for tag, week in [("ts1", "Week 15"), ("ts2", "Week 18"), ("ts3", "Week 20")]:
        path = os.path.join(PRED_DIR, f"{sid}_{tag}.npy")
        if os.path.exists(path):
            out[week] = np.load(path).astype(np.float32)
    return out


def get_vlm_subjects():
    """Return the 32 subject IDs used in the VLM LOSO (NaF, Week 12 + ≥1 future)."""
    with open(VLM_JSON) as f:
        records = json.load(f)
    return sorted(set(r["pid"] for r in records))


# ---------------------------------------------------------------------------
# Build feature matrices
# ---------------------------------------------------------------------------

def build_ts0_features(subjects, raddino):
    """Single-token: ts0 (Week 12 RAD-DINO) only."""
    X, genotypes, sids = [], [], []
    for sid in subjects:
        if "Week 12" not in raddino.get(sid, {}):
            continue
        X.append(raddino[sid]["Week 12"])
        genotypes.append(1 if "_KO_" in sid else 0)
        sids.append(sid)
    return np.array(X), np.array(genotypes), np.array(sids)


def build_real_4token_features(subjects, raddino):
    """4-token real: [ts0, ts1, ts2, ts3] concatenated using actual RAD-DINO embeddings.
    Missing future weeks zero-padded."""
    X, genotypes, sids = [], [], []
    for sid in subjects:
        wk_map = raddino.get(sid, {})
        if "Week 12" not in wk_map:
            continue
        tokens = [wk_map["Week 12"]]
        for wk in FUTURE_WEEKS:
            tokens.append(wk_map.get(wk, np.zeros(EMB_DIM, dtype=np.float32)))
        X.append(np.concatenate(tokens))
        genotypes.append(1 if "_KO_" in sid else 0)
        sids.append(sid)
    return np.array(X), np.array(genotypes), np.array(sids)


def build_predicted_4token_features(subjects, raddino):
    """4-token longitudinal: ts0 real + ts1/ts2/ts3 from MLP predictions.
    Missing predictions zero-padded.

    WARNING — label leakage: the longitudinal MLP is conditioned on genotype,
    so predicted embeddings implicitly encode the genotype label. Genotype
    metrics from this condition (B) are an inflated upper bound, not a fair
    comparison with the VLM. See results.md for details.
    """
    X, genotypes, sids = [], [], []
    for sid in subjects:
        if "Week 12" not in raddino.get(sid, {}):
            continue
        preds = load_predicted(sid)
        tokens = [raddino[sid]["Week 12"]]
        for wk in FUTURE_WEEKS:
            tokens.append(preds.get(wk, np.zeros(EMB_DIM, dtype=np.float32)))
        X.append(np.concatenate(tokens))
        genotypes.append(1 if "_KO_" in sid else 0)
        sids.append(sid)
    return np.array(X), np.array(genotypes), np.array(sids)


# ---------------------------------------------------------------------------
# LOSO evaluations
# ---------------------------------------------------------------------------

def loso_genotype(X, y, sids, label):
    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all, y_prob_all = [], [], []
    for tr, te in logo.split(X, y, groups=sids):
        if len(set(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X[tr], y[tr])
        y_pred_all.extend(clf.predict(X[te]).tolist())
        y_true_all.extend(y[te].tolist())
        y_prob_all.extend(clf.predict_proba(X[te])[:, 1].tolist())

    acc = accuracy_score(y_true_all, y_pred_all)
    try:
        auc = roc_auc_score(y_true_all, y_prob_all)
    except Exception:
        auc = float("nan")
    print(f"  [{label}] Genotype  acc={acc:.3f}  AUC={auc:.3f}  (n={len(y_true_all)}, chance=0.5)")
    return {"accuracy": round(acc, 3), "auc": round(auc, 3), "n": len(y_true_all)}


def loso_tbr(X, sids, tbr, label):
    """
    LOSO Ridge regression: predict TBR at each future week from the embedding.
    One prediction per (subject, future_week) pair where TBR ground truth exists.

    X    : (n_subjects, D) feature matrix, rows aligned with sids
    sids : (n_subjects,) subject IDs — must be in the same order as X rows
    tbr  : dict sid -> {week -> float}
    """
    # Build a fast sid→row-index map so lookup is O(1) instead of O(n) per subject.
    sid_to_row = {sid: i for i, sid in enumerate(sids)}

    logo = LeaveOneGroupOut()
    gt_by_slot   = defaultdict(list)
    pred_by_slot = defaultdict(list)

    for slot, fw in enumerate(FUTURE_WEEKS):
        X_slot, y_slot, g_slot = [], [], []
        for sid in sids:
            tbr_val = tbr.get(sid, {}).get(fw)
            if tbr_val is None:
                continue
            X_slot.append(X[sid_to_row[sid]])
            y_slot.append(tbr_val)
            g_slot.append(sid)

        if len(set(g_slot)) < 4:
            continue

        X_slot = np.array(X_slot)
        y_slot = np.array(y_slot)
        g_slot = np.array(g_slot)
        y_pred = np.zeros_like(y_slot)

        for tr, te in logo.split(X_slot, y_slot, groups=g_slot):
            clf = Ridge(alpha=10.0)
            clf.fit(X_slot[tr], y_slot[tr])
            y_pred[te] = clf.predict(X_slot[te])

        gt_by_slot[slot].extend(y_slot.tolist())
        pred_by_slot[slot].extend(y_pred.tolist())

    slot_results, overall = tbr_slot_metrics(gt_by_slot, pred_by_slot)
    print(f"  [{label}] TBR overall  MAE={overall.get('overall_mae')}  "
          f"r={overall.get('overall_pearson_r')}  R²={overall.get('overall_r2')}  "
          f"n={overall.get('overall_n')}")
    for sl, m in slot_results.items():
        print(f"    {sl}: MAE={m['mae']}  r={m['pearson_r']}  R²={m['r2']}  n={m['n']}")
    return {"by_slot": slot_results, **overall}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    raddino  = load_raddino()
    tbr      = load_tbr()
    subjects = get_vlm_subjects()  # exactly the 32 VLM subjects
    print(f"VLM subjects: {len(subjects)}")

    results = {}

    # ---- Condition A1: ts0 only (Week 12 RAD-DINO) ----
    print("\n=== A1: ts0 only (Week 12 RAD-DINO real embedding) ===")
    X_ts0, y_ts0, sids_ts0 = build_ts0_features(subjects, raddino)
    print(f"  Subjects with Week 12 embedding: {len(sids_ts0)}")
    results["A1_ts0_only"] = {
        "description": "ts0 only — Week 12 RAD-DINO real embedding (768-d)",
        "genotype": loso_genotype(X_ts0, y_ts0, sids_ts0, "A1"),
        "tbr":      loso_tbr(X_ts0, sids_ts0, tbr, "A1"),
    }

    # ---- Condition A2: 4-token real (ts0+ts1+ts2+ts3 all RAD-DINO actual) ----
    print("\n=== A2: 4-token real (ts0-ts3 all actual RAD-DINO embeddings, concatenated) ===")
    X_real4, y_real4, sids_real4 = build_real_4token_features(subjects, raddino)
    print(f"  Subjects: {len(sids_real4)}")
    results["A2_real_4token"] = {
        "description": "4-token real — [ts0,ts1,ts2,ts3] actual RAD-DINO embeddings concatenated (3072-d), missing weeks zero-padded",
        "genotype": loso_genotype(X_real4, y_real4, sids_real4, "A2"),
        "tbr":      loso_tbr(X_real4, sids_real4, tbr, "A2"),
    }

    # ---- Condition B: 4-token longitudinal (ts0 real + ts1/ts2/ts3 MLP-predicted) ----
    print("\n=== B: 4-token longitudinal (ts0 real + ts1/ts2/ts3 MLP-predicted, concatenated) ===")
    X_pred4, y_pred4, sids_pred4 = build_predicted_4token_features(subjects, raddino)
    print(f"  Subjects: {len(sids_pred4)}")
    results["B_predicted_4token"] = {
        "description": "4-token longitudinal — ts0 real RAD-DINO + ts1/ts2/ts3 MLP-predicted (3072-d), missing predictions zero-padded",
        "genotype": loso_genotype(X_pred4, y_pred4, sids_pred4, "B"),
        "tbr":      loso_tbr(X_pred4, sids_pred4, tbr, "B"),
    }

    # ---- VLM results for reference ----
    results["VLM_baseline_1token"] = {
        "description": "VLM baseline — 1-token (ts0 only), LoRA LLaMA-3.1-8B, multitask heads, LOSO",
        "genotype": {"accuracy": 0.219, "auc": None, "n": 32},
        "tbr": {
            "overall_n": 134, "overall_mae": 6.241, "overall_pearson_r": -0.054, "overall_r2": -0.357,
            "by_slot": {
                "delta_3wk (Week 15)": {"n": 64, "mae": 6.806, "pearson_r": -0.090, "r2": -0.088},
                "delta_6wk (Week 18)": {"n": 30, "mae": 7.568, "pearson_r": -0.243, "r2": -0.957},
                "delta_8wk (Week 20)": {"n": 40, "mae": 4.340, "pearson_r":  0.069, "r2": -4.421},
            }
        }
    }
    results["VLM_longitudinal_4token"] = {
        "description": "VLM longitudinal — 4-token (ts0 real + ts1/ts2/ts3 MLP-predicted), LoRA LLaMA-3.1-8B, multitask heads, LOSO",
        "genotype": {"accuracy": 0.531, "auc": None, "n": 32},
        "tbr": {
            "overall_n": 134, "overall_mae": 5.393, "overall_pearson_r": 0.276, "overall_r2": -0.076,
            "by_slot": {
                "delta_3wk (Week 15)": {"n": 64, "mae": 6.123, "pearson_r": 0.447, "r2":  0.131},
                "delta_6wk (Week 18)": {"n": 30, "mae": 6.855, "pearson_r": -0.093, "r2": -0.722},
                "delta_8wk (Week 20)": {"n": 40, "mae": 3.127, "pearson_r":  0.207, "r2": -1.826},
            }
        }
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results saved → {OUT_JSON}")

    # ---- Summary table ----
    print("\n" + "="*70)
    print("SUMMARY: Encoder linear probe vs VLM (LOSO, 32 NaF subjects)")
    print("="*70)
    print(f"{'Condition':<38} {'Geno acc':>9} {'TBR r (Δ3wk)':>13} {'TBR R² (Δ3wk)':>14}")
    print("-"*70)
    for k, v in results.items():
        geno = v["genotype"].get("accuracy")
        tbr_slot = v["tbr"].get("by_slot", {})
        d3 = tbr_slot.get("delta_3wk (Week 15)", {})
        r_val  = d3.get("pearson_r", "—")
        r2_val = d3.get("r2", "—")
        print(f"  {k:<36} {str(geno):>9} {str(r_val):>13} {str(r2_val):>14}")


if __name__ == "__main__":
    main()
