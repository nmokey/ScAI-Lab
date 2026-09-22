"""Read-only audit of current research artifacts; CPU, no model downloads.

Prints a JSON snapshot. No source data, training outputs, or running jobs are changed.
Exploratory baseline scores are descriptive, not confirmatory hypothesis tests.
"""
import argparse
import csv
import json
import hashlib
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_stats import join_mouse_group_ids


def regression(y, p):
    return dict(n=len(y), mae=float(mean_absolute_error(y, p)),
                mse=float(mean_squared_error(y, p)), r2=float(r2_score(y, p)))


def cosine(x, y):
    return np.sum(x*y, axis=-1)/(np.linalg.norm(x, axis=-1)*np.linalg.norm(y, axis=-1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="/data1/Processed_NIfTI_Test")
    args = ap.parse_args()
    root = Path(args.data_root)
    embroot = root / "embeddings"
    records = json.loads((embroot / "vlm/mouse_all_vqa_traj.json").read_text())
    sids = sorted({r["pid"] for r in records})
    out = {"captured_utc": datetime.now(timezone.utc).isoformat(), "encoders": {}}
    for encoder in ["raddino", "colipri", "merlin", "m3d"]:
        data = np.load(embroot / encoder / f"{encoder}_embeddings.npz", allow_pickle=True)
        keys = list(zip(data["subject_ids"].astype(str), data["weeks"].astype(str)))
        e = data["embeddings"]
        out["encoders"][encoder] = dict(shape=list(e.shape), subjects=len(set(data["subject_ids"])),
            duplicate_keys=len(keys)-len(set(keys)), nonfinite=int((~np.isfinite(e)).sum()),
            modalities=sorted(set(data["modalities"].tolist())))
        if encoder == "raddino":
            lookup = dict(zip(keys, e))
            rd = data
    groups = join_mouse_group_ids(np.array(sids), np.array(["Week 12"]*len(sids)), root / "mouse_manifest.csv")
    groups_all = join_mouse_group_ids(rd["subject_ids"], rd["weeks"], root / "mouse_manifest.csv")
    group_by_sid = dict(zip(rd["subject_ids"], groups_all))
    out["vlm_data"] = dict(subjects=len(sids), records=len(records), types=dict(Counter(r["content_type"] for r in records)),
        group_composition={g: dict(Counter(s.split("_")[1] for s, grp in zip(sids, groups) if grp == g)) for g in sorted(set(groups))},
        duplicate_pid_type=len(records)-len({(r["pid"], r["content_type"]) for r in records}),
        ts0_max_error=max(float(np.max(np.abs(load_file(r["embedding_path_ts0"])["embeddings"].reshape(-1)-lookup[(r["pid"], "Week 12")]))) for r in records))
    X = np.stack([lookup[(s, "Week 12")] for s in sids])
    ygeno = np.array([int("_KO_" in s) for s in sids])
    # Exactly the VLM population, with training-fold standardization.
    probes = {}
    for name, g in [("subject", np.array(sids)), ("mousegroup", groups)]:
        pred = np.full(len(sids), np.nan)
        skipped = []
        for tr, te in LeaveOneGroupOut().split(X, ygeno, g):
            if len(set(ygeno[tr])) < 2:
                skipped.extend([sids[i] for i in te])
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, multi_class="multinomial", random_state=42))
            m.fit(X[tr], ygeno[tr]); pred[te] = m.predict_proba(X[te])[:, 1]
        keep = np.isfinite(pred)
        probes[name] = dict(auc=float(roc_auc_score(ygeno[keep], pred[keep])) if len(set(ygeno[keep])) == 2 else None,
                           predicted_subjects=int(keep.sum()), skipped_subjects=skipped)
    out["matched_week12_genotype_probe_auc"] = probes
    # Compare rolled-out exports against simple training-fold week centroids.
    forecast = {}
    for week, tag in [("Week 15", "ts1"), ("Week 18", "ts2"), ("Week 20", "ts3")]:
        samples = [s for s in sids if (s, week) in lookup]
        actual = np.stack([lookup[(s, week)] for s in samples])
        predicted = np.stack([np.load(embroot / "longitudinal_nogeno_f22/predicted_embeddings" / f"{s}_{tag}.npy") for s in samples])
        centroid = np.stack([np.mean([v for (ss, w), v in lookup.items() if ss in sids and ss != s and w == week], axis=0) for s in samples])
        centroid_group = np.stack([np.mean([v for (ss, w), v in lookup.items() if ss in sids and group_by_sid[ss] != group_by_sid[s] and w == week], axis=0) for s in samples])
        persistence = np.stack([lookup[(s, "Week 12")] for s in samples])
        forecast[week] = dict(n=len(samples), predicted_cosine=float(cosine(predicted, actual).mean()),
            subject_excluded_centroid_cosine=float(cosine(centroid, actual).mean()),
            group_excluded_centroid_cosine=float(cosine(centroid_group, actual).mean()),
            baseline_persistence_cosine=float(cosine(persistence, actual).mean()),
            predicted_norm=float(np.linalg.norm(predicted, axis=1).mean()),
            actual_norm=float(np.linalg.norm(actual, axis=1).mean()),
            fraction_beating_subject_centroid=float((cosine(predicted, actual)>cosine(centroid, actual)).mean()))
    out["rollout_vs_centroid"] = forecast
    # Slot targets use the same rendered, rounded labels as the VLM.
    import re
    targets = {r["pid"]: {int(w):float(v) for w,v in re.findall(r"Week\s+(\d+):\s*(\d+(?:\.\d+)?)", r["answer"])} for r in records if r["content_type"] == "tbr"}
    baselines = {}
    for name, g in [("subject", np.array(sids)), ("mousegroup", groups)]:
        byslot = {}
        for delta in [3, 6, 8]:
            ids = np.array([i for i, s in enumerate(sids) if delta in targets[s]])
            yy = np.array([targets[sids[i]][delta] for i in ids])
            preds = {"train_mean":np.full(len(ids), np.nan), "train_median":np.full(len(ids), np.nan), "ridge_alpha100":np.full(len(ids), np.nan)}
            for tr,te in LeaveOneGroupOut().split(X[ids], yy, g[ids]):
                preds["train_mean"][te] = yy[tr].mean(); preds["train_median"][te] = np.median(yy[tr])
                m = make_pipeline(StandardScaler(), Ridge(alpha=100))
                m.fit(X[ids[tr]], yy[tr]); preds["ridge_alpha100"][te] = m.predict(X[ids[te]])
            byslot[str(delta)] = {k:regression(yy,v) for k,v in preds.items()}
        baselines[name] = byslot
    out["current_tbr_baselines"] = baselines
    manifest_rows = list(csv.DictReader((root / "mouse_manifest.csv").open()))
    import nibabel as nib
    geometry = []
    cache = []
    for row in manifest_rows:
        ct_path = Path(row["ct_hi_nifti"])
        key = hashlib.md5(pickle.dumps({"image": str(ct_path)}, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
        cache_path = embroot / "merlin/cache" / f"{key}.pt"
        if row["session_id"] in ("m54406", "m54520"):
            cache.append(dict(subject=row["mouse_id"], cache_exists=cache_path.exists(),
                              cache_older_than_ct=cache_path.stat().st_mtime < ct_path.stat().st_mtime if cache_path.exists() else None))
        if row["cohort"] != "NaF" or not row["pet_nifti"]:
            continue
        ct = nib.load(ct_path); pet = nib.load(row["pet_nifti"])
        centre = np.array(ct.shape)/2
        correct = (np.linalg.inv(pet.affine) @ ct.affine @ np.r_[centre,1])[:3]
        scaled = centre*np.array(pet.shape)/np.array(ct.shape)
        geometry.append(float(np.linalg.norm((correct-scaled)*pet.header.get_zooms()[:3])))
    out["acquisition_checks"] = dict(
        timing_counts={str(k):v for k,v in Counter((r["cohort"],r["timepoint_h"]) for r in manifest_rows).items()},
        different_ct_pet_source=[{k:r[k] for k in ["mouse_id","week","ct_source_scan","pet_source_scan"]} for r in manifest_rows if r["pet_source_scan"] and r["ct_source_scan"]!=r["pet_source_scan"]],
        tbr3_shape_mapping_error_mm=dict(n=len(geometry), median=float(np.median(geometry)), maximum=max(geometry)),
        corrected_crop_merlin_cache=cache)
    runs = {}
    for d in sorted((embroot / "vlm/runs").glob("mouse_vlm_nogeno*")):
        p = d / "vqa_loso.json"
        if not p.exists(): continue
        rows = json.loads(p.read_text())
        counts = Counter((r["pid"], r["qid"]) for r in rows)
        item = dict(records=len(rows), subjects=len({r["pid"] for r in rows}), duplicate_pid_qid=sum(n-1 for n in counts.values()),
            completed_metrics=(d/"loso_results.json").exists())
        if item["completed_metrics"]:
            metrics = json.loads((d/"loso_results.json").read_text())
            item["metrics"] = {k:metrics.get(k) for k in ["genotype_auc", "genotype_n", "tbr_reg_overall_mae", "tbr_reg_overall_r2"]}
            per = {r["pid"]:r["tbr_targets"] for r in rows if r.get("content_type")=="tbr" or ("TBR" in r.get("orig_question","") and "status" not in r.get("orig_question",""))}
            gt, means = [], []
            for pid, ts in per.items():
                for slot, value in enumerate(ts):
                    if value < 0: continue
                    gt.append(value)
                    means.append(float(np.mean([vv[slot] for pp,vv in per.items() if pp!=pid and vv[slot]>=0])))
            item["fold_mean_baseline_on_saved_targets"] = regression(gt, means)
        item["unique_genotype_labels_so_far"] = sorted({r["genotype_label"] for r in rows})
        runs[d.name] = item
    out["runs"] = runs
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
