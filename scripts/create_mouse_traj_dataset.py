"""
create_mouse_traj_dataset.py

Builds the trajectory VQA JSON files and per-scan .safetensors embeddings
needed to train the NephrologyKG VLM on mouse atherosclerosis data.

Analogous to MedTrinity-25M/create_nlst_3d_traj_dataset.py but adapted for:
  - RAD-DINO 768-d embeddings (from raddino_embeddings.npz)
  - Mouse subjects: NaF cohort, WT and KO genotypes
  - Timepoints: Week 12 / 15 / 18 / 20
  - Long topic: aortic TBR trajectory (from tbr_features_NaF.csv)
  - Short topic: genotype (WT vs KO)
  - Input: any timepoint with at least one future timepoint available
  - Splits: random 80/10/10 by subject

All VQA records use Week 12 as the fixed input timepoint. A subject with
W12/W15/W18/W20 contributes records (W12→W15/W18/W20). Using only Week 12
ensures a consistent baseline across all subjects.

Outputs (written to --output-dir):
  embeddings/
    <subject_id>_<week_tag>.safetensors   — one file per scan, shape (1, 768)
  mouse_train_vqa_traj.json
  mouse_val_vqa_traj.json
  mouse_test_vqa_traj.json
  mouse_all_vqa_traj.json

Usage
-----
    python vlm/create_mouse_traj_dataset.py --dry-run
    python vlm/create_mouse_traj_dataset.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.torch import save_file
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vlm"))
from utils.research_io import atomic_json, file_sha256

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEEK_ORDER = ["Week 12", "Week 15", "Week 18", "Week 20"]
WEEK_TAGS  = {"Week 12": "ts0", "Week 15": "ts1", "Week 18": "ts2", "Week 20": "ts3"}

TBR_COL  = "tbr2_p95_median"
WEEK_NUM = {"Week 12": 12, "Week 15": 15, "Week 18": 18, "Week 20": 20}

DEFAULT_EMB_NPZ  = "/data1/Processed_NIfTI_Test/embeddings/raddino/raddino_embeddings.npz"
DEFAULT_TBR_CSV  = "/data1/Processed_NIfTI_Test/embeddings/longitudinal/tbr_features_NaF.csv"
DEFAULT_OUT_DIR  = "/data1/Processed_NIfTI_Test/embeddings/vlm/validated_20260914"

TEST_FRAC = 0.1
VAL_FRAC  = 0.1
SEED      = 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings",     default=DEFAULT_EMB_NPZ)
    p.add_argument("--tbr-csv",        default=DEFAULT_TBR_CSV)
    p.add_argument("--manifest", default="/data1/Processed_NIfTI_Test/mouse_manifest.csv")
    p.add_argument("--output-dir",     default=DEFAULT_OUT_DIR)
    p.add_argument("--min-timepoints", type=int, default=2,
                   help="Minimum total timepoints a subject must have (default 2)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print stats and exit without writing files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings(npz_path):
    """Returns dict: subject_id -> {week -> np.ndarray (768,)}"""
    data        = np.load(npz_path, allow_pickle=True)
    embeddings  = data["embeddings"].astype(np.float32)
    subject_ids = data["subject_ids"].astype(str)
    weeks       = data["weeks"].astype(str)

    emb_map = {}
    for emb, sid, wk in zip(embeddings, subject_ids, weeks):
        if wk in emb_map.get(sid, {}) or emb.shape != (768,) or not np.isfinite(emb).all():
            raise ValueError(f"Duplicate or invalid embedding: {sid}/{wk}")
        emb_map.setdefault(sid, {})[wk] = emb

    print(f"[i] Loaded {len(embeddings)} embeddings across {len(emb_map)} subjects")
    return emb_map


def load_tbr(csv_path):
    """Returns dict: subject_id -> {week -> float}"""
    df = pd.read_csv(csv_path)
    tbr_map = {}
    for _, row in df.iterrows():
        sid, week, value = str(row["subject_id"]), str(row["week"]), float(row[TBR_COL])
        if week in tbr_map.get(sid, {}):
            raise ValueError(f"Duplicate PET target: {sid}/{week}")
        if not np.isfinite(value):
            continue  # Missing measurements remain masked, never numerical labels.
        if value < 0:
            raise ValueError(f"Negative PET target: {sid}/{week}")
        tbr_map.setdefault(sid, {})[week] = value
    print(f"[i] Loaded TBR for {len(tbr_map)} subjects  (column: {TBR_COL})")
    return tbr_map


# ---------------------------------------------------------------------------
# .safetensors export
# ---------------------------------------------------------------------------

def export_safetensors(emb_map, out_emb_dir, dry_run=False):
    """
    Write one .safetensors per (subject, week) with shape (1, 768).
    Returns dict: (subject_id, week) -> absolute path.
    """
    os.makedirs(out_emb_dir, exist_ok=True)
    path_map = {}
    for sid, week_map in emb_map.items():
        for wk, emb in week_map.items():
            tag   = WEEK_TAGS[wk]
            fpath = os.path.join(out_emb_dir, f"{sid}_{tag}.safetensors")
            path_map[(sid, wk)] = fpath
            if not dry_run:
                tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)  # (1, 768)
                save_file({"embeddings": tensor}, fpath)
    if not dry_run:
        print(f"[+] Exported {len(path_map)} .safetensors → {out_emb_dir}")
    return path_map


# ---------------------------------------------------------------------------
# Question / answer formatting
# ---------------------------------------------------------------------------

INPUT_WEEK = "Week 12"   # fixed input timepoint for all VQA records

LONG_FEATURE  = "PET intensity proxy (TBR-2)"
SHORT_FEATURE = "genotype"


def _tbr_str(val):
    return f"{val:.2f}"


def _genotype(sid):
    parts = sid.split("_")
    return parts[1] if len(parts) >= 2 else "unknown"


def _max_delta_weeks(input_wk, future_weeks):
    """Total span from input week to the last future week."""
    return WEEK_NUM[future_weeks[-1]] - WEEK_NUM[input_wk]


def _tbr_trajectory_str(input_wk, future_weeks, tbr_by_week):
    """
    Build the trajectory clause: 'Week 0: X, Week 3: Y, Week 8: Z'
    where week labels are relative deltas from the input week.
    All future weeks are included; weeks missing TBR data show 'NA'.
    """
    parts = []
    for wk in future_weeks:
        delta = WEEK_NUM[wk] - WEEK_NUM[input_wk]
        val   = tbr_by_week.get(wk)
        parts.append(f"Week {delta}: {_tbr_str(val) if val is not None else 'NA'}")
    return ", ".join(parts)


def format_tbr_qa(sid, input_wk, future_weeks, tbr_map):
    """Long topic: TBR trajectory from input_wk over all future_weeks."""
    tbr_by_week = tbr_map.get(sid, {})
    # Require at least one future week with actual TBR data
    if not any(wk in tbr_by_week for wk in future_weeks):
        return None, None, None

    n_weeks  = _max_delta_weeks(input_wk, future_weeks)
    traj_str = _tbr_trajectory_str(input_wk, future_weeks, tbr_by_week)
    question = f"Predict the mouse's trajectory of {LONG_FEATURE} over the next {n_weeks} weeks?"
    answer   = f"The predicted trajectory for {LONG_FEATURE} - {traj_str}."
    # Raw slot-fill values, kept alongside the rendered text so a paraphrase
    # bank (crump_aug_dataset.csv) can re-fill a different template without
    # having to regex the values back out of the rendered answer.
    template_values = {"long_feature": LONG_FEATURE, "n_weeks": n_weeks, "trajectory": traj_str}
    return question, answer, template_values


def format_genotype_qa(sid):
    """Short topic: eventual atherosclerosis status."""
    geno     = _genotype(sid)
    status   = geno
    question = f"What will be the eventual mouse status for {SHORT_FEATURE}?"
    answer   = f"The mouse genotype is {status}."
    template_values = {"short_feature": SHORT_FEATURE, "status": status}
    return question, answer, template_values


def format_combined_qa(sid, input_wk, future_weeks, tbr_map):
    """Combined: TBR trajectory + eventual atherosclerosis status."""
    tbr_by_week = tbr_map.get(sid, {})
    if not any(wk in tbr_by_week for wk in future_weeks):
        return None, None, None

    geno     = _genotype(sid)
    status   = geno
    n_weeks  = _max_delta_weeks(input_wk, future_weeks)
    traj_str = _tbr_trajectory_str(input_wk, future_weeks, tbr_by_week)

    question = (f"Predict the mouse's trajectory of {LONG_FEATURE} over the next {n_weeks} weeks "
                f"and the eventual status of {SHORT_FEATURE}?")
    answer = (f"The predicted trajectory for {LONG_FEATURE} - {traj_str}. "
              f"The mouse genotype is {status}.")
    template_values = {
        "long_feature": LONG_FEATURE, "n_weeks": n_weeks, "trajectory": traj_str,
        "short_feature": SHORT_FEATURE, "status": status,
    }
    return question, answer, template_values


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def build_record(pid, input_wk, future_weeks, question, answer, path_map, qid,
                 content_type="trajectory", template_values=None, tbr_values=None):
    def _path(wk):
        return path_map.get((pid, wk), "")

    return {
        "target_schema_version": 1,
        "target_name": "pet_midband_p95_trimmed_median_proxy",
        "pid":                pid,
        "qid":                qid,
        "input_week":         input_wk,
        "future_weeks":       future_weeks,
        # embedding_path_ts0 = the scan the model actually receives
        "embedding_path_ts0": _path(input_wk),
        # future scan paths stored for reference / multitask use
        "embedding_path_ts1": _path(future_weeks[0]) if len(future_weeks) > 0 else "",
        "embedding_path_ts2": _path(future_weeks[1]) if len(future_weeks) > 1 else "",
        "embedding_path_ts3": _path(future_weeks[2]) if len(future_weeks) > 2 else "",
        "question":           question,
        "answer":             answer,
        # "genotype" | "tbr" | "combined". Previously hardcoded to "trajectory" for
        # every record, which left question-text matching as the only way to tell
        # the three question types apart downstream.
        "content_type":       content_type,
        # Raw slot-fill values (short_feature/status/long_feature/n_weeks/trajectory)
        # so a paraphrase bank can re-render question/answer without parsing them
        # back out of rendered text. See scripts/dataset/create_mouse_vqa_dataset_from_openai.py.
        "template_values":    template_values or {},
        "answer_vqa_numeric": {
            "genotype": 1 if _genotype(pid) == "KO" else 0,
            "tbr": list(tbr_values) if tbr_values is not None else [-1.0]*4,
        },
    }


# ---------------------------------------------------------------------------
# Per-subject record generation
# ---------------------------------------------------------------------------

def get_records_for_subject(sid, emb_map, tbr_map, path_map, qid):
    """
    All records use Week 12 as the fixed input timepoint.
    Generates: genotype-only, TBR-only (if future TBR data exists),
    and combined (if future TBR data exists).
    Skips subjects without a Week 12 scan.
    """
    if INPUT_WEEK not in emb_map[sid]:
        return [], qid

    future_weeks = WEEK_ORDER[1:]  # Query horizons never depend on future attendance.
    # Preserve the existing eligible observation mask, including availability of
    # the visit in the embedding cohort. Missingness affects targets only.
    observed = {wk: value for wk,value in tbr_map.get(sid,{}).items() if wk in emb_map[sid]}
    tbr_map = {sid: observed}
    target_values = [float(_tbr_str(observed[w])) if w in observed else -1.0 for w in future_weeks] + [-1.0]

    if not future_weeks:
        return [], qid

    records = []

    # 1. Genotype / eventual status (always)
    q, a, tv = format_genotype_qa(sid)
    records.append(build_record(sid, INPUT_WEEK, future_weeks, q, a, path_map, qid,
                                content_type="genotype", template_values=tv))
    qid += 1

    # 2. TBR trajectory (only if future TBR data exists)
    q, a, tv = format_tbr_qa(sid, INPUT_WEEK, future_weeks, tbr_map)
    if q is not None:
        records.append(build_record(sid, INPUT_WEEK, future_weeks, q, a, path_map, qid,
                                    content_type="tbr", template_values=tv, tbr_values=target_values))
        qid += 1

    # 3. Combined TBR trajectory + eventual status
    q, a, tv = format_combined_qa(sid, INPUT_WEEK, future_weeks, tbr_map)
    if q is not None:
        records.append(build_record(sid, INPUT_WEEK, future_weeks, q, a, path_map, qid,
                                    content_type="combined", template_values=tv, tbr_values=target_values))
        qid += 1

    return records, qid


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------

def make_random_splits(all_subjects, val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=SEED):
    """Stratified split — WT and KO subjects split independently to balance genotypes."""
    rng = np.random.RandomState(seed)
    wt  = sorted(s for s in all_subjects if "_WT_" in s)
    ko  = sorted(s for s in all_subjects if "_KO_" in s)

    def _split(sids):
        sids = list(sids)
        rng.shuffle(sids)
        n      = len(sids)
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))
        return sids[n_test + n_val:], sids[n_test:n_test + n_val], sids[:n_test]

    wt_train, wt_val, wt_test = _split(wt)
    ko_train, ko_val, ko_test = _split(ko)
    return (
        set(wt_train + ko_train),
        set(wt_val   + ko_val),
        set(wt_test  + ko_test),
    )


def split_records(all_records, train_sids, val_sids, test_sids):
    return (
        [r for r in all_records if r["pid"] in train_sids],
        [r for r in all_records if r["pid"] in val_sids],
        [r for r in all_records if r["pid"] in test_sids],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    out_dir     = args.output_dir
    out_emb_dir = os.path.join(out_dir, "embeddings")
    os.makedirs(out_dir, exist_ok=True)

    emb_map = load_embeddings(args.embeddings)
    tbr_map = load_tbr(args.tbr_csv)
    if not args.dry_run:
        atomic_json(Path(out_dir)/"input_manifest.json", dict(
            target_schema_version=1, embeddings_path=str(Path(args.embeddings).resolve()),
            embeddings_sha256=file_sha256(args.embeddings), tbr_csv_path=str(Path(args.tbr_csv).resolve()),
            mouse_manifest_sha256=file_sha256(args.manifest),
            tbr_csv_sha256=file_sha256(args.tbr_csv), builder_sha256=file_sha256(__file__),
            target_column=TBR_COL, target_rounding_decimals=2,
            future_weeks=WEEK_ORDER[1:], identity_rule="filename order LL, LR, UL, UR plus quadrant_overrides.yaml",
            overrides_sha256=file_sha256(Path(__file__).resolve().parents[1]/"quadrant_overrides.yaml")))

    # NaF subjects with enough timepoints to form at least one (input → future) pair
    valid_sids = sorted([
        sid for sid, wm in emb_map.items()
        if sid.startswith("NaF") and len(wm) >= args.min_timepoints
    ])
    print(f"[i] Eligible subjects (NaF, ≥{args.min_timepoints} timepoints): {len(valid_sids)}")

    if args.dry_run:
        total_records = 0
        for sid in valid_sids:
            available  = [wk for wk in WEEK_ORDER if wk in emb_map[sid]]
            tbr_wks    = sorted(tbr_map.get(sid, {}).keys())
            if INPUT_WEEK not in available:
                print(f"    {sid:20s}  weeks: {available}  tbr: {tbr_wks}  ~0 records (no Week 12)")
                continue
            future     = [wk for wk in available if wk != INPUT_WEEK]
            has_tbr    = any(wk in tbr_map.get(sid, {}) for wk in future)
            est        = 1 + (2 if has_tbr else 0)   # genotype + TBR + combined
            total_records += est
            print(f"    {sid:20s}  weeks: {available}  tbr: {tbr_wks}  ~{est} records")
        print(f"\n[dry-run] Estimated total records: {total_records}")
        return

    # Export .safetensors
    path_map = export_safetensors(emb_map, out_emb_dir)

    # Build all records
    all_records = []
    qid = 0
    for sid in valid_sids:
        records, qid = get_records_for_subject(sid, emb_map, tbr_map, path_map, qid)
        all_records.extend(records)

    print(f"[i] Total records: {len(all_records)}")
    from eval_stats import join_mouse_group_ids
    groups = join_mouse_group_ids([r['pid'] for r in all_records],
                                  [r['input_week'] for r in all_records],args.manifest)
    for record, group in zip(all_records,groups):
        record['acquisition_group'] = str(group)

    # Count by question type and input week
    from collections import Counter
    week_counts  = Counter(r["input_week"] for r in all_records)
    print(f"    By input week : {dict(week_counts)}")

    # Split
    train_sids, val_sids, test_sids = make_random_splits(sorted({r['pid'] for r in all_records}))
    train_records, val_records, test_records = split_records(
        all_records, train_sids, val_sids, test_sids
    )
    print(f"[i] Train: {len(train_records)} records ({len(train_sids)} subjects) | "
          f"Val: {len(val_records)} records ({len(val_sids)} subjects) | "
          f"Test: {len(test_records)} records ({len(test_sids)} subjects)")

    # Save
    for fname, records in [
        ("mouse_train_vqa_traj.json", train_records),
        ("mouse_val_vqa_traj.json",   val_records),
        ("mouse_test_vqa_traj.json",  test_records),
        ("mouse_all_vqa_traj.json",   all_records),
    ]:
        fpath = os.path.join(out_dir, fname)
        atomic_json(fpath, records)
        print(f"[+] {fname}  ({len(records)} records)")


if __name__ == "__main__":
    main()
