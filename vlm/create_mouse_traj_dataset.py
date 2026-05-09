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

One VQA record is generated per (subject, input_timepoint) pair where at least
one future timepoint exists. A subject with W12/W15/W18/W20 contributes 3
records (W12→future, W15→future, W18→future). This lets the model answer
trajectory questions from any point in the disease timeline, not just baseline.

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

import numpy as np
import pandas as pd
from safetensors.torch import save_file
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEEK_ORDER = ["Week 12", "Week 15", "Week 18", "Week 20"]
WEEK_TAGS  = {"Week 12": "ts0", "Week 15": "ts1", "Week 18": "ts2", "Week 20": "ts3"}

TBR_COL  = "tbr2_p95_median"
WEEK_NUM = {"Week 12": 12, "Week 15": 15, "Week 18": 18, "Week 20": 20}

DEFAULT_EMB_NPZ  = "/data1/Processed_NIfTI_Test/embeddings/raddino/raddino_embeddings.npz"
DEFAULT_TBR_CSV  = "/data1/Processed_NIfTI_Test/embeddings/longitudinal/tbr_features_NaF.csv"
DEFAULT_OUT_DIR  = "/data1/Processed_NIfTI_Test/embeddings/vlm"

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
        emb_map.setdefault(sid, {})[wk] = emb

    print(f"[i] Loaded {len(embeddings)} embeddings across {len(emb_map)} subjects")
    return emb_map


def load_tbr(csv_path):
    """Returns dict: subject_id -> {week -> float}"""
    df = pd.read_csv(csv_path)
    tbr_map = {}
    for _, row in df.iterrows():
        tbr_map.setdefault(str(row["subject_id"]), {})[str(row["week"])] = float(row[TBR_COL])
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

def _tbr_str(val):
    return f"{val:.1f}"


def _genotype(sid):
    parts = sid.split("_")
    return parts[1] if len(parts) >= 2 else "unknown"


def _week_delta_str(input_wk, future_weeks):
    """E.g. 'over the next 3 and 8 weeks' from Week 12 → [Week 15, Week 20]."""
    deltas = [WEEK_NUM[wk] - WEEK_NUM[input_wk] for wk in future_weeks]
    delta_strs = [f"{d} week{'s' if d != 1 else ''}" for d in deltas]
    if len(delta_strs) == 1:
        return f"over the next {delta_strs[0]}"
    return "over the next " + ", ".join(delta_strs[:-1]) + f" and {delta_strs[-1]}"


def format_tbr_qa(sid, input_wk, future_weeks, tbr_map):
    """Long topic: TBR trajectory from input_wk into future_weeks."""
    tbr_by_week = tbr_map.get(sid, {})
    future_tbr_weeks = [wk for wk in future_weeks if wk in tbr_by_week]

    if not future_tbr_weeks:
        return None, None  # no TBR data for future weeks — skip

    delta_str = _week_delta_str(input_wk, future_tbr_weeks)
    question  = (f"Given the scan at week {WEEK_NUM[input_wk]}, predict the mouse's "
                 f"aortic TBR trajectory {delta_str}?")

    parts = [f"Week {WEEK_NUM[wk]}: {_tbr_str(tbr_by_week[wk])}"
             for wk in future_tbr_weeks]
    answer = f"The predicted aortic TBR trajectory: {', '.join(parts)}."
    return question, answer


def format_genotype_qa(sid, input_wk):
    """Short topic: genotype."""
    geno     = _genotype(sid)
    question = (f"Given the scan at week {WEEK_NUM[input_wk]}, "
                f"what is the genotype of this mouse (WT or KO)?")
    if geno == "KO":
        answer = "The mouse is KO (ApoE knockout on high-fat diet), predisposed to atherosclerosis."
    else:
        answer = "The mouse is WT (wild-type), on regular chow with no induced atherosclerosis."
    return question, answer


def format_combined_qa(sid, input_wk, future_weeks, tbr_map):
    """Combined: TBR trajectory + genotype."""
    tbr_by_week      = tbr_map.get(sid, {})
    future_tbr_weeks = [wk for wk in future_weeks if wk in tbr_by_week]

    if not future_tbr_weeks:
        return None, None

    geno      = _genotype(sid)
    delta_str = _week_delta_str(input_wk, future_tbr_weeks)
    question  = (f"Given the scan at week {WEEK_NUM[input_wk]}, predict the mouse's "
                 f"aortic TBR trajectory {delta_str} and describe its genotype?")

    tbr_parts = [f"Week {WEEK_NUM[wk]}: {_tbr_str(tbr_by_week[wk])}"
                 for wk in future_tbr_weeks]
    geno_str  = ("KO (ApoE knockout), predisposed to atherosclerosis."
                 if geno == "KO" else
                 "WT (wild-type), no induced atherosclerosis.")
    answer = (f"The predicted aortic TBR trajectory: {', '.join(tbr_parts)}. "
              f"The mouse is {geno_str}")
    return question, answer


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def build_record(pid, input_wk, future_weeks, question, answer, path_map, qid):
    def _path(wk):
        return path_map.get((pid, wk), "")

    return {
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
        "content_type":       "trajectory",
        "answer_vqa_numeric": {
            "genotype": 1 if _genotype(pid) == "KO" else 0,
        },
    }


# ---------------------------------------------------------------------------
# Per-subject record generation
# ---------------------------------------------------------------------------

def get_records_for_subject(sid, emb_map, tbr_map, path_map, qid):
    """
    One record per (subject, input_week) pair where future timepoints exist.
    For each input week, generate: genotype-only, TBR-only (if TBR available),
    and combined (if TBR available).
    """
    available_weeks = [wk for wk in WEEK_ORDER if wk in emb_map[sid]]
    records = []

    for i, input_wk in enumerate(available_weeks[:-1]):      # exclude last — no future
        future_weeks = available_weeks[i + 1:]               # all subsequent weeks

        # 1. Genotype (always)
        q, a = format_genotype_qa(sid, input_wk)
        records.append(build_record(sid, input_wk, future_weeks, q, a, path_map, qid))
        qid += 1

        # 2. TBR trajectory (only if future TBR data exists)
        q, a = format_tbr_qa(sid, input_wk, future_weeks, tbr_map)
        if q is not None:
            records.append(build_record(sid, input_wk, future_weeks, q, a, path_map, qid))
            qid += 1

        # 3. Combined TBR + genotype
        q, a = format_combined_qa(sid, input_wk, future_weeks, tbr_map)
        if q is not None:
            records.append(build_record(sid, input_wk, future_weeks, q, a, path_map, qid))
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

    # NaF subjects with enough timepoints to form at least one (input → future) pair
    valid_sids = sorted([
        sid for sid, wm in emb_map.items()
        if sid.startswith("NaF") and len(wm) >= args.min_timepoints
    ])
    print(f"[i] Eligible subjects (NaF, ≥{args.min_timepoints} timepoints): {len(valid_sids)}")

    if args.dry_run:
        total_records = 0
        for sid in valid_sids:
            available = [wk for wk in WEEK_ORDER if wk in emb_map[sid]]
            tbr_wks   = sorted(tbr_map.get(sid, {}).keys())
            n_input   = len(available) - 1   # number of (input, future) pairs
            # estimate records: per input_wk: 1 genotype + up to 2 TBR-based
            has_tbr_future = any(
                any(wk in tbr_map.get(sid, {}) for wk in available[i+1:])
                for i in range(n_input)
            )
            est = n_input * (3 if has_tbr_future else 1)
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

    # Count by question type and input week
    from collections import Counter
    week_counts  = Counter(r["input_week"] for r in all_records)
    print(f"    By input week : {dict(week_counts)}")

    # Split
    train_sids, val_sids, test_sids = make_random_splits(valid_sids)
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
        with open(fpath, "w") as f:
            json.dump(records, f, indent=2)
        print(f"[+] {fname}  ({len(records)} records)")


if __name__ == "__main__":
    main()
