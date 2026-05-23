"""
LOSO (Leave-One-Subject-Out) cross-validation for the mouse trajectory VLM.

For each of the 32 NaF subjects:
  - Train on the remaining 31 subjects
  - Evaluate on the held-out subject
  - Collect predictions into a single aggregated output file

Run from vlm/ directory:
    cd ~/ScAI-Lab/vlm
    CUDA_VISIBLE_DEVICES=0 python run/run_mouse_vlm_loso.py
    CUDA_VISIBLE_DEVICES=0 python run/run_mouse_vlm_loso.py --start-fold 10  # resume
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import tempfile
import shutil

import torch
import transformers

# Pin to logical device 0 (the physical GPU selected by CUDA_VISIBLE_DEVICES).
# This ensures every from_pretrained call inside each fold lands on the same device.
torch.cuda.set_device(0)

from utils.misc_utils import load_yaml
from utils.run_utils import get_model
from data.eval import calculate_mouse_metrics


_VLM_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML_FILE = os.path.join(_VLM_DIR, "yaml", "viz_emb_params_mouse.yml")
ALL_DATA    = "/data1/Processed_NIfTI_Test/embeddings/vlm/mouse_all_vqa_traj.json"
LOSO_DIR    = "/data1/Processed_NIfTI_Test/embeddings/vlm/runs/mouse_vlm_loso"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start-fold", type=int, default=0,
                   help="Resume from this fold index (0-based)")
    p.add_argument("--all-data", default=ALL_DATA)
    p.add_argument("--output-dir", default=LOSO_DIR)
    return p.parse_args()


def write_fold_jsons(all_records, held_out_sid, tmp_dir):
    train = [r for r in all_records if r["pid"] != held_out_sid]
    val   = [r for r in all_records if r["pid"] == held_out_sid]
    train_path = os.path.join(tmp_dir, "train.json")
    val_path   = os.path.join(tmp_dir, "val.json")
    with open(train_path, "w") as f:
        json.dump(train, f)
    with open(val_path, "w") as f:
        json.dump(val, f)
    return train_path, val_path


def run_fold(sid, fold_idx, all_records, output_dir, base_params):
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx+1}: held-out subject = {sid}")
    print(f"{'='*60}")

    fold_out = os.path.join(output_dir, f"fold_{fold_idx:02d}_{sid}")
    os.makedirs(fold_out, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_path, val_path = write_fold_jsons(all_records, sid, tmp_dir)

        # Patch params for this fold
        params = {k: dict(v) if isinstance(v, dict) else v
                  for k, v in base_params.items()}
        params["exp"]["output_dir"]         = fold_out
        params["data"]["data_path"]         = train_path
        params["data"]["inf_data_path"]     = val_path
        params["inf"]["model_name"]         = os.path.join(fold_out, params["train"]["save_model_name"])
        params["inf"]["train_gt_file"]      = train_path
        params["inf"]["save_file"]          = "vqa.json"
        params["inf"]["results_file"]       = "val.json"

        # Write patched YAML to tmp dir so get_model can load it
        import yaml
        patched_yaml = os.path.join(tmp_dir, "params.yml")
        with open(patched_yaml, "w") as f:
            yaml.dump(params, f)

        model = get_model("viz_emb", patched_yaml)
        model.setup()
        model.run()
        model.evaluate()

        # Free GPU memory between folds
        torch.cuda.empty_cache()

    # Delete model weights and checkpoints — only vqa.json/val.json are needed
    for name in os.listdir(fold_out):
        full = os.path.join(fold_out, name)
        if name in ("mouse_vlm_mdl", "runs") or name.startswith("checkpoint-"):
            shutil.rmtree(full, ignore_errors=True)

    pred_path = os.path.join(fold_out, "vqa.json")
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            return json.load(f)
    return []


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.all_data) as f:
        all_records = json.load(f)

    subjects = sorted(set(r["pid"] for r in all_records))
    print(f"[i] {len(subjects)} subjects, {len(all_records)} total records")
    print(f"[i] Running {len(subjects)} LOSO folds")

    base_params = load_yaml(YAML_FILE)

    # Aggregate predictions file (append across folds for resumability)
    agg_path = os.path.join(args.output_dir, "vqa_loso.json")
    if args.start_fold > 0 and os.path.exists(agg_path):
        with open(agg_path) as f:
            all_preds = json.load(f)
        print(f"[i] Resuming from fold {args.start_fold}, loaded {len(all_preds)} existing predictions")
    else:
        all_preds = []

    for fold_idx, sid in enumerate(subjects):
        if fold_idx < args.start_fold:
            continue

        fold_preds = run_fold(sid, fold_idx, all_records, args.output_dir, base_params)
        all_preds.extend(fold_preds)

        # Save after every fold in case of interruption
        with open(agg_path, "w") as f:
            json.dump(all_preds, f, indent=2)
        print(f"[+] Fold {fold_idx+1}/{len(subjects)} done — {len(fold_preds)} predictions, "
              f"{len(all_preds)} total saved")

    # Final metrics over all held-out predictions
    results_path = os.path.join(args.output_dir, "loso_results.json")
    calculate_mouse_metrics(
        gt_file=None,
        train_gt_file=args.all_data,
        pred_file=agg_path,
        out_file=results_path,
    )
    print(f"\n[+] LOSO complete. Aggregated predictions: {agg_path}")
    print(f"[+] Final metrics: {results_path}")


if __name__ == "__main__":
    main()
