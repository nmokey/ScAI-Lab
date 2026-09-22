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
# Entry-point path setup: add vlm/ to sys.path so bare imports (utils, data, model)
# resolve when this script is run as `cd vlm && python run/run_mouse_vlm_loso.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import json
import hashlib
import tempfile
import shutil
from pathlib import Path
from contextlib import nullcontext

import torch
import transformers

from utils.misc_utils import load_yaml
from utils.run_utils import get_model
from data.eval import calculate_mouse_metrics
from data.validated_metrics import score_predictions
from utils.research_io import atomic_json, verify_forecast_manifest
from utils.run_contract import (make_run_spec, freeze_run, completed_fold,
                                mark_fold_complete, collect_complete_folds)


_VLM_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML_FILE = os.path.join(_VLM_DIR, "yaml", "viz_emb_params_mouse_nogeno_long_repaired.yml")
ALL_DATA    = "/data1/Processed_NIfTI_Test/embeddings/vlm/validated_20260914/mouse_all_vqa_traj.json"
LOSO_DIR    = "/data1/Processed_NIfTI_Test/embeddings/vlm/runs/mouse_vlm_loso_checkpoint_qhead"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start-fold", type=int, default=0,
                   help="Resume from this fold index (0-based)")
    p.add_argument("--end-fold", type=int, default=None, help="Exclusive fold limit for a correctness smoke run")
    p.add_argument("--all-data", default=ALL_DATA)
    p.add_argument("--output-dir", default=LOSO_DIR)
    p.add_argument("--yaml", default=YAML_FILE,
                   help="Path to YAML config (default: viz_emb_params_mouse_nogeno_long_repaired.yml)")
    p.add_argument("--inner-val", action="store_true",
                   help="F2: carve one TRAINING subject per fold out as the Trainer's "
                        "eval_dataset (removed from training). Without this, eval_dataset "
                        "IS the held-out test subject -- harmless while "
                        "load_best_model_at_end is false, but every eval_loss curve is then "
                        "a test-set curve. Required if you turn load_best_model_at_end on. "
                        "Costs one training subject (30 instead of 31).")
    return p.parse_args()


def write_fold_jsons(all_records, held_out_sid, tmp_dir, inner_val_sid=None):
    """
    Split records for one LOSO fold.

    held_out_sid  -> val.json   (the test subject; becomes inf_data_path)
    inner_val_sid -> inner.json (optional; a TRAINING subject removed from train.json
                                 and used as the Trainer's eval_dataset -- F2)
    everything else -> train.json
    """
    held  = [r for r in all_records if r["pid"] == held_out_sid]
    inner = [r for r in all_records if r["pid"] == inner_val_sid] if inner_val_sid else []
    train = [r for r in all_records if r["pid"] not in (held_out_sid, inner_val_sid)]
    train_path = os.path.join(tmp_dir, "train.json")
    val_path   = os.path.join(tmp_dir, "val.json")
    atomic_json(train_path, train)
    atomic_json(val_path, held)
    inner_path = None
    if inner:
        inner_path = os.path.join(tmp_dir, "inner.json")
        with open(inner_path, "w") as f:
            json.dump(inner, f)
    return train_path, val_path, inner_path


def pick_inner_val_subject(subjects, held_out_sid, fold_idx):
    """
    Deterministic inner-validation subject for a fold: the next subject in sorted
    order after the held-out one (wrapping), so every fold carves out a different
    subject and the choice is reproducible without a seed.
    """
    pool = [s for s in subjects if s != held_out_sid]
    return pool[fold_idx % len(pool)]


def run_fold(sid, fold_idx, all_records, output_dir, base_params, inner_val=False, subjects=None, run_id=None):
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx+1}: held-out subject = {sid}")
    print(f"{'='*60}")

    fold_out = os.path.join(output_dir, f"fold_{fold_idx:02d}_{sid}")
    os.makedirs(fold_out, exist_ok=True)

    truth=[r for r in all_records if r['pid']==sid]
    if run_id:
        existing=completed_fold(fold_out,run_id,truth)
        if existing is not None:
            print(f"Verified completed fold {sid}; reusing saved predictions")
            return existing
    split_dir=os.path.join(fold_out,'splits')
    os.makedirs(split_dir,exist_ok=True)
    with nullcontext(split_dir) as tmp_dir:
        inner_sid = pick_inner_val_subject(subjects, sid, fold_idx) if inner_val else None
        train_path, val_path, inner_path = write_fold_jsons(all_records, sid, tmp_dir, inner_sid)
        if inner_sid:
            print(f"  inner-val subject (Trainer eval_dataset): {inner_sid}  "
                  f"-- removed from training; test subject {sid} is never seen during training")

        # Patch params for this fold. deepcopy, not a one-level dict copy: the
        # shallow version shared nested dicts (e.g. inf.decoding_kwargs) across
        # every fold, so any future nested patch would leak between folds.
        params = copy.deepcopy(base_params)
        params["exp"]["output_dir"]         = fold_out
        params["data"]["data_path"]         = train_path
        params["data"]["inf_data_path"]     = val_path
        # F2: with an inner-val subject the Trainer evaluates on a training-fold
        # subject, not on the test subject, so eval curves are honest and
        # load_best_model_at_end becomes a legitimate option.
        params["data"]["eval_data_path"]    = inner_path
        nested_root=params['data'].get('nested_forecast_root')
        if params['data']['img_tokens']==4:
            if not nested_root:
                raise ValueError('Longitudinal evaluation requires nested_forecast_root; global OOF tokens are invalid')
            fold_tokens=Path(nested_root)/sid
            forecast_manifest=verify_forecast_manifest(fold_tokens,sid,subjects)
            from safetensors.torch import load_file
            for query,path in {r['pid']:r['embedding_path_ts0'] for r in all_records}.items():
                vector=load_file(path)['embeddings'].contiguous().numpy()
                if hashlib.sha256(vector.tobytes()).hexdigest()!=forecast_manifest['baseline_sha256'][query]:
                    raise ValueError(f'Forecast baseline differs from VLM baseline for {query}')
            params['data']['predicted_emb_dir']=str(fold_tokens)
        elif params['data'].get('predicted_emb_dir'):
            raise ValueError('Baseline arm must use only its observed token')
        # Technical restart uses only a complete compact Trainer checkpoint.
        checkpoints=sorted(Path(fold_out).glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
        if checkpoints:
            valid=[p for p in checkpoints if all((p/f).exists() for f in
                   ('vlm_config.json','trainer_state.json','optimizer.pt','scheduler.pt','rng_state.pth'))]
            if not valid:raise ValueError('No complete training checkpoint is available for this partial fold')
            params['train']['resume_from_checkpoint']=str(valid[-1])
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
        predictions=json.loads(Path(fold_out,'vqa.json').read_text())
        training=json.loads(Path(train_path).read_text())
        score_predictions(predictions,truth,training)
        if run_id:mark_fold_complete(fold_out,run_id,params['train']['save_model_name'])

        # Free GPU memory between folds
        torch.cuda.empty_cache()

    # Keep the compact final adapter/head checkpoint for verification and rescoring.
    # Intermediate Trainer checkpoints can be removed once final inference succeeds.
    for name in os.listdir(fold_out):
        full = os.path.join(fold_out, name)
        if name == "runs" or name.startswith("checkpoint-"):
            shutil.rmtree(full, ignore_errors=True)

    pred_path = os.path.join(fold_out, "vqa.json")
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            return json.load(f)
    return []


def main():
    # Pin to logical device 0 (the physical GPU selected by CUDA_VISIBLE_DEVICES) so
    # every from_pretrained call inside each fold lands on the same device. Done here
    # rather than at import time: as a module-level side effect it raised on CPU-only
    # machines, making this entry point impossible to import or test off-GPU.
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.all_data) as f:
        all_records = json.load(f)

    subjects = sorted(set(r["pid"] for r in all_records))
    print(f"[i] {len(subjects)} subjects, {len(all_records)} total records")
    print(f"[i] Running {len(subjects)} LOSO folds")

    base_params = load_yaml(args.yaml)
    if base_params["train"].get("load_best_model_at_end") and not args.inner_val:
        raise SystemExit("load_best_model_at_end is true in the yaml but --inner-val is not set: "
                         "that selects checkpoints on the test subject (F2). Pass --inner-val "
                         "or set load_best_model_at_end: false.")

    if args.inner_val:
        raise ValueError('The repaired fixed-epoch protocol uses all outer-training subjects; inner selection requires a separate nested protocol')
    if base_params['train'].get('pool_at')!='question_eos':
        raise ValueError('Validated runs require question-only supervision')
    if base_params['train'].get('load_best_model_at_end'):
        raise ValueError('Validated runs use the prespecified final epoch')
    spec=make_run_spec(base_params,all_records,args.all_data)
    run_id=freeze_run(args.output_dir,spec)
    # Prevent two workers from modifying the same run concurrently.
    import fcntl
    with open(Path(args.output_dir)/'.run.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        end=len(subjects) if args.end_fold is None else args.end_fold
        if not 0<=args.start_fold<=end<=len(subjects):raise ValueError('Invalid fold range')
        for index,sid in enumerate(subjects):
            if index<args.start_fold and completed_fold(Path(args.output_dir)/f'fold_{index:02d}_{sid}',run_id,
                                                        [r for r in all_records if r['pid']==sid]) is None:
                raise ValueError('Cannot skip an unfinished earlier fold')
        for fold_idx in range(args.start_fold,end):
            sid=subjects[fold_idx]
            run_fold(sid,fold_idx,all_records,args.output_dir,base_params,
                     inner_val=False,subjects=subjects,run_id=run_id)
            _,complete=collect_complete_folds(args.output_dir,subjects,all_records,run_id)
        predictions,complete=collect_complete_folds(args.output_dir,subjects,all_records,run_id)
        if complete:
            results=score_predictions(predictions,all_records)
            atomic_json(Path(args.output_dir)/'loso_results.json',results)
            print(f"Complete validated LOSO: {results['tbr_reg_subject_mae']} subject MAE")
        else:
            print('Partial run saved separately; no complete-study metrics produced')


if __name__ == "__main__":
    main()
