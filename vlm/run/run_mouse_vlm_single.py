"""
Single train/val run for quick metric checks — no LOSO CV.
Uses the train/val split already defined in the yaml's data_path/inf_data_path.

Run from vlm/ directory:
    CUDA_VISIBLE_DEVICES=6 python run/run_mouse_vlm_single.py
    CUDA_VISIBLE_DEVICES=6 python run/run_mouse_vlm_single.py --yaml yaml/viz_emb_params_mouse_nogeno_long_aug.yml --output-dir /data1/Processed_NIfTI_Test/embeddings/vlm/runs/mouse_vlm_single_aug
"""

import sys
import os
# Entry-point path setup: add vlm/ to sys.path so bare imports (utils, data, model)
# resolve when this script is run as `cd vlm && python run/run_mouse_vlm_single.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
if torch.cuda.is_available():
    torch.cuda.set_device(0)

from utils.misc_utils import load_yaml
from utils.run_utils import get_model
from data.eval import calculate_mouse_metrics

_VLM_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML_FILE     = os.path.join(_VLM_DIR, "yaml", "viz_emb_params_mouse_nogeno_base_repaired.yml")
DEFAULT_OUT_DIR = "/data1/Processed_NIfTI_Test/embeddings/vlm/runs/mouse_vlm_single_checkpoint_qhead"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", default=YAML_FILE,
                   help="Path to YAML config (default: viz_emb_params_mouse_nogeno_base_repaired.yml)")
    p.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    import yaml
    params = load_yaml(args.yaml)
    if params['data']['img_tokens'] != 1:
        raise ValueError('Use the LOSO runner for longitudinal input: it resolves and verifies each nested forecast fold')
    params["exp"]["output_dir"] = out_dir
    params["inf"]["model_name"] = os.path.join(out_dir, params["train"]["save_model_name"])
    params["inf"]["save_file"]  = "vqa.json"
    params["inf"]["results_file"] = "val.json"

    patched_yaml = os.path.join(out_dir, "params.yml")
    with open(patched_yaml, "w") as f:
        yaml.dump(params, f)

    model = get_model("viz_emb", patched_yaml)
    model.setup()
    model.run()
    model.evaluate()

    print(f"\n[+] Done. Results → {out_dir}/val.json")
