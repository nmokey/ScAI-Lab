"""
Single train/val run for quick metric checks — no LOSO CV.
Uses the train/val split already defined in viz_emb_params_mouse.yml.

Run from vlm/ directory:
    CUDA_VISIBLE_DEVICES=6 python run/run_mouse_vlm_single.py
"""

import sys
import os
# Entry-point path setup: add vlm/ to sys.path so bare imports (utils, data, model)
# resolve when this script is run as `cd vlm && python run/run_mouse_vlm_single.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.cuda.set_device(0)

from utils.misc_utils import load_yaml
from utils.run_utils import get_model
from data.eval import calculate_mouse_metrics

_VLM_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML_FILE = os.path.join(_VLM_DIR, "yaml", "viz_emb_params_mouse.yml")
OUT_DIR   = "/data1/Processed_NIfTI_Test/embeddings/vlm/runs/mouse_vlm_single"

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    import yaml
    params = load_yaml(YAML_FILE)
    params["exp"]["output_dir"] = OUT_DIR
    params["inf"]["model_name"] = os.path.join(OUT_DIR, params["train"]["save_model_name"])
    params["inf"]["save_file"]  = "vqa.json"
    params["inf"]["results_file"] = "val.json"

    patched_yaml = os.path.join(OUT_DIR, "params.yml")
    with open(patched_yaml, "w") as f:
        yaml.dump(params, f)

    model = get_model("viz_emb", patched_yaml)
    model.setup()
    model.run()
    model.evaluate()

    print(f"\n[+] Done. Results → {OUT_DIR}/val.json")
