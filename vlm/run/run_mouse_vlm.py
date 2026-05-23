"""
Train and evaluate the mouse trajectory VLM (text generation, no multitask head).

Run from vlm/ directory:
    cd ~/ScAI-Lab/vlm
    python run/run_mouse_vlm.py
"""

import sys
import os
# Entry-point path setup: add vlm/ to sys.path so bare imports (utils, data, model)
# resolve when this script is run as `cd vlm && python run/run_mouse_vlm.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.run_utils import get_model

if __name__ == "__main__":
    model = get_model("viz_emb", "viz_emb_params_mouse.yml")
    model.setup()
    model.run()
    model.evaluate()
