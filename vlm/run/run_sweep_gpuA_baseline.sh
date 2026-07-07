#!/usr/bin/env bash
# Overnight sweep — GPU A: baseline VLM (ts0 only), epoch ablation.
# Runs 15/20/25/30/40 epochs sequentially, each a full 32-fold LOSO CV.
#
# Usage (from vlm/):
#   cd ~/ScAI-Lab/vlm
#   CUDA_VISIBLE_DEVICES=0 bash run/run_sweep_gpuA_baseline.sh
#
# Config for all runs: exp3 (r=4, q/v LoRA, multitask_wt=5, z-scored TBR),
# img_tokens=1, no predicted_emb_dir. Only epochs vary.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLM_DIR="$(dirname "$SCRIPT_DIR")"
RUNNER="$SCRIPT_DIR/run_mouse_vlm_loso.py"
YAML_DIR="$VLM_DIR/yaml"
RUN_ROOT="/data1/Processed_NIfTI_Test/embeddings/vlm/runs"

EPOCHS=(15 20 25 30 40)

echo "========================================================"
echo " GPU A sweep — BASELINE VLM (ts0 only)"
echo " Epochs: ${EPOCHS[*]}"
echo " Started: $(date)"
echo "========================================================"

for ep in "${EPOCHS[@]}"; do
    name="base_ep${ep}"
    yaml_path="$YAML_DIR/viz_emb_params_mouse_${name}.yml"
    out_dir="$RUN_ROOT/mouse_vlm_${name}"
    log_file="$out_dir/loso_run.log"
    mkdir -p "$out_dir"

    echo ""
    echo "--------------------------------------------------------"
    echo " [GPU A] baseline ${ep} epochs"
    echo " YAML: $yaml_path"
    echo " Out : $out_dir"
    echo " Start: $(date)"
    echo "--------------------------------------------------------"

    python "$RUNNER" --yaml "$yaml_path" --output-dir "$out_dir" 2>&1 | tee "$log_file"

    results="$out_dir/loso_results.json"
    if [[ -f "$results" ]]; then
        echo "[DONE] baseline ${ep}ep:"
        python3 - "$results" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"  genotype_acc        : {r.get('genotype_acc')}")
print(f"  tbr_reg_overall_mae : {r.get('tbr_reg_overall_mae')}")
print(f"  tbr_reg_overall_r   : {r.get('tbr_reg_overall_r')}")
for slot, m in r.get('tbr_reg_by_slot', {}).items():
    print(f"  {slot:12s}: MAE={m['mae']}, r={m['pearson_r']}, n={m['n']}")
PYEOF
    else
        echo "[WARN] baseline ${ep}ep — no loso_results.json"
    fi
    echo " End: $(date)"
done

echo ""
echo "========================================================"
echo " GPU A sweep complete: $(date)"
echo "========================================================"
