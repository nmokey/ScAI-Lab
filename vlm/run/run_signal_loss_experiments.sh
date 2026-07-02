#!/usr/bin/env bash
# Run signal-loss investigation experiments sequentially.
#
# Each experiment is a full LOSO CV run with one variable changed from the
# exp3 baseline (r=4, q/v LoRA, multitask_wt=5, z-scored TBR, 10 epochs,
# LLaMA-3.1-8B). Results land in loso_results.json under each run dir.
#
# Usage (from repo root):
#   cd ~/ScAI-Lab/vlm
#   CUDA_VISIBLE_DEVICES=0 bash run/run_signal_loss_experiments.sh
#   CUDA_VISIBLE_DEVICES=0 bash run/run_signal_loss_experiments.sh --skip ep20,ep50
#
# Logs: one .log file per experiment in the run output directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLM_DIR="$(dirname "$SCRIPT_DIR")"
RUNNER="$SCRIPT_DIR/run_mouse_vlm_loso.py"
YAML_DIR="$VLM_DIR/yaml"

EXPERIMENTS=(
    "ep20:viz_emb_params_mouse_ep20.yml:/data1/Processed_NIfTI_Test/embeddings/vlm/runs/mouse_vlm_ep20"
    "ep50:viz_emb_params_mouse_ep50.yml:/data1/Processed_NIfTI_Test/embeddings/vlm/runs/mouse_vlm_ep50"
    "ep100:viz_emb_params_mouse_ep100.yml:/data1/Processed_NIfTI_Test/embeddings/vlm/runs/mouse_vlm_ep100"
    "tinyllama:viz_emb_params_mouse_tinyllama.yml:/data1/Processed_NIfTI_Test/embeddings/vlm/runs/mouse_vlm_tinyllama"
)

# Parse --skip flag
SKIP=""
for arg in "$@"; do
    case $arg in
        --skip=*) SKIP="${arg#--skip=}" ;;
        --skip)   shift; SKIP="$1" ;;
    esac
done

should_skip() {
    local name="$1"
    [[ ",$SKIP," == *",$name,"* ]]
}

echo "========================================================"
echo " Signal-loss investigation experiments"
echo " Baseline (exp3, 10 epochs): genotype_acc=0.531, TBR_reg_delta3wk_MAE=6.123, r=0.447"
echo "========================================================"

for entry in "${EXPERIMENTS[@]}"; do
    IFS=: read -r name yaml_file out_dir <<< "$entry"

    if should_skip "$name"; then
        echo ""
        echo "[SKIP] $name"
        continue
    fi

    yaml_path="$YAML_DIR/$yaml_file"
    log_file="$out_dir/loso_run.log"
    mkdir -p "$out_dir"

    echo ""
    echo "--------------------------------------------------------"
    echo " Experiment : $name"
    echo " YAML       : $yaml_path"
    echo " Output dir : $out_dir"
    echo " Log        : $log_file"
    echo " Started    : $(date)"
    echo "--------------------------------------------------------"

    python "$RUNNER" \
        --yaml "$yaml_path" \
        --output-dir "$out_dir" \
        2>&1 | tee "$log_file"

    results="$out_dir/loso_results.json"
    if [[ -f "$results" ]]; then
        echo ""
        echo "[DONE] $name — key metrics:"
        python3 - "$results" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"  genotype_acc          : {r.get('genotype_acc')}")
print(f"  tbr_reg_overall_mae   : {r.get('tbr_reg_overall_mae')}")
print(f"  tbr_reg_overall_r     : {r.get('tbr_reg_overall_r')}")
print(f"  tbr_reg_overall_r2    : {r.get('tbr_reg_overall_r2')}")
slots = r.get('tbr_reg_by_slot', {})
for slot, m in slots.items():
    print(f"  {slot:20s}: MAE={m['mae']}, r={m['pearson_r']}, R²={m['r2']}, n={m['n']}")
PYEOF
    else
        echo "[WARN] $name — loso_results.json not found"
    fi

    echo " Finished   : $(date)"
done

echo ""
echo "========================================================"
echo " All experiments complete. Compare results:"
for entry in "${EXPERIMENTS[@]}"; do
    IFS=: read -r name _ out_dir <<< "$entry"
    echo "  $name: $out_dir/loso_results.json"
done
echo "========================================================"
