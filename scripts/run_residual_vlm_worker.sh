#!/usr/bin/env bash
set -euo pipefail
cd /home/ryab/ScAI-Lab
export CUDA_VISIBLE_DEVICES=6
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
export NUMBA_CACHE_DIR=/tmp/scai-numba MPLCONFIGDIR=/tmp/scai-matplotlib
export LD_LIBRARY_PATH=/home/ryab/miniconda3/envs/vlm_env/lib
audit=docs/audit_2026-09-14/residual_vlm
for seed in 0 1 2; do
  .venv-test/bin/python -u scripts/run_residual_vlm.py \
    --yaml "$audit/configs/residual_seed${seed}.yml" \
    --split docs/audit_2026-09-14/stratified/PROTOCOL.json \
    --all-data /data1/Processed_NIfTI_Test/embeddings/vlm/validated_20260914/mouse_all_vqa_traj.json \
    --original-run "/data1/Processed_NIfTI_Test/embeddings/vlm/runs/stratified_long_seed${seed}" \
    --output-dir "/data1/Processed_NIfTI_Test/embeddings/vlm/runs/stratified_residual_seed${seed}" \
    >> "$audit/seed${seed}.log" 2>&1
done
.venv-test/bin/python -u scripts/evaluate_residual_vlm.py > "$audit/verification.log" 2>&1
