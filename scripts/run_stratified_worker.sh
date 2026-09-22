#!/usr/bin/env bash
# Authorized fixed queue: one independent four-fold run at a time per GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${1:?GPU index required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
export NUMBA_CACHE_DIR=/tmp/scai-numba
export LD_LIBRARY_PATH=/home/ryab/miniconda3/envs/vlm_env/lib
audit=docs/audit_2026-09-14/stratified
queue="$audit/queue.txt"
while true; do
  exec 9>"${queue}.lock"
  flock 9
  job=$(head -n 1 "$queue")
  if [[ -z "$job" ]]; then flock -u 9; break; fi
  tail -n +2 "$queue" > "${queue}.next"
  mv "${queue}.next" "$queue"
  flock -u 9
  read -r arm seed <<< "$job"
  run="stratified_${arm}_seed${seed}"
  printf '%s started %s GPU=%s\n' "$(date -Is)" "$run" "$CUDA_VISIBLE_DEVICES"
  if .venv-test/bin/python -u scripts/run_stratified_vlm.py \
    --yaml "$audit/configs/${arm}_seed${seed}.yml" \
    --split "$audit/PROTOCOL.json" \
    --all-data /data1/Processed_NIfTI_Test/embeddings/vlm/validated_20260914/mouse_all_vqa_traj.json \
    --output-dir "/data1/Processed_NIfTI_Test/embeddings/vlm/runs/${run}" \
    >> "$audit/${run}.log" 2>&1; then
    printf '%s completed %s\n' "$(date -Is)" "$run"
  else
    printf '%s FAILED %s; retain artifacts and inspect the error\n' "$(date -Is)" "$run"
    exit 1
  fi
done
