#!/usr/bin/env bash
# Fixed-protocol work queue. One worker per otherwise available GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${1:?GPU index required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
export NUMBA_CACHE_DIR=/tmp/scai-numba
export LD_LIBRARY_PATH=/home/ryab/miniconda3/envs/vlm_env/lib
queue=docs/audit_2026-09-14/run_configs/queue.txt
while true; do
  # Serialize queue claims; each run also locks its own output directory.
  exec 9>"${queue}.lock"
  flock 9
  job=$(head -n 1 "$queue")
  if [[ -z "$job" ]]; then flock -u 9; break; fi
  tail -n +2 "$queue" > "${queue}.next"
  mv "${queue}.next" "$queue"
  flock -u 9
  read -r arm seed <<< "$job"
  run="validated_${arm}_seed${seed}"
  printf '%s started %s GPU=%s\n' "$(date -Is)" "$run" "$CUDA_VISIBLE_DEVICES"
  if .venv-test/bin/python -u vlm/run/run_mouse_vlm_loso.py \
    --yaml "docs/audit_2026-09-14/run_configs/${arm}_seed${seed}.yml" \
    --output-dir "/data1/Processed_NIfTI_Test/embeddings/vlm/runs/${run}" \
    >> "docs/audit_2026-09-14/${run}.log" 2>&1; then
    printf '%s completed %s\n' "$(date -Is)" "$run"
  else
    printf '%s FAILED %s; inspect log and resume this run explicitly\n' "$(date -Is)" "$run"
    exit 1
  fi
done
