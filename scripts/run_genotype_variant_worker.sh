#!/usr/bin/env bash
set -euo pipefail
cd /home/ryab/ScAI-Lab
export CUDA_VISIBLE_DEVICES="${1:?GPU required}"
variant="${2:?Variant required}"
case "$variant" in direct_visual|genotype_only) ;; *) exit 2 ;; esac
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
export NUMBA_CACHE_DIR=/tmp/scai-numba MPLCONFIGDIR=/tmp/scai-matplotlib
export LD_LIBRARY_PATH=/home/ryab/miniconda3/envs/vlm_env/lib
audit=docs/audit_2026-09-14/genotype_vlm_variants
for seed in 0 1 2; do
  .venv-test/bin/python -u scripts/run_genotype_variants.py \
    --variant "$variant" --yaml "$audit/configs/${variant}_seed${seed}.yml" \
    --split docs/audit_2026-09-14/stratified/PROTOCOL.json \
    --all-data /data1/Processed_NIfTI_Test/embeddings/vlm/validated_20260914/mouse_all_vqa_traj.json \
    --original-run "/data1/Processed_NIfTI_Test/embeddings/vlm/runs/stratified_long_seed${seed}" \
    --parent-run "/data1/Processed_NIfTI_Test/embeddings/vlm/runs/stratified_residual_seed${seed}" \
    --output-dir "/data1/Processed_NIfTI_Test/embeddings/vlm/runs/stratified_${variant}_seed${seed}" \
    >> "$audit/${variant}_seed${seed}.log" 2>&1
done
# Whichever queue finishes second verifies and aggregates the whole comparison.
printf 'complete\n' > "$audit/${variant}.done"
exec 9>"$audit/aggregate.lock"
flock 9
if [[ -f "$audit/direct_visual.done" && -f "$audit/genotype_only.done" ]]; then
  .venv-test/bin/python -u scripts/evaluate_genotype_variants.py > "$audit/verification.log" 2>&1
fi
