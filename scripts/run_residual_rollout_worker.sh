#!/usr/bin/env bash
set -euo pipefail
cd /home/ryab/ScAI-Lab
export CUDA_VISIBLE_DEVICES=6
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR=/tmp/scai-matplotlib NUMBA_CACHE_DIR=/tmp/scai-numba
export LD_LIBRARY_PATH=/home/ryab/miniconda3/envs/vlm_env/lib
.venv-test/bin/python scripts/run_residual_rollout_experiment.py --device cuda > docs/audit_2026-09-14/residual_rollout_experiment/run.log 2>&1
