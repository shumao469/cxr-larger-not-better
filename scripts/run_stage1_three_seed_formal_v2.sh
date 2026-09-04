#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -f "${HOME}/miniconda/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda/etc/profile.d/conda.sh"
  conda activate cxr_torch_stable
fi

export TMPDIR=/tmp
export TEMP=/tmp
export TMP=/tmp
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

MANIFEST="${MANIFEST:-data/processed/model_manifest_formal_v2.csv}"

python src/14_plan_or_run_three_seed_recovery.py \
  --project-root "${PROJECT_ROOT}" \
  --manifest "${MANIFEST}" \
  --mode formal \
  --epochs "${EPOCHS:-5}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --prediction-batch-size "${PREDICTION_BATCH_SIZE:-128}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --val-limit "${VAL_LIMIT:-3000}" \
  --val-sampling-seed "${VAL_SAMPLING_SEED:-2026}" \
  --execute \
  "$@"
