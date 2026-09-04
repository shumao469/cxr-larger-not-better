#!/usr/bin/env bash
# Wait for all formal training and final manifest gates; never interrupt training.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$(cd "$script_dir/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"
state_root="${STATE_ROOT:-$project_root/logs/formal_pipeline}"
mkdir -p "$state_root"
cd "$project_root"
export TMPDIR="${TMPDIR:-/tmp}" TEMP="${TEMP:-/tmp}" TMP="${TMP:-/tmp}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec 9>"$state_root/grouped_prediction_queue.lock"
if ! flock -n 9; then
    echo 'Grouped prediction queue already running; duplicate launch refused.'
    exit 1
fi
echo 'Waiting for final manifest and all 30 training completions; GPU remains with training.'
cache_args=()
if [[ -n "${CACHE_PATH:-}" ]]; then
    cache_args=(--cache-path "$CACHE_PATH")
fi
"$python_bin" -u src/19_predict_grouped_cxr_v2.py \
    --project-root "$project_root" --execute --wait-ready --poll-seconds 60 \
    --group-size "${GROUP_SIZE:-5}" "${cache_args[@]}"
echo 'Grouped formal predictions complete. Continue audited metrics and figures through the existing workflow monitor.'
