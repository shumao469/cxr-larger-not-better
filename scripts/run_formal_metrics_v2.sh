#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$(cd "$script_dir/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"
pred_root="$project_root/results_formal_v2/predictions"
metric_root="$project_root/results_formal_v2/metrics"
log_root="${LOG_ROOT:-$project_root/logs/formal_pipeline}"

mkdir -p "$log_root"
cd "$project_root"
exec 9>"$log_root/formal_metrics_queue.lock"
if ! flock -n 9; then
    echo 'Formal metrics queue already running; duplicate launch refused.'
    exit 1
fi

if [[ -d "$metric_root" ]] && find "$metric_root" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite non-empty metrics directory: $metric_root" >&2
    exit 2
elif [[ -e "$metric_root" && ! -d "$metric_root" ]]; then
    echo "Metrics path exists but is not a directory: $metric_root" >&2
    exit 2
fi

mapfile -t model_ids < <(find "$pred_root" -mindepth 1 -maxdepth 1 -type d ! -name '.*' -printf '%f\n' | sort)
if [[ ${#model_ids[@]} -ne 30 ]]; then
    echo "Expected 30 published prediction directories, found ${#model_ids[@]}" >&2
    exit 3
fi

for model_id in "${model_ids[@]}"; do
    pred_dir="$pred_root/$model_id"
    [[ -f "$pred_dir/prediction_completed.json" ]] || { echo "Missing completion marker: $model_id" >&2; exit 4; }
    [[ -f "$pred_dir/predictions.csv" ]] || { echo "Missing predictions.csv: $model_id" >&2; exit 5; }
done

mkdir -p "$metric_root"
for model_id in "${model_ids[@]}"; do
    echo "Evaluating $model_id"
    "$python_bin" src/09_evaluate_predictions_v2.py \
        --pred "$pred_root/$model_id/predictions.csv" \
        --out "$metric_root/${model_id}_metrics.csv"
done

"$python_bin" - "$metric_root" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1]).resolve()
files = sorted(root.glob('*_metrics.csv'))
if len(files) != 30:
    raise RuntimeError(f'Expected 30 metric files, found {len(files)}')

frames = []
for path in files:
    frame = pd.read_csv(path)
    if len(frame) != 15:
        raise RuntimeError(f'{path.name}: expected 15 rows, found {len(frame)}')
    frame['metric_file'] = path.name
    frames.append(frame)

all_metrics = pd.concat(frames, ignore_index=True)
if len(all_metrics) != 450:
    raise RuntimeError(f'Expected 450 consolidated rows, found {len(all_metrics)}')
mean_primary = all_metrics[all_metrics['label'] == 'mean_primary_labels'].copy()
if len(mean_primary) != 90:
    raise RuntimeError(f'Expected 90 mean-primary rows, found {len(mean_primary)}')

outputs = {
    'all_formal_metrics.csv': all_metrics,
    'formal_mean_primary_metrics.csv': mean_primary,
}
hashes = {}
for name, frame in outputs.items():
    target = root / name
    temp = target.with_suffix(target.suffix + '.tmp')
    frame.to_csv(temp, index=False)
    os.replace(temp, target)
    hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()

report = {
    'schema_version': 1,
    'status': 'PASS',
    'completed_at': datetime.now(timezone.utc).astimezone().isoformat(),
    'model_metric_files': 30,
    'all_metric_rows': 450,
    'mean_primary_rows': 90,
    'artifacts_sha256': hashes,
}
target = root / 'metrics_completed.json'
temp = target.with_suffix('.json.tmp')
temp.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
os.replace(temp, target)
print(json.dumps(report, indent=2))
PY

echo 'Formal metrics complete. Figure generation remains a separate audited stage.'
