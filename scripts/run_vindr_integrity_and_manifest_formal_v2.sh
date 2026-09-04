#!/usr/bin/env bash
# New CPU-only audit -> staged final manifest. Never starts a training/GPU job.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo 'Usage: bash scripts/run_vindr_integrity_and_manifest_formal_v2.sh NEW_ABSOLUTE_LOG_DIRECTORY' >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$(cd "$script_dir/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"
: "${VINDR_DICOM_ROOT:?Set VINDR_DICOM_ROOT to the verified VinDr-CXR v1.0.0 directory}"
: "${VINDR_PNG_ROOT:?Set VINDR_PNG_ROOT to the completed formal PNG directory}"
run_dir="$1"
cd "$project_root"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Independent from the training GPU lock; prevents duplicate CPU audit chains.
exec 9>>"$project_root/logs/vindr_integrity_and_manifest.lock"
if ! flock -n 9; then
    echo 'A VinDr integrity/manifest chain is already running.' >&2
    exit 1
fi

"$python_bin" - "$project_root" "$run_dir" <<'PY'
import importlib.util
import os
from pathlib import Path
import sys
project = Path(sys.argv[1]).resolve()
raw = Path(sys.argv[2])
if not raw.is_absolute() or (project / "logs").resolve() not in raw.resolve().parents:
    raise SystemExit("Run directory must be an absolute new child of the project logs directory")
if os.path.lexists(raw):
    raise SystemExit("Run directory already exists; prior audit and FAIL logs will not be overwritten")
spec = importlib.util.spec_from_file_location("finalize_manifest", project / "src/17_finalize_formal_manifest.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.reject_existing_outputs(project / "data/processed")
raw.mkdir(parents=True, exist_ok=False)
PY

"$python_bin" -u src/16_audit_vindr_png_integrity.py \
    --dicom-root "$VINDR_DICOM_ROOT" \
    --png-root "$VINDR_PNG_ROOT" \
    --audit-out "$run_dir/png_integrity.json" \
    > "$run_dir/png_audit.stdout.log" 2> "$run_dir/png_audit.stderr.log"

CXR_PROJECT_ROOT="$project_root" \
VINDR_DICOM_ROOT="$VINDR_DICOM_ROOT" \
VINDR_PNG_ROOT="$VINDR_PNG_ROOT" \
NIH_ROOT="${NIH_ROOT:?Set NIH_ROOT to the NIH ChestX-ray14 directory}" \
CHEXPERT_ROOT="${CHEXPERT_ROOT:?Set CHEXPERT_ROOT to the CheXpert v1.0 directory}" \
"$python_bin" -u src/17_finalize_formal_manifest.py \
    --png-audit "$run_dir/png_integrity.json" \
    --work-dir "$run_dir/manifest" \
    > "$run_dir/finalize.stdout.log" 2> "$run_dir/finalize.stderr.log"
