#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$(cd "$script_dir/.." && pwd)}"
python_bin="${PYTHON_BIN:-python}"
: "${VINDR_DICOM_ROOT:?Set VINDR_DICOM_ROOT to the verified VinDr-CXR v1.0.0 directory}"
: "${VINDR_PNG_ROOT:?Set VINDR_PNG_ROOT to a new or resumable PNG output directory}"
dicom_root="$VINDR_DICOM_ROOT"
png_root="$VINDR_PNG_ROOT"
converter="$project_root/src/04_convert_vindr_dicom_to_png_v2.py"

"$python_bin" - "$dicom_root" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "full_sha256_merge_summary.json").read_text(encoding="utf-8"))
flag = json.loads((root / "download_verified.flag").read_text(encoding="utf-8"))
if summary.get("status") != "PASS" or summary.get("verified") != 18006:
    raise SystemExit(f"VinDr checksum summary gate failed: {summary}")
if flag.get("status") != "PASS" or flag.get("verified") != 18006:
    raise SystemExit(f"VinDr verified flag gate failed: {flag}")
if flag.get("manifest_sha256") != summary.get("manifest_sha256"):
    raise SystemExit("VinDr flag/summary manifest hash mismatch")

for split, expected in (("test", 3000), ("train", 15000)):
    files = [
        p for p in (root / split).iterdir()
        if p.is_file() and p.suffix.lower() in {".dicom", ".dcm"}
    ]
    if len(files) != expected:
        raise SystemExit(f"VinDr {split} DICOM count {len(files)} != {expected}")
print("VinDr preflight PASS: checksum=18006/18006, test=3000, train=15000", flush=True)
PY

mkdir -p "$png_root/test" "$png_root/train"

"$python_bin" -u "$converter" \
  --dicom-root "$dicom_root/test" \
  --png-root "$png_root/test"

"$python_bin" -u "$converter" \
  --dicom-root "$dicom_root/train" \
  --png-root "$png_root/train"

"$python_bin" - "$png_root" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for split, expected in (("test", 3000), ("train", 15000)):
    files = [p for p in (root / split).iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    if len(files) != expected:
        raise SystemExit(f"VinDr {split} PNG count {len(files)} != {expected}")
tmp_files = list(root.rglob("*.tmp"))
failure_files = list(root.rglob("conversion_failures.tsv"))
if tmp_files or failure_files:
    raise SystemExit(f"Incomplete conversion artifacts: tmp={len(tmp_files)}, failure_files={len(failure_files)}")
print("VinDr conversion PASS: test=3000, train=15000", flush=True)
PY
