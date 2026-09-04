#!/usr/bin/env python3
"""CPU-only, non-overwriting finalization after the strict VinDr PNG audit.

Production inputs and the frozen cohort are intentionally not CLI-configurable.
Host paths are bound once from environment variables before the audit starts.
Only a fresh directory beneath the project's logs directory may be selected.
The existing manifest builder runs in a new staging directory. No training,
subset generation, prediction, GPU import, or image modification takes place.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd


PROJECT_ROOT = Path(os.environ.get("CXR_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DICOM_ROOT = Path(os.environ.get("VINDR_DICOM_ROOT", PROJECT_ROOT / "data/raw/vindr-cxr/1.0.0")).resolve()
PNG_ROOT = Path(os.environ.get("VINDR_PNG_ROOT", PROJECT_ROOT / "data/processed/vindr_png_formal_v2")).resolve()
NIH_ROOT = Path(os.environ.get("NIH_ROOT", PROJECT_ROOT / "data/raw/nih")).resolve()
CHEXPERT_ROOT = Path(os.environ.get("CHEXPERT_ROOT", PROJECT_ROOT / "data/raw/CheXpert-v1.0")).resolve()
MANIFEST_SHA256 = "175f64e334b3c8a0773d1580263ac64bd4ba72511cb7391f3ab503025a8ed5d4"
IMAGE_COUNTS = {"test": 3000, "train": 15000}
SPLIT_COUNTS = {
    ("nih", "train"): 79481,
    ("nih", "val"): 11342,
    ("nih", "internal_test"): 21297,
    ("chexpert", "train"): 103286,
    ("chexpert", "val"): 15047,
    ("chexpert", "internal_test"): 28986,
    ("vindr", "external_test"): 15000,
}
FINAL_NAMES = (
    "model_manifest_formal_v2.csv",
    "model_manifest_formal_v2_audit.json",
    "model_manifest_formal_v2_audit_overlap.csv",
    "model_manifest_formal_v2_missing.csv",
    "model_manifest_formal_v2_finalization.json",
)
EVIDENCE_NAMES = (
    "SHA256SUMS.txt", "full_sha256_merge_summary.json",
    "download_verified.flag", "full_sha256_merge_audit.csv",
)
REQUIRED_PNG_CHECKS = (
    "official_manifest_sha256", "summary_pass_and_counts", "flag_pass_and_count",
    "manifest_hash_matches_both_markers", "checksum_manifest_entry_count",
    "source_audit_rows_all_pass", "source_audit_paths_exactly_match_manifest",
    "test_exact_image_counts", "train_exact_image_counts",
    "test_png_stems_match_dicom", "train_png_stems_match_dicom",
    "test_dicom_paths_match_verified_source", "train_dicom_paths_match_verified_source",
    "png_global_stems_unique", "dicom_global_stems_unique",
    "png_no_linked_entries", "dicom_no_linked_entries",
    "png_no_unexpected_image_paths", "dicom_no_unexpected_image_paths",
    "png_no_incomplete_or_failure_artifacts", "dicom_no_incomplete_or_failure_artifacts",
    "all_png_crc_and_decode_pass", "source_evidence_unchanged",
    "source_metadata_unchanged", "decoded_png_metadata_unchanged", "image_inventory_unchanged",
    "dicom_download_sidecars_unchanged",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as stream:
        result = json.load(stream)
    require(isinstance(result, dict), f"Expected JSON object: {path}")
    return result


def reject_existing_outputs(processed: Path) -> None:
    conflicts = [str(processed / name) for name in FINAL_NAMES
                 if os.path.lexists(processed / name)]
    if conflicts:
        raise FileExistsError("Refusing to overwrite existing formal output(s): " + ", ".join(conflicts))


def validate_png_report(report: dict) -> None:
    """Pure report gate; fixed production limits also apply to imported calls."""
    require(report.get("schema_version") == 1 and report.get("status") == "PASS",
            "PNG audit must be schema 1 and PASS")
    require(report.get("failed_check_count") == 0 and report.get("failure_examples") == [],
            "PNG audit must contain zero failures")
    require(report.get("expected_counts") == IMAGE_COUNTS
            and report.get("expected_manifest_entries") == 18006,
            "PNG audit must use the fixed 18000-image / 18006-entry release")
    require(report.get("dicom_root") == str(DICOM_ROOT)
            and report.get("png_root") == str(PNG_ROOT), "PNG audit roots differ from production roots")
    require(report.get("image_counts") == {"dicom": IMAGE_COUNTS, "png": IMAGE_COUNTS},
            "Current DICOM and PNG split counts must both be 3000/15000")
    checks = report.get("checks", {})
    require(isinstance(checks, dict) and set(REQUIRED_PNG_CHECKS).issubset(checks),
            "PNG audit is missing required checks")
    require(all(isinstance(value, dict) and value.get("passed") is True for value in checks.values()),
            "Every PNG audit check must pass")
    validation = report.get("png_validation", {})
    require(all(validation.get(key) == 18000 for key in
                ("attempted", "crc_verify_passed", "all_chunk_crcs_passed", "full_decode_passed", "passed"))
            and validation.get("failed") == 0
            and validation.get("chunks_crc_checked", 0) >= 54000,
            "All 18000 PNGs must pass every chunk CRC, verify, and full decode")
    evidence = report.get("source_evidence", {})
    summary, flag = evidence.get("summary", {}), evidence.get("verified_flag", {})
    require(summary.get("status") == flag.get("status") == "PASS"
            and summary.get("manifest_entries") == 18006
            and summary.get("verified") == flag.get("verified") == 18006
            and summary.get("manifest_sha256") == flag.get("manifest_sha256") == MANIFEST_SHA256
            and all(summary.get(key) == 0 for key in ("missing", "source_mismatch", "errors")),
            "Official DICOM evidence must be PASS 18006/18006 with the pinned manifest hash")
    rows = evidence.get("row_counts", {})
    require(all(rows.get(key) == 18006 for key in
                ("manifest", "audit_csv", "pass_and_matching_hash", "current_path_and_size_match"))
            and rows.get("invalid_rows") == 0, "Official source audit row checks are incomplete")
    require(evidence.get("files", {}).get("SHA256SUMS.txt", {}).get("sha256") == MANIFEST_SHA256,
            "Official manifest digest missing from evidence snapshot")


def verify_current_report_evidence(report: dict, project: Path) -> None:
    """Detect stale evidence or changed tools without rehashing the large DICOMs."""
    for name in EVIDENCE_NAMES:
        require(sha256_file(DICOM_ROOT / name) == report["source_evidence"]["files"][name]["sha256"],
                f"Source evidence changed after PNG audit: {name}")
    for key, path in (
        ("audit_script", project / "src/16_audit_vindr_png_integrity.py"),
        ("converter", project / "src/04_convert_vindr_dicom_to_png_v2.py"),
        ("conversion_runner", project / "scripts/run_vindr_conversion_formal_v2.sh"),
    ):
        require(sha256_file(path) == report.get("provenance", {}).get(key, {}).get("sha256"),
                f"Tool provenance differs from the PNG audit: {key}")


def validate_builder_audit(audit: dict) -> None:
    require(audit.get("output_rows") == audit.get("expected_output_rows") == 274439,
            "Final manifest must contain exactly 274439 rows")
    require(audit.get("counts") == {f"{dataset}/{split}": count
                                    for (dataset, split), count in SPLIT_COUNTS.items()},
            "Final source/split counts differ from the frozen cohort")
    require(audit.get("count_mismatches") == {} and audit.get("unexpected_dataset_splits") == {}
            and audit.get("duplicate_dataset_image_ids") == 0
            and audit.get("patient_split_overlap_total") == 0
            and audit.get("all_paths_resolved") is True,
            "The staged manifest audit did not pass all required checks")
    require(audit.get("source_manifest") == str(PROJECT_ROOT / "data/processed/harmonized_manifest_all_splits.csv")
            and audit.get("selected_datasets") == ["nih", "chexpert", "vindr"],
            "Staged manifest was not built from the fixed frozen source")


def compare_training_rows(final: pd.DataFrame, training_frames: dict[str, pd.DataFrame]) -> dict:
    """All columns, values, and within-source row order must be exactly unchanged."""
    result = {}
    for source in ("nih", "chexpert"):
        expected = training_frames[source].reset_index(drop=True)
        actual = final.loc[final["dataset"] == source].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_dtype=True,
                                      check_like=False, obj=f"Frozen {source} training manifest")
        result[source] = {"rows": len(actual), "all_columns_and_row_order_identical": True}
    return result


def publish_new(source: Path, destination: Path) -> None:
    """Same-filesystem atomic publication with no overwrite, including race cases."""
    os.link(source, destination)


@contextlib.contextmanager
def finalization_lock(path: Path):
    # Production CLI is WSL/Linux. The lock is separate from the GPU lock.
    import fcntl
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another formal-manifest finalizer is running") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def finalize(png_audit: Path, work_dir: Path) -> Path:
    project = PROJECT_ROOT.resolve()
    processed = project / "data/processed"
    log_root = (project / "logs").resolve()
    work_dir = work_dir.resolve()
    require(log_root in work_dir.parents, "Work directory must be a new child of the project logs directory")
    require(not os.path.lexists(work_dir), "Work directory must not already exist; prior FAIL evidence is preserved")
    require(png_audit.is_file() and not png_audit.is_symlink(), "A regular completed PNG audit JSON is required")
    with finalization_lock(log_root / "formal_manifest_finalization.lock"):
        reject_existing_outputs(processed)
        report = read_json(png_audit)
        validate_png_report(report)
        verify_current_report_evidence(report, project)
        inputs = [png_audit.resolve(), processed / "harmonized_manifest_all_splits.csv",
                  project / "src/13_rebuild_recovered_manifest.py"]
        training_paths = {source: processed / f"model_manifest_{source}_training_v2.csv"
                          for source in ("nih", "chexpert")}
        inputs.extend(training_paths.values())
        initial_hashes = {str(path): sha256_file(path) for path in inputs}
        work_dir.mkdir(parents=True, exist_ok=False)
        stage = work_dir / "staging"
        stage.mkdir()
        staged_csv, staged_audit, staged_overlap, staged_missing, staged_completion = [stage / name for name in FINAL_NAMES]
        command = [sys.executable, "-u", str(project / "src/13_rebuild_recovered_manifest.py"),
                   "--datasets", "nih", "chexpert", "vindr",
                   "--source-manifest", str(processed / "harmonized_manifest_all_splits.csv"),
                   "--nih-root", str(NIH_ROOT), "--chexpert-root", str(CHEXPERT_ROOT),
                   "--vindr-png-root", str(PNG_ROOT / "train"),
                   "--out", str(staged_csv), "--audit-out", str(staged_audit),
                   "--missing-out", str(staged_missing)]
        with (work_dir / "builder.stdout.log").open("x", encoding="utf-8") as stdout, \
                (work_dir / "builder.stderr.log").open("x", encoding="utf-8") as stderr:
            subprocess.run(command, cwd=project, stdout=stdout, stderr=stderr, check=True)
        require(not staged_missing.exists(), "Builder left a missing-image report; formal publication is blocked")
        validate_builder_audit(read_json(staged_audit))
        final = pd.read_csv(staged_csv, dtype=str, keep_default_na=False)
        require(len(final) == 274439 and not final.duplicated(["dataset", "image_id"]).any(),
                "Staged CSV row count or dataset/image uniqueness failed")
        actual_counts = final.groupby(["dataset", "split"]).size().to_dict()
        require(actual_counts == SPLIT_COUNTS, "Staged CSV source/split counts failed")
        comparison = compare_training_rows(final, {source: pd.read_csv(path, dtype=str, keep_default_na=False)
                                                   for source, path in training_paths.items()})
        final_hashes = {str(path): sha256_file(path) for path in inputs}
        require(initial_hashes == final_hashes, "Frozen source, training manifests, builder, or PNG audit changed during finalization")
        verify_current_report_evidence(report, project)
        completion = {"schema_version": 1, "status": "PASS", "completed_at": datetime.now(timezone.utc).isoformat(),
                      "manifest_rows": 274439, "png_audit": str(png_audit.resolve()),
                      "input_sha256": initial_hashes, "source_comparison": comparison,
                      "cpu_only": True, "training_queue_modified": False,
                      "publication_contract": "Require all listed files and this final completion report; never consume partial publication.",
                      "artifacts": {name: {"path": str(processed / name), "sha256": sha256_file(stage / name)}
                                    for name in FINAL_NAMES[:3]}}
        with staged_completion.open("x", encoding="utf-8") as stream:
            json.dump(completion, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        reject_existing_outputs(processed)
        # Every operation is create-only. If publication is interrupted, retain
        # all staging/partial artifacts; no cleanup can erase pre-existing data.
        for source in (staged_audit, staged_overlap, staged_csv, staged_completion):
            publish_new(source, processed / source.name)
        return processed / staged_completion.name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png-audit", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        path = finalize(args.png_audit, args.work_dir)
        print(json.dumps({"status": "PASS", "completion_report": str(path)}, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(f"Formal manifest finalization FAILED; existing outputs and staging were preserved: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
