"""Small isolated checks for final-manifest gates; never invoke the real builder."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "src/17_finalize_formal_manifest.py"
SPEC = importlib.util.spec_from_file_location("finalize_formal_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def passing_png_report():
    summary = {"status": "PASS", "manifest_entries": 18006, "verified": 18006,
               "manifest_sha256": MODULE.MANIFEST_SHA256, "missing": 0, "source_mismatch": 0, "errors": 0}
    return {
        "schema_version": 1, "status": "PASS", "failed_check_count": 0, "failure_examples": [],
        "expected_counts": dict(MODULE.IMAGE_COUNTS), "expected_manifest_entries": 18006,
        "dicom_root": str(MODULE.DICOM_ROOT), "png_root": str(MODULE.PNG_ROOT),
        "image_counts": {"dicom": dict(MODULE.IMAGE_COUNTS), "png": dict(MODULE.IMAGE_COUNTS)},
        "checks": {key: {"passed": True} for key in MODULE.REQUIRED_PNG_CHECKS},
        "png_validation": {"attempted": 18000, "crc_verify_passed": 18000, "all_chunk_crcs_passed": 18000,
                           "full_decode_passed": 18000, "passed": 18000, "failed": 0, "chunks_crc_checked": 54000},
        "source_evidence": {"summary": summary, "verified_flag": dict(summary),
                            "row_counts": {"manifest": 18006, "audit_csv": 18006, "pass_and_matching_hash": 18006,
                                           "current_path_and_size_match": 18006, "invalid_rows": 0},
                            "files": {"SHA256SUMS.txt": {"sha256": MODULE.MANIFEST_SHA256}}},
    }


def passing_builder_audit():
    return {"output_rows": 274439, "expected_output_rows": 274439,
            "counts": {f"{d}/{s}": n for (d, s), n in MODULE.SPLIT_COUNTS.items()},
            "count_mismatches": {}, "unexpected_dataset_splits": {},
            "duplicate_dataset_image_ids": 0, "patient_split_overlap_total": 0, "all_paths_resolved": True,
            "source_manifest": str(MODULE.PROJECT_ROOT / "data/processed/harmonized_manifest_all_splits.csv"),
            "selected_datasets": ["nih", "chexpert", "vindr"]}


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cxr-finalize-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_fixed_full_png_report_passes(self):
        MODULE.validate_png_report(passing_png_report())

    def test_failed_report_never_promoted(self):
        report = passing_png_report()
        report["status"] = "FAIL"
        with self.assertRaises(ValueError):
            MODULE.validate_png_report(report)

    def test_missing_failed_or_unexpected_failed_check_rejected(self):
        original = passing_png_report()
        variants = []
        report = copy.deepcopy(original)
        del report["checks"]["all_png_crc_and_decode_pass"]
        variants.append(report)
        report = copy.deepcopy(original)
        report["checks"]["dicom_download_sidecars_unchanged"]["passed"] = False
        variants.append(report)
        report = copy.deepcopy(original)
        report["checks"]["new_extra_check"] = {"passed": False}
        variants.append(report)
        for report in variants:
            with self.subTest(report=report), self.assertRaises(ValueError):
                MODULE.validate_png_report(report)

    def test_every_png_stage_must_reach_18000(self):
        for key in ("attempted", "crc_verify_passed", "all_chunk_crcs_passed", "full_decode_passed", "passed"):
            report = passing_png_report()
            report["png_validation"][key] = 17999
            with self.subTest(key=key), self.assertRaises(ValueError):
                MODULE.validate_png_report(report)
        report = passing_png_report()
        report["png_validation"]["failed"] = 1
        with self.assertRaises(ValueError):
            MODULE.validate_png_report(report)

    def test_counts_roots_hash_and_18006_source_gate(self):
        variants = []
        report = passing_png_report()
        report["expected_counts"] = {"test": 3, "train": 15}
        variants.append(report)
        report = passing_png_report()
        report["png_root"] = str(self.root)
        variants.append(report)
        report = passing_png_report()
        report["source_evidence"]["row_counts"]["current_path_and_size_match"] = 18005
        variants.append(report)
        report = passing_png_report()
        report["source_evidence"]["verified_flag"]["manifest_sha256"] = "0" * 64
        variants.append(report)
        report = passing_png_report()
        report["source_evidence"]["summary"]["status"] = "FAIL"
        variants.append(report)
        for report in variants:
            with self.subTest(report=report), self.assertRaises(ValueError):
                MODULE.validate_png_report(report)

    def test_builder_audit_has_no_status_but_passes(self):
        report = passing_builder_audit()
        self.assertNotIn("status", report)
        MODULE.validate_builder_audit(report)

    def test_builder_count_overlap_missing_or_wrong_source_rejected(self):
        for key, value in (("output_rows", 274438), ("patient_split_overlap_total", 1),
                           ("count_mismatches", {"nih/train": "wrong"}),
                           ("all_paths_resolved", False), ("source_manifest", "legacy.csv")):
            report = passing_builder_audit()
            report[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                MODULE.validate_builder_audit(report)

    def frames(self):
        frames = {
            "nih": pd.DataFrame([{"dataset": "nih", "image_id": "a", "patient_id": "1", "split": "train", "label": "0", "path": "/a"},
                                 {"dataset": "nih", "image_id": "b", "patient_id": "2", "split": "val", "label": "1", "path": "/b"}]),
            "chexpert": pd.DataFrame([{"dataset": "chexpert", "image_id": "c", "patient_id": "3", "split": "train", "label": "", "path": "/c"}]),
        }
        return frames, pd.concat(frames.values(), ignore_index=True)

    def test_exact_rows_all_columns_and_order_pass(self):
        frames, final = self.frames()
        report = MODULE.compare_training_rows(final, frames)
        self.assertTrue(report["nih"]["all_columns_and_row_order_identical"])

    def test_label_path_patient_split_or_row_order_change_fails(self):
        frames, original = self.frames()
        for column in ("image_id", "patient_id", "split", "label", "path"):
            final = original.copy()
            final.loc[0, column] = "changed"
            with self.subTest(column=column), self.assertRaises(AssertionError):
                MODULE.compare_training_rows(final, frames)
        with self.assertRaises(AssertionError):
            MODULE.compare_training_rows(original.iloc[[1, 0, 2]], frames)

    def test_every_existing_formal_target_blocks_without_modification(self):
        for name in MODULE.FINAL_NAMES:
            path = self.root / name
            path.write_bytes(b"preserve existing PASS or FAIL")
            with self.subTest(name=name), self.assertRaises(FileExistsError):
                MODULE.reject_existing_outputs(self.root)
            self.assertEqual(path.read_bytes(), b"preserve existing PASS or FAIL")
            path.unlink()

    def test_dangling_target_symlink_also_blocks(self):
        target = self.root / MODULE.FINAL_NAMES[0]
        try:
            target.symlink_to(self.root / "missing.csv")
        except OSError:
            self.skipTest("Symlink creation unavailable")
        with self.assertRaises(FileExistsError):
            MODULE.reject_existing_outputs(self.root)

    def test_publish_new_refuses_overwrite_including_race(self):
        source = self.root / "staged.json"
        source.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        target = self.root / "published.json"
        MODULE.publish_new(source, target)
        before = target.read_bytes()
        other = self.root / "other.json"
        other.write_bytes(b"replacement")
        with self.assertRaises(FileExistsError):
            MODULE.publish_new(other, target)
        self.assertEqual(target.read_bytes(), before)
        self.assertTrue(source.exists())

    def test_lock_rejects_concurrent_finalizer(self):
        if os.name == "nt":
            self.skipTest("Production lock uses WSL fcntl")
        path = self.root / "finalize.lock"
        with MODULE.finalization_lock(path):
            with self.assertRaises(RuntimeError):
                with MODULE.finalization_lock(path):
                    self.fail("duplicate lock acquired")


if __name__ == "__main__":
    unittest.main(verbosity=2)
