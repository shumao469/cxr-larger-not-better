"""Small, isolated fixtures: never access or modify production image trees."""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "src/16_audit_vindr_png_integrity.py"
SPEC = importlib.util.spec_from_file_location("audit_vindr_png_integrity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vindr-png-audit-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.raw = self.base / "official"
        self.png = self.base / "converted"
        self.expected = {"test": 1, "train": 2}
        for root in (self.raw, self.png):
            for split in self.expected:
                (root / split).mkdir(parents=True)
        self.source_files = []
        for split, stems in (("test", ["test_image"]), ("train", ["train_a", "train_b"])):
            for stem in stems:
                source = self.raw / split / f"{stem}.dicom"
                source.write_bytes(f"Small fake DICOM {split}/{stem}".encode())
                self.source_files.append(source)
                Image.new("L", (8, 9), color=80).save(self.png / split / f"{stem}.png")
        for index in range(6):
            source = self.raw / f"metadata_{index}.txt"
            source.write_text(f"metadata {index}", encoding="utf-8")
            self.source_files.append(source)
        self.rows = []
        manifest_lines = []
        for source in sorted(self.source_files):
            name = source.relative_to(self.raw).as_posix()
            digest = MODULE.sha256_file(source)
            manifest_lines.append(f"{digest} {name}\n")
            self.rows.append({"relative_path": name, "expected_sha256": digest,
                              "actual_sha256": digest, "bytes": source.stat().st_size,
                              "status": "PASS", "error": ""})
        (self.raw / "SHA256SUMS.txt").write_text("".join(manifest_lines), encoding="utf-8")
        self.manifest_hash = MODULE.sha256_file(self.raw / "SHA256SUMS.txt")
        self.summary = {"status": "PASS", "manifest_entries": 9, "verified": 9,
                        "missing": 0, "source_mismatch": 0, "errors": 0,
                        "manifest_sha256": self.manifest_hash,
                        "completed_at": "2026-09-03 02:08:06",
                        "physionet_root": str(self.raw),
                        "manifest": str(self.raw / "SHA256SUMS.txt"),
                        "audit_csv": str(self.raw / "full_sha256_merge_audit.csv")}
        self.flag = {"status": "PASS", "verified": 9,
                     "manifest_sha256": self.manifest_hash,
                     "completed_at": self.summary["completed_at"],
                     "summary": str(self.raw / "full_sha256_merge_summary.json"),
                     "audit_csv": self.summary["audit_csv"]}
        self.write_evidence()

    def write_evidence(self):
        (self.raw / "full_sha256_merge_summary.json").write_text(json.dumps(self.summary), encoding="utf-8")
        (self.raw / "download_verified.flag").write_text(json.dumps(self.flag), encoding="utf-8")
        with (self.raw / "full_sha256_merge_audit.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def run_audit(self):
        return MODULE.audit_dataset(self.raw, self.png, expected_counts=self.expected,
                                    expected_manifest_entries=9,
                                    expected_manifest_sha256=self.manifest_hash)

    def assert_failed(self, report, check=None):
        self.assertEqual(report["status"], "FAIL", report)
        self.assertGreater(report["failed_check_count"], 0)
        if check:
            self.assertFalse(report["checks"][check]["passed"], report)

    def test_pass_and_read_only_images(self):
        before = {p: (MODULE.sha256_file(p), p.stat().st_mtime_ns)
                  for root in (self.raw, self.png) for p in root.rglob("*") if p.is_file()}
        report = self.run_audit()
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["png_validation"]["crc_verify_passed"], 3)
        self.assertEqual(report["png_validation"]["all_chunk_crcs_passed"], 3)
        self.assertEqual(report["png_validation"]["full_decode_passed"], 3)
        self.assertEqual(report["source_evidence"]["row_counts"]["audit_csv"], 9)
        self.assertFalse(report["source_evidence"]["dicom_contents_rehashed"])
        after = {p: (MODULE.sha256_file(p), p.stat().st_mtime_ns)
                 for root in (self.raw, self.png) for p in root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_quarantine_is_not_indexed(self):
        quarantine = self.raw / ".replaced_partial"
        quarantine.mkdir()
        (quarantine / "train_a.dicom").write_bytes(b"old quarantined image")
        self.assertEqual(self.run_audit()["status"], "PASS")

    def test_exact_dicom_directory_indexes_are_recorded_and_allowed(self):
        expected = []
        for split in ("test", "train"):
            path = self.raw / split / "index.html"
            path.write_text(f"<html><title>Index of {split}/</title></html>", encoding="utf-8")
            expected.append({"relative_path": f"{split}/index.html", "path": str(path),
                             "bytes": path.stat().st_size, "sha256": MODULE.sha256_file(path),
                             "stat": list(MODULE.stamp(path))})
        report = self.run_audit()
        self.assertEqual(report["status"], "PASS", report)
        sidecars = report["dicom_download_sidecars"]
        self.assertEqual(sidecars["count"], 2)
        self.assertEqual(sidecars["files"], expected)
        self.assertEqual(sidecars["files_after"], expected)
        self.assertTrue(sidecars["unchanged"])
        self.assertEqual(report["image_counts"]["dicom"], self.expected)
        self.assertEqual(report["source_evidence"]["row_counts"]["audit_csv"], 9)

    def test_other_unknown_split_files_still_fail(self):
        for name in ("train/unknown.txt", "test/index.htm", "test/INDEX.html", "train/extra.jpg"):
            with self.subTest(name=name):
                path = self.raw / name
                path.write_bytes(b"unknown file")
                self.assert_failed(self.run_audit(), "dicom_no_unexpected_image_paths")
                path.unlink()

    def test_png_tree_indexes_and_other_unknown_files_fail(self):
        for name in ("index.html", "train/index.html", "test/index.html", "test/unknown.txt", "train/extra.jpg"):
            with self.subTest(name=name):
                path = self.png / name
                path.write_bytes(b"<html>Not allowed in PNG output</html>")
                self.assert_failed(self.run_audit(), "png_no_unexpected_image_paths")
                path.unlink()

    def test_download_index_content_change_is_detected_even_with_same_metadata(self):
        path = self.raw / "train/index.html"
        path.write_bytes(b"<html>first</html>")
        previous_stat = path.stat()
        original = MODULE.verify_all_png_chunk_crcs
        changed = False

        def change_index_once(png_path):
            nonlocal changed
            if not changed:
                path.write_bytes(b"<html>later</html>")
                os.utime(path, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
                changed = True
            return original(png_path)

        with patch.object(MODULE, "verify_all_png_chunk_crcs", side_effect=change_index_once):
            report = self.run_audit()
        self.assert_failed(report, "dicom_download_sidecars_unchanged")
        sidecars = report["dicom_download_sidecars"]
        self.assertEqual(sidecars["files"][0]["bytes"], sidecars["files_after"][0]["bytes"])
        self.assertEqual(sidecars["files"][0]["stat"], sidecars["files_after"][0]["stat"])
        self.assertNotEqual(sidecars["files"][0]["sha256"], sidecars["files_after"][0]["sha256"])

    def test_download_index_cannot_be_a_directory_or_symlink(self):
        path = self.raw / "train/index.html"
        path.mkdir()
        self.assert_failed(self.run_audit(), "dicom_no_unexpected_image_paths")
        path.rmdir()
        try:
            os.symlink(self.raw / "metadata_0.txt", path)
        except OSError:
            self.skipTest("Symbolic links unavailable")
        self.assert_failed(self.run_audit(), "dicom_no_linked_entries")

    def test_never_rehashes_dicom_contents(self):
        original = MODULE.sha256_file

        def no_dicom_hash(path):
            self.assertNotIn(path.suffix.lower(), MODULE.DICOM_EXTENSIONS)
            return original(path)

        with patch.object(MODULE, "sha256_file", side_effect=no_dicom_hash):
            self.assertEqual(self.run_audit()["status"], "PASS")

    def test_wrong_marker_hash(self):
        self.flag["manifest_sha256"] = "0" * 64
        self.write_evidence()
        self.assert_failed(self.run_audit(), "manifest_hash_matches_both_markers")

    def test_summary_not_pass_or_wrong_count(self):
        self.summary["verified"] = 8
        self.summary["status"] = "FAIL"
        self.write_evidence()
        self.assert_failed(self.run_audit(), "summary_pass_and_counts")

    def test_flag_wrong_count(self):
        self.flag["verified"] = 8
        self.write_evidence()
        self.assert_failed(self.run_audit(), "flag_pass_and_count")

    def test_wrong_official_hash_even_if_markers_agree(self):
        self.manifest_hash = "1" * 64
        self.assert_failed(self.run_audit(), "official_manifest_sha256")

    def test_evidence_points_at_other_root(self):
        self.flag["audit_csv"] = str(self.base / "wrong.csv")
        self.write_evidence()
        self.assert_failed(self.run_audit(), "flag_audit_csv_path")

    def test_csv_digest_mismatch(self):
        self.rows[0]["actual_sha256"] = "0" * 64
        self.write_evidence()
        self.assert_failed(self.run_audit(), "source_audit_rows_all_pass")

    def test_csv_duplicate_and_missing_paths(self):
        self.rows[-1] = dict(self.rows[0])
        self.write_evidence()
        self.assert_failed(self.run_audit(), "source_audit_paths_exactly_match_manifest")

    def test_csv_pass_with_error_is_rejected(self):
        self.rows[0]["error"] = "previous read failed"
        self.write_evidence()
        self.assert_failed(self.run_audit(), "source_audit_rows_all_pass")

    def test_source_size_change_detected_without_dicom_rehash(self):
        (self.raw / "train/train_a.dicom").write_bytes(b"changed size")
        self.assert_failed(self.run_audit(), "source_audit_rows_all_pass")

    def test_missing_png_and_extra_stem(self):
        (self.png / "train/train_a.png").rename(self.png / "train/unexpected.png")
        self.assert_failed(self.run_audit(), "train_png_stems_match_dicom")

    def test_nested_png_cannot_count_as_official(self):
        nested = self.png / "train/nested"
        nested.mkdir()
        Image.new("L", (8, 9)).save(nested / "other.png")
        self.assert_failed(self.run_audit(), "png_no_unexpected_image_paths")

    def test_duplicate_global_png_stem(self):
        Image.new("L", (8, 9)).save(self.png / "test/train_a.png")
        self.assert_failed(self.run_audit(), "png_global_stems_unique")

    def test_temporary_or_failure_artifact(self):
        for name in ("train/image.png.tmp", "train/conversion_failures.tsv", "test/.pending.tmp"):
            with self.subTest(name=name):
                path = self.png / name
                path.write_bytes(b"pending")
                self.assert_failed(self.run_audit(), "png_no_incomplete_or_failure_artifacts")
                path.unlink()

    def test_corrupt_crc(self):
        path = self.png / "test/test_image.png"
        data = bytearray(path.read_bytes())
        # Pillow verify()/load() omit IEND's CRC; our explicit chunk scan catches it.
        data[-1] ^= 0xFF
        path.write_bytes(data)
        self.assert_failed(self.run_audit(), "all_png_crc_and_decode_pass")

    def test_trailing_png_data(self):
        path = self.png / "test/test_image.png"
        path.write_bytes(path.read_bytes() + b"unexpected trailing bytes")
        self.assert_failed(self.run_audit(), "all_png_crc_and_decode_pass")

    def test_idat_corrupt_crc(self):
        path = self.png / "test/test_image.png"
        data = bytearray(path.read_bytes())
        location = data.index(b"IDAT")
        data[location + 4] ^= 0x01
        path.write_bytes(data)
        self.assert_failed(self.run_audit(), "all_png_crc_and_decode_pass")

    def test_truncated_png(self):
        path = self.png / "test/test_image.png"
        path.write_bytes(path.read_bytes()[:35])
        self.assert_failed(self.run_audit(), "all_png_crc_and_decode_pass")

    def test_rgb_png_and_disguised_format(self):
        for mode, file_format in (("RGB", "PNG"), ("L", "BMP")):
            with self.subTest(mode=mode, file_format=file_format):
                Image.new(mode, (8, 9)).save(self.png / "test/test_image.png", format=file_format)
                self.assert_failed(self.run_audit(), "all_png_crc_and_decode_pass")

    def test_multiframe_apng_is_rejected(self):
        path = self.png / "test/test_image.png"
        first = Image.new("L", (8, 9), color=30)
        second = Image.new("L", (8, 9), color=180)
        first.save(path, format="PNG", save_all=True, append_images=[second], duration=100, loop=0)
        with Image.open(path) as image:
            self.assertEqual(image.n_frames, 2)
        report = self.run_audit()
        self.assert_failed(report, "all_png_crc_and_decode_pass")
        self.assertEqual(report["png_validation"]["failed"], 1)
        self.assertTrue(any("single-frame PNG" in str(item)
                            for item in report["failure_examples"]))

    def test_reopened_load_failure(self):
        with patch.object(Image.Image, "load", side_effect=OSError("decode failure")):
            report = self.run_audit()
        self.assert_failed(report, "all_png_crc_and_decode_pass")
        self.assertEqual(report["png_validation"]["crc_verify_passed"], 3)
        self.assertEqual(report["png_validation"]["full_decode_passed"], 0)

    def test_nested_dicom_and_symlink_are_rejected(self):
        nested = self.raw / "train/nested"
        nested.mkdir()
        (nested / "other.dicom").write_bytes(b"extra")
        self.assert_failed(self.run_audit(), "dicom_no_unexpected_image_paths")
        if hasattr(os, "symlink"):
            try:
                os.symlink(self.png / "train/train_a.png", self.png / "train/link.png")
            except OSError:
                return
            self.assert_failed(self.run_audit(), "png_no_linked_entries")

    def test_missing_root_is_fail_not_pass(self):
        report = MODULE.audit_dataset(self.base / "not-found", self.png)
        self.assert_failed(report, "audit_execution_completed")

    def test_source_metadata_failure_is_fail_not_pass(self):
        with patch.object(MODULE, "stamp", side_effect=PermissionError("denied")):
            report = self.run_audit()
        self.assert_failed(report, "audit_execution_completed")

    def test_atomic_new_output_and_refusal_to_overwrite(self):
        target = self.base / "audits/result.json"
        MODULE.atomic_new_json(target, {"status": "PASS"})
        self.assertEqual(json.loads(target.read_text())["status"], "PASS")
        with self.assertRaises(FileExistsError):
            MODULE.atomic_new_json(target, {"status": "FAIL"})
        self.assertEqual(json.loads(target.read_text())["status"], "PASS")
        self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_atomic_publication_on_project_filesystem(self):
        # Exercise /mnt/e NTFS/DrvFS too, rather than only Linux /tmp.
        with tempfile.TemporaryDirectory(prefix=".audit-publish-test-", dir=Path(__file__).parent) as directory:
            target = Path(directory) / "report.json"
            MODULE.atomic_new_json(target, {"status": "PASS"})
            self.assertEqual(json.loads(target.read_text()), {"status": "PASS"})
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["report.json"])

    def test_cli_production_counts_cannot_be_reduced(self):
        target = self.base / "cli.json"
        with contextlib.redirect_stdout(io.StringIO()):
            code = MODULE.main(["--dicom-root", str(self.raw), "--png-root", str(self.png),
                                "--audit-out", str(target)])
        self.assertEqual(code, 1)
        report = json.loads(target.read_text())
        self.assertEqual(report["expected_counts"], {"test": 3000, "train": 15000})
        self.assertEqual(report["expected_manifest_entries"], 18006)
        self.assertEqual(report["status"], "FAIL")

    def test_cli_output_inside_input_is_forbidden(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code = MODULE.main(["--dicom-root", str(self.raw), "--png-root", str(self.png),
                                "--audit-out", str(self.png / "audit.json")])
        self.assertEqual(code, 2)
        self.assertFalse((self.png / "audit.json").exists())

    def test_relative_path_rejects_traversal(self):
        for value in ("../outside.dicom", "/tmp/outside.dicom", "train//image.dicom", "train/./image.dicom", "train\\image.dicom"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MODULE.relative_name(value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
