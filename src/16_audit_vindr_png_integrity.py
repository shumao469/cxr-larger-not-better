#!/usr/bin/env python3
"""Read-only, single-process audit of the official VinDr DICOM/PNG pairs.

The CLI is fixed to the formal 15,000 train / 3,000 test release. It verifies
the existing full-DICOM SHA256 evidence, not the 191-GiB DICOM contents again.
Every PNG is CRC-verified and then reopened and fully decoded with Pillow.
No training, conversion, GPU imports, or changes to either image tree occur.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import platform
import re
import stat
import struct
import sys
import tempfile
import time
import warnings
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image, ImageFile, __version__ as pillow_version


EXPECTED_COUNTS = {"test": 3000, "train": 15000}
EXPECTED_MANIFEST_ENTRIES = 18006
OFFICIAL_MANIFEST_SHA256 = "175f64e334b3c8a0773d1580263ac64bd4ba72511cb7391f3ab503025a8ed5d4"
DICOM_EXTENSIONS = {".dicom", ".dcm"}
DICOM_DOWNLOAD_SIDECARS = {"test/index.html", "train/index.html"}
EVIDENCE_NAMES = (
    "SHA256SUMS.txt", "full_sha256_merge_summary.json",
    "download_verified.flag", "full_sha256_merge_audit.csv",
)
MAX_FAILURE_EXAMPLES = 50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_all_png_chunk_crcs(path: Path) -> int:
    """Pillow verify() omits IEND's CRC; explicitly check every chunk as well."""
    chunks = 0
    seen_idat = False
    with path.open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Invalid PNG signature")
        while True:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError("Missing or truncated PNG chunk header/IEND")
            length, kind = struct.unpack(">I4s", header)
            if length > 0x7FFFFFFF or not re.fullmatch(rb"[A-Za-z]{4}", kind):
                raise ValueError("Invalid PNG chunk length or type")
            if chunks == 0 and (kind != b"IHDR" or length != 13):
                raise ValueError("PNG must start with a 13-byte IHDR")
            if chunks > 0 and kind == b"IHDR":
                raise ValueError("Duplicate PNG IHDR")
            checksum = zlib.crc32(kind)
            remaining = length
            while remaining:
                data = stream.read(min(1024 * 1024, remaining))
                if not data:
                    raise ValueError(f"Truncated {kind!r} chunk")
                checksum = zlib.crc32(data, checksum)
                remaining -= len(data)
            encoded_crc = stream.read(4)
            if len(encoded_crc) != 4 or struct.unpack(">I", encoded_crc)[0] != checksum & 0xFFFFFFFF:
                raise ValueError(f"PNG CRC mismatch in {kind!r}")
            chunks += 1
            seen_idat |= kind == b"IDAT"
            if kind == b"IEND":
                if length != 0 or not seen_idat or stream.read(1):
                    raise ValueError("Invalid IEND, missing IDAT, or trailing PNG data")
                return chunks


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_link(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def stamp(path: Path) -> tuple[int, int, int]:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Not a regular file: {path}")
    return info.st_size, info.st_mtime_ns, info.st_ino


def relative_name(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value or "\\" in value or ":" in value or "\x00" in value
        or path.is_absolute() or ".." in path.parts
        or path.as_posix() != value or value == "."
    ):
        raise ValueError(f"Unsafe or noncanonical relative path: {value!r}")
    return value


def declared_path(value: str) -> Path:
    """Compare Windows evidence paths while running under WSL (and vice versa)."""
    value = str(value).replace("\\", "/")
    if os.name != "nt" and re.match(r"^[A-Za-z]:/", value):
        value = "/mnt/" + value[0].lower() + value[2:]
    elif os.name == "nt" and re.match(r"^/mnt/[A-Za-z]/", value):
        value = value[5].upper() + ":" + value[6:]
    return Path(value).resolve()


class Audit:
    def __init__(self, report: dict):
        self.report = report
        report["checks"] = {}
        report["failure_examples"] = []

    def example(self, category: str, detail: object) -> None:
        if len(self.report["failure_examples"]) < MAX_FAILURE_EXAMPLES:
            self.report["failure_examples"].append(
                {"category": category, "detail": detail}
            )

    def check(self, name: str, passed: bool, detail: object) -> None:
        if name in self.report["checks"]:
            raise ValueError(f"Duplicate audit check: {name}")
        self.report["checks"][name] = {"passed": bool(passed), "detail": detail}
        if not passed:
            self.example(name, detail)


def evidence_snapshot(root: Path) -> dict:
    result = {}
    for name in EVIDENCE_NAMES:
        path = root / name
        if is_link(path):
            raise ValueError(f"Evidence must not be a link: {path}")
        before = stamp(path)
        digest = sha256_file(path)
        if stamp(path) != before:
            raise RuntimeError(f"Evidence changed while hashing: {path}")
        result[name] = {"sha256": digest, "stat": list(before)}
    return result


def check_source_evidence(
    root: Path, audit: Audit, expected_counts: dict[str, int],
    expected_entries: int, expected_manifest_sha256: str,
) -> tuple[dict, dict[str, tuple[int, int, int]]]:
    snapshot = evidence_snapshot(root)
    manifest_hash = snapshot["SHA256SUMS.txt"]["sha256"]
    summary = json.loads((root / EVIDENCE_NAMES[1]).read_text(encoding="utf-8-sig"))
    flag = json.loads((root / EVIDENCE_NAMES[2]).read_text(encoding="utf-8-sig"))
    audit.report["source_evidence"] = {
        "files": snapshot, "summary": summary, "verified_flag": flag,
        "dicom_contents_rehashed": False,
        "scope": "Prior full SHA256 evidence plus current file paths and sizes; not a new DICOM content hash.",
    }
    audit.check("official_manifest_sha256", manifest_hash == expected_manifest_sha256,
                {"actual": manifest_hash, "expected": expected_manifest_sha256})
    audit.check("summary_pass_and_counts", summary.get("status") == "PASS"
                and summary.get("manifest_entries") == expected_entries
                and summary.get("verified") == expected_entries
                and all(summary.get(key) == 0 for key in ("missing", "source_mismatch", "errors")),
                {key: summary.get(key) for key in
                 ("status", "manifest_entries", "verified", "missing", "source_mismatch", "errors")})
    audit.check("flag_pass_and_count", flag.get("status") == "PASS"
                and flag.get("verified") == expected_entries,
                {"status": flag.get("status"), "verified": flag.get("verified")})
    audit.check("manifest_hash_matches_both_markers",
                summary.get("manifest_sha256") == flag.get("manifest_sha256") == manifest_hash,
                {"manifest": manifest_hash, "summary": summary.get("manifest_sha256"),
                 "flag": flag.get("manifest_sha256")})
    audit.check("marker_completion_times_match",
                bool(summary.get("completed_at")) and summary.get("completed_at") == flag.get("completed_at"),
                {"summary": summary.get("completed_at"), "flag": flag.get("completed_at")})
    for owner, document, field, target in (
        ("summary", summary, "physionet_root", root),
        ("summary", summary, "manifest", root / EVIDENCE_NAMES[0]),
        ("summary", summary, "audit_csv", root / EVIDENCE_NAMES[3]),
        ("flag", flag, "summary", root / EVIDENCE_NAMES[1]),
        ("flag", flag, "audit_csv", root / EVIDENCE_NAMES[3]),
    ):
        value = document.get(field)
        audit.check(f"{owner}_{field}_path", isinstance(value, str)
                    and bool(value) and declared_path(value) == target.resolve(),
                    {"declared": value, "expected": str(target)})

    manifest = {}
    manifest_folded = set()
    manifest_rows = 0
    with (root / EVIDENCE_NAMES[0]).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            digest, name = line.rstrip("\r\n").split(maxsplit=1)
            if name.startswith("*"):
                name = name[1:]
            name = relative_name(name)
            if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                raise ValueError(f"Invalid SHA256 at manifest line {line_number}")
            if name.casefold() in manifest_folded:
                raise ValueError(f"Duplicate manifest path: {name}")
            manifest[name] = digest.lower()
            manifest_folded.add(name.casefold())
            manifest_rows += 1
    audit.check("checksum_manifest_entry_count", manifest_rows == expected_entries,
                {"actual": manifest_rows, "expected": expected_entries})
    for split, expected in sorted(expected_counts.items()):
        paths = [name for name in manifest if PurePosixPath(name).parts[0] == split]
        valid = [name for name in paths if len(PurePosixPath(name).parts) == 2
                 and PurePosixPath(name).suffix.lower() in DICOM_EXTENSIONS]
        audit.check(f"checksum_manifest_{split}_official_layout",
                    len(paths) == len(valid) == expected,
                    {"paths": len(paths), "direct_dicoms": len(valid), "expected": expected})

    required = {"relative_path", "expected_sha256", "actual_sha256", "bytes", "status", "error"}
    rows = []
    with (root / EVIDENCE_NAMES[3]).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not required.issubset(fields) or len(fields) != len(set(fields)):
            raise ValueError(f"Invalid audit CSV schema: {fields}")
        rows = list(reader)
    audit_names = []
    counts = Counter()
    source_stats = {}
    for index, row in enumerate(rows, 1):
        try:
            if None in row or any(row.get(key) is None for key in required):
                raise ValueError("Malformed CSV row")
            name = relative_name(row["relative_path"])
            audit_names.append(name)
            expected = row["expected_sha256"]
            actual = row["actual_sha256"]
            if (row["status"] != "PASS" or row["error"] or
                    not re.fullmatch(r"[0-9a-f]{64}", expected) or
                    actual != expected or manifest.get(name) != expected):
                raise ValueError("Not PASS, error present, or SHA256 differs from manifest")
            counts["pass_and_matching_hash"] += 1
            path = root / name
            # Reject links in every path component without traversing the quarantine.
            cursor = root
            for part in PurePosixPath(name).parts:
                cursor = cursor / part
                if is_link(cursor):
                    raise ValueError(f"Linked source path: {name}")
            info = stamp(path)
            if int(row["bytes"]) != info[0] or info[0] <= 0:
                raise ValueError(f"Current size differs from prior audit: {info[0]} vs {row['bytes']}")
            source_stats[name] = info
            counts["current_path_and_size_match"] += 1
        except Exception as exc:
            counts["invalid_rows"] += 1
            audit.example("source_audit_row", {"row": index, "path": row.get("relative_path"),
                                                "error": f"{type(exc).__name__}: {exc}"})
    duplicates = len(audit_names) - len({name.casefold() for name in audit_names})
    for key in ("pass_and_matching_hash", "current_path_and_size_match", "invalid_rows"):
        counts.setdefault(key, 0)
    audit.check("source_audit_rows_all_pass", len(rows) == expected_entries
                and counts["invalid_rows"] == 0 and counts["pass_and_matching_hash"] == expected_entries,
                {"rows": len(rows), "expected": expected_entries, **dict(counts)})
    audit.check("source_audit_paths_exactly_match_manifest",
                duplicates == 0 and set(audit_names) == set(manifest),
                {"duplicate_paths": duplicates, "missing": len(set(manifest) - set(audit_names)),
                 "unexpected": len(set(audit_names) - set(manifest))})
    audit.report["source_evidence"]["row_counts"] = {"manifest": manifest_rows,
                                                    "audit_csv": len(rows), **dict(counts)}
    return snapshot, source_stats


def image_inventory(root: Path, *, png: bool, audit: Audit) -> dict[str, list[Path]]:
    label = "png" if png else "dicom"
    extension_set = {".png"} if png else DICOM_EXTENSIONS
    found = {"test": [], "train": []}
    unexpected = []
    artifacts = []
    linked = []
    download_sidecars = []
    # The DICOM root intentionally also contains annotations and quarantine.
    # Only its two official split directories are inspected/indexed.
    scan_roots = [root] if png else [root / split for split in sorted(found)]
    for scan_root in scan_roots:
        if not scan_root.is_dir() or is_link(scan_root):
            raise ValueError(f"Missing or linked {label} directory: {scan_root}")
        for dirname, directories, filenames in os.walk(scan_root, followlinks=False, onerror=lambda err: (_ for _ in ()).throw(err)):
            directories.sort()
            filenames.sort()
            base = Path(dirname)
            for name in list(directories):
                path = base / name
                if not png and path.relative_to(root).as_posix() in DICOM_DOWNLOAD_SIDECARS:
                    unexpected.append(path.relative_to(root).as_posix())
                lowered = name.lower()
                if lowered.endswith((".tmp", ".part", ".partial")) or ".tmp." in lowered or "failure" in lowered or "failed" in lowered:
                    artifacts.append(path.relative_to(root).as_posix())
                if is_link(path):
                    linked.append(str(path.relative_to(root)))
                    directories.remove(name)
            for name in filenames:
                path = base / name
                relative = path.relative_to(root)
                if is_link(path):
                    linked.append(relative.as_posix())
                    continue
                lowered = name.lower()
                if lowered.endswith((".tmp", ".part", ".partial")) or ".tmp." in lowered or "failure" in lowered or "failed" in lowered:
                    artifacts.append(relative.as_posix())
                if path.suffix.lower() in extension_set:
                    if len(relative.parts) == 2 and relative.parts[0] in found:
                        found[relative.parts[0]].append(path)
                    else:
                        unexpected.append(relative.as_posix())
                elif not png and relative.as_posix() in DICOM_DOWNLOAD_SIDECARS:
                    # wget may retain these exact directory-index pages. They
                    # are not official manifest entries or cohort images.
                    before = stamp(path)
                    digest = sha256_file(path)
                    if stamp(path) != before:
                        raise RuntimeError(f"Download sidecar changed while hashing: {path}")
                    download_sidecars.append({"relative_path": relative.as_posix(),
                                              "path": str(path), "bytes": before[0],
                                              "sha256": digest, "stat": list(before)})
                else:
                    unexpected.append(relative.as_posix())
    if not png:
        audit.report["dicom_download_sidecars"] = {
            "allowed_relative_paths": sorted(DICOM_DOWNLOAD_SIDECARS),
            "classification": "Optional wget directory-index pages; excluded from the official manifest and image cohort.",
            "count": len(download_sidecars), "files": download_sidecars,
        }
    audit.check(f"{label}_no_linked_entries", not linked, {"count": len(linked), "examples": linked[:10]})
    audit.check(f"{label}_no_unexpected_image_paths", not unexpected,
                {"count": len(unexpected), "examples": unexpected[:10]})
    audit.check(f"{label}_no_incomplete_or_failure_artifacts", not artifacts,
                {"count": len(artifacts), "examples": artifacts[:10]})
    return {split: sorted(paths, key=lambda p: p.name) for split, paths in found.items()}


def conversion_provenance() -> dict:
    project = Path(__file__).resolve().parents[1]
    converter = project / "src/04_convert_vindr_dicom_to_png_v2.py"
    runner = project / "scripts/run_vindr_conversion_formal_v2.sh"
    result = {"audit_script": {"path": str(Path(__file__).resolve()),
                                "sha256": sha256_file(Path(__file__))},
              "python": sys.version, "platform": platform.platform(), "pillow": pillow_version}
    for label, path in (("converter", converter), ("conversion_runner", runner)):
        result[label] = {"path": str(path), "available": path.is_file()}
        if path.is_file():
            result[label]["sha256"] = sha256_file(path)
    if converter.is_file():
        defaults = {}
        for node in ast.walk(ast.parse(converter.read_text(encoding="utf-8-sig"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument" and node.args
                    and isinstance(node.args[0], ast.Constant)):
                option = node.args[0].value
                if option in ("--lower-percentile", "--upper-percentile"):
                    for keyword in node.keywords:
                        if keyword.arg == "default":
                            defaults[option] = ast.literal_eval(keyword.value)
        result["converter_argument_defaults"] = defaults
        result["parameter_evidence_scope"] = "Current converter source defaults; hashes identify current scripts, not independent proof of every historical invocation."
        if runner.is_file():
            contents = runner.read_text(encoding="utf-8-sig")
            result["runner_mentions_percentile_override"] = any(key in contents for key in defaults)
    return result


def audit_dataset(
    dicom_root: Path, png_root: Path, *,
    expected_counts: dict[str, int] | None = None,
    expected_manifest_entries: int = EXPECTED_MANIFEST_ENTRIES,
    expected_manifest_sha256: str = OFFICIAL_MANIFEST_SHA256,
) -> dict:
    """Internal expected-count injection is for fixtures; the CLI exposes no override."""
    began = time.monotonic()
    counts_expected = dict(EXPECTED_COUNTS if expected_counts is None else expected_counts)
    report = {"schema_version": 1, "status": "FAIL", "started_at": utc_now(),
              "expected_counts": counts_expected, "expected_manifest_entries": expected_manifest_entries,
              "dicom_root": str(dicom_root), "png_root": str(png_root),
              "execution": {"cpu_only": True, "processes": 1, "order": "test then train; filename ascending"}}
    audit = Audit(report)
    try:
        if set(counts_expected) != {"test", "train"} or any(type(n) is not int or n <= 0 for n in counts_expected.values()):
            raise ValueError("Expected counts must contain positive integer test/train counts")
        for root in (dicom_root, png_root):
            if not root.is_dir() or is_link(root):
                raise ValueError(f"Missing or linked image root: {root}")
        dicom_root, png_root = dicom_root.resolve(), png_root.resolve()
        if dicom_root == png_root or dicom_root in png_root.parents or png_root in dicom_root.parents:
            raise ValueError("DICOM and PNG roots must be disjoint")
        report["dicom_root"], report["png_root"] = str(dicom_root), str(png_root)
        report["provenance"] = conversion_provenance()
        evidence_before, source_stats = check_source_evidence(
            dicom_root, audit, counts_expected, expected_manifest_entries, expected_manifest_sha256)
        dicoms = image_inventory(dicom_root, png=False, audit=audit)
        pngs = image_inventory(png_root, png=True, audit=audit)
        inventories = {}
        for label, inventory in (("dicom", dicoms), ("png", pngs)):
            stems = [path.stem.casefold() for split in sorted(inventory) for path in inventory[split]]
            duplicate_count = len(stems) - len(set(stems))
            audit.check(f"{label}_global_stems_unique", duplicate_count == 0,
                        {"total": len(stems), "duplicates": duplicate_count})
            inventories[label] = {split: len(files) for split, files in inventory.items()}
        report["image_counts"] = inventories
        for split, expected in sorted(counts_expected.items()):
            audit.check(f"{split}_exact_image_counts", len(dicoms[split]) == len(pngs[split]) == expected,
                        {"dicom": len(dicoms[split]), "png": len(pngs[split]), "expected": expected})
            dcm_stems = {path.stem for path in dicoms[split]}
            png_stems = {path.stem for path in pngs[split]}
            audit.check(f"{split}_png_stems_match_dicom", dcm_stems == png_stems,
                        {"missing_png_stems": sorted(dcm_stems - png_stems)[:10],
                         "extra_png_stems": sorted(png_stems - dcm_stems)[:10],
                         "missing_count": len(dcm_stems - png_stems), "extra_count": len(png_stems - dcm_stems)})
            official = {name for name in source_stats if PurePosixPath(name).parts[0] == split}
            actual = {path.relative_to(dicom_root).as_posix() for path in dicoms[split]}
            audit.check(f"{split}_dicom_paths_match_verified_source", actual == official,
                        {"missing": len(official - actual), "unexpected": len(actual - official)})

        ImageFile.LOAD_TRUNCATED_IMAGES = False
        image_counts = Counter()
        dimensions = Counter()
        png_stats = {}
        for split in sorted(pngs):
            for path in pngs[split]:
                relative = path.relative_to(png_root).as_posix()
                image_counts["attempted"] += 1
                try:
                    before = stamp(path)
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", Image.DecompressionBombWarning)
                        with Image.open(path) as image:
                            if image.format != "PNG" or image.mode != "L" or min(image.size) <= 0:
                                raise ValueError(f"Expected PNG/L/positive size, got {image.format}/{image.mode}/{image.size}")
                            if image.n_frames != 1:
                                raise ValueError(f"Expected a single-frame PNG, got n_frames={image.n_frames}")
                            image.verify()
                        image_counts["crc_verify_passed"] += 1
                        image_counts["chunks_crc_checked"] += verify_all_png_chunk_crcs(path)
                        image_counts["all_chunk_crcs_passed"] += 1
                        with Image.open(path) as image:
                            image.load()
                            if image.format != "PNG" or image.mode != "L" or min(image.size) <= 0 or image.n_frames != 1:
                                raise ValueError("Format/mode/size/single-frame check failed after full decode")
                            dimensions[f"{image.width}x{image.height}"] += 1
                        image_counts["full_decode_passed"] += 1
                    if stamp(path) != before:
                        raise RuntimeError("PNG changed during verify/decode")
                    png_stats[relative] = before
                    image_counts["passed"] += 1
                except Exception as exc:
                    image_counts["failed"] += 1
                    audit.example("png_integrity", {"path": relative, "error": f"{type(exc).__name__}: {exc}"})
                if image_counts["attempted"] % 500 == 0:
                    print(f"PNG audit {image_counts['attempted']}/{sum(counts_expected.values())}; failed={image_counts['failed']}", flush=True)
        for key in ("attempted", "crc_verify_passed", "all_chunk_crcs_passed", "chunks_crc_checked", "full_decode_passed", "passed", "failed"):
            image_counts.setdefault(key, 0)
        report["png_validation"] = {**dict(image_counts), "dimension_counts": dict(sorted(dimensions.items()))}
        audit.check("all_png_crc_and_decode_pass", image_counts["failed"] == 0
                    and image_counts["passed"] == sum(counts_expected.values()), dict(image_counts))
        audit.check("source_evidence_unchanged", evidence_snapshot(dicom_root) == evidence_before,
                    "Rehashed the four small evidence files after PNG decoding")
        changed_sources = [name for name, before in sorted(source_stats.items()) if stamp(dicom_root / name) != before]
        audit.check("source_metadata_unchanged", not changed_sources,
                    {"checked": len(source_stats), "changed": len(changed_sources), "examples": changed_sources[:10]})
        changed_pngs = [name for name, before in sorted(png_stats.items()) if stamp(png_root / name) != before]
        audit.check("decoded_png_metadata_unchanged", not changed_pngs,
                    {"checked": len(png_stats), "changed": len(changed_pngs), "examples": changed_pngs[:10]})
        ending_report = {}
        ending_audit = Audit(ending_report)
        ending_pngs = image_inventory(png_root, png=True, audit=ending_audit)
        ending_dicoms = image_inventory(dicom_root, png=False, audit=ending_audit)
        sidecars_before = report["dicom_download_sidecars"]["files"]
        sidecars_after = ending_report["dicom_download_sidecars"]["files"]
        report["dicom_download_sidecars"]["files_after"] = sidecars_after
        report["dicom_download_sidecars"]["unchanged"] = sidecars_before == sidecars_after
        audit.check("dicom_download_sidecars_unchanged", sidecars_before == sidecars_after,
                    {"start_count": len(sidecars_before), "end_count": len(sidecars_after),
                     "comparison": "Exact paths, bytes, SHA256, and file metadata before and after PNG decoding"})
        audit.check("image_inventory_unchanged", ending_pngs == pngs and ending_dicoms == dicoms
                    and all(item["passed"] for item in ending_report["checks"].values()),
                    ending_report)
        report["status"] = "PASS" if all(item["passed"] for item in report["checks"].values()) else "FAIL"
    except Exception as exc:
        audit.check("audit_execution_completed", False, f"{type(exc).__name__}: {exc}")
    report["failed_check_count"] = sum(not item["passed"] for item in report["checks"].values())
    report["completed_at"] = utc_now()
    report["elapsed_seconds"] = round(time.monotonic() - began, 3)
    return report


def atomic_new_json(path: Path, report: dict) -> None:
    """Publish one new audit; never silently replace an older audit result."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Audit output already exists; choose a new path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and, unlike replace(), refuses a
        # concurrent destination. Both files are in the same local filesystem.
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-root", required=True, type=Path)
    parser.add_argument("--png-root", required=True, type=Path)
    parser.add_argument("--audit-out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        target = args.audit_out.resolve()
        for root in (args.dicom_root.resolve(), args.png_root.resolve()):
            if target == root or root in target.parents:
                raise ValueError("Audit output must be outside both read-only image roots")
        if args.audit_out.exists() or args.audit_out.is_symlink():
            raise FileExistsError("Refusing to overwrite existing audit output")
        report = audit_dataset(args.dicom_root, args.png_root)
        atomic_new_json(target, report)
        print(json.dumps({"status": report["status"], "audit_out": str(target),
                          "failed_checks": report["failed_check_count"],
                          "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)
        return 0 if report["status"] == "PASS" else 1
    except Exception as exc:
        print(f"PNG audit failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
