#!/usr/bin/env python3
"""Convert VinDr-CXR DICOM images to deterministic 8-bit PNG files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut


def convert(ds, lower_percentile: float, upper_percentile: float):
    pixels = ds.pixel_array
    try:
        pixels = apply_modality_lut(pixels, ds)
    except Exception:
        pixels = np.asarray(pixels)
    try:
        pixels = apply_voi_lut(pixels, ds)
    except Exception:
        pass

    image = np.asarray(pixels, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("DICOM pixel array has no finite values")
    values = image[finite]
    lo, hi = np.percentile(values, [lower_percentile, upper_percentile])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        raise ValueError("DICOM pixel array has no usable intensity range")

    image = np.clip(image, lo, hi)
    image = (image - lo) / (hi - lo)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        image = 1.0 - image
    return np.rint(image * 255.0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-root", required=True)
    parser.add_argument("--png-root", required=True)
    parser.add_argument("--lower-percentile", type=float, default=0.5)
    parser.add_argument("--upper-percentile", type=float, default=99.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dicom_root = Path(args.dicom_root)
    png_root = Path(args.png_root)
    if not dicom_root.exists():
        raise FileNotFoundError(dicom_root)

    dicoms = sorted(
        p for p in dicom_root.rglob("*") if p.is_file() and p.suffix.lower() in {".dicom", ".dcm"}
    )
    if not dicoms:
        raise RuntimeError(f"No DICOM images found below {dicom_root}")
    print(f"DICOM files: {len(dicoms):,}")

    failures = []
    converted = 0
    skipped = 0
    for index, source in enumerate(dicoms, start=1):
        relative_parent = source.parent.relative_to(dicom_root)
        target = png_root / relative_parent / f"{source.stem}.png"
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            ds = pydicom.dcmread(source)
            array = convert(ds, args.lower_percentile, args.upper_percentile)
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(".png.tmp")
            Image.fromarray(array, mode="L").save(temp, format="PNG")
            os.replace(temp, target)
            converted += 1
        except Exception as exc:
            failures.append((str(source), repr(exc)))
        if index % 500 == 0:
            print(f"Processed {index:,}/{len(dicoms):,}; converted={converted:,}; failures={len(failures):,}")

    if failures:
        failure_path = png_root / "conversion_failures.tsv"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            "source\terror\n" + "\n".join(f"{p}\t{e}" for p, e in failures),
            encoding="utf-8",
        )
        raise RuntimeError(f"Conversion failed for {len(failures):,} files; see {failure_path}")
    print(f"Completed. Converted={converted:,}; skipped={skipped:,}; failures=0")


if __name__ == "__main__":
    main()
