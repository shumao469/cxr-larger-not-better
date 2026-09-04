#!/usr/bin/env python3
"""Rebuild the model manifest from the recovered frozen split manifest.

This script preserves the original patient-level splits and cohort membership.
It only resolves image paths against newly downloaded image roots, then performs
strict count, duplicate, split-overlap, and file-existence checks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath

import pandas as pd


PRIMARY_LABELS = [
    "y_Cardiomegaly",
    "y_Pleural_Effusion",
    "y_Atelectasis",
    "y_Consolidation",
]
KEEP_SPLITS = ["train", "val", "internal_test", "external_test"]
EXPECTED_COUNTS = {
    ("chexpert", "train"): 103286,
    ("chexpert", "val"): 15047,
    ("chexpert", "internal_test"): 28986,
    ("nih", "train"): 79481,
    ("nih", "val"): 11342,
    ("nih", "internal_test"): 21297,
    ("vindr", "external_test"): 15000,
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def iter_images(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"Image root does not exist: {root}")
    for dirpath, _, filenames in os.walk(root):
        base = Path(dirpath)
        for name in filenames:
            if Path(name).suffix.lower() in IMAGE_EXTS:
                yield base / name


def unique_index(paths, key_fn, dataset_name):
    index = {}
    duplicates = []
    for path in paths:
        key = key_fn(path)
        if key in index and index[key] != path:
            duplicates.append((key, str(index[key]), str(path)))
        else:
            index[key] = path
    if duplicates:
        sample = duplicates[:5]
        raise RuntimeError(f"Duplicate {dataset_name} image keys; examples: {sample}")
    print(f"Indexed {len(index):,} {dataset_name} images")
    return index


def chexpert_key_from_path(value: str) -> str:
    value = str(value).replace("\\", "/")
    parts = PurePosixPath(value).parts
    if len(parts) < 3:
        return value
    return "/".join(parts[-3:]).lower()


def patient_split_overlap(df: pd.DataFrame):
    rows = []
    for dataset, group in df.groupby("dataset"):
        sets = {
            split: set(
                sub["patient_id"].fillna(sub["image_id"]).astype(str)
            )
            for split, sub in group.groupby("split")
        }
        names = sorted(sets)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                n = len(sets[left].intersection(sets[right]))
                rows.append(
                    {
                        "dataset": dataset,
                        "split_a": left,
                        "split_b": right,
                        "overlap_patients": n,
                    }
                )
    return pd.DataFrame(rows)


def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False)
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        default="data/processed/harmonized_manifest_all_splits.csv",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["nih", "chexpert", "vindr"],
        default=["nih", "chexpert", "vindr"],
        help=(
            "Datasets to resolve. The default builds the final three-dataset "
            "manifest; use '--datasets nih chexpert' for training before VinDr "
            "is available."
        ),
    )
    parser.add_argument("--nih-root")
    parser.add_argument("--chexpert-root")
    parser.add_argument("--vindr-png-root")
    parser.add_argument(
        "--out", default="data/processed/model_manifest_formal_v2.csv"
    )
    parser.add_argument(
        "--audit-out",
        default="data/processed/model_manifest_formal_v2_audit.json",
    )
    parser.add_argument(
        "--missing-out",
        default="data/processed/model_manifest_formal_v2_missing.csv",
    )
    args = parser.parse_args()

    source_path = Path(args.source_manifest)
    df = pd.read_csv(source_path, low_memory=False)
    missing_cols = [
        col
        for col in [
            "dataset",
            "split",
            "patient_id",
            "image_id",
            "image_path_final",
            *PRIMARY_LABELS,
        ]
        if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"Source manifest is missing columns: {missing_cols}")

    selected_datasets = list(dict.fromkeys(args.datasets))
    df = df[df["dataset"].astype(str).str.lower().isin(selected_datasets)].copy()
    df = df[df["split"].isin(KEEP_SPLITS)].copy()
    df = df[df[PRIMARY_LABELS].notna().any(axis=1)].copy()
    print(f"Frozen labeled cohort rows: {len(df):,}")

    roots = {
        "nih": args.nih_root,
        "chexpert": args.chexpert_root,
        "vindr": args.vindr_png_root,
    }
    missing_roots = [dataset for dataset in selected_datasets if not roots[dataset]]
    if missing_roots:
        raise ValueError(f"Missing image roots for selected datasets: {missing_roots}")

    indexes = {}
    if "nih" in selected_datasets:
        indexes["nih"] = unique_index(
            iter_images(Path(args.nih_root)), lambda p: p.name.lower(), "NIH"
        )
    if "chexpert" in selected_datasets:
        indexes["chexpert"] = unique_index(
            iter_images(Path(args.chexpert_root)),
            lambda p: "/".join(x.lower() for x in p.parts[-3:]),
            "CheXpert",
        )
    if "vindr" in selected_datasets:
        indexes["vindr"] = unique_index(
            iter_images(Path(args.vindr_png_root)), lambda p: p.stem.lower(), "VinDr"
        )

    resolved = []
    missing = []
    for row_index, row in df.iterrows():
        dataset = str(row["dataset"]).lower()
        if dataset == "nih":
            key = str(row["image_id"]).lower()
            path = indexes["nih"].get(key)
        elif dataset == "chexpert":
            key = chexpert_key_from_path(row["image_path_final"])
            path = indexes["chexpert"].get(key)
        elif dataset == "vindr":
            key = str(row["image_id"]).lower()
            path = indexes["vindr"].get(key)
        else:
            key = str(row["image_id"])
            path = None

        if path is None:
            missing.append(
                {
                    "row_index": row_index,
                    "dataset": dataset,
                    "split": row["split"],
                    "image_id": row["image_id"],
                    "lookup_key": key,
                    "old_path": row["image_path_final"],
                }
            )
            resolved.append("")
        else:
            resolved.append(str(path.resolve()))

    df["image_path_final"] = resolved
    df["path"] = resolved
    df["png_path"] = resolved

    missing_df = pd.DataFrame(missing)
    missing_path = Path(args.missing_out)
    if not missing_df.empty:
        atomic_csv(missing_df, missing_path)
        raise RuntimeError(
            f"Missing {len(missing_df):,} images. See {missing_path}; final manifest was not written."
        )
    if missing_path.exists():
        missing_path.unlink()

    duplicate_ids = int(df.duplicated(["dataset", "image_id"]).sum())
    counts = df.groupby(["dataset", "split"]).size().to_dict()
    selected_expected_counts = {
        key: value
        for key, value in EXPECTED_COUNTS.items()
        if key[0] in selected_datasets
    }
    count_mismatches = {
        f"{dataset}/{split}": {"expected": expected, "observed": counts.get((dataset, split), 0)}
        for (dataset, split), expected in selected_expected_counts.items()
        if counts.get((dataset, split), 0) != expected
    }
    unexpected = {
        f"{dataset}/{split}": int(value)
        for (dataset, split), value in counts.items()
        if (dataset, split) not in selected_expected_counts
    }
    overlap = patient_split_overlap(df)
    overlap_total = int(overlap["overlap_patients"].sum()) if len(overlap) else 0

    audit = {
        "source_manifest": str(source_path.resolve()),
        "selected_datasets": selected_datasets,
        "output_rows": int(len(df)),
        "expected_output_rows": int(sum(selected_expected_counts.values())),
        "counts": {f"{k[0]}/{k[1]}": int(v) for k, v in counts.items()},
        "count_mismatches": count_mismatches,
        "unexpected_dataset_splits": unexpected,
        "duplicate_dataset_image_ids": duplicate_ids,
        "patient_split_overlap_total": overlap_total,
        "all_paths_resolved": True,
    }
    audit_path = Path(args.audit_out)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    overlap.to_csv(audit_path.with_name(audit_path.stem + "_overlap.csv"), index=False)

    if count_mismatches or unexpected or duplicate_ids or overlap_total:
        raise RuntimeError(f"Manifest audit failed. See {audit_path}")

    df["file_exists"] = True
    df["primary_label_available"] = True
    out_path = Path(args.out)
    atomic_csv(df, out_path)
    print(f"Saved validated manifest: {out_path}")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
