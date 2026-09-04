#!/usr/bin/env python
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

TARGETS = [
    "Cardiomegaly",
    "Edema",
    "Pleural_Effusion",
    "Atelectasis",
    "Consolidation",
    "No_Finding",
]

# VinDr / VinBigData class names.
VINDR_SOURCE = {
    "Cardiomegaly": {"cardiomegaly"},
    "Pleural_Effusion": {"pleural effusion"},
    "Atelectasis": {"atelectasis"},
    "Consolidation": {"consolidation"},
    "No_Finding": {"no finding"},
    # Edema is not a standard VinDr competition class. Keep as missing.
    "Edema": set(),
}

def norm_text(x):
    return str(x).strip().lower()

def collect_dicoms(root: Path):
    paths = list(root.rglob("*.dicom")) + list(root.rglob("*.dcm"))
    out = {}
    for p in paths:
        out[p.stem] = str(p)
    return out

def find_split_from_path(path_str):
    p = Path(path_str)
    parts = set(p.parts)
    if "train" in parts:
        return "train"
    if "test" in parts:
        return "kaggle_test"
    return "unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/h/kagglehub_cache/competitions/vinbigdata-chest-xray-abnormalities-detection")
    parser.add_argument("--png-root", default="/mnt/h/data/processed/vindr_png")
    parser.add_argument("--out", default="/mnt/h/cxr_larger_not_better/data/processed/vindr_manifest_raw.csv")
    args = parser.parse_args()

    root = Path(args.root)
    png_root = Path(args.png_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    train_csv = root / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Cannot find {train_csv}")

    print(f"Reading {train_csv}")
    ann = pd.read_csv(train_csv)
    if "image_id" not in ann.columns or "class_name" not in ann.columns:
        raise ValueError(f"Unexpected VinDr train.csv columns: {ann.columns.tolist()}")

    ann["class_name_norm"] = ann["class_name"].map(norm_text)

    # Aggregate image-level class names.
    class_by_image = (
        ann.groupby("image_id")["class_name_norm"]
        .apply(lambda s: sorted(set(s.dropna().astype(str))))
        .to_dict()
    )

    print("Indexing DICOM files...")
    dicom_map = collect_dicoms(root)
    print(f"Indexed {len(dicom_map):,} DICOM files.")

    rows = []
    for image_id, dicom_path in sorted(dicom_map.items()):
        split_original = find_split_from_path(dicom_path)
        class_set = set(class_by_image.get(image_id, []))

        item = {
            "dataset": "vindr",
            "patient_id": image_id,   # Kaggle version does not provide patient IDs.
            "study_id": image_id,
            "image_id": image_id,
            "path": str(png_root / ("train" if split_original == "train" else "test") / f"{image_id}.png"),
            "png_path": str(png_root / ("train" if split_original == "train" else "test") / f"{image_id}.png"),
            "dicom_path": dicom_path,
            "view": "PA_or_unknown",
            "split_original": split_original,
            "source_label_text": "|".join(sorted(class_set)) if class_set else "",
            "age": np.nan,
            "sex": "",
        }

        has_any_abnormal = any(c != "no finding" for c in class_set)

        for target in TARGETS:
            if split_original != "train":
                item[f"label_{target}"] = np.nan
                continue

            if target == "Edema":
                item[f"label_{target}"] = np.nan
            elif target == "No_Finding":
                item[f"label_{target}"] = 1 if ("no finding" in class_set and not has_any_abnormal) else 0
            else:
                source_names = VINDR_SOURCE[target]
                item[f"label_{target}"] = 1 if len(class_set.intersection(source_names)) > 0 else 0

        rows.append(item)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out, index=False)

    print(f"Saved: {out}")
    print(f"Rows: {len(out_df):,}")
    print(out_df["split_original"].value_counts(dropna=False))
    for target in TARGETS:
        col = f"label_{target}"
        print(target, out_df[col].sum(skipna=True), "non-missing:", out_df[col].notna().sum())

if __name__ == "__main__":
    main()
