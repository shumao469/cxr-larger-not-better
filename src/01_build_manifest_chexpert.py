#!/usr/bin/env python
from pathlib import Path
import argparse
import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

TARGETS = [
    "Cardiomegaly",
    "Edema",
    "Pleural_Effusion",
    "Atelectasis",
    "Consolidation",
    "No_Finding",
]

CHEXPERT_SOURCE = {
    "Cardiomegaly": "Cardiomegaly",
    "Edema": "Edema",
    "Pleural_Effusion": "Pleural Effusion",
    "Atelectasis": "Atelectasis",
    "Consolidation": "Consolidation",
    "No_Finding": "No Finding",
}

def key_from_path(path_str):
    # Use patient/study/view as matching key.
    parts = Path(str(path_str)).parts
    if len(parts) >= 3:
        return "/".join(parts[-3:])
    return str(path_str)

def infer_view(path_str, row):
    if "Frontal/Lateral" in row.index:
        return str(row["Frontal/Lateral"])
    s = str(path_str).lower()
    if "frontal" in s:
        return "Frontal"
    if "lateral" in s:
        return "Lateral"
    return ""

def build_image_index(root: Path, cache_path: Path):
    if cache_path.exists():
        print(f"Loading cached image index: {cache_path}")
        idx = pd.read_csv(cache_path)
        return dict(zip(idx["key"], idx["path"]))

    print(f"Indexing CheXpert image files under: {root}")
    records = []
    exts = {".jpg", ".jpeg", ".png"}

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if Path(fn).suffix.lower() in exts:
                p = Path(dirpath) / fn
                # Actual extracted paths end with patientXXXX/studyY/viewZ_xxx.jpg
                key = "/".join(p.parts[-3:])
                records.append({"key": key, "path": str(p)})

    idx = pd.DataFrame(records).drop_duplicates("key")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    idx.to_csv(cache_path, index=False)

    print(f"Saved image index: {cache_path}")
    print(f"Indexed images: {len(idx):,}")
    return dict(zip(idx["key"], idx["path"]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/h/data/raw/chexpert/chexpertchestxrays-u20210408")
    parser.add_argument("--out", default="/mnt/h/cxr_larger_not_better/data/processed/chexpert_manifest_raw.csv")
    parser.add_argument("--index-cache", default="/mnt/h/cxr_larger_not_better/data/processed/chexpert_image_index.csv")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Your extracted CheXpert structure:
    csv_dir = root / "CheXpert-v1.0 batch 1 (validate & csv)"
    train_csv = csv_dir / "train.csv"
    valid_csv = csv_dir / "valid.csv"

    print("Using train_csv:", train_csv)
    print("Using valid_csv:", valid_csv)

    if not train_csv.exists():
        raise FileNotFoundError(f"Cannot find train.csv: {train_csv}")
    if not valid_csv.exists():
        raise FileNotFoundError(f"Cannot find valid.csv: {valid_csv}")

    path_map = build_image_index(root, Path(args.index_cache))

    all_rows = []

    for csv_path, split_name in [(train_csv, "train"), (valid_csv, "valid")]:
        df = pd.read_csv(csv_path)
        print(f"Reading {csv_path}: {len(df):,} rows")

        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Building {split_name} manifest"):
            csv_path_value = str(r.get("Path", ""))
            key = key_from_path(csv_path_value)
            actual_path = path_map.get(key, "")

            patient_match = re.search(r"(patient\d+)", csv_path_value)
            study_match = re.search(r"(study\d+)", csv_path_value)

            patient_id = patient_match.group(1) if patient_match else ""
            study_id0 = study_match.group(1) if study_match else ""

            item = {
                "dataset": "chexpert",
                "patient_id": patient_id,
                "study_id": f"{patient_id}_{study_id0}" if patient_id and study_id0 else key,
                "image_id": key.replace("/", "_"),
                "path": actual_path,
                "png_path": actual_path,
                "dicom_path": "",
                "view": infer_view(csv_path_value, r),
                "split_original": split_name,
                "source_label_text": "",
                "age": np.nan,
                "sex": str(r.get("Sex", "")) if "Sex" in r.index else "",
            }

            for target in TARGETS:
                source_col = CHEXPERT_SOURCE[target]
                item[f"label_{target}"] = r[source_col] if source_col in df.columns else np.nan

            all_rows.append(item)

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(out, index=False)

    print(f"Saved: {out}")
    print(f"Rows: {len(out_df):,}")
    print(f"Missing image paths: {(out_df['path'].fillna('') == '').sum():,}")
    print("\nSplit:")
    print(out_df["split_original"].value_counts(dropna=False))
    print("\nView:")
    print(out_df["view"].value_counts(dropna=False).head(20))

if __name__ == "__main__":
    main()
