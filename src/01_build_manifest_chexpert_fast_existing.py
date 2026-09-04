#!/usr/bin/env python
from pathlib import Path
import argparse
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

def key_from_csv_path(path_str):
    # CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg
    p = Path(str(path_str))
    parts = p.parts
    if len(parts) >= 3:
        return "/".join(parts[-3:])
    return str(path_str)

def key_from_real_path(path_str):
    p = Path(str(path_str))
    parts = p.parts
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/h/data/raw/chexpert/chexpertchestxrays-u20210408")
    parser.add_argument("--image-list", default="/mnt/h/cxr_larger_not_better/data/processed/chexpert_image_paths.txt")
    parser.add_argument("--out", default="/mnt/h/cxr_larger_not_better/data/processed/chexpert_manifest_raw.csv")
    parser.add_argument("--drop-missing", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    image_list = Path(args.image_list)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    csv_dir = root / "CheXpert-v1.0 batch 1 (validate & csv)"
    train_csv = csv_dir / "train.csv"
    valid_csv = csv_dir / "valid.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train.csv: {train_csv}")
    if not valid_csv.exists():
        raise FileNotFoundError(f"Missing valid.csv: {valid_csv}")
    if not image_list.exists():
        raise FileNotFoundError(f"Missing image list: {image_list}")

    print("train_csv:", train_csv)
    print("valid_csv:", valid_csv)
    print("image_list:", image_list)

    print("Loading image path list...")
    paths = [x.strip() for x in image_list.read_text().splitlines() if x.strip()]
    path_map = {}
    for p in paths:
        path_map[key_from_real_path(p)] = p
    print(f"Image paths loaded: {len(paths):,}")
    print(f"Unique image keys: {len(path_map):,}")

    rows = []

    for csv_path, split_name in [(train_csv, "train"), (valid_csv, "valid")]:
        df = pd.read_csv(csv_path)
        print(f"Reading {csv_path}: {len(df):,} rows")

        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"CheXpert {split_name}"):
            csv_path_value = str(r.get("Path", ""))
            key = key_from_csv_path(csv_path_value)
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

            rows.append(item)

    out_df = pd.DataFrame(rows)
    missing = (out_df["path"].fillna("") == "").sum()
    print(f"Rows before drop missing: {len(out_df):,}")
    print(f"Missing image paths: {missing:,}")

    if args.drop_missing:
        out_df = out_df[out_df["path"].fillna("") != ""].copy()
        print(f"Rows after drop missing: {len(out_df):,}")

    out_df.to_csv(out, index=False)
    print(f"Saved: {out}")
    print(out_df["split_original"].value_counts(dropna=False))
    print(out_df["view"].value_counts(dropna=False).head(20))

if __name__ == "__main__":
    main()
