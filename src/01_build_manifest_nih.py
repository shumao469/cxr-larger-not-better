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

NIH_SOURCE = {
    "Cardiomegaly": "Cardiomegaly",
    "Edema": "Edema",
    "Pleural_Effusion": "Effusion",
    "Atelectasis": "Atelectasis",
    "Consolidation": "Consolidation",
    "No_Finding": "No Finding",
}

def read_list(path: Path):
    if not path.exists():
        return set()
    return set(x.strip() for x in path.read_text().splitlines() if x.strip())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/h/data/raw/nih")
    parser.add_argument("--out", default="/mnt/h/cxr_larger_not_better/data/processed/nih_manifest_raw.csv")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    data_csv = root / "Data_Entry_2017.csv"
    if not data_csv.exists():
        raise FileNotFoundError(f"Cannot find {data_csv}")

    print(f"Reading {data_csv}")
    df = pd.read_csv(data_csv)

    print("Indexing NIH image paths...")
    img_paths = list(root.rglob("*.png")) + list(root.rglob("*.jpg")) + list(root.rglob("*.jpeg"))
    path_map = {p.name: str(p) for p in img_paths}
    print(f"Indexed {len(path_map):,} image files.")

    train_val = read_list(root / "train_val_list.txt")
    test_list = read_list(root / "test_list.txt")

    rows = []
    missing = 0

    for _, r in df.iterrows():
        image_id = str(r["Image Index"])
        img_path = path_map.get(image_id, "")
        if not img_path:
            missing += 1

        finding_text = str(r.get("Finding Labels", ""))
        findings = set(x.strip() for x in finding_text.split("|") if x.strip())

        item = {
            "dataset": "nih",
            "patient_id": str(r.get("Patient ID", "")),
            "study_id": image_id.rsplit("_", 1)[0],
            "image_id": image_id,
            "path": img_path,
            "png_path": img_path,
            "dicom_path": "",
            "view": str(r.get("View Position", "")),
            "split_original": "test" if image_id in test_list else ("train_val" if image_id in train_val else "unknown"),
            "source_label_text": finding_text,
            "age": r.get("Patient Age", np.nan),
            "sex": r.get("Patient Gender", ""),
        }

        for target in TARGETS:
            source_label = NIH_SOURCE[target]
            item[f"label_{target}"] = 1 if source_label in findings else 0

        rows.append(item)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out, index=False)

    print(f"Saved: {out}")
    print(f"Rows: {len(out_df):,}")
    print(f"Missing image paths: {missing:,}")
    for target in TARGETS:
        print(target, int(out_df[f'label_{target}'].sum()))

if __name__ == "__main__":
    main()
