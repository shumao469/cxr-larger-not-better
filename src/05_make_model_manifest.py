#!/usr/bin/env python
from pathlib import Path
import argparse
import pandas as pd

PRIMARY_LABELS = [
    "y_Cardiomegaly",
    "y_Pleural_Effusion",
    "y_Atelectasis",
    "y_Consolidation",
]

KEEP_SPLITS = ["train", "val", "internal_test", "external_test"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/harmonized_manifest_all_splits.csv")
    parser.add_argument("--out", default="data/processed/model_manifest_all_splits.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)

    print("Input rows:", len(df))
    print(pd.crosstab(df["dataset"], df["split"], dropna=False))

    df["image_path_final"] = df["image_path_final"].fillna("").astype(str)
    df["file_exists"] = df["image_path_final"].map(lambda x: Path(x).exists() if x else False)
    df["primary_label_available"] = df[PRIMARY_LABELS].notna().any(axis=1)

    df = df[df["split"].isin(KEEP_SPLITS)].copy()
    df = df[df["file_exists"]].copy()
    df = df[df["primary_label_available"]].copy()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\nSaved:", out)
    print("Output rows:", len(df))
    print("\nDataset x split:")
    print(pd.crosstab(df["dataset"], df["split"], dropna=False))

    print("\nPrimary label availability:")
    for c in PRIMARY_LABELS:
        print(c, "available:", int(df[c].notna().sum()), "positive:", int(df[c].sum(skipna=True)))

if __name__ == "__main__":
    main()
