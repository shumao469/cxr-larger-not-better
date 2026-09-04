#!/usr/bin/env python
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

LABELS = [
    "y_Cardiomegaly",
    "y_Pleural_Effusion",
    "y_Atelectasis",
    "y_Consolidation",
]

def make_patient_subset(df, n_target, seed):
    rng = np.random.default_rng(seed)

    if str(n_target).lower() == "all":
        return df.copy()

    n_target = int(n_target)

    # Keep only rows with at least one available selected label.
    df = df[df[LABELS].notna().any(axis=1)].copy()

    patients = df["patient_id"].fillna(df["image_id"]).astype(str).drop_duplicates().to_numpy()
    rng.shuffle(patients)

    selected = []
    total_rows = 0
    by_patient = df.groupby(df["patient_id"].fillna(df["image_id"]).astype(str), sort=False)

    for pid in patients:
        selected.append(pid)
        total_rows += len(by_patient.get_group(pid))
        if total_rows >= n_target:
            break

    out = df[df["patient_id"].fillna(df["image_id"]).astype(str).isin(selected)].copy()

    # If patient-based selection exceeds target substantially, keep all rows because this preserves patient integrity.
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/model_manifest_all_splits.csv")
    parser.add_argument("--out-dir", default="data/subsets")
    parser.add_argument("--train-sources", nargs="+", default=["nih", "chexpert"])
    parser.add_argument("--sizes", nargs="+", default=["1000", "5000", "10000", "50000", "all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()

    df = pd.read_csv(args.manifest, low_memory=False)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for source in args.train_sources:
        source_train = df[(df["dataset"] == source) & (df["split"] == "train")].copy()
        source_train = source_train[source_train[LABELS].notna().any(axis=1)].copy()

        print(f"\nSource: {source}")
        print("Available training rows:", len(source_train))
        print("Patients:", source_train["patient_id"].nunique())

        for size in args.sizes:
            for seed in args.seeds:
                sub = make_patient_subset(source_train, size, seed)

                out_path = out_dir / f"{source}_n{size}_seed{seed}.csv"
                sub.to_csv(out_path, index=False)

                row = {
                    "train_source": source,
                    "subset_n": size,
                    "seed": seed,
                    "rows": len(sub),
                    "patients": sub["patient_id"].nunique(),
                    "path": str(out_path),
                }
                for lab in LABELS:
                    row[f"{lab}_available"] = int(sub[lab].notna().sum())
                    row[f"{lab}_positive"] = int(sub[lab].sum(skipna=True))
                summary.append(row)

                print(f"Saved {out_path} rows={len(sub):,} patients={row['patients']:,}")

    summary_df = pd.DataFrame(summary)
    summary_path = out_dir / "subset_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\nSaved summary:", summary_path)
    print(summary_df)

if __name__ == "__main__":
    main()
