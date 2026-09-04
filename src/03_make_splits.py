#!/usr/bin/env python
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def group_random_split(df, group_col, seed=2026, train_frac=0.70, val_frac=0.10):
    rng = np.random.default_rng(seed)

    groups = df[group_col].fillna(df["image_id"]).astype(str).to_numpy()
    groups = pd.Series(groups).drop_duplicates().to_numpy(dtype=str)
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train:n_train + n_val])
    test_groups = set(groups[n_train + n_val:])

    def assign(g):
        g = str(g)
        if g in train_groups:
            return "train"
        if g in val_groups:
            return "val"
        if g in test_groups:
            return "internal_test"
        return "unknown"

    return df[group_col].fillna(df["image_id"]).astype(str).map(assign)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--external-datasets", nargs="+", default=["vindr"])
    parser.add_argument("--frontal-only", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)

    if args.frontal_only and "is_frontal" in df.columns:
        before = len(df)
        df = df[df["is_frontal"].fillna(False)].copy()
        print(f"Frontal-only filter: {before:,} -> {len(df):,}")

    df["split"] = "unassigned"

    # Unlabeled rows.
    if "is_labeled" not in df.columns:
        df["is_labeled"] = True

    df.loc[~df["is_labeled"].fillna(False), "split"] = "unlabeled"

    # External datasets: keep labeled rows as external_test.
    for ds in args.external_datasets:
        mask = (df["dataset"] == ds) & df["is_labeled"].fillna(False)
        df.loc[mask, "split"] = "external_test"

    # Other datasets: patient-level random split.
    for ds in sorted(df["dataset"].dropna().unique()):
        if ds in args.external_datasets:
            continue
        mask = (df["dataset"] == ds) & df["is_labeled"].fillna(False)
        sub = df[mask].copy()
        if sub.empty:
            continue

        group_col = "patient_id" if "patient_id" in sub.columns else "image_id"
        sub_split = group_random_split(
            sub,
            group_col=group_col,
            seed=args.seed,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
        )
        df.loc[mask, "split"] = sub_split.values

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Saved: {out}")
    print(f"Rows: {len(df):,}")
    print(pd.crosstab(df["dataset"], df["split"], dropna=False))

if __name__ == "__main__":
    main()
