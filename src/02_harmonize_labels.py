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

DISEASE_TARGETS = [
    "Cardiomegaly",
    "Edema",
    "Pleural_Effusion",
    "Atelectasis",
    "Consolidation",
]

def normalize_label(x, uncertain="missing"):
    if pd.isna(x):
        return np.nan
    try:
        val = float(x)
    except Exception:
        return np.nan

    if val == 1:
        return 1
    if val == 0:
        return 0
    if val == -1:
        if uncertain == "zero":
            return 0
        if uncertain == "one":
            return 1
        return np.nan
    return np.nan

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", default="/mnt/h/cxr_larger_not_better/data/processed/harmonized_manifest.csv")
    parser.add_argument("--uncertain", choices=["missing", "zero", "one"], default="missing")
    args = parser.parse_args()

    frames = []
    for inp in args.inputs:
        path = Path(inp)
        if not path.exists():
            print(f"Warning: missing input, skip: {path}")
            continue
        df = pd.read_csv(path)
        print(f"Loaded {path}: {len(df):,}")
        frames.append(df)

    if not frames:
        raise RuntimeError("No input manifests loaded.")

    df = pd.concat(frames, ignore_index=True)

    required_base = [
        "dataset", "patient_id", "study_id", "image_id",
        "path", "png_path", "dicom_path", "view", "split_original"
    ]
    for col in required_base:
        if col not in df.columns:
            df[col] = ""

    for target in TARGETS:
        raw_col = f"label_{target}"
        y_col = f"y_{target}"
        if raw_col not in df.columns:
            df[y_col] = np.nan
        else:
            df[y_col] = df[raw_col].apply(lambda x: normalize_label(x, uncertain=args.uncertain))

    # If No_Finding is missing but all disease labels are explicitly 0, set No_Finding=1.
    disease_cols = [f"y_{x}" for x in DISEASE_TARGETS]
    all_disease_known = df[disease_cols].notna().all(axis=1)
    all_disease_zero = (df[disease_cols].fillna(-999) == 0).all(axis=1)
    no_col = "y_No_Finding"
    df.loc[df[no_col].isna() & all_disease_known & all_disease_zero, no_col] = 1

    df["any_target_positive"] = df[disease_cols].max(axis=1, skipna=True)
    df["n_available_targets"] = df[[f"y_{x}" for x in TARGETS]].notna().sum(axis=1)
    df["is_labeled"] = df["n_available_targets"] > 0

    # Prefer PNG path when available.
    df["image_path_final"] = df["png_path"].fillna("").astype(str)
    mask_empty_png = df["image_path_final"].eq("") | df["image_path_final"].eq("nan")
    df.loc[mask_empty_png, "image_path_final"] = df.loc[mask_empty_png, "path"].fillna("").astype(str)

    # Frontal flag.
    view_lower = df["view"].fillna("").astype(str).str.lower()
    path_lower = df["image_path_final"].fillna("").astype(str).str.lower()
    df["is_frontal"] = (
        view_lower.str.contains("frontal|pa|ap", regex=True)
        | path_lower.str.contains("frontal|pa|ap", regex=True)
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Saved: {out}")
    print(f"Rows: {len(df):,}")
    print(df["dataset"].value_counts(dropna=False))
    print("Label availability:")
    for target in TARGETS:
        col = f"y_{target}"
        print(target, "available:", int(df[col].notna().sum()), "positive:", int(df[col].sum(skipna=True)))

if __name__ == "__main__":
    main()
