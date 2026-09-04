from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-missing", default="logs/missing_files.csv")
    args = parser.parse_args()

    print("Reading manifest...", flush=True)
    df = pd.read_csv(args.manifest, low_memory=False)
    print("Rows:", len(df), flush=True)

    print("\nDataset counts:")
    print(df["dataset"].value_counts(dropna=False), flush=True)

    print("\nSplit table:")
    print(pd.crosstab(df["dataset"], df["split"], dropna=False), flush=True)

    path_col = "image_path_final"
    if path_col not in df.columns:
        raise ValueError(f"Missing column: {path_col}")

    print("\nChecking file existence with progress bar...", flush=True)
    paths = df[path_col].fillna("").astype(str).tolist()

    exists = []
    for p in tqdm(paths, total=len(paths), desc="exists"):
        exists.append(Path(p).exists() if p else False)

    df["file_exists"] = exists

    print("\nFile existence by dataset:")
    print(pd.crosstab(df["dataset"], df["file_exists"], dropna=False), flush=True)

    print("\nFile existence by dataset and split:")
    print(pd.crosstab([df["dataset"], df["split"]], df["file_exists"], dropna=False), flush=True)

    missing = df[~df["file_exists"]].copy()
    print("\nMissing files:", len(missing), flush=True)

    Path(args.out_missing).parent.mkdir(parents=True, exist_ok=True)
    missing[["dataset", "split", "image_id", path_col]].to_csv(args.out_missing, index=False)
    print("Saved missing list:", args.out_missing, flush=True)

    label_cols = [c for c in df.columns if c.startswith("y_")]
    print("\nLabel summary:")
    for c in label_cols:
        print(c, "available:", int(df[c].notna().sum()), "positive:", int(df[c].sum(skipna=True)), flush=True)

if __name__ == "__main__":
    main()
