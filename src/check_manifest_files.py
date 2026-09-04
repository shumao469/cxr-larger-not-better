from pathlib import Path
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest, low_memory=False)
    print("Rows:", len(df))
    print(pd.crosstab(df["dataset"], df["split"], dropna=False))

    path_col = "image_path_final"
    if path_col not in df.columns:
        raise ValueError(f"Missing column: {path_col}")

    exists = df[path_col].fillna("").map(lambda x: Path(x).exists() if x else False)
    df["file_exists"] = exists

    print("\nFile existence by dataset:")
    print(pd.crosstab(df["dataset"], df["file_exists"], dropna=False))

    print("\nFile existence by dataset and split:")
    print(pd.crosstab([df["dataset"], df["split"]], df["file_exists"], dropna=False))

    missing = df[~df["file_exists"]]
    print("\nMissing files:", len(missing))
    if len(missing):
        print(missing[["dataset", "split", "image_id", path_col]].head(args.sample).to_string(index=False))

    label_cols = [c for c in df.columns if c.startswith("y_")]
    print("\nLabel summary:")
    for c in label_cols:
        print(c, "available:", int(df[c].notna().sum()), "positive:", int(df[c].sum(skipna=True)))

if __name__ == "__main__":
    main()
