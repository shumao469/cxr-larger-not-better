from pathlib import Path
import pandas as pd

paths = {
    "chexpert": Path("/mnt/h/data/raw/chexpert/chexpertchestxrays-u20210408"),
    "nih": Path("/mnt/h/data/raw/nih"),
    "vindr": Path("/mnt/h/kagglehub_cache/competitions/vinbigdata-chest-xray-abnormalities-detection"),
}

for name, root in paths.items():
    print("\n====", name, "====")
    print("root:", root)
    print("exists:", root.exists())
    if not root.exists():
        continue

    csvs = list(root.rglob("*.csv"))
    imgs = list(root.rglob("*.jpg")) + list(root.rglob("*.jpeg")) + list(root.rglob("*.png"))
    dicoms = list(root.rglob("*.dicom")) + list(root.rglob("*.dcm"))

    print("csv count:", len(csvs))
    for x in csvs[:20]:
        print("  CSV:", x)

    print("image count jpg/jpeg/png:", len(imgs))
    print("dicom count:", len(dicoms))

    for csv in csvs[:8]:
        try:
            df = pd.read_csv(csv, nrows=3)
            print("\nPreview:", csv)
            print(df.head())
            print("Columns:", list(df.columns)[:40])
        except Exception as e:
            print("Could not read", csv, e)
