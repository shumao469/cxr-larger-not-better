from pathlib import Path
import pandas as pd

metric_dir = Path("results/metrics")
files = sorted(metric_dir.glob("*_metrics.csv"))

dfs = []
for f in files:
    if f.name in ["all_stage1_metrics.csv", "stage1_mean_primary_metrics.csv"]:
        continue
    try:
        df = pd.read_csv(f)
        df["metric_file"] = str(f)
        dfs.append(df)
    except Exception as e:
        print("Skip", f, e)

if not dfs:
    print("No metric files found.")
    raise SystemExit(0)

out = pd.concat(dfs, ignore_index=True)
out.to_csv("results/metrics/all_stage1_metrics.csv", index=False)

mean = out[out["label"] == "mean_primary_labels"].copy()
mean.to_csv("results/metrics/stage1_mean_primary_metrics.csv", index=False)

print("Saved:")
print("results/metrics/all_stage1_metrics.csv")
print("results/metrics/stage1_mean_primary_metrics.csv")

cols = [
    "train_source", "subset_n", "seed",
    "eval_dataset", "eval_split",
    "AUROC", "AUPRC", "Brier", "ECE", "n"
]
if len(mean):
    print(mean[cols].sort_values(["train_source", "eval_dataset", "subset_n", "seed"]).to_string(index=False))
