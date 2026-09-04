from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

inp = Path("results/metrics/stage1_mean_primary_metrics.csv")
out_dir = Path("results/figures")
out_dir.mkdir(parents=True, exist_ok=True)

if not inp.exists():
    raise FileNotFoundError(inp)

df = pd.read_csv(inp)

# Order x-axis
order = {"1000": 1000, "5000": 5000, "10000": 10000, "50000": 50000, "all": 1000000}
df["subset_order"] = df["subset_n"].astype(str).map(order)

metrics = ["AUROC", "AUPRC", "Brier", "ECE"]

for train_source in sorted(df["train_source"].dropna().unique()):
    sub = df[df["train_source"] == train_source].copy()

    for metric in metrics:
        plt.figure(figsize=(7, 5))

        for eval_dataset in sorted(sub["eval_dataset"].dropna().unique()):
            g = sub[sub["eval_dataset"] == eval_dataset].copy()
            if g.empty:
                continue

            agg = (
                g.groupby(["subset_n", "subset_order"], as_index=False)[metric]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("subset_order")
            )

            x = agg["subset_n"].astype(str)
            y = agg["mean"]
            yerr = agg["std"].fillna(0)

            plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=eval_dataset)

        plt.xlabel("Training sample size")
        plt.ylabel(metric)
        plt.title(f"{metric} learning curve, trained on {train_source}")
        plt.legend()
        plt.tight_layout()

        out = out_dir / f"learning_curve_{train_source}_{metric}.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print("Saved", out)
