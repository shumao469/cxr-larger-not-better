from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

metric_path = Path("results/metrics/stage1_mean_primary_metrics.csv")
out_dir = Path("results/figures")
table_dir = Path("results/tables")
out_dir.mkdir(parents=True, exist_ok=True)
table_dir.mkdir(parents=True, exist_ok=True)

if not metric_path.exists():
    raise FileNotFoundError(f"Missing metrics file: {metric_path}")

df = pd.read_csv(metric_path)

order_map = {
    "1000": 1000,
    "5000": 5000,
    "10000": 10000,
    "50000": 50000,
    "all": 100000,
}

df["subset_n_str"] = df["subset_n"].astype(str)
df["subset_order"] = df["subset_n_str"].map(order_map)

if df["subset_order"].isna().any():
    print("Warning: some subset_n values were not mapped:")
    print(df.loc[df["subset_order"].isna(), "subset_n"].drop_duplicates().to_string(index=False))

# Clean mean-primary table
cols = [
    "train_source",
    "subset_n_str",
    "seed",
    "eval_dataset",
    "eval_split",
    "AUROC",
    "AUPRC",
    "Brier",
    "ECE",
    "n",
]

clean = (
    df.sort_values(["train_source", "eval_dataset", "subset_order", "seed"])
      [cols]
      .copy()
)

clean.to_csv(table_dir / "stage1_mean_primary_clean.csv", index=False)

# Adjacent sample-size gains
gain_rows = []
for (train_source, eval_dataset), g in df.groupby(["train_source", "eval_dataset"]):
    g = g.sort_values(["subset_order", "seed"]).copy()
    prev = None

    for _, r in g.iterrows():
        if prev is not None:
            gain_rows.append({
                "train_source": train_source,
                "eval_dataset": eval_dataset,
                "from_n": prev["subset_n_str"],
                "to_n": r["subset_n_str"],
                "delta_AUROC": r["AUROC"] - prev["AUROC"],
                "delta_AUPRC": r["AUPRC"] - prev["AUPRC"],
                "delta_Brier": r["Brier"] - prev["Brier"],
                "delta_ECE": r["ECE"] - prev["ECE"],
            })
        prev = r

gain = pd.DataFrame(gain_rows)
gain.to_csv(table_dir / "stage1_adjacent_gains.csv", index=False)

# Internal-external gaps
gap_rows = []

for train_source in sorted(df["train_source"].dropna().unique()):
    g = df[df["train_source"] == train_source].copy()

    for n in sorted(g["subset_n_str"].dropna().unique(), key=lambda x: order_map.get(str(x), 999999)):
        sub = g[g["subset_n_str"] == n]
        internal = sub[sub["eval_dataset"] == train_source]

        if internal.empty:
            continue

        internal_auc = float(internal["AUROC"].iloc[0])
        internal_auprc = float(internal["AUPRC"].iloc[0])
        internal_brier = float(internal["Brier"].iloc[0])
        internal_ece = float(internal["ECE"].iloc[0])

        for _, r in sub[sub["eval_dataset"] != train_source].iterrows():
            gap_rows.append({
                "train_source": train_source,
                "subset_n": n,
                "external_dataset": r["eval_dataset"],
                "internal_AUROC": internal_auc,
                "external_AUROC": r["AUROC"],
                "AUROC_gap": internal_auc - r["AUROC"],
                "internal_AUPRC": internal_auprc,
                "external_AUPRC": r["AUPRC"],
                "AUPRC_gap": internal_auprc - r["AUPRC"],
                "internal_Brier": internal_brier,
                "external_Brier": r["Brier"],
                "Brier_gap": r["Brier"] - internal_brier,
                "internal_ECE": internal_ece,
                "external_ECE": r["ECE"],
                "ECE_gap": r["ECE"] - internal_ece,
            })

gap = pd.DataFrame(gap_rows)
gap.to_csv(table_dir / "stage1_internal_external_gap.csv", index=False)

# Learning-curve plots
metrics = ["AUROC", "AUPRC", "Brier", "ECE"]

for train_source in sorted(df["train_source"].dropna().unique()):
    sub = df[df["train_source"] == train_source].copy()

    for metric in metrics:
        plt.figure(figsize=(7, 5))

        for eval_dataset in sorted(sub["eval_dataset"].dropna().unique()):
            g = sub[sub["eval_dataset"] == eval_dataset].sort_values("subset_order")
            plt.plot(
                g["subset_n_str"],
                g[metric],
                marker="o",
                label=eval_dataset,
            )

        plt.xlabel("Training sample size")
        plt.ylabel(metric)
        plt.title(f"{metric}: trained on {train_source}")
        plt.legend()
        plt.tight_layout()

        out = out_dir / f"stage1_curve_{train_source}_{metric}.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print("Saved", out)

# Generalization-gap plots
if not gap.empty:
    for train_source in sorted(gap["train_source"].dropna().unique()):
        sub = gap[gap["train_source"] == train_source].copy()

        plt.figure(figsize=(7, 5))

        for external_dataset in sorted(sub["external_dataset"].dropna().unique()):
            g = sub[sub["external_dataset"] == external_dataset].copy()
            g["subset_order"] = g["subset_n"].astype(str).map(order_map)
            g = g.sort_values("subset_order")

            plt.plot(
                g["subset_n"],
                g["AUROC_gap"],
                marker="o",
                label=external_dataset,
            )

        plt.axhline(0, linestyle="--", linewidth=1)
        plt.xlabel("Training sample size")
        plt.ylabel("Internal AUROC - External AUROC")
        plt.title(f"Generalization gap: trained on {train_source}")
        plt.legend()
        plt.tight_layout()

        out = out_dir / f"stage1_generalization_gap_{train_source}.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print("Saved", out)

print("\nSaved tables:")
print(table_dir / "stage1_mean_primary_clean.csv")
print(table_dir / "stage1_adjacent_gains.csv")
print(table_dir / "stage1_internal_external_gap.csv")

print("\nClean metrics:")
print(clean.to_string(index=False))

print("\nAdjacent gains:")
print(gain.to_string(index=False))

print("\nInternal-external gap:")
print(gap.to_string(index=False))
