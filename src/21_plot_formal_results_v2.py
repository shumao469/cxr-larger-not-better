#!/usr/bin/env python3
"""Create audited formal CXR figures from the completed three-seed metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = {"1000": 0, "5000": 1, "10000": 2, "50000": 3, "all": 4}
DISPLAY_DATASET = {"nih": "NIH ChestX-ray14", "chexpert": "CheXpert", "vindr": "VinDr-CXR"}
DISPLAY_METRIC = {"AUROC": "AUROC", "AUPRC": "AUPRC", "Brier": "Brier score", "ECE": "Expected calibration error"}
COLORS = {"nih": "#0072B2", "chexpert": "#D55E00", "vindr": "#009E73"}
METRICS = ["AUROC", "AUPRC", "Brier", "ECE"]


def atomic_savefig(fig: plt.Figure, target: Path) -> None:
    temp = target.with_name(target.stem + ".tmp" + target.suffix)
    fig.savefig(temp, dpi=300 if target.suffix == ".png" else None,
                bbox_inches="tight", facecolor="white")
    os.replace(temp, target)


def save_triplet(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    paths = []
    for suffix in (".png", ".pdf", ".svg"):
        target = output_dir / f"{stem}{suffix}"
        atomic_savefig(fig, target)
        paths.append(target)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    metric_path = Path(args.metrics)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metric_path, low_memory=False)
    if len(df) != 90 or set(df["label"]) != {"mean_primary_labels"}:
        raise RuntimeError("Expected exactly 90 mean-primary metric rows")
    df["subset_n"] = df["subset_n"].astype(str)
    df["subset_order"] = df["subset_n"].map(ORDER)
    if df["subset_order"].isna().any():
        raise RuntimeError("Unexpected subset_n values")
    expected_sources = {"nih", "chexpert"}
    expected_eval = {"nih", "chexpert", "vindr"}
    if set(df["train_source"]) != expected_sources or set(df["eval_dataset"]) != expected_eval:
        raise RuntimeError("Unexpected train/evaluation dataset identities")
    counts = df.groupby(["train_source", "subset_n", "eval_dataset"])["seed"].nunique()
    if len(counts) != 30 or not (counts == 3).all():
        raise RuntimeError("Every train-size/evaluation condition must contain seeds 1, 2, and 3")

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    agg = (df.groupby(["train_source", "subset_n", "subset_order", "eval_dataset"], as_index=False)
             [METRICS].agg(["mean", "std"]))
    agg.columns = ["_".join(part for part in col if part) for col in agg.columns]
    agg = agg.sort_values(["train_source", "eval_dataset", "subset_order"])
    agg.to_csv(out_dir / "formal_metric_mean_sd.csv", index=False)

    artifacts: list[Path] = [out_dir / "formal_metric_mean_sd.csv"]
    x = np.arange(5)
    xlabels = ["1k", "5k", "10k", "50k", "All"]
    for train_source in sorted(expected_sources):
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
            for eval_dataset in ("nih", "chexpert", "vindr"):
                group = agg[(agg["train_source"] == train_source) &
                            (agg["eval_dataset"] == eval_dataset)].sort_values("subset_order")
                if len(group) != 5:
                    raise RuntimeError(f"Incomplete learning curve: {train_source}/{eval_dataset}/{metric}")
                mean = group[f"{metric}_mean"].to_numpy(float)
                std = group[f"{metric}_std"].to_numpy(float)
                ax.plot(x, mean, marker="o", linewidth=1.8, markersize=4.5,
                        color=COLORS[eval_dataset], label=DISPLAY_DATASET[eval_dataset])
                ax.fill_between(x, mean - std, mean + std, color=COLORS[eval_dataset], alpha=0.14)
            ax.set_xticks(x, xlabels)
            ax.set_xlabel("Training sample size")
            ax.set_ylabel(DISPLAY_METRIC[metric])
            ax.set_title(f"Training source: {DISPLAY_DATASET[train_source]}")
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
            ax.legend(frameon=False, ncol=3, loc="best")
            artifacts.extend(save_triplet(fig, out_dir, f"learning_curve_{train_source}_{metric.lower()}"))
            plt.close(fig)

    gap_rows = []
    for train_source in sorted(expected_sources):
        internal = df[
            (df["train_source"] == train_source)
            & (df["eval_dataset"] == train_source)
        ]
        for external_dataset in sorted(expected_eval - {train_source}):
            external = df[
                (df["train_source"] == train_source)
                & (df["eval_dataset"] == external_dataset)
            ]
            paired = internal.merge(
                external,
                on=["train_source", "subset_n", "subset_order", "seed"],
                suffixes=("_internal", "_external"),
                validate="one_to_one",
            )
            for _, row in paired.iterrows():
                gap_rows.append({
                    "train_source": train_source,
                    "subset_n": row["subset_n"],
                    "subset_order": row["subset_order"],
                    "seed": row["seed"],
                    "external_dataset": external_dataset,
                    "AUROC_gap_internal_minus_external": row["AUROC_internal"] - row["AUROC_external"],
                    "AUPRC_gap_internal_minus_external": row["AUPRC_internal"] - row["AUPRC_external"],
                    "Brier_gap_external_minus_internal": row["Brier_external"] - row["Brier_internal"],
                    "ECE_gap_external_minus_internal": row["ECE_external"] - row["ECE_internal"],
                })
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(out_dir / "formal_paired_generalization_gaps.csv", index=False)
    artifacts.append(out_dir / "formal_paired_generalization_gaps.csv")
    gap_agg = (gaps.groupby(["train_source", "subset_n", "subset_order", "external_dataset"], as_index=False)
                    .agg(AUROC_gap_mean=("AUROC_gap_internal_minus_external", "mean"),
                         AUROC_gap_std=("AUROC_gap_internal_minus_external", "std")))
    gap_agg.to_csv(out_dir / "formal_paired_generalization_gap_mean_sd.csv", index=False)
    artifacts.append(out_dir / "formal_paired_generalization_gap_mean_sd.csv")

    for train_source in sorted(expected_sources):
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        for external_dataset in sorted(expected_eval - {train_source}):
            group = gap_agg[(gap_agg["train_source"] == train_source) &
                            (gap_agg["external_dataset"] == external_dataset)].sort_values("subset_order")
            mean = group["AUROC_gap_mean"].to_numpy(float)
            std = group["AUROC_gap_std"].to_numpy(float)
            ax.plot(x, mean, marker="o", linewidth=1.8, markersize=4.5,
                    color=COLORS[external_dataset], label=DISPLAY_DATASET[external_dataset])
            ax.fill_between(x, mean - std, mean + std, color=COLORS[external_dataset], alpha=0.14)
        ax.axhline(0, color="#555555", linestyle="--", linewidth=1)
        ax.set_xticks(x, xlabels)
        ax.set_xlabel("Training sample size")
        ax.set_ylabel("Internal AUROC − external AUROC")
        ax.set_title(f"Generalization gap: trained on {DISPLAY_DATASET[train_source]}")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        ax.legend(frameon=False)
        artifacts.extend(save_triplet(fig, out_dir, f"generalization_gap_{train_source}_auroc"))
        plt.close(fig)

    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}
    report = {
        "schema_version": 1,
        "status": "PASS",
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "input": str(metric_path),
        "input_sha256": hashlib.sha256(metric_path.read_bytes()).hexdigest(),
        "source_rows": len(df),
        "seed_count_per_condition": 3,
        "learning_curve_figures": 8,
        "generalization_gap_figures": 2,
        "format_triplets": ["png", "pdf", "svg"],
        "artifact_count_excluding_report": len(artifacts),
        "artifacts_sha256": hashes,
    }
    report_path = out_dir / "figures_completed.json"
    temp = report_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
