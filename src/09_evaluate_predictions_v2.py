#!/usr/bin/env python3
"""Evaluate recovered or newly generated predictions with strict column checks."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


DEFAULT_LABELS = [
    "y_Cardiomegaly",
    "y_Pleural_Effusion",
    "y_Atelectasis",
    "y_Consolidation",
]


def ece_score(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[mask], np.clip(y_prob[mask], 0.0, 1.0)
    if not len(y_true):
        return np.nan
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        lo, hi = bins[index], bins[index + 1]
        selected = (y_prob >= lo) & (y_prob <= hi if index == n_bins - 1 else y_prob < hi)
        if selected.any():
            ece += selected.mean() * abs(y_true[selected].mean() - y_prob[selected].mean())
    return float(ece)


def strip_y(label):
    return label[2:] if label.startswith("y_") else label


def find_true_col(df, label):
    base = strip_y(label)
    candidates = [label, f"true_{label}", f"true_{base}", f"target_{label}", f"target_{base}"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Cannot find truth column for {label}")


def find_prob_col(df, label):
    base = strip_y(label)
    candidates = [
        f"p_{base}", f"prob_{base}", f"pred_{base}", f"prediction_{base}",
        f"{base}_prob", f"{base}_pred", f"{label}_prob", f"{label}_pred",
        f"p_{label}", f"prob_{label}", f"pred_{label}", f"prediction_{label}",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    disease = base.lower()
    for column in df.columns:
        lower = column.lower()
        if disease in lower and (lower.startswith(("p_", "prob", "pred")) or lower.endswith(("_prob", "_pred"))):
            return column
    raise KeyError(f"Cannot find probability column for {label}")


def infer_meta(path):
    match = re.match(r"(?P<src>.+?)_n(?P<n>.+?)_seed(?P<seed>\d+)", Path(path).parent.name)
    return (match.group("src"), match.group("n"), int(match.group("seed"))) if match else (None, None, None)


def compute_one(y, p, n_bins):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], np.clip(p[mask], 1e-7, 1 - 1e-7)
    if not len(y):
        raise ValueError("No finite labels/probabilities")
    prevalence = float(y.mean())
    auprc = float(average_precision_score(y, p))
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "prevalence": prevalence,
        "AUROC": np.nan if len(np.unique(y)) < 2 else float(roc_auc_score(y, p)),
        "AUPRC": auprc,
        "AUPRC_lift": auprc / prevalence if prevalence > 0 else np.nan,
        "normalized_AUPRC": (auprc - prevalence) / (1 - prevalence) if prevalence < 1 else np.nan,
        "Brier": float(brier_score_loss(y, p)),
        "ECE": ece_score(y, p, n_bins),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--expected-prediction-rows", type=int, default=65283)
    args = parser.parse_args()

    pred_path = Path(args.pred)
    df = pd.read_csv(pred_path, low_memory=False)
    if len(df) != args.expected_prediction_rows:
        raise RuntimeError(f"Prediction row mismatch: {len(df):,} != {args.expected_prediction_rows:,}")
    train_source, subset_n, seed = infer_meta(pred_path)
    for column, current in [("train_source", train_source), ("subset_n", subset_n), ("seed", seed)]:
        if column in df.columns and df[column].notna().any():
            value = df.loc[df[column].notna(), column].iloc[0]
            if column == "train_source": train_source = value
            elif column == "subset_n": subset_n = value
            else: seed = value

    dataset_col = "eval_dataset" if "eval_dataset" in df.columns else "dataset"
    split_col = "eval_split" if "eval_split" in df.columns else "split"
    rows = []
    for (eval_dataset, eval_split), group in df.groupby([dataset_col, split_col], dropna=False):
        label_rows = []
        for label in args.labels:
            truth_col = find_true_col(group, label)
            probability_col = find_prob_col(group, label)
            metrics = compute_one(group[truth_col], group[probability_col], args.n_bins)
            row = {
                "train_source": train_source, "subset_n": subset_n, "seed": seed,
                "eval_dataset": eval_dataset, "eval_split": eval_split,
                "label": label, "truth_col": truth_col, "prob_col": probability_col, "method": "raw",
                **metrics,
            }
            rows.append(row)
            label_rows.append(row)
        mean_row = {
            "train_source": train_source, "subset_n": subset_n, "seed": seed,
            "eval_dataset": eval_dataset, "eval_split": eval_split,
            "label": "mean_primary_labels", "truth_col": "", "prob_col": "", "method": "raw",
        }
        for metric in ["AUROC", "AUPRC", "AUPRC_lift", "normalized_AUPRC", "Brier", "ECE", "prevalence"]:
            values = [row[metric] for row in label_rows if np.isfinite(row[metric])]
            mean_row[metric] = float(np.mean(values)) if values else np.nan
        mean_row["n"] = int(np.mean([row["n"] for row in label_rows]))
        mean_row["positive"] = float(np.mean([row["positive"] for row in label_rows]))
        rows.append(mean_row)

    output = pd.DataFrame(rows)
    expected_rows = 3 * (len(args.labels) + 1)
    if len(output) != expected_rows:
        raise RuntimeError(f"Metric row mismatch: {len(output)} != {expected_rows}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp = out_path.with_suffix(".csv.tmp")
    output.to_csv(temp, index=False)
    os.replace(temp, out_path)
    print("Saved:", out_path)
    print(output[output["label"] == "mean_primary_labels"].to_string(index=False))


if __name__ == "__main__":
    main()
