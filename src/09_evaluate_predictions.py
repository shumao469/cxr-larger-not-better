#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


DEFAULT_LABELS = [
    "y_Cardiomegaly",
    "y_Pleural_Effusion",
    "y_Atelectasis",
    "y_Consolidation",
]


def ece_score(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)

    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    if len(y_true) == 0:
        return np.nan

    y_prob = np.clip(y_prob, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            idx = (y_prob >= lo) & (y_prob <= hi)
        else:
            idx = (y_prob >= lo) & (y_prob < hi)

        if idx.sum() == 0:
            continue

        conf = y_prob[idx].mean()
        acc = y_true[idx].mean()
        ece += (idx.sum() / len(y_true)) * abs(acc - conf)

    return float(ece)


def strip_y(label):
    return label[2:] if label.startswith("y_") else label


def find_prob_col(df, label):
    base = strip_y(label)
    candidates = [
        f"p_{base}",
        f"prob_{base}",
        f"pred_{base}",
        f"prediction_{base}",
        f"{base}_prob",
        f"{base}_pred",
        f"{label}_prob",
        f"{label}_pred",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    # Fallback: contains disease name and starts with p/prob/pred
    pattern = base.lower()
    for c in df.columns:
        cl = c.lower()
        if pattern in cl and (
            cl.startswith("p_")
            or cl.startswith("prob")
            or cl.startswith("pred")
            or cl.endswith("_prob")
            or cl.endswith("_pred")
        ):
            return c

    raise KeyError(f"Cannot find probability column for {label}. Available columns: {list(df.columns)[:30]}...")


def infer_meta_from_path(pred_path):
    name = Path(pred_path).parent.name
    # expected: nih_n1000_seed1 or chexpert_nall_seed1
    m = re.match(r"(?P<src>.+?)_n(?P<n>.+?)_seed(?P<seed>\d+)", name)
    if m:
        return m.group("src"), m.group("n"), int(m.group("seed"))
    return None, None, None


def compute_one(y, p, n_bins=15):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = np.clip(p[mask], 1e-7, 1 - 1e-7)

    n = len(y)
    if n == 0:
        return {
            "n": 0,
            "positive": np.nan,
            "prevalence": np.nan,
            "AUROC": np.nan,
            "AUPRC": np.nan,
            "AUPRC_lift": np.nan,
            "normalized_AUPRC": np.nan,
            "Brier": np.nan,
            "ECE": np.nan,
        }

    positive = int(y.sum())
    prevalence = float(y.mean())

    if len(np.unique(y)) < 2:
        auroc = np.nan
    else:
        auroc = float(roc_auc_score(y, p))

    auprc = float(average_precision_score(y, p))

    if prevalence > 0:
        auprc_lift = auprc / prevalence
    else:
        auprc_lift = np.nan

    if prevalence < 1:
        normalized_auprc = (auprc - prevalence) / (1 - prevalence)
    else:
        normalized_auprc = np.nan

    brier = float(brier_score_loss(y, p))
    ece = ece_score(y, p, n_bins=n_bins)

    return {
        "n": n,
        "positive": positive,
        "prevalence": prevalence,
        "AUROC": auroc,
        "AUPRC": auprc,
        "AUPRC_lift": auprc_lift,
        "normalized_AUPRC": normalized_auprc,
        "Brier": brier,
        "ECE": ece,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()

    pred_path = Path(args.pred)
    df = pd.read_csv(pred_path, low_memory=False)

    train_source, subset_n, seed = infer_meta_from_path(pred_path)

    if "train_source" in df.columns:
        train_source = df["train_source"].dropna().iloc[0] if df["train_source"].notna().any() else train_source
    if "subset_n" in df.columns:
        subset_n = df["subset_n"].dropna().iloc[0] if df["subset_n"].notna().any() else subset_n
    if "seed" in df.columns:
        seed = df["seed"].dropna().iloc[0] if df["seed"].notna().any() else seed

    group_cols = []
    for c in ["eval_dataset", "dataset"]:
        if c in df.columns:
            group_cols.append(c)
            break
    for c in ["eval_split", "split"]:
        if c in df.columns:
            group_cols.append(c)
            break

    if not group_cols:
        df["_all"] = "all"
        group_cols = ["_all"]

    rows = []

    for group_values, g in df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        meta = dict(zip(group_cols, group_values))

        eval_dataset = meta.get("eval_dataset", meta.get("dataset", "all"))
        eval_split = meta.get("eval_split", meta.get("split", "all"))

        per_label_metrics = []

        for label in args.labels:
            if label not in g.columns:
                continue

            pcol = find_prob_col(g, label)
            m = compute_one(g[label].values, g[pcol].values, n_bins=args.n_bins)

            row = {
                "train_source": train_source,
                "subset_n": subset_n,
                "seed": seed,
                "eval_dataset": eval_dataset,
                "eval_split": eval_split,
                "label": label,
                "prob_col": pcol,
                "method": "raw",
            }
            row.update(m)
            rows.append(row)
            per_label_metrics.append(row)

        if per_label_metrics:
            mean_row = {
                "train_source": train_source,
                "subset_n": subset_n,
                "seed": seed,
                "eval_dataset": eval_dataset,
                "eval_split": eval_split,
                "label": "mean_primary_labels",
                "prob_col": "",
                "method": "raw",
            }

            for metric in [
                "AUROC",
                "AUPRC",
                "AUPRC_lift",
                "normalized_AUPRC",
                "Brier",
                "ECE",
                "prevalence",
            ]:
                vals = [r[metric] for r in per_label_metrics if np.isfinite(r[metric])]
                mean_row[metric] = float(np.mean(vals)) if vals else np.nan

            mean_row["n"] = int(np.nanmean([r["n"] for r in per_label_metrics]))
            mean_row["positive"] = float(np.nanmean([r["positive"] for r in per_label_metrics]))
            rows.append(mean_row)

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print("Saved:", args.out)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
