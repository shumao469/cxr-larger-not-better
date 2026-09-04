#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


LABELS = [
    "y_Cardiomegaly",
    "y_Pleural_Effusion",
    "y_Atelectasis",
    "y_Consolidation",
]


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


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

        ece += (idx.sum() / len(y_true)) * abs(y_true[idx].mean() - y_prob[idx].mean())

    return float(ece)


def compute_metrics(y, p, n_bins=15):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = np.clip(p[mask], 1e-7, 1 - 1e-7)

    n = len(y)
    if n == 0:
        return dict(
            n=0,
            positive=np.nan,
            prevalence=np.nan,
            AUROC=np.nan,
            AUPRC=np.nan,
            AUPRC_lift=np.nan,
            normalized_AUPRC=np.nan,
            Brier=np.nan,
            ECE=np.nan,
        )

    positive = int(y.sum())
    prevalence = float(y.mean())

    auroc = np.nan if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))
    auprc = float(average_precision_score(y, p))
    auprc_lift = auprc / prevalence if prevalence > 0 else np.nan
    normalized_auprc = (auprc - prevalence) / (1 - prevalence) if prevalence < 1 else np.nan
    brier = float(brier_score_loss(y, p))
    ece = ece_score(y, p, n_bins=n_bins)

    return dict(
        n=n,
        positive=positive,
        prevalence=prevalence,
        AUROC=auroc,
        AUPRC=auprc,
        AUPRC_lift=auprc_lift,
        normalized_AUPRC=normalized_auprc,
        Brier=brier,
        ECE=ece,
    )


def parse_model_name(model_dir):
    name = Path(model_dir).name
    m = re.match(r"(?P<src>.+?)_n(?P<n>.+?)_seed(?P<seed>\d+)", name)
    if not m:
        return None, None, None
    return m.group("src"), m.group("n"), int(m.group("seed"))


def get_path_col(df):
    for c in ["path", "image_path", "filepath", "file_path", "png_path"]:
        if c in df.columns:
            return c
    raise KeyError("Cannot find image path column. Expected one of: path, image_path, filepath, file_path, png_path")


class CXRDataset(Dataset):
    def __init__(self, df, labels, img_size):
        self.df = df.reset_index(drop=True)
        self.labels = labels
        self.path_col = get_path_col(df)
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        path = r[self.path_col]

        img = Image.open(path).convert("RGB")
        x = self.tf(img)

        y = []
        for lab in self.labels:
            v = r[lab]
            if pd.isna(v):
                y.append(np.nan)
            else:
                y.append(float(v))

        return x, torch.tensor(y, dtype=torch.float32), idx


def build_model(n_labels):
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, n_labels)
    return model


def load_model(model_dir, n_labels, device):
    ckpt_path = Path(model_dir) / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model", "model_state"]:
            if key in ckpt:
                state = ckpt[key]
                break
        else:
            state = ckpt
    else:
        state = ckpt

    # Remove DataParallel prefix if needed
    clean_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        clean_state[nk] = v

    model = build_model(n_labels)
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        print("Warning missing keys:", missing[:10])
    if unexpected:
        print("Warning unexpected keys:", unexpected[:10])

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_logits(model, df, labels, img_size, batch_size, num_workers, device):
    ds = CXRDataset(df, labels, img_size)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_logits = []
    all_y = []
    all_idx = []

    for x, y, idx in dl:
        x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().numpy()
        all_logits.append(logits)
        all_y.append(y.numpy())
        all_idx.append(idx.numpy())

    logits = np.concatenate(all_logits, axis=0)
    y = np.concatenate(all_y, axis=0)
    idx = np.concatenate(all_idx, axis=0)

    order = np.argsort(idx)
    return logits[order], y[order]


def fit_temperature_per_label(logits, y):
    """
    Per-label temperature scaling.
    Fits one positive scalar T per label using source validation logits.
    """
    n_labels = logits.shape[1]
    temps = np.ones(n_labels, dtype=float)

    for j in range(n_labels):
        z = logits[:, j]
        yy = y[:, j]
        mask = np.isfinite(yy) & np.isfinite(z)

        if mask.sum() < 20 or len(np.unique(yy[mask])) < 2:
            temps[j] = 1.0
            continue

        z_t = torch.tensor(z[mask], dtype=torch.float32)
        y_t = torch.tensor(yy[mask], dtype=torch.float32)

        log_t = torch.nn.Parameter(torch.zeros(()))
        opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=100)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            T = torch.exp(log_t).clamp(0.05, 20.0)
            loss = loss_fn(z_t / T, y_t)
            loss.backward()
            return loss

        try:
            opt.step(closure)
            temps[j] = float(torch.exp(log_t).detach().clamp(0.05, 20.0))
        except Exception:
            temps[j] = 1.0

    return temps


def fit_platt_per_label(logits, y):
    models_out = []
    n_labels = logits.shape[1]

    for j in range(n_labels):
        z = logits[:, j]
        yy = y[:, j]
        mask = np.isfinite(yy) & np.isfinite(z)

        if mask.sum() < 20 or len(np.unique(yy[mask])) < 2:
            models_out.append(None)
            continue

        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr.fit(z[mask].reshape(-1, 1), yy[mask].astype(int))
        models_out.append(lr)

    return models_out


def fit_isotonic_per_label(logits, y):
    models_out = []
    n_labels = logits.shape[1]

    for j in range(n_labels):
        z = logits[:, j]
        yy = y[:, j]
        mask = np.isfinite(yy) & np.isfinite(z)

        if mask.sum() < 20 or len(np.unique(yy[mask])) < 2:
            models_out.append(None)
            continue

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(z[mask], yy[mask])
        models_out.append(iso)

    return models_out


def apply_calibration(logits, method, calibrator):
    if method == "raw":
        return sigmoid_np(logits)

    if method == "temperature":
        temps = np.asarray(calibrator, dtype=float)
        return sigmoid_np(logits / temps.reshape(1, -1))

    if method == "platt":
        probs = np.zeros_like(logits, dtype=float)
        for j, lr in enumerate(calibrator):
            if lr is None:
                probs[:, j] = sigmoid_np(logits[:, j])
            else:
                probs[:, j] = lr.predict_proba(logits[:, j].reshape(-1, 1))[:, 1]
        return probs

    if method == "isotonic":
        probs = np.zeros_like(logits, dtype=float)
        for j, iso in enumerate(calibrator):
            if iso is None:
                probs[:, j] = sigmoid_np(logits[:, j])
            else:
                probs[:, j] = iso.predict(logits[:, j])
        return np.clip(probs, 0, 1)

    raise ValueError(method)


def filter_labeled(df, labels):
    return df[df[labels].notna().any(axis=1)].copy()


def get_eval_df(manifest, dataset):
    if dataset == "vindr":
        split = "external_test"
    else:
        split = "internal_test"

    df = manifest[(manifest["dataset"] == dataset) & (manifest["split"] == split)].copy()
    return filter_labeled(df, LABELS), split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--train-source", default=None)
    parser.add_argument("--subset-n", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-datasets", nargs="+", default=["nih", "chexpert", "vindr"])
    parser.add_argument("--labels", nargs="+", default=LABELS)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--calib-limit", type=int, default=5000)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--metrics-out", required=True)
    args = parser.parse_args()

    inferred_src, inferred_n, inferred_seed = parse_model_name(args.model_dir)
    train_source = args.train_source or inferred_src
    subset_n = args.subset_n or inferred_n
    seed = args.seed if args.seed is not None else inferred_seed

    if train_source is None:
        raise ValueError("train_source could not be inferred. Pass --train-source.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Train source:", train_source)
    print("Subset n:", subset_n)
    print("Seed:", seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest, low_memory=False)

    # Source validation for calibration
    calib_df = manifest[
        (manifest["dataset"] == train_source) &
        (manifest["split"] == "val")
    ].copy()
    calib_df = filter_labeled(calib_df, args.labels)

    if args.calib_limit and args.calib_limit > 0 and len(calib_df) > args.calib_limit:
        calib_df = calib_df.sample(n=args.calib_limit, random_state=seed).copy()

    print("Calibration rows:", len(calib_df))

    model = load_model(args.model_dir, len(args.labels), device)

    calib_logits, calib_y = predict_logits(
        model=model,
        df=calib_df,
        labels=args.labels,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    temps = fit_temperature_per_label(calib_logits, calib_y)
    platt_models = fit_platt_per_label(calib_logits, calib_y)
    isotonic_models = fit_isotonic_per_label(calib_logits, calib_y)

    print("Temperatures:", dict(zip(args.labels, temps)))

    rows = []
    pred_tables = []

    calibrators = {
        "raw": None,
        "temperature": temps,
        "platt": platt_models,
        "isotonic": isotonic_models,
    }

    for eval_dataset in args.eval_datasets:
        eval_df, eval_split = get_eval_df(manifest, eval_dataset)
        print(f"Predicting eval_dataset={eval_dataset}, split={eval_split}, rows={len(eval_df)}")

        logits, y = predict_logits(
            model=model,
            df=eval_df,
            labels=args.labels,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        base_pred_df = eval_df.reset_index(drop=True).copy()
        base_pred_df["train_source"] = train_source
        base_pred_df["subset_n"] = subset_n
        base_pred_df["seed"] = seed
        base_pred_df["eval_dataset"] = eval_dataset
        base_pred_df["eval_split"] = eval_split

        for method, calibrator in calibrators.items():
            probs = apply_calibration(logits, method, calibrator)

            pred_df = base_pred_df[[
                "train_source", "subset_n", "seed",
                "eval_dataset", "eval_split",
                "dataset", "split"
            ] + [get_path_col(base_pred_df)] + args.labels].copy()

            pred_df["calibration_method"] = method

            for j, lab in enumerate(args.labels):
                base = lab[2:] if lab.startswith("y_") else lab
                pred_df[f"logit_{base}"] = logits[:, j]
                pred_df[f"p_{base}"] = probs[:, j]

            pred_tables.append(pred_df)

            per_label = []
            for j, lab in enumerate(args.labels):
                base = lab[2:] if lab.startswith("y_") else lab
                m = compute_metrics(y[:, j], probs[:, j], n_bins=args.n_bins)

                row = {
                    "train_source": train_source,
                    "subset_n": subset_n,
                    "seed": seed,
                    "eval_dataset": eval_dataset,
                    "eval_split": eval_split,
                    "label": lab,
                    "method": method,
                }
                row.update(m)
                rows.append(row)
                per_label.append(row)

            mean_row = {
                "train_source": train_source,
                "subset_n": subset_n,
                "seed": seed,
                "eval_dataset": eval_dataset,
                "eval_split": eval_split,
                "label": "mean_primary_labels",
                "method": method,
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
                vals = [r[metric] for r in per_label if np.isfinite(r[metric])]
                mean_row[metric] = float(np.mean(vals)) if vals else np.nan

            mean_row["n"] = int(np.nanmean([r["n"] for r in per_label]))
            mean_row["positive"] = float(np.nanmean([r["positive"] for r in per_label]))
            rows.append(mean_row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.metrics_out, index=False)

    all_preds = pd.concat(pred_tables, ignore_index=True)
    pred_out = out_dir / "calibrated_predictions.csv"
    all_preds.to_csv(pred_out, index=False)

    calib_info = pd.DataFrame({
        "label": args.labels,
        "temperature": temps,
    })
    calib_info.to_csv(out_dir / "calibration_parameters.csv", index=False)

    print("Saved metrics:", args.metrics_out)
    print("Saved predictions:", pred_out)
    print("Saved calibration params:", out_dir / "calibration_parameters.csv")

    show = metrics[metrics["label"] == "mean_primary_labels"].copy()
    cols = [
        "train_source", "subset_n", "seed",
        "eval_dataset", "eval_split",
        "method",
        "AUROC", "AUPRC", "prevalence",
        "AUPRC_lift", "normalized_AUPRC",
        "Brier", "ECE", "n"
    ]
    print(show[cols].to_string(index=False))


if __name__ == "__main__":
    main()
