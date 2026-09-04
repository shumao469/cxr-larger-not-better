#!/usr/bin/env python
from pathlib import Path
import argparse
import json
import time
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import roc_auc_score
from tqdm import tqdm

LABELS = [
    "y_Cardiomegaly",
    "y_Pleural_Effusion",
    "y_Atelectasis",
    "y_Consolidation",
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CXRDataset(Dataset):
    def __init__(self, df, labels, transform=None):
        self.df = df.reset_index(drop=True)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        path = r["image_path_final"]

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # fallback blank image; should not happen after file check
            img = Image.new("RGB", (224, 224), color=0)

        if self.transform:
            img = self.transform(img)

        y = []
        m = []
        for lab in self.labels:
            val = r[lab]
            if pd.isna(val):
                y.append(0.0)
                m.append(0.0)
            else:
                y.append(float(val))
                m.append(1.0)

        return {
            "image": img,
            "target": torch.tensor(y, dtype=torch.float32),
            "mask": torch.tensor(m, dtype=torch.float32),
        }

def make_transforms(img_size, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def make_model(n_labels):
    weights = models.DenseNet121_Weights.IMAGENET1K_V1
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, n_labels)
    return model

def compute_pos_weight(df, labels, cap=20.0):
    weights = []
    for lab in labels:
        x = df[lab].dropna()
        pos = float((x == 1).sum())
        neg = float((x == 0).sum())
        if pos <= 0:
            w = 1.0
        else:
            w = neg / pos
        w = max(1.0, min(cap, w))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)

@torch.no_grad()
def evaluate_val(model, loader, labels, device):
    model.eval()
    all_y, all_p, all_m = [], [], []

    for batch in tqdm(loader, desc="val", leave=False):
        x = batch["image"].to(device)
        y = batch["target"].cpu().numpy()
        m = batch["mask"].cpu().numpy()

        logits = model(x)
        p = torch.sigmoid(logits).cpu().numpy()

        all_y.append(y)
        all_p.append(p)
        all_m.append(m)

    y = np.concatenate(all_y, axis=0)
    p = np.concatenate(all_p, axis=0)
    m = np.concatenate(all_m, axis=0)

    aucs = {}
    for j, lab in enumerate(labels):
        mask = m[:, j] == 1
        if mask.sum() < 10 or len(np.unique(y[mask, j])) < 2:
            aucs[lab] = np.nan
        else:
            aucs[lab] = roc_auc_score(y[mask, j], p[mask, j])

    valid_aucs = [v for v in aucs.values() if not np.isnan(v)]
    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else np.nan
    return mean_auc, aucs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/model_manifest_all_splits.csv")
    parser.add_argument("--subset-dir", default="data/subsets")
    parser.add_argument("--train-source", required=True, choices=["nih", "chexpert"])
    parser.add_argument("--subset-n", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--labels", nargs="+", default=LABELS)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--val-limit", type=int, default=3000, help="Limit validation rows for faster model selection; 0 means full validation.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-pos-weight", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    subset_path = Path(args.subset_dir) / f"{args.train_source}_n{args.subset_n}_seed{args.seed}.csv"
    if not subset_path.exists():
        raise FileNotFoundError(subset_path)

    train_df = pd.read_csv(subset_path, low_memory=False)
    full_df = pd.read_csv(args.manifest, low_memory=False)

    val_df = full_df[
        (full_df["dataset"] == args.train_source) &
        (full_df["split"] == "val")
    ].copy()

    # Keep rows with at least one selected label available.
    train_df = train_df[train_df[args.labels].notna().any(axis=1)].copy()
    val_df = val_df[val_df[args.labels].notna().any(axis=1)].copy()

    # Use a fixed validation subset for faster model selection.
    # Full internal/external evaluation is done later by 08_predict + 09_evaluate.
    if args.val_limit and args.val_limit > 0 and len(val_df) > args.val_limit:
        val_df = val_df.sample(n=args.val_limit, random_state=args.seed).copy()

    print("Train rows:", len(train_df))
    print("Val rows:", len(val_df))
    print("Labels:", args.labels)

    out_dir = Path(args.out_dir or f"results/models/{args.train_source}_n{args.subset_n}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CXRDataset(train_df, args.labels, make_transforms(args.img_size, train=True))
    val_ds = CXRDataset(val_df, args.labels, make_transforms(args.img_size, train=False))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = make_model(len(args.labels)).to(device)

    if args.use_pos_weight:
        pos_weight = compute_pos_weight(train_df, args.labels).to(device)
        print("pos_weight:", pos_weight.detach().cpu().numpy())
        loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_auc = -1.0
    history = []

    config = vars(args).copy()
    config["labels"] = args.labels
    config["device"] = device
    config["subset_path"] = str(subset_path)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        start = time.time()

        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs} train"):
            x = batch["image"].to(device)
            y = batch["target"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                raw_loss = loss_fn(logits, y)
                denom = mask.sum().clamp_min(1.0)
                loss = (raw_loss * mask).sum() / denom

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.detach().cpu()))

        val_auc, val_aucs = evaluate_val(model, val_loader, args.labels, device)
        mean_loss = float(np.mean(losses))
        elapsed = time.time() - start

        row = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "val_mean_auc": val_auc,
            "elapsed_sec": elapsed,
        }
        for k, v in val_aucs.items():
            row[f"val_auc_{k}"] = v
        history.append(row)

        print("Epoch", epoch, "loss", mean_loss, "val_mean_auc", val_auc)

        hist_df = pd.DataFrame(history)
        hist_df.to_csv(out_dir / "history.csv", index=False)

        torch.save(
            {
                "model_state": model.state_dict(),
                "labels": args.labels,
                "epoch": epoch,
                "val_mean_auc": val_auc,
                "config": config,
            },
            out_dir / "last.pt",
        )

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "labels": args.labels,
                    "epoch": epoch,
                    "val_mean_auc": val_auc,
                    "config": config,
                },
                out_dir / "best.pt",
            )
            print("Saved best:", out_dir / "best.pt")

    print("Finished. Best val mean AUROC:", best_auc)

if __name__ == "__main__":
    main()
