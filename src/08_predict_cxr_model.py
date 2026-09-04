#!/usr/bin/env python
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

class CXRPredDataset(Dataset):
    def __init__(self, df, labels, transform):
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
            img = Image.new("RGB", (224, 224), color=0)

        img = self.transform(img)

        y = []
        for lab in self.labels:
            val = r[lab] if lab in r else np.nan
            y.append(np.nan if pd.isna(val) else float(val))

        meta = {
            "dataset": r["dataset"],
            "split": r["split"],
            "patient_id": str(r["patient_id"]),
            "study_id": str(r["study_id"]),
            "image_id": str(r["image_id"]),
            "image_path_final": str(r["image_path_final"]),
        }

        return img, torch.tensor([0 if pd.isna(v) else v for v in y], dtype=torch.float32), torch.tensor([0 if pd.isna(v) else 1 for v in y], dtype=torch.float32), meta

def collate_fn(batch):
    imgs, ys, ms, metas = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    ys = torch.stack(ys, dim=0)
    ms = torch.stack(ms, dim=0)
    return imgs, ys, ms, metas

def make_model(n_labels):
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, n_labels)
    return model

def make_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/model_manifest_all_splits.csv")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--eval-datasets", nargs="+", default=["nih", "chexpert", "vindr"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(args.model_dir)
    ckpt_path = model_dir / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)

    labels = ckpt["labels"]
    cfg = ckpt.get("config", {})
    train_source = cfg.get("train_source", "unknown")
    subset_n = cfg.get("subset_n", "unknown")
    seed = cfg.get("seed", "unknown")

    model = make_model(len(labels))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    df = pd.read_csv(args.manifest, low_memory=False)

    eval_parts = []
    for ds in args.eval_datasets:
        if ds == "vindr":
            part = df[(df["dataset"] == ds) & (df["split"] == "external_test")].copy()
        else:
            part = df[(df["dataset"] == ds) & (df["split"] == "internal_test")].copy()
        part = part[part[labels].notna().any(axis=1)].copy()
        eval_parts.append(part)

    eval_df = pd.concat(eval_parts, ignore_index=True)
    print("Evaluation rows:", len(eval_df))
    print(pd.crosstab(eval_df["dataset"], eval_df["split"], dropna=False))

    ds = CXRPredDataset(eval_df, labels, make_transform(args.img_size))
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn
    )

    rows = []
    with torch.no_grad():
        for imgs, ys, ms, metas in tqdm(loader, desc="predict"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            ys_np = ys.numpy()
            ms_np = ms.numpy()

            for i, meta in enumerate(metas):
                row = dict(meta)
                row["train_source"] = train_source
                row["subset_n"] = subset_n
                row["seed"] = seed
                for j, lab in enumerate(labels):
                    row[f"true_{lab}"] = ys_np[i, j] if ms_np[i, j] == 1 else np.nan
                    row[f"pred_{lab}"] = probs[i, j]
                rows.append(row)

    out_dir = Path(args.out_dir or f"results/predictions/{train_source}_n{subset_n}_seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print("Saved:", out_path)

if __name__ == "__main__":
    main()
