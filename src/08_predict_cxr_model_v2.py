#!/usr/bin/env python3
"""Strict prediction with atomic output and frozen-cohort checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


class CXRPredDataset(Dataset):
    def __init__(self, df, labels, transform, allow_blank_images=False):
        self.df = df.reset_index(drop=True)
        self.labels = labels
        self.transform = transform
        self.allow_blank_images = allow_blank_images

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(row["image_path_final"])
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            if not self.allow_blank_images:
                raise RuntimeError(f"Cannot read evaluation image: {path}") from exc
            image = Image.new("RGB", (224, 224), color=0)
        image = self.transform(image)
        values = [row[label] if label in row else np.nan for label in self.labels]
        target = torch.tensor([0 if pd.isna(v) else float(v) for v in values], dtype=torch.float32)
        mask = torch.tensor([0 if pd.isna(v) else 1 for v in values], dtype=torch.float32)
        meta = {
            "dataset": row["dataset"],
            "split": row["split"],
            "patient_id": str(row["patient_id"]),
            "study_id": str(row["study_id"]),
            "image_id": str(row["image_id"]),
            "image_path_final": path,
        }
        return image, target, mask, meta


def collate_fn(batch):
    images, targets, masks, metas = zip(*batch)
    return torch.stack(images), torch.stack(targets), torch.stack(masks), metas


def make_model(n_labels):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, n_labels)
    return model


def transform(img_size):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--eval-datasets", nargs="+", default=["nih", "chexpert", "vindr"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--expected-total", type=int, default=65283)
    parser.add_argument("--allow-blank-images", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path(args.model_dir) / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in ["labels", "model_state"]:
        if key not in checkpoint:
            raise KeyError(f"Checkpoint missing {key}: {checkpoint_path}")
    labels = checkpoint["labels"]
    config = checkpoint.get("config", {})
    train_source = config.get("train_source", "unknown")
    subset_n = config.get("subset_n", "unknown")
    seed = config.get("seed", "unknown")

    model = make_model(len(labels))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()

    manifest = pd.read_csv(args.manifest, low_memory=False)
    parts = []
    for dataset in args.eval_datasets:
        split = "external_test" if dataset == "vindr" else "internal_test"
        part = manifest[(manifest["dataset"] == dataset) & (manifest["split"] == split)].copy()
        part = part[part[labels].notna().any(axis=1)].copy()
        parts.append(part)
    eval_df = pd.concat(parts, ignore_index=True)
    if len(eval_df) != args.expected_total:
        raise RuntimeError(
            f"Frozen evaluation cohort mismatch: observed {len(eval_df):,}, expected {args.expected_total:,}"
        )
    duplicate_count = int(eval_df.duplicated(["dataset", "image_id"]).sum())
    if duplicate_count:
        raise RuntimeError(f"Evaluation cohort has {duplicate_count} duplicate dataset/image_id rows")
    print(pd.crosstab(eval_df["dataset"], eval_df["split"]))

    loader = DataLoader(
        CXRPredDataset(eval_df, labels, transform(args.img_size), args.allow_blank_images),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )
    rows = []
    with torch.no_grad():
        for images, targets, masks, metas in tqdm(loader, desc="predict"):
            probabilities = torch.sigmoid(model(images.to(device))).cpu().numpy()
            targets_np = targets.numpy()
            masks_np = masks.numpy()
            for i, meta in enumerate(metas):
                row = dict(meta)
                row.update({"train_source": train_source, "subset_n": subset_n, "seed": seed})
                for j, label in enumerate(labels):
                    row[f"true_{label}"] = targets_np[i, j] if masks_np[i, j] == 1 else np.nan
                    row[f"pred_{label}"] = probabilities[i, j]
                rows.append(row)

    output = pd.DataFrame(rows)
    if len(output) != args.expected_total:
        raise RuntimeError(f"Prediction output row mismatch: {len(output):,}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.csv"
    temp = out_path.with_suffix(".csv.tmp")
    output.to_csv(temp, index=False)
    os.replace(temp, out_path)
    metadata = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "prediction_rows": int(len(output)),
        "counts": output.groupby(["dataset", "split"]).size().astype(int).to_dict(),
        "labels": labels,
    }
    metadata["counts"] = {f"{k[0]}/{k[1]}": v for k, v in metadata["counts"].items()}
    (out_dir / "prediction_audit.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
