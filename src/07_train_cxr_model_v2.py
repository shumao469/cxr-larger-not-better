#!/usr/bin/env python3
"""Strict and reproducible DenseNet-121 training for the formal three-seed rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


LABELS = [
    "y_Cardiomegaly",
    "y_Pleural_Effusion",
    "y_Atelectasis",
    "y_Consolidation",
]


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CXRDataset(Dataset):
    def __init__(self, df, labels, transform=None, allow_blank_images=False,
                 image_cache=None, img_size=224):
        self.df = df.reset_index(drop=True)
        self.labels = labels
        self.transform = transform
        self.allow_blank_images = allow_blank_images
        self.image_cache = image_cache
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(row["image_path_final"])
        try:
            image = (self.image_cache.load_rgb(path, self.img_size)
                     if self.image_cache is not None
                     else Image.open(path).convert("RGB"))
        except Exception as exc:
            if not self.allow_blank_images:
                raise RuntimeError(f"Cannot read training image: {path}") from exc
            image = Image.new("RGB", (224, 224), color=0)

        if self.transform:
            image = self.transform(image)
        target, mask = [], []
        for label in self.labels:
            value = row[label]
            if pd.isna(value):
                target.append(0.0)
                mask.append(0.0)
            else:
                target.append(float(value))
                mask.append(1.0)
        return {
            "image": image,
            "target": torch.tensor(target, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
        }


def make_transforms(img_size, train=True):
    ops = [transforms.Resize((img_size, img_size))]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(ops)


def make_model(n_labels):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, n_labels)
    return model


def compute_pos_weight(df, labels, cap=20.0):
    weights = []
    for label in labels:
        values = df[label].dropna()
        positive = float((values == 1).sum())
        negative = float((values == 0).sum())
        value = 1.0 if positive <= 0 else negative / positive
        weights.append(max(1.0, min(cap, value)))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate_val(model, loader, labels, device):
    model.eval()
    all_y, all_p, all_m = [], [], []
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device)
        labels_true = batch["target"].cpu().numpy()
        masks = batch["mask"].cpu().numpy()
        probabilities = torch.sigmoid(model(images)).cpu().numpy()
        all_y.append(labels_true)
        all_p.append(probabilities)
        all_m.append(masks)
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    m = np.concatenate(all_m)
    aucs = {}
    for j, label in enumerate(labels):
        available = m[:, j] == 1
        if available.sum() < 10 or len(np.unique(y[available, j])) < 2:
            aucs[label] = np.nan
        else:
            aucs[label] = float(roc_auc_score(y[available, j], p[available, j]))
    valid = [value for value in aucs.values() if np.isfinite(value)]
    return (float(np.mean(valid)) if valid else np.nan), aucs


def atomic_torch_save(payload, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def open_validated_image_cache(args):
    """Enable only a locally verified, opt-in cache; never change the protocol."""
    info = {"enabled": False, "reason": "disabled_or_not_configured"}
    path = Path(args.image_cache_config)
    if args.disable_image_cache or not path.is_file():
        return None, info
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
        if not settings.get("enabled", False):
            return None, info
        if args.num_workers != 0 or args.allow_blank_images or args.img_size != 224:
            raise ValueError("Cache rollout is validated only for strict 224px, workers=0")
        report_path = Path(settings["validation_report"])
        report_bytes = report_path.read_bytes()
        if hashlib.sha256(report_bytes).hexdigest() != settings["validation_report_sha256"]:
            raise ValueError("Cache validation report hash changed")
        report = json.loads(report_bytes)
        if report.get("status") != "PASS":
            raise ValueError("Cache validation did not pass")
        for filename, field in [("cxr_lossless_cache.py", "cache_module_sha256"),
                                (Path(__file__).name, "trainer_sha256")]:
            digest = hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
            if report.get(field) != digest:
                raise ValueError(f"Validated implementation changed: {filename}")
        from cxr_lossless_cache import LosslessResizeCache
        cache = LosslessResizeCache(
            settings["cache_path"], max_bytes=int(settings["max_bytes"]),
            reserve_bytes=int(settings["reserve_bytes"]), space_root=settings["space_root"],
        )
        info = {"enabled": True, "settings": settings,
                "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "equivalence": "PIL_RGB_Resize224_prefix_only_original_transforms_preserved"}
        print("Lossless image cache enabled:", settings["cache_path"], flush=True)
        return cache, info
    except Exception as exc:
        # Acceleration is optional. An unavailable/invalid cache must not stop
        # training or replace a bad original image with a synthetic image.
        info["reason"] = f"fallback_to_original: {type(exc).__name__}: {exc}"
        print("Image cache unavailable; using original images:", info["reason"], flush=True)
        return None, info


def save_cache_stats(cache, out_dir, epoch):
    if cache is None:
        return
    try:
        stats = {"epoch": epoch, "updated_at": time.time(), **cache.snapshot_stats()}
        path = out_dir / "image_cache_stats.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        os.replace(temp, path)
        print("Image cache:", json.dumps(stats, sort_keys=True), flush=True)
    except Exception as exc:
        print("Image cache statistics unavailable (training preserved):", exc, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--subset-dir", required=True)
    parser.add_argument("--train-source", required=True, choices=["nih", "chexpert"])
    parser.add_argument("--subset-n", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--labels", nargs="+", default=LABELS)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--val-limit", type=int, default=3000)
    parser.add_argument("--val-sampling-seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument("--allow-blank-images", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--image-cache-config", default=str(
        Path(__file__).resolve().parents[1] / "config/lossless_cache_formal_v2.json"))
    parser.add_argument("--disable-image-cache", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed, deterministic=not args.non_deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    subset_path = Path(args.subset_dir) / f"{args.train_source}_n{args.subset_n}_seed{args.seed}.csv"
    if not subset_path.exists():
        raise FileNotFoundError(subset_path)
    train_df = pd.read_csv(subset_path, low_memory=False)
    manifest = pd.read_csv(args.manifest, low_memory=False)
    val_df = manifest[
        (manifest["dataset"] == args.train_source) & (manifest["split"] == "val")
    ].copy()
    train_df = train_df[train_df[args.labels].notna().any(axis=1)].copy()
    val_df = val_df[val_df[args.labels].notna().any(axis=1)].copy()

    if args.val_limit > 0 and len(val_df) > args.val_limit:
        val_df = val_df.sample(
            n=args.val_limit, random_state=args.val_sampling_seed
        ).copy()
    val_df = val_df.sort_values(["patient_id", "image_id"]).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_df[["dataset", "split", "patient_id", "study_id", "image_id", "image_path_final", *args.labels]].to_csv(
        out_dir / "validation_cohort.csv", index=False
    )
    print("Train rows:", len(train_df), "Val rows:", len(val_df))

    image_cache, image_cache_info = open_validated_image_cache(args)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        CXRDataset(train_df, args.labels, make_transforms(args.img_size, True), args.allow_blank_images,
                   image_cache=image_cache, img_size=args.img_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        CXRDataset(val_df, args.labels, make_transforms(args.img_size, False), args.allow_blank_images,
                   image_cache=image_cache, img_size=args.img_size),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        worker_init_fn=seed_worker,
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

    config = vars(args).copy()
    config.update(
        {
            "labels": args.labels,
            "device": device,
            "subset_path": str(subset_path.resolve()),
            "formal_protocol_version": "v2-fixed-validation-strict-images",
            "image_cache": image_cache_info,
        }
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_auc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        started = time.time()
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs} train"):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(images)
                raw_loss = loss_fn(logits, targets)
                loss = (raw_loss * masks).sum() / masks.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        val_auc, val_aucs = evaluate_val(model, val_loader, args.labels, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_mean_auc": val_auc,
            "elapsed_sec": time.time() - started,
            **{f"val_auc_{key}": value for key, value in val_aucs.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        payload = {
            "model_state": model.state_dict(),
            "labels": args.labels,
            "epoch": epoch,
            "val_mean_auc": val_auc,
            "config": config,
        }
        atomic_torch_save(payload, out_dir / "last.pt")
        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            atomic_torch_save(payload, out_dir / "best.pt")
            print("Saved best:", out_dir / "best.pt")
        print("Epoch", epoch, "loss", row["train_loss"], "val_mean_auc", val_auc)
        save_cache_stats(image_cache, out_dir, epoch)
    print("Finished. Best val mean AUROC:", best_auc)
    if image_cache is not None:
        try:
            image_cache.close()
        except Exception as exc:
            print("Image cache close warning (training completed):", exc, flush=True)


if __name__ == "__main__":
    main()
