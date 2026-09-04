#!/usr/bin/env python3
"""Train the 30 formal NIH/CheXpert models before VinDr is available.

This stage creates only subsets, validation cohorts, histories, and checkpoints.
Prediction and evaluation remain deferred until the final three-dataset manifest
is available, at which point 14_plan_or_run_three_seed_recovery.py reuses these
formal checkpoints.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch


SOURCES = ["nih", "chexpert"]
SIZES = ["1000", "5000", "10000", "50000", "all"]
SEEDS = [1, 2, 3]
EXPECTED_ROWS = 259_439
EXPECTED_COUNTS = {
    ("chexpert", "train"): 103_286,
    ("chexpert", "val"): 15_047,
    ("chexpert", "internal_test"): 28_986,
    ("nih", "train"): 79_481,
    ("nih", "val"): 11_342,
    ("nih", "internal_test"): 21_297,
}


def checkpoint_is_complete(model_dir: Path, epochs: int):
    checkpoint_path = model_dir / "best.pt"
    history_path = model_dir / "history.csv"
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size < 1_000_000:
        return False, "missing_or_too_small"
    if not history_path.exists():
        return False, "history_missing"
    try:
        history = pd.read_csv(history_path)
        if len(history) < epochs or int(history["epoch"].max()) < epochs:
            return False, "history_incomplete"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or not {"model_state", "labels", "config"}.issubset(checkpoint):
            return False, "checkpoint_keys_missing"
        config = checkpoint["config"]
        if int(config.get("epochs", -1)) != epochs:
            return False, "epoch_config_mismatch"
        return True, "complete"
    except Exception as exc:
        return False, f"invalid:{type(exc).__name__}"


def run_command(command, log_path: Path, cwd: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN:", " ".join(str(item) for item in command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND: " + " ".join(str(item) for item in command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed with code {result.returncode}; see {log_path}")


def audit_manifest(manifest_path: Path):
    df = pd.read_csv(manifest_path, low_memory=False)
    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(f"Training manifest row mismatch: {len(df):,} != {EXPECTED_ROWS:,}")
    counts = df.groupby(["dataset", "split"]).size().to_dict()
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Training manifest count mismatch: {counts}")
    missing = int((~df["image_path_final"].map(lambda value: Path(str(value)).is_file())).sum())
    if missing:
        raise RuntimeError(f"Training manifest has {missing:,} missing images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", default="data/processed/model_manifest_training_v2.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=3000)
    parser.add_argument("--val-sampling-seed", type=int, default=2026)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    manifest = (root / args.manifest).resolve()
    audit_manifest(manifest)

    python = sys.executable
    subset_dir = root / "data" / "subsets_formal_v2"
    run_root = root / "results_formal_v2"
    model_root = run_root / "models"
    log_root = run_root / "logs"
    model_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    subset_summary = subset_dir / "subset_summary.csv"
    if not subset_summary.exists():
        run_command(
            [
                python,
                str(root / "src/06_make_sample_size_subsets.py"),
                "--manifest", str(manifest),
                "--out-dir", str(subset_dir),
                "--seeds", "1", "2", "3",
            ],
            log_root / "00_make_subsets.log",
            root,
        )

    plan = []
    for source in SOURCES:
        for size in SIZES:
            for seed in SEEDS:
                experiment = f"{source}_n{size}_seed{seed}"
                model_dir = model_root / experiment
                complete, note = checkpoint_is_complete(model_dir, args.epochs)
                action = "complete" if complete else "train"
                plan.append({"experiment": experiment, "action": action, "checkpoint_note": note})
                pd.DataFrame(plan).to_csv(run_root / "training_plan.csv", index=False)
                if complete:
                    print("SKIP complete:", experiment, flush=True)
                    continue
                run_command(
                    [
                        python,
                        str(root / "src/07_train_cxr_model_v2.py"),
                        "--manifest", str(manifest),
                        "--subset-dir", str(subset_dir),
                        "--train-source", source,
                        "--subset-n", size,
                        "--seed", str(seed),
                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--val-limit", str(args.val_limit),
                        "--val-sampling-seed", str(args.val_sampling_seed),
                        "--num-workers", str(args.num_workers),
                        "--use-pos-weight",
                        "--out-dir", str(model_dir),
                    ],
                    log_root / f"train_{experiment}.log",
                    root,
                )
                complete, note = checkpoint_is_complete(model_dir, args.epochs)
                if not complete:
                    raise RuntimeError(f"Checkpoint audit failed for {experiment}: {note}")
                plan[-1] = {"experiment": experiment, "action": "complete", "checkpoint_note": note}
                pd.DataFrame(plan).to_csv(run_root / "training_plan.csv", index=False)

    completed = sum(row["action"] == "complete" for row in plan)
    summary = {
        "status": "complete" if completed == 30 else "incomplete",
        "completed_models": completed,
        "expected_models": 30,
        "manifest": str(manifest),
        "prediction_and_external_evaluation_deferred": True,
    }
    (run_root / "training_only_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
