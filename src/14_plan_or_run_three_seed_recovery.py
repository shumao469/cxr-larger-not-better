#!/usr/bin/env python3
"""Plan or execute a resumable three-seed Stage 1 recovery.

Formal mode retrains all 30 experiments in a separate result tree. Salvage mode
reuses complete legacy predictions or loadable checkpoints when possible and is
intended only for rapid exploratory recovery.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd


SOURCES = ["nih", "chexpert"]
SIZES = ["1000", "5000", "10000", "50000", "all"]
SEEDS = [1, 2, 3]
EXPECTED_PREDICTION_ROWS = 65283


def checkpoint_is_loadable(path: Path):
    if not path.exists() or path.stat().st_size < 1_000_000:
        return False, "missing_or_too_small"
    try:
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        valid = isinstance(checkpoint, dict) and "model_state" in checkpoint and "labels" in checkpoint
        return valid, "torch_load_ok" if valid else "missing_checkpoint_keys"
    except Exception as exc:
        return False, f"torch_load_failed:{type(exc).__name__}"


def prediction_is_complete(path: Path):
    if not path.exists() or path.stat().st_size < 1_000_000:
        return False, 0, "missing_or_too_small"
    try:
        df = pd.read_csv(path, low_memory=False)
        required = {"dataset", "split", "image_id"}
        if not required.issubset(df.columns):
            return False, len(df), "missing_required_columns"
        truth = [column for column in df.columns if column.startswith("true_y_")]
        pred = [column for column in df.columns if column.startswith("pred_y_")]
        valid = len(df) == EXPECTED_PREDICTION_ROWS and len(truth) == 4 and len(pred) == 4
        return valid, len(df), "ok" if valid else "row_or_label_column_mismatch"
    except Exception as exc:
        return False, 0, f"csv_read_failed:{type(exc).__name__}"


def metric_is_complete(path: Path):
    try:
        df = pd.read_csv(path)
        return len(df) == 15 and (df["label"] == "mean_primary_labels").sum() == 3
    except Exception:
        return False


def run_command(command, log_path, cwd):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN:", " ".join(str(item) for item in command))
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND: " + " ".join(str(item) for item in command) + "\n")
        result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed with code {result.returncode}; see {log_path}")


def merge_metrics(metric_dir: Path):
    frames = []
    for path in sorted(metric_dir.glob("*_metrics.csv")):
        if path.name in {"all_stage1_metrics.csv", "stage1_mean_primary_metrics.csv"}:
            continue
        df = pd.read_csv(path)
        if len(df) == 15:
            df["metric_file"] = str(path)
            frames.append(df)
    if len(frames) != 30:
        raise RuntimeError(f"Expected 30 complete metric files, found {len(frames)}")
    all_metrics = pd.concat(frames, ignore_index=True)
    all_metrics.to_csv(metric_dir / "all_stage1_metrics.csv", index=False)
    all_metrics[all_metrics["label"] == "mean_primary_labels"].to_csv(
        metric_dir / "stage1_mean_primary_metrics.csv", index=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", default="data/processed/model_manifest_formal_v2.csv")
    parser.add_argument("--mode", choices=["formal", "salvage"], default="formal")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--with-calibration", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=3000)
    parser.add_argument("--val-sampling-seed", type=int, default=2026)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    manifest = (root / args.manifest).resolve()
    if manifest.exists():
        manifest_df = pd.read_csv(
            manifest,
            low_memory=False,
            usecols=["dataset", "split", "image_id", "image_path_final"],
        )
        if len(manifest_df) != 274439:
            raise RuntimeError(f"Validated manifest row mismatch: {len(manifest_df):,}")
        missing_images = int(
            (~manifest_df["image_path_final"].map(lambda value: Path(str(value)).is_file())).sum()
        )
        if missing_images and args.execute:
            raise RuntimeError(
                f"Validated manifest has {missing_images:,} missing images; execution is blocked"
            )
        if missing_images:
            print(f"Plan warning: validated manifest has {missing_images:,} missing images")
    elif args.execute:
        raise FileNotFoundError(f"Validated manifest not found: {manifest}")
    else:
        print(f"Plan warning: validated manifest not found yet: {manifest}")

    python = sys.executable
    if args.mode == "formal":
        run_root = root / "results_formal_v2"
        subset_dir = root / "data" / "subsets_formal_v2"
    else:
        run_root = root / "results_salvage_v2"
        subset_dir = root / "data" / "subsets_salvage_v2"
    model_root = run_root / "models"
    prediction_root = run_root / "predictions"
    metric_root = run_root / "metrics"
    calibration_root = run_root / "calibration"
    log_root = run_root / "logs"
    for directory in [model_root, prediction_root, metric_root, log_root]:
        directory.mkdir(parents=True, exist_ok=True)

    if args.execute and not (subset_dir / "subset_summary.csv").exists():
        run_command(
            [python, str(root / "src/06_make_sample_size_subsets.py"), "--manifest", str(manifest), "--out-dir", str(subset_dir), "--seeds", "1", "2", "3"],
            log_root / "00_make_subsets.log",
            root,
        )

    plan = []
    for source in SOURCES:
        for size in SIZES:
            for seed in SEEDS:
                experiment = f"{source}_n{size}_seed{seed}"
                formal_model_dir = model_root / experiment
                formal_pred_dir = prediction_root / experiment
                metric_path = metric_root / f"{experiment}_metrics.csv"
                legacy_model_dir = root / "results" / "models" / experiment
                legacy_pred_path = root / "results" / "predictions" / experiment / "predictions.csv"

                metric_complete = metric_is_complete(metric_path)
                current_pred_complete, current_pred_rows, _ = prediction_is_complete(formal_pred_dir / "predictions.csv")
                current_ckpt_ok, current_ckpt_note = checkpoint_is_loadable(formal_model_dir / "best.pt")
                legacy_pred_complete, legacy_pred_rows, legacy_pred_note = prediction_is_complete(legacy_pred_path)
                legacy_ckpt_ok, legacy_ckpt_note = checkpoint_is_loadable(legacy_model_dir / "best.pt")

                if metric_complete:
                    action = "complete"
                    selected_model_dir = formal_model_dir
                    selected_pred_path = formal_pred_dir / "predictions.csv"
                elif current_pred_complete:
                    action = "evaluate"
                    selected_model_dir = formal_model_dir
                    selected_pred_path = formal_pred_dir / "predictions.csv"
                elif current_ckpt_ok:
                    action = "predict_current_checkpoint"
                    selected_model_dir = formal_model_dir
                    selected_pred_path = formal_pred_dir / "predictions.csv"
                elif args.mode == "salvage" and legacy_pred_complete:
                    action = "evaluate_legacy_prediction"
                    selected_model_dir = legacy_model_dir
                    selected_pred_path = legacy_pred_path
                elif args.mode == "salvage" and legacy_ckpt_ok:
                    action = "predict_legacy_checkpoint"
                    selected_model_dir = legacy_model_dir
                    selected_pred_path = formal_pred_dir / "predictions.csv"
                else:
                    action = "train_predict_evaluate"
                    selected_model_dir = formal_model_dir
                    selected_pred_path = formal_pred_dir / "predictions.csv"

                plan.append(
                    {
                        "experiment": experiment,
                        "mode": args.mode,
                        "action": action,
                        "current_checkpoint_loadable": current_ckpt_ok,
                        "current_checkpoint_note": current_ckpt_note,
                        "legacy_prediction_complete": legacy_pred_complete,
                        "legacy_prediction_rows": legacy_pred_rows,
                        "legacy_prediction_note": legacy_pred_note,
                        "legacy_checkpoint_loadable": legacy_ckpt_ok,
                        "legacy_checkpoint_note": legacy_ckpt_note,
                    }
                )
                if not args.execute or action == "complete":
                    continue

                if action == "train_predict_evaluate":
                    run_command(
                        [
                            python, str(root / "src/07_train_cxr_model_v2.py"),
                            "--manifest", str(manifest), "--subset-dir", str(subset_dir),
                            "--train-source", source, "--subset-n", size, "--seed", str(seed),
                            "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                            "--val-limit", str(args.val_limit), "--val-sampling-seed", str(args.val_sampling_seed),
                            "--num-workers", str(args.num_workers), "--use-pos-weight", "--out-dir", str(formal_model_dir),
                        ],
                        log_root / f"train_{experiment}.log", root,
                    )
                    selected_model_dir = formal_model_dir
                    action = "predict"

                if action in {"predict", "predict_current_checkpoint", "predict_legacy_checkpoint"}:
                    run_command(
                        [
                            python, str(root / "src/08_predict_cxr_model_v2.py"),
                            "--manifest", str(manifest), "--model-dir", str(selected_model_dir),
                            "--out-dir", str(formal_pred_dir), "--batch-size", str(args.prediction_batch_size),
                            "--num-workers", str(args.num_workers),
                        ],
                        log_root / f"predict_{experiment}.log", root,
                    )
                    selected_pred_path = formal_pred_dir / "predictions.csv"

                run_command(
                    [python, str(root / "src/09_evaluate_predictions_v2.py"), "--pred", str(selected_pred_path), "--out", str(metric_path)],
                    log_root / f"evaluate_{experiment}.log", root,
                )

                if args.with_calibration:
                    calib_dir = calibration_root / experiment
                    calib_metric = metric_root / f"{experiment}_calibrated_metrics.csv"
                    run_command(
                        [
                            python, str(root / "src/12_calibrate_and_evaluate.py"),
                            "--manifest", str(manifest), "--model-dir", str(selected_model_dir),
                            "--train-source", source, "--subset-n", size, "--seed", str(seed),
                            "--eval-datasets", "nih", "chexpert", "vindr", "--calib-limit", "0",
                            "--batch-size", str(args.prediction_batch_size), "--num-workers", str(args.num_workers),
                            "--out-dir", str(calib_dir), "--metrics-out", str(calib_metric),
                        ],
                        log_root / f"calibrate_{experiment}.log", root,
                    )

    plan_df = pd.DataFrame(plan)
    plan_path = run_root / "recovery_plan.csv"
    plan_df.to_csv(plan_path, index=False)
    print(plan_df.groupby("action").size().to_string())
    print("Saved plan:", plan_path)
    if args.execute:
        merge_metrics(metric_root)
        print("All 30 experiments completed and merged:", metric_root)
    else:
        print("Plan only. Re-run with --execute after reviewing the plan.")


if __name__ == "__main__":
    main()
