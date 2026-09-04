#!/usr/bin/env python3
"""Train formal NIH and/or CheXpert checkpoints with audited resumability."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch


ALL_SOURCES = ["nih", "chexpert"]
SIZES = ["1000", "5000", "10000", "50000", "all"]
SEEDS = [1, 2, 3]
LABELS = [
    "y_Cardiomegaly",
    "y_Pleural_Effusion",
    "y_Atelectasis",
    "y_Consolidation",
]
EXPECTED_COUNTS = {
    ("chexpert", "train"): 103_286,
    ("chexpert", "val"): 15_047,
    ("chexpert", "internal_test"): 28_986,
    ("nih", "train"): 79_481,
    ("nih", "val"): 11_342,
    ("nih", "internal_test"): 21_297,
}


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, path)


def atomic_csv(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def load_checkpoint(path: Path):
    if not path.exists() or path.stat().st_size < 1_000_000:
        raise RuntimeError(f"Checkpoint is missing or too small: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_state", "labels", "epoch", "val_mean_auc", "config"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise RuntimeError(f"Checkpoint keys are incomplete: {path}")
    return payload


def audit_checkpoint_config(
    payload,
    *,
    source: str,
    size: str,
    seed: int,
    epochs: int,
    manifest: Path,
    subset_path: Path,
):
    config = payload["config"]
    expected = {
        "train_source": source,
        "subset_n": size,
        "seed": seed,
        "epochs": epochs,
    }
    for key, value in expected.items():
        if str(config.get(key)) != str(value):
            raise RuntimeError(
                f"Checkpoint config mismatch for {key}: {config.get(key)!r} != {value!r}"
            )
    if list(payload["labels"]) != LABELS or list(config.get("labels", [])) != LABELS:
        raise RuntimeError("Checkpoint label order mismatch")
    if Path(config.get("manifest", "")).resolve() != manifest.resolve():
        raise RuntimeError("Checkpoint manifest path mismatch")
    if Path(config.get("subset_path", "")).resolve() != subset_path.resolve():
        raise RuntimeError("Checkpoint subset path mismatch")


def audit_finished_experiment(
    model_dir: Path,
    *,
    source: str,
    size: str,
    seed: int,
    epochs: int,
    manifest: Path,
    subset_path: Path,
    manifest_sha256: str,
    subset_sha256: str,
    require_marker: bool,
):
    history_path = model_dir / "history.csv"
    if not history_path.exists():
        raise RuntimeError("history_missing")
    history = pd.read_csv(history_path)
    if history["epoch"].astype(int).tolist() != list(range(1, epochs + 1)):
        raise RuntimeError("history_epoch_sequence_mismatch")

    last = load_checkpoint(model_dir / "last.pt")
    best = load_checkpoint(model_dir / "best.pt")
    if int(last["epoch"]) != epochs:
        raise RuntimeError("last_checkpoint_epoch_mismatch")
    for payload in (last, best):
        audit_checkpoint_config(
            payload,
            source=source,
            size=size,
            seed=seed,
            epochs=epochs,
            manifest=manifest,
            subset_path=subset_path,
        )

    finite_history = history[pd.to_numeric(history["val_mean_auc"], errors="coerce").notna()]
    if finite_history.empty:
        raise RuntimeError("no_finite_validation_auc")
    history_best = float(finite_history["val_mean_auc"].max())
    checkpoint_best = float(best["val_mean_auc"])
    if not math.isclose(history_best, checkpoint_best, rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError("best_checkpoint_auc_mismatch")

    marker_path = model_dir / "completed.json"
    if require_marker:
        if not marker_path.exists():
            raise RuntimeError("completion_marker_missing")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        expected_marker = {
            "source": source,
            "subset_n": size,
            "seed": seed,
            "epochs": epochs,
            "manifest_sha256": manifest_sha256,
            "subset_sha256": subset_sha256,
        }
        for key, value in expected_marker.items():
            if str(marker.get(key)) != str(value):
                raise RuntimeError(f"completion_marker_mismatch:{key}")
    return {
        "source": source,
        "subset_n": size,
        "seed": seed,
        "epochs": epochs,
        "manifest": str(manifest),
        "subset_path": str(subset_path),
        "manifest_sha256": manifest_sha256,
        "subset_sha256": subset_sha256,
        "best_checkpoint_sha256": sha256_file(model_dir / "best.pt"),
        "last_checkpoint_sha256": sha256_file(model_dir / "last.pt"),
        "best_val_mean_auc": checkpoint_best,
        "status": "complete",
    }


def run_command(command, log_path: Path, cwd: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN:", " ".join(str(item) for item in command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND: " + " ".join(str(item) for item in command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed with code {result.returncode}; see {log_path}")


def audit_manifest(manifest_path: Path, sources: list[str]):
    frame = pd.read_csv(manifest_path, low_memory=False)
    selected_counts = {
        key: value for key, value in EXPECTED_COUNTS.items() if key[0] in sources
    }
    expected_rows = sum(selected_counts.values())
    if len(frame) != expected_rows:
        raise RuntimeError(
            f"Training manifest row mismatch: {len(frame):,} != {expected_rows:,}"
        )
    counts = frame.groupby(["dataset", "split"]).size().to_dict()
    if counts != selected_counts:
        raise RuntimeError(f"Training manifest count mismatch: {counts}")
    if int(frame.duplicated(["dataset", "image_id"]).sum()):
        raise RuntimeError("Training manifest has duplicate dataset/image IDs")
    missing = int(
        (~frame["image_path_final"].map(lambda value: Path(str(value)).is_file())).sum()
    )
    if missing:
        raise RuntimeError(f"Training manifest has {missing:,} missing images")


def acquire_gpu_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    print(f"Waiting for the single-GPU lock: {lock_path}", flush=True)
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "argv": sys.argv}) + "\n")
    handle.flush()
    print("Acquired the single-GPU lock", flush=True)
    return handle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--sources", nargs="+", choices=ALL_SOURCES, default=ALL_SOURCES
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=3000)
    parser.add_argument("--val-sampling-seed", type=int, default=2026)
    args = parser.parse_args()

    sources = list(dict.fromkeys(args.sources))
    slug = "_".join(sources)
    root = Path(args.project_root).resolve()
    manifest = (root / args.manifest).resolve()
    audit_manifest(manifest, sources)
    manifest_sha256 = sha256_file(manifest)

    python = sys.executable
    run_root = root / "results_formal_v2"
    model_root = run_root / "models"
    log_root = run_root / "logs"
    model_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    source_subset_dirs = {}
    for source in sources:
        subset_dir = root / "data" / "subsets_formal_v2" / source
        source_subset_dirs[source] = subset_dir
        required_subsets = [
            subset_dir / f"{source}_n{size}_seed{seed}.csv"
            for size in SIZES
            for seed in SEEDS
        ]
        if not all(path.exists() and path.stat().st_size > 0 for path in required_subsets):
            run_command(
                [
                    python,
                    str(root / "src/06_make_sample_size_subsets.py"),
                    "--manifest", str(manifest),
                    "--out-dir", str(subset_dir),
                    "--train-sources", source,
                    "--seeds", "1", "2", "3",
                ],
                log_root / f"00_make_subsets_{source}.log",
                root,
            )
        if not all(path.exists() and path.stat().st_size > 0 for path in required_subsets):
            raise RuntimeError(f"Subset generation incomplete for {source}")

    gpu_lock = acquire_gpu_lock(run_root / "gpu_training.lock")
    try:
        plan = []
        plan_path = run_root / f"training_plan_{slug}.csv"
        for source in sources:
            subset_dir = source_subset_dirs[source]
            for size in SIZES:
                for seed in SEEDS:
                    experiment = f"{source}_n{size}_seed{seed}"
                    model_dir = model_root / experiment
                    subset_path = subset_dir / f"{source}_n{size}_seed{seed}.csv"
                    subset_sha256 = sha256_file(subset_path)
                    note = "completion_marker_missing"
                    try:
                        audit_finished_experiment(
                            model_dir,
                            source=source,
                            size=size,
                            seed=seed,
                            epochs=args.epochs,
                            manifest=manifest,
                            subset_path=subset_path,
                            manifest_sha256=manifest_sha256,
                            subset_sha256=subset_sha256,
                            require_marker=True,
                        )
                        complete = True
                        note = "complete"
                    except Exception as exc:
                        complete = False
                        note = str(exc)

                    plan.append(
                        {
                            "experiment": experiment,
                            "train_source": source,
                            "subset_n": size,
                            "seed": seed,
                            "action": "complete" if complete else "train",
                            "checkpoint_note": note,
                        }
                    )
                    atomic_csv(pd.DataFrame(plan), plan_path)
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
                    marker = audit_finished_experiment(
                        model_dir,
                        source=source,
                        size=size,
                        seed=seed,
                        epochs=args.epochs,
                        manifest=manifest,
                        subset_path=subset_path,
                        manifest_sha256=manifest_sha256,
                        subset_sha256=subset_sha256,
                        require_marker=False,
                    )
                    atomic_json(marker, model_dir / "completed.json")
                    audit_finished_experiment(
                        model_dir,
                        source=source,
                        size=size,
                        seed=seed,
                        epochs=args.epochs,
                        manifest=manifest,
                        subset_path=subset_path,
                        manifest_sha256=manifest_sha256,
                        subset_sha256=subset_sha256,
                        require_marker=True,
                    )
                    plan[-1]["action"] = "complete"
                    plan[-1]["checkpoint_note"] = "complete"
                    atomic_csv(pd.DataFrame(plan), plan_path)

        completed = sum(row["action"] == "complete" for row in plan)
        expected_models = len(sources) * len(SIZES) * len(SEEDS)
        summary = {
            "status": "complete" if completed == expected_models else "incomplete",
            "sources": sources,
            "completed_models": completed,
            "expected_models": expected_models,
            "manifest": str(manifest),
            "manifest_sha256": manifest_sha256,
            "prediction_and_external_evaluation_deferred": True,
        }
        atomic_json(summary, run_root / f"training_only_summary_{slug}.json")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_UN)
        gpu_lock.close()


if __name__ == "__main__":
    main()
