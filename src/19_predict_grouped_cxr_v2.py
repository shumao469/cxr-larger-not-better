#!/usr/bin/env python3
"""Reuse each unchanged FP32 evaluation batch across up to five formal models.

No training or model parallelism is introduced. Dataset, transform, collate and
model construction come directly from 08_predict_cxr_model_v2.py. Production
waits for the final manifest and all 30 completed training experiments, then
holds the existing gpu_training.lock. A group is fully computed and checked
before any of its create-only prediction outputs are published.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


LABELS = ["y_Cardiomegaly", "y_Pleural_Effusion", "y_Atelectasis", "y_Consolidation"]
SOURCES = ("nih", "chexpert")
SIZES = ("1000", "5000", "10000", "50000", "all")
SEEDS = (1, 2, 3)
EVAL_DATASETS = ("nih", "chexpert", "vindr")
EVAL_COUNTS = {("nih", "internal_test"): 21297, ("chexpert", "internal_test"): 28986,
               ("vindr", "external_test"): 15000}
EXPECTED_ROWS = 65283
BATCH_SIZE = 128
IMG_SIZE = 224
META_COLUMNS = ["dataset", "split", "patient_id", "study_id", "image_id", "image_path_final"]
OUTPUT_NAMES = ("predictions.csv", "prediction_audit.json", "prediction_completed.json")
PROTOCOL = "08-equivalent-fp32-batch128-v1"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_reference(project):
    path = Path(project) / "src/08_predict_cxr_model_v2.py"
    spec = importlib.util.spec_from_file_location("cxr_single_prediction_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def experiments():
    return [(f"{source}_n{size}_seed{seed}", source, size, seed)
            for source in SOURCES for size in SIZES for seed in SEEDS]


def build_eval_frame(manifest, labels, *, expected_rows=EXPECTED_ROWS, expected_counts=EVAL_COUNTS):
    require(list(labels) == LABELS, "Checkpoint labels/order differ from the four frozen labels")
    # These are the same filtering and concatenation operations, in the same
    # dataset order, as the single-model reference main().
    parts = []
    for dataset in EVAL_DATASETS:
        split = "external_test" if dataset == "vindr" else "internal_test"
        part = manifest[(manifest["dataset"] == dataset) & (manifest["split"] == split)].copy()
        parts.append(part[part[labels].notna().any(axis=1)].copy())
    frame = pd.concat(parts, ignore_index=True)
    require(len(frame) == expected_rows, f"Evaluation cohort is {len(frame)}, expected {expected_rows}")
    require(not frame.duplicated(["dataset", "image_id"]).any(), "Duplicate evaluation dataset/image_id")
    require(frame.groupby(["dataset", "split"]).size().to_dict() == expected_counts,
            "Evaluation source/split counts differ")
    return frame


def shared_metadata(frame):
    result = frame[META_COLUMNS].copy().reset_index(drop=True)
    for column in META_COLUMNS[2:]:
        result[column] = result[column].astype(str)
    return result


def cohort_digest(metadata, truth):
    digest = hashlib.sha256(metadata.to_csv(index=False).encode("utf-8"))
    digest.update(np.asarray(truth, dtype=np.float32).tobytes())
    return digest.hexdigest()


def make_loader(reference, frame, labels, *, batch_size=BATCH_SIZE, device="cpu", cache=None):
    dataset = reference.CXRPredDataset(frame, labels, reference.transform(IMG_SIZE), False)
    if cache is not None:
        class CachedDataset(reference.CXRPredDataset):
            def __getitem__(self, index):
                row = self.df.iloc[index]
                path = str(row["image_path_final"])
                try:
                    image = cache.load_rgb(path, IMG_SIZE)
                except Exception as exc:
                    raise RuntimeError(f"Cannot read evaluation image: {path}") from exc
                image = self.transform(image)
                values = [row[label] if label in row else np.nan for label in self.labels]
                target = torch.tensor([0 if pd.isna(v) else float(v) for v in values], dtype=torch.float32)
                mask = torch.tensor([0 if pd.isna(v) else 1 for v in values], dtype=torch.float32)
                meta = {"dataset": row["dataset"], "split": row["split"],
                        "patient_id": str(row["patient_id"]), "study_id": str(row["study_id"]),
                        "image_id": str(row["image_id"]), "image_path_final": path}
                return image, target, mask, meta
        dataset = CachedDataset(frame, labels, reference.transform(IMG_SIZE), False)
    return DataLoader(dataset,
                      batch_size=batch_size, shuffle=False, num_workers=0,
                      pin_memory=(str(device).startswith("cuda")), collate_fn=reference.collate_fn)


def predict_group(models, loader, expected_metadata, expected_truth, *, device):
    """One tensor per batch, independent eval-mode forwards, compact matrices."""
    require(torch.get_default_dtype() == torch.float32, "FP32 default dtype required")
    require(not torch.is_autocast_enabled(), "Autocast must remain disabled")
    rows, n_labels = expected_truth.shape
    predictions = {name: np.empty((rows, n_labels), dtype=np.float32) for name in models}
    require(models, "Empty model group")
    for model in models.values():
        require(all(not layer.training for layer in model.modules()), "Every model/layer must be in eval mode")
    offset = 0
    with torch.no_grad():
        for images, targets, masks, metas in loader:
            count = len(metas)
            require(images.dtype == torch.float32 and count > 0 and offset + count <= rows,
                    "Invalid FP32 evaluation batch size or dtype")
            observed_metadata = pd.DataFrame(list(metas), columns=META_COLUMNS).reset_index(drop=True)
            pd.testing.assert_frame_equal(observed_metadata,
                                          expected_metadata.iloc[offset:offset + count].reset_index(drop=True),
                                          check_exact=True, check_dtype=True)
            observed_truth = np.where(masks.numpy() == 1, targets.numpy(), np.nan).astype(np.float32)
            np.testing.assert_array_equal(observed_truth, expected_truth[offset:offset + count])
            # This is the only device transfer. Each model sees exactly the same
            # full batch (including original dataset boundaries and final tail).
            device_images = images.to(device)
            for name, model in models.items():
                probabilities = torch.sigmoid(model(device_images)).cpu().numpy()
                require(probabilities.dtype == np.float32 and probabilities.shape == (count, n_labels),
                        f"Unexpected FP32 output shape/dtype: {name}")
                require(np.isfinite(probabilities).all() and (probabilities >= 0).all()
                        and (probabilities <= 1).all(), f"Invalid probabilities: {name}")
                predictions[name][offset:offset + count] = probabilities
            offset += count
            if offset == rows or offset % (BATCH_SIZE * 20) == 0:
                print(f"Grouped prediction {offset}/{rows}; models={len(models)}", flush=True)
    require(offset == rows, f"Incomplete grouped predictions: {offset}/{rows}")
    return predictions


def output_frame(metadata, truth, probabilities, config, labels=LABELS):
    frame = metadata.copy()
    for key in ("train_source", "subset_n", "seed"):
        frame[key] = config[key]
    for index, label in enumerate(labels):
        # The reference builds scalar rows; any missing truth promotes its
        # column to float64, whereas entirely observed columns stay float32.
        dtype = np.float64 if np.isnan(truth[:, index]).any() else np.float32
        frame[f"true_{label}"] = truth[:, index].astype(dtype)
        frame[f"pred_{label}"] = probabilities[:, index]
    return frame


def reference_output(reference, model, loader, config, labels=LABELS, *, device="cpu"):
    """Exact scalar-row FP32 loop used by 08, reserved for CPU equivalence QA."""
    rows = []
    with torch.no_grad():
        for images, targets, masks, metas in loader:
            probabilities = torch.sigmoid(model(images.to(device))).cpu().numpy()
            targets_np, masks_np = targets.numpy(), masks.numpy()
            for index, meta in enumerate(metas):
                row = dict(meta)
                row.update({key: config[key] for key in ("train_source", "subset_n", "seed")})
                for column, label in enumerate(labels):
                    row[f"true_{label}"] = targets_np[index, column] if masks_np[index, column] == 1 else np.nan
                    row[f"pred_{label}"] = probabilities[index, column]
                rows.append(row)
    return pd.DataFrame(rows)


def readiness_counts(project):
    processed = project / "data/processed"
    final_ready = all((processed / name).is_file() for name in
                      ("model_manifest_formal_v2.csv", "model_manifest_formal_v2_audit.json",
                       "model_manifest_formal_v2_audit_overlap.csv", "model_manifest_formal_v2_finalization.json"))
    if final_ready:
        final = read_json(processed / "model_manifest_formal_v2_finalization.json")
        require(final.get("status") == "PASS" and final.get("manifest_rows") == 274439
                and final.get("training_queue_modified") is False, "Final manifest PASS gate failed before GPU lock")
    counts = {source: 0 for source in SOURCES}
    for name, source, size, seed in experiments():
        marker = project / "results_formal_v2/models" / name / "completed.json"
        if marker.is_file():
            # Malformed/FAIL markers are errors, not indefinite silent waiting.
            evidence = read_json(marker)
            require(evidence.get("status") == "complete" and evidence.get("source") == source
                    and str(evidence.get("subset_n")) == size and evidence.get("seed") == seed
                    and evidence.get("epochs") == 5, f"Invalid training completion: {marker}")
            counts[source] += 1
    return final_ready, counts


@contextlib.contextmanager
def gpu_lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        print(f"Waiting for existing single-GPU lock: {path}", flush=True)
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(json.dumps({"pid": os.getpid(), "stage": "grouped_formal_prediction"}) + "\n")
            stream.flush()
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def audit_inputs(project, reference):
    processed = project / "data/processed"
    manifest_path = processed / "model_manifest_formal_v2.csv"
    completion_path = processed / "model_manifest_formal_v2_finalization.json"
    completion = read_json(completion_path)
    require(completion.get("status") == "PASS" and completion.get("manifest_rows") == 274439
            and completion.get("training_queue_modified") is False, "Final manifest completion gate failed")
    for source, expected in (("nih", 112120), ("chexpert", 147319)):
        entry = completion.get("source_comparison", {}).get(source, {})
        require(entry.get("rows") == expected and entry.get("all_columns_and_row_order_identical") is True,
                f"Final manifest does not preserve {source} training rows")
    input_hashes = {str(completion_path): sha256_file(completion_path)}
    for name in ("model_manifest_formal_v2.csv", "model_manifest_formal_v2_audit.json",
                 "model_manifest_formal_v2_audit_overlap.csv"):
        path = processed / name
        entry = completion.get("artifacts", {}).get(name, {})
        digest = sha256_file(path)
        require(Path(entry.get("path", "")).resolve() == path.resolve() and entry.get("sha256") == digest,
                f"Final artifact hash/path mismatch: {name}")
        input_hashes[str(path)] = digest
    require(not (processed / "model_manifest_formal_v2_missing.csv").exists(), "Final missing-image report exists")
    audit = read_json(processed / "model_manifest_formal_v2_audit.json")
    require(audit.get("output_rows") == audit.get("expected_output_rows") == 274439
            and audit.get("count_mismatches") == {} and audit.get("unexpected_dataset_splits") == {}
            and audit.get("duplicate_dataset_image_ids") == 0 and audit.get("patient_split_overlap_total") == 0
            and audit.get("all_paths_resolved") is True, "Final manifest audit failed")
    manifest = pd.read_csv(manifest_path, low_memory=False)
    require(len(manifest) == 274439, "Final manifest must contain 274439 rows")
    evaluation = build_eval_frame(manifest, LABELS)
    metadata = shared_metadata(evaluation)
    truth = evaluation[LABELS].to_numpy(dtype=np.float32)
    specs = []
    for name, source, size, seed in experiments():
        model_dir = project / "results_formal_v2/models" / name
        marker_path, checkpoint_path = model_dir / "completed.json", model_dir / "best.pt"
        marker = read_json(marker_path)
        require(marker.get("status") == "complete" and marker.get("source") == source
                and str(marker.get("subset_n")) == size and marker.get("seed") == seed
                and marker.get("epochs") == 5, f"Invalid completed training protocol: {name}")
        training_manifest = processed / f"model_manifest_{source}_training_v2.csv"
        subset_path = project / "data/subsets_formal_v2" / source / f"{name}.csv"
        for path, key in ((training_manifest, "manifest_sha256"), (subset_path, "subset_sha256"),
                          (checkpoint_path, "best_checkpoint_sha256"), (model_dir / "last.pt", "last_checkpoint_sha256")):
            digest = sha256_file(path)
            require(marker.get(key) == digest, f"Training evidence hash mismatch: {name}/{key}")
            input_hashes[str(path)] = digest
        require(completion.get("input_sha256", {}).get(str(training_manifest)) == input_hashes[str(training_manifest)],
                f"Training manifest differs from finalization evidence: {source}")
        history_path = model_dir / "history.csv"
        history = pd.read_csv(history_path)
        require(history["epoch"].astype(int).tolist() == [1, 2, 3, 4, 5], f"Incomplete training history: {name}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        require(isinstance(payload, dict) and "model_state" in payload and "config" in payload,
                f"Invalid checkpoint structure: {name}")
        config = payload["config"]
        require(list(payload.get("labels", [])) == LABELS and list(config.get("labels", [])) == LABELS,
                f"Checkpoint label order mismatch: {name}")
        require(config.get("train_source") == source and str(config.get("subset_n")) == size
                and config.get("seed") == seed and config.get("epochs") == 5
                and config.get("img_size") == IMG_SIZE, f"Checkpoint configuration mismatch: {name}")
        require(Path(config.get("manifest", "")).resolve() == training_manifest.resolve()
                and Path(config.get("subset_path", "")).resolve() == subset_path.resolve(),
                f"Checkpoint frozen input paths mismatch: {name}")
        other = build_eval_frame(manifest, payload["labels"])
        pd.testing.assert_frame_equal(evaluation, other, check_exact=True, check_dtype=True)
        input_hashes[str(marker_path)] = sha256_file(marker_path)
        input_hashes[str(history_path)] = sha256_file(history_path)
        specs.append({"experiment": name, "checkpoint": checkpoint_path,
                      "checkpoint_sha256": input_hashes[str(checkpoint_path)],
                      "config": {key: config[key] for key in ("train_source", "subset_n", "seed")}})
        del payload, other
    reference_path = project / "src/08_predict_cxr_model_v2.py"
    input_hashes[str(reference_path)] = sha256_file(reference_path)
    return evaluation, metadata, truth, specs, input_hashes


def validate_prediction_frame(frame, metadata, truth, config, labels=LABELS):
    require(len(frame) == len(metadata), "Prediction row count mismatch")
    expected_columns = META_COLUMNS + ["train_source", "subset_n", "seed"]
    for label in labels:
        expected_columns.extend((f"true_{label}", f"pred_{label}"))
    require(list(frame.columns) == expected_columns, "Prediction columns or label order mismatch")
    for column in META_COLUMNS:
        require(frame[column].astype(str).tolist() == metadata[column].astype(str).tolist(),
                f"Prediction metadata/order mismatch: {column}")
    for column in ("train_source", "subset_n", "seed"):
        require((frame[column].astype(str) == str(config[column])).all(), f"Prediction model metadata mismatch: {column}")
    require(not frame.duplicated(["dataset", "image_id"]).any(), "Duplicate prediction image IDs")
    for index, label in enumerate(labels):
        actual_truth = pd.to_numeric(frame[f"true_{label}"], errors="raise").to_numpy(dtype=np.float32)
        np.testing.assert_array_equal(actual_truth, truth[:, index])
        values = pd.to_numeric(frame[f"pred_{label}"], errors="raise").to_numpy(dtype=np.float32)
        require(np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all(),
                f"Invalid prediction values: {label}")


def reusable_prediction(directory, spec, metadata, truth, manifest_hash, digest, reference_hash=None):
    if not directory.exists():
        return False
    require(directory.is_dir() and not directory.is_symlink(), f"Invalid prediction directory: {directory}")
    contents = list(directory.iterdir())
    if not contents:
        return False
    require({path.name for path in contents} == set(OUTPUT_NAMES),
            f"Existing outputs are partial or unexpected; refusing overwrite: {directory}")
    require(all(path.is_file() and not path.is_symlink() for path in contents), "Prediction artifacts must be regular files")
    marker = read_json(directory / OUTPUT_NAMES[2])
    audit = read_json(directory / OUTPUT_NAMES[1])
    require(marker.get("status") == "complete" and marker.get("protocol") == PROTOCOL
            and marker.get("experiment") == spec["experiment"], f"Existing prediction completion is invalid: {directory}")
    for key, expected in (("manifest_sha256", manifest_hash), ("cohort_sha256", digest),
                          ("checkpoint_sha256", spec["checkpoint_sha256"]), ("prediction_rows", len(metadata))):
        require(marker.get(key) == audit.get(key) == expected, f"Existing prediction evidence mismatch: {key}")
    require(audit.get("labels") == LABELS and audit.get("counts") ==
            {f"{a}/{b}": int(n) for (a, b), n in metadata.groupby(["dataset", "split"]).size().items()},
            "Existing prediction labels/counts mismatch")
    require(audit.get("precision") == "float32" and audit.get("batch_size") == BATCH_SIZE
            and audit.get("img_size") == IMG_SIZE and audit.get("protocol") == PROTOCOL,
            "Existing prediction numerical protocol mismatch")
    require(audit.get("cpu_equivalence", {}).get("status") == "PASS"
            and audit.get("runtime_equivalence", {}).get("status") == "PASS"
            and audit.get("runtime_equivalence", {}).get("exact_probabilities") is True,
            "Existing prediction equivalence gates missing or failed")
    require(Path(audit.get("checkpoint", "")).resolve() == spec["checkpoint"].resolve(),
            "Existing checkpoint path mismatch")
    if reference_hash is not None:
        require(audit.get("reference_script_sha256") == reference_hash, "Existing reference version mismatch")
    for name in OUTPUT_NAMES[:2]:
        require(marker.get("output_sha256", {}).get(name) == sha256_file(directory / name),
                f"Existing prediction checksum mismatch: {name}")
    frame = pd.read_csv(directory / OUTPUT_NAMES[0], keep_default_na=False,
                        dtype={column: str for column in META_COLUMNS + ["train_source", "subset_n", "seed"]})
    # Empty true cells are missing labels, as written by the reference CSV.
    for label in LABELS:
        frame[f"true_{label}"] = frame[f"true_{label}"].replace("", np.nan)
    validate_prediction_frame(frame, metadata, truth, spec["config"])
    return True


def prepare_prediction(stage, frame, audit, spec, metadata, truth):
    """Prepare and audit the complete disk roundtrip before exposing any result."""
    stage.mkdir(exist_ok=False)
    csv_path = stage / OUTPUT_NAMES[0]
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)
        stream.flush()
        os.fsync(stream.fileno())
    write_new_json(stage / OUTPUT_NAMES[1], audit)
    marker = {key: audit[key] for key in ("experiment", "manifest_sha256", "cohort_sha256", "checkpoint_sha256", "prediction_rows")}
    marker.update({"status": "complete", "protocol": PROTOCOL,
                   "completed_at": datetime.now(timezone.utc).isoformat(),
                   "output_sha256": {name: sha256_file(stage / name) for name in OUTPUT_NAMES[:2]}})
    write_new_json(stage / OUTPUT_NAMES[2], marker)
    require(reusable_prediction(stage, spec, metadata, truth, audit["manifest_sha256"], audit["cohort_sha256"],
                                audit["reference_script_sha256"]), "Staged prediction roundtrip failed")


def publish_prediction(stage, directory):
    if directory.exists():
        require(directory.is_dir() and not directory.is_symlink() and not list(directory.iterdir()),
                f"Refusing to replace existing predictions: {directory}")
    else:
        directory.mkdir(parents=True, exist_ok=False)
    # Completion marker is the commit point. Interrupted/partial publication is
    # preserved and rejected on a later run, never overwritten or silently used.
    for name in OUTPUT_NAMES:
        os.link(stage / name, directory / name)


def gate_sample_indices(evaluation):
    """Select intact original batches, including both source boundaries/tail."""
    require(len(evaluation) > 0, "Empty runtime equivalence cohort")
    starts = {0, ((len(evaluation) - 1) // BATCH_SIZE) * BATCH_SIZE}
    datasets = evaluation["dataset"].astype(str).to_numpy()
    starts.update((index // BATCH_SIZE) * BATCH_SIZE
                  for index in np.flatnonzero(datasets[1:] != datasets[:-1]) + 1)
    return [index for start in sorted(starts) for index in range(start, min(start + BATCH_SIZE, len(evaluation)))]


def runtime_equivalence(reference, models, evaluation, configs, *, device, cache=None):
    """Run only after readiness/lock, before any full prediction or publication."""
    indices = gate_sample_indices(evaluation)
    sample = evaluation.iloc[indices].reset_index(drop=True)
    metadata = shared_metadata(sample)
    truth = sample[LABELS].to_numpy(dtype=np.float32)
    if cache is not None:
        # Exercise both cold and warm cache; compare every transformed FP32
        # pixel as well as labels/metas. Original 08 remains the reference.
        for _ in range(2):
            baseline = make_loader(reference, sample, LABELS, device=device)
            cached = make_loader(reference, sample, LABELS, device=device, cache=cache)
            for original, accelerated in zip(baseline, cached, strict=True):
                for left, right in zip(original[:3], accelerated[:3]):
                    require(torch.equal(left, right), "Cached input tensor/target/mask differs from original 08")
                require(original[3] == accelerated[3], "Cached metadata differs from original 08")
    got = predict_group(models, make_loader(reference, sample, LABELS, device=device, cache=cache),
                        metadata, truth, device=device)
    for name, model in models.items():
        expected = reference_output(reference, model, make_loader(reference, sample, LABELS, device=device),
                                    configs[name], device=device)
        pd.testing.assert_frame_equal(output_frame(metadata, truth, got[name], configs[name]), expected,
                                      check_exact=True, check_dtype=True)
    return {"status": "PASS", "device": device, "sample_rows": len(indices), "original_row_indices": indices,
            "batch_size": BATCH_SIZE, "tail_batch": len(evaluation) % BATCH_SIZE or BATCH_SIZE,
            "models": list(models), "exact_probabilities": True, "cache_enabled": cache is not None}


def open_image_cache(project, requested_path):
    """Optional acceleration with the same validated config and host limits as 07."""
    info = {"enabled": False, "reason": "not_requested"}
    if requested_path is None:
        return None, info, {}
    try:
        config_path = project / "config/lossless_cache_formal_v2.json"
        settings = read_json(config_path)
        require(settings.get("enabled") is True, "Cache configuration is not enabled")
        require(Path(settings["cache_path"]).resolve() == requested_path.resolve(), "Requested cache path differs from validated config")
        require(0 < int(settings["max_bytes"]) <= 19 * 1024**3
                and int(settings["reserve_bytes"]) >= 25 * 1024**3
                and settings["space_root"] == "/mnt/c", "Cache host capacity limits differ")
        report_path = Path(settings["validation_report"])
        report_hash = sha256_file(report_path)
        require(report_hash == settings["validation_report_sha256"], "Cache benchmark hash mismatch")
        report = read_json(report_path)
        require(report.get("status") == "PASS" and report.get("cpu_only") is True, "Cache benchmark did not pass")
        module_path = project / "src/cxr_lossless_cache.py"
        module_hash = sha256_file(module_path)
        require(report.get("cache_module_sha256") == module_hash, "Validated cache module changed")
        from cxr_lossless_cache import LosslessResizeCache
        cache = LosslessResizeCache(requested_path, max_bytes=int(settings["max_bytes"]),
                                   reserve_bytes=int(settings["reserve_bytes"]), space_root=settings["space_root"])
        hashes = {str(config_path): sha256_file(config_path), str(report_path): report_hash, str(module_path): module_hash}
        info = {"enabled": True, "settings": settings, "evidence_sha256": hashes}
        print(f"Validated lossless image cache enabled: {requested_path}", flush=True)
        return cache, info, hashes
    except Exception as exc:
        info["reason"] = f"fallback_to_original: {type(exc).__name__}: {exc}"
        print(f"Image cache unavailable; using original images: {info['reason']}", flush=True)
        return None, info, {}


def self_test(project):
    """CPU-only exact comparison including a full 128 batch and a one-row tail."""
    reference = load_reference(project)
    from PIL import Image

    class TinyEvalModel(nn.Module):
        def __init__(self, seed):
            super().__init__()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.norm = nn.BatchNorm1d(3)
                self.dropout = nn.Dropout(0.4)
                self.head = nn.Linear(3, 4)

        def forward(self, images):
            return self.head(self.dropout(self.norm(images.mean(dim=(2, 3)))))

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with tempfile.TemporaryDirectory(prefix="cxr-grouped-self-test-") as temporary:
            root = Path(temporary)
            for index in range(3):
                yy, xx = np.indices((229 + index, 237 + index))
                array = np.stack(((xx + index * 9) % 255, (yy * 2) % 255, (xx + yy) % 255), axis=-1).astype(np.uint8)
                Image.fromarray(array).save(root / f"source_{index}.png")
            rows = []
            for index in range(129):
                dataset = "nih" if index < 65 else "chexpert" if index < 128 else "vindr"
                row = {"dataset": dataset, "split": "external_test" if dataset == "vindr" else "internal_test",
                       "patient_id": str(index), "study_id": str(index), "image_id": f"image_{index}",
                       "image_path_final": str(root / f"source_{index % 3}.png")}
                row.update({label: np.nan if index % 9 == 0 and column == 1 else float((index + column) % 2)
                            for column, label in enumerate(LABELS)})
                rows.append(row)
            evaluation = pd.DataFrame(rows)
            metadata, truth = shared_metadata(evaluation), evaluation[LABELS].to_numpy(dtype=np.float32)
            models = {f"fixture_{seed}": TinyEvalModel(seed).eval() for seed in (11, 22, 33)}
            before = {name: {key: tensor.clone() for key, tensor in model.state_dict().items()} for name, model in models.items()}
            grouped = predict_group(models, make_loader(reference, evaluation, LABELS), metadata, truth, device="cpu")
            for name, model in models.items():
                config = {"train_source": "nih", "subset_n": "1000", "seed": 1}
                expected = reference_output(reference, model, make_loader(reference, evaluation, LABELS), config)
                actual = output_frame(metadata, truth, grouped[name], config)
                pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_dtype=True)
                for key, tensor in model.state_dict().items():
                    require(torch.equal(tensor, before[name][key]), "Model/BatchNorm state changed during inference")
            # Also exercise the actual DenseNet121 constructor and unchanged
            # 224 transform on a small CPU batch plus tail, not just a toy net.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(44)
                actual_model = reference.make_model(4).eval()
            small = evaluation.iloc[[0, 65, 128]].reset_index(drop=True)
            small_meta, small_truth = shared_metadata(small), small[LABELS].to_numpy(dtype=np.float32)
            got = predict_group({"densenet121": actual_model}, make_loader(reference, small, LABELS, batch_size=2),
                                small_meta, small_truth, device="cpu")["densenet121"]
            expected = reference_output(reference, actual_model, make_loader(reference, small, LABELS, batch_size=2), config)
            pd.testing.assert_frame_equal(output_frame(small_meta, small_truth, got, config), expected,
                                          check_exact=True, check_dtype=True)
        return {"status": "PASS", "device": "cpu", "reference": "08_predict_cxr_model_v2.py",
                "full_batch": 128, "rows": 129, "dataset_boundary": 65, "tail_batch": 1,
                "fixture_models": 3, "actual_densenet121_rows": 3,
                "exact_dataframe_and_probabilities": True, "model_and_batchnorm_unchanged": True}
    finally:
        torch.set_num_threads(previous_threads)


def execute(project, group_size, wait, poll_seconds=60, cache_path=None):
    require(1 <= group_size <= 5, "Group size must be between 1 and 5")
    require(10 <= poll_seconds <= 3600, "Poll seconds must be between 10 and 3600")
    test_result = self_test(project)
    while True:
        final_ready, counts = readiness_counts(project)
        if final_ready and counts == {"nih": 15, "chexpert": 15}:
            break
        require(wait, f"Not ready: final_manifest={final_ready}, training={counts}")
        print(f"Waiting for audited final manifest and all training: final={final_ready}, models={counts}", flush=True)
        time.sleep(poll_seconds)
    run_root = project / "results_formal_v2"
    with gpu_lock(run_root / "gpu_training.lock"):
        final_ready, counts = readiness_counts(project)
        require(final_ready and counts == {"nih": 15, "chexpert": 15}, "Readiness changed while waiting for GPU lock")
        reference = load_reference(project)
        evaluation, metadata, truth, specs, hashes = audit_inputs(project, reference)
        manifest_path = project / "data/processed/model_manifest_formal_v2.csv"
        manifest_hash = hashes[str(manifest_path)]
        digest = cohort_digest(metadata, truth)
        reference_hash = hashes[str(project / "src/08_predict_cxr_model_v2.py")]
        prediction_root = run_root / "predictions"
        pending = [spec for spec in specs if not reusable_prediction(prediction_root / spec["experiment"], spec,
                                                                    metadata, truth, manifest_hash, digest, reference_hash)]
        if not pending:
            print("All 30 prediction outputs passed reuse audits; no GPU inference needed", flush=True)
            return
        require(torch.cuda.is_available(), "CUDA unavailable; refusing silent CPU production prediction")
        device = "cuda"
        prediction_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".grouped_staging_", dir=prediction_root))
        write_new_json(staging / "cpu_equivalence.json", test_result)
        cache, cache_info, cache_hashes = open_image_cache(project, cache_path)
        hashes.update(cache_hashes)
        try:
            for begin in range(0, len(pending), group_size):
                group = pending[begin:begin + group_size]
                models = {}
                for spec in group:
                    require(sha256_file(spec["checkpoint"]) == spec["checkpoint_sha256"], "Checkpoint changed before load")
                    payload = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
                    require(list(payload["labels"]) == LABELS, "Checkpoint label order changed")
                    model = reference.make_model(len(LABELS))
                    model.load_state_dict(payload["model_state"], strict=True)
                    models[spec["experiment"]] = model.to(device).eval()
                    del payload, model
                gate = runtime_equivalence(reference, models, evaluation,
                                           {spec["experiment"]: spec["config"] for spec in group}, device=device, cache=cache)
                probabilities = predict_group(models, make_loader(reference, evaluation, LABELS, device=device, cache=cache),
                                              metadata, truth, device=device)
                # No publication if any group output or frozen input changed.
                for path, expected in hashes.items():
                    require(sha256_file(path) == expected, f"Frozen input changed during inference: {path}")
                for spec in group:
                    frame = output_frame(metadata, truth, probabilities[spec["experiment"]], spec["config"])
                    validate_prediction_frame(frame, metadata, truth, spec["config"])
                    audit = {"experiment": spec["experiment"], "checkpoint": str(spec["checkpoint"].resolve()),
                             "checkpoint_sha256": spec["checkpoint_sha256"], "prediction_rows": len(frame),
                             "counts": {f"{a}/{b}": int(n) for (a, b), n in frame.groupby(["dataset", "split"]).size().items()},
                             "labels": LABELS, "manifest_sha256": manifest_hash, "cohort_sha256": digest,
                             "precision": "float32", "batch_size": BATCH_SIZE, "img_size": IMG_SIZE,
                             "group_size": len(group), "protocol": PROTOCOL, "cpu_equivalence": test_result,
                             "runtime_equivalence": gate, "reference_script_sha256": reference_hash,
                             "cache_configuration": cache_info,
                             "cache": cache.snapshot_stats() if cache is not None else None,
                             "cache_script_sha256": hashes.get(str(project / "src/cxr_lossless_cache.py"))}
                    prepare_prediction(staging / spec["experiment"], frame, audit, spec, metadata, truth)
                # Every member's disk roundtrip passed before the first publish.
                for spec in group:
                    publish_prediction(staging / spec["experiment"], prediction_root / spec["experiment"])
                    print(f"Published complete audited predictions: {spec['experiment']}", flush=True)
                del models, probabilities
        finally:
            if cache is not None:
                cache.close()
        print("All 30 formal predictions completed; metrics/plots remain separate stages", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--group-size", type=int, default=5)
    waiting = parser.add_mutually_exclusive_group()
    waiting.add_argument("--wait-ready", action="store_true", help="Wait without the GPU lock for all 30 models and final manifest")
    waiting.add_argument("--no-wait", action="store_true", help="Fail immediately instead of waiting (also the default)")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--cache-path", type=Path, help="Optional bounded lossless RGB/Resize cache; disabled by default")
    args = parser.parse_args(argv)
    project = args.project_root.resolve()
    try:
        if args.self_test:
            print(json.dumps(self_test(project)), flush=True)
        elif args.execute:
            execute(project, args.group_size, args.wait_ready, args.poll_seconds, args.cache_path)
        else:
            final_ready, counts = readiness_counts(project)
            print(json.dumps({"mode": "readiness_only", "final_manifest_files_present": final_ready,
                              "completed_training_markers": counts, "gpu_work_started": False}), flush=True)
        return 0
    except Exception as exc:
        print(f"Grouped prediction FAILED; no outputs were overwritten: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
