#!/usr/bin/env python3
"""CPU-only paired input/RNG equivalence audit; no GPU or model execution."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cxr_lossless_cache import LosslessResizeCache


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_trainer(path):
    spec = importlib.util.spec_from_file_location("cache_benchmark_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reset_rng():
    random.seed(20260903)
    np.random.seed(20260903)
    torch.manual_seed(20260903)


def rng_digest():
    digest = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    digest.update(repr(random.getstate()).encode())
    state = np.random.get_state()
    digest.update(state[0].encode())
    digest.update(state[1].tobytes())
    digest.update(repr(state[2:]).encode())
    return digest.hexdigest()


def tensor_digest(item):
    digest = hashlib.sha256()
    for key in ("image", "target", "mask"):
        tensor = item[key].contiguous()
        digest.update(str((key, tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/mnt/e/cxr_larger_not_better")
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-source", type=int, default=256)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES=-1; never benchmark beside GPU training")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(args.project_root)
    trainer = load_trainer(root / "src/07_train_cxr_model_v2.py")
    cache_path = Path(args.cache_path)
    if cache_path.exists():
        raise RuntimeError("Paired cold-cache benchmark requires a new cache path; do not overwrite")
    cache = LosslessResizeCache(cache_path, max_bytes=19 * 1024**3,
                                reserve_bytes=25 * 1024**3, space_root="/mnt/c")
    report = {"status": "FAIL", "cpu_only": True,
              "scope": "input pipeline only, not end-to-end GPU training speed",
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "cache_path": str(cache_path), "per_source": args.per_source,
              "cache_module_sha256": sha_file(root / "src/cxr_lossless_cache.py"),
              "trainer_sha256": sha_file(root / "src/07_train_cxr_model_v2.py"),
              "sources": [], "failures": []}
    try:
        test_run = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests",
             "-p", "test_*cache*.py", "-v"], cwd=root, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        report["cpu_tests"] = {"returncode": test_run.returncode, "output": test_run.stdout}
        if test_run.returncode:
            raise RuntimeError("Cache unit/integration tests failed")
        for source in ("nih", "chexpert"):
            path = root / f"data/processed/model_manifest_{source}_training_v2.csv"
            frame = pd.read_csv(path, low_memory=False)
            frame = frame[frame["split"] == "train"].sample(
                n=args.per_source, random_state=20260903).reset_index(drop=True)
            expected = None
            expected_rng = None
            entry = {"source": source, "rows": len(frame),
                     "manifest_sha256": sha_file(path), "modes": {}}
            for mode, selected_cache in (("original", None), ("cold", cache), ("warm", cache)):
                reset_rng()
                dataset = trainer.CXRDataset(frame, trainer.LABELS,
                    trainer.make_transforms(224, True), image_cache=selected_cache, img_size=224)
                digests = []
                started = time.perf_counter()
                for index in range(len(dataset)):
                    digests.append(tensor_digest(dataset[index]))
                    if (index + 1) % 64 == 0:
                        print(source, mode, index + 1, "of", len(dataset), flush=True)
                elapsed = time.perf_counter() - started
                state = rng_digest()
                if expected is None:
                    expected, expected_rng = digests, state
                matches = sum(a == b for a, b in zip(expected, digests))
                entry["modes"][mode] = {"elapsed_sec": elapsed,
                    "images_per_sec": len(dataset) / elapsed,
                    "matching_samples": matches, "rng_equal": state == expected_rng,
                    "input_digest": hashlib.sha256("".join(digests).encode()).hexdigest()}
                if matches != len(dataset) or state != expected_rng:
                    report["failures"].append(f"{source}:{mode}:input_or_rng_mismatch")
                print(json.dumps({"source": source, "mode": mode,
                                  **entry["modes"][mode]}), flush=True)
            entry["warm_input_speedup"] = (entry["modes"]["original"]["elapsed_sec"] /
                                           entry["modes"]["warm"]["elapsed_sec"])
            report["sources"].append(entry)
        report["cache_stats"] = cache.snapshot_stats()
        # Every sampled image must have actually been served from cache in warm mode.
        if report["cache_stats"].get("hits", 0) < 2 * args.per_source:
            report["failures"].append("insufficient_verified_cache_hits")
        report["status"] = "PASS" if not report["failures"] else "FAIL"
    except Exception as exc:
        report["failures"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        report["cache_stats"] = cache.snapshot_stats()
        cache.close()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        temp = out.with_suffix(out.suffix + ".tmp")
        temp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(temp, out)
        print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
