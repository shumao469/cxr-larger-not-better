"""CPU-only trajectory checks for the optional deterministic image cache.

Compare the preserved pre-cache trainer with the cache-aware trainer, including
the otherwise easy-to-miss global RNG consumption of validation iterators.
All generated images and cache databases live under a temporary directory.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


def load_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


ORIGINAL = load_module(
    "trainer_before_lossless_cache",
    PROJECT / "logs/acceleration_20260903/07_train_cxr_model_v2.before_cache.py",
)
UPDATED = load_module("trainer_with_lossless_cache", SOURCE / "07_train_cxr_model_v2.py")


def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_snapshot(generator):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "loader_generator": generator.get_state().clone(),
    }


class TrainingCacheIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Prevent this small CPU-only check from monopolizing host CPU threads.
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cxr-cache-integration-")
        self.root = Path(self.temporary.name)
        self.labels = ORIGINAL.LABELS
        records = []
        dimensions = [(251, 239), (224, 224), (389, 173), (131, 305), (1, 1), (77, 113), (231, 228), (301, 279)]
        for index, (width, height) in enumerate(dimensions):
            yy, xx = np.indices((height, width), dtype=np.uint32)
            pixels = np.stack(
                [
                    (xx * 7 + yy * 3 + index * 19) % 256,
                    (xx * 11 + yy * 17 + index * 23) % 256,
                    (xx * 5 + yy * 13 + index * 31) % 256,
                ],
                axis=-1,
            ).astype(np.uint8)
            source = self.root / f"source_{index}.png"
            Image.fromarray(pixels).save(source)
            values = [float(index % 2), float((index // 2) % 2), np.nan if index % 3 == 0 else 1.0, 0.0]
            records.append({"image_path_final": str(source), **dict(zip(self.labels, values))})
        self.training = pd.DataFrame(records[:5])
        self.validation = pd.DataFrame(records[5:])

    def tearDown(self):
        self.temporary.cleanup()

    def make_cache(self, filename):
        from cxr_lossless_cache import LosslessResizeCache

        return LosslessResizeCache(
            self.root / filename,
            max_bytes=64 * 1024 * 1024,
            reserve_bytes=0,
            space_root=self.root,
        )

    def collect_run(self, trainer, cache_name=None, warm=False):
        cache = None
        if warm:
            cache = self.make_cache(cache_name)
            for frame in (self.training, self.validation):
                for path in frame["image_path_final"]:
                    cache.load_rgb(path, 224)
            cache.close()

        # Include cache constructor effects in the equivalence check, because
        # production opens it after set_seed() and before creating its loaders.
        reset_rng(314159)
        if cache_name is not None:
            cache = self.make_cache(cache_name)
        extra = {"image_cache": cache, "img_size": 224} if trainer is UPDATED else {}
        generator = torch.Generator()
        generator.manual_seed(42)
        training = DataLoader(
            trainer.CXRDataset(self.training, self.labels, trainer.make_transforms(224, True), False, **extra),
            batch_size=3,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            worker_init_fn=trainer.seed_worker,
            generator=generator,
        )
        validation = DataLoader(
            trainer.CXRDataset(self.validation, self.labels, trainer.make_transforms(224, False), False, **extra),
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            worker_init_fn=trainer.seed_worker,
            # Deliberately no generator: preserve the production global RNG use.
        )
        trace = [("created", {}, rng_snapshot(generator))]
        try:
            for epoch in range(2):
                for phase, loader in (("train", training), ("val", validation)):
                    for batch_index, batch in enumerate(loader):
                        trace.append(
                            (
                                (epoch, phase, batch_index),
                                {key: value.clone() for key, value in batch.items()},
                                rng_snapshot(generator),
                            )
                        )
                if cache is not None:
                    cache.snapshot_stats()
                trace.append(((epoch, "end"), {}, rng_snapshot(generator)))
            if cache is not None:
                # Equality alone would be a false positive if every lookup
                # silently fell back to the uncached path.
                expected_hits = (2 if warm else 1) * (len(self.training) + len(self.validation))
                self.assertGreaterEqual(cache.snapshot_stats().get("hits", 0), expected_hits)
            return trace
        finally:
            if cache is not None:
                cache.close()

    def assert_trace_equal(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for (expected_label, expected_batch, expected_rng), (actual_label, actual_batch, actual_rng) in zip(expected, actual):
            self.assertEqual(expected_label, actual_label)
            self.assertEqual(expected_batch.keys(), actual_batch.keys())
            for key in expected_batch:
                self.assertTrue(torch.equal(expected_batch[key], actual_batch[key]), (expected_label, key))
            self.assertEqual(expected_rng["python"], actual_rng["python"], expected_label)
            for expected_part, actual_part in zip(expected_rng["numpy"], actual_rng["numpy"]):
                np.testing.assert_equal(expected_part, actual_part, err_msg=str(expected_label))
            for key in ("torch", "loader_generator"):
                self.assertTrue(torch.equal(expected_rng[key], actual_rng[key]), (expected_label, key))

    def test_disabled_cache_preserves_two_epoch_trajectory(self):
        self.assert_trace_equal(self.collect_run(ORIGINAL), self.collect_run(UPDATED))

    def test_cold_cache_preserves_two_epoch_trajectory(self):
        self.assert_trace_equal(self.collect_run(ORIGINAL), self.collect_run(UPDATED, "cold.sqlite3"))

    def test_warm_cache_preserves_two_epoch_trajectory(self):
        self.assert_trace_equal(self.collect_run(ORIGINAL), self.collect_run(UPDATED, "warm.sqlite3", warm=True))

    def test_missing_source_is_not_hidden_by_cached_image(self):
        cache = self.make_cache("missing.sqlite3")
        path = Path(self.training.iloc[0]["image_path_final"])
        try:
            cache.load_rgb(str(path), 224)
            path.unlink()
            dataset = UPDATED.CXRDataset(self.training, self.labels, UPDATED.make_transforms(224, True), False, image_cache=cache)
            with self.assertRaisesRegex(RuntimeError, "Cannot read training image"):
                dataset[0]
        finally:
            cache.close()


if __name__ == "__main__":
    unittest.main()
