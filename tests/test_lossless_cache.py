"""CPU-only fixtures for cache integrity, RNG equivalence, and fail-open policy."""

import concurrent.futures
import hashlib
import importlib.util
import os
from pathlib import Path
import random
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

SPEC = importlib.util.spec_from_file_location("cxr_lossless_cache", Path(__file__).resolve().parents[1] / "src/cxr_lossless_cache.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cxr-cache-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.png"
        pixels = (np.arange(117 * 93 * 3, dtype=np.uint32) % 251).astype(np.uint8).reshape(117, 93, 3)
        Image.fromarray(pixels).save(self.source)
        self.cache = self.make_cache()
        self.addCleanup(self.cache.close)

    def make_cache(self, **overrides):
        args = {"max_bytes": 8 * 1024 ** 2, "reserve_bytes": 0, "space_root": self.root}
        args.update(overrides)
        return MODULE.LosslessResizeCache(self.root / "cache/cache.sqlite3", **args)

    def assert_rng_equal(self, left, right):
        self.assertEqual(left[0], right[0])
        self.assertEqual(left[1][0], right[1][0])
        np.testing.assert_array_equal(left[1][1], right[1][1])
        self.assertEqual(left[1][2:], right[1][2:])
        self.assertTrue(torch.equal(left[2], right[2]))

    def transform(self, size=224):
        return transforms.Compose([transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(5),
            transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

    def test_miss_hit_pixels_rng_and_augmented_tensors_are_exact(self):
        before_source = self.source.read_bytes()
        before = rng_state()
        miss = self.cache.load_rgb(self.source, 224)
        self.assert_rng_equal(before, rng_state())
        self.assertEqual(miss.size, (93, 117))
        hit = self.cache.load_rgb(self.source, 224)
        self.assert_rng_equal(before, rng_state())
        self.assertEqual(hit.size, (224, 224))
        expected_prefix = transforms.Resize((224,224))(miss)
        self.assertEqual(expected_prefix.tobytes(), hit.tobytes())
        transform = self.transform()
        for seed in (1, 2, 3):
            state = torch.Generator(device="cpu").manual_seed(seed).get_state()
            torch.set_rng_state(state)
            original_tensor = transform(miss)
            original_after = rng_state()
            torch.set_rng_state(state)
            cached_tensor = transform(hit)
            self.assertTrue(torch.equal(original_tensor, cached_tensor))
            self.assert_rng_equal(original_after, rng_state())
        self.assertEqual(self.source.read_bytes(), before_source)
        stats = self.cache.snapshot_stats()
        self.assertEqual((stats["misses"],stats["hits"],stats["writes"]), (1,1,1))

    def test_changed_source_is_a_miss(self):
        self.cache.load_rgb(self.source)
        Image.new("RGB", (93,117), (40,60,80)).save(self.source)
        current = self.cache.load_rgb(self.source)
        self.assertEqual(current.size, (93,117))
        self.assertEqual(current.getpixel((0,0)), (40,60,80))
        self.assertEqual(self.cache.snapshot_stats()["source_changes"], 1)
        self.assertEqual(self.cache.load_rgb(self.source).getpixel((0,0)), (40,60,80))

    def test_mtime_change_alone_is_a_miss(self):
        self.cache.load_rgb(self.source)
        info = self.source.stat()
        os.utime(self.source, ns=(info.st_atime_ns, info.st_mtime_ns + 1000000))
        self.assertEqual(self.cache.load_rgb(self.source).size, (93,117))
        self.assertEqual(self.cache.snapshot_stats()["source_changes"], 1)

    def test_source_stamp_uses_one_stat_and_rejects_nonregular_files(self):
        expected = self.source.stat()
        original_stat = MODULE.os.stat
        with patch.object(MODULE.os, "stat", wraps=original_stat) as observed:
            actual = MODULE._source_stamp(str(self.source))
            self.assertEqual(observed.call_count, 1)
        self.assertEqual(actual, (expected.st_size, expected.st_mtime_ns,
                                  expected.st_ino, expected.st_dev))
        with patch.object(MODULE.os, "stat", wraps=original_stat) as observed:
            with self.assertRaisesRegex(OSError, "not a regular file"):
                MODULE._source_stamp(str(self.root))
            self.assertEqual(observed.call_count, 1)

    def test_warm_hit_keeps_exactly_two_source_identity_checks(self):
        self.cache.load_rgb(self.source)
        original_stat = MODULE.os.stat
        with patch.object(MODULE.os, "stat", wraps=original_stat) as observed:
            self.assertEqual(self.cache.load_rgb(self.source).size, (224, 224))
            self.assertEqual(observed.call_count, 2)
        self.assertEqual(self.cache.snapshot_stats()["hits"], 1)

    def test_source_change_after_cache_lookup_is_not_returned_as_a_hit(self):
        self.cache.load_rgb(self.source)
        original_lookup = self.cache._lookup

        def replace_source_after_lookup(*args):
            cached = original_lookup(*args)
            Image.new("RGB", (93, 117), (40, 60, 80)).save(self.source)
            return cached

        with patch.object(self.cache, "_lookup", side_effect=replace_source_after_lookup):
            result = self.cache.load_rgb(self.source)
        self.assertEqual(result.size, (93, 117))
        self.assertEqual(result.getpixel((0, 0)), (40, 60, 80))
        self.assertEqual(self.cache.snapshot_stats()["hits"], 0)
        self.assertEqual(self.cache.snapshot_stats()["source_changes"], 1)

    def test_corrupt_blob_and_pixel_hash_fall_back_to_source(self):
        for sql in ("UPDATE cache_entries SET png_blob=x'00010203'", "UPDATE cache_entries SET pixel_sha256='wrong'"):
            with self.subTest(sql=sql):
                self.cache.load_rgb(self.source)
                self.cache._connection.execute(sql)
                image = self.cache.load_rgb(self.source)
                self.assertEqual(image.size, (93,117))
        self.assertEqual(self.cache.snapshot_stats()["corrupt_entries"], 2)

    def test_corrupt_database_is_preserved_and_source_returned(self):
        self.cache.cache_path.parent.mkdir()
        self.cache.cache_path.write_bytes(b"not a sqlite database")
        before = self.cache.cache_path.read_bytes()
        self.assertEqual(self.cache.load_rgb(self.source).size, (93,117))
        self.assertGreater(self.cache.snapshot_stats()["cache_errors"], 0)
        self.assertEqual(self.cache.cache_path.read_bytes(), before)

    def test_tiny_limit_never_creates_cache(self):
        cache = self.make_cache(max_bytes=1)
        self.addCleanup(cache.close)
        before = rng_state()
        self.assertEqual(cache.load_rgb(self.source).size, (93,117))
        self.assertFalse(cache.cache_path.exists())
        self.assertGreater(cache.snapshot_stats()["capacity_skips"], 0)
        self.assert_rng_equal(before, rng_state())

    def test_existing_bundle_near_limit_skips_new_rows(self):
        self.cache.load_rgb(self.source)
        initial_bytes = self.cache._bundle_bytes()
        self.cache.max_bytes = initial_bytes + MODULE.WRITE_MARGIN
        second = self.root / "second.png"
        Image.new("RGB", (70,80)).save(second)
        self.assertEqual(self.cache.load_rgb(second).size, (70,80))
        self.assertEqual(self.cache._bundle_bytes(), initial_bytes)
        self.assertLessEqual(self.cache._bundle_bytes(), self.cache.max_bytes)
        self.assertGreater(self.cache.snapshot_stats()["capacity_skips"], 0)

    def test_source_change_while_sqlite_row_written_is_not_committed(self):
        original_decode = MODULE._decode_cached
        calls = 0

        def change_source_after_database_roundtrip(*args):
            nonlocal calls
            result = original_decode(*args)
            calls += 1
            if calls == 2:
                info = self.source.stat()
                os.utime(self.source, ns=(info.st_atime_ns, info.st_mtime_ns + 1000000))
            return result

        with patch.object(MODULE, "_decode_cached", side_effect=change_source_after_database_roundtrip):
            self.assertEqual(self.cache.load_rgb(self.source).size, (93,117))
        count = self.cache._connection.execute("SELECT count(*) FROM cache_entries").fetchone()[0]
        self.assertEqual(count, 0)
        self.assertEqual(self.cache.snapshot_stats()["writes"], 0)
        self.assertEqual(self.cache.snapshot_stats()["source_changes"], 1)

    def test_wrong_schema_metadata_table_falls_back(self):
        self.cache.load_rgb(self.source)
        self.cache._connection.execute("ALTER TABLE cache_schemas RENAME COLUMN specification TO wrong_name")
        self.cache.close()
        self.assertEqual(self.cache.load_rgb(self.source).size, (93,117))
        self.assertGreater(self.cache.snapshot_stats()["cache_errors"], 0)

    def test_stats_include_bundle_and_host_space(self):
        self.cache.load_rgb(self.source)
        stats = self.cache.snapshot_stats()
        self.assertGreater(stats["bundle_bytes"], 0)
        self.assertGreater(stats["host_free_bytes"], 0)
        with patch.object(MODULE.shutil, "disk_usage", side_effect=OSError("unavailable")):
            self.assertIsNone(self.cache.snapshot_stats()["host_free_bytes"])

    def test_low_host_space_disables_writes_but_preserves_reads(self):
        self.cache.load_rgb(self.source)
        usage = type("Usage", (), {"free": 0})()
        with patch.object(MODULE.shutil, "disk_usage", return_value=usage):
            self.assertEqual(self.cache.load_rgb(self.source).size, (224,224))
            new_source = self.root / "second.png"
            Image.new("RGB", (70,80)).save(new_source)
            self.assertEqual(self.cache.load_rgb(new_source).size, (70,80))
        self.assertGreater(self.cache.snapshot_stats()["capacity_skips"], 0)

    def test_disk_errors_only_disable_cache(self):
        with patch.object(MODULE.shutil, "disk_usage", side_effect=OSError("unavailable volume")):
            self.assertEqual(self.cache.load_rgb(self.source).size, (93,117))
        self.assertGreater(self.cache.snapshot_stats()["cache_errors"], 0)

    def test_cache_write_failure_falls_back_without_blank(self):
        with patch.object(self.cache, "_store", side_effect=sqlite3.OperationalError("database full")):
            result = self.cache.load_rgb(self.source)
        with Image.open(self.source) as source:
            self.assertEqual(result.tobytes(), source.convert("RGB").tobytes())
        self.assertEqual(self.cache.snapshot_stats()["cache_errors"], 1)

    def test_invalid_original_raises_even_if_prior_cache_exists(self):
        self.cache.load_rgb(self.source)
        self.source.write_bytes(b"invalid image")
        with self.assertRaises(Exception):
            self.cache.load_rgb(self.source)
        self.assertEqual(self.cache.snapshot_stats()["source_read_errors"], 1)

    def test_different_sizes_and_schema_versions_have_distinct_keys(self):
        self.cache.load_rgb(self.source, 32)
        self.cache.load_rgb(self.source, 64)
        self.assertEqual(self.cache.load_rgb(self.source, 32).size, (32,32))
        self.assertEqual(self.cache.load_rgb(self.source, 64).size, (64,64))
        with patch.dict(MODULE._VERSIONS, {"Pillow": "test-other-version"}):
            self.assertEqual(self.cache.load_rgb(self.source, 32).size, (93,117))
        count = self.cache._connection.execute("SELECT count(*) FROM cache_entries").fetchone()[0]
        self.assertEqual(count, 3)

    def test_missing_source_does_not_use_stale_cache(self):
        self.cache.load_rgb(self.source)
        self.source.unlink()
        with self.assertRaises(FileNotFoundError):
            self.cache.load_rgb(self.source)

    def test_multiple_connections_are_safe_and_busy_writer_falls_back(self):
        self.cache.load_rgb(self.source)
        second = self.make_cache()
        self.addCleanup(second.close)
        self.assertEqual(second.load_rgb(self.source).size, (224,224))
        second._connection.execute("BEGIN IMMEDIATE")
        try:
            another = self.root / "third.png"
            Image.new("RGB", (70,80)).save(another)
            self.assertEqual(self.cache.load_rgb(another).size, (70,80))
            self.assertGreater(self.cache.snapshot_stats()["cache_errors"], 0)
        finally:
            second._connection.execute("ROLLBACK")

    def test_same_instance_threaded_reads_are_serialized(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            dimensions = list(pool.map(lambda _: self.cache.load_rgb(self.source).size, range(8)))
        self.assertEqual(dimensions.count((93,117)), 1)
        self.assertEqual(dimensions.count((224,224)), 7)
        self.assertLessEqual(self.cache._bundle_bytes(), self.cache.max_bytes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
