"""Bounded, fail-open-to-source cache for the existing PIL RGB/Resize prefix.

This module never substitutes blank images and never seeds or draws from the
Python, NumPy, or torch RNG. Cache misses return the original RGB image; hits
return the losslessly verified resized RGB image. Keep the original Resize and
all downstream transforms in the caller, including their iterator/RNG behavior.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import threading

from PIL import Image, ImageFile
from torchvision.transforms import InterpolationMode, Resize


GIB = 1024 ** 3
WRITE_MARGIN = 128 * 1024
DB_VERSION = 1
_VERSIONS = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "Pillow")}
_COLUMNS = (
    "cache_key", "source_path", "img_size", "schema_sha256", "source_size",
    "source_mtime_ns", "source_inode", "source_device", "blob_sha256",
    "pixel_sha256", "png_blob",
)
_STAT_KEYS = ("requests", "hits", "misses", "writes", "source_changes",
              "corrupt_entries", "capacity_skips", "cache_errors",
              "fallback_reads", "source_read_errors")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_stamp(path: str) -> tuple[int, int, int, int]:
    info = os.stat(path)
    # Reuse this stat result; load_rgb still checks identity before and after
    # every hit. Avoid a redundant second trip to the mounted source volume.
    if not stat.S_ISREG(info.st_mode):
        raise OSError(f"Source is not a regular file: {path}")
    return info.st_size, info.st_mtime_ns, info.st_ino, info.st_dev


def _schema(img_size: int) -> tuple[str, str]:
    specification = {
        "schema": "cxr-pil-rgb-resize-lossless-v1", "versions": _VERSIONS,
        "decode": "PIL.Image.open(path).convert('RGB')", "exif_transpose": False,
        "load_truncated_images": bool(ImageFile.LOAD_TRUNCATED_IMAGES),
        "resize": {"height": img_size, "width": img_size, "backend": "PIL",
                   "interpolation": "BILINEAR", "antialias": True},
        "storage": "single-frame RGB uint8 PNG; no random augmentation",
    }
    encoded = json.dumps(specification, sort_keys=True, separators=(",", ":"))
    return encoded, _sha(encoded.encode("utf-8"))


def _decode_cached(blob: bytes, blob_hash: str, pixel_hash: str, img_size: int) -> Image.Image:
    if _sha(blob) != blob_hash:
        raise ValueError("Cached PNG byte hash mismatch")
    with Image.open(io.BytesIO(blob)) as probe:
        if probe.format != "PNG" or probe.mode != "RGB" or probe.size != (img_size, img_size) or probe.n_frames != 1:
            raise ValueError("Cache must be a single-frame RGB PNG at the exact requested size")
        probe.verify()
    with Image.open(io.BytesIO(blob)) as decoded:
        decoded.load()
        image = decoded.copy()
    if _sha(image.tobytes()) != pixel_hash:
        raise ValueError("Cached RGB prefix pixel hash mismatch")
    return image


class LosslessResizeCache:
    """SQLite cache whose failures only disable acceleration, never source reads.

    Production defaults cap the whole SQLite bundle at 20 GiB and require at
    least 25 GiB remaining on the actual Windows C: volume (/mnt/c), rather than
    trusting the virtual free space reported by WSL's ext4 filesystem.
    """

    def __init__(self, cache_path, *, max_bytes=20 * GIB, reserve_bytes=25 * GIB,
                 space_root="/mnt/c", busy_timeout_ms=200):
        self.cache_path = Path(cache_path).absolute()
        self.max_bytes = max(0, int(max_bytes))
        self.reserve_bytes = max(0, int(reserve_bytes))
        self.space_root = Path(space_root)
        self.busy_timeout_ms = max(0, int(busy_timeout_ms))
        self._connection = None
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._stats = dict.fromkeys(_STAT_KEYS, 0)
        self._last_error = None

    def __getstate__(self):
        result = dict(self.__dict__)
        result["_connection"] = None
        result["_lock"] = None
        return result

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._connection = None
        self._pid = os.getpid()
        self._lock = threading.RLock()

    def _error(self, exc):
        self._stats["cache_errors"] += 1
        self._last_error = f"{type(exc).__name__}: {exc}"

    def _bundle_bytes(self):
        total = 0
        for suffix in ("", "-journal", "-wal", "-shm"):
            path = Path(str(self.cache_path) + suffix)
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                pass
        return total

    def _capacity(self, additional):
        if self._bundle_bytes() + additional > self.max_bytes:
            return False
        ancestor = self.cache_path.parent
        while not ancestor.exists():
            ancestor = ancestor.parent
        # The host volume is the binding constraint for an expanding ext4.vhdx.
        free = min(shutil.disk_usage(self.space_root).free, shutil.disk_usage(ancestor).free)
        return free - additional >= self.reserve_bytes

    def _db(self):
        if self._pid != os.getpid():
            if self._connection is not None:
                self._connection.close()
            self._connection = None
            self._pid = os.getpid()
        if self._connection is not None:
            return self._connection
        existing = self.cache_path.exists()
        if not existing and not self._capacity(2 * WRITE_MARGIN):
            self._stats["capacity_skips"] += 1
            return None
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.cache_path), timeout=self.busy_timeout_ms / 1000,
                                     isolation_level=None, check_same_thread=False)
        try:
            connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
            # Rollback journals avoid unbounded WAL growth. One writer is
            # serialized with BEGIN IMMEDIATE; independent processes may read.
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if version == 0 and not tables:
                if not self._capacity(2 * WRITE_MARGIN):
                    connection.execute("ROLLBACK")
                    connection.close()
                    self._stats["capacity_skips"] += 1
                    return None
                connection.execute("""CREATE TABLE cache_entries (
                    cache_key TEXT PRIMARY KEY, source_path TEXT NOT NULL,
                    img_size INTEGER NOT NULL, schema_sha256 TEXT NOT NULL,
                    source_size INTEGER NOT NULL, source_mtime_ns INTEGER NOT NULL,
                    source_inode INTEGER NOT NULL, source_device INTEGER NOT NULL,
                    blob_sha256 TEXT NOT NULL, pixel_sha256 TEXT NOT NULL,
                    png_blob BLOB NOT NULL)""")
                connection.execute("CREATE TABLE cache_schemas (schema_sha256 TEXT PRIMARY KEY, specification TEXT NOT NULL)")
                connection.execute(f"PRAGMA user_version={DB_VERSION}")
            elif version != DB_VERSION:
                raise ValueError(f"Unsupported cache DB version {version}; preserving existing cache")
            columns = tuple(row[1] for row in connection.execute("PRAGMA table_info(cache_entries)"))
            if columns != _COLUMNS:
                raise ValueError("Unexpected cache table schema; preserving existing cache")
            schema_columns = tuple(row[1] for row in connection.execute("PRAGMA table_info(cache_schemas)"))
            if schema_columns != ("schema_sha256", "specification"):
                raise ValueError("Unexpected cache schema-metadata table; preserving existing cache")
            connection.execute("COMMIT")
            self._connection = connection
            return connection
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            connection.close()
            raise

    def _lookup(self, path, img_size, schema_hash, stamp):
        db = self._db()
        if db is None:
            return None
        key = _sha(json.dumps([path, img_size, schema_hash], ensure_ascii=False).encode("utf-8"))
        row = db.execute("SELECT source_path,img_size,schema_sha256,source_size,source_mtime_ns,"
                         "source_inode,source_device,blob_sha256,pixel_sha256,png_blob "
                         "FROM cache_entries WHERE cache_key=?", (key,)).fetchone()
        if row is None:
            return None
        if row[:3] != (path, img_size, schema_hash):
            self._stats["corrupt_entries"] += 1
            raise ValueError("Cache identity metadata mismatch")
        if tuple(row[3:7]) != stamp:
            self._stats["source_changes"] += 1
            return None
        try:
            return _decode_cached(row[9], row[7], row[8], img_size)
        except Exception:
            self._stats["corrupt_entries"] += 1
            raise

    def _store(self, path, img_size, schema_text, schema_hash, stamp, original):
        db = self._db()
        if db is None:
            return
        if not self._capacity(2 * WRITE_MARGIN):
            self._stats["capacity_skips"] += 1
            return
        prefix = Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR, antialias=True)(original)
        pixels = prefix.tobytes()
        memory = io.BytesIO()
        prefix.save(memory, format="PNG")
        blob = memory.getvalue()
        blob_hash, pixel_hash = _sha(blob), _sha(pixels)
        roundtrip = _decode_cached(blob, blob_hash, pixel_hash, img_size)
        if roundtrip.tobytes() != pixels:
            raise ValueError("PNG encoding did not preserve every RGB prefix pixel")
        if _source_stamp(path) != stamp:
            self._stats["source_changes"] += 1
            return
        key = _sha(json.dumps([path, img_size, schema_hash], ensure_ascii=False).encode("utf-8"))
        db.execute("BEGIN IMMEDIATE")
        try:
            old = db.execute("SELECT length(png_blob) FROM cache_entries WHERE cache_key=?", (key,)).fetchone()
            # Cover DB growth, old-row rollback journal, and B-tree overhead.
            additional = 2 * len(blob) + (old[0] if old else 0) + 2 * WRITE_MARGIN
            if not self._capacity(additional):
                db.execute("ROLLBACK")
                self._stats["capacity_skips"] += 1
                return
            page_size = db.execute("PRAGMA page_size").fetchone()[0]
            # Leave headroom for journal pages as well as the database file.
            page_limit = max(1, (self.max_bytes - additional) // page_size)
            db.execute(f"PRAGMA max_page_count={page_limit}")
            db.execute("INSERT OR IGNORE INTO cache_schemas VALUES (?,?)", (schema_hash, schema_text))
            db.execute("INSERT OR REPLACE INTO cache_entries VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (key, path, img_size, schema_hash, *stamp, blob_hash, pixel_hash, sqlite3.Binary(blob)))
            stored = db.execute("SELECT png_blob,blob_sha256,pixel_sha256 FROM cache_entries WHERE cache_key=?", (key,)).fetchone()
            if _decode_cached(stored[0], stored[1], stored[2], img_size).tobytes() != pixels:
                raise ValueError("SQLite roundtrip changed cached prefix pixels")
            if _source_stamp(path) != stamp:
                db.execute("ROLLBACK")
                self._stats["source_changes"] += 1
                return
            db.execute("COMMIT")
            self._stats["writes"] += 1
        except Exception:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise

    def load_rgb(self, source_path, img_size=224):
        """Return source RGB on miss, validated resized RGB on hit.

        Original-image exceptions propagate to the caller's existing strict
        image-read handler. Cache errors never cause a fabricated image.
        """
        if type(img_size) is not int or img_size <= 0:
            raise ValueError("img_size must be a positive integer")
        path = os.path.abspath(os.fspath(source_path))
        schema_text, schema_hash = _schema(img_size)
        with self._lock:
            self._stats["requests"] += 1
            try:
                initial_stamp = _source_stamp(path)
            except Exception:
                self._stats["source_read_errors"] += 1
                raise
            try:
                cached = self._lookup(path, img_size, schema_hash, initial_stamp)
                if cached is not None and _source_stamp(path) == initial_stamp:
                    self._stats["hits"] += 1
                    return cached
            except Exception as exc:
                self._error(exc)
            self._stats["misses"] += 1
            self._stats["fallback_reads"] += 1
            try:
                with Image.open(path) as handle:
                    original = handle.convert("RGB")
            except Exception:
                self._stats["source_read_errors"] += 1
                raise
            try:
                if _source_stamp(path) == initial_stamp:
                    self._store(path, img_size, schema_text, schema_hash, initial_stamp, original)
                else:
                    self._stats["source_changes"] += 1
            except Exception as exc:
                self._error(exc)
            return original

    def snapshot_stats(self):
        with self._lock:
            try:
                bundle_bytes = self._bundle_bytes()
            except OSError:
                bundle_bytes = None
            try:
                host_free_bytes = shutil.disk_usage(self.space_root).free
            except OSError:
                host_free_bytes = None
            return {**self._stats, "cache_path": str(self.cache_path),
                    "max_bytes": self.max_bytes, "reserve_bytes": self.reserve_bytes,
                    "space_root": str(self.space_root), "last_cache_error": self._last_error,
                    "bundle_bytes": bundle_bytes, "host_free_bytes": host_free_bytes}

    def close(self):
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
