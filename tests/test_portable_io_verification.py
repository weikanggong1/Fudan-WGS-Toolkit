"""Portable metadata IO contracts; generated fixtures are not benchmarks."""
from collections import OrderedDict
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pytest

from fudan_wgs_toolkit.cache_runtime import portable as metadata_io
from test_cache_portable import portable


@pytest.fixture(autouse=True)
def isolated_digest_cache(monkeypatch):
    monkeypatch.setattr(metadata_io, "_SHA_CACHE", OrderedDict())
    monkeypatch.setattr(metadata_io, "_SHA_METRICS", dict(calls=0, identity_hits=0,
        content_hashes=0, content_bytes=0, content_hash_seconds=0.0))
    # Most tests exercise stable immutable metadata, without wall-clock sleeps.
    now = time.time_ns()
    monkeypatch.setattr(metadata_io.time, "time_ns", lambda: now + 10_000_000_000)


def test_checksum_hashes_complete_bytes_once_and_reuses_exact_file_identity(tmp_path):
    path = tmp_path/"metadata.bin"
    payload = b"verified annotations" * 1000
    path.write_bytes(payload)
    assert metadata_io._sha(path) == hashlib.sha256(payload).hexdigest()
    first = dict(metadata_io._SHA_METRICS)
    for _ in range(4):
        assert metadata_io._sha(path) == hashlib.sha256(payload).hexdigest()
    assert first["content_hashes"] == 1 and first["content_bytes"] == len(payload)
    assert metadata_io._SHA_METRICS["content_hashes"] == 1
    assert metadata_io._SHA_METRICS["identity_hits"] == 4


@pytest.mark.parametrize("change", ["rewrite", "replace", "touch"])
def test_changed_identity_cannot_reuse_the_previous_checksum(tmp_path, change, monkeypatch):
    if change == "rewrite":
        # Include coarse filesystems where a same-tick rewrite and restored
        # mtime retain the entire stat identity. Recent bytes must be rehashed.
        monkeypatch.setattr(metadata_io, "_SHA_CACHE_SETTLE_NS", 60_000_000_000)
    path = tmp_path/"metadata.bin"
    first, second = b"first exact bytes", b"second exact data"
    path.write_bytes(first)
    before = path.stat()
    metadata_io._sha(path)
    if change == "rewrite":
        path.write_bytes(second)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif change == "replace":
        replacement = tmp_path/"replacement"
        replacement.write_bytes(second)
        replacement.replace(path)
    else:
        second = first
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns+1_000_000))
    assert metadata_io._sha(path) == hashlib.sha256(second).hexdigest()
    assert metadata_io._SHA_METRICS["content_hashes"] == 2
    assert metadata_io._SHA_METRICS["identity_hits"] == 0


def test_recent_identical_stat_identity_rehashes_changed_bytes(tmp_path, monkeypatch):
    path = tmp_path / "coarse-clock.bin"
    path.write_bytes(b"first exact bytes")
    identity = metadata_io._identity(path)
    monkeypatch.setattr(metadata_io.time, "time_ns", lambda: max(identity[3:]) + 1)
    monkeypatch.setattr(metadata_io, "_identity", lambda _: identity)
    monkeypatch.setattr(metadata_io, "_stat_identity", lambda _: identity)
    metadata_io._sha(path)
    path.write_bytes(b"second exact data")
    assert metadata_io._sha(path) == hashlib.sha256(b"second exact data").hexdigest()
    assert metadata_io._SHA_METRICS["content_hashes"] == 2
    assert metadata_io._SHA_METRICS["identity_hits"] == 0


@pytest.mark.parametrize("shape,dtype", [((0,), "<f8"), ((3, 0), "<i8"),
    ((17,), "<u8"), ((7, 3), ">f4"), ((9,), "|b1")])
def test_metadata_finalization_copies_bounded_bytes_without_mapping(tmp_path, monkeypatch, shape, dtype):
    values = np.arange(int(np.prod(shape))).reshape(shape).astype(dtype)
    writer = metadata_io._ArrayWriter(tmp_path, "bounded", dtype, shape[1:])
    writer.append(values)
    def reject_mapping(*args, **kwargs):
        raise AssertionError("metadata finalization must not map full chromosome fields")
    monkeypatch.setattr(np.lib.format, "open_memmap", reject_mapping)
    monkeypatch.setattr(np, "memmap", reject_mapping)
    specification = writer.finish()
    observed = np.load(tmp_path / specification["file"], allow_pickle=False)
    assert observed.dtype == values.dtype
    np.testing.assert_array_equal(observed, values)
    assert not writer.raw_path.exists()


def test_checksum_never_caches_bytes_changed_during_the_initial_scan(tmp_path, monkeypatch):
    path = tmp_path/"metadata.bin"
    path.write_bytes(b"original exact bytes")
    original_hash = hashlib.sha256
    class MutatingDigest:
        def __init__(self):
            self.digest = original_hash()
        def update(self, payload):
            self.digest.update(payload)
            path.write_bytes(b"changed exact bytes of another size")
        def hexdigest(self):
            return self.digest.hexdigest()
    monkeypatch.setattr(metadata_io.hashlib, "sha256", MutatingDigest)
    with pytest.raises(ValueError, match="changed during checksum"):
        metadata_io._sha(path)
    assert str(path.resolve()) not in metadata_io._SHA_CACHE


def test_checksum_cache_retains_a_bounded_number_of_file_identities(tmp_path, monkeypatch):
    monkeypatch.setattr(metadata_io, "_SHA_CACHE_LIMIT", 2)
    paths = [tmp_path/f"field_{i}" for i in range(3)]
    for index, path in enumerate(paths):
        path.write_bytes(bytes([index]))
        metadata_io._sha(path)
    assert len(metadata_io._SHA_CACHE) == 2
    assert str(paths[0].resolve()) not in metadata_io._SHA_CACHE
    metadata_io._sha(paths[0])
    assert metadata_io._SHA_METRICS["content_hashes"] == 4


def test_new_reader_keeps_first_content_verification_and_reuses_unchanged_annotations(portable):
    native, cache, metadata, _, _, _ = portable
    # Export was a separate preparation step; require this reader's complete
    # cold verification before comparing the second reader's content IO.
    metadata_io._SHA_CACHE.clear()
    field = "annotation/info/unicode"
    with metadata_io.PortableMetadataReader(metadata, cache) as first:
        np.testing.assert_array_equal(first.read_field(field), native.fields[field])
        report = first.reader_metadata
        assert report["metadata_checksum_process_delta"]["content_hashes"] > 0
    before = dict(metadata_io._SHA_METRICS)
    with metadata_io.PortableMetadataReader(metadata, cache) as second:
        np.testing.assert_array_equal(second.read_field(field), native.fields[field])
        report = second.reader_metadata
        assert report["metadata_checksums_enabled"]
        assert report["metadata_checked_files"] > 0
        assert report["metadata_checksum_process_delta"]["content_hashes"] == 0
    assert metadata_io._SHA_METRICS["content_hashes"] == before["content_hashes"]
    assert metadata_io._SHA_METRICS["identity_hits"] > before["identity_hits"]


def test_warm_checksum_cache_does_not_accept_changed_annotation_contents(portable):
    _, cache, metadata, manifest, _, _ = portable
    field = "annotation/info/weight"
    with metadata_io.PortableMetadataReader(metadata, cache) as reader:
        reader.read_field(field)
    path = metadata/manifest["fields"][field]["file"]
    value = np.load(path, allow_pickle=False)
    np.save(path, value+1)
    with metadata_io.PortableMetadataReader(metadata, cache) as reader:
        with pytest.raises(ValueError, match="checksum"):
            reader.read_field(field)


def test_field_change_between_checksum_and_mmap_cannot_enter_reader_cache(portable, monkeypatch):
    _, cache, metadata, manifest, _, _ = portable
    field = metadata/manifest["fields"]["position"]["file"]
    original_map = metadata_io.readonly_npy_mmap
    reader = metadata_io.PortableMetadataReader(metadata, cache)
    def changed_map(path):
        if path == field:
            values = np.load(path, allow_pickle=False)
            np.save(path, values+1)
        return original_map(path)
    monkeypatch.setattr(metadata_io, "readonly_npy_mmap", changed_map)
    with pytest.raises(ValueError, match="changed"):
        reader.read_field("position")
    assert field.name not in reader._arrays
    with pytest.raises(ValueError, match="changed"):
        reader.close()
    assert reader._closed
