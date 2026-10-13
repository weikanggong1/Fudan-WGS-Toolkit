"""Read-only NPY geometry, layout and bounded-header IO contracts."""
from pathlib import Path

import numpy as np
import pytest

from fudan_wgs_toolkit import npy_io


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
@pytest.mark.parametrize("order", ["C", "F"])
def test_unbuffered_npy_mapping_preserves_layout_and_values(tmp_path, version, order):
    path = tmp_path/"field.npy"
    expected = np.array(np.arange(100).reshape(10, 10), dtype=np.float64, order=order)
    with path.open("wb") as stream:
        np.lib.format.write_array(stream, expected, version=version, allow_pickle=False)
    actual = npy_io.readonly_npy_mmap(path)
    np.testing.assert_array_equal(actual, expected)
    assert isinstance(actual, np.memmap) and not actual.flags.writeable
    assert actual.strides == expected.strides
    with pytest.raises(ValueError):
        actual.setflags(write=True)


def test_unbuffered_npy_header_reads_no_array_payload(tmp_path, monkeypatch):
    path = tmp_path/"field.npy"
    np.save(path, np.arange(500000, dtype=np.float64))
    identity = npy_io._identity(path.stat())
    original_open = Path.open
    observed = []
    class CountedStream:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def read(self, count):
            payload = self.stream.read(count)
            observed.append(len(payload))
            return payload
        def __getattr__(self, name):
            return getattr(self.stream, name)
    def tracked_open(current, *args, **kwargs):
        if current == path:
            assert kwargs.get("buffering") == 0
            return CountedStream(original_open(current, *args, **kwargs))
        return original_open(current, *args, **kwargs)
    monkeypatch.setattr(Path, "open", tracked_open)
    actual = npy_io.readonly_npy_mmap(path)
    assert sum(observed) < 1024 and actual.shape == (500000,)
    assert identity[2] > 3*2**20


def test_unbuffered_mapping_rejects_short_shape_and_object_pickle(tmp_path):
    path = tmp_path/"field.npy"
    np.save(path, np.arange(5))
    with path.open("r+b") as stream:
        stream.truncate(path.stat().st_size-1)
    with pytest.raises(ValueError, match="shorter"):
        npy_io.readonly_npy_mmap(path)
    np.save(path, np.asarray(["object arrays unsupported"], dtype=object))
    with pytest.raises(ValueError, match="pickle"):
        npy_io.readonly_npy_mmap(path)


@pytest.mark.parametrize("expected", [np.asarray(7, dtype=np.int32),
    np.empty((0, 3), dtype=np.float32),
    np.asarray([(2, 3.)], dtype=[("integer", "<i4"), ("floating", "<f8")])])
def test_scalar_empty_and_structured_maps_preserve_standard_geometry(tmp_path, expected):
    path = tmp_path/"field.npy"
    np.save(path, expected, allow_pickle=False)
    actual = npy_io.readonly_npy_mmap(path)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert not actual.flags.writeable


@pytest.mark.parametrize("change", ["inplace", "replace"])
def test_header_read_rejects_changed_file_or_replaced_path(tmp_path, monkeypatch, change):
    path = tmp_path/"field.npy"
    np.save(path, np.arange(10))
    original_header = np.lib.format._read_array_header
    def changed_header(stream, version):
        geometry = original_header(stream, version)
        if change == "inplace":
            np.save(path, np.arange(11))
        else:
            replacement = tmp_path/"replacement.npy"
            np.save(replacement, np.arange(10)+1)
            replacement.replace(path)
        return geometry
    monkeypatch.setattr(np.lib.format, "_read_array_header", changed_header)
    with pytest.raises(ValueError, match="identity changed"):
        npy_io.readonly_npy_mmap(path)


def test_map_construction_rejects_changed_identity(tmp_path, monkeypatch):
    path = tmp_path/"field.npy"
    np.save(path, np.arange(10))
    original_map = np.memmap
    def changed_map(*args, **kwargs):
        result = original_map(*args, **kwargs)
        np.save(path, np.arange(11))
        return result
    monkeypatch.setattr(np, "memmap", changed_map)
    with pytest.raises(ValueError, match="map construction"):
        npy_io.readonly_npy_mmap(path)


def test_invalid_magic_or_truncated_header_cannot_construct_a_map(tmp_path):
    path = tmp_path/"field.npy"
    for payload in (b"not a numpy array", b"\x93NUMPY\x01\x00"):
        path.write_bytes(payload)
        with pytest.raises(ValueError):
            npy_io.readonly_npy_mmap(path)
