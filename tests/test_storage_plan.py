"""Measured filesystem accounting; synthetic estimates are tested separately."""
from pathlib import Path

import pytest

from fudan_wgs_toolkit.storage_plan import directory_storage, source_storage


def test_sparse_lengths_allocated_blocks_and_hard_links_are_distinct(tmp_path):
    source = tmp_path/"sparse.bin"
    with source.open("wb") as stream:
        stream.seek(4*2**20)
        stream.write(b"X")
    (tmp_path/"alias.bin").hardlink_to(source)
    expected = source.stat()
    measured = directory_storage(tmp_path)
    assert measured["logical_bytes"] == 2*expected.st_size
    assert measured["unique_logical_bytes"] == expected.st_size
    assert measured["files"] == 2 and measured["unique_files"] == 1
    assert measured["hardlink_duplicate_bytes"] == expected.st_size
    if hasattr(expected, "st_blocks"):
        assert measured["allocated_bytes"] == expected.st_blocks*1024
        assert measured["unique_allocated_bytes"] == expected.st_blocks*512
    assert source_storage([source, tmp_path/"alias.bin"])["unique_logical_bytes"] == expected.st_size


def test_storage_exclusion_is_explicit_and_tree_links_are_not_followed(tmp_path):
    target = tmp_path/"source"
    target.mkdir()
    (target/"data").write_bytes(b"abcdef")
    (tmp_path/"external-link").symlink_to(target, target_is_directory=True)
    (tmp_path/"report.json").write_bytes(b"{}")
    measured = directory_storage(tmp_path, exclude=("report.json",))
    assert measured["logical_bytes"] == 6
    assert measured["symlinks_not_followed"] == 1
    assert source_storage(tmp_path/"external-link")["logical_bytes"] == 6
    with pytest.raises(ValueError, match="relative"):
        directory_storage(tmp_path, exclude=("../outside",))
    with pytest.raises(FileNotFoundError):
        directory_storage(tmp_path/"missing")


def test_filesystem_errors_are_not_silently_counted_as_zero(tmp_path, monkeypatch):
    target = tmp_path/"unreadable"
    target.write_bytes(b"secret")
    original = Path.lstat
    def denied(path, *args, **kwargs):
        if path == target:
            raise PermissionError("read denied")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "lstat", denied)
    with pytest.raises(PermissionError, match="read denied"):
        directory_storage(tmp_path)
