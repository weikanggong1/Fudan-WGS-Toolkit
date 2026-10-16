"""Filesystem accounting shared by real and synthetic preparation.

Logical lengths, allocated blocks and unique inodes answer different questions.
These measurements never traverse directory symlinks or treat unreadable paths
as empty. Estimates supplied by a caller are kept separate from measurements.
"""
from __future__ import annotations

import os
from pathlib import Path
import stat


def source_storage(paths):
    """Measure explicit files/directories; count a hard-linked inode once.

    Explicit roots are resolved once, so a deliberately supplied source link
    measures its target. Links encountered while walking a tree are counted as
    links and never followed. Missing paths and permission errors propagate.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    return _measure([Path(value).expanduser().resolve(strict=True) for value in paths])


def directory_storage(path, *, exclude=()):
    """Measure a tree with optional relative paths/subtrees omitted.

    ``allocated_bytes`` is null where ``st_blocks`` is unavailable. The unique
    values deduplicate (device, inode), including shared metadata hard links.
    Omitting a report being written avoids self-size recursion in that report.
    """
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    omitted = []
    for value in exclude:
        item = Path(value)
        if item.is_absolute() or ".." in item.parts:
            raise ValueError("excluded storage paths must stay relative to the measured root")
        omitted.append(item.parts)
    return _measure([root], excluded_root=root, excluded=omitted)


def _measure(roots, excluded_root=None, excluded=()):
    logical = allocated = unique_logical = unique_allocated = files = links = 0
    allocated_known = True
    identities = set()
    stack = list(roots)
    while stack:
        path = stack.pop()
        if excluded_root is not None:
            parts = path.relative_to(excluded_root).parts
            if any(parts[:len(value)] == value for value in excluded):
                continue
        entry = path.lstat()
        if stat.S_ISLNK(entry.st_mode):
            links += 1
            continue
        if stat.S_ISDIR(entry.st_mode):
            with os.scandir(path) as entries:
                stack.extend(Path(item.path) for item in entries)
            continue
        if not stat.S_ISREG(entry.st_mode):
            continue
        files += 1
        logical += entry.st_size
        blocks = getattr(entry, "st_blocks", None)
        allocated_known &= blocks is not None
        allocated += 0 if blocks is None else blocks*512
        identity = entry.st_dev, entry.st_ino
        if identity not in identities:
            identities.add(identity)
            unique_logical += entry.st_size
            unique_allocated += 0 if blocks is None else blocks*512
    return dict(logical_bytes=logical,
        allocated_bytes=allocated if allocated_known else None,
        unique_logical_bytes=unique_logical,
        unique_allocated_bytes=unique_allocated if allocated_known else None,
        files=files, unique_files=len(identities), symlinks_not_followed=links,
        hardlink_duplicate_bytes=logical-unique_logical,
        accounting="regular file lengths and filesystem blocks; directory entries and external symlink targets excluded")
