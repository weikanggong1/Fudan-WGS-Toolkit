"""Read-only NPY maps with bounded header IO and identity-bracketed files."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np


def _identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def readonly_npy_mmap(path):
    """Map a nonobject NPY without buffered header read-ahead or a data copy.

    The NumPy parser preserves versions 1/2/3, shape, dtype and C/F layout.
    The mapping uses the same unbuffered descriptor that supplied the header;
    complete identity checks bracket parsing and mmap construction. Callers
    retain responsibility for their supplied manifest's checksum and geometry.
    """
    path = Path(path)
    before = _identity(path.stat())
    with path.open("rb", buffering=0) as stream:
        if _identity(os.fstat(stream.fileno())) != before:
            raise ValueError("NPY file identity changed before header read")
        version = np.lib.format.read_magic(stream)
        shape, fortran_order, dtype = np.lib.format._read_array_header(stream, version)
        offset = stream.tell()
        if dtype.hasobject:
            raise ValueError("read-only NPY maps do not accept object pickle arrays")
        required = offset + math.prod(shape)*dtype.itemsize
        if required > before[2]:
            raise ValueError("NPY file is shorter than its declared shape and dtype")
        if _identity(os.fstat(stream.fileno())) != before or _identity(path.stat()) != before:
            raise ValueError("NPY file identity changed during header read")
        result = np.memmap(stream, dtype=dtype, mode="r", offset=offset,
                           shape=shape, order="F" if fortran_order else "C")
        if _identity(os.fstat(stream.fileno())) != before or _identity(path.stat()) != before:
            raise ValueError("NPY file identity changed during map construction")
    return result
