"""Authenticated, optional mmap persistence for model-independent annotations.

Derived files contain candidate rows or numeric mask metadata only. They are
not genotypes, cohort frequencies, imputed dosages or association results.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import errno
import json
import mmap
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time

import numpy as np

from .annotation_index import CandidateAnnotationIndex
from .npy_io import readonly_npy_mmap

FORMAT = "candidate-annotation-mmap-v1"
MAX_MANIFEST_BYTES = 8 * 2**20
MAX_DATA_BYTES = 8 * 2**30
IO_BYTES = 8 * 2**20


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _private(value, *, directory=False):
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    return (kind(value.st_mode) and hasattr(os, "getuid") and value.st_uid == os.getuid()
            and value.st_mode & 0o077 == 0)


def _sha(path):
    before = _identity(path)
    digest = hashlib.sha256()
    with Path(path).open("rb", buffering=0) as stream:
        if _identity(path) != before or _stat(os.fstat(stream.fileno())) != before:
            raise ValueError("annotation cache file changed before checksum")
        for block in iter(lambda: stream.read(IO_BYTES), b""):
            digest.update(block)
        if _identity(path) != before or _stat(os.fstat(stream.fileno())) != before:
            raise ValueError("annotation cache file changed during checksum")
    return digest.hexdigest()


def _stat(s):
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _drop_pages(array):
    # Only unpublished validation maps are advised. This changes residency,
    # never array contents, and keeps a long scan's private RSS bounded.
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and hasattr(mapping, "madvise"):
        try:
            mapping.madvise(mmap.MADV_DONTNEED)
        except (OSError, ValueError):
            pass


def _validate_vector(array, *, minimum=0, maximum=None, ordered=False):
    step = max(1, IO_BYTES // max(1, array.dtype.itemsize))
    previous = None
    view = np.asarray(array)
    for start in range(0, len(view), step):
        part = view[start:start + step]
        if np.any(part < minimum) or (maximum is not None and np.any(part >= maximum)):
            raise ValueError("annotation cache row/code bounds differ")
        if ordered and len(part):
            if (previous is not None and part[0] <= previous) or np.any(part[1:] <= part[:-1]):
                raise ValueError("annotation cache rows are not unique physical order")
            previous = int(part[-1])
        _drop_pages(array)


def _write_array(path, shape, dtype, parts):
    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    offset = 0
    for source in parts:
        source = np.asarray(source)
        if source.ndim != 1 or source.dtype != np.dtype(dtype):
            raise ValueError("derived annotation array dtype/shape differs")
        step = max(1, IO_BYTES // max(1, source.dtype.itemsize))
        for start in range(0, len(source), step):
            stop = min(start + step, len(source))
            array[offset + start:offset + stop] = source[start:stop]
            array.flush()
            _drop_pages(array)
        offset += len(source)
    if offset != shape[0]:
        raise ValueError("derived annotation row count differs")
    array.flush()
    del array
    with Path(path).open("rb", buffering=0) as stream:
        os.fsync(stream.fileno())
    return dict(size=path.stat().st_size, sha256=_sha(path), shape=list(shape), dtype=np.dtype(dtype).str)


class AnnotationCache:
    """Best-effort derived cache; invalid entries fall back to original readers.

    Source authentication is supplied by the metadata reader and runs outside
    fallback handling: an altered original annotation is always an error.
    Only cache failures are recoverable. One flat row map owns all gene views.
    """
    def __init__(self, directory, source, validate_source, metrics, *, future=None):
        self.directory = Path(directory)
        self.source = source
        self.validate_source = validate_source
        self.metrics = metrics
        self.future = future
        self._invalid_path = None

    def _binding(self, kind, rules):
        return dict(format=FORMAT, kind=kind, source=self.source, rules=rules)

    def _path(self, binding):
        return self.directory / hashlib.sha256(_json(binding)).hexdigest()

    def _count(self, key, amount=1):
        self.metrics[key] = self.metrics.get(key, 0) + amount

    def record_build(self):
        """Count original-selector work, including uncached readonly fallback."""
        self._count("builds")

    @contextmanager
    def _workspace(self):
        # SHA buffers, manifest parsing, and bounded vector validation only.
        # Genotype/cohort preparation leases are acquired separately afterwards.
        if callable(self.future):
            self.future(2 * IO_BYTES + 8 * MAX_MANIFEST_BYTES, phase="annotation_cache_io")
        try:
            yield
        finally:
            if callable(self.future):
                self.future(0, phase="annotation_cache_io_released")

    def _read(self, binding):
        path = self._path(binding)
        if not path.is_dir():
            self._count("misses")
            return None
        if (not _private(self.directory.lstat(), directory=True)
                or not _private(path.lstat(), directory=True)):
            raise ValueError("derived annotation cache must be private and owned")
        raw_path = path / "manifest.json"
        if not _private(raw_path.lstat()) or not _private((path / "COMPLETE").lstat()):
            raise ValueError("derived annotation markers must be private owned files")
        if raw_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError("derived annotation manifest exceeds bound")
        raw = raw_path.read_bytes()
        if (path / "COMPLETE").read_text() != hashlib.sha256(raw).hexdigest():
            raise ValueError("derived annotation completion marker differs")
        manifest = json.loads(raw)
        if manifest.get("binding") != binding or manifest.get("format") != FORMAT:
            raise ValueError("derived annotation input/rule binding differs")
        specs = manifest.get("files", {})
        if not isinstance(specs, dict) or not specs or sum(s["size"] for s in specs.values()) > MAX_DATA_BYTES:
            raise ValueError("derived annotation files exceed bound")
        arrays, identities = {}, {}
        for name, spec in specs.items():
            if Path(name).name != name or not name.endswith(".npy"):
                raise ValueError("derived annotation filename is invalid")
            file = path / name
            # Never follow a cache-provided path outside this private entry.
            if file.resolve().parent != path.resolve():
                raise ValueError("derived annotation file escaped its directory")
            if not _private(file.lstat()):
                raise ValueError("derived annotation arrays must be private owned files")
            before = _identity(file)
            if before[2] != spec["size"] or _sha(file) != spec["sha256"]:
                raise ValueError("derived annotation checksum differs")
            value = readonly_npy_mmap(file)
            if _identity(file) != before or list(value.shape) != spec["shape"] or value.dtype.str != spec["dtype"]:
                raise ValueError("derived annotation geometry/identity differs")
            arrays[name] = value
            identities[name] = before
        if _sha(raw_path) != hashlib.sha256(raw).hexdigest():
            raise ValueError("derived annotation manifest changed during read")
        return manifest, arrays, identities

    def _load(self, kind, rules, decode):
        self.validate_source()
        started = time.perf_counter()
        try:
            with self._workspace():
                loaded = self._read(self._binding(kind, rules))
                if loaded is None:
                    return None
                manifest, arrays, identities = loaded
                result = decode(manifest, arrays)
                path = self._path(self._binding(kind, rules))
                if any(_identity(path / name) != identity for name, identity in identities.items()):
                    raise ValueError("derived annotation changed during validation")
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            self._count("fallback_invalid")
            self._invalid_path = self._path(self._binding(kind, rules))
            return None
        finally:
            self._count("read_seconds", time.perf_counter() - started)
        self.validate_source()
        self._count("hits")
        return result

    def load_array(self, name, rules, *, dtype, size, maximum=None):
        def decode(manifest, arrays):
            if set(arrays) != {"values.npy"}:
                raise ValueError("numeric annotation array fields differ")
            array = arrays["values.npy"]
            if array.dtype != np.dtype(dtype) or array.shape != (size,):
                raise ValueError("numeric annotation geometry differs")
            _validate_vector(array, maximum=maximum)
            self._count("mapped_bytes", array.nbytes)
            return array
        return self._load("array:" + name, rules, decode)

    def load_index(self, rules):
        def decode(manifest, arrays):
            if set(arrays) != {"rows.npy", "offsets.npy"}:
                raise ValueError("candidate annotation array fields differ")
            rows, offsets = arrays["rows.npy"], arrays["offsets.npy"]
            if rows.dtype != np.int64 or rows.ndim != 1 or offsets.dtype != np.int64 or offsets.ndim != 1:
                raise ValueError("candidate annotation row/offset dtype differs")
            groups = manifest["groups"]
            pairs = sum(len(categories) for gene, categories in groups)
            if (offsets.shape != (pairs + 1,) or offsets[0] != 0 or offsets[-1] != len(rows)
                    or np.any(offsets[1:] < offsets[:-1])):
                raise ValueError("candidate annotation offsets differ")
            index = CandidateAnnotationIndex(rules["chromosome"], rules["variant_type"])
            index.prepared_categories = set(rules["categories"])
            index.promoter_signature = (None if rules["promoter_signature"] is None else
                                        tuple(map(tuple, rules["promoter_signature"])))
            seen = set()
            at = 0
            for gene, categories in groups:
                if not isinstance(gene, str) or not gene or gene in seen:
                    raise ValueError("candidate annotation gene directory differs")
                seen.add(gene)
                local = index._groups.setdefault(gene, {})
                for category in categories:
                    if category not in index.prepared_categories or category in local:
                        raise ValueError("candidate annotation category directory differs")
                    a = rows[int(offsets[at]):int(offsets[at + 1])]
                    _validate_vector(a, maximum=self.source["n_variants"], ordered=True)
                    local[category] = a
                    at += 1
            _drop_pages(rows)
            self._count("mapped_bytes", rows.nbytes + offsets.nbytes)
            return index
        return self._load("index", rules, decode)

    @contextmanager
    def _writer(self, binding):
        """Nonblocking process lock plus atomic directory publication."""
        try:
            import fcntl
        except ImportError as error:
            raise OSError("process locking is unavailable for optional annotation cache") from error
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not _private(self.directory.lstat(), directory=True):
            raise PermissionError("derived annotation cache must be private and owned")
        target = self._path(binding)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        lock_fd = os.open(self.directory / (target.name + ".lock"), flags, 0o600)
        with os.fdopen(lock_fd, "a+b") as lock:
            if not _private(os.fstat(lock.fileno())):
                raise PermissionError("derived annotation lock must be a private owned file")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self._count("lock_busy")
                yield None
                return
            temporary = Path(tempfile.mkdtemp(prefix=".annotation-", dir=self.directory))
            try:
                yield temporary
                if (temporary / "COMPLETE").is_file():
                    if target.exists():
                        # Keep a valid concurrent entry; retire invalid cache
                        # data only while holding its exclusive writer lock.
                        try:
                            if target == self._invalid_path:
                                raise ValueError("entry failed semantic validation")
                            self._read(binding)
                            return
                        except (OSError, ValueError, KeyError, TypeError, OverflowError):
                            retired = self.directory / (".retired-" + temporary.name[1:])
                            os.replace(target, retired)
                            if retired.is_symlink() or not retired.is_dir():
                                retired.unlink()
                            else:
                                shutil.rmtree(retired)
                    publication_bytes = sum(file.stat().st_size for file in temporary.iterdir())
                    os.replace(temporary, target)
                    # Persist the atomic directory entry too, when supported.
                    fd = os.open(self.directory, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    self._count("publishes")
                    self._count("published_bytes", publication_bytes)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _save(self, kind, rules, write):
        self.validate_source()
        started = time.perf_counter()
        binding = self._binding(kind, rules)
        try:
            with self._workspace(), self._writer(binding) as temporary:
                if temporary is None:
                    return
                manifest = dict(format=FORMAT, binding=binding, **write(temporary))
                raw = _json(manifest)
                if len(raw) > MAX_MANIFEST_BYTES or sum(s["size"] for s in manifest["files"].values()) > MAX_DATA_BYTES:
                    self._count("fallback_size")
                    return
                # Recheck source before COMPLETE and publication; changed
                # annotations are errors, never a valid derived entry.
                self.validate_source()
                (temporary / "manifest.json").write_bytes(raw)
                (temporary / "COMPLETE").write_text(hashlib.sha256(raw).hexdigest())
                for file in temporary.iterdir():
                    file.chmod(0o600)
                    with file.open("rb", buffering=0) as stream:
                        os.fsync(stream.fileno())
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            self._count("fallback_write")
            if isinstance(error, PermissionError) or (isinstance(error, OSError)
                    and error.errno in (errno.EROFS, errno.EACCES, errno.EPERM)):
                self._count("fallback_read_only")
        finally:
            self._count("write_seconds", time.perf_counter() - started)
        self.validate_source()

    def save_array(self, name, rules, values):
        values = np.asarray(values)
        if values.nbytes + 4096 > MAX_DATA_BYTES:
            self._count("fallback_size")
            return
        def write(path):
            return dict(files={"values.npy": _write_array(path / "values.npy", values.shape, values.dtype, [values])})
        self._save("array:" + name, rules, write)

    def save_index(self, rules, index):
        categories = set(rules["categories"])
        groups, parts, stops = [], [], [0]
        for gene, annotations in index._groups.items():
            names = []
            for category, rows in annotations.items():
                if category in categories:
                    names.append(category)
                    parts.append(rows)
                    stops.append(stops[-1] + len(rows))
            if names:
                groups.append([gene, names])
        # References to existing immutable row arrays only; no full concatenate.
        if 8 * (stops[-1] + len(stops)) + 4096 > MAX_DATA_BYTES:
            self._count("fallback_size")
            return
        def write(path):
            flat = _write_array(path / "rows.npy", (stops[-1],), np.int64, parts)
            offsets = np.asarray(stops, dtype=np.int64)
            directory = _write_array(path / "offsets.npy", offsets.shape, offsets.dtype, [offsets])
            return dict(groups=groups, files={"rows.npy": flat, "offsets.npy": directory})
        self._save("index", rules, write)
