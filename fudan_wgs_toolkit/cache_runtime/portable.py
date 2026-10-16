"""Self-contained metadata for a complete six-state chromosome cache.

Runtime readers use NumPy memory maps and the existing validated CSR adapter.
They never open, stat or import a native genotype SDK. ``export_metadata`` is a
preparation operation on a supplied reader; it does not export genotypes.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from ..npy_io import readonly_npy_mmap
from .adapter_fast import CachedGenotypeAdapter
from .fast_container import Container


FORMAT = "portable-six-state-metadata-v1"
_SHA_CACHE_LIMIT = 4096
# Some shared filesystems expose coarse ctime ticks. A same-size rewrite that
# restores mtime can retain an identity inside that tick. Rehash recent files
# until the timestamp has settled; immutable older metadata keeps the fast path.
_SHA_CACHE_SETTLE_NS = 2_000_000_000
_SHA_CACHE = OrderedDict()
_SHA_METRICS = dict(calls=0, identity_hits=0, content_hashes=0,
                    content_bytes=0, content_hash_seconds=0.0)


def _stat_identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _sha(path):
    """Hash once per unchanged file identity; never cache an unbracketed scan."""
    path = Path(path)
    key = str(path.resolve())
    identity = _identity(path)
    _SHA_METRICS["calls"] += 1
    previous = _SHA_CACHE.get(key)
    settled = time.time_ns() - max(identity[3:]) >= _SHA_CACHE_SETTLE_NS
    if previous is not None and previous[0] == identity and settled:
        if _identity(path) != identity:
            raise ValueError("portable file changed during checksum verification")
        _SHA_CACHE.move_to_end(key)
        _SHA_METRICS["identity_hits"] += 1
        return previous[1]
    digest = hashlib.sha256()
    started = time.perf_counter()
    with path.open("rb", buffering=0) as stream:
        if _stat_identity(os.fstat(stream.fileno())) != identity:
            raise ValueError("portable file changed before checksum verification")
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
        if _stat_identity(os.fstat(stream.fileno())) != identity or _identity(path) != identity:
            raise ValueError("portable file changed during checksum verification")
    result = digest.hexdigest()
    _SHA_METRICS["content_hashes"] += 1
    _SHA_METRICS["content_bytes"] += identity[2]
    _SHA_METRICS["content_hash_seconds"] += time.perf_counter()-started
    _SHA_CACHE[key] = (identity, result)
    _SHA_CACHE.move_to_end(key)
    while len(_SHA_CACHE) > _SHA_CACHE_LIMIT:
        _SHA_CACHE.popitem(last=False)
    return result


def _identity(path):
    return _stat_identity(Path(path).stat())


class _ArrayWriter:
    """Append a fixed trailing-shape array without object pickle or padding."""
    def __init__(self, directory, name, dtype, trailing_shape=()):
        self.directory, self.name = Path(directory), name
        self.dtype, self.trailing_shape = np.dtype(dtype), tuple(trailing_shape)
        self.count = 0
        self.raw_path = self.directory / (name + ".raw")
        self.stream = self.raw_path.open("wb")

    def append(self, values):
        values = np.asarray(values)
        if values.shape[1:] != self.trailing_shape or values.dtype != self.dtype:
            raise ValueError("metadata array type or trailing shape changed between chunks")
        self.stream.write(np.ascontiguousarray(values).tobytes())
        self.count += len(values)

    def finish(self):
        self.close()
        path = self.directory / (self.name + ".npy")
        shape = (self.count, *self.trailing_shape)
        expected_bytes = self.dtype.itemsize * math.prod(shape)
        if self.raw_path.stat().st_size != expected_bytes:
            raise ValueError("metadata raw byte count differs from its declared shape")
        # Copy the already ordered bytes behind a standard NPY header. Mapping
        # full chromosome fields here retained hundreds of MB of dirty/resident
        # pages per worker; fixed byte blocks keep this finalization bounded.
        with path.open("wb") as destination, self.raw_path.open("rb") as source:
            np.lib.format.write_array_header_2_0(destination, dict(
                descr=np.lib.format.dtype_to_descr(self.dtype),
                fortran_order=False, shape=shape))
            for block in iter(lambda: source.read(8 * 2**20), b""):
                destination.write(block)
            destination.flush()
            os.fsync(destination.fileno())
        self.raw_path.unlink()
        return dict(kind="dense", file=path.name, dtype=self.dtype.str,
                    shape=[self.count, *self.trailing_shape])

    def close(self):
        self.stream.close()


class _StringWriter:
    def __init__(self, directory, name, dtype, trailing_shape=()):
        self.directory, self.name = Path(directory), name
        self.dtype, self.trailing_shape = np.dtype(dtype), tuple(trailing_shape)
        self.data = (self.directory / (name + ".utf8")).open("wb")
        self.offsets = _ArrayWriter(directory, name + ".offsets", "<u8")
        self.valid = _ArrayWriter(directory, name + ".valid", "|b1")
        self.offsets.append(np.asarray([0], dtype="<u8"))
        self.count = self.bytes_written = self.max_characters = self.max_bytes = self.missing_count = 0

    def append(self, values):
        values = np.asarray(values)
        if values.shape[1:] != self.trailing_shape or values.dtype.kind not in "OUS":
            raise ValueError("metadata text shape or type changed between chunks")
        flat = values.reshape(-1)
        offsets, valid = np.empty(len(flat), dtype="<u8"), np.ones(len(flat), dtype=bool)
        for index, value in enumerate(flat):
            if value is None:
                valid[index] = False
                self.missing_count += 1
                payload = b""
            elif isinstance(value, (str, np.str_)):
                payload = str(value).encode("utf-8")
                self.max_characters = max(self.max_characters, len(value))
            elif isinstance(value, (bytes, np.bytes_)):
                payload = bytes(value)
                self.max_characters = max(self.max_characters, len(payload.decode("utf-8")))
            else:
                raise ValueError("object metadata must contain only strings or None")
            self.data.write(payload)
            self.max_bytes = max(self.max_bytes, len(payload))
            self.bytes_written += len(payload)
            offsets[index] = self.bytes_written
        self.offsets.append(offsets)
        self.valid.append(valid)
        self.count += len(values)

    def finish(self):
        self.data.close()
        dtype = (np.dtype("U%d" % max(1, self.max_characters)) if self.dtype.kind == "U" else
                 np.dtype("S%d" % max(1, self.max_bytes)) if self.dtype.kind == "S" else self.dtype)
        return dict(kind="utf8", file=self.name + ".utf8",
            offsets=self.offsets.finish(), valid=self.valid.finish(),
            shape=[self.count, *self.trailing_shape], original_dtype=dtype.str,
            max_characters=self.max_characters, missing_count=self.missing_count)

    def close(self):
        self.data.close()
        self.offsets.close()
        self.valid.close()


def _value_writer(directory, name, values):
    array = np.asarray(values)
    if array.ndim < 1:
        raise ValueError("variant metadata must have a leading axis")
    if array.dtype.kind in "OUS":
        return _StringWriter(directory, name, array.dtype, array.shape[1:])
    if array.dtype.kind not in "biufc":
        raise ValueError("unsupported metadata dtype")
    return _ArrayWriter(directory, name, array.dtype, array.shape[1:])


def _export_field(reader, path, directory, name, block_size):
    writer = flat_writer = row_offsets = None
    total_rows = flat_count = 0
    for begin in range(0, reader.n_variants, block_size):
        indices = np.arange(begin, min(begin + block_size, reader.n_variants), dtype=np.int64)
        values = reader.read_field(path, indices)
        if isinstance(values, list):
            if writer is not None:
                raise ValueError("fixed metadata became variable-length")
            if row_offsets is None:
                row_offsets = _ArrayWriter(directory, name + ".rows", "<u8")
                row_offsets.append(np.asarray([0], dtype="<u8"))
            ends = np.empty(len(values), dtype="<u8")
            for index, row in enumerate(values):
                row = np.asarray(row)
                if row.ndim < 1:
                    raise ValueError("ragged metadata rows must be arrays")
                # Empty native rows can have a generic float dtype. Defer
                # selecting flat type until actual stored values arrive.
                if len(row):
                    if flat_writer is None:
                        flat_writer = _value_writer(directory, name + ".values", row)
                    flat_writer.append(row)
                flat_count += len(row)
                ends[index] = flat_count
            row_offsets.append(ends)
        else:
            if row_offsets is not None:
                raise ValueError("variable-length metadata became fixed")
            values = np.asarray(values)
            if len(values) != len(indices):
                raise ValueError("metadata read changed the variant axis")
            if writer is None:
                writer = _value_writer(directory, name, values)
            writer.append(values)
        if len(values) != len(indices):
            raise ValueError("metadata read changed the variant axis")
        total_rows += len(indices)
    if total_rows != reader.n_variants:
        raise ValueError("metadata field coverage differs from variants")
    if row_offsets is not None:
        if flat_writer is None:
            flat_writer = _ArrayWriter(directory, name + ".values", "<f8")
        return dict(kind="ragged", shape=[total_rows], offsets=row_offsets.finish(),
                    values=flat_writer.finish())
    if writer is None:
        writer = _ArrayWriter(directory, name, "<f8")
    return writer.finish()


def _complete_annotation_coverage(coverage, fields, directory, n_variants, block_size):
    """Prove optional complete coverage from the actual exported boolean field."""
    if not isinstance(coverage, dict):
        raise ValueError("annotation_coverage must be a complete coverage mapping")
    result = dict(coverage)
    expected = dict(n_variants=n_variants, matched_variants=n_variants,
                    missing_snv=0, missing_non_snv=0)
    if any(type(result.get(name)) is not int or result[name] != count
           for name, count in expected.items()):
        raise ValueError("annotation_coverage must describe complete selected-variant coverage")
    if result.get("allow_missing_non_snv", False) is not False:
        raise ValueError("complete annotation_coverage cannot allow missing annotations")
    field = result.get("available_field", "annotation_available")
    if not isinstance(field, str) or field not in fields:
        raise ValueError("annotation_coverage requires an exported availability field")
    spec = fields[field]
    if (spec.get("kind") != "dense" or spec.get("shape") != [n_variants]
            or np.dtype(spec.get("dtype")).kind != "b"):
        raise ValueError("annotation availability must be a boolean selected-variant vector")
    values = np.load(Path(directory) / spec["file"], mmap_mode="r", allow_pickle=False)
    try:
        if values.shape != (n_variants,) or values.dtype.kind != "b":
            raise ValueError("annotation availability must be a boolean selected-variant vector")
        matched = sum(int(np.count_nonzero(values[begin:begin + block_size]))
                      for begin in range(0, n_variants, block_size))
        if matched != n_variants:
            raise ValueError("annotation availability differs from complete coverage proof")
    finally:
        mapping = getattr(values, "_mmap", None)
        if mapping is not None:
            mapping.close()
    result.update(available_field=field, allow_missing_non_snv=False)
    return result


def export_metadata(reader, container_directory, output_directory, field_paths, *,
                    annotation_catalog=None, annotation_names=(), qc_path="annotation/filter",
                    block_size=65536, sample_pairs=None, annotation_coverage=None):
    """Prepare immutable metadata beside a verified complete genotype cache.

    ``reader`` supplies sample IDs and chunked metadata. ``field_paths`` are
    aligned variant fields, including position/chromosome/variant.id/allele,
    QC and every requested mask/weight annotation. Native genotype is never
    read here. The completed manifest binds the exact target cache manifest.
    Optional ``annotation_coverage`` declares complete selected-axis coverage;
    its counts are checked against an exported all-true boolean availability
    vector before the same atomic metadata/COMPLETE publication. Omitting it
    preserves the previous metadata schema and export behavior.
    """
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("refusing to overwrite portable metadata")
    cache = Path(container_directory)
    container = Container(cache)
    temporary = output.with_name(output.name + ".partial-%d" % os.getpid())
    if temporary.exists():
        raise FileExistsError("portable metadata preparation directory already exists")
    started = time.perf_counter()
    try:
        if reader.n_variants != container.manifest["m"]:
            raise ValueError("native metadata and target cache variant dimensions differ")
        from ..identity import sample_keys, validate_sample_pairs
        if sample_pairs is None:
            if not callable(getattr(reader, "sample_pairs", None)):
                raise ValueError("metadata export requires explicit full-source FID/IID sample_pairs")
            sample_pairs = reader.sample_pairs()
        pairs = validate_sample_pairs(sample_pairs, where="metadata source sample pairs")
        if pairs.shape != (reader.n_samples, 2):
            raise ValueError("source sample dimension differs from FID/IID pairs")
        if np.any(container.samples < 0) or np.any(container.samples >= reader.n_samples):
            raise ValueError("cache samples exceed metadata source axis")
        temporary.mkdir(parents=True, mode=0o700)
        # The portable sample axis is the cached population, not the native
        # file's larger source population. Retain original rows as provenance;
        # association indices are positions in this portable population axis.
        pairs = pairs[container.samples]
        samples = sample_keys(pairs)
        np.save(temporary / "sample_pairs.npy", pairs, allow_pickle=False)
        np.save(temporary / "sample_ids.npy", samples, allow_pickle=False)
        np.save(temporary / "source_sample_rows.npy", container.samples, allow_pickle=False)
        fields = {}
        for index, path in enumerate(dict.fromkeys(field_paths)):
            if not isinstance(path, str) or not path or path.startswith("genotype/"):
                raise ValueError("only named variant metadata fields can be exported")
            fields[path] = _export_field(reader, path, temporary, "field_%03d" % index, block_size)
        required = {"position", "chromosome", "variant.id", "allele", qc_path}
        required.update((annotation_catalog or {}).values())
        if not required <= fields.keys():
            raise ValueError("portable metadata lacks required variant/QC/annotation fields")
        coverage = (None if annotation_coverage is None else
                    _complete_annotation_coverage(annotation_coverage, fields, temporary,
                                                  reader.n_variants, block_size))
        files = {p.name: dict(sha256=_sha(p), size=p.stat().st_size)
                 for p in temporary.iterdir() if p.is_file()}
        manifest = dict(schema_version=1, format=FORMAT, n_samples=len(container.samples),
            n_source_samples=reader.n_samples, source_sample_rows="source_sample_rows.npy",
            n_variants=reader.n_variants, sample_ids="sample_ids.npy",
            sample_pairs="sample_pairs.npy", sample_identifier_format="fid_iid_json",
            genotype_manifest_sha256=_sha(cache / "manifest.json"),
            genotype_sample_axis_sha256=container.manifest["sample_sha256"],
            fields=fields, files=files,
            analysis=dict(annotation_catalog=dict(annotation_catalog or {}),
                          annotation_names=list(annotation_names), qc_path=qc_path),
            preparation=dict(metadata_only=True, genotype_read=False,
                             block_size=block_size, seconds=time.perf_counter() - started))
        if coverage is not None:
            manifest["annotation_coverage"] = coverage
        document = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n"
        (temporary / "manifest.json").write_bytes(document)
        (temporary / "COMPLETE").write_text(hashlib.sha256(document).hexdigest())
        for path in temporary.iterdir():
            path.chmod(0o600)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    finally:
        container.close()


class PortableMetadataReader:
    """Read aligned fields lazily from a completed metadata directory."""
    sample_axis_kind = "prepared_population"
    def __init__(self, metadata_directory, container_directory, *, verify_checksums=True,
                 legacy_numeric=False):
        if type(legacy_numeric) is not bool:
            raise ValueError("legacy_numeric must be an explicit boolean")
        self.directory = Path(metadata_directory).resolve()
        self._closed = False
        self._arrays, self._identities, self._validated_offsets = {}, {}, set()
        self._verify_checksums = bool(verify_checksums)
        self._metadata_reads = self._checked_files = 0
        self._utf8_index_metrics = dict(ordered_requests=0, reordered_requests=0,
                                       reused_index_requests=0)
        self._metadata_seconds = self._verification_seconds = 0.0
        self._sha_baseline = dict(_SHA_METRICS)
        self._sample_lookup = None
        self._annotation_cache_metrics = {}
        raw = (self.directory / "manifest.json").read_bytes()
        if (self.directory / "COMPLETE").read_text() != hashlib.sha256(raw).hexdigest():
            raise ValueError("portable metadata completion marker differs")
        self._metadata_manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        if self.manifest.get("format") != FORMAT or self.manifest.get("schema_version") != 1:
            raise ValueError("unsupported portable metadata format")
        self.n_samples, self.n_variants = self.manifest["n_samples"], self.manifest["n_variants"]
        if any(type(n) is not int or n < 0 for n in (self.n_samples, self.n_variants)):
            raise ValueError("invalid portable metadata dimensions")
        cache = Path(container_directory)
        if _sha(cache / "manifest.json") != self.manifest["genotype_manifest_sha256"]:
            raise ValueError("portable metadata is bound to another genotype cache")
        genotype = json.loads((cache / "manifest.json").read_text())
        if (genotype["m"] != self.n_variants or genotype["n"] != self.n_samples or
                genotype["sample_sha256"] != self.manifest["genotype_sample_axis_sha256"]):
            raise ValueError("portable metadata/genotype physical axes differ")
        self._manifest_identity = _identity(self.directory / "manifest.json")
        self.genotype_raw_memory_bytes = 0
        self.genotype_max_gap_layers = 0
        self.ploidy = 2
        samples = self._array(self.manifest["sample_ids"])
        if samples.shape != (self.n_samples,) or samples.dtype.hasobject:
            raise ValueError("portable sample identifier geometry differs")
        if self.manifest["sample_identifier_format"] == "fid_iid_json":
            from ..identity import sample_keys, validate_sample_pairs
            pairs = validate_sample_pairs(self._array(self.manifest["sample_pairs"]),
                                          where="prepared sample pairs")
            if pairs.shape[0] != self.n_samples or not np.array_equal(samples, sample_keys(pairs)):
                raise ValueError("prepared FID/IID pairs differ from sample keys")
        elif legacy_numeric and self.manifest["sample_identifier_format"] == "positive_decimal_int64":
            # Explicit PheWAS compatibility only: preserve the completed older
            # cache's exact participant IDs; never manufacture FID/IID pairs.
            if samples.dtype != np.int64 or np.any(samples <= 0):
                raise ValueError("legacy portable sample identifiers must be positive int64")
            if "sample_pairs" in self.manifest:
                raise ValueError("legacy metadata cannot also declare a FID/IID sample axis")
        else:
            raise ValueError("prepared metadata requires explicit FID/IID sample pairs")
        if len(np.unique(samples)) != self.n_samples:
            raise ValueError("portable sample identifiers are not unique")
        source_rows = self._array(self.manifest["source_sample_rows"])
        if (source_rows.dtype != np.int64 or source_rows.shape != (self.n_samples,) or
                np.any(source_rows < 0) or np.any(source_rows >= self.manifest["n_source_samples"]) or
                hashlib.sha256(source_rows.astype("<i8").tobytes()).hexdigest() != genotype["sample_sha256"]):
            raise ValueError("portable original physical sample axis differs")

    def _path(self, name):
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError("portable field file must be a single relative filename")
        path = (self.directory / name).resolve()
        if path.parent != self.directory:
            raise ValueError("portable field escaped metadata directory")
        return path

    def _verify(self, name):
        if self._closed:
            raise ValueError("portable metadata reader is closed")
        if _identity(self.directory / "manifest.json") != self._manifest_identity:
            raise ValueError("immutable portable metadata manifest changed")
        path = self._path(name)
        spec = self.manifest["files"].get(name)
        if spec is None:
            raise ValueError("unmanifested portable metadata file")
        identity = _identity(path)
        if name in self._identities:
            if identity != self._identities[name]:
                raise ValueError("immutable portable metadata file changed")
            return path
        started = time.perf_counter()
        if identity[2] != spec["size"] or (self._verify_checksums and _sha(path) != spec["sha256"]):
            raise ValueError("portable metadata file checksum or size differs")
        if _identity(path) != identity:
            raise ValueError("portable metadata changed during verification")
        self._verification_seconds += time.perf_counter() - started
        self._checked_files += 1
        self._identities[name] = identity
        return path

    def _array(self, name):
        path = self._verify(name)
        if name not in self._arrays:
            value = readonly_npy_mmap(path)
            if _identity(path) != self._identities[name]:
                raise ValueError("immutable portable metadata file changed")
            self._arrays[name] = value
        return self._arrays[name]

    def _read(self, spec, indices):
        shape = tuple(spec["shape"])
        if spec["kind"] == "dense":
            values = self._array(spec["file"])
            if values.shape != shape or values.dtype.str != spec["dtype"]:
                raise ValueError("portable dense field geometry differs")
            return values if indices is None else values[indices]
        if spec["kind"] == "utf8":
            # Retain the authenticated read-only mappings as zero-copy base
            # ndarray views. A scalar memmap lookup otherwise invokes its
            # Python subclass wrapper for every decoded offset and validity
            # flag, although no new mapped view is needed for those scalars.
            offsets = np.asarray(self._read(spec["offsets"], None))
            valid = np.asarray(self._read(spec["valid"], None))
            flat_count = int(np.prod(shape, dtype=np.int64))
            path = self._verify(spec["file"])
            if offsets.shape != (flat_count + 1,) or valid.shape != (flat_count,) or int(offsets[-1]) != path.stat().st_size:
                raise ValueError("portable UTF8 field geometry differs")
            if spec["file"] not in self._validated_offsets:
                if np.any(offsets[1:] < offsets[:-1]) or int(offsets[0]) != 0:
                    raise ValueError("portable UTF8 offsets are not ordered")
                self._validated_offsets.add(spec["file"])
            selected_count = shape[0] if indices is None else len(indices)
            width = int(np.prod(shape[1:], dtype=np.int64))
            original_dtype = np.dtype(spec["original_dtype"])
            result = np.empty((selected_count, *shape[1:]), dtype=original_dtype if original_dtype.kind in "US" else object)
            target = result.reshape((selected_count, width))
            # Coalesce nearby requests into bounded byte windows. Whole-field
            # reads use at most 65536 rows per pread; sparse order is retained.
            # Complete fields (notably QC) need no full-length request/order
            # arrays: their offsets and values are already in physical order.
            if indices is None:
                order = sorted_indices = None
                request_count = flat_count
            else:
                ordered = len(indices) < 2 or np.all(indices[1:] > indices[:-1])
                if ordered:
                    # Single blocks and gene masks already use physical order.
                    # Scalar fields reuse their exact axis without expanded,
                    # sorting or gather arrays. Wider fields retain row-major
                    # expansion; only the redundant stable sort is omitted.
                    order = None
                    sorted_indices = (indices if width == 1 else
                        (indices[:, None] * width + np.arange(width)).reshape(-1))
                    self._utf8_index_metrics["ordered_requests"] += 1
                    self._utf8_index_metrics["reused_index_requests"] += int(width == 1)
                else:
                    flat_indices = (indices[:, None] * width + np.arange(width)).reshape(-1)
                    order = np.argsort(flat_indices, kind="stable")
                    sorted_indices = flat_indices[order]
                    self._utf8_index_metrics["reordered_requests"] += 1
                request_count = len(sorted_indices)
            flat_target = target.reshape(-1)
            with path.open("rb", buffering=0) as stream:
                begin_at = 0
                while begin_at < request_count:
                    first = begin_at if sorted_indices is None else int(sorted_indices[begin_at])
                    byte_start = int(offsets[first])
                    stop = min(request_count, begin_at + 65536)
                    candidate_ends = (offsets[begin_at + 1:stop + 1] if sorted_indices is None else
                                      offsets[sorted_indices[begin_at:stop] + 1])
                    # A single unusually long string remains legal, but does
                    # not cause adjacent requests to expand a larger window.
                    stop = begin_at + max(1, int(np.searchsorted(candidate_ends, byte_start + 16 * 2**20, side="right")))
                    last = stop - 1 if sorted_indices is None else int(sorted_indices[stop - 1])
                    byte_end = int(offsets[last + 1])
                    payload = os.pread(stream.fileno(), byte_end - byte_start, byte_start)
                    if len(payload) != byte_end - byte_start:
                        raise ValueError("short portable UTF8 read")
                    for position in range(begin_at, stop):
                        index = position if sorted_indices is None else int(sorted_indices[position])
                        left, right = int(offsets[index]) - byte_start, int(offsets[index + 1]) - byte_start
                        value = payload[left:right]
                        destination = position if order is None else order[position]
                        flat_target[destination] = ((value if original_dtype.kind == "S" else value.decode("utf-8"))
                                                    if valid[index] else None)
                    begin_at = stop
            return result
        if spec["kind"] == "ragged":
            offsets = self._read(spec["offsets"], None)
            if offsets.shape != (shape[0] + 1,) or int(offsets[0]) != 0:
                raise ValueError("portable ragged offsets differ")
            key = spec["offsets"]["file"]
            if key not in self._validated_offsets:
                if np.any(offsets[1:] < offsets[:-1]):
                    raise ValueError("portable ragged offsets differ")
                self._validated_offsets.add(key)
            if int(offsets[-1]) != spec["values"]["shape"][0]:
                raise ValueError("portable ragged values dimension differs")
            selected = np.arange(shape[0]) if indices is None else indices
            lengths = np.asarray(offsets[selected + 1] - offsets[selected], dtype=np.int64)
            indices = np.concatenate([np.arange(int(offsets[row]), int(offsets[row + 1])) for row in selected]) if len(selected) else np.empty(0, dtype=np.int64)
            values = self._read(spec["values"], indices)
            stops = np.concatenate(([0], np.cumsum(lengths)))
            return [values[stops[index]:stops[index + 1]] for index in range(len(selected))]
        raise ValueError("unknown portable metadata field encoding")

    def sample_ids(self):
        return np.asarray(self._array(self.manifest["sample_ids"]), dtype=str)

    def sample_indices(self, sample_ids):
        requested = np.asarray(sample_ids, dtype=str)
        if requested.ndim != 1 or len(np.unique(requested)) != len(requested):
            raise ValueError("requested sample identifiers must be a unique vector")
        if self._sample_lookup is None:
            available = self.sample_ids()
            order = np.argsort(available)
            self._sample_lookup = available[order], order
        values, rows = self._sample_lookup
        positions = np.searchsorted(values, requested)
        if np.any(positions >= len(values)) or (len(values) and np.any(values[np.minimum(positions, len(values) - 1)] != requested)):
            raise ValueError("requested sample identifier is absent from portable cache")
        return rows[positions].astype(np.int64)

    def read_ref_alt(self, variant_indices):
        alleles = self.read_field("allele", variant_indices)
        ref, alt = np.empty(len(alleles), dtype=object), np.empty(len(alleles), dtype=object)
        for index, value in enumerate(alleles):
            text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
            ref[index], _, alt[index] = text.partition(",")
        return ref, alt

    def read_field(self, path, variant_indices=None):
        from ..genotype import _indices
        indices = None if variant_indices is None else _indices(variant_indices, self.n_variants, "variant_indices")
        if path in ("$ref", "$alt", "$num_allele"):
            selected = np.arange(self.n_variants, dtype=np.int64) if indices is None else indices
            if path == "$num_allele":
                return np.asarray([(value.decode("utf-8") if isinstance(value, bytes) else str(value)).count(",") + 1
                                   for value in self.read_field("allele", selected)], dtype=np.int64)
            ref, alt = self.read_ref_alt(selected)
            return ref if path == "$ref" else alt
        if path not in self.manifest["fields"]:
            raise KeyError("field is absent from portable metadata")
        started = time.perf_counter()
        values = self._read(self.manifest["fields"][path], indices)
        self._metadata_reads += 1
        self._metadata_seconds += time.perf_counter() - started
        return values

    def annotation_cache_source(self, fields):
        """Bind optional derived masks to authenticated original annotations.

        This verifies the files without decoding strings. Repeated requests use
        the existing identity-bracketed verification, not another content scan.
        Unchecked readers cannot authorize persistent annotation reuse.
        """
        if not self._verify_checksums:
            self._annotation_cache_metrics["disabled_unverified_source"] = True
            return None
        names = set()
        def files(spec):
            if "file" in spec:
                names.add(spec["file"])
            for value in spec.values():
                if isinstance(value, dict):
                    files(value)
        selected = sorted(set(fields))
        for field in selected:
            if field not in self.manifest["fields"]:
                return None  # The original selector reports a missing field.
            files(self.manifest["fields"][field])
        names = tuple(sorted(names))
        def validate():
            for name in names:
                self._verify(name)
        validate()
        return dict(directory=self.directory / ".candidate_annotations",
            binding=dict(metadata_manifest_sha256=self._metadata_manifest_sha256,
                genotype_manifest_sha256=self.manifest["genotype_manifest_sha256"],
                n_variants=self.n_variants,
                fields={field:self.manifest["fields"][field] for field in selected},
                files={name:self.manifest["files"][name] for name in names}),
            validate=validate, metrics=self._annotation_cache_metrics)

    @property
    def reader_metadata(self):
        manifest_checksum = _sha(self.directory / "manifest.json")
        return dict(backend="portable_metadata_mmap", original_genotype_required=False,
            original_genotype_sdk_imported=False, genotype_sdk_fallback_count=0,
            metadata_reads=self._metadata_reads, metadata_read_seconds=self._metadata_seconds,
            metadata_utf8_index_requests=dict(self._utf8_index_metrics),
            metadata_checked_files=self._checked_files,
            metadata_checksum_verification_seconds=self._verification_seconds,
            metadata_checksums_enabled=self._verify_checksums,
            annotation_cache=dict(self._annotation_cache_metrics,
                metrics_scope="reader lifetime; mapped_bytes is cumulative file size, not peak RAM"),
            metadata_checksum_process_delta={key: value-self._sha_baseline[key]
                for key, value in _SHA_METRICS.items()},
            metadata_checksum_metrics_scope="process cumulative delta since this reader opened",
            genotype_manifest_sha256=self.manifest["genotype_manifest_sha256"],
            metadata_manifest_sha256=manifest_checksum)

    def close(self):
        if self._closed:
            return
        try:
            for name in self._identities:
                self._verify(name)
        finally:
            self._arrays.clear()
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise ValueError("portable metadata reader is closed")
        return self

    def __exit__(self, *_):
        self.close()


class PortableGenotypeReader(CachedGenotypeAdapter):
    """Standalone validated chromosome reader with mature Single packing.

    ``supports_resident_minor_blocks`` advertises lossless CUDA dosage
    materialization without a native genotype SDK. The pipeline still admits each
    gene's storage and decoder workspace before selecting this route.
    """
    supports_resident_minor_blocks = True

    def __init__(self, container_directory, metadata_directory=None, *, device="cuda:0",
                 compact_cache_bytes=64 * 2**20, verify_checksums=True,
                 prepared_cache_directory=None, prepared_cache_max_bytes=64 * 2**30,
                 prefetch_depth=0, prefetch_memory_bytes=256 * 2**20,
                 prefetch_processes=0):
        cache = Path(container_directory)
        metadata = PortableMetadataReader(metadata_directory or cache / "metadata", cache,
                                          verify_checksums=verify_checksums)
        container = None
        try:
            container = Container(cache)
            if not np.array_equal(container.samples, metadata._array(metadata.manifest["source_sample_rows"])):
                raise ValueError("portable metadata/cache source sample order differs")
            # Container already proved its original physical axis checksum.
            # Rebind only the adapter facade to portable population positions;
            # CSR exception rows, counts and immutable streams are unchanged.
            class LogicalContainer:
                complete = True

                def __init__(self, original):
                    self.original = original
                    self.samples = np.arange(len(original.samples), dtype=np.int64)
                    self.samples.flags.writeable = False
                    self.index, self.manifest = original.index, original.manifest

                def read_frame(self, index):
                    return self.original.read_frame(index)

                def close(self):
                    self.original.close()

            prepared_cache = None
            process_descriptor = None
            if prepared_cache_directory is not None:
                from .cohort_compact_cache import CohortCompactCache
                # Bind physical population, logical sample axis and exact decoder
                # semantics. Changing a model alone does not invalidate compact
                # genotype data; changing samples or requested variants does.
                import hashlib
                from . import adapter_fast, sparse_decode_fast
                from .. import genotype
                digest = hashlib.sha256()
                for module in (adapter_fast, sparse_decode_fast, genotype):
                    digest.update(Path(module.__file__).read_bytes())
                source_binding = dict(schema_version=1,
                    genotype_manifest=container.manifest,
                    metadata_manifest_sha256=hashlib.sha256(
                        (metadata.directory / "manifest.json").read_bytes()).hexdigest(),
                    decoder_semantics_sha256=digest.hexdigest())
                prepared_cache = CohortCompactCache(prepared_cache_directory, source_binding,
                    max_bytes=prepared_cache_max_bytes)
                process_descriptor = dict(container_directory=str(cache.resolve()),
                    metadata_directory=str(metadata.directory.resolve()),
                    compact_cache_bytes=compact_cache_bytes, verify_checksums=verify_checksums,
                    prepared_cache_directory=str(Path(prepared_cache_directory).resolve()),
                    prepared_cache_max_bytes=prepared_cache_max_bytes,
                    source_binding=source_binding)
            super().__init__(metadata, LogicalContainer(container), device=device,
                             compact_cache_bytes=compact_cache_bytes, own_reader=True,
                             prepared_cache=prepared_cache, prefetch_depth=prefetch_depth,
                             prefetch_memory_bytes=prefetch_memory_bytes,
                             prefetch_processes=prefetch_processes,
                             process_descriptor=process_descriptor)
            if prepared_cache is not None:
                prepared_cache.metrics = self._metrics
        except BaseException:
            if container is not None:
                container.close()
            metadata.close()
            raise

    @property
    def _stage_profiler(self):
        return getattr(self._reader, "_stage_profiler", None)

    @_stage_profiler.setter
    def _stage_profiler(self, value):
        self._reader._stage_profiler = value
