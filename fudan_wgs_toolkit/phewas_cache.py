"""Read completed older PheWAS caches without changing genotype or identity.

The generic main workflow owns the separate schema-2 FID/IID interface. This
module preserves the completed schema-1 cache's ordered positive int64 IDs.
SPDX-License-Identifier: GPL-3.0-only
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat

import numpy as np


_GENOTYPE_STREAM_NAMES = ("data.bin", "headers.bin", "counts.bin")
_STREAM_STAT_FIELDS = ("dev", "ino", "size", "mtime_ns", "ctime_ns")


def _stream_stat(path):
    """A regular-file identity without reading the potentially large stream."""
    try:
        value = Path(path).stat()
    except OSError as error:
        raise ValueError("genotype stream is missing or inaccessible") from error
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("genotype stream must be a regular file")
    return dict(dev=value.st_dev, ino=value.st_ino, size=value.st_size,
                mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)


def _stream_bindings(entries):
    """Bind all three immutable genotype streams across successive attempts.

    Logical absolute paths preserve deliberate cache links. ``stat`` follows
    their targets, so replacing a stream or retargeting a link changes this
    binding even if the new file has the same byte length. These inexpensive
    identities complement the container manifest and per-frame checksums;
    they do not reread or hash chromosome-sized data during continuation.
    """
    bindings = {}
    for entry in entries:
        directory = entry.get("container_directory")
        if not isinstance(directory, (str, os.PathLike)) or not str(directory):
            raise ValueError("genotype stream bindings require a container directory")
        for name in _GENOTYPE_STREAM_NAMES:
            path = os.path.abspath(Path(directory) / name)
            bindings[path] = _stream_stat(path)
    if not bindings:
        raise ValueError("genotype stream bindings require at least one container")
    return dict(sorted(bindings.items()))


def _verify_stream_bindings(bindings):
    """Reject missing, replaced or changed streams before accepting reuse."""
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("genotype stream bindings are missing")
    expected_fields = set(_STREAM_STAT_FIELDS)
    directories = {}
    for path, expected in bindings.items():
        if (not isinstance(path, str) or not Path(path).is_absolute()
                or Path(path).name not in _GENOTYPE_STREAM_NAMES
                or not isinstance(expected, dict) or set(expected) != expected_fields
                or any(type(expected[key]) is not int for key in expected_fields)
                or any(expected[key] < 0 for key in ("dev", "ino", "size"))):
            raise ValueError("genotype stream binding has invalid identity fields")
        directories.setdefault(str(Path(path).parent), set()).add(Path(path).name)
        if _stream_stat(path) != expected:
            raise ValueError("genotype stream changed after the analysis binding")
    if any(names != set(_GENOTYPE_STREAM_NAMES) for names in directories.values()):
        raise ValueError("genotype stream binding must include all three container streams")


def _cache_path(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError("cache manifest paths must be nonempty strings")
    if Path(value).is_absolute():
        raise ValueError("cache manifest paths must be relative to the cache directory")
    # Check the logical tree. Existing deliberate links to transferred caches
    # are allowed; their immutable targets are verified by the portable reader.
    root = Path(root).resolve()
    path = Path(os.path.abspath(root / value))
    if not path.is_relative_to(root):
        raise ValueError("cache manifest paths must remain within the cache directory")
    return path


def _annotation_settings(root, manifest):
    required = ("annotation_catalog", "annotation_names", "qc_path")
    if any(name not in manifest for name in required):
        raise ValueError("legacy cache requires explicit annotation catalog, names and QC path")
    catalog = manifest["annotation_catalog"]
    if isinstance(catalog, str):
        catalog = json.loads(_cache_path(root, catalog).read_text())
    names, qc_path = manifest["annotation_names"], manifest["qc_path"]
    if (not isinstance(catalog, dict) or
            any(not isinstance(key, str) or not key or not isinstance(value, str) or not value
                for key, value in catalog.items())):
        raise ValueError("legacy annotation_catalog must map names to field paths")
    if (not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or set(names) - set(catalog)):
        raise ValueError("legacy annotation_names must be distinct catalog names in weight order")
    if not isinstance(qc_path, str) or not qc_path:
        raise ValueError("legacy qc_path must be a nonempty field path")
    return dict(annotation_catalog=catalog, annotation_names=names, qc_path=qc_path)


def _dataset(cache_directory, chromosomes=None):
    """Validate selected schema-1 caches and return their unchanged sample axis.

    Read only manifests and small sample/metadata arrays, never genotype frames.
    Schema 2 is validated by main and then explicitly rejected for this numeric
    PheWAS input contract; full FID/IID data use the public main workflow.
    """
    root = Path(cache_directory).resolve()
    manifest_path = root / "cache_dataset.json"
    if not manifest_path.exists():
        from .run import _dataset as modern_dataset
        modern_dataset(root, chromosomes)
        raise ValueError("PheWAS numeric inputs cannot consume schema-2 FID/IID sample pairs")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported cache_dataset.json schema_version")
    analysis = _annotation_settings(root, manifest)
    entries = manifest.get("chromosomes")
    if isinstance(entries, dict):
        entries = [dict(value, name=str(key)) for key, value in entries.items()]
    if not isinstance(entries, list) or not entries:
        raise ValueError("cache_dataset.json must declare chromosome entries")
    if isinstance(chromosomes, (str, int)):
        chromosomes = [chromosomes]
    wanted = None if chromosomes is None else {str(value).removeprefix("chr") for value in chromosomes}
    selected, seen = [], set()
    for raw in entries:
        if not isinstance(raw, dict):
            raise ValueError("cache chromosome entries must be mappings")
        entry = dict(raw)
        name = str(entry.get("name", "")).removeprefix("chr")
        if not re.fullmatch(r"[1-9][0-9]*", name) or not 1 <= int(name) <= 22 or name in seen:
            raise ValueError("chromosome entries must have distinct names from 1 to 22")
        seen.add(name)
        if wanted is not None and name not in wanted:
            continue
        container = entry.get("container_directory", entry.get("cache_directory"))
        entry.update(name=name, container_directory=str(_cache_path(root, container)))
        entry["metadata_directory"] = str(_cache_path(root, entry.get("metadata_directory", container + "/metadata")))
        for key in ("gene_catalog", "promoter_intervals"):
            if key in entry:
                entry[key] = str(_cache_path(root, entry[key]))
        selected.append(entry)
    if wanted is not None and wanted - seen:
        raise ValueError("requested chromosomes are absent from the cache")
    if not selected:
        raise ValueError("no chromosomes selected")
    selected.sort(key=lambda entry: int(entry["name"]))
    sample_path = _cache_path(root, manifest.get("sample_ids", "sample_ids.npy"))
    samples = np.load(sample_path, mmap_mode="r", allow_pickle=False)
    if (samples.dtype != np.int64 or samples.ndim != 1 or np.any(samples <= 0)
            or len(np.unique(samples)) != len(samples)):
        raise ValueError("legacy cache sample axis must contain unique positive int64 IDs")
    from .cache_runtime.portable import PortableMetadataReader
    for entry in selected:
        with PortableMetadataReader(entry["metadata_directory"], entry["container_directory"],
                                    verify_checksums=True, legacy_numeric=True) as metadata:
            if metadata.manifest["sample_identifier_format"] != "positive_decimal_int64":
                raise ValueError("legacy PheWAS cache requires its original positive-int64 identity format")
            if not np.array_equal(samples, metadata._array(metadata.manifest["sample_ids"])):
                raise ValueError("legacy cache and chromosome metadata sample axes differ")
            if metadata.manifest.get("analysis") != analysis:
                raise ValueError("legacy dataset and chromosome metadata annotation settings differ")
            required_fields = {"position", "chromosome", "variant.id", "allele", analysis["qc_path"]}
            required_fields.update(analysis["annotation_catalog"].values())
            if not required_fields <= metadata.manifest["fields"].keys():
                raise ValueError("legacy chromosome metadata lacks declared annotation/QC fields")
    return root, manifest_path, manifest, selected, samples


def _analysis_defaults(memory_limit_gib, covariance_block_size, long_mask_threshold, long_mask_rank, seed):
    # Preserve the source-11 PheWAS policy; callers retain its mature per-mask
    # exact/approximation selection rather than silently inheriting new defaults.
    return dict(memory_limit_gib=memory_limit_gib, genotype_block_size=128,
        annotation_block_size=250_000, variant_tile_size=512,
        covariance_backend="cached", cached_variant_tile_size=covariance_block_size,
        long_mask_threshold=long_mask_threshold, long_mask_method="fastskat",
        long_mask_rank=long_mask_rank, long_mask_seed=seed, wrapper_semantics="base", variant_type="variant")


__all__ = ["_dataset", "_cache_path", "_analysis_defaults",
           "_stream_bindings", "_verify_stream_bindings"]
