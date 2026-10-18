"""Bounded leases for immutable null-fit geometry, never phenotype state.

SPDX-License-Identifier: GPL-3.0-only
Keys bind ordered samples and complete input contents. Only designs, exact GRM
subsets and spectra belong here; weights, precision, beta and tau do not.
"""
from __future__ import annotations
from ._cuda_memory import allocator_bytes

from collections import OrderedDict
from contextlib import contextmanager
import copy
from dataclasses import dataclass
import hashlib
import json
import threading

import numpy as np
import torch


def available_cuda_bytes(device):
    """Use the existing allocator cap/live-free policy; CPU has no CUDA cap."""
    device = torch.device(device)
    if device.type != "cuda":
        return None
    from . import tf32
    policy = tf32.execution_metadata()["tf32_memory_guard"]
    allocated, reserved = allocator_bytes(device)
    limit = policy["process_allocated_limit_bytes"]
    reserve = policy["live_cuda_reserve_bytes"]
    with torch.cuda.device(device):
        free, _ = torch.cuda.mem_get_info()
    snapshot = tf32._product_workspace_availability(allocated=allocated,
        reserved=reserved, free=free, limit=limit, reserve=reserve)
    return int(snapshot["available_bytes"])


def guard_cuda_workspace(required_bytes, device, *, context="null-fit structure"):
    """Reject new explicit tensors before allocation; no precision fallback."""
    required_bytes = int(required_bytes)
    available = available_cuda_bytes(device)
    if available is not None and required_bytes > available:
        raise MemoryError(f"{context} requires estimated {required_bytes} new bytes, "
                          f"available {available} under CUDA allocator/live-memory guard")


def _key(kind, arrays, **settings):
    digest = hashlib.sha256()
    digest.update(json.dumps([kind, settings], sort_keys=True,
                             separators=(",", ":")).encode())
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([value.dtype.str, list(value.shape)],
                                 separators=(",", ":")).encode())
        digest.update(memoryview(value).cast("B"))
    return kind, digest.hexdigest()


def _spectrum_versions(spectrum):
    tensors = [spectrum.eigenvalues]
    for rows, rotation in spectrum.blocks:
        tensors.extend((rows, rotation))
    try:
        return tuple((id(value), value._version) for value in tensors)
    except RuntimeError:
        # Inference tensors cannot prove absence of mutation: use uncached.
        return None


@dataclass
class _Entry:
    value: object
    nbytes: int
    leases: int = 0
    versions: tuple | None = None


class FitStructureCache:
    """Worker-local LRU bounded by retained bytes and entry count.

    Active leases cannot be evicted. Oversized entries or entries that cannot
    fit alongside active leases are used uncached. Caller-owned fitted models
    still count in the live CUDA allocator guard after their fit lease ends.
    """
    def __init__(self, *, max_bytes=128 * 1024**2, max_entries=16,
                 eigh_batch_bytes=64 * 1024**2):
        if (type(max_bytes) is not int or max_bytes < 0
                or type(max_entries) is not int or max_entries < 0
                or type(eigh_batch_bytes) is not int or eigh_batch_bytes <= 0):
            raise ValueError("fit structure cache budgets must be nonnegative integers")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.eigh_batch_bytes = eigh_batch_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()
        self._hits = self._misses = self._evictions = 0
        self._active_leases = 0
        self._uncached = 0

    def _can_admit(self, incoming_bytes):
        """Avoid evicting hot geometry for a candidate that still cannot fit."""
        active = [entry for entry in self._entries.values() if entry.leases]
        return (incoming_bytes <= self.max_bytes
                and incoming_bytes + sum(entry.nbytes for entry in active) <= self.max_bytes
                and len(active) < self.max_entries)

    def _evict(self, incoming_bytes):
        for key in list(self._entries):
            if (self._bytes + incoming_bytes <= self.max_bytes
                    and len(self._entries) < self.max_entries):
                break
            entry = self._entries[key]
            if not entry.leases:
                self._bytes -= entry.nbytes
                del self._entries[key]
                self._evictions += 1

    @contextmanager
    def _acquire(self, key, builder, sizeof, *, fingerprint=None):
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and fingerprint is not None:
                if entry.versions != fingerprint(entry.value):
                    if entry.leases:
                        raise RuntimeError("leased fit structure was modified in place")
                    self._bytes -= entry.nbytes
                    del self._entries[key]
                    entry = None
            if entry is not None:
                self._hits += 1
                self._entries.move_to_end(key)
                entry.leases += 1
            else:
                self._misses += 1
                # Source-GRM bytes and raw-design bytes are not the retained
                # subset/design size. Guessing here can evict a reusable spectrum
                # on every trait even when the guessed candidate cannot fit.
                # CUDA builders admit their explicit allocations using live
                # allocator counts; cache admission uses the actual owned size.
                value = builder()
                entry = _Entry(value, sizeof(value), leases=1,
                    versions=fingerprint(value) if fingerprint is not None else None)
                if (self._can_admit(entry.nbytes)
                        and (fingerprint is None or entry.versions is not None)):
                    self._evict(entry.nbytes)
                    self._entries[key] = entry
                    self._bytes += entry.nbytes
                else:
                    self._uncached += 1
            self._active_leases += 1
        try:
            yield entry.value
        finally:
            with self._lock:
                entry.leases -= 1
                self._active_leases -= 1

    @contextmanager
    def acquire_design(self, sample_ids, covariates, names, builder):
        """Reuse only the identical ordered raw design and design algorithm."""
        ids = np.asarray(sample_ids, dtype=str)
        raw = np.asarray(covariates, dtype=np.float64)
        key = _key("design-v1-two-pass-gs", (ids, raw), names=list(names))
        def build():
            x, metadata = builder(raw, names)
            x = np.array(x, dtype=np.float64, copy=True)
            x.flags.writeable = False
            return x, copy.deepcopy(metadata)
        def sizeof(value):
            return value[0].nbytes + len(json.dumps(value[1]).encode())
        def fingerprint(value):
            return _key("design-result-v1", (value[0],), metadata=value[1])
        with self._acquire(key, build, sizeof, fingerprint=fingerprint) as value:
            # A caller can annotate its own metadata without changing the cache.
            yield value[0], copy.deepcopy(value[1])

    @contextmanager
    def acquire_kinship_subset(self, kinship, sample_ids):
        """Bind source GRM contents as well as requested sample order."""
        ids = np.asarray(sample_ids, dtype=str)
        arrays = (kinship.sample_ids, kinship.diagonal, kinship.edge_rows,
                  kinship.edge_cols, kinship.edge_values, ids)
        key = _key("grm-subset-v1", arrays)
        def sizeof(value):
            return sum(getattr(value, name).nbytes for name in
                ("sample_ids", "diagonal", "edge_rows", "edge_cols", "edge_values"))
        def fingerprint(value):
            return _key("grm-subset-result-v1", tuple(getattr(value, name) for name in
                ("sample_ids", "diagonal", "edge_rows", "edge_cols", "edge_values")))
        with self._acquire(key, lambda: kinship.subset(ids), sizeof,
                           fingerprint=fingerprint) as value:
            yield value

    @contextmanager
    def acquire_spectrum(self, sample_ids, diagonal, edge_rows=(), edge_cols=(),
                         edge_values=(), *, device="cpu", dtype=torch.float64,
                         max_block_size=2048):
        """Borrow exact geometry in an independent mutable spectrum shell."""
        from .null_model import KinshipSpectrum
        ids = np.asarray(sample_ids, dtype=str)
        diagonal = np.asarray(diagonal, dtype=np.float64)
        if (ids.ndim != 1 or not len(ids) or ids.shape != diagonal.shape
                or len(np.unique(ids)) != len(ids)):
            raise ValueError("fit spectrum IDs must be unique and match the GRM axis")
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        key = _key("spectrum-v1", (ids, diagonal,
            np.asarray(edge_rows, dtype=np.int64), np.asarray(edge_cols, dtype=np.int64),
            np.asarray(edge_values, dtype=np.float64)), dtype=str(dtype),
            device=str(torch.device(device)), max_block_size=max_block_size,
            eigh_batch_bytes=self.eigh_batch_bytes)
        def sizeof(value):
            tensors = [value.eigenvalues]
            for rows, rotation in value.blocks:
                tensors.extend((rows, rotation))
            return sum(value.numel() * value.element_size() for value in tensors)
        def build():
            return KinshipSpectrum.from_sparse(diagonal, edge_rows, edge_cols,
                edge_values, device=device, dtype=dtype, max_block_size=max_block_size,
                eigh_batch_bytes=self.eigh_batch_bytes)
        with self._acquire(key, build, sizeof, fingerprint=_spectrum_versions) as value:
            # Gaussian set_matmul_mode replaces these fields. It must never
            # downcast the cached FP64 shell used by the next independent fit.
            yield KinshipSpectrum(value.eigenvalues, list(value.blocks))

    def stats(self):
        with self._lock:
            return dict(hits=self._hits, misses=self._misses,
                        evictions=self._evictions, uncached_builds=self._uncached,
                        retained_bytes=self._bytes,
                        entries=len(self._entries),
                        active_leases=self._active_leases,
                        max_bytes=self.max_bytes, max_entries=self.max_entries)

    def close(self):
        with self._lock:
            if self._active_leases:
                raise RuntimeError("cannot close fit structure cache with active leases")
            self._entries.clear()
            self._bytes = 0
