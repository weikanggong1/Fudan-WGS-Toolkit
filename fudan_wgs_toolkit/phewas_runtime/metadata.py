"""Bounded shared metadata reads; no phenotype or association state is cached."""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import numpy as np


class SharedMetadataReader:
    def __init__(self, reader, *, capacity_bytes=256 * 2**20):
        if type(capacity_bytes) is not int or capacity_bytes < 1:
            raise ValueError('metadata cache capacity must be a positive integer')
        self.reader, self.capacity = reader, capacity_bytes
        self._cache, self._bytes = OrderedDict(), 0
        self._pinned = {}
        self._samples = None
        self._cache_admission = True
        self.metrics = dict(reads=0, hits=0, evictions=0, highwater_bytes=0)

    def __getattr__(self, name):
        return getattr(self.reader, name)

    def _key(self, method, field, rows):
        if rows is None:
            axis = None
        else:
            rows = np.asarray(rows)
            if rows.ndim != 1 or rows.dtype.kind not in 'iu':
                raise ValueError('metadata rows must be a one-dimensional integer axis')
            normalized = rows.astype(np.int64, copy=False)
            axis = (len(normalized), hashlib.sha256(normalized.tobytes()).digest())
        return method, field, axis

    def _read(self, key, operation):
        if key in self._cache:
            self.metrics['hits'] += 1
            self._cache.move_to_end(key)
            return self._cache[key][0]
        result = operation()
        self.metrics['reads'] += 1
        arrays = result if isinstance(result, tuple) else (result,)
        size = sum(np.asarray(value).nbytes for value in arrays)
        # Object arrays own strings outside the ndarray pointer buffer.
        for value in arrays:
            a = np.asarray(value)
            if a.dtype.hasobject:
                size += sum(len(str(x).encode()) + 64 for x in a.flat)
        if self._cache_admission and size <= self.capacity:
            while self._cache and self._bytes + size > self.capacity:
                _, (_, old_size) = self._cache.popitem(last=False)
                self._bytes -= old_size
                self.metrics['evictions'] += 1
            self._cache[key] = result, size
            self._bytes += size
            self.metrics['highwater_bytes'] = max(self.metrics['highwater_bytes'], self._bytes)
        return result

    def pin_field(self, field):
        """Share structural metadata once; report this fixed host storage separately."""
        if field not in self._pinned:
            value = self.reader.read_field(field)
            if field == 'position':value = np.asarray(value, dtype=np.int64)
            np.asarray(value).flags.writeable = False
            self._pinned[field] = value
            self.metrics['reads'] += 1
            self._structural_storage()
        return self._pinned[field]

    def _structural_storage(self):
        self.metrics["structural_bytes"] = (sum(np.asarray(x).nbytes for x in self._pinned.values())
            + (0 if self._samples is None else np.asarray(self._samples).nbytes))
        self.metrics["structural_storage_scope"] = "ndarray buffers; object payloads and Python objects excluded"

    def read_field(self, field, rows=None):
        if field in self._pinned:
            self.metrics['hits'] += 1
            if rows is None:return self._pinned[field]
            from ..genotype import _indices
            return self._pinned[field][_indices(rows, self.n_variants, 'metadata_rows')]
        return self._read(self._key('field', field, rows), lambda: self.reader.read_field(field, rows))

    def sample_ids(self):
        if self._samples is None:
            self._samples = self.reader.sample_ids()
            np.asarray(self._samples).flags.writeable = False
            self._structural_storage()
            self.metrics["reads"] += 1
        else:self.metrics["hits"] += 1
        return self._samples

    def read_ref_alt(self, rows=None):
        return self._read(self._key('alleles', None, rows), lambda: self.reader.read_ref_alt(rows))

    @property
    def reader_metadata(self):
        return dict(self.reader.reader_metadata, shared_metadata=dict(self.metrics, capacity_bytes=self.capacity))

    def clear(self):
        self._cache.clear()
        self._pinned.clear()
        self._samples = None
        self._bytes = 0

    @contextmanager
    def pressure_recovery(self):
        """Drop only optional LRU owners and avoid refilling during one retry.

        Pinned fields, canonical samples, underlying mmaps and caller-owned
        views remain valid. Released bytes are diagnostic, never admission
        credit: the host gate re-reads actual live pressure before allocating.
        """
        previous = self._cache_admission
        entries, retained = len(self._cache), self._bytes
        self._cache.clear()
        self._bytes = 0
        self._cache_admission = False
        try:
            self.metrics['evictions'] += entries
            self.metrics['pressure_evictions'] = self.metrics.get('pressure_evictions', 0)+entries
            self.metrics['pressure_evicted_estimated_bytes'] = (
                self.metrics.get('pressure_evicted_estimated_bytes', 0)+retained)
            yield
        finally:
            self._cache_admission = previous
