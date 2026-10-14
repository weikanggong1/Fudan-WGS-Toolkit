"""Read-only shared six-state cache transport for independent phenotypes.

The physical cache keeps partial allele calls. A phenotype obtains its own
source AF/MAC, orientation and ordered sample axis before the existing Single
or gene statistic receives a DeviceMinorBlock. CPU mode is a small contract
oracle; production CUDA uses bounded Triton count/scatter kernels.
"""
from collections import OrderedDict
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, field
import math
import time
import hashlib
import os
import weakref

import numpy as np
import torch

from ..cache_runtime import sparse_decode_fast, store
from ..genotype import _allele_frequency_summary, _indices
from ..genotype_device import DeviceMinorBlock

_KERNELS = None


def _immutable(values):
    a = np.asarray(values, dtype=np.int64)
    return np.frombuffer(a.tobytes(), dtype=np.int64)


def _array_geometry(value):
    return (id(value), value.dtype.str, value.shape, value.strides,
            value.__array_interface__['data'][0], bool(value.flags.writeable))


def _readonly_source(value):
    """Prove immutable backing, rather than trusting a reversible readonly flag.

    A read-only mmap additionally binds its inode/geometry/change timestamps.
    Ordinary readonly views of writable NumPy storage are deliberately excluded.
    """
    if not isinstance(value, np.ndarray) or value.flags.writeable:
        return None
    current, seen = value, set()
    while isinstance(current, np.ndarray):
        if id(current) in seen or current.flags.writeable:
            return None
        seen.add(id(current))
        if isinstance(current, np.memmap) and current.mode == 'r':
            status = os.stat(current.filename)
            return ('readonly-mmap', os.fspath(current.filename), current.offset,
                    status.st_dev, status.st_ino, status.st_size,
                    status.st_mtime_ns, status.st_ctime_ns)
        current = current.base
    return ('immutable-bytes', id(current)) if isinstance(current, bytes) else None


def _stamp(tensor):
    return (id(tensor), tensor._version, tuple(tensor.shape), tensor.dtype, tensor.device)


def _kernels():
    global _KERNELS, triton, tl
    if _KERNELS is not None:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def scatter_raw(O, OFFSETS, R, S, COLS, M, BLOCK: tl.constexpr):
        col = tl.program_id(1)
        source_col = tl.load(COLS + col)
        begin = tl.load(OFFSETS + source_col).to(tl.int64)
        end = tl.load(OFFSETS + source_col + 1).to(tl.int64)
        x = begin + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = x < end
        row = tl.load(R + x, valid, other=0).to(tl.int64)
        state = tl.load(S + x, valid, other=0)
        tl.store(O + row * M + col, state, valid)

    @triton.jit
    def count_trait(S, ROWS, O, M, N, TILES, BLOCK: tl.constexpr):
        tile, col = tl.program_id(0), tl.program_id(1)
        r = tile * BLOCK + tl.arange(0, BLOCK)
        valid = r < N
        source_row = tl.load(ROWS + r, valid, other=0).to(tl.int64)
        state = tl.load(S + source_row * M + col, valid, other=3)
        ref = tl.where(state == 0, 2, tl.where((state == 1) | (state == 4), 1, 0))
        called = tl.where(state < 3, 2, tl.where(state >= 4, 1, 0))
        whole_ref = tl.where(state < 3, 2 - state.to(tl.int32), 0)
        missing = state >= 3
        offset = (col * TILES + tile) * 4
        tl.store(O + offset, tl.sum(tl.where(valid, ref, 0).to(tl.int64), 0))
        tl.store(O + offset + 1, tl.sum(tl.where(valid, called, 0).to(tl.int64), 0))
        tl.store(O + offset + 2, tl.sum(tl.where(valid, whole_ref, 0).to(tl.int64), 0))
        tl.store(O + offset + 3, tl.sum((valid & missing).to(tl.int64), 0))

    @triton.jit
    def materialize_trait(S, ROWS, COLS, FLIP, O, SOURCE_M, M, N, BLOCK: tl.constexpr):
        x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = x < M * N
        col = x % M
        source_col = tl.load(COLS + col, valid, other=0)
        source_row = tl.load(ROWS + x // M, valid, other=0).to(tl.int64)
        state = tl.load(S + source_row * SOURCE_M + source_col, valid, other=3)
        flip = tl.load(FLIP + col, valid, other=0)
        dosage = tl.where(state < 3, tl.where(flip, state, 2 - state), 3)
        tl.store(O + x, dosage, valid)

    _KERNELS = scatter_raw, count_trait, materialize_trait
    return _KERNELS


@dataclass(frozen=True)
class RawStateBlock:
    """Lossless uint8 [physical cache samples, requested variants] states."""
    states: object
    sample_indices: np.ndarray
    variant_indices: np.ndarray
    _seal: object = field(default=None, repr=False, compare=False)
    _state_stamp: object = field(default=None, repr=False, compare=False)

    @property
    def shape(self):
        return tuple(self.states.shape)


@dataclass
class _Axis:
    original: object
    samples: np.ndarray
    cache_rows: np.ndarray
    device_rows: object
    identity_rows: object = None
    single_binding: object = None
    digest: object = None
    pins: int = 0
    host_bytes: int = 0
    aliases: set = field(default_factory=set)
    geometry: object = None
    backing: object = None


@dataclass(frozen=True)
class _SingleAxisBinding:
    """Private proof constructed from this broker's validated immutable axis."""
    broker: object
    axis: object
    seal: object
    sample_geometry: object
    identity_geometry: object

    def validate(self, reader, samples, identity, n):
        broker, axis = self.broker, self.axis
        if (broker._closed or reader._closed or reader._broker is not broker
                or reader._axis is not axis or broker._seal is not self.seal
                or broker._axes_by_object.get(id(axis.samples)) is not axis
                or samples is not axis.samples or identity is not axis.identity_rows
                or n != len(axis.samples)
                or _array_geometry(samples) != self.sample_geometry
                or _readonly_source(samples) != axis.backing
                or _array_geometry(identity) != self.identity_geometry):
            raise ValueError('verified Single sample-axis binding was altered')
        return axis


@dataclass(frozen=True)
class _FrameSummaries:
    offsets: np.ndarray
    ref_ac: np.ndarray
    called: np.ndarray
    n: int


class SharedStateBroker:
    """Share verified frames and CSR H2D across phenotype-specific views.

    ``sample_indices`` means rows of the metadata reader. For a prepared
    population these are logical rows; original source rows remain provenance. ``read_states``
    preserves caller variant order. Its optional MAC bound rejects only sites
    that no cache sample subset can retain; final MAC is always cohort-specific.
    This object never writes cache files and never falls back to SDK genotypes.
    """
    def __init__(self, original_reader, container, *, device='cuda:0',
                 memory_limit_gib=None, device_cache_bytes=512 * 2**20,
                 compact_cache_bytes=64 * 2**20, own_reader=False, own_container=True,
                 portable_axis=False, memory_reserve_bytes=256 * 2**20,
                 sample_axis_cache_bytes=64 * 2**20,
                 sample_axis_host_cache_bytes=64 * 2**20,
                 device_cache_live_fraction=None):
        if not isinstance(container, store.Container) or not container.complete:
            raise ValueError('a complete verified six-state Container is required')
        if memory_limit_gib is not None and (not isinstance(memory_limit_gib, (int, float))
                or isinstance(memory_limit_gib, bool) or not math.isfinite(memory_limit_gib)
                or memory_limit_gib <= 0):
            raise ValueError('memory_limit_gib must be positive finite or None')
        if type(memory_reserve_bytes) is not int or memory_reserve_bytes < 0:
            raise ValueError('memory_reserve_bytes must be a nonnegative integer')
        if type(portable_axis) is not bool:
            raise ValueError('portable_axis must be boolean')
        if device_cache_live_fraction is not None:
            self._validate_fraction(device_cache_live_fraction)
        for value in (device_cache_bytes, compact_cache_bytes, sample_axis_cache_bytes,
                      sample_axis_host_cache_bytes):
            if type(value) is not int or value < 0:
                raise ValueError('shared cache capacities must be nonnegative integers')
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('shared six-state device must be cpu or cuda')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA shared cache requested but CUDA is unavailable')
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self._reader, self._container = original_reader, container
        self._own_reader, self._own_container = own_reader, own_container
        self._limit = None if memory_limit_gib is None else int(memory_limit_gib * 2**30)
        self._reserve = memory_reserve_bytes
        self._device_capacity, self._compact_capacity = device_cache_bytes, compact_cache_bytes
        self._device_capacity_configured = device_cache_bytes
        self._device_cache_live_fraction = device_cache_live_fraction
        if self.device.type == 'cuda' and device_cache_live_fraction is not None:
            self._device_capacity = min(device_cache_bytes,
                int(self._live_available_bytes()*device_cache_live_fraction))
        self._sources, self._device_sources = OrderedDict(), OrderedDict()
        self._source_bytes = self._device_bytes = 0
        self._raw_sources, self._raw_source_bytes, self._raw_capacity = OrderedDict(), 0, 0
        self._raw_scope_active = False
        self._closed, self._seal, self._axes = False, object(), OrderedDict()
        self._axes_by_object, self._axes_by_digest = {}, {}
        self._axis_aliases = OrderedDict()
        self._axis_proofs = OrderedDict()
        self._proof_capacity = 4096
        self._host_axis_capacity, self._host_axis_bytes = sample_axis_host_cache_bytes, 0
        self._alias_capacity = 256
        self._device_axes, self._device_axis_bytes = OrderedDict(), 0
        self._axis_capacity = sample_axis_cache_bytes
        self._identity_rows = None
        self._workspace_admission = None
        self._optional_device_cache_releases = []
        if portable_axis:
            # Explicit legacy opt-in still proves the physical Container rows.
            manifest = getattr(original_reader, 'manifest', None)
            if not isinstance(manifest, dict) or not hasattr(original_reader, '_array'):
                raise ValueError('portable axis requires verified portable metadata')
            source_rows = original_reader._array(manifest['source_sample_rows'])
            if not np.array_equal(source_rows, container.samples):
                raise ValueError('portable source sample rows differ from physical cache')
        # Portable metadata exposes logical prepared-population rows. Original
        # source rows remain in the immutable container/metadata binding proof.
        logical_axis = portable_axis or getattr(original_reader, 'sample_axis_kind', None) == 'prepared_population'
        self.samples = _immutable(np.arange(container.manifest['n'], dtype=np.int64)
                                  if logical_axis else container.samples)
        self._starts = np.asarray(container.index['start'], dtype=np.int64)
        self._sizes = np.asarray(container.index['m'], dtype=np.int64)
        if (self.samples.shape != (container.manifest['n'],)
                or np.any(self.samples < 0) or np.any(self.samples >= original_reader.n_samples)
                or len(np.unique(self.samples)) != len(self.samples)):
            raise ValueError('physical cache sample binding differs from source')
        if (container.manifest['m'] != original_reader.n_variants
                or (len(self._starts) and (self._starts[0] != 0
                    or np.any(self._sizes < 1)
                    or np.any(self._starts[1:] != self._starts[:-1] + self._sizes[:-1])
                    or self._starts[-1] + self._sizes[-1] != original_reader.n_variants))):
            raise ValueError('physical cache variant coverage differs from source')
        self._identity_mapping = np.array_equal(self.samples, np.arange(len(self.samples), dtype=np.int64))
        self._sample_sort = None if self._identity_mapping else np.argsort(self.samples)
        self._sorted_samples = self.samples if self._identity_mapping else self.samples[self._sample_sort]
        self._metrics = dict(frame_reads=0, compact_frame_hits=0, compact_frame_evictions=0,
            gpu_csr_uploads=0, gpu_csr_hits=0, gpu_csr_evictions=0,
            device_csr_uploads=0, device_csr_hits=0,
            csr_h2d_bytes=0, csr_host_copy_bytes=0, index_h2d_bytes=0, h2d_bytes=0,
            raw_slab_merge_calls=0, raw_slab_merge_copy_bytes=0,
            raw_slab_merge_new_bytes_highwater=0, state_read_calls=0,
            requested_variants=0, bound_rejected_variants=0, returned_raw_variants=0,
            trait_block_calls=0, trait_count_calls=0, returned_trait_variants=0,
            bound_sample_axes=0, sample_axis_reuse_calls=0, sample_axis_validation_calls=0,
            sample_axis_digest_calls=0, sample_axis_immutable_alias_hits=0,
            sample_axis_host_bytes_highwater=0, sample_axis_host_evictions=0,
            sample_axis_identity_mapping_aliases=0, sample_axis_borrowed_immutable_axes=0,
            sample_axis_verified_rebinds=0,
            compact_bytes_highwater=0, gpu_csr_bytes_highwater=0,
            sample_axis_device_bytes_highwater=0, sample_axis_device_uploads=0,
            sample_axis_device_evictions=0,
            raw_slab_bytes_highwater=0, memory_required_bytes_highwater=0,
            raw_state_cache_hits=0, raw_state_cache_evictions=0,
            raw_state_cache_bytes_highwater=0, raw_state_materializations=0,
            frame_read_validate_seconds=0., csr_upload_host_seconds=0.,
            raw_materialize_host_seconds=0., trait_count_host_seconds=0.,
            trait_materialize_host_seconds=0., trait_summary_d2h_calls=0,
            trait_summary_d2h_bytes=0, trait_summary_d2h_host_seconds=0.,
            batched_trait_summary_calls=0, batched_trait_summary_traits=0,
            trait_pending_summary_bytes_highwater=0, batched_trait_count_guard_calls=0)

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()

    def _check_open(self):
        if self._closed:
            raise RuntimeError('shared six-state broker is closed')

    @property
    def metrics(self):
        return dict(self._metrics, backend='shared-lossless-six-state-CSR', device=str(self.device),
            compressed_frame_prefetch=dict(getattr(self._container, 'prefetch_metrics', {})),
            compact_cache_limit_bytes=self._compact_capacity,
            gpu_csr_cache_limit_bytes=self._device_capacity,
            gpu_csr_cache_configured_limit_bytes=self._device_capacity_configured,
            gpu_csr_cache_live_fraction=self._device_cache_live_fraction,
            current_compact_cache_bytes=self._source_bytes,
            current_gpu_csr_cache_bytes=self._device_bytes,
            sample_axis_cache_limit_bytes=self._axis_capacity,
            current_sample_axis_device_bytes=self._device_axis_bytes,
            sample_axis_host_cache_limit_bytes=self._host_axis_capacity,
            current_sample_axis_host_bytes=self._host_axis_bytes,
            current_sample_axis_host_count=len(self._axes),
            current_sample_axis_pinned_count=sum(axis.pins > 0 for axis in self._axes.values()),
            current_sample_axis_weak_alias_count=len(self._axis_aliases),
            current_sample_axis_weak_proof_count=len(self._axis_proofs),
            raw_state_cache_limit_bytes=self._raw_capacity,
            current_raw_state_cache_bytes=self._raw_source_bytes,
            current_raw_state_cache_slabs=len(self._raw_sources),
            memory_limit_bytes=self._limit, genotype_sdk_fallback_count=0,
            live_memory_reserve_bytes=self._reserve,
            timer_contract='host times include enqueue; summary D2H includes wait/copy; stages may overlap')

    def _evict_device(self):
        _, (_, size, _) = self._device_sources.popitem(last=False)
        self._device_bytes -= size
        self._metrics['gpu_csr_evictions'] += 1

    def release_device_cache(self):
        """Retire only owned CSR/raw references; active slabs remain valid."""
        released = self._device_bytes+self._raw_source_bytes
        while self._raw_sources:
            self._evict_raw()
        while self._device_sources:
            self._evict_device()
        for release in tuple(self._optional_device_cache_releases):
            released += release()
        return released

    @contextmanager
    def optional_device_cache(self, release):
        """Register a mask-local owner with the existing live pressure guard.

        The callback retires only unpinned references and allocates nothing.
        It gives no credit for still borrowed tensors or allocator fragments.
        """
        if not callable(release):
            raise TypeError('optional cache release must be callable')
        self._optional_device_cache_releases.append(release)
        try:
            yield
        finally:
            self._optional_device_cache_releases.remove(release)

    def _evict_raw(self):
        _, (_, size) = self._raw_sources.popitem(last=False)
        self._raw_source_bytes -= size
        self._metrics['raw_state_cache_evictions'] += 1

    @staticmethod
    def _validate_fraction(value):
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or not 0 < value <= 1):
            raise ValueError('cache live_fraction must be in (0, 1]')

    def _live_available_bytes(self):
        free, total = torch.cuda.mem_get_info(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        unused = max(0, torch.cuda.memory_reserved(self.device)-allocated)
        cap = int(total) if self._limit is None else min(int(total), self._limit)
        return max(0, min(int(free)+unused, cap-allocated)-self._reserve)

    @contextmanager
    def raw_state_cache(self, capacity_bytes, *, live_fraction=.25):
        """Retain bounded physical uint8 slabs only across this genetic job.

        The cache never contains cohort AF/MAC or floating point dosage. It
        allocates nothing up front and yields under live workspace pressure.
        An active caller's slab still counts in the CUDA allocator after its
        cached reference is retired; retirement does not invent free credit.
        """
        self._check_open()
        if type(capacity_bytes) is not int or capacity_bytes < 0:
            raise ValueError('raw-state cache capacity must be a nonnegative integer')
        self._validate_fraction(live_fraction)
        if self._raw_scope_active:
            raise RuntimeError('raw-state cache scopes cannot be nested')
        capacity = capacity_bytes
        if self.device.type == 'cuda':
            capacity = min(capacity, int(self._live_available_bytes()*live_fraction))
        self._raw_capacity, self._raw_scope_active = capacity, True
        try:
            yield self
        finally:
            while self._raw_sources:
                self._evict_raw()
            self._raw_capacity, self._raw_scope_active = 0, False

    @contextmanager
    def workspace_guard(self, admission):
        """Coordinate a job's fitted-state bank with existing incremental guards."""
        if admission is not None and not callable(admission):
            raise TypeError('workspace admission must be callable or None')
        previous = self._workspace_admission
        self._workspace_admission = admission
        try:
            yield
        finally:
            self._workspace_admission = previous

    def _guard(self, required):
        self._metrics['memory_required_bytes_highwater'] = max(
            self._metrics['memory_required_bytes_highwater'], required)
        if self._limit is not None and required > self._limit:
            raise MemoryError('shared six-state workspace exceeds configured memory budget')
        if self.device.type == 'cuda':
            if self._workspace_admission is not None:
                self._workspace_admission(required, phase='shared_genotype_or_covariance')
            # Own LRU entries may be evicted; independent model allocations are
            # only retired through the explicitly installed owner callback.
            while self._raw_sources or self._device_sources:
                allocated = torch.cuda.memory_allocated(self.device)
                unused = max(0, torch.cuda.memory_reserved(self.device)-allocated)
                free, _ = torch.cuda.mem_get_info(self.device)
                if ((self._limit is None or allocated+required+self._reserve <= self._limit)
                        and required <= max(0, free+unused-self._reserve)):
                    break
                if self._raw_sources:
                    self._evict_raw()
                else:
                    self._evict_device()
            allocated = torch.cuda.memory_allocated(self.device)
            unused = torch.cuda.memory_reserved(self.device) - allocated
            free, _ = torch.cuda.mem_get_info(self.device)
            if ((self._limit is not None and allocated + required + self._reserve > self._limit)
                    or required > max(0, free + unused - self._reserve)):
                raise MemoryError('shared six-state workspace exceeds live CUDA budget')

    def _source(self, frame):
        if frame in self._sources:
            self._metrics['compact_frame_hits'] += 1
            self._sources.move_to_end(frame)
            return self._sources[frame][0]
        if frame in self._device_sources:
            # Tiny source summaries remain with the device CSR. They do not
            # retain the large decompressed exception buffers after CPU LRU
            # eviction, and avoid another disk/decompress cycle on GPU hits.
            return self._device_sources[frame][2]
        started = time.perf_counter()
        f = self._container.read_frame(int(frame))
        prefetch = getattr(self._container, 'prefetch', None)
        following = int(frame)+1
        if (callable(prefetch) and following < len(self._starts)
                and following not in self._sources and following not in self._device_sources):
            prefetch(following)
        if (f['start'] != self._starts[frame] or f['m'] != self._sizes[frame]
                or f['n'] != len(self.samples)):
            raise ValueError('verified frame geometry differs from cache index')
        source = sparse_decode_fast.validate_source(f['offsets'], f['sample_index'], f['state'],
            f['reference_alleles'], f['called_alleles'], int(f['n']))
        self._metrics['frame_reads'] += 1
        self._metrics['frame_read_validate_seconds'] += time.perf_counter() - started
        size = sum(a.nbytes for a in (source.offsets, source.sample_index, source.state,
                                      source.ref_ac, source.called))
        if size <= self._compact_capacity:
            while self._sources and self._source_bytes + size > self._compact_capacity:
                _, (_, old) = self._sources.popitem(last=False)
                self._source_bytes -= old
                self._metrics['compact_frame_evictions'] += 1
            self._sources[frame] = (source, size)
            self._source_bytes += size
            self._metrics['compact_bytes_highwater'] = max(
                self._metrics['compact_bytes_highwater'], self._source_bytes)
        return source

    def _upload(self, frame, source):
        if frame in self._device_sources:
            self._metrics['device_csr_hits'] += 1
            if self.device.type == 'cuda':
                self._metrics['gpu_csr_hits'] += 1
            self._device_sources.move_to_end(frame)
            return self._device_sources[frame][0]
        if isinstance(source, _FrameSummaries):
            # A live-budget eviction between metadata selection and upload
            # may remove the CSR. Revalidate its immutable frame if needed.
            source = self._source(frame)
        started = time.perf_counter()
        payload_bytes = source.offsets.nbytes + source.sample_index.nbytes + source.state.nbytes
        self._guard(payload_bytes + 64 * 2**20)
        payload = tuple(torch.from_numpy(np.array(a, copy=True)).to(self.device)
                        for a in (source.offsets, source.sample_index, source.state))
        self._metrics['csr_host_copy_bytes'] += payload_bytes
        self._metrics['device_csr_uploads'] += 1
        if self.device.type == 'cuda':
            self._metrics['gpu_csr_uploads'] += 1
            self._metrics['csr_h2d_bytes'] += payload_bytes
            self._metrics['h2d_bytes'] += payload_bytes
        self._metrics['csr_upload_host_seconds'] += time.perf_counter() - started
        if payload_bytes <= self._device_capacity:
            while self._device_sources and self._device_bytes + payload_bytes > self._device_capacity:
                self._evict_device()
            metadata = _FrameSummaries(source.offsets, source.ref_ac, source.called, source.n)
            self._device_sources[frame] = (payload, payload_bytes, metadata)
            self._device_bytes += payload_bytes
            self._metrics['gpu_csr_bytes_highwater'] = max(
                self._metrics['gpu_csr_bytes_highwater'], self._device_bytes)
        return payload

    def _indices_to_device(self, values):
        a = np.asarray(values)
        result = torch.from_numpy(np.array(a, copy=True)).to(self.device)
        if self.device.type == 'cuda':
            self._metrics['index_h2d_bytes'] += a.nbytes
            self._metrics['h2d_bytes'] += a.nbytes
        return result

    @staticmethod
    def _minimum_mac(value):
        if value is not None and (not np.isscalar(value) or not np.isfinite(value) or value < 0):
            raise ValueError('minimum MAC must be finite and nonnegative')

    def read_states(self, variant_indices, minimum_mac_bound=None):
        """Read bounded physical state slabs, sharing verified CSR transport.

        A subset can have at most ``min(full_REF, full_ALT + 1)`` initial MAC.
        The extra allele preserves the original odd called-count rounding.
        """
        self._check_open()
        self._minimum_mac(minimum_mac_bound)
        variants = _indices(variant_indices, self._reader.n_variants, 'variant_indices')
        self._metrics['state_read_calls'] += 1
        self._metrics['requested_variants'] += len(variants)
        raw_key = (variants.tobytes(), minimum_mac_bound) if self._raw_capacity else None
        cached = self._raw_sources.get(raw_key) if raw_key is not None else None
        if cached is not None:
            raw = cached[0]
            self._validate_raw(raw)
            self._raw_sources.move_to_end(raw_key)
            self._metrics['raw_state_cache_hits'] += 1
            self._metrics['bound_rejected_variants'] += len(variants)-len(raw.variant_indices)
            self._metrics['returned_raw_variants'] += len(raw.variant_indices)
            return raw
        frames = np.searchsorted(self._starts, variants, side='right') - 1
        parts, positions = [], []
        for frame in np.unique(frames):
            source = self._source(int(frame))
            request_positions = np.flatnonzero(frames == frame)
            columns = variants[request_positions] - self._starts[frame]
            if minimum_mac_bound is not None:
                upper = np.minimum(source.ref_ac[columns], source.called[columns] - source.ref_ac[columns] + 1)
                keep = upper >= minimum_mac_bound
                self._metrics['bound_rejected_variants'] += int((~keep).sum())
                columns, request_positions = columns[keep], request_positions[keep]
            if not len(columns):
                continue
            cells = len(self.samples) * len(columns)
            if cells > np.iinfo(np.int32).max:
                raise MemoryError('raw six-state slab exceeds validated int32 geometry')
            self._guard(cells * 2 + 64 * 2**20)
            payload = self._upload(int(frame), source)
            self._guard(cells + 64 * 2**20)
            started = time.perf_counter()
            output = torch.zeros((len(self.samples), len(columns)), dtype=torch.uint8, device=self.device)
            c = self._indices_to_device(columns)
            offsets, r, state = payload
            if len(state):
                if self.device.type == 'cuda':
                    scatter, _, _ = _kernels()
                    largest_column = int(np.max(np.diff(source.offsets)[columns], initial=0))
                    with torch.cuda.device(self.device):
                        if largest_column:
                            scatter[(triton.cdiv(largest_column, 256), len(columns))](
                                output, offsets, r, state, c, len(columns), BLOCK=256)
                else:
                    for destination, column in enumerate(columns):
                        begin, end = map(int, source.offsets[column:column+2])
                        output[r[begin:end].to(torch.int64), destination] = state[begin:end]
            self._metrics['raw_materialize_host_seconds'] += time.perf_counter() - started
            self._metrics['raw_slab_bytes_highwater'] = max(
                self._metrics['raw_slab_bytes_highwater'], output.numel())
            parts.append(output)
            positions.append(request_positions)
            # The LRU owns cached payloads. A stale loop-local alias must not
            # retain an evicted CSR through later-frame/merge admission.
            del payload, offsets, r, state, c, output, source
        if parts:
            all_positions = np.concatenate(positions)
            order = np.argsort(all_positions)
            if len(parts) == 1 and np.array_equal(order, np.arange(len(order))):
                states = parts[0]
            else:
                cells = len(self.samples) * len(all_positions)
                # The old cat-then-index_select kept two additional complete
                # slabs live. One destination plus exact uint8 index_copy
                # preserves caller order with only one extra slab. Existing
                # parts already count in the live allocator, not NEW bytes.
                self._guard(cells + 64 * 2**20)
                states = torch.empty((len(self.samples), len(all_positions)),
                                     dtype=torch.uint8, device=self.device)
                destination = np.empty(len(order), dtype=np.int64)
                destination[order] = np.arange(len(order), dtype=np.int64)
                begin = 0
                while parts:
                    part = parts.pop(0)
                    end = begin + part.shape[1]
                    states.index_copy_(1, self._indices_to_device(destination[begin:end]), part)
                    self._metrics['raw_slab_merge_copy_bytes'] += part.numel() * part.element_size()
                    begin = end
                    del part
                self._metrics['raw_slab_merge_calls'] += 1
                self._metrics['raw_slab_merge_new_bytes_highwater'] = max(
                    self._metrics['raw_slab_merge_new_bytes_highwater'], cells)
            selected = variants[all_positions[order]]
        else:
            states = torch.empty((len(self.samples), 0), dtype=torch.uint8, device=self.device)
            selected = np.empty(0, dtype=np.int64)
        self._metrics['returned_raw_variants'] += len(selected)
        self._metrics['raw_state_materializations'] += 1
        raw = RawStateBlock(states, self.samples, _immutable(selected), self._seal, _stamp(states))
        if raw_key is not None:
            size = states.numel()*states.element_size()+raw.variant_indices.nbytes+len(raw_key[0])
            if size <= self._raw_capacity:
                while self._raw_sources and self._raw_source_bytes+size > self._raw_capacity:
                    self._evict_raw()
                self._raw_sources[raw_key] = (raw, size)
                self._raw_source_bytes += size
                self._metrics['raw_state_cache_bytes_highwater'] = max(
                    self._metrics['raw_state_cache_bytes_highwater'], self._raw_source_bytes)
        return raw

    def _touch_axis(self, axis, *, pin=False):
        self._axes.move_to_end(id(axis.samples))
        if pin:
            axis.pins += 1
        # The axis about to be consumed survives this lookup. Active reader
        # views/trait kernels explicitly pin it; inactive history is bounded.
        self._trim_host_axes(protect=axis)
        return axis

    def _forget_axis(self, axis):
        key = id(axis.samples)
        self._axes.pop(key)
        self._axes_by_object.pop(key)
        if self._axes_by_digest.get(axis.digest) is axis:
            self._axes_by_digest.pop(axis.digest)
        for alias_key in tuple(axis.aliases):
            self._axis_aliases.pop(alias_key, None)
        axis.aliases.clear()
        # Single's proof holds this axis and broker. Break that retired cycle
        # immediately rather than waiting for a generational GC traversal.
        axis.single_binding = None
        device_key = id(axis)
        if self._device_axes.pop(device_key, None) is not None:
            self._device_axis_bytes -= axis.cache_rows.nbytes
            axis.device_rows = None
            self._metrics['sample_axis_device_evictions'] += 1
        self._host_axis_bytes -= axis.host_bytes
        self._metrics['sample_axis_host_evictions'] += 1

    def _trim_host_axes(self, *, protect=None):
        while self._host_axis_bytes > self._host_axis_capacity:
            victim = next((axis for axis in self._axes.values()
                           if not axis.pins and axis is not protect), None)
            if victim is None:
                break
            self._forget_axis(victim)

    def _unpin_axis(self, axis):
        if axis.pins <= 0:
            raise RuntimeError('sample-axis pin was released twice')
        axis.pins -= 1
        if not self._closed:
            self._trim_host_axes()

    @contextmanager
    def _acquire_axis(self, samples):
        axis = self._axis(samples, pin=True)
        try:
            yield axis
        finally:
            self._unpin_axis(axis)

    def _axis(self, samples, *, pin=False):
        self._check_open()
        # Preserve the source mmap/array object so its admitted identity can be
        # checked without constructing a new ndarray wrapper on every call.
        candidate = samples if isinstance(samples, np.ndarray) else np.asarray(samples)
        if candidate.ndim != 1 or candidate.dtype.kind not in 'iu':
            raise ValueError('sample_indices must be a one-dimensional integer array')
        # Integer comparison must never pass through floating point. In
        # particular uint64 vs int64 can promote to double in NumPy; reject
        # unrepresentable unsigned values before casting to the canonical axis.
        if candidate.dtype.kind == 'u' and np.any(candidate > np.uint64(np.iinfo(np.int64).max)):
            raise IndexError('sample_indices exceeds the canonical integer dimension')
        candidate_values = candidate.astype(np.int64, copy=False)
        alias = self._axis_aliases.get(id(candidate))
        if alias is not None:
            source, axis, geometry, backing = alias
            if (source() is not candidate or _array_geometry(candidate) != geometry
                    or _readonly_source(candidate) != backing):
                raise ValueError('verified readonly sample source binding was altered')
            self._axis_aliases.move_to_end(id(candidate))
            self._metrics['sample_axis_reuse_calls'] += 1
            self._metrics['sample_axis_immutable_alias_hits'] += 1
            return self._touch_axis(axis, pin=pin)
        axis = self._axes_by_object.get(id(samples))
        if axis is not None and samples is axis.samples:
            if (candidate.dtype != np.int64 or candidate.shape != axis.cache_rows.shape
                    or _array_geometry(candidate) != axis.geometry):
                raise ValueError('immutable sample binding metadata was altered')
            if _readonly_source(candidate) != axis.backing:
                raise ValueError('verified readonly sample source binding was altered')
            self._metrics['sample_axis_reuse_calls'] += 1
            self._metrics['sample_axis_immutable_alias_hits'] += 1
            return self._touch_axis(axis, pin=pin)
        # Different complete-case cohorts must not incur a quadratic scan of
        # every prior N-row axis. The digest only locates a candidate: equality
        # still proves all values before reusing its validated immutable axis.
        source_binding = _readonly_source(candidate)
        proof = self._axis_proofs.get(id(candidate))
        if proof is not None:
            reference, geometry, backing, digest = proof
            if (reference() is not candidate or _array_geometry(candidate) != geometry
                    or source_binding != backing):
                raise ValueError('verified readonly sample source binding was altered')
            self._axis_proofs.move_to_end(id(candidate))
        else:
            digest = (len(candidate_values), hashlib.sha256(candidate_values.tobytes()).digest())
            self._metrics['sample_axis_digest_calls'] += 1
        axis = self._axes_by_digest.get(digest)
        if axis is not None and np.array_equal(candidate_values, axis.samples):
            self._remember_axis_alias(candidate, axis, source_binding)
            self._remember_axis_proof(candidate, source_binding, digest)
            self._metrics['sample_axis_reuse_calls'] += 1
            return self._touch_axis(axis, pin=pin)
        if proof is None:
            self._metrics['sample_axis_validation_calls'] += 1
            selected = _indices(samples, self._reader.n_samples, 'sample_indices')
            positions = np.searchsorted(self._sorted_samples, selected)
            if (np.any(positions >= len(self.samples)) or
                    np.any(self._sorted_samples[np.minimum(positions, max(0, len(self.samples)-1))] != selected)):
                raise ValueError('phenotype samples are outside the physical cache binding')
        else:
            # Only a weak immutable-source proof survives LRU retirement. It
            # owns no cohort array or Axis and prevents repeated N-row scans.
            selected = candidate_values
            positions = None if self._identity_mapping else np.searchsorted(self._sorted_samples, selected)
            self._metrics['sample_axis_verified_rebinds'] += 1
        # Immutable authenticated model rows already own their exact ordering.
        # Reuse that storage, and the same index when cache rows are logical
        # population rows. Readonly flags over mutable storage still get copied.
        canonical = (candidate if candidate.dtype == np.int64 and source_binding is not None
                     else _immutable(selected))
        if canonical is candidate:
            self._metrics['sample_axis_borrowed_immutable_axes'] += 1
        rows = canonical if self._identity_mapping else _immutable(self._sample_sort[positions])
        if rows is canonical:
            self._metrics['sample_axis_identity_mapping_aliases'] += 1
        self._guard(rows.nbytes + 64 * 2**20)
        axis = _Axis(canonical, canonical, rows, None, digest=digest,
            host_bytes=canonical.nbytes+(0 if rows is canonical else rows.nbytes),
            geometry=_array_geometry(canonical), backing=_readonly_source(canonical))
        self._axes[id(axis.samples)] = axis
        self._axes_by_object[id(axis.samples)] = axis
        self._axes_by_digest[digest] = axis
        self._host_axis_bytes += axis.host_bytes
        self._metrics['sample_axis_host_bytes_highwater'] = max(
            self._metrics['sample_axis_host_bytes_highwater'], self._host_axis_bytes)
        self._remember_axis_alias(candidate, axis, source_binding)
        self._remember_axis_proof(candidate, source_binding, digest)
        self._metrics['bound_sample_axes'] += 1
        return self._touch_axis(axis, pin=pin)

    def _remember_axis_proof(self, candidate, backing, digest):
        if backing is None:
            return
        key, owner = id(candidate), weakref.ref(self)
        def expired(reference):
            broker = owner()
            if broker is not None:
                prior = broker._axis_proofs.get(key)
                if prior is not None and prior[0] is reference:
                    broker._axis_proofs.pop(key)
        self._axis_proofs[key] = (weakref.ref(candidate, expired),
            _array_geometry(candidate), backing, digest)
        self._axis_proofs.move_to_end(key)
        while len(self._axis_proofs) > self._proof_capacity:
            self._axis_proofs.popitem(last=False)

    def _remember_axis_alias(self, candidate, axis, source_binding):
        backing = _readonly_source(candidate)
        if backing != source_binding:
            raise ValueError('readonly sample source changed while its axis was verified')
        if backing is not None and candidate is not axis.samples:
            # Weak aliases cannot retain another cohort-sized allocation. The
            # weakref identity and geometry still exclude id reuse and mutation.
            key, owner = id(candidate), weakref.ref(self)
            def expired(reference):
                broker = owner()
                if broker is not None:
                    prior = broker._axis_aliases.get(key)
                    if prior is not None and prior[0] is reference:
                        broker._axis_aliases.pop(key)
                        prior[1].aliases.discard(key)
            prior = self._axis_aliases.pop(key, None)
            if prior is not None:
                prior[1].aliases.discard(key)
            self._axis_aliases[key] = (weakref.ref(candidate, expired), axis,
                _array_geometry(candidate), backing)
            axis.aliases.add(key)
            while len(self._axis_aliases) > self._alias_capacity:
                old_key, old = self._axis_aliases.popitem(last=False)
                old[1].aliases.discard(old_key)

    def _device_rows(self, axis):
        key = id(axis)
        if key in self._device_axes:
            self._device_axes.move_to_end(key)
            return axis.device_rows
        size = axis.cache_rows.nbytes
        # A single oversized axis is allowed; there is no all-cohort device
        # residency. The current cohort's immutable index survives its tile.
        while self._device_axes and self._device_axis_bytes+size > self._axis_capacity:
            _, old = self._device_axes.popitem(last=False)
            self._device_axis_bytes -= old.cache_rows.nbytes
            old.device_rows = None
            self._metrics['sample_axis_device_evictions'] += 1
        self._guard(size + 64*2**20)
        axis.device_rows = self._indices_to_device(axis.cache_rows)
        self._device_axes[key] = axis
        self._device_axis_bytes += size
        self._metrics['sample_axis_device_uploads'] += 1
        self._metrics['sample_axis_device_bytes_highwater'] = max(
            self._metrics['sample_axis_device_bytes_highwater'], self._device_axis_bytes)
        return axis.device_rows

    def _validate_raw(self, raw):
        if (not isinstance(raw, RawStateBlock) or raw._seal is not self._seal
                or raw._state_stamp != _stamp(raw.states)
                or raw.sample_indices is not self.samples or raw.states.dtype != torch.uint8
                or raw.variant_indices.ndim != 1 or raw.variant_indices.dtype != np.int64
                or raw.variant_indices.flags.writeable
                or raw.states.device != self.device or raw.shape != (len(self.samples), len(raw.variant_indices))):
            raise ValueError('unaltered raw block from this shared broker required')

    def _count_device(self, raw, axis, *, guard_counts=True):
        """Keep integer summaries on device; release each trait's partials."""
        n, m = len(axis.samples), raw.shape[1]
        if not m:
            return torch.empty((4, 0), dtype=torch.int64, device=self.device)
        started = time.perf_counter()
        device_rows = self._device_rows(axis)
        if self.device.type == 'cuda':
            _, count, _ = _kernels()
            tiles = triton.cdiv(n, 1024)
            if guard_counts:
                self._guard(m * (max(1, tiles) + 2) * 32 + 64 * 2**20)
            partial = torch.empty((m, tiles, 4), dtype=torch.int64, device=self.device)
            if tiles:
                with torch.cuda.device(self.device):
                    count[(tiles, m)](raw.states, device_rows, partial, m, n, tiles, BLOCK=1024)
            result = partial.sum(1).T.contiguous()
            del partial
        else:
            # Column chunks bound oracle scratch, including table lookups.
            result = torch.empty((4, m), dtype=torch.int64)
            ref = torch.tensor([2, 1, 0, 0, 1, 0], dtype=torch.int64)
            called = torch.tensor([2, 2, 2, 0, 1, 1], dtype=torch.int64)
            for begin in range(0, m, 128):
                end = min(begin + 128, m)
                if guard_counts:
                    self._guard(40 * n * (end-begin) + 64 * 2**20)
                s = raw.states[:, begin:end].index_select(0, device_rows).to(torch.int64)
                values = torch.stack((ref[s].sum(0), called[s].sum(0),
                    torch.where(s < 3, 2-s, 0).sum(0), (s >= 3).sum(0)))
                result[:, begin:end] = values
        self._metrics['trait_count_calls'] += 1
        self._metrics['trait_count_host_seconds'] += time.perf_counter() - started
        return result

    def _summaries_to_cpu(self, counts):
        started = time.perf_counter()
        result = counts.detach().cpu().numpy()
        if self.device.type == 'cuda' and counts.numel():
            self._metrics['trait_summary_d2h_calls'] += 1
            self._metrics['trait_summary_d2h_bytes'] += counts.numel() * counts.element_size()
        self._metrics['trait_summary_d2h_host_seconds'] += time.perf_counter()-started
        return result

    def _count(self, raw, axis):
        return self._summaries_to_cpu(self._count_device(raw, axis))

    def trait_block(self, raw_block, sample_indices, minimum_mac=None):
        """Return an independently oriented cohort block, preserving row order."""
        self._check_open()
        self._validate_raw(raw_block)
        self._minimum_mac(minimum_mac)
        with self._acquire_axis(sample_indices) as axis:
            counts = self._count(raw_block, axis)
            return self._materialize_from_counts(raw_block, axis, counts, minimum_mac)

    def trait_blocks(self, raw_block, sample_axes, minimum_mac=None):
        """Count independent cohorts, then transfer their small summaries once.

        Each existing integer count kernel retains its sample order and
        half-call meanings. Only [trait, four summaries, variants] tensors are
        stacked. No model, genotype imputation or matrix K dimension is merged.
        """
        self._check_open()
        self._validate_raw(raw_block)
        self._minimum_mac(minimum_mac)
        with ExitStack() as pins:
            axes = [pins.enter_context(self._acquire_axis(samples)) for samples in sample_axes]
            return self._trait_blocks_pinned(raw_block, axes, minimum_mac)

    def _trait_blocks_pinned(self, raw_block, axes, minimum_mac):
        self._metrics['batched_trait_summary_calls'] += 1
        self._metrics['batched_trait_summary_traits'] += len(axes)
        if not axes:
            return []
        pending = []
        m = raw_block.shape[1]
        if m:
            largest_axis = max(len(axis.samples) for axis in axes)
            if self.device.type == 'cuda':
                partial = m * (max(1, (largest_axis+1023)//1024) + 2) * 32
            else:
                partial = 40 * largest_axis * min(m, 128) + 2 * m * 32
            # Worst trait partial/reduction plus every retained summary and
            # the stack output. Check once before the owned sequence so live
            # memory queries do not insert a per-trait count synchronization.
            self._guard(partial + 2 * len(axes) * 4 * m * 8 + 64 * 2**20)
            self._metrics['batched_trait_count_guard_calls'] += 1
        for axis in axes:
            pending.append(self._count_device(raw_block, axis, guard_counts=False))
            pending_bytes = len(pending) * 4 * m * 8
            self._metrics['trait_pending_summary_bytes_highwater'] = max(
                self._metrics['trait_pending_summary_bytes_highwater'], pending_bytes)
        # The ownership guard already includes pending summaries and stack.
        stacked = torch.stack(pending)
        counts_cpu = self._summaries_to_cpu(stacked)
        pending.clear()
        del stacked
        return [self._materialize_from_counts(raw_block, axis, counts, minimum_mac)
                for axis, counts in zip(axes, counts_cpu)]

    def _materialize_from_counts(self, raw_block, axis, counts, minimum_mac):
        reference, called, whole_ref, missing = counts
        summaries = _allele_frequency_summary(reference, called, len(axis.samples))
        columns = (np.arange(raw_block.shape[1], dtype=np.int64) if minimum_mac is None
                   else np.flatnonzero(summaries[2] >= minimum_mac))
        af, missing_rate, initial_mac, reference_ac, called_alleles = tuple(a[columns] for a in summaries)
        n, m = len(axis.samples), len(columns)
        if n * m > np.iinfo(np.int32).max:
            raise MemoryError('trait dosage exceeds validated int32 geometry')
        self._guard(n * m * (1 if self.device.type == 'cuda' else 20) + 64 * 2**20)
        started = time.perf_counter()
        dosage = torch.empty((n, m), dtype=torch.uint8, device=self.device)
        if n * m:
            c = self._indices_to_device(columns)
            flip = self._indices_to_device(af >= .5)
            if self.device.type == 'cuda':
                _, _, materialize = _kernels()
                with torch.cuda.device(self.device):
                    materialize[(triton.cdiv(n*m, 256),)](raw_block.states, self._device_rows(axis),
                        c, flip, dosage, raw_block.shape[1], m, n, BLOCK=256)
            else:
                s = raw_block.states.index_select(0, self._device_rows(axis)).index_select(1, c)
                dosage[:] = torch.where(s < 3, torch.where(flip[None, :], s, 2-s), 3)
        result = DeviceMinorBlock(dosage, axis.samples, raw_block.variant_indices[columns],
            af, initial_mac, missing_rate, reference_ac, called_alleles)
        observed = np.where(af >= .5, 2*(n-missing[columns])-whole_ref[columns], whole_ref[columns])
        result._store_counts(np.arange(n, dtype=np.int64), observed.astype(np.float64), missing[columns])
        self._metrics['trait_block_calls'] += 1
        self._metrics['returned_trait_variants'] += m
        self._metrics['trait_materialize_host_seconds'] += time.perf_counter()-started
        return result

    def reader_view(self, trait_rows):
        """Metadata-compatible genotype facade fixed to one ordered cohort axis."""
        self._check_open()
        return SharedTraitReader(self, self._axis(trait_rows, pin=True))

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._sources.clear()
        self._device_sources.clear()
        self._raw_sources.clear()
        self._raw_source_bytes = self._raw_capacity = 0
        for axis in self._axes.values():
            axis.aliases.clear()
            axis.single_binding = None
            axis.device_rows = None
        self._axes.clear()
        self._axes_by_object.clear()
        self._axes_by_digest.clear()
        self._axis_aliases.clear()
        self._axis_proofs.clear()
        self._device_axes.clear()
        self._device_axis_bytes = 0
        self._host_axis_bytes = 0
        self._identity_rows = None
        self._source_bytes = self._device_bytes = 0
        try:
            if self._own_container:
                self._container.close()
        finally:
            if self._own_reader:
                self._reader.close()


class SharedTraitReader:
    """One single-model reader view; closing it never closes another view."""
    def __init__(self, broker, axis):
        self._broker, self._axis, self._closed = broker, axis, False
        self._starts, self._sizes = broker._starts, broker._sizes
        self._device = str(broker.device)
        self._metrics = dict(minor_block_calls=0, single_effective_blocks=0, single_effective_columns=0)

    def __getattr__(self, name):
        return getattr(self._broker._reader, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if not self._closed:
            self._closed = True
            self._broker._unpin_axis(self._axis)

    def _single_axis_binding(self):
        self._broker._check_open()
        if self._closed:
            raise RuntimeError('shared trait reader is closed')
        axis = self._axis
        if axis.identity_rows is None:
            # Every singleton's local rows are 0..N-1. All cohort identities are
            # views of one immutable population-sized rowmap, never N*traits.
            if self._broker._identity_rows is None:
                self._broker._identity_rows = _immutable(np.arange(len(self._broker.samples), dtype=np.int64))
            axis.identity_rows = self._broker._identity_rows[:len(axis.samples)]
        if axis.single_binding is None:
            axis.single_binding = _SingleAxisBinding(self._broker, axis, self._broker._seal,
                _array_geometry(axis.samples), _array_geometry(axis.identity_rows))
        axis.single_binding.validate(self, axis.samples, axis.identity_rows, len(axis.samples))
        return axis.single_binding

    @property
    def reader_metadata(self):
        result = dict(self._broker._reader.reader_metadata)
        result['analysis_cache'] = dict(self._metrics, backend='shared-lossless-six-state-view',
            genotype_sdk_fallback_count=0)
        result['phewas_shared_state'] = self._broker.metrics
        return result

    def read_genotype(self, *args, **kwargs):
        raise RuntimeError('shared cache views never fall back to SDK genotypes')

    def read_ref_dosage(self, *args, **kwargs):
        raise RuntimeError('shared cache views require cohort minor_block')

    def _samples(self, values):
        if self._closed:
            raise RuntimeError('shared trait reader is closed')
        self._broker._check_open()
        a = np.asarray(values)
        # The only admitted axis is already validated and immutable. Equality
        # proves range/uniqueness/order; rerunning np.unique on hundreds of
        # thousands of samples for every mask would repeat the old bottleneck.
        if (a.ndim != 1 or a.dtype.kind not in 'iu' or a.shape != self._axis.samples.shape
                or not np.array_equal(a, self._axis.samples)):
            raise ValueError('trait reader sample order differs from its fixed cohort')
        return self._axis.samples

    def minor_block(self, variant_indices, union_sample_indices, *, device=None,
                    minimum_mac=None, resident=True):
        samples = self._samples(union_sample_indices)
        if device is not None:
            requested = torch.device(device)
            if requested.type == 'cuda' and requested.index is None:
                requested = torch.device('cuda', torch.cuda.current_device())
            if requested != self._broker.device:
                raise ValueError('trait reader device differs from shared broker')
        raw = self._broker.read_states(variant_indices, minimum_mac_bound=minimum_mac)
        result = self._broker.trait_block(raw, samples, minimum_mac=minimum_mac)
        self._metrics['minor_block_calls'] += 1
        return result

    def iter_minor_blocks(self, variant_indices, union_sample_indices, block_size=256, **options):
        if type(block_size) is not int or block_size < 1:
            raise ValueError('block_size must be a positive integer')
        variants = _indices(variant_indices, self.n_variants, 'variant_indices')
        for start in range(0, len(variants), block_size):
            yield self.minor_block(variants[start:start+block_size], union_sample_indices, **options)

    def iter_effective_minor_blocks(self, variant_indices, union_sample_indices, block_size=1024,
                                    *, effective_block_size=1024, **options):
        """Standalone view contract; multi-trait scheduler owns shared packing."""
        if type(effective_block_size) is not int or effective_block_size < 1:
            raise ValueError('effective_block_size must be a positive integer')
        from ..cache_runtime.single_batches import _physical_requests
        variants = _indices(variant_indices, self.n_variants, 'variant_indices')
        if type(block_size) is not int or block_size < 1:
            raise ValueError('block_size must be a positive integer')
        parts, pending = [], 0

        def merge():
            cells = len(self._axis.samples) * sum(part.shape[1] for part in parts)
            if cells > np.iinfo(np.int32).max:
                raise MemoryError('effective dosage exceeds validated int32 geometry')
            self._broker._guard(cells + 64 * 2**20)
            dosage = torch.cat([part.dosage for part in parts], 1)
            result = DeviceMinorBlock(dosage, self._axis.samples,
                np.concatenate([part.variant_indices for part in parts]),
                *[np.concatenate([getattr(part, name) for part in parts]) for name in
                  ('union_ref_af', 'union_initial_mac', 'union_missing_rate', 'union_ref_ac', 'union_called_alleles')])
            self._metrics['single_effective_blocks'] += 1
            self._metrics['single_effective_columns'] += result.shape[1]
            return result

        for request in _physical_requests(self, variants, block_size):
            block = self.minor_block(request, union_sample_indices, **options)
            begin = 0
            while begin < block.shape[1]:
                stop = min(block.shape[1], begin + effective_block_size-pending)
                part = block.select_columns(np.arange(begin, stop, dtype=np.int64))
                parts.append(part)
                pending += stop-begin
                begin = stop
                if pending == effective_block_size:
                    result = merge()
                    parts.clear()
                    pending = 0
                    yield result
                    del result
        if pending:
            yield merge()
