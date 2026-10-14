"""Optional mask-scoped native TF32 preparation across Gaussian models.

Only the kinship rotation and fitted covariate cross-products are prepared.
Every covariance still uses the original model's inverse precision and fixed
covariate covariance. The row-packed product uses the mature RNA TF32 kernel
and its unchanged ordered BK=32 reduction; real-data acceptance is required
before enabling this candidate in a continuous run.
"""
from contextlib import AbstractContextManager, contextmanager
import weakref

import numpy as np
import torch

from . import tf32
from ._blocked_covariance import _storage_bytes
from ._gaussian_covariance_shared import (
    _array_stamp, _basis_stamp, _model_stamp, _same_tensor, _tensor_stamp,
)


class SharedGaussianResidentPreparation(AbstractContextManager):
    """Borrow one immutable resident genotype; own only bounded preparation.

    ``batched_cross(models)`` returns an aligned tuple of q-by-M tensors, or
    None entries for the mature fallback. ``rotated_for(model)`` similarly
    returns the common rotated G or None. Consumers use ``pin()`` around the
    covariance calculation so pressure retirement preserves active borrowers.
    No fitted model, N-by-M-by-trait tensor, score or covariance is retained.
    """
    def __init__(self, reference_model, genotype, broker, *, capacity_bytes=None,
                 profiler=None, report=None, maximum_batch_size=32):
        if (not isinstance(genotype, torch.Tensor) or genotype.ndim != 2
                or genotype.layout != torch.strided or genotype.dtype != torch.float32
                or min(genotype.shape) < 1):
            raise ValueError('resident genotype must be a nonempty dense FP32 matrix')
        if type(maximum_batch_size) is not int or not 1 <= maximum_batch_size <= 32:
            raise ValueError('maximum_batch_size must be an integer between 1 and 32')
        if capacity_bytes is not None and (type(capacity_bytes) is not int or capacity_bytes < 0):
            raise ValueError('capacity_bytes must be a nonnegative integer')
        live = getattr(broker, '_live_available_bytes', None)
        self.capacity_bytes = (min(8*2**30, int(max(0, live())*.15))
                               if capacity_bytes is None and callable(live)
                               else 0 if capacity_bytes is None else min(8*2**30, capacity_bytes))
        self._genotype, self._broker = genotype, broker
        self._genotype_stamp = _tensor_stamp(genotype)
        self._spectrum = reference_model.spectrum
        self._basis = _basis_stamp(reference_model)
        self._axis = reference_model.sample_ids
        self._axis_stamp = _array_stamp(self._axis)
        self._certificates = {}
        self._rotated = self._cross_output = None
        self._pins = 0
        self._closed = False
        self._release_scope = None
        self._profiler, self._report = profiler, report
        self.maximum_batch_size = maximum_batch_size
        self._metrics = dict(capacity_bytes=self.capacity_bytes, peak_storage_bytes=0,
            rotation_preparations=0, rotation_hits=0, rotation_admission_fallbacks=0,
            cross_batches=0, cross_columns=0, cross_mature_products=0,
            cross_admission_fallbacks=0, cross_route_fallbacks=0,
            cross_geometry_changes=0, largest_trait_batch=0,
            maximum_new_workspace_bytes=0, pressure_releases=0,
            unsupported_model_fallbacks=0)

    def _source_unchanged(self):
        if self._closed or self._genotype_stamp is None or self._basis is None:
            return False
        current = (id(self._spectrum), _tensor_stamp(self._spectrum.eigenvalues),
            tuple((_tensor_stamp(a), _tensor_stamp(b)) for a, b in self._spectrum.blocks))
        return (current == self._basis and _tensor_stamp(self._genotype) == self._genotype_stamp
                and _array_stamp(self._axis) == self._axis_stamp)

    def supports(self, model):
        """Reuse only an exactly certified unchanged sample axis and rotation."""
        if (not self._source_unchanged() or self._genotype.requires_grad
                or getattr(model, 'family', None) != 'gaussian'
                or getattr(model, 'n_pheno', None) != 1 or getattr(model, 'use_spa', False)
                or getattr(model, 'matmul_mode', None) != 'tf32'
                or model.device != self._genotype.device
                or len(model.sample_ids) != self._genotype.shape[0]):
            return False
        stamp = _model_stamp(model)
        if (stamp is None or stamp[0] is None
                or any(getattr(model, name).dtype != torch.float32
                       or getattr(model, name).requires_grad for name in
                       ('x', 'precision_x', 'fixed_effect_covariance', 'inverse_variance'))):
            return False
        previous = self._certificates.get(id(model))
        if previous is not None and previous[0]() is model and previous[1] == stamp:
            return True
        if model.sample_ids is not self._axis and not np.array_equal(model.sample_ids, self._axis):
            return False
        spectrum = model.spectrum
        if (not _same_tensor(spectrum.eigenvalues, self._spectrum.eigenvalues)
                or len(spectrum.blocks) != len(self._spectrum.blocks)
                or any(not _same_tensor(a, c) or not _same_tensor(b, d)
                       for (a, b), (c, d) in zip(spectrum.blocks, self._spectrum.blocks))):
            return False
        self._certificates[id(model)] = (weakref.ref(model), stamp)
        return True

    @property
    def storage_bytes(self):
        rotated = self._rotated
        if (rotated is not None and rotated.untyped_storage().data_ptr()
                == self._genotype.untyped_storage().data_ptr()):
            rotated = None  # A diagonal spectrum borrows the existing G.
        return _storage_bytes((rotated, self._cross_output))

    def _peak(self):
        self._metrics['peak_storage_bytes'] = max(self._metrics['peak_storage_bytes'], self.storage_bytes)

    def _guard(self, required):
        self._metrics['maximum_new_workspace_bytes'] = max(
            self._metrics['maximum_new_workspace_bytes'], required)
        try:
            self._broker._guard(required)
        except MemoryError:
            self.release_unpinned()
            self._broker._guard(required)

    def _measure(self, phase):
        from contextlib import nullcontext
        return (self._profiler.measure(phase, gpu=True) if self._profiler is not None
                else nullcontext())

    def rotated_for(self, model):
        """Borrow the same native rotated matrix; None selects the mature path."""
        if not self.supports(model):
            self._metrics['unsupported_model_fallbacks'] += 1
            return None
        if self._rotated is not None:
            self._metrics['rotation_hits'] += 1
            return self._rotated
        n, m = self._genotype.shape
        retained = 4*n*m if self._spectrum.blocks else 0
        if retained+self.storage_bytes > self.capacity_bytes:
            self._metrics['rotation_admission_fallbacks'] += 1
            return None
        largest_family = max((len(rows) for rows, _ in self._spectrum.blocks), default=0)
        try:
            self._guard(retained+16*largest_family*m+256*2**20)
        except MemoryError:
            self._metrics['rotation_admission_fallbacks'] += 1
            return None
        with self._measure('gene_resident_shared_rotation'):
            self._rotated = self._spectrum.rotate(self._genotype, matmul_mode='tf32')
        self._metrics['rotation_preparations'] += 1
        self._peak()
        return self._rotated

    def batched_cross(self, models):
        """Compute original Px.T @ G with one bounded row-packed TF32 product.

        Singleton GEMV/dot/outer routes retain the separate mature products.
        Failure to admit the extra stack returns None entries, allowing the
        original covariance helper to allocate only its current model's cross.
        Actual numeric/kernel errors propagate; no lower precision is used.
        """
        models = tuple(models)
        packed = output = outputs = model = None
        try:
            count = len(models)
            if count > self.maximum_batch_size:
                raise ValueError('trait count exceeds the bounded resident preparation batch')
            if not count:
                return ()
            if not all(self.supports(model) for model in models):
                self._metrics['unsupported_model_fallbacks'] += count
                return (None,)*count
            if self._cross_output is not None:
                # Only one tile output is optional cache. Callers close their
                # previous tile's borrow before requesting a new batch.
                if self._pins:
                    raise RuntimeError('cannot replace a pinned resident cross batch')
                self._cross_output = None
            n, m = self._genotype.shape
            widths = tuple(model.precision_x.shape[1] for model in models)
            total = sum(widths)
            routes = tuple(tf32._native_route((q, n), (n, m)) for q in widths)
            batched = count > 1 and all(route == 'tf32_mma' for route in routes)
            retained = 4*total*m
            if retained+self.storage_bytes > self.capacity_bytes:
                self._metrics['cross_admission_fallbacks'] += 1
                return (None,)*count
            packed_bytes = 4*n*total if batched else 0
            estimate = (tf32._estimated_product_workspace((total, n), (n, m), mode='tf32',
                        left_dtype=torch.float32, right_dtype=torch.float32)
                        if batched else sum(tf32._estimated_product_workspace((q, n), (n, m),
                            mode='tf32', left_dtype=torch.float32, right_dtype=torch.float32)
                            for q in widths))
            try:
                # Separate vector results coexist briefly with their combined
                # returned output; admit that second output as NEW storage.
                self._guard(packed_bytes+estimate+(0 if batched else retained))
            except MemoryError:
                self._metrics['cross_admission_fallbacks'] += 1
                return (None,)*count
            with self._measure('gene_resident_batched_covariate_cross'):
                if batched:
                    # Exact FP32 fitted input values; no broadcasted NxMxT.
                    packed = torch.cat(tuple(model.precision_x.T for model in models), dim=0)
                    output = tf32.matmul(packed, self._genotype, mode='tf32')
                    self._metrics['cross_batches'] += 1
                    geometry = tf32._native_geometry(total, m, n)
                    self._metrics['cross_geometry_changes'] += sum(
                        tf32._native_geometry(q, m, n) != geometry for q in widths)
                else:
                    # Native vector/outer formulas must not become rounded MMA.
                    outputs = []
                    for model in models:
                        outputs.append(tf32.matmul(model.precision_x.T, self._genotype, mode='tf32'))
                    output = torch.cat(tuple(outputs), dim=0)
                    outputs = None
                    self._metrics['cross_mature_products'] += count
                    self._metrics['cross_route_fallbacks'] += 1
            self._cross_output = output
            self._metrics['cross_columns'] += count
            self._metrics['largest_trait_batch'] = max(self._metrics['largest_trait_batch'], count)
            self._peak()
            return tuple(output.split(widths, dim=0))
        finally:
            # Packed Nx(sum q) workspace is transient even under a retained
            # exception. Never retain the models through a traceback frame.
            models = packed = output = outputs = model = None

    @contextmanager
    def pin(self):
        if self._closed:
            raise RuntimeError('shared resident preparation is closed')
        self._pins += 1
        try:
            yield self
        finally:
            self._pins -= 1

    def release_unpinned(self):
        if self._pins:
            return 0
        released = self.storage_bytes
        self._rotated = self._cross_output = None
        self._metrics['pressure_releases'] += 1
        return released  # Dropped owned references, never CUDA allocator credit.

    def __enter__(self):
        if self._closed or self._release_scope is not None:
            raise RuntimeError('shared resident scope is closed or already entered')
        register = getattr(self._broker, 'optional_device_cache', None)
        if callable(register):
            self._release_scope = register(self.release_unpinned)
            self._release_scope.__enter__()
        return self

    def close(self):
        if self._closed:
            return
        if self._pins:
            raise RuntimeError('cannot close pinned resident preparation')
        metrics = dict(self._metrics, storage_bytes_before_close=self.storage_bytes,
            genotype_copied=False, covariance_shared=False, precision_shared=False,
            collection_forced=False, pinned_after_close=0)
        if self._release_scope is not None:
            self._release_scope.__exit__(None, None, None)
            self._release_scope = None
        self._rotated = self._cross_output = None
        metrics['storage_bytes_after_close'] = self.storage_bytes
        if self._report is not None:
            summary = self._report.setdefault('gene_shared_resident_preparation', {})
            for name, value in metrics.items():
                if isinstance(value, bool): summary[name] = value
                elif name in ('capacity_bytes', 'peak_storage_bytes', 'largest_trait_batch',
                              'maximum_new_workspace_bytes'):
                    summary[name] = max(summary.get(name, 0), value)
                else: summary[name] = summary.get(name, 0)+value
        self._genotype = self._spectrum = self._axis = self._broker = None
        self._profiler = self._report = self._basis = self._genotype_stamp = self._axis_stamp = None
        self._certificates.clear()
        self._closed = True

    def __exit__(self, *exc):
        self.close()
        return False


def shared_gaussian_resident(reference_model, genotype, broker, *, capacity_bytes=None,
                             profiler=None, report=None, maximum_batch_size=32):
    """Create an experimental preparation scope, with the mature fallback."""
    return SharedGaussianResidentPreparation(reference_model, genotype, broker,
        capacity_bytes=capacity_bytes, profiler=profiler, report=report,
        maximum_batch_size=maximum_batch_size)
