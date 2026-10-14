"""Mask-scoped reuse of native Gaussian raw and kinship-rotated panels.

Each model still uses its own fitted precision and covariate projection. The
mature directed TF32 covariance products are unchanged. This optional owner
has the same preparation capacity as the single-model blocked path and must
be closed at the end of one prepared, immutable host-genotype mask.
"""
from contextlib import AbstractContextManager
import weakref

import numpy as np
import torch

from ._blocked_covariance import _PanelPreparation, _storage_bytes


def _tensor_stamp(value):
    """Identity, geometry and mutation counter, without retaining the tensor."""
    if not isinstance(value, torch.Tensor):
        return None
    try:
        return (id(value), value.data_ptr(), tuple(value.shape), tuple(value.stride()),
                value.dtype, value.device, value._version)
    except RuntimeError:  # Inference tensors cannot certify subsequent mutation.
        return None


def _basis_stamp(model):
    spectrum = getattr(model, 'spectrum', None)
    if spectrum is None:
        return None
    eigen = _tensor_stamp(spectrum.eigenvalues)
    blocks = tuple((_tensor_stamp(rows), _tensor_stamp(rotation))
                   for rows, rotation in spectrum.blocks)
    if eigen is None or any(a is None or b is None for a, b in blocks):
        return None
    return (id(spectrum), eigen, blocks)


def _model_stamp(model):
    states = tuple(_tensor_stamp(getattr(model, name, None)) for name in
                   ('x', 'precision_x', 'fixed_effect_covariance', 'inverse_variance'))
    if any(state is None for state in states):
        return None
    return (_basis_stamp(model), states, _array_stamp(model.sample_ids))


def _array_stamp(value):
    if not isinstance(value, np.ndarray):
        return None
    return (id(value), value.__array_interface__['data'][0], value.shape,
            value.strides, value.dtype)


def _same_tensor(left, right):
    # The shared fitted-array bank usually reaches this identity fast path.
    if left is right:
        return True
    return (left.shape == right.shape and left.stride() == right.stride()
            and left.dtype == right.dtype and left.device == right.device
            and torch.equal(left, right))


class _ModelPreparationView:
    """One model's projections over a mask-owned raw/rotated preparation bank."""
    def __init__(self, bank, model, profiler):
        self._bank = weakref.proxy(bank)
        self._model = weakref.ref(model)
        self._stamp = _model_stamp(model)
        self._closed = False
        self._start = dict(bank._cache.metrics)
        bank._cache.model = weakref.proxy(model)
        bank._cache.profiler = profiler
        bank._cache.metrics['peak_storage_bytes'] = bank._cache.panel_bytes
        self._final_metrics = None

    def _checked_cache(self):
        if self._closed:
            raise RuntimeError('shared panel model view is closed')
        model = self._model()
        try:
            if (model is None or _model_stamp(model) != self._stamp
                    or not self._bank._source_unchanged()):
                raise ValueError('fitted state or shared rotation changed during panel use')
            return self._bank._cache
        finally:
            model = None  # A retained exception must not retain a fitted model.

    def borrow(self, begin, end):
        # Mutation checks use counters and geometry only; no panel-level GPU
        # download, full n-row comparison, or model precision sharing occurs.
        return self._checked_cache().borrow(begin, end)

    def _guard(self, new_bytes):
        return self._checked_cache()._guard(new_bytes)

    def measure(self, phase):
        return self._checked_cache().measure(phase)

    def release_unpinned(self):
        return self._bank.release_unpinned()

    @property
    def metrics(self):
        if self._closed:
            return self._final_metrics
        cache = self._bank._cache
        metrics = {name: value-self._start.get(name, 0)
                   for name, value in cache.metrics.items()}
        metrics.update(capacity_bytes=cache.capacity_bytes,
                       peak_storage_bytes=cache.metrics['peak_storage_bytes'],
                       shared_raw_rotated_storage_bytes_at_model_end=cache.panel_bytes,
                       shared_mask_preparation=True)
        return metrics

    @property
    def panel_bytes(self):
        # After close this view owns no panels. Retained mask-level ownership
        # is reported separately; it is not an allocator/free-memory credit.
        return 0 if self._closed else self._bank._cache.panel_bytes

    @property
    def projection_bytes(self):
        return 0 if self._closed else self._bank._cache.projection_bytes

    @property
    def pins(self):
        return {} if self._closed else self._bank._cache.pins

    def close(self):
        if self._closed:
            return
        cache = self._bank._cache
        if cache.pins:
            raise RuntimeError('cannot close a model view with borrowed panels')
        self._final_metrics = dict(self.metrics)
        self._bank._peak_storage = max(self._bank._peak_storage,
                                       cache.metrics['peak_storage_bytes'])
        cache.projections.clear()
        cache.projection_bytes = 0
        cache.model = cache.profiler = None
        self._bank._model_scopes += 1
        self._bank._active_view = None
        self._model = self._stamp = None
        self._closed = True


class SharedGaussianPanelBank(AbstractContextManager):
    """Bounded preparation owner for one immutable host mask and sample axis.

    Input host storage and the fitted sample IDs are borrowed, not copied.
    Their content must remain unchanged for this mask scope, matching the
    existing prepared-genotype contract. Tensor mutation counters certify the
    shared kinship basis and each active model. No fitted model is retained.
    """
    def __init__(self, model, host, broker, *, variant_tile_size=512,
                 capacity_bytes=None, report=None):
        if type(variant_tile_size) is not int or variant_tile_size < 1:
            raise ValueError('variant_tile_size must be a positive integer')
        if (not isinstance(host, np.ndarray) or host.ndim != 2
                or host.dtype != np.float32 or min(host.shape) < 1):
            raise ValueError('host must be a nonempty two-dimensional FP32 array')
        if capacity_bytes is not None and (type(capacity_bytes) is not int
                                           or capacity_bytes < 0):
            raise ValueError('capacity_bytes must be a nonnegative integer')
        self._host_stamp = _array_stamp(host)
        self._width = min(host.shape[1], variant_tile_size)
        self._limit = min(8*2**30, capacity_bytes) if capacity_bytes is not None else 8*2**30
        self._cache = _PanelPreparation(None, host, broker, self._width, 0)
        self._spectrum = getattr(model, 'spectrum', None)
        self._basis = _basis_stamp(model)
        self._axis = getattr(model, 'sample_ids', None)
        self._device = model.device
        self._certificates = {}
        self._active_view = None
        self._closed = False
        self._model_scopes = self._fallback_scopes = 0
        self._peak_storage = self._peak_capacity = 0
        self._report = report
        self._release_scope = None
        # Construction copies no genotype, rotates no panel, and owns no
        # projection or numeric covariance. The anchor is never retained.
        self._enabled = self._supported_state(model)

    def _supported_state(self, model):
        if (getattr(model, 'family', None) != 'gaussian'
                or getattr(model, 'n_pheno', None) != 1
                or getattr(model, 'use_spa', False)
                or getattr(model, 'matmul_mode', None) != 'tf32'
                or model.device != self._device
                or len(model.sample_ids) != self._cache.host.shape[0]):
            return False
        stamp = _model_stamp(model)
        return (stamp is not None and stamp[0] is not None
                and all(getattr(model, name).dtype == torch.float32 for name in
                        ('x', 'precision_x', 'fixed_effect_covariance', 'inverse_variance')))

    def _source_unchanged(self):
        if self._closed or self._cache.host is None or self._spectrum is None:
            return False
        # The spectrum does not reference its model. Check its actual tensor
        # counter/identity using a lightweight attribute carrier.
        current = (_tensor_stamp(self._spectrum.eigenvalues),
                   tuple((_tensor_stamp(a), _tensor_stamp(b))
                         for a, b in self._spectrum.blocks))
        return (self._basis is not None
                and (id(self._spectrum), *current) == self._basis
                and _array_stamp(self._cache.host) == self._host_stamp)

    def supports(self, model, host):
        """Certify exact axis/rotation once per unchanged fitted-state object."""
        if (not self._enabled or not self._source_unchanged()
                or _array_stamp(host) != self._host_stamp
                or not self._supported_state(model)):
            return False
        stamp = _model_stamp(model)
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

    def panel_cache_factory(self, model, host, broker, width, capacity_bytes, profiler=None):
        """Factory hook for ``blocked_gaussian_covariance``; strict fallback.

        The incoming capacity is the mature live-budget limit computed after
        admitting the covariance output. A shared bank never increases it.
        """
        if self._closed:
            raise RuntimeError('shared panel bank is closed')
        if self._active_view is not None:
            raise RuntimeError('shared panel bank permits one model view at a time')
        if (broker is not self._cache.broker or width != self._width
                or not self.supports(model, host)):
            self._fallback_scopes += 1
            # Optional shared owners retire before allocating an unrelated
            # mature cache; no fictitious credit is applied to the broker.
            self.release_unpinned()
            return _PanelPreparation(model, host, broker, width, capacity_bytes, profiler)
        capacity = min(self._limit, capacity_bytes)
        if self.storage_bytes > capacity:
            self.release_unpinned()
        self._cache.capacity_bytes = capacity
        self._peak_capacity = max(self._peak_capacity, capacity)
        self._cache.metrics['capacity_bytes'] = capacity
        view = _ModelPreparationView(self, model, profiler)
        self._active_view = weakref.ref(view)
        return view

    @property
    def storage_bytes(self):
        return self._cache.panel_bytes+self._cache.projection_bytes

    def release_unpinned(self):
        return self._cache.release_unpinned()

    def __enter__(self):
        if self._closed or self._release_scope is not None:
            raise RuntimeError('shared panel mask scope is closed or already entered')
        register = getattr(self._cache.broker, 'optional_device_cache', None)
        if callable(register):
            # Model-local callbacks end before the statistical tail. Keep a
            # mask callback while retained panels can affect tail/model load.
            self._release_scope = register(self.release_unpinned)
            self._release_scope.__enter__()
        return self

    def close(self):
        if self._closed:
            return
        if self._cache.pins:
            raise RuntimeError('cannot close a mask bank with borrowed panels')
        view = self._active_view() if self._active_view is not None else None
        if view is not None:
            view.close()
        final = dict(self._cache.metrics)
        final.update(capacity_bytes=self._peak_capacity,
                     peak_storage_bytes=max(self._peak_storage,
                                            final['peak_storage_bytes']))
        final.update(model_scopes=self._model_scopes, fallback_scopes=self._fallback_scopes,
                     storage_bytes_before_close=self.storage_bytes,
                     borrowed_basis_storage_bytes=(_storage_bytes(
                         [self._spectrum.eigenvalues]+[tensor for pair in self._spectrum.blocks
                                                      for tensor in pair])
                         if self._spectrum is not None else 0),
                     borrowed_sample_axis_bytes=(self._axis.nbytes
                         if isinstance(self._axis, np.ndarray) else 0),
                     borrowed_host_bytes=self._cache.host.nbytes,
                     covariance_shared=False, precision_shared=False,
                     model_projection_shared=False, prepared_host_copied=False,
                     collection_forced=False)
        if self._release_scope is not None:
            self._release_scope.__exit__(None, None, None)
            self._release_scope = None
        self._cache.close()
        final.update(storage_bytes_after_close=self.storage_bytes,
                     pinned_panels_after_close=len(self._cache.pins))
        if self._report is not None:
            summary = self._report.setdefault('gene_shared_panel_preparation_cache', {})
            for key, value in final.items():
                if isinstance(value, bool):
                    summary[key] = value
                elif key in ('capacity_bytes', 'peak_storage_bytes'):
                    summary[key] = max(summary.get(key, 0), value)
                else:
                    summary[key] = summary.get(key, 0)+value
        self._spectrum = self._axis = self._basis = self._host_stamp = self._report = None
        self._certificates.clear()
        self._active_view = None
        self._closed = True

    def __exit__(self, *exc):
        self.close()
        return False


def shared_gaussian_panels(reference_model, host, broker, *, variant_tile_size=512,
                           capacity_bytes=None, report=None):
    """Create a mask owner; unsupported models use the original factory path."""
    return SharedGaussianPanelBank(reference_model, host, broker,
                                  variant_tile_size=variant_tile_size,
                                  capacity_bytes=capacity_bytes, report=report)
