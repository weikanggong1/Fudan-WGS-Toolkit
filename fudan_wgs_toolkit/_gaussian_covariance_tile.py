"""Optional pair-major native Gaussian covariance for at most four models.

Only raw and exactly equivalent kinship-rotated genotype panels are shared.
Every model retains its own precision, covariate projection, two directed
TF32 products and independent covariance output. This is an execution-path
candidate, not an approximation or a mask-size statistical threshold.
"""
from contextlib import contextmanager, nullcontext
import weakref

import numpy as np
import torch

from ._blocked_covariance import _PanelPreparation
from ._gaussian_covariance_shared import SharedGaussianPanelBank, _model_stamp


def _record_report(report, summary):
    """Fixed cumulative counters and at most eight recent anonymous calls."""
    aggregate = report.setdefault('gene_pair_major_covariance', {})
    for name, increment in (('calls', 1), ('completed_calls', int(summary['completed'])),
                            ('admitted_calls', int(summary['admitted'])),
                            ('declined_calls', int(summary['fallback_reason'] is not None)),
                            ('failed_calls', int(summary['error_type'] is not None)),
                            ('model_count', summary['model_count'])):
        aggregate[name] = aggregate.get(name, 0)+increment
    for name in ('panel_pairs', 'directed_products', 'panel_preparations', 'panel_hits',
                 'panel_evictions', 'projection_preparations', 'projection_hits',
                 'projection_evictions', 'storage_bytes_after_close', 'pinned_panels_after_close'):
        aggregate[name] = aggregate.get(name, 0)+summary.get(name, 0)
    for name in ('capacity_bytes', 'peak_storage_bytes', 'new_output_bytes', 'admission_new_bytes'):
        aggregate[name] = max(aggregate.get(name, 0), summary.get(name, 0))
    reasons = aggregate.setdefault('fallback_reasons', {})
    if summary['fallback_reason'] is not None:
        reason = summary['fallback_reason']  # Only the two defined admission/certificate buckets.
        reasons[reason] = reasons.get(reason, 0)+1
    aggregate.update(covariance_shared=False, precision_shared=False,
                     model_projection_shared=False, directed_product_order_preserved=True,
                     collection_forced=False)
    recent = aggregate.setdefault('recent_calls', [])
    recent.append(dict(summary))
    del recent[:-8]


class _BorrowedPreparation(_PanelPreparation):
    """Use the mature projection preparation over already pinned raw panels.

    Borrowed raw storage is owned and counted by the first model's cache.
    Each independent projection cache has an exclusive capacity partition;
    the sum of all partitions never exceeds the original preparation budget.
    """
    @contextmanager
    def pair(self, begin, left, other, right):
        entries = ((begin, left),) if begin == other else ((begin, left), (other, right))
        try:
            for key, payload in entries:
                if key in self.panels or key in self.pins:
                    raise RuntimeError('a borrowed covariance pair is already active')
                self.panels[key] = (payload, 0)
                self.pins[key] = 1  # Protect actual borrowed owners during admission.
            yield self
        finally:
            for key, _ in entries:
                self.panels.pop(key, None)
                self.pins.pop(key, None)
            entries = left = right = payload = _ = None


def gaussian_covariance_tile_workspace_bytes(models, host, *, variant_tile_size=512):
    """NEW independent outputs plus the original bounded two-panel allowance.

    The caller may use this once in its existing workspace/future scope.
    Already allocated model state and cached panels remain actual live owners;
    this estimate does not credit their storage or register another future.
    """
    if type(variant_tile_size) is not int or variant_tile_size < 1:
        raise ValueError('variant_tile_size must be a positive integer')
    if (not isinstance(host, np.ndarray) or host.ndim != 2
            or host.dtype != np.float32 or min(host.shape) < 1):
        raise ValueError('host must be a nonempty two-dimensional FP32 array')
    if not 1 <= len(models) <= 4:
        raise ValueError('a covariance tile must contain one to four models')
    n, m = host.shape
    width = min(m, variant_tile_size)
    scratch = 20*n*width+16*width**2+256*2**20
    return 4*len(models)*m*m+scratch


def blocked_gaussian_covariance_tile(models, host, broker, *, variant_tile_size=512,
                                     profiler=None, report=None, capacity_bytes=None):
    """Return ordered independent covariances, or decline before allocation.

    ``None`` means that exact sample/rotation certification or the combined
    NEW-output admission declined this optional path. The caller then uses
    the mature per-model implementation. Once computation has started, all
    exceptions propagate after closing the panel and projection owners.
    No covariance, inverse precision, or fitted projection is shared.

    Capacity is at most the mature 8 GiB / 15% live-space limit. Other models'
    small projection partitions are reserved within that same limit; the
    remainder belongs to the first model's raw/rotated preparation cache.
    A capacity of zero still shares the current pair across the model tile.
    """
    from .tf32 import matmul
    certificate = raw_cache = scope = release_scope = None
    caches = outputs = stamps = None
    model = cache = covariance = block = reverse = weighted = None
    left_values = right_values = borrowed_left = borrowed_right = None
    left_raw = left_rotated = left_cross = projected_left = None
    right_raw = right_rotated = right_cross = projected_right = None
    left_scope = right_scope = pair_scope = None
    summary = dict(model_count=len(models), covariance_shared=False,
                   precision_shared=False, model_projection_shared=False,
                   directed_product_order_preserved=True, collection_forced=False,
                   admitted=False, completed=False, error_type=None, fallback_reason=None)
    try:
        new_bytes = gaussian_covariance_tile_workspace_bytes(
            models, host, variant_tile_size=variant_tile_size)
        if capacity_bytes is not None and (type(capacity_bytes) is not int
                                           or capacity_bytes < 0):
            raise ValueError('capacity_bytes must be a nonnegative integer')
        models = tuple(models)
        n, m = host.shape
        width = min(m, variant_tile_size)
        scratch = new_bytes-4*len(models)*m*m
        summary.update(new_output_bytes=4*len(models)*m*m,
                       admission_new_bytes=new_bytes, variant_tile_size=width)
        certificate = SharedGaussianPanelBank(models[0], host, broker,
            variant_tile_size=width, capacity_bytes=0)
        if not all(certificate.supports(model, host) for model in models):
            summary['fallback_reason'] = 'incompatible_exact_axis_rotation_or_state'
            return None
        stamps = tuple(_model_stamp(model) for model in models)
        try:
            # Only this preliminary resource refusal selects the sequential
            # fallback. Allocator/kernel errors after admission are not hidden.
            broker._guard(new_bytes)
        except MemoryError:
            summary['fallback_reason'] = 'combined_new_output_admission_declined'
            return None
        summary['admitted'] = True
        outputs = []
        for ordinal in range(len(models)):
            # Integer iteration does not leave a fitted model in the frame of
            # a failed partial-allocation comprehension's retained traceback.
            outputs.append(torch.empty((m, m), dtype=torch.float32, device=models[0].device))
        live = getattr(broker, '_live_available_bytes', None)
        live_capacity = (min(8*2**30, int(max(0, live()-scratch)*.15))
                         if callable(live) else 0)
        limit = min(8*2**30, capacity_bytes) if capacity_bytes is not None else live_capacity
        if callable(live):
            limit = min(limit, live_capacity)
        remaining = limit
        partitions = []
        for model in models[1:]:
            partition = min(remaining, 8*m*model.x.shape[1])
            partitions.append(partition)
            remaining -= partition
        raw_cache = _PanelPreparation(weakref.proxy(models[0]), host, broker,
                                     width, remaining, profiler)
        caches = [raw_cache]+[_BorrowedPreparation(weakref.proxy(model), host, broker,
                   width, partition, profiler)
                   for model, partition in zip(models[1:], partitions)]
        summary.update(admitted=True, capacity_bytes=limit,
                       raw_preparation_capacity_bytes=remaining,
                       independent_projection_capacity_bytes=sum(partitions),
                       peak_storage_bytes=0, panel_pairs=0, directed_products=0)

        def release_unpinned():
            return sum(current.release_unpinned() for current in caches)

        release_scope = getattr(broker, 'optional_device_cache', None)
        scope = release_scope(release_unpinned) if callable(release_scope) else nullcontext()
        with scope:
            for begin in range(0, m, width):
                end = min(begin+width, m)
                with raw_cache.borrow(begin, end) as left_values:
                    for other in range(begin, m, width):
                        stop = min(other+width, m)
                        right_scope = (nullcontext(left_values) if other == begin
                                       else raw_cache.borrow(other, stop))
                        with right_scope as right_values:
                            summary['panel_pairs'] += 1
                            for ordinal, model in enumerate(models):
                                if (not certificate._source_unchanged()
                                        or _model_stamp(model) != stamps[ordinal]):
                                    raise ValueError('fitted state or shared rotation changed during tile use')
                                cache = caches[ordinal]
                                pair_scope = (nullcontext(cache) if ordinal == 0 else
                                    cache.pair(begin, left_values[:2], other, right_values[:2]))
                                with pair_scope:
                                    left_scope = (nullcontext(left_values) if ordinal == 0
                                                  else cache.borrow(begin, end))
                                    with left_scope as borrowed_left:
                                        right_scope_model = (nullcontext(borrowed_left) if other == begin
                                            else nullcontext(right_values) if ordinal == 0
                                            else cache.borrow(other, stop))
                                        with right_scope_model as borrowed_right:
                                            left_raw, left_rotated, left_cross, projected_left = borrowed_left
                                            right_raw, right_rotated, right_cross, projected_right = borrowed_right
                                            cache._guard(8*n*width+16*width**2+256*2**20)
                                            with cache.measure('gene_panel_covariance_products'):
                                                weighted = model.inverse_variance[:, None]*right_rotated
                                                block = (matmul(left_rotated.T, weighted, mode='tf32')
                                                    -matmul(projected_left, right_cross, mode='tf32'))
                                                weighted = None
                                                summary['directed_products'] += 1
                                                if other == begin:
                                                    block = (block+block.T)/2
                                                else:
                                                    weighted = model.inverse_variance[:, None]*left_rotated
                                                    reverse = (matmul(right_rotated.T, weighted, mode='tf32')
                                                        -matmul(projected_right, left_cross, mode='tf32'))
                                                    weighted = None
                                                    block = (block+reverse.T)/2
                                                    reverse = None
                                                    summary['directed_products'] += 1
                                                covariance = outputs[ordinal]
                                                covariance[begin:end, other:stop].copy_(block)
                                                if other != begin:
                                                    covariance[other:stop, begin:end].copy_(block.T)
                                            borrowed_right = right_raw = right_rotated = right_cross = projected_right = block = covariance = None
                                        right_scope_model = None
                                        borrowed_left = left_raw = left_rotated = left_cross = projected_left = None
                                    left_scope = None
                                pair_scope = cache = model = None
                                summary['peak_storage_bytes'] = max(summary['peak_storage_bytes'],
                                    sum(current.panel_bytes+current.projection_bytes for current in caches))
                            right_values = None
                        right_scope = None
                    left_values = None
        if (not certificate._source_unchanged()
                or any(_model_stamp(model) != stamp for model, stamp in zip(models, stamps))):
            raise ValueError('fitted state or shared rotation changed during tile use')
        summary['completed'] = True
        return tuple(outputs)
    except BaseException as error:
        summary['error_type'] = type(error).__name__
        raise
    finally:
        if caches is not None:
            summary.update(panel_preparations=raw_cache.metrics['panel_preparations'],
                           panel_hits=raw_cache.metrics['panel_hits'],
                           panel_evictions=raw_cache.metrics['panel_evictions'],
                           projection_preparations=sum(c.metrics['projection_preparations'] for c in caches),
                           projection_hits=sum(c.metrics['projection_hits'] for c in caches),
                           projection_evictions=sum(c.metrics['projection_evictions'] for c in caches))
            for cache in caches:
                cache.close()
            summary.update(storage_bytes_after_close=sum(c.panel_bytes+c.projection_bytes for c in caches),
                           pinned_panels_after_close=sum(len(c.pins) for c in caches))
        if certificate is not None:
            certificate.close()
        if report is not None:
            _record_report(report, summary)
        caches = outputs = stamps = certificate = raw_cache = scope = release_scope = None
        model = cache = covariance = block = reverse = weighted = None
        left_values = right_values = borrowed_left = borrowed_right = None
        left_raw = left_rotated = left_cross = projected_left = None
        right_raw = right_rotated = right_cross = projected_right = None
        left_scope = right_scope = right_scope_model = pair_scope = None
        models = host = broker = profiler = report = None
