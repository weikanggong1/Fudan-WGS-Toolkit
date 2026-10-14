"""Merge already-complete native SKAT tails across independent model groups.

Only Q vectors and full spectra are accepted. Covariance matrices, genotype
arrays and fitted models are never retained or rebuilt in FP64. All formulas
and exceptional/zero-statistic semantics remain in the mature native tail.
An explicit native FULL preparation can return PendingCompleteTail. Its
finalizer uses the same probability-field/CCT routine as association_test.
"""
from collections import defaultdict
import copy

import torch

_METRICS = dict(requested_groups=0, requested_rows=0, actual_tail_calls=0,
    merged_tail_calls=0, merged_groups=0, original_group_fallbacks=0,
    live_budget_declines=0, explicit_budget_declines=0,
    max_estimated_new_workspace_bytes=0, pending_preparations=0,
    pending_finalizations=0, pending_closes=0)


def _validate_pending_rows(q_values, complete_spectra, pvalues, labels, num_variant):
    weights = 2 * (len(labels) + 1)
    if (type(num_variant) is not int or num_variant < 1
            or not isinstance(q_values, torch.Tensor)
            or not isinstance(complete_spectra, torch.Tensor)
            or not isinstance(pvalues, torch.Tensor)
            or q_values.dtype != torch.float32
            or complete_spectra.dtype != torch.float64
            or pvalues.dtype != torch.float64
            or q_values.shape != (weights,)
            or complete_spectra.shape != (weights, num_variant)
            or pvalues.shape != (3, weights)
            or q_values.device != complete_spectra.device
            or q_values.device != pvalues.device
            or len(set(labels)) != len(labels)
            or any(not isinstance(label, str) or not label for label in labels)):
        raise ValueError('pending native FULL requires complete Q/spectrum/probability rows')
    return weights


class PendingCompleteTail:
    """Small detached FULL-tail inputs; no covariance, score or model owner.

    Only association_test's explicit native FULL preparation creates these
    objects. Q and the two completed elementary probability rows are copied
    as small owners. The already-complete FP64 spectrum is detached without
    rebuilding a covariance. A caller bounds pending spectra across models;
    finish_complete_tails closes every supplied object on success or failure.
    """
    __slots__ = ('q_values', 'complete_spectra', 'pvalues', 'labels',
                 'num_variant', 'cmac', 'tail_optimization',
                 'output_batch_optimization', '_closed')

    def __init__(self, q_values, complete_spectra, pvalues, labels, *,
                 num_variant, cmac, tail_optimization=False,
                 output_batch_optimization=True):
        labels = tuple(labels)
        weights = _validate_pending_rows(q_values, complete_spectra, pvalues, labels, num_variant)
        if (type(tail_optimization) is not bool
                or type(output_batch_optimization) is not bool):
            raise ValueError('pending probability controls must be booleans')
        self.q_values = q_values.detach().clone()
        self.complete_spectra = complete_spectra.detach()
        self.pvalues = torch.empty((3, weights), dtype=torch.float64,
                                  device=pvalues.device)
        self.pvalues[1:].copy_(pvalues[1:].detach())
        # An uncompleted SKAT row is explicit, never uninitialized storage.
        self.pvalues[0].fill_(float('nan'))
        self.labels = labels
        self.num_variant = num_variant
        self.cmac = float(cmac)
        self.tail_optimization = tail_optimization
        self.output_batch_optimization = output_batch_optimization
        self._closed = False
        _METRICS['pending_preparations'] += 1

    @property
    def retained_bytes(self):
        return sum(value.numel()*value.element_size()
                   for value in (self.q_values, self.complete_spectra, self.pvalues)
                   if value is not None)

    def close(self):
        if not self._closed:
            self.q_values = self.complete_spectra = self.pvalues = None
            self._closed = True
            _METRICS['pending_closes'] += 1


def finish_complete_tails(pendings, *, maximum_workspace_bytes=64 * 2**20,
                          maximum_rows=256):
    """Complete original field dictionaries in order, then release all inputs.

    FULL spectrum rows and elementary Burden/ACAT-V probabilities are supplied
    by the unchanged preparation. Only SKAT tail calls may merge. The mature
    CCT/final-field routine retains its original branches and output order.
    An invalid, duplicate, closed or failed pending group is a real failure;
    it is never replaced by an approximate tail or a probability floor.
    """
    from .statistics import _association_probability_fields
    items = list(pendings)
    groups = probabilities = outputs = item = probability = result = None
    try:
        if (any(not isinstance(item, PendingCompleteTail) or item._closed
                for item in items) or len({id(item) for item in items}) != len(items)):
            raise ValueError('complete-tail finalization requires distinct open pending objects')
        for item in items:
            _validate_pending_rows(item.q_values, item.complete_spectra, item.pvalues,
                                   item.labels, item.num_variant)
        groups = [(item.q_values, item.complete_spectra) for item in items]
        probabilities = complete_tail_groups(groups,
            maximum_workspace_bytes=maximum_workspace_bytes, maximum_rows=maximum_rows)
        outputs = []
        for item, probability in zip(items, probabilities):
            item.pvalues[0].copy_(probability)
            result = dict(num_variant=item.num_variant, cMAC=item.cmac)
            result.update(_association_probability_fields(item.pvalues, item.labels,
                tail_optimization=item.tail_optimization,
                batch_copy=item.output_batch_optimization))
            outputs.append(result)
            _METRICS['pending_finalizations'] += 1
        return outputs
    except BaseException as error:
        from ._lobpcg_stream_pool import clear_exception_frames
        clear_exception_frames(error)
        raise
    finally:
        for item in items:
            if isinstance(item, PendingCompleteTail):
                item.close()
        items.clear()
        pendings = groups = probabilities = outputs = item = probability = result = None


def _workspace(rows, variants):
    # Concatenated Q/spectra plus the mature FP64 normalization/root/branch
    # temporaries. This is admission estimation, not a universal allocator peak.
    return rows * (96*variants + 256)


def complete_tail_groups(groups, *, maximum_workspace_bytes=64 * 2**20,
                         maximum_rows=256):
    """Return P vectors in original group order using unchanged complete tails.

Each item is (Q[J], complete_spectra[J,M]), on one device. Q is FP32/FP64;
spectra are the complete FP64 output of the native spectrum stage. Equal M
and devices may merge; neither model parameters nor probabilities are reused.
Limits bound an optional merge. A declined merge evaluates original groups
separately under their caller's normal full-tail workspace budget.
"""
    for name, value in (('maximum_workspace_bytes', maximum_workspace_bytes),
                        ('maximum_rows', maximum_rows)):
        if type(value) is not int or value < 1:
            raise ValueError(name + ' must be a positive integer')
    from .statistics import _saddlepoint_skat_pvalues
    items = list(groups)
    buckets = defaultdict(list)
    outputs = [None] * len(items)
    q_batch = spectrum_batch = probability = q = spectrum = None
    pending = []
    try:
        for index, item in enumerate(items):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError('each complete-tail group must contain Q and full spectra')
            q, spectrum = item
            if (not isinstance(q, torch.Tensor) or not isinstance(spectrum, torch.Tensor)
                    or q.dtype not in (torch.float32, torch.float64)
                    or spectrum.dtype != torch.float64 or spectrum.ndim != 2
                    or min(spectrum.shape) < 1 or q.shape != (spectrum.shape[0],)
                    or q.device != spectrum.device):
                raise ValueError('complete-tail Q/full-spectrum shapes, devices or dtypes differ')
            buckets[(str(spectrum.device), spectrum.shape[1])].append(index)
            _METRICS['requested_groups'] += 1
            _METRICS['requested_rows'] += len(q)

        def flush():
            nonlocal q_batch, spectrum_batch, probability
            if not pending:
                return
            rows = sum(len(items[index][0]) for index in pending)
            variants = items[pending[0]][1].shape[1]
            required = _workspace(rows, variants)
            merge = len(pending) > 1 and required <= maximum_workspace_bytes
            device = items[pending[0]][1].device
            if merge and device.type == 'cuda':
                from ._fit_structure_cache import guard_cuda_workspace
                try:
                    guard_cuda_workspace(required, device,
                                         context='complete native tail-group merge')
                except MemoryError:
                    _METRICS['live_budget_declines'] += 1
                    merge = False
            if merge:
                _METRICS['max_estimated_new_workspace_bytes'] = max(
                    _METRICS['max_estimated_new_workspace_bytes'], required)
                q_batch = torch.cat([items[index][0].double() for index in pending])
                spectrum_batch = torch.cat([items[index][1] for index in pending])
                _METRICS['actual_tail_calls'] += 1
                _METRICS['merged_tail_calls'] += 1
                _METRICS['merged_groups'] += len(pending)
                probability = _saddlepoint_skat_pvalues(q_batch, spectrum_batch)
                offset = 0
                for index in pending:
                    count = len(items[index][0])
                    # Own just the J returned probabilities, not the full merged
                    # root output storage or any spectrum/projection temporary.
                    outputs[index] = probability[offset:offset+count].clone()
                    offset += count
                q_batch = spectrum_batch = probability = None
            else:
                if len(pending) > 1 and required > maximum_workspace_bytes:
                    _METRICS['explicit_budget_declines'] += 1
                for index in pending:
                    _METRICS['actual_tail_calls'] += 1
                    _METRICS['original_group_fallbacks'] += 1
                    outputs[index] = _saddlepoint_skat_pvalues(*items[index])
            pending.clear()

        with torch.no_grad():
            for indices in buckets.values():
                rows = 0
                for index in indices:
                    count = len(items[index][0])
                    variants = items[index][1].shape[1]
                    if pending and (rows+count > maximum_rows
                            or _workspace(rows+count, variants) > maximum_workspace_bytes):
                        flush()
                        rows = 0
                    pending.append(index)
                    rows += count
                flush()
        return outputs
    except BaseException as error:
        from ._lobpcg_stream_pool import clear_exception_frames
        clear_exception_frames(error)
        raise
    finally:
        pending.clear()
        items.clear()
        buckets.clear()
        groups = q_batch = spectrum_batch = probability = q = spectrum = item = None


def execution_metadata(*, reset=False):
    report = copy.deepcopy(_METRICS)
    report.update(method='unchanged complete native Saddle/fourth-moment Gamma per independent row',
        workspace_scope='conservative optional merge estimate; original single-group caller budget retained',
        matrix_products='none; no FP64 covariance or fitted-model reconstruction',
        integration_scope='explicit native FULL pending/finalizer only; normal association output remains immediate')
    if reset:
        for key in _METRICS:
            _METRICS[key] = 0
    return report
