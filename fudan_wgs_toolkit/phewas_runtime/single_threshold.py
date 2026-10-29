"""Restore strict binary SPA admission using the original physical core.

The guard band requests an arithmetic replay; it never changes the P cutoff.
Only bounded near-boundary Score/variance cells are replaced, before the same
normal P, effect estimates and FP64 SPA are evaluated. No new data cache.
"""
from __future__ import annotations

import math
import numpy as np
import torch

BINARY_SINGLE_THRESHOLD_POLICY = 'binary-single-legacy-physical-cutoff-v1'
BINARY_SINGLE_THRESHOLD_RELATIVE_GUARD = 2e-3


def threshold_host_scratch_bytes(legacy_block_size, traits):
    """Additional bounded host indices/masks/metadata; no host dense dosage."""
    if (type(legacy_block_size) is not int or legacy_block_size < 1
            or type(traits) is not int or traits < 1):
        raise ValueError('threshold replay bounds must be positive integers')
    return legacy_block_size * (512 + 256 * traits)


class LegacySingleRequestPlan:
    """Borrow one job's selected axis and locate its old requests directly.

    Sorted axes use binary search within the original physical frame and raw
    genotype block bound. The uncommon unsorted axis uses the original fixed
    chunks, with a bounded wanted-set scan. Neither route reads preceding
    frames or retains an index/dosage cache. The owner keeps this axis fixed.
    """
    def __init__(self, broker, selected_variants, block_size):
        from ..genotype import _indices
        if type(block_size) is not int or block_size < 1:
            raise ValueError('legacy genotype block size must be a positive integer')
        self.broker = broker
        self.selected = _indices(selected_variants, broker._reader.n_variants, 'variant_indices')
        self.block_size = block_size
        self.sorted = len(self.selected) < 2 or bool(np.all(self.selected[1:] > self.selected[:-1]))

    def requests_for(self, wanted_variants):
        from ..genotype import _indices
        wanted = _indices(wanted_variants, self.broker._reader.n_variants, 'variant_indices')
        if not len(wanted):
            return
        if self.sorted:
            positions = np.searchsorted(self.selected, wanted)
            if (np.any(positions >= len(self.selected))
                    or not np.array_equal(self.selected[positions], wanted)):
                raise ValueError('threshold replay variant is outside the original selected axis')
        else:
            lookup = {int(value): index for index, value in enumerate(wanted)}
            positions = np.full(len(wanted), -1, dtype=np.int64)
            remaining = len(wanted)
            for position, value in enumerate(self.selected):
                index = lookup.get(int(value))
                if index is not None:
                    positions[index] = position
                    remaining -= 1
                    if not remaining:
                        break
            if remaining:
                raise ValueError('threshold replay variant is outside the original selected axis')
        bounds = set()
        for variant, position in zip(wanted, positions):
            position = int(position)
            if self.sorted:
                frame = int(np.searchsorted(self.broker._starts, variant, side='right') - 1)
                lower = int(np.searchsorted(self.selected, self.broker._starts[frame], side='left'))
                upper = int(np.searchsorted(self.selected,
                    self.broker._starts[frame] + self.broker._sizes[frame], side='left'))
                begin = lower + ((position-lower)//self.block_size)*self.block_size
                stop = min(begin+self.block_size, upper)
            else:
                begin = (position//self.block_size)*self.block_size
                stop = min(begin+self.block_size, len(self.selected))
            bounds.add((begin, stop))
        for begin, stop in sorted(bounds):
            yield self.selected[begin:stop]


class LegacySingleThresholdRecheck:
    """Callable normal-core repair for one cross-frame Single job.

    Fits, models, complete-case cohorts, MAC, orientation, missing fill, all
    FP32/TF32 products, whole-trait extreme-tail vector fallback and FP64 SPA
    are unchanged. The one legacy G and all normal replay outputs are released
    before the normal caller enters SPA or publishes a current-order record.
    """
    def __init__(self, broker, selected_variants, *, genotype_block_size,
                 mac_cutoff, p_filter_cutoff=.05, metrics=None):
        if (isinstance(p_filter_cutoff, bool) or not isinstance(p_filter_cutoff, (int, float))
                or not math.isfinite(p_filter_cutoff) or not 0 < p_filter_cutoff <= 1):
            raise ValueError('SPA P cutoff must be finite in (0,1]')
        self.broker = broker
        self.plan = LegacySingleRequestPlan(broker, selected_variants, genotype_block_size)
        self.mac_cutoff = mac_cutoff
        self.p_filter_cutoff = float(p_filter_cutoff)
        self.metrics = {} if metrics is None else metrics
        for name in ('guard_batches', 'guard_control_d2h_bytes', 'near_boundary_cells',
                     'legacy_requests_replayed', 'legacy_variants_replayed', 'restored_cells',
                     'maximum_legacy_request_columns', 'maximum_estimated_new_device_bytes'):
            self.metrics.setdefault(name, 0)

    def __call__(self, pipelines, prepared, scores, variances):
        from ..statistics import _chi1_from_score
        from .single import _shared_single_normal_statistics
        models = [pipeline.models[0] for pipeline in pipelines]
        binary = [index for index, model in enumerate(models) if model.family == 'binomial']
        if not binary:
            return
        selected = np.asarray(prepared['selected'], dtype=np.int64)
        if scores.shape != (len(selected), len(models)) or len(variances) != len(models):
            raise ValueError('threshold replay normal core geometry differs')
        with pipelines[0].profiler.measure('single_binary_threshold_guard', gpu=True):
            nominal = _chi1_from_score(scores.T, torch.stack(variances))
            near = (nominal-self.p_filter_cutoff).abs() <= BINARY_SINGLE_THRESHOLD_RELATIVE_GUARD*self.p_filter_cutoff
            for index, model in enumerate(models):
                if model.family != 'binomial':
                    near[index].zero_()
            near_host = near.detach().cpu().numpy()
        self.metrics['guard_batches'] += 1
        if scores.is_cuda:
            self.metrics['guard_control_d2h_bytes'] += near.numel()*near.element_size()
        self.metrics['near_boundary_cells'] += int(near_host.sum())
        del nominal, near
        wanted_columns = np.flatnonzero(near_host.any(axis=0))
        if not len(wanted_columns):
            return
        remaining = set(map(int, wanted_columns))
        restored = 0
        requests = self.plan.requests_for(selected[wanted_columns])
        raw = block = state = replay = legacy_scores = legacy_variances = None
        try:
            for request in requests:
                width = len(request)
                # All current G/model/core allocations remain live and count
                # in broker admission. Reserve NEW replay G/raw/core only.
                required = (max(pipeline._workspace_estimate(model, width, individual=True)
                    for pipeline, model in zip(pipelines, models))
                    + 2*models[0].n*width + 128*width*len(models) + 64*2**20)
                self.broker._guard(required)
                self.metrics['maximum_estimated_new_device_bytes'] = max(
                    self.metrics['maximum_estimated_new_device_bytes'], required)
                try:
                    with pipelines[0].profiler.measure('single_binary_threshold_legacy_replay', gpu=True):
                        raw = self.broker.read_states(request, minimum_mac_bound=self.mac_cutoff)
                        block = self.broker.trait_block(raw, pipelines[0].union_rows,
                                                       minimum_mac=self.mac_cutoff)
                        state, _ = pipelines[0]._prepare_individual_block(block, 0,
                                                                         mac_cutoff=self.mac_cutoff)
                        replay = (None if state is None else
                                  pipelines[0]._prepare_individual_trait(state, 0, mac_cutoff=self.mac_cutoff))
                        if replay is None:
                            raise ValueError('threshold replay lost an eligible current variant')
                        legacy_scores, legacy_variances = _shared_single_normal_statistics(
                            pipelines, replay['genotype'])
                        lookup = {int(value): index for index, value in enumerate(replay['selected'])}
                        matched = [column for column in remaining if int(selected[column]) in lookup]
                        if not matched:
                            raise ValueError('threshold replay request does not contain a current variant')
                        for trait in binary:
                            columns = [column for column in matched if near_host[trait, column]]
                            if not columns:
                                continue
                            destination = torch.as_tensor(columns, dtype=torch.int64, device=scores.device)
                            source = torch.as_tensor([lookup[int(selected[column])] for column in columns],
                                                     dtype=torch.int64, device=scores.device)
                            scores[:, trait].index_copy_(0, destination, legacy_scores[:, trait].index_select(0, source))
                            variances[trait].index_copy_(0, destination, legacy_variances[trait].index_select(0, source))
                            restored += len(columns)
                            del destination, source
                        remaining.difference_update(matched)
                    self.metrics['legacy_requests_replayed'] += 1
                    self.metrics['legacy_variants_replayed'] += len(replay['selected'])
                    self.metrics['maximum_legacy_request_columns'] = max(
                        self.metrics['maximum_legacy_request_columns'], width)
                finally:
                    raw = block = state = replay = legacy_scores = legacy_variances = None
            if remaining:
                raise ValueError('threshold replay did not restore every near-boundary variant')
            self.metrics['restored_cells'] += restored
        finally:
            requests.close()
            # Keep no request G, fitted model/pipeline or current output owner.
            raw = block = state = replay = legacy_scores = legacy_variances = None
