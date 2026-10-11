"""Bounded original-weight FastSKAT pairs; no dense operator stack.

SPDX-License-Identifier: GPL-3.0-only
Only independent original LOBPCG solves use fixed persistent streams. FP32
weighting, original Gaussian RNG draw order, refinement and tail remain the
caller's original per-weight operations. Every NEW operator/start/workspace
is included before lookahead materialization.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from ._fit_structure_cache import available_cuda_bytes, guard_cuda_workspace
from ._lobpcg_stream_pool import (LOBPCGWorkerPool, independent_lobpcg,
                                 workspace_estimate, clear_exception_frames, close_preserving_error)


_POOL_SCOPE = ContextVar('wgs-lobpcg-pipeline-owner', default=None)
_WORKSPACE_SCOPE = ContextVar('wgs-lobpcg-evaluation-workspace', default=None)


class _LazyPoolOwner:
    def __init__(self, max_workers):
        if type(max_workers) is not int or not 1 <= max_workers <= 2:
            raise ValueError('LOBPCG pipeline pool requires 1 to 2 workers')
        self.max_workers, self.pool, self.closed = max_workers, None, False
        self._closed_metadata = None
        self._pool_created_count = 0
        self._evaluation_scopes = 0
        self._evaluation_high_water = 0
        self._additional_new_high_water = 0

    def acquire(self):
        if self.closed:
            raise RuntimeError('LOBPCG pipeline owner is closed')
        if self.pool is None:
            self.pool = LOBPCGWorkerPool(self.max_workers)
            self._pool_created_count += 1
        return self.pool

    def metadata(self):
        return dict(closed=self.closed, owner_count=0 if self.closed else 1,
                    pool_created=self._pool_created_count > 0,
                    pool_created_count=self._pool_created_count,
                    active_evaluation_workspace_scopes=self._evaluation_scopes,
                    evaluation_scope_high_water=self._evaluation_high_water,
                    additional_new_peak_estimate_high_water_bytes=self._additional_new_high_water,
                    pool=(self.pool.metadata() if self.pool is not None else self._closed_metadata))

    def close(self):
        self.closed = True
        pool, self.pool = self.pool, None
        try:
            if pool is not None:
                pool.close()
        finally:
            if pool is not None:
                self._closed_metadata = pool.metadata()
            pool = None


@dataclass
class _WorkspaceSettings:
    maximum_new_bytes: int | None
    callback: object


@contextmanager
def lobpcg_pool_scope(*, max_workers=2, owner=None):
    """Pipeline-scoped lazy owner; nested scopes restore the outer owner."""
    owned_here = owner is None
    if owned_here:
        owner = _LazyPoolOwner(max_workers)
    elif not isinstance(owner, _LazyPoolOwner) or owner.closed:
        raise ValueError('LOBPCG borrowed owner must be open and explicitly scoped')
    token = _POOL_SCOPE.set(owner)
    primary = None
    try:
        yield owner
    except BaseException as error:
        primary = error
        raise
    finally:
        _POOL_SCOPE.reset(token)
        if owned_here:
            close_preserving_error(owner.close, primary)
        owner = token = primary = None


@contextmanager
def lobpcg_workspace_scope(*, maximum_new_bytes=None, callback=None):
    """Evaluation-scoped NEW allowance and complete-tail bank callback.

    Callback receives only the nonnegative added weighted-stage demand. Its
    pipeline caller maps original tail_new + addition to its existing lease.
    Rejection must raise MemoryError or return False, selecting original
    per-weight computation. This scope neither raises caps nor owns tensors.
    """
    if maximum_new_bytes is not None and (type(maximum_new_bytes) is not int or maximum_new_bytes < 0):
        raise ValueError('LOBPCG evaluation workspace must be nonnegative bytes')
    if callback is not None and not callable(callback):
        raise ValueError('LOBPCG evaluation callback must be callable')
    settings = _WorkspaceSettings(maximum_new_bytes, callback)
    owner = _POOL_SCOPE.get()
    if owner is not None:
        owner._evaluation_scopes += 1
        owner._evaluation_high_water = max(owner._evaluation_high_water, owner._evaluation_scopes)
    token = _WORKSPACE_SCOPE.set(settings)
    try:
        yield settings
    finally:
        _WORKSPACE_SCOPE.reset(token)
        # Copied thread contexts refer to this same scoped settings object.
        # Clear the callback so no completed Future can retain a lease owner.
        settings.callback = None
        if owner is not None:
            owner._evaluation_scopes -= 1
        settings = token = callback = owner = None


def current_lobpcg_context():
    """Return current internal owners; do not create a pool or cache globally."""
    return _POOL_SCOPE.get(), _WORKSPACE_SCOPE.get()


def tile_peak_estimate(m, k, width, *, native_new_bytes=0):
    if any(type(v) is not int or v <= 0 for v in (m, k, width)) or m < 3*k:
        raise ValueError('FastSKAT tile requires positive geometry and M >= 3*rank')
    dense, start = 4*m*m, 4*m*k
    creation = (width+1)*dense + width*start
    solving = width*(dense+start) + workspace_estimate(m, k, width)['required_bytes']
    # All completed operators/X/Q remain resident only within this bounded
    # tile while one original FP64 refiner executes. Its blockwise temporary
    # casts/QR/output/Ritz/deflation and solver scratch are conservatively
    # included, without constructing a full dense FP64 covariance.
    refinement = width*(dense+2*start) + 64*m*k + 64*k*k + 24*min(256, m)*m
    return dict(required_new_bytes=max(creation, solving, refinement)+native_new_bytes,
                creation_new_bytes=creation, solver_new_bytes=solving,
                refinement_new_bytes=refinement,
                initial_native_handle_workspace_estimate_bytes=native_new_bytes,
                resident_base_covariance_excluded=True,
                separate_dense_weighted_operators=width,
                whole_operator_stack_bytes=0)


def additional_workspace_estimate(m, k, width=2, *, native_new_bytes=0):
    """Extra NEW demand above the original FastSKAT weighted-stage allowance.

    The caller's whole STAAR tail allowance still covers rare/symmetrization,
    ACAT and other original copies. Add only this nonnegative difference to
    that allowance; do not add a second original weighted-stage reservation.
    V and already materialized operators are counted by live residency.
    """
    candidate = tile_peak_estimate(m, k, width, native_new_bytes=native_new_bytes)
    original = 8*m*m+24*m*k
    return dict(additional_new_bytes=max(0, candidate['required_new_bytes']-original),
                original_weighted_stage_new_bytes=original,
                candidate_weighted_stage_new_bytes=candidate['required_new_bytes'],
                candidate=candidate,
                caller_preserves_original_whole_staar_tail_allowance=True,
                resident_covariance_not_recounted=True)


class FastSKATPairTiles:
    """Own lookahead state and a scoped worker pool; consume in weight order.

    A failed batch returns to original serial LOBPCG for the same starts.
    Thus a later weight's batch error cannot outrank an earlier original
    refinement/tail failure. No probability approximation or skip is used.
    The pool can be supplied by a pipeline owner and must be closed by it.
    """
    def __init__(self, covariance, weights, generator, rank, *, worker_pool=None,
                 max_workers=2, max_new_workspace_bytes=None, workspace_callback=None):
        self.covariance, self.weights, self.generator = covariance, weights, generator
        self.rank, self.next_index = int(rank), 0
        self.pending, self.starts, self.records = {}, {}, []
        self.owned_pool = None
        self.pool = worker_pool
        self.scope_owner = None
        scope, settings = current_lobpcg_context()
        self.scope_owner = scope
        if settings is not None:
            if max_new_workspace_bytes is None:
                max_new_workspace_bytes = settings.maximum_new_bytes
            if workspace_callback is None:
                workspace_callback = settings.callback
        self.max_workers = min(2, int(max_workers))
        self.max_new_workspace_bytes = max_new_workspace_bytes
        self.workspace_callback = workspace_callback
        if max_new_workspace_bytes is not None and (
                type(max_new_workspace_bytes) is not int or max_new_workspace_bytes < 0):
            raise ValueError('FastSKAT lookahead workspace allowance must be nonnegative bytes')
        if workspace_callback is not None and not callable(workspace_callback):
            raise ValueError('FastSKAT workspace callback must be callable')
        if self.max_workers < 1:
            raise ValueError('FastSKAT tile requires at least one worker')
        self.supported = (str(torch.__version__).split('+')[0] == '2.5.1'
                          and covariance.is_cuda and covariance.dtype == torch.float32
                          and covariance.shape[0] >= 3*self.rank)
        if self.pool is None and self.supported and scope is not None:
            self.pool = scope.acquire()
        # No per-mask streams or thread owners: only an explicit pipeline
        # scope may enable this factor. Public calls retain original scalar.
        self.supported = self.supported and self.pool is not None

    def _initial(self, index):
        value = self.starts.pop(index, None)
        if value is not None:
            return value
        return torch.randn((self.covariance.shape[0], self.rank),
                           generator=self.generator, device=self.covariance.device,
                           dtype=self.covariance.dtype)

    def _original(self, index):
        weights = weighted = initial = pair = None
        try:
            weights = self.weights[:, index].to(dtype=self.covariance.dtype,
                                                device=self.covariance.device)
            weighted = self.covariance * weights[:, None] * weights[None, :]
            initial = self._initial(index)
            pair = torch.lobpcg(weighted, k=self.rank, B=None, X=initial,
                               largest=True, niter=200, tol=1e-5)
            return weights, weighted, initial, pair[0], pair[1]
        finally:
            weights = weighted = initial = pair = None

    def _width(self, offset):
        if not self.supported:
            return 1, None
        available = available_cuda_bytes(self.covariance.device)
        if self.max_new_workspace_bytes is not None:
            available = min(available, self.max_new_workspace_bytes)
        maximum = min(self.max_workers, self.pool.max_workers,
                      self.weights.shape[1]-offset)
        native = self.pool.initial_native_workspace_estimate(self.covariance.device)
        m = self.covariance.shape[0]
        for width in range(maximum, 1, -1):
            estimate = tile_peak_estimate(m, self.rank, width, native_new_bytes=native)
            if estimate['required_new_bytes'] <= available:
                if self.workspace_callback is not None:
                    addition = additional_workspace_estimate(m, self.rank, width,
                                                               native_new_bytes=native)
                    try:
                        admitted = self.workspace_callback(addition['additional_new_bytes'],
                                                           phase='fastskat_lookahead_new_peak')
                    except MemoryError:
                        continue
                    if admitted is False:
                        continue
                    if self.scope_owner is not None:
                        self.scope_owner._additional_new_high_water = max(
                            self.scope_owner._additional_new_high_water, addition['additional_new_bytes'])
                return width, estimate
        return 1, None

    def _populate(self, offset, width, estimate):
        matrices = initials = columns = pairs = records = None
        weights = weighted = initial = error = pair = None
        try:
            guard_cuda_workspace(estimate['required_new_bytes'], self.covariance.device,
                                 context='FastSKAT lookahead operator/X/solver/refinement NEW peak')
            matrices, initials, columns = [], [], []
            for index in range(offset, offset+width):
                weights = self.weights[:, index].to(dtype=self.covariance.dtype,
                                                    device=self.covariance.device)
                weighted = self.covariance * weights[:, None] * weights[None, :]
                initial = self._initial(index)
                # Preserve exact original X draw order if a later batched
                # eigensolve fails and original per-weight replay is needed.
                self.starts[index] = initial
                matrices.append(weighted)
                initials.append(initial)
                columns.append(weights)
            pairs, records = independent_lobpcg(matrices, initials, k=self.rank,
                niter=200, tol=1e-5, largest=True, max_active_workers=width,
                worker_pool=self.pool)
            for index, column, matrix, initial, pair in zip(
                    range(offset, offset+width), columns, matrices, initials, pairs):
                self.pending[index] = (column, matrix, initial, pair[0], pair[1])
                self.starts.pop(index, None)
            self.records.append(dict(offset=offset, weights=width,
                estimate=estimate, route='bounded_original_weight_pairs', solver=records))
        except (RuntimeError, MemoryError) as error:
            # Clear completed worker/driver frames before retaining an error
            # summary; the original per-weight caller determines error order.
            clear_exception_frames(error)
            self.pending.clear()
            self.records.append(dict(offset=offset, weights=width,
                estimate=estimate, route='original_serial_replay_after_batch_failure',
                failure_type=type(error).__name__, scientific_failure_not_suppressed=True))
        finally:
            if matrices is not None:
                matrices.clear()
            if initials is not None:
                initials.clear()
            if columns is not None:
                columns.clear()
            if pairs is not None:
                pairs.clear()
            matrices = initials = columns = pairs = records = None
            weights = weighted = initial = error = pair = column = matrix = None

    def take(self, index):
        result = None
        try:
            if index != self.next_index:
                raise ValueError('FastSKAT weights must be consumed in original order')
            if index not in self.pending and index not in self.starts:
                width, estimate = self._width(index)
                if width > 1:
                    self._populate(index, width, estimate)
                else:
                    self.records.append(dict(offset=index, weights=1,
                                             route='original_sequential_pressure_or_unsupported'))
            result = self.pending.pop(index, None)
            if result is None:
                result = self._original(index)
            self.next_index += 1
            return result
        finally:
            result = None

    def close(self):
        self.pending.clear()
        self.starts.clear()
        self.covariance = self.weights = self.generator = None
        if self.owned_pool is not None:
            self.owned_pool.close()
        self.owned_pool = self.pool = None
        self.scope_owner = None
        self.workspace_callback = None
