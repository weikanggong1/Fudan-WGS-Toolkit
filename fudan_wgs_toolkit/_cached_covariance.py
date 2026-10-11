"""Bounded cached TF32 covariance for an explicit diagonal single null model.

Reuse oriented/imputed FP32 genotype panels and covariate projections within
one call. Covariance products use the model's native TF32 implementation;
score/projection preparation retains 512 columns and both directed products
are averaged. No sample/variant selection or statistical tail is changed.

SPDX-License-Identifier: GPL-3.0-only
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import math
import time

import torch


_GIB = 2**30
_RESERVE = 256 * 2**20
_PREPARE_TILE = 512


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def cached_workspace_estimate(n, m, q, *, variant_tile_size, panel_variant_size,
                              full_resident=False):
    """Conservative NEW CUDA storage, excluding the already resident model.

    Full: original+weighted N*M. Panels: two original+weighted N*C panels.
    One covariance, six product block temporaries, cached C/D projections,
    scores, a bounded GEMV layout-copy allowance and bounded GPU input-check
    temporaries are also reserved. No complete covariance symmetrization copy
    or final covariance host transfer is used.

    ``n``, ``m`` and ``q`` are sample, variant and covariate counts.
    ``variant_tile_size`` is the covariance product width;
    ``panel_variant_size`` is the genotype cache width. ``full_resident``
    selects full original/weighted storage instead of two cached panels.
    Return bytes and the FP32 element count for each workspace component.
    """
    b, c = min(variant_tile_size, max(m, 1)), min(panel_variant_size, max(m, 1))
    preparation = min(_PREPARE_TILE, max(m, 1))
    genotype = 2 * n * m if full_resident else 4 * n * c
    terms = dict(genotype_cache_elements=genotype, covariance_elements=m * m,
                 product_temporary_elements=6 * b * b,
                 projection_elements=2 * q * m, score_elements=m,
                 vector_layout_allowance_elements=2 * n * preparation,
                 input_validation_allowance_elements=3 * n * preparation)
    return dict(new_storage_bytes=4 * sum(terms.values()), **terms)


def plan_cached_workspace(n, m, q, *, variant_tile_size=4096,
                          panel_variant_size=None, memory_limit_gib=40,
                          allocated_bytes=0, reserved_bytes=0,
                          free_bytes=None, reserve_bytes=_RESERVE):
    """Select full or two-panel CUDA storage without allocating a tensor.

    Existing allocator reservations count against the process cap, including
    unused fragments that cannot necessarily satisfy a large allocation.
    ``free_bytes`` adds the live device constraint, including other processes.
    The caller may release unused blocks and remeasure before planning;
    this pure planner never grants credit for unverified reusable fragments.
    The estimate
    reserves validation/GEMV layouts and product temporaries. Admission is a
    preflight check; simultaneous allocations or fragmentation can still OOM.
    Automatic panels may reduce the requested covariance product width to a
    multiple of 512. Preparation remains 512 columns. Explicit panels retain
    the requested width and must fit as specified. The caller must retain its
    configured CUDA allocator cap.
    """
    _positive_integer(n, "samples")
    for value, name in ((m, "variants"), (q, "covariates")):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    _positive_integer(variant_tile_size, "variant_tile_size")
    if variant_tile_size % _PREPARE_TILE:
        raise ValueError("variant_tile_size must be a multiple of 512")
    if panel_variant_size is not None:
        _positive_integer(panel_variant_size, "panel_variant_size")
        if panel_variant_size % variant_tile_size:
            raise ValueError("panel_variant_size must be a multiple of variant_tile_size")
    if (isinstance(memory_limit_gib, bool)
            or not isinstance(memory_limit_gib, (int, float))
            or not math.isfinite(memory_limit_gib) or not 0 < memory_limit_gib):
        raise ValueError("memory_limit_gib must be positive and finite")
    for value, name in ((allocated_bytes, "allocated_bytes"),
                        (reserved_bytes, "reserved_bytes"),
                        (reserve_bytes, "reserve_bytes")):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    limit = int(memory_limit_gib * _GIB)
    if free_bytes is None:
        free_bytes = limit
    if type(free_bytes) is not int or free_bytes < 0:
        raise ValueError("free_bytes must be a nonnegative integer")
    budget_basis = max(allocated_bytes, reserved_bytes)
    available = min(limit - budget_basis, free_bytes) - reserve_bytes
    if not m:
        return dict(cache_mode="empty", full_resident=False,
                    requested_variant_tile_size=variant_tile_size,
                    effective_variant_tile_size=variant_tile_size,
                    panel_variant_size=0, available_new_bytes=available,
                    allocator_budget_basis_bytes=budget_basis,
                    allocator_reservation_credit_bytes=0,
                    conservative_new_workspace_bytes=0, workspace_breakdown={})

    def automatic_plan(tile_size):
        full = cached_workspace_estimate(
            n, m, q, variant_tile_size=tile_size,
            panel_variant_size=m, full_resident=True)
        if full["new_storage_bytes"] <= available:
            return True, m, full
        # Two original+weighted panels cost 16*N bytes per cached column.
        # Keep all remaining allowances intact while changing only the
        # product width and the corresponding panel alignment.
        fixed = cached_workspace_estimate(
            n, m, q, variant_tile_size=tile_size,
            panel_variant_size=1)["new_storage_bytes"] - 16 * n
        maximum = (available - fixed) // (16 * n)
        panel_size = min(
            ((m + tile_size - 1) // tile_size) * tile_size,
            (maximum // tile_size) * tile_size)
        if panel_size < tile_size:
            return None
        estimate = cached_workspace_estimate(
            n, m, q, variant_tile_size=tile_size,
            panel_variant_size=panel_size)
        if estimate["new_storage_bytes"] > available:
            return None
        return False, panel_size, estimate

    effective_tile = variant_tile_size
    if panel_variant_size is None:
        selected = automatic_plan(effective_tile)
        if selected is None:
            # Admission is monotone in the minimum panel/product width.
            # Binary search avoids scanning an unbounded user-supplied width.
            low, high = 1, variant_tile_size // _PREPARE_TILE - 1
            while low <= high:
                units = (low + high) // 2
                candidate = automatic_plan(units * _PREPARE_TILE)
                if candidate is None:
                    high = units - 1
                else:
                    effective_tile, selected = units * _PREPARE_TILE, candidate
                    low = units + 1
            if selected is None:
                raise MemoryError("configured budget cannot admit two original+weighted covariance panels at the minimum 512-column product width")
        full_resident, panel_size, estimate = selected
    else:
        full_resident, panel_size = False, panel_variant_size
        estimate = cached_workspace_estimate(
            n, m, q, variant_tile_size=effective_tile,
            panel_variant_size=panel_size)
    if estimate["new_storage_bytes"] > available:
        raise MemoryError(f"cached covariance requires {estimate['new_storage_bytes']} new bytes; {available} available within process/live GPU budget")
    return dict(
        cache_mode="full_original_and_weighted" if full_resident else "two_original_and_weighted_panels",
        full_resident=full_resident, panel_variant_size=panel_size,
        requested_variant_tile_size=variant_tile_size,
        effective_variant_tile_size=effective_tile,
        available_new_bytes=available,
        allocator_budget_basis_bytes=budget_basis,
        allocator_reservation_credit_bytes=0,
        conservative_new_workspace_bytes=estimate["new_storage_bytes"],
        workspace_breakdown=estimate)


def _cuda_memory_snapshot(device):
    """Measure the selected CUDA allocator and live device without allocation."""
    with torch.cuda.device(device):
        free, _ = torch.cuda.mem_get_info()
        return dict(allocated_bytes=int(torch.cuda.memory_allocated(device)),
                    reserved_bytes=int(torch.cuda.memory_reserved(device)),
                    free_bytes=int(free))


def _release_unused_cuda_blocks(device):
    """Release only unoccupied allocator blocks, retaining every live tensor.

    Re-measure rather than assume that all unused reservations were released:
    blocks containing live allocations can remain fragmented after cleanup.
    CUDA cache cleanup is restricted to the device used by this call.
    """
    before = _cuda_memory_snapshot(device)
    started = time.perf_counter()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    after = _cuda_memory_snapshot(device)
    return dict(before=before, after=after, host_wall_seconds=elapsed,
                released_bytes=max(0, before["reserved_bytes"] - after["reserved_bytes"]))


def _allocate_with_pressure_reclaim(factory, device, record_cleanup=None):
    """Retry one non-mutating allocation only after an actual CUDA OOM."""
    try:
        return factory()
    except torch.cuda.OutOfMemoryError:
        cleanup = _release_unused_cuda_blocks(device)
        if record_cleanup is not None:
            record_cleanup(cleanup)
        return factory()


class _Timers:
    def __init__(self, device, enabled):
        self.device, self.enabled = device, enabled
        self.stages, self.events = {}, []

    @contextmanager
    def measure(self, name):
        started = time.perf_counter()
        if self.enabled:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(torch.cuda.current_stream(self.device))
        try:
            yield
        finally:
            if self.enabled:
                end.record(torch.cuda.current_stream(self.device))
                self.events.append((name, begin, end))
            stage = self.stages.setdefault(name, dict(calls=0, host_wall_seconds=0.0))
            stage["calls"] += 1
            stage["host_wall_seconds"] += time.perf_counter() - started

    def finish(self):
        if self.enabled:
            torch.cuda.synchronize(self.device)
            for name, begin, end in self.events:
                stage = self.stages[name]
                stage["cuda_stream_seconds"] = stage.get("cuda_stream_seconds", 0.0) + begin.elapsed_time(end) / 1000
        return self.stages


def score_covariance_cached(model, genotype, *, variant_tile_size=4096,
                            panel_variant_size=None, memory_limit_gib=40,
                            symmetry="average", matmul_mode="tf32", profile=False,
                            compute_score=True):
    """Return ``(score, covariance, report)`` without changing a fitted model.

    ``genotype`` is the finite, already oriented/imputed host N-by-M FP32
    dosage. ``variant_tile_size`` requests the covariance GEMM output size;
    score/covariate projections keep the original 512-column preparation.
    ``panel_variant_size`` controls cached columns, with automatic admission
    when None, including a smaller 512-aligned product width if needed. The
    report records requested/effective widths; ``variant_tile_size`` in the
    report is the effective width. Admission counts current allocator reservations
    against the process ceiling, including existing model tensors. Unused cache
    is reclaimed only after failed admission or an actual allocation OOM;
    healthy calls retain allocator reuse. Cleanup never changes a live tensor.
    ``symmetry='average'`` retains both TF32 directions. Other symmetry
    rules are rejected because they change the accepted covariance contract.
    ``profile=True`` adds CUDA event measurements and synchronizes once at the
    end. Host stage times and CUDA stream intervals overlap; neither denotes
    pure-kernel time. H2D counts include genotype panels; D2H counts include
    one Boolean finite flag per newly encountered panel. Covariance has no
    host transfer. Each panel is checked before its score/projection products.
    ``compute_score=False`` returns None for score and omits its allocation
    and products when a caller already owns the shared phenotype scores. It
    keeps preparation layouts and every covariance product unchanged.
    """
    started = time.perf_counter()
    _positive_integer(variant_tile_size, "variant_tile_size")
    if variant_tile_size % _PREPARE_TILE:
        raise ValueError("variant_tile_size must be a multiple of 512")
    if panel_variant_size is not None:
        _positive_integer(panel_variant_size, "panel_variant_size")
        if panel_variant_size % variant_tile_size:
            raise ValueError("panel_variant_size must be a multiple of variant_tile_size")
    if isinstance(memory_limit_gib, bool) or not isinstance(memory_limit_gib, (int, float)) or not math.isfinite(memory_limit_gib) or not 0 < memory_limit_gib:
        raise ValueError("memory_limit_gib must be positive and finite")
    if symmetry != "average":
        raise ValueError("cached covariance only admits symmetry=average")
    if type(profile) is not bool:
        raise ValueError("profile must be Boolean")
    if type(compute_score) is not bool:
        raise ValueError("compute_score must be Boolean")
    if matmul_mode != "tf32" or getattr(model, "matmul_mode", "tf32") != "tf32":
        raise ValueError("cached covariance requires a native TF32 model")
    # Binary fitted state has no kinship eigenspectrum. Admit only its explicit
    # diagonal non-SPA protocol; retain the established Gaussian gate and state.
    from .binary_null import BinaryNullModel
    binary_state = isinstance(model, BinaryNullModel)
    if binary_state:
        inverse_variance, precision_x, residual, fixed_cov = model._diagonal_tf32_covariance_state(matmul_mode)
        if isinstance(genotype, torch.Tensor) and genotype.requires_grad:
            raise ValueError("cached binary genotype must be host data without gradients")
    elif getattr(model, "family", None) != "gaussian" or getattr(model, "n_pheno", 1) != 1 or getattr(model, "use_spa", False) or model.spectrum.blocks:
        raise NotImplementedError("cached covariance requires single Gaussian diagonal precision without SPA")
    device = torch.device(model.device)
    if device.type != "cuda":
        raise ValueError("cached native TF32 covariance requires a CUDA model")
    if isinstance(genotype, torch.Tensor) and genotype.is_cuda:
        raise ValueError("genotype must reside on the host; implicit full D2H is not supported")
    layout_started = time.perf_counter()
    source = torch.as_tensor(genotype, dtype=torch.float32, device="cpu")
    if source.ndim != 2 or source.shape[0] != model.n:
        raise ValueError("genotype must be a samples-by-variants matrix aligned to the model")
    n, m = map(int, source.shape)
    source_column_major = source.stride(0) == 1 and source.stride(1) == n
    validation_host_layout_seconds = time.perf_counter() - layout_started
    covariance_scale = None
    if not binary_state:
        inverse_variance, precision_x, fixed_cov, covariance_scale = model._covariance_product_state(matmul_mode)
        residual = model.scaled_residuals
    for name, value in (("precision_x", precision_x), ("inverse_variance", inverse_variance),
                        ("scaled_residuals", residual), ("fixed_effect_covariance", fixed_cov)):
        if value.device != device or value.dtype != torch.float32:
            raise ValueError(f"model.{name} must be CUDA FP32 on the model device")
    q = int(precision_x.shape[1])
    # Bind the exact model implementation, never copy or replace the TF32 kernel.
    module = importlib.import_module(type(model).__module__)
    mm = getattr(module, "matmul")
    limit = int(memory_limit_gib * _GIB)
    peak_before = torch.cuda.max_memory_allocated(device)
    initial_cleanup = None
    snapshot = _cuda_memory_snapshot(device)
    def make_plan(current):
        return plan_cached_workspace(
            n, m, q, variant_tile_size=variant_tile_size,
            panel_variant_size=panel_variant_size, memory_limit_gib=memory_limit_gib,
            allocated_bytes=current['allocated_bytes'], reserved_bytes=current['reserved_bytes'],
            free_bytes=current['free_bytes'])
    try:
        plan = make_plan(snapshot)
    except MemoryError:
        if snapshot['reserved_bytes'] <= snapshot['allocated_bytes']:
            raise
        initial_cleanup = _release_unused_cuda_blocks(device)
        snapshot = initial_cleanup['after']
        plan = make_plan(snapshot)
    allocated_before = snapshot["allocated_bytes"]
    reserved_before = snapshot["reserved_bytes"]
    free_before = snapshot["free_bytes"]
    effective_tile = plan["effective_variant_tile_size"]
    timers = _Timers(device, profile)
    report = dict(backend="cached_native_tf32", cache_version=1,
                  compute_score=compute_score, score_retained=compute_score,
                  samples=n, variants=m, covariates=q, symmetry=symmetry,
                  variant_tile_size=effective_tile,
                  requested_variant_tile_size=variant_tile_size,
                  effective_variant_tile_size=effective_tile,
                  preparation_tile_size=_PREPARE_TILE,
                  memory_limit_bytes=limit, admission_reserve_bytes=_RESERVE,
                  allocated_before_bytes=allocated_before, reserved_before_bytes=reserved_before,
                  free_before_bytes=free_before, peak_before_bytes=peak_before,
                  admission_budget_basis="remaining_cuda_allocator_reservations",
                  pre_cleanup_allocated_bytes=(initial_cleanup["before"]["allocated_bytes"] if initial_cleanup else allocated_before),
                  pre_cleanup_reserved_bytes=(initial_cleanup["before"]["reserved_bytes"] if initial_cleanup else reserved_before),
                  pre_cleanup_free_bytes=(initial_cleanup["before"]["free_bytes"] if initial_cleanup else free_before),
                  allocator_cleanup_calls=int(initial_cleanup is not None),
                  allocator_cleanup_released_bytes=(initial_cleanup["released_bytes"] if initial_cleanup else 0),
                  allocator_cleanup_host_wall_seconds=(initial_cleanup["host_wall_seconds"] if initial_cleanup else 0.0),
                  h2d_bytes=0, h2d_calls=0, d2h_bytes=0, d2h_calls=0,
                  covariance_d2h_bytes=0, covariance_d2h_calls=0,
                  validation_host_layout_seconds=validation_host_layout_seconds,
                  input_validation_columns=_PREPARE_TILE,
                  validation_sync_count=0, validation_d2h_bytes=0,
                  score_product_calls=0, cross_product_calls=0,
                  projection_left_product_calls=0, covariance_product_calls=0,
                  projection_product_calls=0, weighted_panel_calls=0,
                  score_layout_copy_calls=0, score_layout_copy_bytes=0,
                  singleton_product_blocks=0, singleton_layout_copy_calls=0,
                  singleton_layout_copy_bytes=0,
                  singleton_covariance_tile_size=_PREPARE_TILE,
                  matmul_module=module.__name__, max_observed_allocated_bytes=allocated_before,
                  timing_contract="host wall and CUDA stream elapsed overlap; events are not pure-kernel timings")
    if not m:
        score = torch.empty((0,), dtype=torch.float32, device=device) if compute_score else None
        covariance = torch.empty((0, 0), dtype=torch.float32, device=device)
        report.update(cache_mode="empty", panel_variant_size=0,
                      conservative_new_workspace_bytes=0, stages=timers.finish(),
                      wall_seconds=time.perf_counter() - started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        return score, covariance, report

    full_resident = plan["full_resident"]
    panel_size = plan["panel_variant_size"]
    report.update({key: value for key, value in plan.items() if key != "full_resident"})

    def observe_memory():
        observed = torch.cuda.memory_allocated(device)
        report["max_observed_allocated_bytes"] = max(report["max_observed_allocated_bytes"], observed)
        if observed > limit:
            raise MemoryError("cached covariance exceeded the configured allocated-process memory limit")

    def record_pressure_cleanup(cleanup):
        report["allocator_cleanup_calls"] += 1
        report["allocator_cleanup_released_bytes"] += cleanup["released_bytes"]
        report["allocator_cleanup_host_wall_seconds"] += cleanup["host_wall_seconds"]

    def allocate(factory):
        return _allocate_with_pressure_reclaim(factory, device, record_pressure_cleanup)

    score = (allocate(lambda: torch.empty((m,), dtype=torch.float32, device=device))
             if compute_score else None)
    cross = allocate(lambda: torch.empty((q, m), dtype=torch.float32, device=device))
    projected_left = allocate(lambda: torch.empty((m, q), dtype=torch.float32, device=device))
    covariance = allocate(lambda: torch.empty((m, m), dtype=torch.float32, device=device))
    prepared = set()
    validated_panels = set()

    def load_panel(start, stop):
        with timers.measure("genotype_h2d"):
            panel = allocate(lambda: torch.as_tensor(source[:, start:stop], dtype=torch.float32, device=device))
        report["h2d_bytes"] += 4 * n * (stop - start)
        report["h2d_calls"] += 1
        observe_memory()
        identity = (start, stop)
        if identity not in validated_panels:
            # Check the uploaded FP32 input without repeatedly reading the
            # complete host matrix. Bounded checks avoid an N*M GPU Boolean
            # allocation; only the final flag synchronizes to the host.
            with timers.measure("input_validation_cuda"):
                finite = torch.ones((), dtype=torch.bool, device=device)
                for column in range(0, stop - start, _PREPARE_TILE):
                    finite.logical_and_(torch.isfinite(panel[:, column:column + _PREPARE_TILE]).all())
            with timers.measure("input_validation_sync"):
                panel_is_finite = bool(finite)
            report["validation_sync_count"] += 1
            report["validation_d2h_bytes"] += 1
            report["d2h_calls"] += 1
            report["d2h_bytes"] += 1
            del finite
            if not panel_is_finite:
                raise ValueError("genotype must be finite")
            validated_panels.add(identity)
            observe_memory()
        # Once per variant: retain the old 512-column mathematical geometry.
        for column in range(start, stop, _PREPARE_TILE):
            if column in prepared:
                continue
            end = min(column + _PREPARE_TILE, stop)
            raw = panel[:, column - start:end - start]
            # Host C-order input formerly became a compact C-order 512 tile.
            # Restore that layout for the CUDA GEMV score if the panel is wider.
            score_tile = raw
            if not source_column_major and not raw.is_contiguous():
                with timers.measure("score_layout_copy"):
                    score_tile = raw.contiguous()
                report["score_layout_copy_calls"] += 1
                report["score_layout_copy_bytes"] += 4 * n * (end - column)
            if compute_score:
                with timers.measure("score_prepare"):
                    score[column:end] = mm(score_tile.T, residual, mode="tf32")
                report["score_product_calls"] += 1
            with timers.measure("projection_prepare"):
                # A one-column tail uses CUDA FP32 GEMV. Preserve the same
                # compact host-C layout as legacy preparation for that route,
                # reusing the layout copy already required by the score.
                c = mm(precision_x.T, score_tile, mode="tf32")
                cross[:, column:end] = c
                projected_left[column:end] = mm(c.T, fixed_cov, mode="tf32")
            report["cross_product_calls"] += 1
            report["projection_left_product_calls"] += 1
            prepared.add(column)
            del c, raw, score_tile
        with timers.measure("weighted_panel_prepare"):
            weighted = allocate(lambda: inverse_variance[:, None] * panel)
        report["weighted_panel_calls"] += 1
        observe_memory()
        return panel, weighted

    def product_block(left, weighted_right, row, row_stop, column, column_stop):
        singleton = row_stop - row == 1 or column_stop - column == 1
        projection_right = cross[:, column:column_stop]
        if singleton:
            report["singleton_product_blocks"] += 1
            # GEMV/dot use cuBLAS FP32 rather than the strided TF32 MMA
            # kernel. Their arithmetic depends on shape and storage strides.
            # Preserve legacy 512-column shapes and compact C layouts, while
            # retaining uploaded original/weighted panels and cached C/D.
            with timers.measure("singleton_layout_copy"):
                if not source_column_major:
                    if not left.is_contiguous():
                        left = left.contiguous()
                        report["singleton_layout_copy_calls"] += 1
                        report["singleton_layout_copy_bytes"] += left.numel() * 4
                    if not weighted_right.is_contiguous():
                        weighted_right = weighted_right.contiguous()
                        report["singleton_layout_copy_calls"] += 1
                        report["singleton_layout_copy_bytes"] += weighted_right.numel() * 4
                if not projection_right.is_contiguous():
                    projection_right = projection_right.contiguous()
                    report["singleton_layout_copy_calls"] += 1
                    report["singleton_layout_copy_bytes"] += projection_right.numel() * 4
            observe_memory()
        with timers.measure("covariance_gemm"):
            gram = mm(left.T, weighted_right, mode="tf32")
        report["covariance_product_calls"] += 1
        with timers.measure("projection_apply"):
            projection = mm(projected_left[row:row_stop], projection_right, mode="tf32")
        report["projection_product_calls"] += 1
        with timers.measure("covariance_subtract"):
            block = gram - projection
        del gram, projection
        observe_memory()
        return block

    def product_ranges(start, stop):
        boundaries = list(range(start, stop, effective_tile))
        # An old 512 tail of width one must stay a GEMV/dot, even if a wider
        # cached output tile would otherwise absorb it into a TF32 MMA.
        if m % _PREPARE_TILE == 1 and start < m - 1 < stop:
            boundaries.append(m - 1)
        boundaries = sorted(set(boundaries + [stop]))
        return zip(boundaries[:-1], boundaries[1:])

    def panel_pair(left, weighted_left, left_start, left_stop,
                   right, weighted_right, right_start, right_stop):
        def emit_block(row, row_stop, column, column_stop):
            l = left[:, row - left_start:row_stop - left_start]
            r = right[:, column - right_start:column_stop - right_start]
            wr = weighted_right[:, column - right_start:column_stop - right_start]
            block = product_block(l, wr, row, row_stop, column, column_stop)
            if row == column:
                with timers.measure("symmetry_average"):
                    block = (block + block.T) / 2
            else:
                wl = weighted_left[:, row - left_start:row_stop - left_start]
                reverse = product_block(r, wl, column, column_stop, row, row_stop)
                with timers.measure("symmetry_average"):
                    block = (block + reverse.T) / 2
                del reverse, wl
            with timers.measure("covariance_output_device_copy"):
                covariance[row:row_stop, column:column_stop].copy_(block)
                if row != column:
                    covariance[column:column_stop, row:row_stop].copy_(block.T)

        same = left_start == right_start
        for outer_row, outer_row_stop in product_ranges(left_start, left_stop):
            for outer_column, outer_column_stop in product_ranges(right_start, right_stop):
                if same and outer_column < outer_row:
                    continue
                singleton = outer_row_stop - outer_row == 1 or outer_column_stop - outer_column == 1
                row_step = _PREPARE_TILE if singleton else effective_tile
                column_step = _PREPARE_TILE if singleton else effective_tile
                for row in range(outer_row, outer_row_stop, row_step):
                    row_stop = min(row + row_step, outer_row_stop)
                    for column in range(outer_column, outer_column_stop, column_step):
                        column_stop = min(column + column_step, outer_column_stop)
                        emit_block(row, row_stop, column, column_stop)

    if full_resident:
        original, weighted = load_panel(0, m)
        panel_pair(original, weighted, 0, m, original, weighted, 0, m)
        del original, weighted
    else:
        for left_start in range(0, m, panel_size):
            left_stop = min(left_start + panel_size, m)
            original_left, weighted_left = load_panel(left_start, left_stop)
            panel_pair(original_left, weighted_left, left_start, left_stop,
                       original_left, weighted_left, left_start, left_stop)
            for right_start in range(left_stop, m, panel_size):
                right_stop = min(right_start + panel_size, m)
                original_right, weighted_right = load_panel(right_start, right_stop)
                panel_pair(original_left, weighted_left, left_start, left_stop,
                           original_right, weighted_right, right_start, right_stop)
                del original_right, weighted_right
            del original_left, weighted_left
    del cross, projected_left
    if covariance_scale is not None:
        with timers.measure("covariance_unit_restore"):
            covariance.mul_(covariance_scale)
    observe_memory()
    report["stages"] = timers.finish()
    report.update(wall_seconds=time.perf_counter() - started,
                  allocated_after_bytes=torch.cuda.memory_allocated(device),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                  peak_counter_scope="process peak includes earlier allocations; caller may reset peak stats before an isolated benchmark",
                  prepared_variant_tiles=len(prepared),
                  validated_panel_count=len(validated_panels),
                  covariance_host_transfer_removed=True)
    return score, covariance, report
