"""Shared genotype scores use IEEE FP32 while covariance retains native TF32."""
from __future__ import annotations

from copy import deepcopy
import os

import torch

_METRICS = dict(score_batches=0, fp32_simt_products=0, fp32_vector_products=0,
                simt_admission_fallbacks=0, workspace_queries=0,
                maximum_estimated_new_workspace_bytes=0)


def execution_metadata(*, reset=False):
    """Report actual score routes, separately from covariance/Burden products."""
    result = deepcopy(_METRICS)
    if reset:
        for key in _METRICS:
            _METRICS[key] = 0
    return result


def _simt_admission(genotype, traits):
    """Check NEW stacked residuals/output plus conservative BLAS copy allowance."""
    if not genotype.is_cuda:
        return True
    from .. import tf32
    from .._burden import estimate_workspace
    n, m = genotype.shape
    # Genotype and fitted vectors already exist. A hypothetical BLAS layout
    # copy is NEW workspace, distinct from the existing shared G storage.
    estimate = estimate_workspace((m, n), (n, traits))
    required = 4*n*traits + estimate['required_bytes']
    _METRICS['maximum_estimated_new_workspace_bytes'] = max(
        _METRICS['maximum_estimated_new_workspace_bytes'], required)
    _METRICS['workspace_queries'] += 1
    allocated = torch.cuda.memory_allocated(genotype.device)
    reserved = torch.cuda.memory_reserved(genotype.device)
    with torch.cuda.device(genotype.device):
        free, _ = torch.cuda.mem_get_info()
    available = tf32._product_workspace_availability(allocated=allocated,
        reserved=reserved, free=free, limit=tf32._memory_limit_bytes,
        reserve=tf32._memory_reserve_bytes)
    return required <= available['available_bytes']


def fp32_score_columns(genotype, residuals, *, out=None):
    """Compute ``G.T @ U`` using original FP32 fitted residuals and one shared G.

    A singleton uses the mature FP32 GEMV. Wider CUDA tiles use an IEEE FP32 SIMT
    matrix product: TF32 input rounding is disabled for Score only, under the
    existing native IEEE product lock and restored even after an exception.
    Covariance/projection and FP64 SPA retain their original scientific routes.
    NEW stacked ``N x traits`` storage is admitted before allocating it; when
    this extra workspace cannot fit, the mature per-vector route needs no
    stack. Every CUDA vector retains its existing native workspace guard.
    Caller admission covers the output and optional copy into ``out``.
    """
    if (not isinstance(genotype, torch.Tensor) or genotype.ndim != 2
            or genotype.layout != torch.strided or genotype.dtype != torch.float32):
        raise ValueError('shared score genotype must be a dense FP32 matrix')
    vectors = tuple(residuals)
    n, m = genotype.shape
    for residual in vectors:
        if (not isinstance(residual, torch.Tensor) or residual.ndim != 1
                or residual.shape != (n,) or residual.layout != torch.strided
                or residual.dtype != torch.float32 or residual.device != genotype.device):
            raise ValueError('shared scores require aligned original fitted FP32 residual vectors')
    if out is not None:
        if (not isinstance(out, torch.Tensor) or out.shape != (m, len(vectors))
                or out.layout != torch.strided or out.dtype != torch.float32
                or out.device != genotype.device):
            raise ValueError('shared score output must match the FP32 variant-by-trait matrix')
        if out.numel() and any(value.numel() and value.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
                               for value in (genotype, *vectors)):
            raise ValueError('shared score output cannot alias fitted residuals or genotype storage')
    _METRICS['score_batches'] += 1
    wide = len(vectors) > 1 and n > 0 and m > 0
    simt = wide and genotype.is_cuda
    if simt and os.getenv('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') == '1':
        raise ValueError('unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE: shared Score requires IEEE FP32')
    if simt and _simt_admission(genotype, len(vectors)):
        from .._burden import _LOCK, _ieee_flags
        packed = torch.stack(vectors, dim=1)
        try:
            # Use the existing lock/flags without relabelling these Score
            # products as Burden products in resource/execution telemetry.
            with _LOCK, _ieee_flags():
                value = torch.mm(genotype.T, packed)
            _METRICS['fp32_simt_products'] += 1
        finally:
            del packed
        if out is None:
            return value
        out.copy_(value)
        del value
        return out
    if simt:
        _METRICS['simt_admission_fallbacks'] += 1
    result = out if out is not None else torch.empty(
        (m, len(vectors)), dtype=torch.float32, device=genotype.device)
    if genotype.is_cuda:
        from ..tf32 import matmul
        for index, residual in enumerate(vectors):
            result[:, index].copy_(matmul(genotype.T, residual, mode='tf32'))
            _METRICS['fp32_vector_products'] += 1
    else:
        for index, residual in enumerate(vectors):
            result[:, index].copy_(torch.mv(genotype.T, residual))
            _METRICS['fp32_vector_products'] += 1
    return result
