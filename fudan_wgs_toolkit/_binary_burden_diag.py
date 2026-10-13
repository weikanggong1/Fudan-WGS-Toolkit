"""Bounded diagonal binary burden products; no samples-by-traits-by-weights slab.

SPDX-License-Identifier: GPL-3.0-only
CUDA contractions split the sample axis for scheduling, accumulate in FP32,
and retain the supplied independent fitted precision/covariates/residuals.
This is a change of execution order, without fitting a replacement null model.
"""
from __future__ import annotations

import torch
import weakref

_COO_PATTERN_BOUNDS = weakref.WeakKeyDictionary()
_COO_OWNER_BOUNDS = {}

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _score_diagonal(Z, P, SCORE, BASE, N, B, T, ZS0, ZS1,
                        CHUNK: tl.constexpr, RN: tl.constexpr, RB: tl.constexpr):
        # The long sample/trait axis belongs in CUDA grid.x (up to 2^31-1),
        # avoiding the 65535-block y/z bound for very small sample chunks.
        trait = (tl.program_id(0) % T).to(tl.int64)
        part = (tl.program_id(0) // T).to(tl.int64)
        cols = (tl.program_id(1)*RB+tl.arange(0, RB)).to(tl.int64)
        rows = tl.arange(0, RN).to(tl.int64)
        precision = tl.load(P+trait).to(tl.pointer_type(tl.float32))
        precision_stride = tl.load(P+T+trait)
        precision_kind = tl.load(P+2*T+trait)
        residual = tl.load(P+3*T+trait).to(tl.pointer_type(tl.float32))
        residual_stride = tl.load(P+4*T+trait)
        score = tl.full((RB,), 0, tl.float32)
        base = tl.full((RB,), 0, tl.float32)
        for start in range(tl.cdiv(CHUNK, RN)):
            sample = part*CHUNK+start*RN+rows
            valid = (sample < N) & (sample < (part+1)*CHUNK)
            z = tl.load(Z+sample[:, None]*ZS0+cols[None, :]*ZS1,
                        valid[:, None] & (cols[None, :] < B), other=0)
            r = tl.load(residual+sample*residual_stride, valid, other=0)
            # Diagonal base is sum_i Z_ib * (precision_it * Z_ib).
            # These are IEEE FP32 products, with no TF32 component splitting.
            if precision_kind != 2:
                p = tl.load(precision+sample*precision_stride, valid, other=0)
                if precision_kind == 1:
                    p = p*(1-p)
                weighted = p[:, None]*z
                base += tl.sum(z*weighted, axis=0)
            score += tl.sum(z*r[:, None], axis=0)
        output = (part*T+trait)*B+cols
        tl.store(SCORE+output, score, cols < B)
        tl.store(BASE+output, base, cols < B)

    @triton.jit
    def _sparse_base(Z, P, BASE, B, T, ZS0, ZS1,
                     CHUNK: tl.constexpr, RE: tl.constexpr, RB: tl.constexpr):
        trait = (tl.program_id(0) % T).to(tl.int64)
        part = (tl.program_id(0) // T).to(tl.int64)
        cols = (tl.program_id(1)*RB+tl.arange(0, RB)).to(tl.int64)
        edges = tl.arange(0, RE).to(tl.int64)
        kind = tl.load(P+2*T+trait)
        result = tl.full((RB,), 0, tl.float32)
        if kind == 2:
            values = tl.load(P+trait).to(tl.pointer_type(tl.float32))
            value_stride = tl.load(P+T+trait)
            indices = tl.load(P+8*T+trait).to(tl.pointer_type(tl.int64))
            index_row_stride = tl.load(P+9*T+trait)
            index_edge_stride = tl.load(P+10*T+trait)
            nnz = tl.load(P+11*T+trait)
            for start in range(tl.cdiv(CHUNK, RE)):
                edge = part*CHUNK+start*RE+edges
                valid = (edge < nnz) & (edge < (part+1)*CHUNK)
                row = tl.load(indices+edge*index_edge_stride, valid, other=0)
                column = tl.load(indices+index_row_stride+edge*index_edge_stride,
                                 valid, other=0)
                value = tl.load(values+edge*value_stride, valid, other=0)
                left = tl.load(Z+row[:, None]*ZS0+cols[None, :]*ZS1,
                               valid[:, None] & (cols[None, :] < B), other=0)
                right = tl.load(Z+column[:, None]*ZS0+cols[None, :]*ZS1,
                                valid[:, None] & (cols[None, :] < B), other=0)
                # Every original COO entry contributes, including all signed
                # off-diagonal values. No diagonal approximation or N*N slab.
                result += tl.sum(left*(value[:, None]*right), axis=0)
        tl.store(BASE+(part*T+trait)*B+cols, result, cols < B)

    @triton.jit
    def _cross(Z, P, CROSS, N, B, Q, T, ZS0, ZS1,
               CHUNK: tl.constexpr, RQ: tl.constexpr,
               RB: tl.constexpr, RK: tl.constexpr):
        trait = tl.program_id(0) % T
        part = tl.program_id(0) // T
        trait = trait.to(tl.int64)
        part = part.to(tl.int64)
        rows = (tl.program_id(1)*RQ+tl.arange(0, RQ)).to(tl.int64)
        cols = (tl.program_id(2)*RB+tl.arange(0, RB)).to(tl.int64)
        inner = tl.arange(0, RK).to(tl.int64)
        sx = tl.load(P+5*T+trait).to(tl.pointer_type(tl.float32))
        sx0 = tl.load(P+6*T+trait)
        sx1 = tl.load(P+7*T+trait)
        acc = tl.full((RQ, RB), 0, tl.float32)
        for start in range(tl.cdiv(CHUNK, RK)):
            sample = part*CHUNK+start*RK+inner
            valid = (sample < N) & (sample < (part+1)*CHUNK)
            left = tl.load(sx+sample[None, :]*sx0+rows[:, None]*sx1,
                           (rows[:, None] < Q) & valid[None, :], other=0)
            right = tl.load(Z+sample[:, None]*ZS0+cols[None, :]*ZS1,
                            valid[:, None] & (cols[None, :] < B), other=0)
            left = tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", "=r,r",
                [left], dtype=tl.float32, is_pure=True, pack=1)
            right = tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", "=r,r",
                [right], dtype=tl.float32, is_pure=True, pack=1)
            acc = tl.dot(left, right, acc, input_precision='tf32')
        output = ((part*T+trait)*Q+rows[:, None])*B+cols[None, :]
        tl.store(CROSS+output, acc, (rows[:, None] < Q) & (cols[None, :] < B))

    @triton.jit
    def _project_ieee(CROSS, P, OUTPUT, B, Q, T, HS0, HS1, HS2,
                      RB: tl.constexpr, RQ: tl.constexpr, RK: tl.constexpr):
        # Flatten the entire grid to avoid CUDA's 65535 y/z dimension limit.
        groups_b, groups_q = tl.cdiv(B, RB), tl.cdiv(Q, RQ)
        task = tl.program_id(0).to(tl.int64)
        trait = task // (groups_b*groups_q)
        group_b = (task // groups_q) % groups_b
        group_q = task % groups_q
        burdens = (group_b*RB+tl.arange(0, RB)).to(tl.int64)
        columns = (group_q*RQ+tl.arange(0, RQ)).to(tl.int64)
        inner = tl.arange(0, RK).to(tl.int64)
        covariance = tl.load(P+trait).to(tl.pointer_type(tl.float32))
        covariance_s0 = tl.load(P+T+trait)
        covariance_s1 = tl.load(P+2*T+trait)
        acc = tl.full((RB, RQ), 0, tl.float32)
        for start in range(tl.cdiv(Q, RK)):
            rows = start*RK+inner
            left = tl.load(CROSS+trait*HS0+rows[None, :]*HS1+burdens[:, None]*HS2,
                           (burdens[:, None] < B) & (rows[None, :] < Q), other=0)
            right = tl.load(covariance+rows[:, None]*covariance_s0
                            + columns[None, :]*covariance_s1,
                            (rows[:, None] < Q) & (columns[None, :] < Q), other=0)
            # Ordinary FP32 products preserve the supplied small covariance
            # entries. The q*q matrix is never rounded to TF32 and no
            # traits*burdens*q*q global temporary is constructed.
            acc += tl.sum(left[:, :, None]*right[None, :, :], axis=1)
        output = (trait*B+burdens[:, None])*Q+columns[None, :]
        tl.store(OUTPUT+output, acc, (burdens[:, None] < B) & (columns[None, :] < Q))


def _descriptor_rows(models):
    """Retain fitted tensor pointers and their actual strides, without stacking."""
    rows = [[] for _ in range(12)]
    for model in models:
        precision = model.precision if model.precision is not None else model.fitted_probability
        sx = model.precision_x if model.precision_x is not None else model.xw.T
        residual = model.scaled_residuals
        if precision.layout == torch.sparse_coo:
            indices, sparse_values = precision.indices(), precision.values()
            values = (sparse_values.data_ptr(), sparse_values.stride(0), 2,
                residual.data_ptr(), residual.stride(0), sx.data_ptr(), sx.stride(0), sx.stride(1),
                indices.data_ptr(), indices.stride(0), indices.stride(1), precision._nnz())
        else:
            values = (precision.data_ptr(), precision.stride(0), int(model.precision is None),
                residual.data_ptr(), residual.stride(0), sx.data_ptr(), sx.stride(0), sx.stride(1),
                0, 0, 0, 0)
        for row, value in zip(rows, values):
            row.append(value)
    return rows


def _validate_coo_geometry(precision, n):
    """Validate source COO metadata/bounds once per immutable shared pattern.

    The owner cache is weak and keyed by mutation geometry. Production fitted
    bank contexts separately validate every alias of their shared pattern.
    No precision values are approximated, reordered, copied or cached here.
    """
    if (precision.layout != torch.sparse_coo or precision.shape != (n, n)
            or not precision.is_coalesced() or precision.dtype != torch.float32
            or precision.requires_grad):
        raise ValueError('native sparse burden precision must be coalesced FP32 COO')
    indices, values = precision.indices(), precision.values()
    nnz = precision._nnz()
    if (indices.shape != (2, nnz) or indices.dtype != torch.int64
            or indices.device != precision.device or values.shape != (nnz,)
            or values.device != precision.device or values.dtype != torch.float32
            or values.requires_grad):
        raise ValueError('native sparse burden COO indices/values have invalid source geometry')
    pattern = getattr(precision, '_immutable_shared_pattern', None)
    if pattern is not None:
        canonical = pattern.indices
        if (indices.data_ptr() != canonical.data_ptr()
                or indices.shape != canonical.shape or indices.stride() != canonical.stride()):
            raise ValueError('native sparse burden COO differs from its fitted shared pattern')
        stamp = (n, canonical.data_ptr(), canonical._version, tuple(canonical.shape),
                 tuple(canonical.stride()), canonical.device)
        if _COO_PATTERN_BOUNDS.get(pattern) == stamp:
            return
    else:
        stamp = (n, precision._version, indices.data_ptr(), indices._version,
                 tuple(indices.shape), tuple(indices.stride()), precision.device)
        cached = _COO_OWNER_BOUNDS.get(id(precision))
        if cached is not None and cached[0]() is precision and cached[1] == stamp:
            return
    if nnz and (bool((indices < 0).any()) or bool((indices >= n).any())):
        raise ValueError('native sparse burden COO indices are outside the sample axis')
    if pattern is not None:
        _COO_PATTERN_BOUNDS[pattern] = stamp
    else:
        token = id(precision)
        _COO_OWNER_BOUNDS[token] = (weakref.ref(precision,
            lambda reference, token=token: _COO_OWNER_BOUNDS.pop(token, None)), stamp)


def _cpu_partials(models, burdens, sample_chunk_size, parts):
    """CPU formula oracle, separately labelled from actual CUDA TF32 kernels."""
    for part in range(parts[0].shape[0]):
        start, stop = part*sample_chunk_size, min((part+1)*sample_chunk_size, len(burdens))
        z = burdens[start:stop]
        for trait, model in enumerate(models):
            precision = model.precision
            if precision is None:
                probability = model.fitted_probability[start:stop]
                p = probability*(1-probability)
            elif precision.layout == torch.strided:
                p = precision[start:stop]
            r = model.scaled_residuals[start:stop]
            sx = model.precision_x if model.precision_x is not None else model.xw.T
            parts[0][part, trait] = (z*r[:, None]).sum(0)
            parts[1][part, trait] = ((z*(p[:, None]*z)).sum(0)
                                   if precision is None or precision.layout == torch.strided else 0.)
            parts[2][part, trait] = sx[start:stop].T@z


def _cpu_sparse_partials(models, burdens, sample_chunk_size, output):
    for part in range(output.shape[0]):
        for trait, model in enumerate(models):
            precision = model.precision
            if precision is None or precision.layout != torch.sparse_coo:
                output[part, trait].zero_()
                continue
            start, stop = part*sample_chunk_size, min((part+1)*sample_chunk_size, precision._nnz())
            indices, values = precision.indices()[:, start:stop], precision.values()[start:stop]
            output[part, trait] = (burdens[indices[0]]*(values[:, None]*burdens[indices[1]])).sum(0)


def _ieee_covariance_projection(cross, models):
    """Multiply ``H.T @ C`` in bounded ordinary FP32 tiles.

    Resident fitted covariances retain their actual strides. This helper has
    one ``[traits, burdens, covariates]`` output and a 3-row pointer table.
    Its CUDA kernel owns only a 4*32*16-element register tile. It neither
    changes global CUDA matmul policy nor allocates a covariate-square slab.
    The containing diagonal-product admission includes these allocations.
    CPU execution is a formula contract oracle, not a GPU performance path.
    """
    models = tuple(models)
    if (not isinstance(cross, torch.Tensor) or cross.layout != torch.strided
            or cross.ndim != 3 or cross.dtype != torch.float32 or cross.requires_grad):
        raise ValueError('IEEE covariance projection requires an FP32 traits-by-covariates-by-burdens tensor')
    t, q, b = cross.shape
    if not 1 <= t <= 32 or len(models) != t or q < 1 or b < 1:
        raise ValueError('IEEE covariance projection has invalid trait, covariate or burden geometry')
    for model in models:
        covariance = model.fixed_effect_covariance
        if (covariance.layout != torch.strided or covariance.shape != (q, q)
                or covariance.dtype != torch.float32 or covariance.device != cross.device
                or covariance.requires_grad):
            raise ValueError('IEEE covariance projection requires aligned fitted FP32 covariances')
    if cross.is_cuda:
        if triton is None:
            raise RuntimeError('CUDA IEEE covariance projection requires Triton')
        from .tf32 import _device_capability
        if _device_capability(cross.device)[0] < 8:
            raise RuntimeError('CUDA IEEE covariance projection requires Ampere or newer hardware')
    output = descriptor = None
    try:
        output = torch.empty((t, b, q), dtype=torch.float32, device=cross.device)
        if cross.is_cuda:
            descriptor = torch.tensor([
                [model.fixed_effect_covariance.data_ptr() for model in models],
                [model.fixed_effect_covariance.stride(0) for model in models],
                [model.fixed_effect_covariance.stride(1) for model in models]],
                dtype=torch.int64, device=cross.device)
            _project_ieee[(t*triton.cdiv(b, 4)*triton.cdiv(q, 16),)](
                cross, descriptor, output, b, q, t, *cross.stride(),
                RB=4, RQ=16, RK=32, num_warps=4, enable_fp_fusion=False)
        else:
            for trait, model in enumerate(models):
                covariance = model.fixed_effect_covariance
                for first_b in range(0, b, 4):
                    last_b = min(first_b+4, b)
                    for first_q in range(0, q, 16):
                        last_q = min(first_q+16, q)
                        acc = torch.zeros((last_b-first_b, last_q-first_q),
                                          dtype=torch.float32, device=cross.device)
                        for first_k in range(0, q, 32):
                            last_k = min(first_k+32, q)
                            left = cross[trait, first_k:last_k, first_b:last_b].T
                            right = covariance[first_k:last_k, first_q:last_q]
                            acc.add_((left[:, :, None]*right[None, :, :]).sum(1))
                        output[trait, first_b:last_b, first_q:last_q] = acc
        result = output
        output = None
        return result
    finally:
        # An allocation or kernel exception must not pin new result storage.
        output = descriptor = None


def diagonal_burden_products(models, burdens, *, sample_chunk_size=8192):
    """Return independent ``score[T,B]`` and ``diag(V)[T,B]`` with bounded partials.

    The caller validates and admits workspace before entering this function.
    Dense FP32 diagonal or original coalesced FP32 COO precision is accepted.
    COO computes every entry in an independent NNZ-split kernel. CUDA uses an
    IEEE FP32 score/base reduction and an explicit RNA TF32 sample-split cross
    projection. Singleton cross outputs retain the mature IEEE FP32 GEMV/dot
    route. Partial products and their reduction remain FP32. The small fitted
    covariance projection uses bounded ordinary FP32 products; its supplied
    small entries are not rounded to TF32. Singleton matrix outputs retain
    their mature ordinary FP32 vector route.
    CPU execution is a formula oracle, never reported as a TF32 GPU execution.
    """
    models = tuple(models)
    if not models or len(models) > 32:
        raise ValueError('diagonal burden products require between 1 and 32 traits')
    if type(sample_chunk_size) is not int or sample_chunk_size < 1:
        raise ValueError('sample_chunk_size must be a positive integer')
    if (not isinstance(burdens, torch.Tensor) or burdens.ndim != 2
            or burdens.layout != torch.strided or burdens.dtype != torch.float32
            or burdens.requires_grad):
        raise ValueError('diagonal burdens require a dense FP32 samples-by-weights tensor without gradients')
    n, b = burdens.shape
    q, t = models[0].x.shape[1], len(models)
    sparse_models, maximum_nnz = 0, 0
    for model in models:
        sx = model.precision_x if model.precision_x is not None else model.xw.T
        precision = model.precision if model.precision is not None else model.fitted_probability
        values = ((sx, (n, q)),
                  (model.scaled_residuals, (n,)), (model.fixed_effect_covariance, (q, q)))
        if precision.device != burdens.device:
            raise ValueError('diagonal burden precision must share the burden device')
        if precision.layout == torch.sparse_coo:
            _validate_coo_geometry(precision, n)
            maximum_nnz = max(maximum_nnz, precision._nnz())
            sparse_models += 1
        elif (precision.layout != torch.strided or precision.dtype != torch.float32
                or precision.shape != (n,) or precision.device != burdens.device
                or precision.requires_grad):
            raise ValueError('diagonal burden products require aligned independent fitted FP32 diagonal states')
        if model.n != n or model.x.shape[1] != q or any(
                value.layout != torch.strided or value.dtype != torch.float32
                or value.shape != shape or value.device != burdens.device or value.requires_grad
                for value, shape in values):
            raise ValueError('diagonal burden products require aligned independent fitted FP32 diagonal states')
    chunks = (n+sample_chunk_size-1)//sample_chunk_size
    nnz_chunks = (maximum_nnz+sample_chunk_size-1)//sample_chunk_size
    shape = (chunks, t, b)
    parts = (torch.empty(shape, dtype=torch.float32, device=burdens.device),
             torch.empty(shape, dtype=torch.float32, device=burdens.device),
             torch.empty((chunks, t, q, b), dtype=torch.float32, device=burdens.device))
    descriptor = projected = sparse_parts = None
    try:
        if sparse_models:
            sparse_parts = torch.empty((nnz_chunks, t, b), dtype=torch.float32, device=burdens.device)
        if burdens.is_cuda:
            if triton is None:
                raise RuntimeError('CUDA diagonal burden products require Triton')
            from .tf32 import _device_capability
            if _device_capability(burdens.device)[0] < 8:
                raise RuntimeError('CUDA diagonal burden products require Ampere or newer hardware')
            if chunks and b:
                descriptor = torch.tensor(_descriptor_rows(models), dtype=torch.int64,
                                          device=burdens.device)
                _score_diagonal[(chunks*t, triton.cdiv(b, 16))](
                    burdens, descriptor, parts[0], parts[1], n, b, t,
                    *burdens.stride(), CHUNK=sample_chunk_size, RN=256, RB=16,
                    num_warps=4, enable_fp_fusion=False)
                if q > 1 and b > 1:
                    _cross[(chunks*t, triton.cdiv(q, 16), triton.cdiv(b, 16))](
                        burdens, descriptor, parts[2], n, b, q, t,
                        *burdens.stride(), CHUNK=sample_chunk_size, RQ=16, RB=16, RK=32,
                        num_warps=4)
            if nnz_chunks and b:
                if descriptor is None:
                    descriptor = torch.tensor(_descriptor_rows(models), dtype=torch.int64,
                                              device=burdens.device)
                _sparse_base[(nnz_chunks*t, triton.cdiv(b, 16))](
                    burdens, descriptor, sparse_parts, b, t, *burdens.stride(),
                    CHUNK=sample_chunk_size, RE=256, RB=16, num_warps=4,
                    enable_fp_fusion=False)
        else:
            _cpu_partials(models, burdens, sample_chunk_size, parts)
            if sparse_models:
                _cpu_sparse_partials(models, burdens, sample_chunk_size, sparse_parts)
        scores, base = (part.sum(0) for part in parts[:2])
        if sparse_models:
            base.add_(sparse_parts.sum(0))
            sparse_parts = None
        singleton_cross = burdens.is_cuda and (q == 1 or b == 1)
        if singleton_cross:
            # Native matrix products intentionally use ordinary FP32 vectors
            # when either output dimension is one. Preserve that precision
            # policy, rather than rounding these vector inputs to TF32.
            from .tf32 import batched_matmul
            cross = batched_matmul([
                (model.precision_x if model.precision_x is not None else model.xw.T).T
                for model in models], burdens, mode='tf32')
        else:
            cross = parts[2].sum(0)
        parts = ()
        ieee_covariance = burdens.is_cuda and q > 1 and b > 1
        if ieee_covariance:
            projected = _ieee_covariance_projection(cross, models)
        elif burdens.is_cuda:
            from .tf32 import batched_matmul
            projected = batched_matmul(cross.transpose(1, 2),
                [model.fixed_effect_covariance for model in models], mode='tf32')
        else:
            projected = torch.stack([value.T@model.fixed_effect_covariance
                                     for value, model in zip(cross, models)])
        variances = base-(projected*cross.transpose(1, 2)).sum(-1)
        return scores, variances, dict(sample_chunks=chunks,
            sparse_nnz_chunks=nnz_chunks,
            sparse_precision_products=sparse_models,
            sparse_nnz_terms=sum(model.precision._nnz() for model in models
                                 if model.precision is not None and model.precision.layout == torch.sparse_coo),
            split_projection_products=chunks*t*int(not singleton_cross),
            legacy_vector_cross_products=t*int(singleton_cross),
            cuda_diagonal_kernel_launches=int(burdens.is_cuda and chunks > 0 and b > 0)
                + int(burdens.is_cuda and chunks > 0 and b > 1 and q > 1)
                + int(burdens.is_cuda and nnz_chunks > 0 and b > 0)
                + int(ieee_covariance),
            ieee_covariance_projection_batches=int(ieee_covariance),
            cpu_diagonal_oracle_batches=int(not burdens.is_cuda))
    finally:
        # An OOM traceback must not pin partials or caller-derived descriptors.
        parts = (); descriptor = projected = sparse_parts = None
