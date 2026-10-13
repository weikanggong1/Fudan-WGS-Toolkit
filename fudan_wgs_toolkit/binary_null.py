"""Ordinary logistic IRLS and explicit prefitted binary association state.

SPDX-License-Identifier: GPL-3.0-only
IRLS follows the binomial identity used by R's stats::glm.
The mixed fitter below implements one explicitly supplied block-sparse GRM;
it never replaces that GRM with an ordinary model implicitly.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import threading
import weakref
import numpy as np
import torch
from .statistics import score_covariance
from .tf32 import matmul, validate_mode
from .tensor_validation import tensor_all_finite
from .precision_audit import explicit_binary_fitted_projection_fp64


def binary_burden_batch_workspace(n, burdens, covariates, traits):
    """NEW bytes for at most 32 independent normal projections, excluding Z.

    SigmaX views retain their original storage; only precision-applied Z is
    stacked. The bound also covers the IEEE FP32 residual stack, one transient
    weighted Z, small covariance/cross products, and bounded validation.
    """
    t = min(32, traits)
    return (4*n*burdens*(t+1) + 4*n*t
            + 64*t*(burdens*burdens+covariates*burdens) + 256*2**20)


def _guard_binary_burden_batch_workspace(burdens, required):
    """Independent interface admission precedes every optional sample stack."""
    if burdens.is_cuda:
        from .tf32 import _guard_workspace
        _guard_workspace(burdens.device, dict(required_bytes=required, cast_bytes=0,
            layout_copy_allowance_bytes=0, output_bytes=required,
            backend='binary_burden_trait_workspace'),
            left_shape=burdens.shape, right_shape=(burdens.shape[0], 32))


def binary_burden_score_covariance_batch(models, burdens, *, allocation_guard=None):
    """Independent fitted products, using a trait axis for native TF32 GEMM.

    Every trait keeps its own precision, SigmaX, covariance and residual. Sparse
    precision applications remain independent. Ordinary FP64 models and mixed
    covariate geometries retain the established scalar-model calculation.
    A rejected optional batch falls back before any stacked allocation.
    """
    from .tf32 import batched_matmul
    from .phewas_runtime.scores import fp32_score_columns
    models = tuple(models)
    if not models:
        raise ValueError("binary burden batches require at least one model")
    n, b = burdens.shape
    native = (burdens.dtype == torch.float32 and len(models) > 1
        and all(model.family == 'binomial' and model.matmul_mode == 'tf32'
                and not model.use_spa and model.n == n and model.device == burdens.device
                and model.scaled_residuals.dtype == torch.float32
                and model.fixed_effect_covariance.dtype == torch.float32
                and (model.precision is None or model.precision.dtype == torch.float32)
                and (model.precision_x if model.precision_x is not None else model.xw).dtype == torch.float32
                for model in models)
        and len({model.x.shape[1] for model in models}) == 1)
    scores, covariances = [], []
    batches = fallbacks = 0
    for start in range(0, len(models), 32):
        tile = models[start:start+32]
        required = binary_burden_batch_workspace(n, b, max(model.x.shape[1] for model in tile), len(tile))
        admitted = native and len(tile) > 1
        if admitted and allocation_guard is not None:
            try:
                admitted = allocation_guard(required) is not False
            except MemoryError:
                admitted = False
        if admitted:
            try:
                _guard_binary_burden_batch_workspace(burdens, required)
            except MemoryError:
                admitted = False
        if not admitted:
            fallbacks += int(native and len(tile) > 1)
            for model in tile:
                score, covariance = model.score_covariance(burdens)
                scores.append(score)
                covariances.append(covariance)
            continue
        # One finite check for this shared source, rather than one per trait.
        if not tensor_all_finite(burdens):
            raise ValueError("burdens must be a finite samples-by-weights matrix")
        weighted = torch.empty((len(tile), n, b), dtype=torch.float32, device=burdens.device)
        sx, cov = [], []
        for i, model in enumerate(tile):
            precision = model.precision
            if precision is None:
                precision = model.fitted_probability*(1-model.fitted_probability)
            if precision.ndim == 1:
                value = precision[:, None]*burdens
            elif precision.layout != torch.strided:
                value = torch.sparse.mm(precision, burdens)
            else:
                value = matmul(precision, burdens, mode='tf32')
            weighted[i].copy_(value)
            del value
            sx.append((model.precision_x if model.precision_x is not None else model.xw.T).T)
            cov.append(model.fixed_effect_covariance)
        cross = batched_matmul(sx, burdens, mode='tf32')
        base = batched_matmul(burdens.T, weighted, mode='tf32')
        del weighted
        projected = batched_matmul(cross.transpose(1, 2), cov, mode='tf32')
        covariance = base-batched_matmul(projected, cross, mode='tf32')
        covariance = (covariance+covariance.transpose(1, 2))/2
        score = fp32_score_columns(burdens, [model.scaled_residuals for model in tile]).T
        scores.extend(score.unbind(0))
        covariances.extend(covariance.unbind(0))
        batches += 1
    return torch.stack(scores), torch.stack(covariances), dict(
        native_projection_batches=batches, admission_fallbacks=fallbacks)


def binary_burden_diag_workspace(n, burdens, covariates, traits, *, sample_chunk_size=8192,
                                 maximum_nnz=0):
    """Bound NEW FP32 sample partials and small products; fitted state is resident.

    This estimate contains no ``traits x samples x burdens`` or burden-by-burden
    covariance. At most 32 independent traits are processed in one tile.
    """
    if type(sample_chunk_size) is not int or sample_chunk_size < 1:
        raise ValueError('sample_chunk_size must be a positive integer')
    if type(maximum_nnz) is not int or maximum_nnz < 0:
        raise ValueError('maximum_nnz must be a nonnegative integer')
    t = min(32, traits)
    parts = (n+sample_chunk_size-1)//sample_chunk_size
    nnz_parts = (maximum_nnz+sample_chunk_size-1)//sample_chunk_size
    vector_layout = (4*t*n*(burdens+covariates)
                     if burdens == 1 or covariates == 1 else 0)
    return (4*parts*t*burdens*(2+covariates) + 4*nnz_parts*t*burdens
            + 32*t*burdens*(2+covariates) + vector_layout + 120*t + 256*2**20)


def _guard_binary_burden_diag_workspace(burdens, required):
    """Admit all NEW sample partials before allocating their first tensor."""
    if burdens.is_cuda:
        from .tf32 import _guard_workspace
        _guard_workspace(burdens.device, dict(required_bytes=required, cast_bytes=0,
            layout_copy_allowance_bytes=0, output_bytes=required,
            backend='binary_burden_diagonal_workspace'),
            left_shape=burdens.shape, right_shape=(burdens.shape[0], 32))


def _normal_fp32_burden_accuracy_required(model, burdens):
    return (burdens.is_cuda and burdens.dtype == torch.float32
            and model.matmul_mode == 'tf32' and not model.use_spa)


def _native_burden_diagonal_supported(models, burdens):
    if not (burdens.is_cuda and burdens.dtype == torch.float32 and not burdens.requires_grad):
        return False
    n, unused_b = burdens.shape
    q = models[0].x.shape[1]
    for model in models:
        sx = model.precision_x if model.precision_x is not None else model.xw.T
        precision = model.precision if model.precision is not None else model.fitted_probability
        precision_supported = (precision.dtype == torch.float32
            and precision.device == burdens.device and not precision.requires_grad
            and ((precision.layout == torch.strided and precision.shape == (n,))
                 or (precision.layout == torch.sparse_coo and precision.shape == (n, n)
                     and precision.is_coalesced())))
        if not (model.family == 'binomial' and model.n_pheno == 1
                and model.matmul_mode == 'tf32' and not model.use_spa
                and model.x.shape[1] == q and precision_supported and all(
                    value.layout == torch.strided and value.dtype == torch.float32
                    and value.shape == shape and value.device == burdens.device
                    and not value.requires_grad for value, shape in (
                        (sx, (n, q)),
                        (model.scaled_residuals, (n,)),
                        (model.fixed_effect_covariance, (q, q))))):
            return False
    return True


def _try_burden_diagonal_products(models, burdens, *, allocation_guard, sample_chunk_size):
    """Optional tile admission precedes every native partial allocation."""
    from ._binary_burden_diag import diagonal_burden_products
    n, b = burdens.shape
    q = models[0].x.shape[1]
    maximum_nnz = max((model.precision._nnz() for model in models
        if model.precision is not None and model.precision.layout == torch.sparse_coo), default=0)
    required = binary_burden_diag_workspace(n, b, q, len(models),
        sample_chunk_size=sample_chunk_size, maximum_nnz=maximum_nnz)
    try:
        if allocation_guard is not None and allocation_guard(required) is False:
            return None
        _guard_binary_burden_diag_workspace(burdens, required)
    except MemoryError:
        return None
    return diagonal_burden_products(models, burdens, sample_chunk_size=sample_chunk_size)


def _accurate_independent_burden_workspace(model, burdens):
    """Conservative NEW weighted/layout/cast space for one original model."""
    n, b = burdens.shape
    q = model.x.shape[1]
    precision = model.precision
    precision_bytes = 0
    if precision is not None:
        if precision.layout == torch.strided and precision.ndim == 2:
            # Native dense TF32 products may need a contiguous precision copy;
            # a dtype conversion is separately allowed before this layout copy.
            precision_bytes = 4*precision.numel()
            if precision.dtype != torch.float32:
                precision_bytes += 4*precision.numel()
        elif precision.layout != torch.strided:
            # Sparse casts/coalescing/SPMM may own indices plus values. Count
            # complete extra sparse source storage twice rather than an N*N.
            values = precision.values() if precision.layout != torch.sparse_coo else precision._values()
            if precision.layout == torch.sparse_coo:
                index_bytes = precision._indices().numel()*precision._indices().element_size()
            elif precision.layout in (torch.sparse_csr, torch.sparse_bsr):
                index_bytes = (precision.crow_indices().numel()*precision.crow_indices().element_size()
                               + precision.col_indices().numel()*precision.col_indices().element_size())
            elif precision.layout in (torch.sparse_csc, torch.sparse_bsc):
                index_bytes = (precision.ccol_indices().numel()*precision.ccol_indices().element_size()
                               + precision.row_indices().numel()*precision.row_indices().element_size())
            else:
                raise ValueError('accurate independent burden fallback requires supported sparse precision storage')
            precision_bytes = 2*(index_bytes+4*values.numel())
    return (24*n*b + 16*n*q + 32*b*q + 8*q*q + 128*q
            + precision_bytes + 256*2**20)


def _accurate_independent_burden_products(model, burdens, *, allocation_guard):
    """Keep original precision/long-H products with an accurate small C.

    This independent fallback never constructs a burden-square covariance.
    It admits weighted[N,B], source casts/layouts and its bounded tiny-C
    projection before calling the mature supplied-precision implementation.
    Failure to admit raises; it cannot silently return the old rounded C.
    """
    from types import SimpleNamespace
    from ._binary_burden_diag import _ieee_covariance_projection
    required = _accurate_independent_burden_workspace(model, burdens)
    if allocation_guard is not None and allocation_guard(required) is False:
        raise MemoryError('accurate independent binary burden workspace was not admitted')
    _guard_binary_burden_diag_workspace(burdens, required)
    g = weighted = cross = projected = covariance = covariance_model = score = variance = None
    try:
        g, mode = model._association_genotype(burdens, None)
        weighted, cross = model._fitted_projection(g, mode)
        covariance = model.fixed_effect_covariance.to(dtype=g.dtype)
        if g.shape[1] == 1 or cross.shape[0] == 1:
            from .tf32 import batched_matmul
            projected = batched_matmul(cross.T[None, :, :], [covariance], mode='tf32')[0]
        else:
            covariance_model = SimpleNamespace(fixed_effect_covariance=covariance)
            projected = _ieee_covariance_projection(cross[None, :, :], [covariance_model])[0]
        variance = (g*weighted).sum(0)-(projected*cross.T).sum(-1)
        score = matmul(g.T, model.scaled_residuals.to(dtype=g.dtype), mode=mode)
        return score, variance
    finally:
        g = weighted = cross = projected = covariance = covariance_model = score = variance = None


def binary_burden_score_variance_batch(models, burdens, *, allocation_guard=None,
                                      sample_chunk_size=8192, trait_batch_size=32):
    """Return ``score[traits, weights]`` and only ``diag(V)[traits, weights]``.

    CUDA FP32 diagonal or coalesced COO precision uses independent sample/NNZ
    chunks to expose GPU parallelism. Every COO entry, including off-diagonal
    precision, is retained. IEEE FP32 diagonal sums avoid unused burden
    products. The long fixed-effect cross projection retains native TF32,
    while its small fitted covariance contraction uses ordinary IEEE FP32.
    All reductions remain FP32: this does not reconstruct FP64 products.
    Other CUDA normal FP32 sparse/dense precision uses an independently
    admitted original weighted-genotype/long-cross product and the same
    accurate small covariance contraction, without a burden-square matrix.
    Rejected trait batches retry single-trait native work; a denied accurate
    single-trait budget raises MemoryError. CPU/FP64/SPA controls retain their
    mature full-covariance calculation and discard its unused off-diagonal
    results.
    The original full-covariance interface remains available independently.
    """
    models = tuple(models)
    if not models:
        raise ValueError('binary burden batches require at least one model')
    if type(trait_batch_size) is not int or not 1 <= trait_batch_size <= 32:
        raise ValueError('trait_batch_size must be an integer between 1 and 32')
    if type(sample_chunk_size) is not int or sample_chunk_size < 1:
        raise ValueError('sample_chunk_size must be a positive integer')
    if not isinstance(burdens, torch.Tensor) or burdens.ndim != 2:
        raise ValueError('burdens must be a samples-by-weights matrix')
    n, b = burdens.shape
    if any(model.n != n or model.device != burdens.device for model in models):
        raise ValueError('burdens and fitted models must share the ordered sample dimension and device')
    if any(_normal_fp32_burden_accuracy_required(model, burdens) for model in models):
        for model in models:
            if not _normal_fp32_burden_accuracy_required(model, burdens):
                continue
            sx = model.precision_x if model.precision_x is not None else model.xw.T
            precision = model.precision if model.precision is not None else model.fitted_probability
            if burdens.requires_grad or any(value.requires_grad for value in
                    (sx, precision, model.scaled_residuals, model.fixed_effect_covariance)):
                raise ValueError('normal FP32 binary burden products require inference tensors without gradients')
    if not tensor_all_finite(burdens):
        raise ValueError('burdens must be a finite samples-by-weights matrix')
    scores, variances = [], []
    counts = dict(native_projection_batches=0, admission_fallbacks=0,
                  native_diagonal_batches=0, diagonal_admission_fallbacks=0,
                  sample_chunks=0, split_projection_products=0,
                  legacy_vector_cross_products=0, cuda_diagonal_kernel_launches=0,
                  cpu_diagonal_oracle_batches=0, sparse_nnz_chunks=0,
                  sparse_precision_products=0, sparse_nnz_terms=0,
                  native_sparse_diagonal_batches=0,
                  ieee_covariance_projection_batches=0,
                  accurate_independent_projection_batches=0,
                  independent_diagonal_retries=0)
    def append_diagonal(result):
        score, variance, execution = result
        scores.extend(score.unbind(0))
        variances.extend(variance.unbind(0))
        counts['native_projection_batches'] += 1
        counts['native_diagonal_batches'] += 1
        counts['native_sparse_diagonal_batches'] += int(execution['sparse_precision_products'] > 0)
        for key, value in execution.items():
            counts[key] += value
    for start in range(0, len(models), trait_batch_size):
        tile = models[start:start+trait_batch_size]
        native = _native_burden_diagonal_supported(tile, burdens)
        result = (_try_burden_diagonal_products(tile, burdens,
            allocation_guard=allocation_guard, sample_chunk_size=sample_chunk_size) if native else None)
        if result is not None:
            append_diagonal(result)
            continue
        if native:
            counts['admission_fallbacks'] += 1
            counts['diagonal_admission_fallbacks'] += 1
        if not any(_normal_fp32_burden_accuracy_required(model, burdens) for model in tile):
            # CPU/FP64/SPA controls retain the mature full-covariance contract.
            score, covariance, execution = binary_burden_score_covariance_batch(
                tile, burdens, allocation_guard=allocation_guard)
            scores.extend(score.unbind(0))
            variances.extend(covariance.diagonal(dim1=-2, dim2=-1).clone().unbind(0))
            counts['native_projection_batches'] += execution['native_projection_batches']
            counts['admission_fallbacks'] += execution['admission_fallbacks']
            continue
        # Different user-selected covariate dimensions and rejected optional
        # trait stacks use one independently admitted original model at a time.
        for model in tile:
            if _native_burden_diagonal_supported((model,), burdens):
                counts['independent_diagonal_retries'] += 1
                result = _try_burden_diagonal_products((model,), burdens,
                    allocation_guard=allocation_guard, sample_chunk_size=sample_chunk_size)
                if result is None:
                    raise MemoryError('accurate independent binary diagonal workspace was not admitted')
                append_diagonal(result)
            elif _normal_fp32_burden_accuracy_required(model, burdens):
                score, variance = _accurate_independent_burden_products(model, burdens,
                    allocation_guard=allocation_guard)
                scores.append(score)
                variances.append(variance)
                counts['accurate_independent_projection_batches'] += 1
                ieee_projection = int(burdens.is_cuda and model.x.shape[1] > 1 and b > 1)
                counts['ieee_covariance_projection_batches'] += ieee_projection
                counts['cuda_diagonal_kernel_launches'] += ieee_projection
            else:
                score, covariance, execution = binary_burden_score_covariance_batch(
                    (model,), burdens, allocation_guard=allocation_guard)
                scores.extend(score.unbind(0))
                variances.extend(covariance.diagonal(dim1=-2, dim2=-1).clone().unbind(0))
                counts['native_projection_batches'] += execution['native_projection_batches']
                counts['admission_fallbacks'] += execution['admission_fallbacks']
    return torch.stack(scores), torch.stack(variances), counts


@dataclass
class BinaryNullModel:
    sample_ids: np.ndarray
    x: torch.Tensor
    scaled_residuals: torch.Tensor
    fitted_probability: torch.Tensor
    xw: torch.Tensor
    projection_left: torch.Tensor
    fixed_effect_covariance: torch.Tensor
    precision: torch.Tensor | None = None
    precision_x: torch.Tensor | None = None
    coefficients: torch.Tensor | None = None
    phenotype: torch.Tensor | None = None
    iterations: int = 0
    converged: bool = True
    has_kinship: bool = False
    family: str = "binomial"
    n_pheno: int = 1
    use_spa: bool = True
    fit_method: str = "prefitted_binary_state"
    matmul_mode: str = "fp64"
    working_phenotype: torch.Tensor | None = None

    @property
    def n(self):
        return self.x.shape[0]

    @property
    def device(self):
        return self.x.device

    @property
    def dof(self):
        return self.n - self.x.shape[1]

    def set_matmul_mode(self, mode):
        """Convert fitted tensors without replacing or refitting binary state.

        Native TF32 applies only to non-SPA association; the established SPA
        interface keeps its FP64 arithmetic. Source mode is retained separately
        from the selected association storage mode.
        """
        mode = validate_mode(mode)
        if mode == "tf32" and self.use_spa:
            raise ValueError("binary SPA requires fp64; native TF32 supports use_spa=False")
        if not hasattr(self, "source_matmul_mode"):
            self.source_matmul_mode = self.matmul_mode
        dtype = torch.float32 if mode == "tf32" else torch.float64
        for name in ("x", "scaled_residuals", "fitted_probability", "xw",
                     "projection_left", "fixed_effect_covariance", "precision",
                     "precision_x", "coefficients", "phenotype", "working_phenotype"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(dtype=dtype))
        self.matmul_mode = mode
        self._centered_single_projection_cache = None
        return self

    def _association_genotype(self, genotype, matmul_mode):
        mode = validate_mode(self.matmul_mode if matmul_mode is None else matmul_mode)
        if mode == "tf32" and self.use_spa:
            raise ValueError("binary SPA requires fp64; native TF32 supports use_spa=False")
        dtype = torch.float32 if mode == "tf32" else torch.float64
        g = torch.as_tensor(genotype, dtype=dtype, device=self.device)
        if g.ndim != 2 or g.shape[0] != self.n or not tensor_all_finite(g):
            raise ValueError("genotype must be a finite samples-by-variants matrix")
        return g, mode

    def _fitted_projection(self, genotype, mode):
        """Apply supplied Sigma_i, retaining original Sigma_iX and cov.

        Diagonal precision uses a rowwise product. Explicit sparse precision
        uses a sparse FP32/FP64 product without materializing an N by N matrix.
        Dense products and fixed-effect projection use the selected mode.
        """
        g = genotype
        precision = self.precision
        if precision is None:
            precision = self.fitted_probability * (1 - self.fitted_probability)
        precision = precision.to(dtype=g.dtype)
        if precision.ndim == 1:
            weighted = precision[:, None] * g
        elif precision.layout != torch.strided:
            weighted = torch.sparse.mm(precision, g)
        else:
            weighted = matmul(precision, g, mode=mode)
        sx = self.precision_x if self.precision_x is not None else self.xw.T
        cross = matmul(sx.to(dtype=g.dtype).T, g, mode=mode)
        return weighted, cross

    def score_covariance(self, genotype, *, matmul_mode=None):
        """Return u=G.T r and V=G.T Sigma_i G-cross.T cov cross.

        Supplied precision_covariates and fixed_effect_covariance are used
        directly; no projection is recomputed from the design matrix.
        """
        g, mode = self._association_genotype(genotype, matmul_mode)
        # Keep the historical ordinary-logistic/SPA path unchanged in FP64.
        if mode == "fp64" and self.precision is None:
            return score_covariance(g, self.scaled_residuals, covariates=self.x,
                                    working_weights=self.fitted_probability * (1-self.fitted_probability))
        if mode == "fp64" and self.precision.ndim == 1 and (self.precision_x is None or self.use_spa):
            return score_covariance(g, self.scaled_residuals, covariates=self.x,
                                    working_weights=self.precision)
        weighted, cross = self._fitted_projection(g, mode)
        cov = self.fixed_effect_covariance.to(dtype=g.dtype)
        covariance = matmul(g.T, weighted, mode=mode) - matmul(
            matmul(cross.T, cov, mode=mode), cross, mode=mode)
        return (matmul(g.T, self.scaled_residuals.to(dtype=g.dtype), mode=mode),
                (covariance + covariance.T) / 2)

    def _diagonal_tf32_covariance_state(self, matmul_mode=None):
        """Validate an explicit fitted diagonal state for bounded covariance.

        Return the original precision, Sigma_iX, residual and fitted covariance.
        This never derives or fits a replacement projection and never treats a
        dense or sparse precision as diagonal. Cached and tiled association are
        inference interfaces; tensors requiring gradients are rejected so panel
        lifetimes remain bounded.
        """
        mode = validate_mode(self.matmul_mode if matmul_mode is None else matmul_mode)
        if mode != "tf32" or self.matmul_mode != "tf32":
            raise ValueError("bounded binary covariance requires a native TF32 model")
        if self.family != "binomial" or self.n_pheno != 1 or self.use_spa:
            raise NotImplementedError("bounded binary covariance requires one non-SPA binomial model")
        precision, sx = self.precision, self.precision_x
        if precision is None or sx is None:
            raise NotImplementedError("bounded binary covariance requires explicit fitted precision and precision_covariates")
        n, q = self.x.shape
        values = (("x", self.x, (n, q)), ("precision", precision, (n,)),
                  ("precision_x", sx, (n, q)),
                  ("scaled_residuals", self.scaled_residuals, (n,)),
                  ("fixed_effect_covariance", self.fixed_effect_covariance, (q, q)))
        for name, value, shape in values:
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or value.shape != shape or value.dtype != torch.float32
                    or value.device != self.device or value.requires_grad):
                raise ValueError(f"model.{name} must be aligned dense FP32 without gradients on the model device")
            if not tensor_all_finite(value):
                raise ValueError(f"model.{name} must be finite")
        if n < 1 or bool((precision <= 0).any()):
            raise ValueError("binary diagonal precision must be positive")
        return precision, sx, self.scaled_residuals, self.fixed_effect_covariance

    def score_covariance_cached(self, genotype, *, variant_tile_size=4096,
                                panel_variant_size=None, memory_limit_gib=40,
                                matmul_mode=None, profile=False, compute_score=True):
        """Return fitted binary score, covariance and a bounded cache report.

        Host FP32 dosage is already oriented/imputed and ordered [N, M]. The
        shared native TF32 cache retains 512-column score/projection geometry,
        bounds original/weighted panels against live CUDA memory, and averages
        both directed covariance products. All supplied fitted state is reused.
        Output score [M] and covariance [M, M] stay CUDA FP32. Only explicit
        diagonal non-SPA state is supported; no fitting or statistical fallback
        occurs. Parameter meanings and admission rules match the Gaussian cache.
        ``compute_score=False`` omits only an already shared score and returns
        None in its place; covariance preparation/products remain unchanged.
        """
        mode = validate_mode(self.matmul_mode if matmul_mode is None else matmul_mode)
        from ._cached_covariance import score_covariance_cached
        return score_covariance_cached(
            self, genotype, variant_tile_size=variant_tile_size,
            panel_variant_size=panel_variant_size, memory_limit_gib=memory_limit_gib,
            matmul_mode=mode, symmetry="average", profile=profile, compute_score=compute_score)

    def score_covariance_tiled(self, genotype, *, variant_tile_size=256, matmul_mode=None):
        """Bounded host-to-device variant tiles using the fitted binary formula.

        This uses the existing Gaussian tiled product order with binary fitted
        diagonal precision: U=G.T r and V=G.T Sigma_i G-cross.T cov cross.
        Both directions are averaged before host FP32 assembly. The full [M,M]
        covariance is returned on the model device; callers retain their dense
        output/solver workspace guards. This does not alter SPA or FP64 paths.
        """
        inv, precision_x, residual, fixed_cov = self._diagonal_tf32_covariance_state(matmul_mode)
        if type(variant_tile_size) is not int or variant_tile_size < 1:
            raise ValueError("variant_tile_size must be a positive integer")
        if isinstance(genotype, torch.Tensor) and (genotype.is_cuda or genotype.requires_grad):
            raise ValueError("tiled binary genotype must be host data without gradients")
        source = torch.as_tensor(genotype, dtype=torch.float32, device="cpu")
        if source.ndim != 2 or source.shape[0] != self.n:
            raise ValueError("genotype must be a samples-by-variants matrix aligned to the model")
        if not tensor_all_finite(source):
            raise ValueError("genotype must be finite")
        n, m = map(int, source.shape)
        if not m:
            return (torch.empty((0,), dtype=torch.float32, device=self.device),
                    torch.empty((0, 0), dtype=torch.float32, device=self.device))
        device = self.device
        score = torch.empty((m,), dtype=torch.float32, device=device)
        covariance_host = torch.empty((m, m), dtype=torch.float32, device="cpu")
        for start in range(0, m, variant_tile_size):
            stop = min(start + variant_tile_size, m)
            left = torch.as_tensor(source[:, start:stop], dtype=torch.float32, device=device)
            score[start:stop] = matmul(left.T, residual, mode="tf32")
            left_cross = matmul(precision_x.T, left, mode="tf32")
            for other in range(start, m, variant_tile_size):
                other_stop = min(other + variant_tile_size, m)
                right = torch.as_tensor(source[:, other:other_stop], dtype=torch.float32, device=device)
                right_cross = matmul(precision_x.T, right, mode="tf32")
                block = (matmul(left.T, inv[:, None] * right, mode="tf32")
                         - matmul(matmul(left_cross.T, fixed_cov, mode="tf32"), right_cross, mode="tf32"))
                if start == other:
                    block = (block + block.T) / 2
                else:
                    reverse = (matmul(right.T, inv[:, None] * left, mode="tf32")
                               - matmul(matmul(right_cross.T, fixed_cov, mode="tf32"), left_cross, mode="tf32"))
                    block = (block + reverse.T) / 2
                    del reverse
                covariance_host[start:stop, other:other_stop].copy_(block.detach().cpu())
                if other != start:
                    covariance_host[other:other_stop, start:stop].copy_(block.T.detach().cpu())
                del right, right_cross, block
            del left, left_cross
        covariance = covariance_host.to(device=device, dtype=torch.float32)
        return score, covariance

    def _centered_single_projection_state(self):
        """Constant-direction correction for a fixed non-SPA diagonal model."""
        from .single_projection import centered_projection_state
        return centered_projection_state(
            self, self.precision, self.precision_x, product=matmul,
            blocked=self.use_spa)

    def individual_score_variance(self, genotype, *, matmul_mode=None):
        """Use the same fitted score and only diag(V), without M by M output."""
        g, mode = self._association_genotype(genotype, matmul_mode)
        if mode == "fp64":
            # Keep the established control/SPA score_covariance branch and its
            # projection arithmetic. Diagonal-only optimization is native TF32.
            score, covariance = self.score_covariance(g, matmul_mode=mode)
            return score, covariance.diagonal()
        centered_state = (self._centered_single_projection_state()
                          if not g.requires_grad else None)
        if centered_state is not None:
            score = matmul(g.T, self.scaled_residuals, mode=mode)
            mean = g.mean(dim=0)
            centered = g - mean[None, :]
            weighted = self.precision[:, None] * centered
            cross = matmul(self.precision_x.T, centered, mode=mode)
            projected = matmul(cross.T, self.fixed_effect_covariance, mode=mode)
            variance = ((centered * weighted).sum(dim=0)
                        - (projected * cross.T).sum(dim=1))
            del weighted, cross, projected
            p_one, p_one_sum = centered_state
            variance += (2 * mean * matmul(centered.T, p_one, mode=mode)
                         + mean.square() * p_one_sum)
            return score, variance
        weighted, cross = self._fitted_projection(g, mode)
        projected = matmul(cross.T, self.fixed_effect_covariance.to(dtype=g.dtype), mode=mode)
        variance = (g * weighted).sum(dim=0) - (projected * cross.T).sum(dim=1)
        return matmul(g.T, self.scaled_residuals.to(dtype=g.dtype), mode=mode), variance


def binary_prefitted_state(*, sample_ids, covariates, residual, fitted_probability,
                           xw, projection_left, fixed_effect_covariance,
                           precision=None, precision_covariates=None,
                           coefficients=None, phenotype=None, has_kinship=False,
                           use_spa=True, device="cuda", provenance="prefitted_binary_state",
                           matmul_mode="fp64", working_phenotype=None):
    """Use explicitly supplied state; this function performs no null fitting.

    Ordinary xw=X.T*mu*(1-mu), left=X*inv(X.T*W*X). Mixed xw=X.T*Sigma_i,
    left=X*cov. A mixed state additionally needs its fitted precision and
    precision_covariates for score-based association and SPA filtering.
    """
    mode = validate_mode(matmul_mode)
    if mode == "tf32" and use_spa:
        raise ValueError("binary SPA requires fp64; native TF32 supports use_spa=False")
    dtype = torch.float32 if mode == "tf32" else torch.float64
    def tensor(value):
        return None if value is None else torch.as_tensor(value, dtype=dtype, device=device)
    x,r,mu,wx,left,cov = map(tensor,(covariates,residual,fitted_probability,xw,projection_left,fixed_effect_covariance))
    if x.ndim!=2 or x.shape[1]<1:
        raise ValueError("binary covariates must include a fixed-effect design")
    n,p=x.shape;ids=np.asarray(sample_ids,dtype=str)
    if ids.shape!=(n,) or len(set(ids))!=n or r.shape!=(n,) or mu.shape!=(n,) or wx.shape!=(p,n) or left.shape!=(n,p) or cov.shape!=(p,p):
        raise ValueError("prefitted binary arrays and sample IDs have incompatible shapes")
    if not all(bool(torch.isfinite(value).all()) for value in (x,r,mu,wx,left,cov)) or bool(((mu<=0)|(mu>=1)).any()):
        raise ValueError("binary state must be finite with fitted probabilities strictly in (0,1)")
    inverse,sx = tensor(precision),tensor(precision_covariates)
    if has_kinship and (inverse is None or sx is None):
        raise ValueError("mixed prefitted binary state requires fitted precision and precision_covariates")
    if inverse is not None and inverse.shape not in ((n,),(n,n)):
        raise ValueError("binary precision must be aligned diagonal or samples-by-samples")
    if sx is not None and sx.shape!=(n,p):
        raise ValueError("binary precision_covariates must have shape samples-by-covariates")
    for value in (inverse, sx):
        if value is not None:
            finite_value = value if value.layout == torch.strided else value.to_sparse_coo().coalesce().values()
            if not bool(torch.isfinite(finite_value).all()):
                raise ValueError("binary fitted precision state must be finite")
    if inverse is not None and inverse.ndim == 1 and bool((inverse <= 0).any()):
        raise ValueError("binary diagonal precision must be positive")
    # Check labels before FP32 conversion; rounding a working response to 0/1
    # must not turn an invalid typed phenotype into apparently valid labels.
    raw_y = None if phenotype is None else torch.as_tensor(phenotype, dtype=torch.float64, device=device)
    if raw_y is not None and (raw_y.shape != (n,) or not bool(torch.isfinite(raw_y).all()) or bool(((raw_y != 0) & (raw_y != 1)).any())):
        raise ValueError("binary phenotype must be aligned finite 0/1 labels; working response is separate")
    y, working = tensor(phenotype), tensor(working_phenotype)
    if working is not None and (working.shape != (n,) or not bool(torch.isfinite(working).all())):
        raise ValueError("binary working_phenotype must be an aligned finite vector")
    return BinaryNullModel(ids,x,r,mu,wx,left,cov,inverse,sx,tensor(coefficients),y,
                           has_kinship=has_kinship,use_spa=use_spa,fit_method=provenance,
                           matmul_mode=mode,working_phenotype=working)


def _compact_fitted_arrays(*, covariates, residual, fitted_probability,
                           fixed_effect_covariance, precision,
                           null_fit_source_sha256, verified_fit_source_sha256,
                           device):
    binding = null_fit_source_sha256
    if (not isinstance(binding, str) or len(binding) != 64
            or binding != verified_fit_source_sha256
            or any(ch not in "0123456789abcdef" for ch in binding)):
        raise ValueError("compact binary state requires the same verified fitted-source SHA256")
    def source_tensor(value, name):
        if isinstance(value, torch.Tensor):
            if value.dtype != torch.float64 or value.requires_grad:
                raise ValueError(f"compact binary {name} must be original FP64 source without gradients")
            return value.to(device=device)
        array = np.asarray(value)
        if array.dtype != np.float64:
            raise ValueError(f"compact binary {name} must be original FP64 source")
        return torch.as_tensor(array, dtype=torch.float64, device=device)
    x = source_tensor(covariates, "covariates")
    r = source_tensor(residual, "residual")
    mu = source_tensor(fitted_probability, "fitted_probability")
    cov = source_tensor(fixed_effect_covariance, "fixed_effect_covariance")
    inverse = source_tensor(precision, "precision")
    if (x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1 or r.shape != (x.shape[0],)
            or mu.shape != r.shape or cov.shape != (x.shape[1], x.shape[1])):
        raise ValueError("compact binary arrays must retain the fitted sample/design shape")
    if not (inverse.ndim == 1 or (inverse.ndim == 2 and inverse.layout != torch.strided)):
        raise ValueError("compact binary precision must be diagonal or sparse; dense N-by-N is refused")
    finite_precision = inverse if inverse.layout == torch.strided else inverse.to_sparse_coo().coalesce().values()
    if (not all(bool(torch.isfinite(value).all()) for value in (x, r, mu, cov, finite_precision))
            or bool(((mu <= 0) | (mu >= 1)).any())):
        raise ValueError("compact binary fitted arrays must be finite with probabilities strictly in (0,1)")
    if inverse.ndim == 1:
        if inverse.shape != r.shape or bool((inverse <= 0).any()):
            raise ValueError("compact binary diagonal precision must be aligned and positive")
        sx = inverse[:, None] * x
        method = "FP64_diagonal_source_precision"
    elif inverse.ndim == 2 and inverse.layout != torch.strided:
        if inverse.shape != (x.shape[0], x.shape[0]):
            raise ValueError("compact binary sparse precision has the wrong shape")
        inverse = inverse.to_sparse_coo().coalesce()
        with explicit_binary_fitted_projection_fp64():
            sx = torch.sparse.mm(inverse, x)
        method = "FP64_sparse_source_precision"
    else:
        raise ValueError("compact binary precision must be diagonal or sparse; dense N-by-N is refused")
    return x, r, mu, cov, inverse, sx, binding, method


def compact_spa_state_from_fitted_arrays(*, sample_ids, covariates, residual,
                                       fitted_probability, fixed_effect_covariance,
                                       precision, null_fit_source_sha256,
                                       verified_fit_source_sha256, device="cpu",
                                       coefficients=None, phenotype=None,
                                       has_kinship=False, iterations=0,
                                       provenance="verified_fitted_binary_state"):
    """Rebuild only the FP64 SPA projection from verified source fitted arrays.

    The caller must verify the immutable source receipts and array SHA values
    before calling, and supply the common fitted-state SHA of the normal bank
    as ``verified_fit_source_sha256``. Matching hashes here bind that admission;
    this array-only function cannot authenticate files that it does not read.

    ``covariates`` is the original FP64 X, never an upcast normal FP32 matrix.
    Sigma_i and the supplied covariance are retained at their original finite
    fit tolerance: SX=Sigma_i X, xw=SX.T, and left=X cov. No refit, inversion of
    X'Sigma_iX, working-response reconstruction or dense N-by-N allocation is
    performed. A shared source X can remain resident across sidecars.

    Diagonal SX is the original elementwise product. Sparse SX uses FP64 sparse
    multiplication; its summation order may differ from the fitter's small
    dense family products, and real SPA validation must record that difference.
    """
    x, r, mu, cov, inverse, sx, binding, method = _compact_fitted_arrays(
        covariates=covariates, residual=residual, fitted_probability=fitted_probability,
        fixed_effect_covariance=fixed_effect_covariance, precision=precision,
        null_fit_source_sha256=null_fit_source_sha256,
        verified_fit_source_sha256=verified_fit_source_sha256, device=device)
    with explicit_binary_fitted_projection_fp64():
        left = x @ cov
    model = binary_prefitted_state(sample_ids=sample_ids, covariates=x,
        residual=r, fitted_probability=mu, xw=sx.T, projection_left=left,
        fixed_effect_covariance=cov, precision=inverse, precision_covariates=sx,
        coefficients=coefficients, phenotype=phenotype, has_kinship=has_kinship,
        use_spa=True, device=device, provenance=provenance, matmul_mode="fp64")
    model.null_fit_source_sha256 = binding
    model.iterations = int(iterations)
    model.spa_projection_reconstructed = True
    model.spa_projection_method = method
    return model


def compact_binary_normal_from_fitted_arrays(*, sample_ids, covariates, residual,
                                            fitted_probability, fixed_effect_covariance,
                                            precision, null_fit_source_sha256,
                                            verified_fit_source_sha256, device="cpu",
                                            coefficients=None, has_kinship=False, iterations=0,
                                            provenance="verified_fitted_binary_state",
                                            association_mode="tf32"):
    """Create a temporary normal association tile from the FP64 source bank.

    Original SX=Sigma_i X is reconstructed in FP64, then the supplied fitted
    arrays are cast to the requested association mode. No FP64 left projection,
    labels or working response is materialized; normal association does not use
    them. The repository can release the whole temporary tile afterwards.
    Sparse summation can change final ulps before casting, and is reported rather
    than asserted to be universally bitwise equal to a pre-saved normal bank.
    Source receipt/array verification is the caller's responsibility as for the
    compact SPA helper. The same fitted-state SHA is retained on the output.
    """
    mode = validate_mode(association_mode)
    x, r, mu, cov, inverse, sx, binding, method = _compact_fitted_arrays(
        covariates=covariates, residual=residual, fitted_probability=fitted_probability,
        fixed_effect_covariance=fixed_effect_covariance, precision=precision,
        null_fit_source_sha256=null_fit_source_sha256,
        verified_fit_source_sha256=verified_fit_source_sha256, device=device)
    ids = np.asarray(sample_ids, dtype=str)
    if ids.shape != (x.shape[0],) or len(np.unique(ids)) != len(ids) or np.any(ids == ""):
        raise ValueError("compact binary sample IDs must be unique and aligned")
    dtype = torch.float32 if mode == "tf32" else torch.float64
    beta = None
    if coefficients is not None:
        beta = torch.as_tensor(coefficients, device=device)
        if (beta.dtype != torch.float64 or beta.shape != (x.shape[1],)
                or not bool(torch.isfinite(beta).all())):
            raise ValueError("compact binary coefficients must retain their original FP64 fitted shape")
        beta = beta.to(dtype=dtype)
    x, r, mu, cov, inverse, sx = (value.to(dtype=dtype) for value in (x, r, mu, cov, inverse, sx))
    model = BinaryNullModel(ids, x, r, mu, sx.T, None, cov, precision=inverse,
        precision_x=sx, coefficients=beta, iterations=int(iterations), has_kinship=has_kinship,
        use_spa=False, fit_method=provenance, matmul_mode=mode)
    model.null_fit_source_sha256 = binding
    model.source_matmul_mode = "fp64"
    model.normal_projection_reconstructed = True
    model.normal_projection_method = method + "_cast_" + mode
    return model


# Weak ownership retains an axis only while admitted sources or derived models
# actually use it. Equal cohorts avoid one anonymous ID bytes copy per trait.
_IMMUTABLE_BINARY_ID_AXES = weakref.WeakValueDictionary()
_IMMUTABLE_BINARY_ID_AXES_LOCK = threading.Lock()


def _intern_immutable_binary_sample_ids(ids):
    """Reuse validated ordered IDs without owning their lifetime or fitted state."""
    contiguous = np.ascontiguousarray(ids)
    key = (contiguous.dtype.str, contiguous.shape,
           hashlib.sha256(memoryview(contiguous).cast("B")).digest())
    with _IMMUTABLE_BINARY_ID_AXES_LOCK:
        existing = _IMMUTABLE_BINARY_ID_AXES.get(key)
        if existing is not None and np.array_equal(existing, contiguous):
            return existing
        # bytes backing makes readonly irreversible; a NumPy flag alone would
        # permit mutation of the shared identity axis via setflags(True).
        owned = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype)
        _IMMUTABLE_BINARY_ID_AXES[key] = owned
        return owned


class ImmutableFittedBinarySource:
    """Validate a receipt-bound FP64 bank entry once, then derive normal/SPA tiles.

    The constructor accepts the same verified array mapping as the compact
    helpers. File receipts must already have been authenticated by the caller;
    this object binds that admission to the common fitted-state SHA. It never
    accepts a boolean ``trusted`` shortcut. NumPy inputs are copied once so a
    mutable external array cannot silently change the admitted tensors. Existing
    Torch sources, including a shared GPU X, retain their original storage.
    Equal ordered ID axes share immutable bytes through weak ownership; the
    pool does not retain unused axes or share fitted model tensors.

    Source tensors must remain immutable while this object is alive. Each
    derivation checks storage, geometry and Torch mutation versions in constant
    time per source array. Full finite-value and sample-ID checks occur only at
    admission. Mutation through unsupported ``.data`` writes is prohibited by
    this ownership contract, as it bypasses Torch's own version mechanism.
    """
    __slots__ = ("_ids", "_tensors", "_stamps", "_binding", "_method",
                 "_has_kinship", "_iterations", "_provenance", "_sealed", "__weakref__")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("fitted binary source is immutable")
        object.__setattr__(self, name, value)

    @staticmethod
    def _owned_array(value):
        if value is None or isinstance(value, torch.Tensor):
            return value
        array = np.asarray(value)
        if array.dtype != np.float64:
            raise ValueError("immutable binary source must retain original FP64 arrays")
        # torch.tensor owns its CPU storage; np.asarray/as_tensor would leave
        # untracked writes through the caller's NumPy allocation possible.
        return torch.tensor(array)

    @staticmethod
    def _stamp(value):
        if value is None:
            return None
        try:
            version = value._version
        except RuntimeError as error:
            raise ValueError("immutable fitted source requires mutation-tracked tensors") from error
        base = (id(value), tuple(value.shape), value.dtype, value.device,
                value.layout, value.requires_grad, version)
        if value.layout == torch.strided:
            return base + (value.data_ptr(), tuple(value.stride()), value.storage_offset())
        indices, values = value.indices(), value.values()
        return base + (indices.data_ptr(), indices._version,
                       values.data_ptr(), values._version, value.is_coalesced())

    def __init__(self, *, sample_ids, covariates, residual, fitted_probability,
                 fixed_effect_covariance, precision, null_fit_source_sha256,
                 verified_fit_source_sha256, device="cpu", coefficients=None,
                 phenotype=None, has_kinship=False, iterations=0,
                 provenance="verified_fitted_binary_state"):
        arrays = {name: self._owned_array(value) for name, value in (
            ("covariates", covariates), ("residual", residual),
            ("fitted_probability", fitted_probability),
            ("fixed_effect_covariance", fixed_effect_covariance),
            ("precision", precision))}
        x, r, mu, cov, inverse, sx, binding, method = _compact_fitted_arrays(
            **arrays, null_fit_source_sha256=null_fit_source_sha256,
            verified_fit_source_sha256=verified_fit_source_sha256, device=device)
        if not tensor_all_finite(sx):
            raise ValueError("reconstructed binary fitted projection is not finite")
        del sx
        with explicit_binary_fitted_projection_fp64():
            left = x @ cov
        if not tensor_all_finite(left):
            raise ValueError("reconstructed binary SPA projection is not finite")
        del left
        ids = np.asarray(sample_ids, dtype=str)
        if ids.shape != (x.shape[0],) or len(np.unique(ids)) != len(ids) or np.any(ids == ""):
            raise ValueError("immutable binary sample IDs must be unique and aligned")
        # A bytes-backed array cannot be made writable via setflags(True).
        self._ids = _intern_immutable_binary_sample_ids(ids)
        beta, y = None, None
        if coefficients is not None:
            beta = self._owned_array(coefficients).to(device=device)
            if (beta.dtype != torch.float64 or beta.requires_grad
                    or beta.shape != (x.shape[1],) or not tensor_all_finite(beta)):
                raise ValueError("immutable binary coefficients must retain their FP64 fitted shape")
        if phenotype is not None:
            y = self._owned_array(phenotype).to(device=device)
            if (y.dtype != torch.float64 or y.requires_grad or y.shape != r.shape
                    or not tensor_all_finite(y) or bool(((y != 0) & (y != 1)).any())):
                raise ValueError("immutable binary labels must be aligned FP64 0/1")
        if type(iterations) is not int or iterations < 0:
            raise ValueError("immutable binary iteration count must be nonnegative")
        self._tensors = (x, r, mu, cov, inverse, beta, y)
        self._stamps = tuple(self._stamp(value) for value in self._tensors)
        self._binding, self._method = binding, method
        self._has_kinship, self._iterations = bool(has_kinship), iterations
        self._provenance = str(provenance)
        self._sealed = True

    @property
    def device(self):
        return self._tensors[0].device

    @property
    def sample_ids(self):
        return self._ids

    @property
    def null_fit_source_sha256(self):
        return self._binding

    def _projection(self):
        if tuple(self._stamp(value) for value in self._tensors) != self._stamps:
            raise ValueError("immutable binary fitted source was mutated")
        x, r, mu, cov, inverse, beta, y = self._tensors
        with explicit_binary_fitted_projection_fp64():
            sx = (inverse[:, None] * x if inverse.ndim == 1
                  else torch.sparse.mm(inverse, x))
        return x, r, mu, cov, inverse, sx, beta, y

    def normal(self, *, association_mode="tf32"):
        """Derive one temporary normal state from the admitted original FP64 fit."""
        mode = validate_mode(association_mode)
        x, r, mu, cov, inverse, sx, beta, _ = self._projection()
        dtype = torch.float32 if mode == "tf32" else torch.float64
        x, r, mu, cov, inverse, sx = (v.to(dtype=dtype)
                                     for v in (x, r, mu, cov, inverse, sx))
        beta = None if beta is None else beta.to(dtype=dtype)
        model = BinaryNullModel(self._ids, x, r, mu, sx.T, None, cov,
            precision=inverse, precision_x=sx, coefficients=beta,
            iterations=self._iterations, has_kinship=self._has_kinship,
            use_spa=False, fit_method=self._provenance, matmul_mode=mode)
        model.null_fit_source_sha256 = self._binding
        model.source_matmul_mode = "fp64"
        model.normal_projection_reconstructed = True
        model.normal_projection_method = self._method + "_cast_" + mode
        model.fitted_source_validation = "immutable_once_with_tensor_version_guard"
        return model

    def spa(self):
        """Derive an FP64 SPA projection only when selected tests need it."""
        x, r, mu, cov, inverse, sx, beta, y = self._projection()
        with explicit_binary_fitted_projection_fp64():
            left = x @ cov
        model = BinaryNullModel(self._ids, x, r, mu, sx.T, left, cov,
            precision=inverse, precision_x=sx, coefficients=beta, phenotype=y,
            iterations=self._iterations, has_kinship=self._has_kinship,
            use_spa=True, fit_method=self._provenance, matmul_mode="fp64")
        model.null_fit_source_sha256 = self._binding
        model.spa_projection_reconstructed = True
        model.spa_projection_method = self._method
        model.fitted_source_validation = "immutable_once_with_tensor_version_guard"
        return model


def _binomial_logit_state(eta):
    """R binomial(logit) means, derivative and working weights in FP64.

    R's logit link is numerically bounded outside [-30, 30], and its
    ``mu.eta`` is evaluated independently of ``mu * (1 - mu)``.  Using a
    plain sigmoid can round a finite positive predictor to exactly one,
    incorrectly turning a valid PQL update into a zero working weight.
    These are the stats logit-link boundary conventions, not an eta clip or
    a change to the mixed-model likelihood/variance-component equations.
    https://github.com/wch/r-source/blob/trunk/src/library/stats/src/family.c
    """
    if eta.dtype != torch.float64 or not bool(torch.isfinite(eta).all()):
        raise ArithmeticError("binomial logit predictor must be finite FP64")
    epsilon = torch.finfo(torch.float64).eps
    exponential = torch.exp(eta.clamp(min=-30., max=30.))
    mu = exponential / (1 + exponential)
    derivative = exponential / (1 + exponential).square()
    lower, upper = eta < -30., eta > 30.
    mu = torch.where(lower, epsilon / (1 + epsilon), mu)
    mu = torch.where(upper, (1 / epsilon) / (1 + 1 / epsilon), mu)
    derivative = torch.where(lower | upper, epsilon, derivative)
    weights = derivative.square() / (mu * (1 - mu))
    if (not bool(torch.isfinite(weights).all())
            or bool((weights <= 0).any())):
        raise ArithmeticError("binomial logit working weights are not finite positive")
    return mu, derivative, weights


def fit_logistic_null(phenotype,covariates=None,*,sample_ids=None,device="cuda",
                      use_spa=True,maxiter=100,tol=1e-8):
    """Float64 GPU binomial IRLS, with stable likelihood step halving."""
    y=torch.as_tensor(phenotype,dtype=torch.float64,device=device)
    if y.ndim!=1 or y.numel()<2 or not bool(torch.isfinite(y).all()) or bool(((y!=0)&(y!=1)).any()) or not 0<float(y.mean())<1:
        raise ValueError("logistic phenotype must contain both finite 0 and 1 outcomes")
    n=y.numel();x=torch.ones((n,1),dtype=y.dtype,device=y.device) if covariates is None else torch.as_tensor(covariates,dtype=y.dtype,device=y.device)
    if x.ndim!=2 or x.shape[0]!=n or not bool(torch.isfinite(x).all()) or int(torch.linalg.matrix_rank(x))!=x.shape[1]:
        raise ValueError("logistic design must be finite, aligned and full rank")
    if not 0<tol<1 or maxiter<2:
        raise ValueError("invalid logistic fitting tolerance or iteration limit")
    ids=np.asarray([str(j) for j in range(n)] if sample_ids is None else sample_ids,dtype=str)
    beta=torch.zeros(x.shape[1],dtype=y.dtype,device=y.device)
    # stats::binomial initialization for an unweighted 0/1 response.
    eta=torch.logit((y+.5)/2)
    boundary_peak = 0
    eta_abs_peak = float(eta.abs().max())
    def loss(z):return (torch.nn.functional.softplus(z)-y*z).sum()
    current_loss=loss(eta)
    for iteration in range(1,maxiter+1):
        mu,mu_eta,w=_binomial_logit_state(eta)
        z=eta+(y-mu)/mu_eta
        sqrtw=torch.sqrt(w)
        candidate=torch.linalg.lstsq(x*sqrtw[:,None],z*sqrtw,driver="gels").solution
        delta=candidate-beta
        for _ in range(64):
            candidate=beta+delta;next_eta=x@candidate;next_loss=loss(next_eta)
            if bool(torch.isfinite(next_loss)&(((iteration==1)|(next_loss<=current_loss+1e-10)))):
                break
            delta/=2
        else:raise ArithmeticError("logistic IRLS step halving did not find a finite likelihood improvement")
        change=float((next_loss-current_loss).abs()/(.05+next_loss.abs()))
        beta,eta,current_loss=candidate,next_eta,next_loss
        boundary_peak = max(boundary_peak, int((eta.abs()>30).sum()))
        eta_abs_peak = max(eta_abs_peak, float(eta.abs().max()))
        if change<tol:break
    else:raise ArithmeticError("logistic IRLS did not converge")
    mu,mu_eta,w=_binomial_logit_state(eta)
    signed_eta = (2*y-1)*eta
    if bool((signed_eta >= 0).all()) and bool((signed_eta > 0).any()):
        # The returned finite vector itself is a separating direction.  Link
        # boundaries prevent an arithmetic zero; they do not identify an MLE
        # when the unpenalized fixed-effects design admits separation.
        raise ArithmeticError("logistic fixed-effects design admits separation; finite initialization is not identifiable")
    cov=torch.linalg.inv(x.T@(w[:,None]*x))
    model=binary_prefitted_state(sample_ids=ids,covariates=x,residual=y-mu,fitted_probability=mu,
        xw=(x*w[:,None]).T,projection_left=x@cov,fixed_effect_covariance=cov,
        precision=w,precision_covariates=x*w[:,None],
        coefficients=beta,phenotype=y,device=device,use_spa=use_spa,provenance="ordinary_logistic_IRLS")
    model.iterations=iteration
    model.linear_predictors=eta
    model.binomial_link_boundary_count=int((eta.abs()>30).sum())
    model.binomial_link_method="R_stats_logit_linkinv_mu_eta"
    model.binomial_link_boundary_peak_count=boundary_peak
    model.binomial_link_eta_abs_peak=eta_abs_peak
    return model


_BINARY_FAMILY_BATCH_BYTES = 64 * 1024**2
_BINARY_FAMILY_BATCH_COUNT = 4096


def _binary_family_batch_width(device, remaining, per_block_bytes, *, context):
    """Bound independent-family batches by explicit new tensors and live CUDA.

    A single supported family may exceed the batching target; its actual
    bytes must still pass the same runtime CUDA cap/reserve guard. Library
    workspaces use that reserve, rather than an invented NNZ multiplier.
    """
    from ._fit_structure_cache import available_cuda_bytes, guard_cuda_workspace
    width = min(remaining, _BINARY_FAMILY_BATCH_COUNT,
                max(1, _BINARY_FAMILY_BATCH_BYTES // per_block_bytes))
    available = available_cuda_bytes(device)
    if available is not None:
        if available < per_block_bytes:
            guard_cuda_workspace(per_block_bytes, device, context=context)
        width = min(width, max(1, available // per_block_bytes))
    guard_cuda_workspace(width * per_block_bytes, device, context=context)
    return width


def _binary_family_apply(diagonal, blocks, batches, value):
    """Apply independent dense family matrices without a samples-square slab."""
    from ._fit_structure_cache import guard_cuda_workspace
    element_bytes = max(diagonal.element_size(), value.element_size())
    guard_cuda_workspace(value.numel() * element_bytes, value.device,
                         context="binary family apply output")
    result = diagonal.reshape((-1,) + (1,) * (value.ndim - 1)) * value
    # The fitting kernels pass vectors or matrices. Retain the original
    # higher-dimensional behavior for callers of this private operator.
    if value.ndim not in (1, 2):
        try:
            trailing = int(np.prod(value.shape[1:]))
            for rows, matrix in blocks:
                guard_cuda_workspace(2 * len(rows) * trailing * element_bytes,
                                     value.device, context="binary family apply fallback")
                result[rows] = matrix @ value[rows]
        except Exception:
            result = None
            raise
        return result
    columns = 1 if value.ndim == 1 else value.shape[1]
    if not columns:
        return result
    try:
        for _, rows, matrices in batches:
            size = rows.shape[1]
            begin = 0
            while begin < len(rows):
                width = _binary_family_batch_width(value.device, len(rows) - begin,
                    2 * size * columns * element_bytes, context="binary family apply batch")
                stop = begin + width
                selected = product = rhs = None
                try:
                    selected = value[rows[begin:stop]]
                    rhs = selected.unsqueeze(-1) if value.ndim == 1 else selected
                    product = torch.bmm(matrices[begin:stop], rhs)
                    result[rows[begin:stop]] = product.squeeze(-1) if value.ndim == 1 else product
                finally:
                    selected = product = rhs = None
                begin = stop
    except Exception:
        result = None
        raise
    return result


class _BlockKinship:
    """Validated PSD kinship, with no dense samples-by-samples allocation."""
    def __init__(self, diagonal, edge_rows, edge_cols, edge_values, *, device,
                 max_block_size, spectrum=None):
        from .null_model import KinshipSpectrum
        from ._fit_structure_cache import guard_cuda_workspace
        if spectrum is None:
            spectrum = KinshipSpectrum.from_sparse(
                diagonal, edge_rows, edge_cols, edge_values, device=device,
                max_block_size=max_block_size, dtype=torch.float64)
        self.diagonal = self.grouped = self.mean_diagonal = None
        self.blocks = [None] * len(spectrum.blocks)
        self._batches = []
        groups = {}
        for index, (rows, _) in enumerate(spectrum.blocks):
            groups.setdefault(len(rows), []).append(index)
        try:
            guard_cuda_workspace(len(diagonal) * 9, device, context="binary kinship vectors")
            self.diagonal = torch.tensor(np.asarray(diagonal), dtype=torch.float64, device=device)
            if (spectrum.eigenvalues.shape != self.diagonal.shape
                    or spectrum.eigenvalues.dtype != torch.float64
                    or spectrum.eigenvalues.device != self.diagonal.device):
                raise ValueError("binary shared spectrum must match the FP64 sample axis and device")
            self.grouped = torch.zeros(len(self.diagonal), dtype=torch.bool, device=device)
            for size, indices in groups.items():
                begin = 0
                while begin < len(indices):
                    # Rotation, scaled rotation and matrix; rows and values.
                    width = _binary_family_batch_width(self.diagonal.device,
                        len(indices) - begin, 3 * size * size * 8 + 2 * size * 8,
                        context="binary kinship reconstruction batch")
                    chosen = indices[begin:begin + width]
                    rows = rotations = values = matrix = None
                    try:
                        rows = torch.stack([spectrum.blocks[i][0] for i in chosen])
                        rotations = torch.stack([spectrum.blocks[i][1] for i in chosen])
                        values = spectrum.eigenvalues[rows]
                        matrix = torch.bmm(rotations * values[:, None, :], rotations.transpose(-2, -1))
                        rotations = values = None
                        matrix = (matrix + matrix.transpose(-2, -1)) / 2
                        self._batches.append((chosen, rows, matrix))
                        for offset, index in enumerate(chosen):
                            self.blocks[index] = (spectrum.blocks[index][0], matrix[offset])
                        self.grouped[rows] = True
                    finally:
                        rows = rotations = values = matrix = None
                    begin += width
            self.mean_diagonal = self.diagonal.mean()
            if not bool(torch.isfinite(self.mean_diagonal)) or float(self.mean_diagonal) <= 0:
                raise ValueError("mixed binary GRM must have positive mean diagonal")
        except Exception:
            self.blocks.clear()
            self._batches.clear()
            self.diagonal = self.grouped = self.mean_diagonal = None
            raise

    def apply(self, value):
        return _binary_family_apply(self.diagonal, self.blocks, self._batches, value)


class _BlockPrecision:
    """Inverse of diag(1/W)+tau*K, retaining only disjoint family blocks."""
    def __init__(self, kinship, weights, tau, *, compute_logdet=False):
        from ._fit_structure_cache import guard_cuda_workspace
        self.diagonal = self.logdet = self.kinship = None
        self.blocks, self._batches = [], []
        logterms = None
        try:
            element_bytes = max(kinship.diagonal.element_size(), weights.element_size())
            guard_cuda_workspace(3 * weights.numel() * element_bytes, weights.device,
                                 context="binary precision diagonal")
            self.diagonal = 1 / (1 / weights + tau * kinship.diagonal)
            self.blocks = [None] * len(kinship.blocks)
            self._batches = []
            self.kinship = kinship
            self.logdet = None
            logterms = [None] * len(kinship.blocks) if compute_logdet else None
            if compute_logdet:
                guard_cuda_workspace(weights.numel() * (2 * element_bytes + 1) + element_bytes,
                                     weights.device, context="binary precision singleton logdet")
                self.logdet = -torch.log(self.diagonal[~kinship.grouped]).sum()
            for indices, rows, matrices in kinship._batches:
                size = rows.shape[1]
                begin = 0
                while begin < len(indices):
                    # Covariance, factor, inverse; symmetrization has the same
                    # three-matrix peak after the first two are released.
                    matrix_bytes = size * size * element_bytes
                    per_block = max(3 * matrix_bytes + (element_bytes if compute_logdet else 0),
                                    matrix_bytes + 2 * size * element_bytes,
                                    2 * matrix_bytes + size * element_bytes + 2 * element_bytes
                                    if compute_logdet else 0)
                    width = _binary_family_batch_width(weights.device,
                        len(indices) - begin, per_block, context="binary precision factor batch")
                    stop = begin + width
                    covariance = factor = inverse = batch_log = None
                    try:
                        covariance = tau * matrices[begin:stop]
                        covariance.diagonal(dim1=-2, dim2=-1).add_(1 / weights[rows[begin:stop]])
                        # Keep the same Cholesky/inverse algorithms and hard
                        # failure semantics. One call checks this family batch.
                        factor = torch.linalg.cholesky(covariance)
                        if compute_logdet:
                            batch_log = 2 * torch.log(factor.diagonal(dim1=-2, dim2=-1)).sum(-1)
                        inverse = torch.cholesky_inverse(factor)
                        covariance = factor = None
                        inverse = (inverse + inverse.transpose(-2, -1)) / 2
                        chosen = indices[begin:stop]
                        self._batches.append((chosen, rows[begin:stop], inverse))
                        for offset, index in enumerate(chosen):
                            self.blocks[index] = (kinship.blocks[index][0], inverse[offset])
                            if compute_logdet:
                                logterms[index] = batch_log[offset]
                        self.diagonal[rows[begin:stop]] = inverse.diagonal(dim1=-2, dim2=-1)
                    finally:
                        covariance = factor = inverse = batch_log = None
                    begin = stop
            if compute_logdet:
                # Preserve the original family-order scalar accumulation.
                for term in logterms:
                    self.logdet = self.logdet + term
        except Exception:
            self.blocks.clear()
            self._batches.clear()
            self.diagonal = self.logdet = self.kinship = None
            raise
        finally:
            # A held constructor traceback must not extend caller-owned
            # geometry/weight lifetimes after a failed stage. Successful
            # precision objects keep the geometry through self.kinship.
            kinship = weights = tau = rows = matrices = term = None
            covariance = factor = inverse = batch_log = logterms = None

    def apply(self, value):
        return _binary_family_apply(self.diagonal, self.blocks, self._batches, value)

    def trace_kinship(self):
        value = (self.diagonal[~self.kinship.grouped]
                 * self.kinship.diagonal[~self.kinship.grouped]).sum()
        for (_, inverse), (_, matrix) in zip(self.blocks, self.kinship.blocks):
            value = value + (inverse * matrix.T).sum()
        return value

    def sparse_tensor(self):
        """COO N-by-N shape, O(N+sum family_size**2) stored entries."""
        if not self.blocks:
            return self.diagonal.clone()
        rows = torch.nonzero(~self.kinship.grouped, as_tuple=True)[0]
        indices = [torch.stack((rows, rows))]
        values = [self.diagonal[rows]]
        for family_rows, inverse in self.blocks:
            size = len(family_rows)
            indices.append(torch.stack((family_rows.repeat_interleave(size),
                                        family_rows.repeat(size))))
            values.append(inverse.reshape(-1))
        return torch.sparse_coo_tensor(
            torch.cat(indices, dim=1), torch.cat(values),
            (len(self.diagonal), len(self.diagonal)),
            device=self.diagonal.device).coalesce()


class _MixedAINonconvergence(ArithmeticError):
    """A failed AI trajectory, eligible for recovery with the same mixed model."""


def _block_binomial_reml(kinship, working, weights, x, tau):
    """Fixed-dispersion REML objective and GLS state for one PQL working fit.

    This is twice the negative restricted Gaussian log likelihood, up to
    constants. Binomial dispersion stays one: the quadratic term is not
    replaced by a profiled-dispersion ``(n-p)*log(quadratic)`` term.
    Only singleton vectors, independent family blocks, and a p-by-p system
    are allocated. The supplied PSD kinship is retained without the whitened
    eigenvalue truncation used by GMMAT's dense Brent implementation.

    Objective: GMMAT src/fitglmm.cpp:236-280,374-445.
    https://github.com/hanchenphd/GMMAT/blob/master/src/fitglmm.cpp
    """
    tau = torch.as_tensor(tau, dtype=working.dtype, device=working.device)
    inverse = _BlockPrecision(kinship, weights, tau, compute_logdet=True)
    sx = inverse.apply(x)
    gram = x.T @ sx
    factor = torch.linalg.cholesky((gram + gram.T) / 2)
    covariance = torch.cholesky_inverse(factor)
    coefficients = covariance @ (sx.T @ working)
    # Evaluate the residual quadratic directly, avoiding subtraction of two
    # large quadratics when the fixed effects explain much of working Y.
    residual = working - x @ coefficients
    py = inverse.apply(residual)
    objective = (inverse.logdet + 2 * torch.log(factor.diagonal()).sum()
                 + (residual * py).sum())
    if not bool(torch.isfinite(objective)):
        raise ArithmeticError("mixed logistic Brent REML objective is nonfinite")
    return objective, (inverse, sx, covariance, coefficients, py)


def _block_binomial_brent_step(kinship, working, weights, x, tol):
    """Optimize tau on GMMAT's ten log intervals, with dispersion fixed one."""
    from scipy.optimize import minimize_scalar
    import math

    # These are GMMAT's glmmkin defaults, not a case-count-dependent bound.
    # R/glmmkin.R:0,533 and src/fitglmm.cpp:424-438 search log(tau).
    lower, upper, regions = 1e-5, 1e5, 10
    left, width = math.log(lower), math.log(upper / lower) / regions

    def objective(log_tau):
        value, _ = _block_binomial_reml(
            kinship, working, weights, x, math.exp(log_tau))
        return float(value)

    best = None
    for region in range(regions):
        bounds = (left + region * width, left + (region + 1) * width)
        result = minimize_scalar(objective, bounds=bounds, method="bounded",
                                 options={"xatol": tol})
        if (not result.success or not math.isfinite(result.fun)
                or not math.isfinite(result.x)):
            raise ArithmeticError("mixed logistic Brent variance search did not converge")
        if best is None or result.fun < best.fun:
            best = result
    tau = torch.as_tensor(math.exp(best.x), dtype=working.dtype,
                          device=working.device)
    objective_value, fitted = _block_binomial_reml(
        kinship, working, weights, x, tau)
    return tau, objective_value, fitted


def _fit_logistic_block_brent(y, x, ordinary, kinship, *, maxiter, tol,
                              trace_callback=None):
    """Restart original fit0 and recover the same single-GRM PQL model.

    This follows GMMAT's AI-to-Brent wrapper and explicit zero-boundary refit
    (R/glmmkin.R:236-246,533-592). The optimizer changes; y, X, K and fixed
    binomial dispersion do not. Nonconvergence remains an error.
    """
    total_iterations = 0
    boundary_peak = ordinary.binomial_link_boundary_peak_count
    eta_abs_peak = ordinary.binomial_link_eta_abs_peak
    tau_peak = 1.
    fixed_boundary = False
    for refit in range(2):
        coefficients_previous = ordinary.coefficients.clone()
        eta = x @ coefficients_previous
        mu, mu_eta, weights = _binomial_logit_state(eta)
        working = eta + (y - mu) / mu_eta
        tau = torch.tensor(0. if fixed_boundary else 1., dtype=y.dtype,
                           device=y.device)
        converged = False
        for iteration in range(1, maxiter + 1):
            total_iterations += 1
            old_tau = tau.clone()
            old_working, old_weights = working, weights
            if fixed_boundary:
                objective, fitted = _block_binomial_reml(
                    kinship, working, weights, x, tau)
            else:
                tau, objective, fitted = _block_binomial_brent_step(
                    kinship, working, weights, x, tol)
            inverse, sx, covariance, coefficients, py = fitted
            eta = working - py / weights
            mu, mu_eta, weights = _binomial_logit_state(eta)
            working = eta + (y - mu) / mu_eta
            if not bool(torch.isfinite(working).all()):
                raise ArithmeticError("mixed logistic Brent working response is nonfinite")
            boundary_peak = max(boundary_peak, int((eta.abs() > 30).sum()))
            eta_abs_peak = max(eta_abs_peak, float(eta.abs().max()))
            tau_peak = max(tau_peak, float(tau))
            change = 2 * max(
                float(((coefficients - coefficients_previous).abs()
                       / (coefficients.abs() + coefficients_previous.abs() + tol)).max()),
                float((tau - old_tau).abs() / (tau.abs() + old_tau.abs() + tol)))
            if trace_callback is not None:
                trace_callback(dict(optimizer="Brent", refit=refit,
                    iteration=iteration, tau_old=old_tau, tau=tau.clone(),
                    Y_old=old_working, Y=working, weights_old=old_weights,
                    weights=weights, alpha=coefficients, cov=covariance, PY=py,
                    objective=objective, eta=eta, change=change,
                    link_boundary_count=int((eta.abs() > 30).sum())))
            coefficients_previous = coefficients
            if change < tol and iteration < maxiter:
                converged = True
                break
        if not converged:
            raise ArithmeticError("mixed logistic Brent PQL did not converge; no model fallback is used")
        if not fixed_boundary and float(tau) < 1.01 * tol:
            fixed_boundary = True
            continue
        return dict(inverse=inverse, sx=sx, covariance=covariance,
            coefficients=coefficients, py=py, eta=eta, mu=mu, weights=weights,
            working=working, tau=tau, old_tau=tau.clone(),
            total_iterations=total_iterations, boundary_peak=boundary_peak,
            eta_abs_peak=eta_abs_peak, tau_peak=tau_peak,
            fixed_boundary=fixed_boundary, converged=True)
    raise ArithmeticError("mixed logistic Brent boundary refit did not converge")


def fit_logistic_mixed_null(phenotype, covariates=None, *, sample_ids=None,
                            kinship_diagonal, edge_rows=(), edge_cols=(),
                            edge_values=(), device="cpu", use_spa=True,
                            maxiter=500, tol=1e-5, max_block_size=2048,
                            trace_callback=None, fit_structure_cache=None):
    """Fit one logistic mixed null using GMMAT's PQL/AI REML equations.

    The aligned PSD GRM is diagonal plus disjoint small family blocks. The
    binomial dispersion is fixed at one; the single random-effect variance is
    fitted by one EM initialization and average-information updates. Negative
    updates are step-halved and a zero-variance boundary is refitted explicitly.
    An AI trajectory that does not converge is restarted from the original
    fixed-effect fit using bounded Brent REML optimization of the same GRM,
    following GMMAT's single-GRM recovery. Failed recovery or separation raises.
    The recovery uses exact family-block log determinants, without a dense
    N-by-N matrix or GMMAT Brent's whitened-eigenvalue truncation.

    Fitting and the SPA projection state use FP64. The returned precision is a
    diagonal vector or sparse COO, never a dense N-by-N matrix. Genotype products
    may subsequently use the independent non-SPA TF32 state. Equations and the
    finite-tolerance return convention follow GMMAT R/glmmkin.R:278-418,655-702,
    https://github.com/hanchenphd/GMMAT/blob/master/R/glmmkin.R.

    ``fit_structure_cache`` may lease a verified immutable GRM spectrum across
    exactly matching ordered samples. Working weights, precision, coefficients
    and variance components remain private to this fit.
    """
    if not 0 < tol < 1 or type(maxiter) is not int or maxiter < 2:
        raise ValueError("invalid mixed logistic tolerance or iteration limit")
    if type(max_block_size) is not int or max_block_size < 1:
        raise ValueError("max_block_size must be a positive integer")
    ordinary = fit_logistic_null(phenotype, covariates, sample_ids=sample_ids,
                                device=device, use_spa=use_spa,
                                maxiter=maxiter, tol=min(tol, 1e-8))
    y, x, ids = ordinary.phenotype, ordinary.x, ordinary.sample_ids
    if len(kinship_diagonal) != len(y):
        raise ValueError("kinship diagonal does not match the aligned phenotype")
    from contextlib import nullcontext
    lease = (nullcontext(None) if fit_structure_cache is None else
             fit_structure_cache.acquire_spectrum(ids, kinship_diagonal,
                 edge_rows, edge_cols, edge_values, device=device,
                 dtype=torch.float64, max_block_size=max_block_size))
    with lease as spectrum:
        return _fit_logistic_mixed_from_ordinary(ordinary,
            kinship_diagonal=kinship_diagonal, edge_rows=edge_rows,
            edge_cols=edge_cols, edge_values=edge_values, device=device,
            use_spa=use_spa, maxiter=maxiter, tol=tol,
            max_block_size=max_block_size, trace_callback=trace_callback,
            spectrum=spectrum)


def _fit_logistic_mixed_from_ordinary(ordinary, *, kinship_diagonal, edge_rows,
                                    edge_cols, edge_values, device, use_spa,
                                    maxiter, tol, max_block_size, trace_callback,
                                    spectrum):
    """Original per-trait PQL/AI/Brent state within the shared-structure lease."""
    y, x, ids = ordinary.phenotype, ordinary.x, ordinary.sample_ids
    kinship = _BlockKinship(kinship_diagonal, edge_rows, edge_cols, edge_values,
                           device=device, max_block_size=max_block_size, spectrum=spectrum)
    n = len(y)

    def state(working, weights, tau):
        inverse = _BlockPrecision(kinship, weights, tau)
        sx = inverse.apply(x)
        gram = x.T @ sx
        covariance = torch.cholesky_inverse(torch.linalg.cholesky((gram + gram.T) / 2))
        coefficients = covariance @ (sx.T @ working)
        py = inverse.apply(working) - sx @ coefficients
        sx_cov = sx @ covariance
        return inverse, sx, covariance, coefficients, py, sx_cov

    def derivative(inverse, sx, py, sx_cov, working):
        apy = kinship.apply(py)
        papy = inverse.apply(apy) - sx @ (sx_cov.T @ apy)
        trace_pk = inverse.trace_kinship() - (sx * kinship.apply(sx_cov)).sum()
        score = (working * papy).sum() - trace_pk
        information = (py * kinship.apply(papy)).sum()
        return score, information

    total_iterations = 0
    boundary_peak = ordinary.binomial_link_boundary_peak_count
    eta_abs_peak = ordinary.binomial_link_eta_abs_peak
    tau_peak = 0.
    fixed_boundary = False
    converged = False
    mixed_optimizer = "AI"
    recovery_reason = None
    brent_iterations = 0
    try:
        for refit in range(2):
            coefficients_previous = ordinary.coefficients.clone()
            eta = x @ coefficients_previous
            mu, mu_eta, weights = _binomial_logit_state(eta)
            working = eta + (y - mu) / mu_eta
            tau = working.var(correction=1) / (2 * kinship.mean_diagonal)
            tau_peak = max(tau_peak, float(tau))
            if fixed_boundary:
                tau = torch.zeros_like(tau)
            else:
                inverse, sx, covariance, coefficients, py, sx_cov = state(working, weights, tau)
                score, _ = derivative(inverse, sx, py, sx_cov, working)
                tau = torch.clamp(tau + tau.square() * score / n, min=0)
            for iteration in range(1, maxiter + 1):
                total_iterations += 1
                old_tau = tau.clone()
                old_working, old_weights = working, weights
                inverse, sx, covariance, coefficients, py, sx_cov = state(working, weights, old_tau)
                score, information = derivative(inverse, sx, py, sx_cov, working)
                if not fixed_boundary:
                    if not bool(torch.isfinite(information)) or float(information) <= 0:
                        raise _MixedAINonconvergence("AI information is nonpositive")
                    delta = score / information
                    for _ in range(2048):
                        tau = old_tau + delta
                        if float(tau) < tol and float(old_tau) < tol:
                            tau = torch.zeros_like(tau)
                        if float(tau) >= 0:
                            break
                        delta = delta / 2
                    else:
                        raise _MixedAINonconvergence("AI step left the variance parameter space")
                    if float(tau) < tol:
                        tau = torch.zeros_like(tau)
                eta = working - py / weights
                mu, mu_eta, weights = _binomial_logit_state(eta)
                boundary_peak = max(boundary_peak, int((eta.abs() > 30).sum()))
                eta_abs_peak = max(eta_abs_peak, float(eta.abs().max()))
                tau_peak = max(tau_peak, float(tau))
                working = eta + (y - mu) / mu_eta
                if not bool(torch.isfinite(working).all()):
                    raise _MixedAINonconvergence("AI working response is nonfinite")
                change = 2 * max(
                    float(((coefficients - coefficients_previous).abs()
                           / (coefficients.abs() + coefficients_previous.abs() + tol)).max()),
                    float((tau - old_tau).abs() / (tau.abs() + old_tau.abs() + tol)))
                if trace_callback is not None:
                    trace_callback(dict(refit=refit, iteration=iteration,
                        tau_old=old_tau, tau=tau.clone(), Y_old=old_working,
                        Y=working, weights_old=old_weights, weights=weights,
                        alpha=coefficients, cov=covariance, PY=py,
                        score=score, AI=information, eta=eta, change=change,
                        link_boundary_count=int((eta.abs() > 30).sum())))
                coefficients_previous = coefficients
                if change < tol and iteration < maxiter:
                    converged = True
                    break
                if not bool(torch.isfinite(tau)) or float(tau) > tol ** -2:
                    break
            if not converged:
                raise _MixedAINonconvergence("AI REML did not converge")
            if not fixed_boundary and float(tau) < 1.01 * tol:
                fixed_boundary = True
                converged = False
                continue
            break
    except _MixedAINonconvergence as error:
        recovery_reason = str(error)
        ai_iterations = total_iterations
        recovered = _fit_logistic_block_brent(
            y, x, ordinary, kinship, maxiter=maxiter, tol=tol,
            trace_callback=trace_callback)
        inverse, sx = recovered['inverse'], recovered['sx']
        covariance, coefficients = recovered['covariance'], recovered['coefficients']
        eta, mu, weights = recovered['eta'], recovered['mu'], recovered['weights']
        working, tau, old_tau = recovered['working'], recovered['tau'], recovered['old_tau']
        fixed_boundary, converged = recovered['fixed_boundary'], recovered['converged']
        boundary_peak = max(boundary_peak, recovered['boundary_peak'])
        eta_abs_peak = max(eta_abs_peak, recovered['eta_abs_peak'])
        tau_peak = max(tau_peak, recovered['tau_peak'])
        brent_iterations = recovered['total_iterations']
        total_iterations += brent_iterations
        mixed_optimizer = "Brent"
    else:
        ai_iterations = total_iterations
    # GMMAT returns Sigma_i/cov from its final pre-update state, while mu and
    # residual use the updated predictor. Keep that finite-tolerance convention.
    model = binary_prefitted_state(
        sample_ids=ids, covariates=x, residual=y-mu, fitted_probability=mu,
        xw=sx.T, projection_left=x @ covariance,
        fixed_effect_covariance=covariance, precision=inverse.sparse_tensor(),
        precision_covariates=sx, coefficients=coefficients, phenotype=y,
        has_kinship=True, use_spa=use_spa, device=device,
        provenance=("logistic_block_sparse_GMMAT_PQL_AI_REML" if mixed_optimizer == "AI"
                    else "logistic_block_sparse_GMMAT_PQL_Brent_REML"),
        matmul_mode="fp64", working_phenotype=working)
    model.iterations = total_iterations
    model.converged = converged
    model.theta = torch.stack((torch.ones_like(tau), tau))
    model.precision_theta = torch.stack((torch.ones_like(old_tau), old_tau))
    model.linear_predictors = eta
    model.boundary_refit = fixed_boundary
    model.binomial_link_boundary_count = int((eta.abs() > 30).sum())
    model.binomial_link_method = "R_stats_logit_linkinv_mu_eta"
    model.binomial_link_boundary_peak_count = boundary_peak
    model.binomial_link_eta_abs_peak = eta_abs_peak
    model.variance_component_peak = tau_peak
    model.kinship_block_sizes = tuple(len(rows) for rows, _ in kinship.blocks)
    model.mixed_optimizer = mixed_optimizer
    model.mixed_ai_recovery_reason = recovery_reason
    model.mixed_ai_iterations = ai_iterations
    model.mixed_brent_iterations = brent_iterations
    model.mixed_optimizer_search_bounds = (1e-5, 1e5) if mixed_optimizer == "Brent" else None
    model.mixed_optimizer_search_regions = 10 if mixed_optimizer == "Brent" else None
    return model
