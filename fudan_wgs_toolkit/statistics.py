"""Weighted score tests with explicit matrix precision.

SPDX-License-Identifier: GPL-3.0-only
Algorithm attribution is recorded in NOTICE.md.
Preserves the original saddlepoint and fourth-moment fallback choices."""

from __future__ import annotations

import copy
import math
import warnings
from typing import Sequence

import torch

from .tf32 import matmul, validate_mode
from .tensor_validation import tensor_all_finite
from ._native_tail_groups import PendingCompleteTail

from ._precision_eigen import record_gpu_eigen_route, record_ordered_tail


_STATISTICS_METADATA = {
    "matmul_mode_calls": {},
    "burden_matrix_products": 0,
    "rare_burden_matrix_products": 0,
    "weight_batch_optimization_calls": 0,
    "matrix_product_backend": "explicit tf32.matmul mode; no FP64 fallback",
    "scalar_reduction_dtype": "float64",
    "eigenvalue_dtype": "float64",
    "tail_dtype": "float64",
    "weight_transform_dtype": "mode dependent: device FP32 native, CPU FP64 explicit control",
    "native_batch_calls": 0,
    "native_complete_spectra": 0,
    "native_spectra_reused": 0,
    "native_relation_row_batches": 0,
    "native_relation_bulk_d2h_calls": 0,
    "native_relation_max_scratch_bytes": 0,
    "probability_output_d2h_calls": 0,
    "probability_output_d2h_values": 0,
    "cct_validation_flag_batches": 0,
    "probability_output_transfer_scope": "final elementary and combined P only; excludes validation, spectrum, root and cMAC synchronization",
    "native_eigen_core_dtype": "float32",
    "native_spectrum_reuse": "exact FP64 cross-products of represented FP32 weights; no tolerance",
    "native_relation_transfer_scope": "one complete Boolean relation table per CUDA test; excludes validation and probability transfers",
    "native_tail_batch": "original Saddle/moment/CCT branches, FP64 probability arithmetic",
    "tail_method": "saddlepoint_gamma_or_fastskat_hybrid",
    "exact_spectrum_threshold": None,
    "spectrum_policy": "full",
    "davies_calls": 0,
    "davies_faults": 0,
    "davies_retry_calls": 0,
    "davies_retry_faults": 0,
    "davies_kuonen_fallback_calls": 0,
    "davies_kuonen_fallback_failures": 0,
    "davies_psd_clipped_calls": 0,
    "davies_psd_clip_max_relative": 0.0,
    "liu_approximate_calls": 0,
    "liu_approximate_failures": 0,
    "liu_failure_records": [],
    "liu_lobpcg_max_residual": 0.0,
    "liu_lobpcg_max_orthogonality": 0.0,
    "liu_trace_max_relative_se": 0.0,
    "fastskat_approximate_calls": 0,
    "fastskat_approximate_failures": 0,
    "fastskat_lobpcg_max_residual": 0.0,
    "fastskat_lobpcg_max_orthogonality": 0.0,
    "fastskat_subspace_residual_limit": 0.25,
    "fastskat_residual_trace_method": "exact_dense",
    "fastskat_failure_records": [],
    "tail_metadata": [],
}


def statistics_execution_metadata(*, reset=False):
    """Actual WGS precision boundaries, without data or trait identifiers."""
    result = copy.deepcopy(_STATISTICS_METADATA)
    if reset:
        _STATISTICS_METADATA["matmul_mode_calls"].clear()
        _STATISTICS_METADATA["burden_matrix_products"] = 0
        _STATISTICS_METADATA["rare_burden_matrix_products"] = 0
        _STATISTICS_METADATA["weight_batch_optimization_calls"] = 0
        for key in ("native_batch_calls", "native_complete_spectra", "native_spectra_reused",
                    "native_relation_row_batches", "native_relation_bulk_d2h_calls", "native_relation_max_scratch_bytes",
                    "probability_output_d2h_calls", "probability_output_d2h_values", "cct_validation_flag_batches"):
            _STATISTICS_METADATA[key] = 0
        _STATISTICS_METADATA["davies_calls"] = 0
        _STATISTICS_METADATA["davies_faults"] = 0
        _STATISTICS_METADATA["davies_retry_calls"] = 0
        _STATISTICS_METADATA["davies_retry_faults"] = 0
        _STATISTICS_METADATA["davies_kuonen_fallback_calls"] = 0
        _STATISTICS_METADATA["davies_kuonen_fallback_failures"] = 0
        _STATISTICS_METADATA["davies_psd_clipped_calls"] = 0
        _STATISTICS_METADATA["davies_psd_clip_max_relative"] = 0.0
        _STATISTICS_METADATA["liu_approximate_calls"] = 0
        _STATISTICS_METADATA["liu_approximate_failures"] = 0
        _STATISTICS_METADATA["liu_failure_records"] = []
        _STATISTICS_METADATA["liu_lobpcg_max_residual"] = 0.0
        _STATISTICS_METADATA["liu_lobpcg_max_orthogonality"] = 0.0
        _STATISTICS_METADATA["liu_trace_max_relative_se"] = 0.0
        _STATISTICS_METADATA["fastskat_approximate_calls"] = 0
        _STATISTICS_METADATA["fastskat_approximate_failures"] = 0
        _STATISTICS_METADATA["fastskat_lobpcg_max_residual"] = 0.0
        _STATISTICS_METADATA["fastskat_lobpcg_max_orthogonality"] = 0.0
        _STATISTICS_METADATA["fastskat_subspace_residual_limit"] = 0.25
        _STATISTICS_METADATA["fastskat_residual_trace_method"] = "exact_dense"
        _STATISTICS_METADATA["fastskat_failure_records"] = []
        _STATISTICS_METADATA["tail_metadata"] = []
    return result


class DegenerateTestError(ValueError):
    """The requested test has no positive variance or valid weight mass."""


def _double(value, *, device=None) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def _finite(value: torch.Tensor, name: str) -> None:
    # A samples-by-variants CUDA matrix can occupy several GiB. The mature
    # blocked validator preserves its strides and checks every value without
    # materializing another full abs/Boolean matrix.
    if not tensor_all_finite(value):
        raise ValueError(f"{name} must contain only finite values")


def score_covariance(
    genotype,
    residual,
    *,
    covariates=None,
    working_weights=None,
    dispersion: float = 1.0,
    projector=None,
    precision=None,
    precision_covariates=None,
    fixed_effect_covariance=None,
    matmul_mode: str = "fp64",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute U=G' r and V=G' P G with mode-selected core storage.

    Native tf32 uses FP32 storage, solves and reductions; fp64 is an explicit control.
    Sparse precision/projector products are supported only in fp64 mode.

    Supply exactly one of (a) covariates with optional working_weights,
    (b) an already fitted projector P, or (c) precision=Sigma_i,
    precision_covariates=Sigma_iX, fixed_effect_covariance=cov.
    The latter two accept sparse COO/CSR matrices. For ordinary Gaussian
    models, dispersion is sigma**2 and residual is the unscaled residual.
    For mixed models, residual must already be the fitted scaled residual;
    dispersion must be 1 because P/Sigma_i include the fitted scale.
    No null-model fitting, sample alignment, imputation or allele flipping
    takes place here.
    """
    validate_mode(matmul_mode)
    def mm(left, right):
        if matmul_mode != "fp64" and (left.layout != torch.strided or right.layout != torch.strided):
            raise ValueError("TF32 statistics require dense matrix operands; sparse precision/projector is unsupported")
        return matmul(left, right, mode=matmul_mode)
    core_dtype = torch.float32 if matmul_mode == "tf32" else torch.float64
    def core(value, **kwargs):
        return torch.as_tensor(value, dtype=core_dtype, **kwargs)
    g = core(genotype)
    if g.ndim != 2 or g.shape[0] < 1 or g.shape[1] < 1:
        raise ValueError("genotype must be a nonempty samples-by-variants matrix")
    if g.layout != torch.strided:
        g = g.to_dense()
    r = core(residual, device=g.device)
    if r.ndim != 1 or r.shape[0] != g.shape[0]:
        raise ValueError("residual must have one value per genotype row")
    _finite(g, "genotype")
    _finite(r, "residual")
    if not math.isfinite(dispersion) or dispersion <= 0:
        raise ValueError("dispersion must be positive and finite")
    modes = sum((covariates is not None, projector is not None, precision is not None))
    if modes != 1:
        raise ValueError("supply exactly one of covariates, projector, precision")
    if projector is not None:
        if dispersion != 1 or working_weights is not None:
            raise ValueError("an explicit projector already contains fitted scaling")
        p = core(projector, device=g.device)
        if p.shape != (g.shape[0], g.shape[0]):
            raise ValueError("projector must have shape samples-by-samples")
        projected_g = mm(p, g)
        covariance = mm(projected_g.T, g)
    elif precision is not None:
        if dispersion != 1 or working_weights is not None:
            raise ValueError("precision already contains fitted scaling")
        if precision_covariates is None or fixed_effect_covariance is None:
            raise ValueError("precision also requires precision_covariates and fixed_effect_covariance")
        inverse_sigma = core(precision, device=g.device)
        inverse_sigma_x = core(precision_covariates, device=g.device)
        cov = core(fixed_effect_covariance, device=g.device)
        if inverse_sigma.shape != (g.shape[0], g.shape[0]):
            raise ValueError("precision must have shape samples-by-samples")
        if inverse_sigma_x.ndim != 2 or inverse_sigma_x.shape[0] != g.shape[0]:
            raise ValueError("precision_covariates must have one row per sample")
        if cov.shape != (inverse_sigma_x.shape[1], inverse_sigma_x.shape[1]):
            raise ValueError("fixed_effect_covariance has an incompatible shape")
        cross = mm(inverse_sigma_x.T, g)
        covariance = mm(mm(inverse_sigma, g).T, g) - mm(mm(cross.T, cov), cross)
    else:
        x = core(covariates, device=g.device)
        if x.ndim != 2 or x.shape[0] != g.shape[0]:
            raise ValueError("covariates must have one row per genotype sample")
        _finite(x, "covariates")
        v = torch.ones(g.shape[0], dtype=g.dtype, device=g.device)
        if working_weights is not None:
            v = core(working_weights, device=g.device)
            if v.shape != (g.shape[0],):
                raise ValueError("working_weights must have one value per sample")
            _finite(v, "working_weights")
            if bool((v < 0).any()):
                raise ValueError("working_weights cannot be negative")
        wg = v[:, None] * g
        cross = mm(x.T, wg)
        if x.shape[1]:
            gram = mm(x.T, v[:, None] * x)
            covariance = mm(g.T, wg) - mm(cross.T, torch.linalg.solve(gram, cross))
        else:
            covariance = mm(g.T, wg)
        covariance = dispersion * covariance
    score = mm(g.T, r)
    _finite(covariance, "score covariance")
    return score, (covariance + covariance.T) * 0.5


def _cauchy_sf_tensor(statistic: torch.Tensor) -> torch.Tensor:
    reciprocal = torch.atan(1 / statistic) / math.pi
    far = torch.where(statistic > 0, reciprocal, 1 + reciprocal)
    return torch.where(statistic.abs() > 1, far, 0.5 - torch.atan(statistic) / math.pi)


def _cct_tensor(pvalues, weights=None, *, internal: bool = False, sync_light: bool = False, _normal_transform=None) -> torch.Tensor:
    p = _double(pvalues).reshape(-1)
    if sync_light:
        from ._statistics_sync import host_flags
        if not p.numel():
            raise ValueError("pvalues must be a nonempty finite vector in [0, 1]")
        finite, out_of_range, has_zero, has_one = host_flags(
            torch.isfinite(p).all(), ((p < 0) | (p > 1)).any(), (p == 0).any(), (p == 1).any())
        if not finite or out_of_range:
            raise ValueError("pvalues must be a nonempty finite vector in [0, 1]")
        if not internal:
            if has_zero and has_one:
                raise ValueError("cannot combine both exact 0 and exact 1 pvalues")
            if has_zero:
                return p.new_tensor(0.)
            if has_one:
                warnings.warn("there are pvalues that are exactly 1", RuntimeWarning, stacklevel=3)
                return p.new_tensor(1.)
    else:
        if not p.numel() or not bool(torch.isfinite(p).all()) or bool(((p < 0) | (p > 1)).any()):
            raise ValueError("pvalues must be a nonempty finite vector in [0, 1]")
        if not internal:
            if bool((p == 0).any()) and bool((p == 1).any()):
                raise ValueError("cannot combine both exact 0 and exact 1 pvalues")
            if bool((p == 0).any()):
                return p.new_tensor(0.)
            if bool((p == 1).any()):
                warnings.warn("there are pvalues that are exactly 1", RuntimeWarning, stacklevel=3)
                return p.new_tensor(1.)
    if weights is None:
        w = torch.full_like(p, 1 / p.numel())
    else:
        w = _double(weights, device=p.device).reshape(-1)
        if sync_light:
            if w.shape != p.shape:
                raise ValueError("weights must be finite, nonnegative and match pvalues")
            mass = w.sum()
            finite, negative, mass_finite, mass_nonpositive = host_flags(
                torch.isfinite(w).all(), (w < 0).any(), torch.isfinite(mass), mass <= 0)
            if not finite or negative:
                raise ValueError("weights must be finite, nonnegative and match pvalues")
            if not mass_finite or mass_nonpositive:
                raise DegenerateTestError("CCT requires a positive finite weight sum")
        else:
            if w.shape != p.shape or not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
                raise ValueError("weights must be finite, nonnegative and match pvalues")
            mass = w.sum()
            if not bool(torch.isfinite(mass)) or bool(mass <= 0):
                raise DegenerateTestError("CCT requires a positive finite weight sum")
        w = w / mass
    small = p < 1e-16
    statistic = (w[small] / p[small] / math.pi).sum()
    normal_transform = (torch.tan((0.5 - p[~small]) * math.pi)
                        if _normal_transform is None else _normal_transform[~small])
    statistic = statistic + (w[~small] * normal_transform).sum()
    if sync_light:
        is_nan, large, nonpositive = host_flags(torch.isnan(statistic), statistic > 1e15, statistic <= 0)
        if is_nan:
            raise DegenerateTestError("CCT statistic is undefined")
        if large:
            return (1 / statistic) / math.pi
        tail = _cauchy_sf_tensor(statistic)
        return tail if internal or nonpositive else 1 - (1 - tail)
    if bool(torch.isnan(statistic)):
        raise DegenerateTestError("CCT statistic is undefined")
    if bool(statistic > 1e15):
        return (1 / statistic) / math.pi
    tail = _cauchy_sf_tensor(statistic)
    return tail if internal or bool(statistic <= 0) else 1 - (1 - tail)


def cct(pvalues, weights=None, *, internal: bool = False, sync_light: bool = False) -> float:
    """Cauchy combination on the input device, with R/C++ boundary rules.

    Default matches exported R CCT: any exact 0 returns 0, any exact 1
    returns 1 with a warning, and simultaneous 0/1 is an error (even for
    zero-weight elements). internal=True matches CCT_pval.cpp used inside
    ACAT-V. Invalid or zero weight mass raises an error instead of R NaN.
    """
    return float(_cct_tensor(pvalues, weights, internal=internal, sync_light=sync_light))


def _ordered_sum(values: torch.Tensor) -> torch.Tensor:
    """Original scalar-addition order for CUDA rows; CPU retains Torch sum."""
    if values.is_cuda:
        from ._ordered_cuda import ordered_rows_cuda
        return ordered_rows_cuda(values.reshape(values.shape[0], -1)).reshape(values.shape[1:])
    return values.sum(dim=0)


def _quadratic_form_sf_tensor(statistic, eigenvalues, *, moment_eigenvalues=None, reference_reduction=False, sync_light=False) -> torch.Tensor:
    raw = _double(eigenvalues).reshape(-1)
    q_raw = _double(statistic, device=raw.device)
    if not raw.numel() or not bool(torch.isfinite(raw).all()) or q_raw.numel() != 1 or not bool(torch.isfinite(q_raw)) or bool(q_raw < 0):
        raise ValueError("quadratic form requires a nonnegative finite statistic and finite spectrum")
    spectrum = torch.where(raw < 1e-8, 0., raw)
    maximum = spectrum.max()
    if bool(maximum <= 0):
        raise DegenerateTestError("SKAT covariance has no eigenvalue at or above 1e-8")
    if bool(q_raw == 0):
        return raw.new_tensor(1.)
    if reference_reduction and raw.is_cuda:
        record_ordered_tail()
    reduce = _ordered_sum if reference_reduction else lambda values: values.sum()
    scaled, q = spectrum / maximum, q_raw / maximum
    if sync_light:
        from ._statistics_sync import bisection_root
        root = bisection_root(scaled, q, reduce)
    else:
        lower = q.new_tensor(-0.01) if bool(q > reduce(scaled)) else -torch.full_like(q, scaled.numel()) / (2 * q)
        upper, root = q.new_tensor(0.499995), q.new_tensor(0.)
        for _ in range(2048):
            if bool((upper - lower).abs() <= 1e-8):
                break
            root = (upper + lower) / 2
            derivative = reduce(scaled / (1 - 2 * scaled * root)) - q
            if bool(derivative == 0):
                break
            upper = torch.where(derivative > 0, root, upper)
            lower = torch.where(derivative > 0, lower, root)
        else:
            raise ArithmeticError("WGS saddlepoint bisection did not converge")
    if bool(root.abs() < 1e-4):
        moments = raw if moment_eigenvalues is None else _double(moment_eigenvalues, device=raw.device).reshape(-1)
        _finite(moments, "moment_eigenvalues")
        c1, c2, c4 = moments.sum(), moments.square().sum(), moments.pow(4).sum()
        if bool(c2 <= 0) or bool(c4 <= 0):
            raise DegenerateTestError("SKAT moment fallback has zero variance")
        dof = c2.square() / c4
        adjusted = (q_raw - c1) / torch.sqrt(2 * c2) * torch.sqrt(2 * dof) + dof
        return raw.new_tensor(1.) if bool(adjusted <= 0) else torch.special.gammaincc(dof / 2, adjusted / 2)
    cumulant = -0.5 * reduce(torch.log(1 - 2 * scaled * root))
    w2 = 2 * (root * q - cumulant)
    if bool(w2 <= 0):
        raise ArithmeticError("WGS saddlepoint has an invalid signed root")
    signed_root = torch.copysign(torch.sqrt(w2), root)
    second_derivative = 2 * reduce(scaled.square() / (1 - 2 * scaled * root).square())
    v = root * torch.sqrt(second_derivative)
    z = signed_root + torch.log(v / signed_root) / signed_root
    return 0.5 * torch.erfc(z / math.sqrt(2))


def quadratic_form_sf(statistic: float, eigenvalues, *, moment_eigenvalues=None) -> float:
    """WGS Saddle tail and fourth-moment fallback in float64 PyTorch.

    The absolute eigenvalue cutoff is 1e-8, the bisection tolerance 1e-8,
    and the moment switch abs(root)<1e-4, matching the original package.
    The complete unthresholded spectrum supplies the fallback moments.
    """
    return float(_quadratic_form_sf_tensor(statistic, eigenvalues, moment_eigenvalues=moment_eigenvalues))


def _beta_continued_fraction(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Modified Lentz evaluation of the incomplete-beta continued fraction."""
    tiny = 1e-300
    def nonzero(value):
        return torch.where(value.abs() < tiny, torch.copysign(torch.full_like(value, tiny), value), value)
    qab, qap, qam = a + b, a + 1, a - 1
    c = torch.ones_like(x)
    d = 1 / nonzero(1 - qab * x / qap)
    h = d
    for m in range(1, 10001):
        aa = m * (b - m) * x / ((qam + 2 * m) * (a + 2 * m))
        d, c = 1 / nonzero(1 + aa * d), nonzero(1 + aa / c)
        h = h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + 2 * m) * (qap + 2 * m))
        d, c = 1 / nonzero(1 + aa * d), nonzero(1 + aa / c)
        delta = d * c
        h = h * delta
        if bool(((delta - 1).abs() <= 8 * torch.finfo(x.dtype).eps).all()):
            return h
    raise ArithmeticError("Student t incomplete-beta continued fraction did not converge")


def _student_t_two_sided(t_squared: torch.Tensor, dof: int) -> torch.Tensor:
    """P(|T_dof| >= sqrt(t_squared)) via the regularized incomplete beta."""
    x = dof / (dof + t_squared)
    complement = t_squared / (dof + t_squared)
    result = torch.ones_like(x)
    interior = (x > 0) & (x < 1)
    z = x[interior]
    if z.numel():
        a, b = z.new_tensor(dof / 2), z.new_tensor(0.5)
        flip = z > (a + 1) / (a + b + 2)
        aa, bb = torch.where(flip, b, a), torch.where(flip, a, b)
        # Compute the complement directly: 1-x loses small t^2/dof digits.
        xx = torch.where(flip, complement[interior], z)
        leading = torch.exp(torch.lgamma(aa + bb) - torch.lgamma(aa) - torch.lgamma(bb)
                            + aa * torch.log(xx) + bb * torch.log1p(-xx))
        tail = leading * _beta_continued_fraction(aa, bb, xx) / aa
        result[interior] = torch.where(flip, 1 - tail, tail)
    result[x == 0] = 0
    return result


def annotation_weights(maf, annotations=None, *, dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return three distinct weight families in beta-then-annotation order.

    dtype=float32 computes all columns on the input device; dtype=float64
    selects the explicit scalar control. PHRED ranks keep the original formula.
    """
    f = _double(maf)
    if f.ndim != 1 or not f.numel():
        raise ValueError("maf must be a nonempty vector")
    _finite(f, "maf")
    if bool(((f <= 0) | (f > 0.5)).any()):
        raise ValueError("weights require minor-allele frequencies in (0, 0.5]")
    phred = None
    if annotations is not None:
        phred = _double(annotations, device=f.device)
        if phred.ndim != 2 or phred.shape[0] != f.numel():
            raise ValueError("annotations must have shape variants-by-annotations")
        _finite(phred, "annotations")
        if bool((phred < 0).any()):
            raise ValueError("PHRED annotations cannot be negative")
    if dtype == torch.float32:
        from ._reference_weights import native_annotation_weights
        return native_annotation_weights(f, phred)
    # The original scalar libm transformations are retained at this small
    # weight boundary. Score/covariance and probability calculations stay on
    # the input device; the helper reports its CPU and transfer work.
    from ._reference_weights import reference_annotation_weights
    return reference_annotation_weights(f, phred)


def _chi1_from_score(score: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    # Probability arithmetic alone uses FP64 to retain small positive P.
    score = _double(score)
    variance = _double(variance, device=score.device)
    if bool((variance <= 0).any()):
        raise DegenerateTestError("score test variance must be positive")
    return torch.erfc(torch.sqrt(score.square() / variance / 2))


def _cct_rows(pvalues, weights, *, batch_validation=True):
    """Internal CCT for all annotation columns, with unchanged tail branches."""
    p = pvalues.double()
    w = weights.double()
    if p.ndim != 2 or p.shape != w.shape or not p.shape[1]:
        raise ValueError("pvalues must be a nonempty finite vector in [0, 1]")
    if batch_validation:
        from ._statistics_sync import host_flags
        mass = w.sum(dim=1)
        _STATISTICS_METADATA["cct_validation_flag_batches"] += 1
        finite_p, out_of_range, finite_w, negative_w, finite_mass, nonpositive_mass = host_flags(
            torch.isfinite(p).all(), ((p<0)|(p>1)).any(), torch.isfinite(w).all(),
            (w<0).any(), torch.isfinite(mass).all(), (mass<=0).any())
        if not finite_p or out_of_range:
            raise ValueError("pvalues must be a nonempty finite vector in [0, 1]")
        if not finite_w or negative_w:
            raise ValueError("weights must be finite, nonnegative and match pvalues")
        if not finite_mass or nonpositive_mass:
            raise DegenerateTestError("CCT requires a positive finite weight sum")
    else:
        if not bool(torch.isfinite(p).all()) or bool(((p<0)|(p>1)).any()):
            raise ValueError("pvalues must be a nonempty finite vector in [0, 1]")
        mass = w.sum(dim=1)
        if not bool(torch.isfinite(w).all()) or bool((w<0).any()):
            raise ValueError("weights must be finite, nonnegative and match pvalues")
        if not bool(torch.isfinite(mass).all()) or bool((mass<=0).any()):
            raise DegenerateTestError("CCT requires a positive finite weight sum")
    w = w / mass[:,None]
    small = p < 1e-16
    statistic = torch.where(small,w/p/math.pi,0.).sum(dim=1)
    statistic += torch.where(~small,w*torch.tan((.5-p)*math.pi),0.).sum(dim=1)
    if bool(torch.isnan(statistic).any()):raise DegenerateTestError("CCT statistic is undefined")
    return torch.where(statistic>1e15,(1/statistic)/math.pi,_cauchy_sf_tensor(statistic))


def _native_weight_relations(rows, *, scratch_limit=64 * 2**20):
    """Prove proportionality exactly, batching only elementwise comparisons.

    The scratch bound covers two FP64 cross-products, their Boolean equality,
    and the pivot gather. These are weight identity checks, not FP64 dense
    score/covariance products. One table reaches the CPU to retain the original
    representative order; pivot indexes remain on the device.
    """
    if rows.dtype != torch.float32 or rows.ndim != 2:
        raise ValueError("native weight relations require an FP32 row matrix")
    if bool((~(rows != 0).any(dim=1)).any()):
        raise DegenerateTestError("SKAT covariance has no eigenvalue at or above 1e-8")
    number_weights, number_variants = rows.shape
    if number_weights < 1 or number_variants < 1:
        raise ValueError("native weight relations require nonempty rows")
    persistent_bytes = 8 * number_weights * number_variants + 16 * number_weights + number_weights**2
    budget = int(scratch_limit)
    if rows.is_cuda:
        from . import tf32
        with torch.cuda.device(rows.device):
            free, _ = torch.cuda.mem_get_info()
        available = tf32._product_workspace_availability(
            allocated=torch.cuda.memory_allocated(rows.device),
            reserved=torch.cuda.memory_reserved(rows.device), free=free,
            limit=int(tf32._memory_limit_bytes),
            reserve=tf32._memory_reserve_bytes)["available_bytes"]
        budget = min(budget, available - persistent_bytes)
    bytes_per_row = 17 * number_weights * number_variants + 9 * number_weights
    if budget < bytes_per_row:
        raise MemoryError("exact weight relation scratch cannot fit one row within the memory budget")
    row_chunk = min(number_weights, budget // bytes_per_row)
    exact = rows.double()
    pivots = (rows != 0).to(torch.int32).argmax(dim=1)
    pivot_values = exact.gather(1, pivots[:, None])[:, 0]
    related = torch.empty((number_weights, number_weights), dtype=torch.bool, device=rows.device)
    for start in range(0, number_weights, row_chunk):
        end = min(number_weights, start + row_chunk)
        left = exact[None, :, :] * pivot_values[start:end, None, None]
        right = exact[:, pivots[start:end]].T[:, :, None] * exact[start:end, None, :]
        related[start:end] = (left == right).all(dim=2)
        del left, right
        _STATISTICS_METADATA["native_relation_row_batches"] += 1
    _STATISTICS_METADATA["native_relation_max_scratch_bytes"] = max(
        _STATISTICS_METADATA["native_relation_max_scratch_bytes"], row_chunk * bytes_per_row)
    _STATISTICS_METADATA["native_relation_bulk_d2h_calls"] += int(rows.is_cuda)
    return exact, pivots, related.cpu().tolist()


def _native_weighted_spectra(covariance, weights):
    from ._weighted_spectra import native_weighted_spectra
    return native_weighted_spectra(covariance, weights)


def _record_tail_metadata(metadata):
    """Keep a bounded, JSON-serializable tail-method audit trail."""
    entry = {str(key): value for key, value in metadata.items()}
    _STATISTICS_METADATA["tail_metadata"].append(entry)
    del _STATISTICS_METADATA["tail_metadata"][:-64]


def _liu_failure(*, index, m, rank, probes, seed, reason, **extra):
    """Record a reproducible long-mask failure before stopping that mask."""
    _STATISTICS_METADATA["liu_approximate_failures"] += 1
    record = {"method": "liu_hutchinson_approx", "approximate": True,
              "m": int(m), "M": int(m), "rank": int(rank),
              "probes": int(probes), "probe_count": int(probes),
              "seed": int(seed), "weight_index": int(index),
              "reason": str(reason)}
    record.update({str(key): value for key, value in extra.items()})
    _STATISTICS_METADATA["liu_failure_records"].append(record)
    raise ArithmeticError(f"Liu approximation stopped for weight {index}: {reason}")




def _saddlepoint_skat_pvalues(q_values, spectra):
    """Use the established complete-spectrum Saddle/moment tail for all legal M."""
    from ._fused_saddle import quadratic_form_sf_batch
    values = quadratic_form_sf_batch(q_values, spectra).to(dtype=torch.float64)
    if not bool(torch.isfinite(values).all()) or bool(((values < 0) | (values > 1)).any()):
        raise ArithmeticError("Saddle/moment SKAT tail failed for one or more weights")
    _record_tail_metadata({
        "method": "saddlepoint_gamma",
        "m": int(spectra.shape[1]),
        "weights": int(spectra.shape[0]),
        "eigenvalue_cutoff": 1e-8,
        "root_tolerance": 1e-8,
        "moment_switch": 1e-4,
        "approximate": False,
    })
    return values.to(dtype=torch.float64, device=q_values.device)


def _liu_hutchinson_pvalues(q_values, covariance, skat_weights, *, rank=512,
                            probes=32, seed=1729, threshold=5000):
    """Original Liu approximation above the configured mask threshold without a full eigensolve.

    The largest ``rank`` eigenvalues are obtained with LOBPCG.  Fixed-seed
    Rademacher probes estimate the residual c2--c4 traces; c1 is the exact
    covariance diagonal sum.  A negative residual beyond its probe standard
    error is a hard failure, so the caller cannot silently return an unstable
    tail.  The standard Liu branch uses l=1/s1^2 when s1^2<=s2 (Liu.mod's
    l=1/s2 branch is deliberately excluded).
    """
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Liu approximation requires a square covariance")
    m = int(covariance.shape[0])
    if m <= int(threshold):
        raise ValueError("Liu approximation is reserved above the configured mask threshold")
    device = covariance.device
    dtype = covariance.dtype
    k = min(int(rank), m - 1)
    p = min(int(probes), m)
    if k < 1 or p < 4:
        raise ValueError("Liu approximation requires positive rank and at least four probes")
    if covariance.is_cuda:
        from . import tf32
        limit_bytes = int(tf32._memory_limit_bytes)
        allocated = int(torch.cuda.memory_allocated(covariance.device))
        reserved = int(torch.cuda.memory_reserved(covariance.device))
        free_bytes, _ = torch.cuda.mem_get_info(covariance.device)
        # v is already resident.  Account for the temporary and final
        # weighted matrices, LOBPCG basis/work vectors, and probes before
        # allocating a second M-by-M object.
        estimate_bytes = (8 * m * m + 20 * m * k + 4 * m * p +
                          int(tf32._memory_reserve_bytes))
        if (allocated + estimate_bytes > limit_bytes
                or estimate_bytes > int(free_bytes) + max(0, reserved-allocated)):
            _liu_failure(index=0, m=m, rank=k, probes=p, seed=seed,
                         reason="estimated CUDA workspace exceeds configured or live memory",
                         allocated_bytes=allocated, estimated_bytes=estimate_bytes,
                         limit_bytes=limit_bytes, live_free_bytes=int(free_bytes),
                         reusable_reserved_bytes=max(0, reserved-allocated))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    z = torch.randint(0, 2, (m, p), generator=generator, device=device,
                      dtype=torch.int8).to(dtype=dtype).mul_(2).sub_(1)
    _STATISTICS_METADATA["liu_init"] = "fixed-seed-rademacher-X"
    pvalues = []
    trace_se = []
    weight_records = []
    for index in range(int(skat_weights.shape[1])):
        weights = skat_weights[:, index].to(dtype=dtype, device=device)
        weighted = covariance * weights[:, None] * weights[None, :]
        init = torch.randint(0, 2, (m, k), generator=generator, device=device,
                             dtype=torch.int8).to(dtype=dtype).mul_(2).sub_(1)
        top, basis = torch.lobpcg(weighted, k=k, B=None, X=init,
                                  largest=True, niter=100, tol=1e-5)
        order = top.flatten().argsort(descending=True)
        top = top.flatten().gather(0, order).to(dtype=torch.float64)
        basis = basis[:, order].to(dtype=dtype)
        if not bool(torch.isfinite(top).all()) or not bool(torch.isfinite(basis).all()):
            _liu_failure(index=index, m=m, rank=k, probes=p, seed=seed,
                         reason="LOBPCG produced non-finite eigenpairs")
        if bool((top < -1e-5).any()):
            _liu_failure(index=index, m=m, rank=k, probes=p, seed=seed,
                         reason="LOBPCG produced a negative eigenvalue")
        trace = weighted.diagonal().to(dtype=torch.float64).sum()
        top_trace_gap = trace - top.sum()
        if bool(top_trace_gap < -1e-4 * trace.abs().clamp_min(1)):
            _liu_failure(index=index, m=m, rank=k, probes=p, seed=seed,
                         reason="leading eigenvalue trace exceeds covariance trace",
                         top_trace_gap=float(top_trace_gap.detach().cpu()))
        projected = torch.matmul(weighted, basis)
        residual_norm = torch.linalg.vector_norm(projected - basis * top.to(dtype=dtype)) / torch.linalg.vector_norm(projected).clamp_min(1e-12)
        orthogonality = torch.linalg.vector_norm(torch.matmul(basis.T, basis) - torch.eye(k, dtype=dtype, device=device)) / math.sqrt(float(k))
        residual_norm_value = float(residual_norm.detach().cpu())
        orthogonality_value = float(orthogonality.detach().cpu())
        _STATISTICS_METADATA["liu_lobpcg_max_residual"] = max(_STATISTICS_METADATA["liu_lobpcg_max_residual"], residual_norm_value)
        _STATISTICS_METADATA["liu_lobpcg_max_orthogonality"] = max(_STATISTICS_METADATA["liu_lobpcg_max_orthogonality"], orthogonality_value)
        if residual_norm_value > 1e-3 or orthogonality_value > 1e-2:
            _liu_failure(index=index, m=m, rank=k, probes=p, seed=seed,
                         reason="LOBPCG convergence or orthogonality guard failed",
                         residual_norm=residual_norm_value,
                         orthogonality=orthogonality_value)
        # Deflate the leading subspace before Hutchinson estimation.  This
        # estimates the residual spectrum directly and avoids spending probe
        # variance on the rank-k component already obtained by LOBPCG.
        projections = torch.matmul(basis.T, z)
        vectors = z
        estimates = []
        errors = []
        for power in range(1, 5):
            vectors = torch.matmul(weighted, vectors)
            leading = (projections.square() * top.to(dtype=dtype).pow(power)[:, None]).sum(dim=0)
            samples = ((z * vectors).sum(dim=0) - leading).to(dtype=torch.float64)
            estimates.append(samples.mean())
            errors.append(samples.std(unbiased=True) / math.sqrt(float(p)))
        c1 = weighted.diagonal().to(dtype=torch.float64).sum()
        top_moments = torch.stack([top.pow(power).sum() for power in range(1, 5)])
        estimates_tensor = torch.stack(estimates)
        errors_tensor = torch.stack(errors)
        residual = estimates_tensor
        # The probe estimate may be slightly below the exact leading moment;
        # tolerate one standard error, but reject a materially negative tail.
        tolerance = torch.maximum(errors_tensor * 4, residual.new_tensor(1e-10) * top_moments.abs().clamp_min(1))
        if bool((residual[1:] < -tolerance[1:]).any()):
            _liu_failure(index=index, m=m, rank=k, probes=p, seed=seed,
                         reason="negative residual trace exceeds probe tolerance")
        relative_se = errors_tensor[1:] / (top_moments[1:] + residual[1:].abs()).clamp_min(1e-12)
        max_relative_se = float(relative_se.max().detach().cpu())
        _STATISTICS_METADATA["liu_trace_max_relative_se"] = max(_STATISTICS_METADATA["liu_trace_max_relative_se"], max_relative_se)
        if max_relative_se > 0.25:
            _liu_failure(index=index, m=m, rank=k, probes=p, seed=seed,
                         reason="Hutchinson trace relative standard error exceeds 0.25",
                         max_relative_se=max_relative_se,
                         trace_se=errors_tensor.detach().cpu().tolist())
        moments = torch.cat((c1.reshape(1), top_moments[1:] + residual[1:].clamp_min(0)))
        c1, c2, c3, c4 = moments
        if bool((~torch.isfinite(moments)).any()) or bool(c2 <= 0) or bool(c3 <= 0) or bool(c4 <= 0):
            _liu_failure(index=index, m=m, rank=k, probes=p, seed=seed,
                         reason="Liu moments are non-finite or non-positive")
        s1 = c3 / c2.pow(1.5)
        s2 = c4 / c2.square()
        if bool((~torch.isfinite(torch.stack((s1, s2))) | (s1 <= 0) | (s2 <= 0)).any()):
            _liu_failure(index=index, m=m, rank=k, probes=p, seed=seed,
                         reason="Liu standardized moments are invalid")
        if bool(s1.square() > s2):
            a = 1 / (s1 - torch.sqrt(s1.square() - s2))
            delta = s1 * a.pow(3) - a.square()
            degrees = a.square() - 2 * delta
        else:
            a = 1 / s1
            delta = s1.new_zeros(())
            degrees = 1 / s1.square()
        transformed = (q_values[index].to(dtype=torch.float64) - c1) / torch.sqrt(2 * c2)
        transformed = transformed * (math.sqrt(2) * a) + degrees + delta
        if bool(((~torch.isfinite(transformed)) | (degrees <= 0) | (delta < 0)).any()):
            _liu_failure(index=index, m=m, rank=k, probes=p, seed=seed,
                         reason="Liu transformed statistic is invalid")
        # SciPy is already a project runtime dependency and supplies the
        # noncentral chi-square survival function absent from PyTorch 2.5.
        from scipy.stats import ncx2
        probability = float(ncx2.sf(float(transformed.detach().cpu()),
                                    float(degrees.detach().cpu()),
                                    float(delta.detach().cpu())))
        if not math.isfinite(probability) or probability < 0 or probability > 1:
            _liu_failure(index=index, m=m, rank=k, probes=p, seed=seed,
                         reason="Liu probability is invalid")
        pvalues.append(q_values.new_tensor(probability, dtype=torch.float64))
        trace_se.append(errors_tensor.detach().cpu().tolist())
        weight_records.append({"index": index, "c1": float(c1), "c2": float(c2),
                               "c3": float(c3), "c4": float(c4),
                               "degrees": float(degrees), "delta": float(delta),
                               "lobpcg_residual": residual_norm_value,
                               "lobpcg_orthogonality": orthogonality_value,
                               "trace_relative_se": max_relative_se})
        del weighted, top, basis, init, projected, projections, vectors
    _STATISTICS_METADATA["liu_approximate_calls"] += 1
    _record_tail_metadata({
        "method": "liu_hutchinson_approx",
        "m": m,
        "M": m,
        "rank": k,
        "probes": p,
        "probe_count": p,
        "seed": int(seed),
        "trace_se": trace_se,
        "lobpcg_init": "fixed-seed-rademacher-X",
        "weights": weight_records,
        "approximate": True,
    })
    return torch.stack(pvalues).to(device=q_values.device)


def _fastskat_failure(*, index, m, rank, seed, reason, **extra):
    """Stop a long mask when the top-k or residual moments are not reliable."""
    _STATISTICS_METADATA["fastskat_approximate_failures"] += 1
    record = {"method": "fastskat_hybrid", "approximate": True,
              "m": int(m), "M": int(m), "rank": int(rank),
              "probes": 0, "probe_count": 0, "seed": int(seed),
              "weight_index": int(index), "reason": str(reason)}
    record.update({str(key): value for key, value in extra.items()})
    _STATISTICS_METADATA["fastskat_failure_records"].append(record)
    raise ArithmeticError(f"FastSKAT hybrid stopped for weight {index}: {reason}")


def _fastskat_hybrid_sf_tensor(statistic, top_eigenvalues, residual_mean,
                               residual_second, diagnostics=None):
    """Saddlepoint tail for top eigenvalues plus a Satterthwaite residual.

    If the residual spectrum has first moment ``mu`` and second power sum
    ``s2``, it is represented by ``a * chi2_nu`` with ``a=s2/mu`` and
    ``nu=mu**2/s2``.  The cgf and its derivatives then retain the leading
    eigenvalues individually and include the residual as a continuous
    chi-square multiplicity. Near the mean a two-moment Satterthwaite
    Gamma fallback is used for the hybrid distribution.
    """
    top = _double(top_eigenvalues).reshape(-1)
    q_raw = _double(statistic, device=top.device)
    mu = _double(residual_mean, device=top.device)
    s2 = _double(residual_second, device=top.device)
    if (not top.numel() or q_raw.numel() != 1 or not bool(torch.isfinite(top).all())
            or not bool(torch.isfinite(torch.stack((q_raw, mu, s2))).all())
            or bool(q_raw < 0) or bool(mu < 0) or bool(s2 < 0)):
        raise ValueError("FastSKAT hybrid requires finite nonnegative moments")
    if bool(q_raw == 0):
        if diagnostics is not None:
            diagnostics.update(tail_branch="zero_statistic", has_residual=bool(mu > 0 and s2 > 0))
        return q_raw.new_tensor(1.)
    if bool((top < -1e-6).any()):
        raise ArithmeticError("FastSKAT hybrid received a materially negative top eigenvalue")
    top = top.clamp_min(0)
    top_scale = top.max() if top.numel() else q_raw.new_zeros(())
    # A residual component is absent only when both moments are zero. A
    # comparison between moments of different orders is not scale invariant.
    if bool((mu > 0) != (s2 > 0)):
        raise ArithmeticError("FastSKAT hybrid residual moments are inconsistent")
    has_residual = bool(mu > 0 and s2 > 0)
    if has_residual:
        residual_scale = s2 / mu
        residual_dof = mu.square() / s2
    else:
        residual_scale = q_raw.new_zeros(())
        residual_dof = q_raw.new_zeros(())
    scale = torch.maximum(top_scale, residual_scale).clamp_min(1e-30)
    scaled_top = top / scale
    scaled_residual_scale = residual_scale / scale
    q = q_raw / scale
    mean = scaled_top.sum()
    if has_residual:
        mean = mean + residual_dof * scaled_residual_scale
    total_mean = top.sum() + mu
    total_second = top.square().sum() + s2
    if bool(total_second <= 0) or bool(total_mean <= 0):
        raise DegenerateTestError("FastSKAT hybrid has zero total variance")

    def derivative(root):
        value = (scaled_top / (1 - 2 * scaled_top * root)).sum()
        if has_residual:
            value = value + (mu / scale) / (1 - 2 * scaled_residual_scale * root)
        return value

    if bool(q >= mean):
        lower = q.new_zeros(())
    else:
        lower = q.new_tensor(-1.)
        for _ in range(80):
            if bool(derivative(lower) <= q):
                break
            lower = lower * 2
        else:
            raise ArithmeticError("FastSKAT hybrid could not bracket a negative saddlepoint root")
    upper = q.new_tensor(0.499999999)
    if bool(derivative(upper) < q):
        raise ArithmeticError("FastSKAT hybrid could not bracket a positive saddlepoint root")
    root = q.new_zeros(())
    for iteration in range(256):
        root = (lower + upper) / 2
        residual = derivative(root) - q
        if bool(residual.abs() <= 1e-10 * (q.abs() + 1)):
            break
        upper = torch.where(residual > 0, root, upper)
        lower = torch.where(residual > 0, lower, root)
    else:
        raise ArithmeticError("FastSKAT hybrid saddlepoint root did not converge")
    if diagnostics is not None:
        diagnostics.update(
            has_residual=has_residual,
            saddlepoint_iterations=iteration + 1,
            saddlepoint_root=float(root.detach().cpu()),
            saddlepoint_relative_residual=float((residual.abs() / (q.abs() + 1)).detach().cpu()),
            tail_branch=("satterthwaite_two_moment_gamma" if bool(root.abs() < 1e-4)
                         else "hybrid_saddlepoint"),
        )
    if bool(root.abs() < 1e-4):
        dof = total_mean.square() / total_second
        adjusted = ((q_raw - total_mean) / torch.sqrt(2 * total_second)
                    * torch.sqrt(2 * dof) + dof)
        return torch.where(adjusted <= 0, q_raw.new_tensor(1.),
                           torch.special.gammaincc(dof / 2, adjusted / 2))
    cumulant = -0.5 * torch.log1p(-2 * scaled_top * root).sum()
    if has_residual:
        x = -2 * scaled_residual_scale * root
        safe_x = torch.where(x == 0, x.new_tensor(1.), x)
        log_ratio = torch.where(x == 0, x.new_tensor(1.), torch.log1p(x) / safe_x)
        cumulant = cumulant + (mu / scale) * root * log_ratio
    w2 = 2 * (root * q - cumulant)
    if bool(w2 <= 0):
        raise ArithmeticError("FastSKAT hybrid saddlepoint has a non-positive signed root")
    signed_root = torch.copysign(torch.sqrt(w2), root)
    second_derivative = 2 * (scaled_top.square() /
                             (1 - 2 * scaled_top * root).square()).sum()
    if has_residual:
        second_derivative = second_derivative + 2 * (s2 / scale.square()) / (1 - 2 * scaled_residual_scale * root).square()
    v = root * torch.sqrt(second_derivative)
    z = signed_root + torch.log(v / signed_root) / signed_root
    return (0.5 * torch.erfc(z / math.sqrt(2))).clamp(0, 1)


def _fastskat_ritz_pairs(weighted, basis):
    """Pair Ritz values with the corresponding vectors of the same subspace.

    Sorting the values without rotating the basis gives an invalid eigenpair
    residual.  The small orthogonal rotation changes neither the retained
    subspace nor the residual-moment definition.
    """
    ritz = torch.matmul(basis.T, torch.matmul(weighted, basis))
    values, rotation = torch.linalg.eigh(ritz, UPLO="U")
    return (values.flip(0).to(dtype=torch.float64),
            torch.matmul(basis, rotation.flip(1)))


def _fastskat_hybrid_pvalues(q_values, covariance, skat_weights, *, rank=512,
                             seed=1729, threshold=5000, _lobpcg_worker_pool=None,
                             _lobpcg_new_workspace_limit=None, _lobpcg_workspace_callback=None):
    """FastSKAT-style top-k spectrum plus exact-dense residual moments.

    The dense covariance already held by the current WGS pipeline makes the
    residual trace moments available without Hutchinson probes: ``c1`` is the
    diagonal trace and ``c2=||A||_F^2``.  Only the leading ``rank`` eigenpairs
    use LOBPCG; the residual is reduced to a Satterthwaite chi-square and is
    included analytically in the saddlepoint cgf.  This is approximate because
    the residual spectrum is collapsed to two moments.
    """
    completed = False
    pvalues = pair_tiles = None
    try:
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("FastSKAT hybrid requires a square covariance")
        m = int(covariance.shape[0])
        if m <= int(threshold):
            raise ValueError("FastSKAT hybrid is reserved above the configured mask threshold")
        if q_values.ndim != 1 or skat_weights.ndim != 2 or skat_weights.shape[0] != m:
            raise ValueError("FastSKAT hybrid q/weights have incompatible shapes")
        device = covariance.device
        dtype = covariance.dtype
        k = min(int(rank), m - 1)
        if k < 1:
            raise ValueError("FastSKAT hybrid requires a positive rank")
        if covariance.is_cuda:
            from . import tf32
            limit_bytes = int(tf32._memory_limit_bytes)
            allocated = int(torch.cuda.memory_allocated(covariance.device))
            reserved = int(torch.cuda.memory_reserved(covariance.device))
            free_bytes, _ = torch.cuda.mem_get_info(covariance.device)
            estimate_bytes = 8 * m * m + 24 * m * k + int(tf32._memory_reserve_bytes)
            if (allocated + estimate_bytes > limit_bytes
                    or estimate_bytes > int(free_bytes) + max(0, reserved-allocated)):
                _fastskat_failure(index=0, m=m, rank=k, seed=seed,
                                  reason="estimated CUDA workspace exceeds configured or live memory",
                                  allocated_bytes=allocated, estimated_bytes=estimate_bytes,
                                  limit_bytes=limit_bytes, live_free_bytes=int(free_bytes),
                                  reusable_reserved_bytes=max(0, reserved-allocated))
        from ._fastskat_numerics import refine_spectrum_and_moments
        from .precision_audit import explicit_fp64_spectral_refinement
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        pvalues = []
        weight_records = []
        from ._fastskat_lobpcg_tiles import FastSKATPairTiles
        pair_tiles = FastSKATPairTiles(covariance, skat_weights, generator, k,
                                      worker_pool=_lobpcg_worker_pool,
                                      max_new_workspace_bytes=_lobpcg_new_workspace_limit,
                                      workspace_callback=_lobpcg_workspace_callback)
        for index in range(int(skat_weights.shape[1])):
            try:
                weights, weighted, init, top, basis = pair_tiles.take(index)
            except RuntimeError as error:
                _fastskat_failure(index=index, m=m, rank=k, seed=seed,
                                  reason="LOBPCG failed", error=str(error))
            top = top.flatten()
            basis = basis.to(dtype=dtype)
            if not bool(torch.isfinite(top).all()) or not bool(torch.isfinite(basis).all()):
                _fastskat_failure(index=index, m=m, rank=k, seed=seed,
                                  reason="LOBPCG produced non-finite eigenpairs")
            # Refine the sampled subspace against the stored matrix itself.
            # FP64 is confined to spectral refinement; genotype/covariance GEMMs
            # retain their explicit TF32 route and the dense matrix stays FP32.
            try:
                with explicit_fp64_spectral_refinement():
                    refined = refine_spectrum_and_moments(weighted, basis)
            except (RuntimeError, ArithmeticError) as error:
                _fastskat_failure(index=index, m=m, rank=k, seed=seed,
                                  reason="FP64 spectral refinement failed", error=str(error))
            top, basis, projected = (refined[key] for key in ("top", "basis", "projected"))
            trace, trace2 = refined["trace"], refined["trace2"]
            residual_mean, residual_second = refined["residual_mean"], refined["residual_second"]
            residual_norm_value = float(refined["residual_norm"].detach().cpu())
            orthogonality_value = float(refined["orthogonality"].detach().cpu())
            _STATISTICS_METADATA["fastskat_lobpcg_max_residual"] = max(
                _STATISTICS_METADATA["fastskat_lobpcg_max_residual"], residual_norm_value)
            _STATISTICS_METADATA["fastskat_lobpcg_max_orthogonality"] = max(
                _STATISTICS_METADATA["fastskat_lobpcg_max_orthogonality"], orthogonality_value)
            if residual_norm_value > 0.25 or orthogonality_value > 1e-2:
                _fastskat_failure(index=index, m=m, rank=k, seed=seed,
                                  reason="LOBPCG convergence or orthogonality guard failed",
                                  residual_norm=residual_norm_value,
                                  orthogonality=orthogonality_value)
            tail_diagnostics = {}
            pvalue = _fastskat_hybrid_sf_tensor(q_values[index], top, residual_mean,
                                                residual_second, diagnostics=tail_diagnostics)
            if not bool(torch.isfinite(pvalue)):
                _fastskat_failure(index=index, m=m, rank=k, seed=seed,
                                  reason="FastSKAT hybrid probability is non-finite")
            pvalues.append(pvalue.to(dtype=torch.float64))
            weight_records.append({
                "index": index,
                "residual_mean": float(residual_mean.detach().cpu()),
                "residual_second": float(residual_second.detach().cpu()),
                "residual_scale": float((residual_second / residual_mean).detach().cpu())
                if bool(residual_mean > 0) else 0.0,
                "residual_dof": float((residual_mean.square() / residual_second).detach().cpu())
                if bool(residual_second > 0) else 0.0,
                "lobpcg_residual": residual_norm_value,
                "lobpcg_orthogonality": orthogonality_value,
                **tail_diagnostics,
                "spectral_refinement": refined["metadata"],
            })
            del weighted, top, basis, init, projected, refined
        _STATISTICS_METADATA["fastskat_approximate_calls"] += 1
        _record_tail_metadata({
            "method": "fastskat_hybrid",
            "tail_method": "top_k_plus_residual_satterthwaite_saddlepoint",
            "m": m,
            "M": m,
            "rank": k,
            "probes": 0,
            "probe_count": 0,
            "seed": int(seed),
            "lobpcg_init": "fixed-seed-gaussian-X",
            "lobpcg_max_iter": 200,
            "lobpcg_batch_tiles": pair_tiles.records,
            "subspace_residual_limit": 0.25,
            "subspace_power_iterations": 2,
            "spectral_refinement_precision": "blockwise_fp64",
            "dense_storage_precision": "fp32",
            "residual_second_reduction": "fp64_square_and_sum",
            "residual_definition": "matched_ritz_pairs_frobenius_relative",
            "residual_trace_method": "exact_dense",
            "weights": weight_records,
            "approximate": True,
        })
        result = torch.stack(pvalues).to(device=q_values.device)
        completed = True
        return result
    except BaseException as error:
        from ._lobpcg_stream_pool import clear_exception_frames
        clear_exception_frames(error)
        raise
    finally:
        if pair_tiles is not None:
            pair_tiles.close()
        if not completed and pvalues is not None:
            pvalues.clear()
        pair_tiles = _lobpcg_worker_pool = _lobpcg_workspace_callback = None
        q_values = covariance = skat_weights = pvalues = generator = None
        weighted = top = basis = init = projected = refined = weights = None
        trace = trace2 = residual_mean = residual_second = pvalue = result = None


def _association_probability_fields(pvalues, labels, *, tail_optimization=False, batch_copy=True):
    """Retain CCT calls and key order; copy final FP64 P values together.

    The scalar path is an explicit differential control. Only the final output
    transfer changes; CCT validation/branches/reductions execute as before.
    """
    if pvalues.ndim != 2 or pvalues.shape != (3, 2 * (len(labels) + 1)) or pvalues.dtype != torch.float64:
        raise ValueError("WGS output requires complete FP64 probability rows")
    fields = {}
    def emit(value):
        if not batch_copy and value.is_cuda:
            _STATISTICS_METADATA["probability_output_d2h_calls"] += 1
            _STATISTICS_METADATA["probability_output_d2h_values"] += 1
        return value if batch_copy else float(value)
    width = len(labels) + 1
    for row, method, combined in ((0, "SKAT", "WGS-S"), (1, "Burden", "WGS-B"), (2, "ACAT-V", "WGS-A")):
        for beta_index, beta_name in enumerate(("1,25", "1,1")):
            values = pvalues[row, beta_index * width:(beta_index + 1) * width]
            base = f"{method}({beta_name})"
            fields[base] = emit(values[0])
            for label, value in zip(labels, values[1:]):
                fields[f"{base}-{label}"] = emit(value)
            fields[f"{combined}({beta_name})"] = emit(_cct_tensor(values, sync_light=tail_optimization))
    fields["ACAT-O"] = emit(_cct_tensor(pvalues[:, (0, width)].reshape(-1), sync_light=tail_optimization))
    fields["WGS-O"] = emit(_cct_tensor(pvalues.reshape(-1), sync_light=tail_optimization))
    if batch_copy:
        if pvalues.is_cuda:
            _STATISTICS_METADATA["probability_output_d2h_calls"] += 1
            _STATISTICS_METADATA["probability_output_d2h_values"] += len(fields)
        values = torch.stack(list(fields.values())).detach().to(device="cpu").tolist()
        return dict(zip(fields, values))
    return fields


def association_test(
    score,
    covariance,
    maf,
    mac,
    annotations=None,
    names: Sequence[str] | None = None,
    *,
    acat_calibration: str = "chi2",
    dof: int | None = None,
    n: int | None = None,
    covariate_count: int | None = None,
    mac_threshold: int = 10,
    rare_maf_cutoff: float = 0.01,
    rv_num_cutoff: int = 2,
    rv_num_cutoff_max: int = 1_000_000_000,
    cmac: float | None = None,
    _skat_pvalues: torch.Tensor | None = None,
    matmul_mode: str = "fp64",
    long_mask_threshold: int = 5000,
    long_mask_method: str = "full",
    long_mask_rank: int = 512,
    long_mask_seed: int = 1729,
    tail_optimization: bool = False,
    weight_batch_optimization: bool = False,
    output_batch_optimization: bool = True,
    cct_validation_optimization: bool = True,
    _defer_full_tail: bool = False,
) -> dict[str, float | int] | PendingCompleteTail:
    """Return WGS columns from already oriented, imputed score inputs.

    Native tf32 uses FP32 scores, covariance and weights. The default full
    policy computes independent complete H32 spectra for every legal mask,
    with the established Saddle/fourth-moment Gamma tails. GPU admission uses
    the live process budget. Explicit fastskat/liu retain their earlier
    approximate path above long_mask_threshold; that threshold does not
    dispatch the default full spectrum. Probability arithmetic stays FP64.
    Explicit fp64 controls preserve the earlier scalar implementation.
    weight_batch_optimization groups score products and chi-square tails,
    retains individual one-dimensional reductions for explicit FP64 controls.
    Native mode always batches all columns and reuses common ACAT-V inputs.
    output_batch_optimization copies final P values together;
    cct_validation_optimization combines ACAT-V input flags. Independent false
    controls retain scalar output transfers and separate validation checks.
    covariance includes the fitted dispersion. acat_calibration='chi2'
    is the SMMAT/relatedness (or binary ordinary) branch;
    'gaussian_glm' requires dof=n-rank(X), and uses t(dof-1) only for
    ACAT-V variants with MAC>mac_threshold. It does not change burden.
    annotations are PHRED values, not pre-transformed ranks. This routine
    never fits or approximates a binary SPA or mixed-model null fit.
    cmac optionally supplies exact sum(G_rare), important after imputation;
    otherwise the sum of the caller's MAC values is returned.
    _defer_full_tail is an internal, explicit native FULL preparation control.
    It retains only detached Q, complete spectra, elementary P rows and small
    output metadata for finish_complete_tails; ordinary calls return a dict.
    """
    validate_mode(matmul_mode)
    if type(_defer_full_tail) is not bool:
        raise ValueError('_defer_full_tail must be a boolean')
    if _defer_full_tail and (matmul_mode != 'tf32' or long_mask_method != 'full'
                            or _skat_pvalues is not None):
        raise ValueError('deferred tails require native tf32 FULL without precomputed SKAT values')
    if type(long_mask_threshold) is not int or long_mask_threshold < 1:
        raise ValueError("long_mask_threshold must be a positive integer")
    _STATISTICS_METADATA["exact_spectrum_threshold"] = None if long_mask_method == "full" else long_mask_threshold
    _STATISTICS_METADATA["spectrum_policy"] = long_mask_method
    if long_mask_method not in ("full", "fastskat", "liu"):
        raise ValueError("long_mask_method must be full, fastskat or liu")
    if int(long_mask_rank) < 1:
        raise ValueError("long_mask_rank must be positive")
    if int(long_mask_seed) < 0:
        raise ValueError("long_mask_seed must be nonnegative")
    native = matmul_mode == "tf32"
    core_dtype = torch.float32 if native else torch.float64
    u = torch.as_tensor(score, dtype=core_dtype)
    v = torch.as_tensor(covariance, dtype=core_dtype, device=u.device)
    f = _double(maf, device=u.device)
    m = _double(mac, device=u.device)
    if u.ndim != 1 or v.shape != (u.numel(), u.numel()) or f.shape != u.shape or m.shape != u.shape:
        raise ValueError("score/maf/mac must be vectors and covariance must be variants-by-variants")
    for value, name in ((u, "score"), (v, "covariance"), (f, "maf"), (m, "mac")):
        _finite(value, name)
    if bool(((f < 0) | (f > 0.5) | (m < 0)).any()):
        raise ValueError("maf must lie in [0,.5] and mac cannot be negative")
    if not 0 < rare_maf_cutoff <= 0.5 or not 1 <= rv_num_cutoff < rv_num_cutoff_max:
        raise ValueError("invalid rare-variant cutoffs")
    if acat_calibration not in ("chi2", "gaussian_glm"):
        raise ValueError("acat_calibration must be chi2 or gaussian_glm")
    if acat_calibration == "gaussian_glm":
        if dof is None and n is not None and covariate_count is not None:
            dof = n - covariate_count
        if dof is None or dof <= 1:
            raise ValueError("gaussian_glm ACAT-V requires residual dof=n-rank(X)>1")
    if mac_threshold < 0:
        raise ValueError("mac_threshold cannot be negative")
    mask = (f > 0) & (f < rare_maf_cutoff)
    count = int(mask.sum())
    if count < rv_num_cutoff or count >= rv_num_cutoff_max:
        raise ValueError("rare-variant count is outside the allowed interval")
    phred = None
    if annotations is not None:
        phred = _double(annotations, device=u.device)
        if phred.ndim != 2 or phred.shape[0] != u.numel():
            raise ValueError("annotations must have one row per original variant")
        phred = phred[mask]
    from ._complete_spectrum_scope import current_reuse_scope, register_covariance_transform
    covariance_origin = v if native and current_reuse_scope() is not None else None
    u, v, f, m = u[mask], v[mask][:, mask], f[mask], m[mask]
    if not bool(torch.allclose(v, v.T, atol=1e-10, rtol=1e-10)):
        raise ValueError("covariance must be symmetric")
    v = (v + v.T) * 0.5
    if covariance_origin is not None:
        register_covariance_transform(covariance_origin, v, mask)
    covariance_origin = None
    k = 0 if phred is None else phred.shape[1]
    labels = [f"annotation_{i + 1}" for i in range(k)] if names is None else list(names)
    if len(labels) != k or len(set(labels)) != k or any(not isinstance(s, str) or not s for s in labels):
        raise ValueError("names must be nonempty distinct annotation names")
    burden_weights, skat_weights, acat_weights = annotation_weights(f, phred, dtype=core_dtype)
    _STATISTICS_METADATA["scalar_reduction_dtype"] = str(core_dtype)
    _STATISTICS_METADATA["eigenvalue_dtype"] = str(core_dtype)
    _STATISTICS_METADATA["weight_transform_dtype"] = "device float32" if native else "CPU float64 explicit control"
    if native:
        weight_batch_optimization = True
        _STATISTICS_METADATA["native_batch_calls"] += 1
    mode_calls = _STATISTICS_METADATA["matmul_mode_calls"]
    mode_calls[matmul_mode] = mode_calls.get(matmul_mode, 0) + 1
    common = m > mac_threshold
    very_rare = ~common
    common_p = _chi1_from_score(u[common], v.diagonal()[common])
    if acat_calibration == "gaussian_glm" and bool(common.any()):
        explained = _double(u[common]).square() / _double(v.diagonal()[common])
        if bool((explained > dof).any()):
            raise ArithmeticError("Gaussian single-variant explained sum of squares exceeds the null total")
        t_squared = explained / (dof - explained) * (dof - 1)
        common_p = _student_t_two_sided(t_squared, dof - 1)
    number_weights = burden_weights.shape[1]
    has_very_rare = bool(very_rare.any())
    all_very_rare = has_very_rare and not bool(common.any())
    # Native mode prepares this only after the full burden product is reduced.
    # The explicit FP64 path retains its established covariance lifetime.
    rare_covariance = None
    if not native and has_very_rare:
        rare_covariance = v if all_very_rare else v[very_rare][:, very_rare]
    burden_variances = rare_variances = None
    if matmul_mode != "fp64":
        # One matrix product across annotation columns. Native quadratic forms
        # use FP32 elementwise products and reductions; only P arithmetic is FP64.
        # No TF32 request reaches the FP64 matrix/vector path below.
        from ._burden import ieee_burden_product
        products = ieee_burden_product(v, burden_weights) if native else matmul(v, burden_weights, mode=matmul_mode)
        burden_variances = (burden_weights * products).sum(dim=0)
        if native:
            del products
        _STATISTICS_METADATA["burden_matrix_products"] += 1
        if all_very_rare:
            rare_variances = burden_variances
        elif has_very_rare:
            if native:
                rare_covariance = v[very_rare][:, very_rare]
            rare_weights = burden_weights[very_rare]
            rare_products = ieee_burden_product(rare_covariance, rare_weights) if native else matmul(rare_covariance, rare_weights, mode=matmul_mode)
            rare_variances = (rare_weights * rare_products).sum(dim=0)
            if native:
                del rare_products, rare_weights
                rare_covariance = None
            _STATISTICS_METADATA["rare_burden_matrix_products"] += 1
    pvalues = torch.empty((3, number_weights), dtype=torch.float64, device=u.device)
    if _skat_pvalues is not None:
        _skat_pvalues = _double(_skat_pvalues, device=u.device)
        if _skat_pvalues.shape != (number_weights,):
            raise ValueError("precomputed SKAT values do not match the annotation weights")
    small_spectra = None
    approximate_spectrum = None
    if native and _skat_pvalues is None and long_mask_method == "full":
        from ._weighted_spectra import sequential_complete_spectra
        small_spectra = sequential_complete_spectra(v, skat_weights)
    elif native and _skat_pvalues is None and count <= long_mask_threshold:
        small_spectra = _native_weighted_spectra(v, skat_weights)
    elif native and _skat_pvalues is None:
        approximate_spectrum = True
    elif _skat_pvalues is None and u.is_cuda and len(u) <= 32 and number_weights > 1:
        # A natural batch of annotation weights selects the small-matrix
        # CUDA Jacobi solver. Reuse that spectrum through the scalar tail.
        matrices = torch.stack([v * ws[:, None] * ws[None, :]
                                for ws in skat_weights.T])
        small_spectra = torch.linalg.eigvalsh(matrices, UPLO="U")
        record_gpu_eigen_route(matrices)
    q_values = ((u.square()[:, None] * skat_weights.square()).sum(dim=0) if native and _skat_pvalues is None
                else _ordered_sum(u.square()[:, None] * skat_weights.square())
                if _skat_pvalues is None and u.is_cuda else None)
    burden_pvalues = rare_pvalues = rare_acat_means = None
    common_normal_transform = rare_normal_transform = None
    if weight_batch_optimization:
        _STATISTICS_METADATA["weight_batch_optimization_calls"] += 1
        # Native mode batches all rows in the FP32 core. Explicit FP64 controls
        # retain separate one-dimensional reductions for the scalar comparator.
        burden_products = burden_weights.T.contiguous() * u[None, :]
        burden_scores = burden_products.sum(dim=1) if native else torch.stack([row.sum() for row in burden_products])
        if burden_variances is None:
            burden_variances = torch.stack([wb @ v @ wb for wb in burden_weights.T])
        burden_pvalues = _chi1_from_score(burden_scores, burden_variances)
        if has_very_rare:
            if all_very_rare:
                rare_scores = burden_scores
                rare_pvalues = burden_pvalues
                rare_variances = burden_variances
            else:
                rare_burden_products = burden_weights[very_rare].T.contiguous() * u[very_rare][None, :]
                rare_scores = rare_burden_products.sum(dim=1) if native else torch.stack([row.sum() for row in rare_burden_products])
                if rare_variances is None:
                    rare_variances = torch.stack([wb @ rare_covariance @ wb
                                                 for wb in burden_weights[very_rare].T])
                rare_pvalues = _chi1_from_score(rare_scores, rare_variances)
            rare_acat_rows = acat_weights[very_rare].T.contiguous()
            rare_acat_means = rare_acat_rows.mean(dim=1) if native else torch.stack([row.mean() for row in rare_acat_rows])
            rare_normal_transform = torch.tan((0.5 - rare_pvalues) * math.pi)
        common_normal_transform = torch.tan((0.5 - common_p) * math.pi)
        if q_values is None and _skat_pvalues is None:
            q_products = skat_weights.T.contiguous().square() * u.square()[None, :]
            q_values = torch.stack([row.sum() for row in q_products])
    if native:
        if _skat_pvalues is None:
            if approximate_spectrum:
                if long_mask_method == "fastskat":
                    pvalues[0] = _fastskat_hybrid_pvalues(
                        q_values, v, skat_weights, rank=int(long_mask_rank),
                        seed=int(long_mask_seed), threshold=long_mask_threshold)
                else:
                    pvalues[0] = _liu_hutchinson_pvalues(
                        q_values, v, skat_weights, rank=int(long_mask_rank),
                        seed=int(long_mask_seed), threshold=long_mask_threshold)
            elif not _defer_full_tail:
                pvalues[0] = _saddlepoint_skat_pvalues(q_values, small_spectra)
        else:
            pvalues[0] = _skat_pvalues
        pvalues[1] = burden_pvalues
        acat_rows = common_p[None,:].expand(number_weights,-1)
        acat_weight_rows = acat_weights[common].T.contiguous()
        if has_very_rare:
            acat_rows = torch.cat((acat_rows,rare_pvalues[:,None]),dim=1)
            acat_weight_rows = torch.cat((acat_weight_rows,rare_acat_means[:,None]),dim=1)
        pvalues[2] = _cct_rows(acat_rows,acat_weight_rows, batch_validation=cct_validation_optimization)
    for i in range(0 if native else number_weights):
        if _skat_pvalues is None:
            ws = skat_weights[:, i]
            if small_spectra is None:
                weighted_covariance = v * ws[:, None] * ws[None, :]
                eigenvalues = torch.linalg.eigvalsh(weighted_covariance, UPLO="U")
                record_gpu_eigen_route(weighted_covariance)
            else:
                eigenvalues = small_spectra[i]
            q = q_values[i] if q_values is not None else torch.sum(u.square() * ws.square())
            refined = False
            pvalues[0, i] = _quadratic_form_sf_tensor(q, eigenvalues, reference_reduction=refined, sync_light=tail_optimization)
        else:
            pvalues[0, i] = _skat_pvalues[i]
        wb = burden_weights[:, i]
        if burden_pvalues is None:
            burden_score = torch.sum(u * wb)
            burden_variance = wb @ v @ wb if burden_variances is None else burden_variances[i]
            pvalues[1, i] = _chi1_from_score(burden_score, burden_variance)
        else:
            pvalues[1, i] = burden_pvalues[i]
        acat_p = common_p
        acat_w = acat_weights[common, i]
        normal_transform = common_normal_transform
        if has_very_rare:
            if rare_pvalues is None:
                rare_wb = wb[very_rare]
                rare_score = torch.sum(u[very_rare] * rare_wb)
                rare_variance = rare_wb @ rare_covariance @ rare_wb if rare_variances is None else rare_variances[i]
                rare_p = _chi1_from_score(rare_score, rare_variance).reshape(1)
                rare_weight = acat_weights[very_rare, i].mean().reshape(1)
            else:
                rare_p = rare_pvalues[i].reshape(1)
                rare_weight = rare_acat_means[i].reshape(1)
                normal_transform = torch.cat((common_normal_transform, rare_normal_transform[i].reshape(1)))
            acat_p = torch.cat((acat_p, rare_p))
            acat_w = torch.cat((acat_w, rare_weight))
        pvalues[2, i] = _cct_tensor(acat_p, acat_w, internal=True, sync_light=tail_optimization,
                                  _normal_transform=normal_transform)
    actual_cmac = float(m.sum()) if cmac is None else float(cmac)
    if _defer_full_tail:
        return PendingCompleteTail(q_values, small_spectra, pvalues, labels,
            num_variant=count, cmac=actual_cmac, tail_optimization=tail_optimization,
            output_batch_optimization=output_batch_optimization)
    result: dict[str, float | int] = {"num_variant": count, "cMAC": actual_cmac}
    result.update(_association_probability_fields(pvalues, labels, tail_optimization=tail_optimization,
                                           batch_copy=output_batch_optimization))
    return result
