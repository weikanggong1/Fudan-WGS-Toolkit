"""Independent continuous/binary PheWAS null preparation from aligned arrays.

SPDX-License-Identifier: GPL-3.0-only
This is model preparation, not a joint MultiSTAAR model. Genotype, CSV routing
and disk-cache readers are separate; the sample axis returned here is explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from .binary_null import fit_logistic_null, fit_logistic_mixed_null
from .null_model import fit_gaussian_null
from .rint import rank_inverse_normal_tensor
from .tf32 import validate_mode


@dataclass(frozen=True)
class SparseKinshipData:
    """ID-bound sparse GRM: diagonal and zero-based off-diagonal COO edges.

    Edges may be one triangle or a consistent symmetric pair. No thresholding,
    rounding of IDs, imputation or GRM construction occurs during loading.
    """
    sample_ids: np.ndarray
    diagonal: np.ndarray
    edge_rows: np.ndarray
    edge_cols: np.ndarray
    edge_values: np.ndarray

    def __post_init__(self):
        ids = np.asarray(self.sample_ids, dtype=str)
        diagonal = np.asarray(self.diagonal, dtype=np.float64)
        rows = np.asarray(self.edge_rows)
        cols = np.asarray(self.edge_cols)
        values = np.asarray(self.edge_values, dtype=np.float64)
        if (ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids)
                or np.any(ids == "") or diagonal.shape != ids.shape
                or not np.isfinite(diagonal).all() or np.any(diagonal < 0)):
            raise ValueError("GRM IDs/diagonal must be unique, aligned and finite")
        if (rows.ndim != 1 or cols.shape != rows.shape or values.shape != rows.shape
                or not np.issubdtype(rows.dtype, np.integer)
                or not np.issubdtype(cols.dtype, np.integer)
                or not np.isfinite(values).all()
                or np.any(rows < 0) or np.any(cols < 0)
                or np.any(rows >= len(ids)) or np.any(cols >= len(ids))
                or np.any(rows == cols)):
            raise ValueError("GRM edges must be aligned integer off-diagonal indices")
        for name, value in (("sample_ids", ids), ("diagonal", diagonal),
                            ("edge_rows", rows.astype(np.int64)),
                            ("edge_cols", cols.astype(np.int64)),
                            ("edge_values", values)):
            owned = value.copy()
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)

    @classmethod
    def load_npz(cls, path):
        """Read a portable NPZ without pickle, including all GRM sample IDs.

        Keys: sample_ids, diagonal, edge_rows, edge_cols, edge_values. Prepared
        input aliases ids/grm_diagonal/grm_edge_row/grm_edge_col/grm_edge_value
        are accepted as a complete alternate schema.
        """
        with np.load(Path(path), allow_pickle=False) as data:
            names = ("sample_ids", "diagonal", "edge_rows", "edge_cols", "edge_values")
            alternate = ("ids", "grm_diagonal", "grm_edge_row", "grm_edge_col", "grm_edge_value")
            selected = names if all(name in data for name in names) else alternate
            if not all(name in data for name in selected):
                raise ValueError("GRM NPZ requires explicit IDs, diagonal and edge arrays")
            return cls(*(data[name] for name in selected))

    def subset(self, sample_ids):
        """Exact K[S,S], preserving the requested per-phenotype ID order."""
        ids = np.asarray(sample_ids, dtype=str)
        if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
            raise ValueError("GRM subset IDs must be a nonempty unique vector")
        lookup = {value: i for i, value in enumerate(self.sample_ids.tolist())}
        if any(value not in lookup for value in ids.tolist()):
            raise ValueError("GRM does not contain every analysis sample")
        source = np.asarray([lookup[value] for value in ids], dtype=np.int64)
        inverse = np.full(len(self.sample_ids), -1, dtype=np.int64)
        inverse[source] = np.arange(len(ids))
        keep = (inverse[self.edge_rows] >= 0) & (inverse[self.edge_cols] >= 0)
        return SparseKinshipData(ids, self.diagonal[source],
            inverse[self.edge_rows[keep]], inverse[self.edge_cols[keep]], self.edge_values[keep])

    def fit_parameters(self):
        return dict(kinship_diagonal=self.diagonal, edge_rows=self.edge_rows,
                    edge_cols=self.edge_cols, edge_values=self.edge_values)


@dataclass
class PreparedPheWASModel:
    """One fitted model plus an optional FP64 SPA state from the same fit."""
    model: object
    spa_model: object | None
    sample_indices: np.ndarray
    metadata: dict


def _fitted_binary_sha256(model):
    """Bind normal/SPA banks to the same exact FP64 fitted state, not filenames."""
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(family=model.family, fit_method=model.fit_method,
        has_kinship=model.has_kinship, iterations=model.iterations,
        converged=model.converged), sort_keys=True).encode())
    digest.update("\0".join(model.sample_ids.tolist()).encode("utf-8"))
    for name in ("x", "scaled_residuals", "fitted_probability", "fixed_effect_covariance",
                 "precision", "precision_x", "projection_left", "coefficients",
                 "phenotype", "working_phenotype"):
        value = getattr(model, name, None)
        digest.update(name.encode())
        if value is None:
            digest.update(b"None")
            continue
        digest.update(json.dumps(list(value.shape)).encode())
        tensors = (value,) if value.layout == torch.strided else (
            value.to_sparse_coo().coalesce().indices(), value.to_sparse_coo().coalesce().values())
        for tensor in tensors:
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(array.dtype.str.encode())
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def infer_family(phenotype):
    """Exact finite 0/1 labels are binomial; other finite values Gaussian."""
    value = np.asarray(phenotype, dtype=np.float64)
    finite = value[np.isfinite(value)]
    if value.ndim != 1 or not len(finite):
        raise ValueError("phenotype must contain finite observations")
    return "binomial" if np.all((finite == 0) | (finite == 1)) else "gaussian"


def continuous_paper_transform(phenotype, covariates, *, device="cpu"):
    """OLS residual -> average-tie Blom RINT -> original phenotype sample SD.

    The fitted Gaussian null must still contain the same fixed covariates and,
    when supplied, the same sparse GRM. This step uses FP64. The rank convention
    is explicit because the paper's prose specifies RINT but not its rank offset.
    """
    y = torch.as_tensor(phenotype, dtype=torch.float64, device=device)
    x = torch.as_tensor(covariates, dtype=torch.float64, device=device)
    if (y.ndim != 1 or len(y) < 3 or x.ndim != 2 or x.shape[0] != len(y)
            or x.shape[1] < 1 or x.shape[1] >= len(y)
            or not bool(torch.isfinite(y).all()) or not bool(torch.isfinite(x).all())
            or int(torch.linalg.matrix_rank(x)) != x.shape[1]):
        raise ValueError("paper transform requires finite aligned full-rank covariates")
    sd = y.std(correction=1)
    if not bool(torch.isfinite(sd)) or float(sd) <= 0:
        raise ValueError("continuous phenotype has zero variance")
    coefficients = torch.linalg.lstsq(x, y, driver="gels").solution
    residual = y - x @ coefficients
    if float(residual.std(correction=1)) <= torch.finfo(y.dtype).eps * max(1., float(sd)):
        raise ValueError("continuous phenotype has zero residual variance")
    transformed = rank_inverse_normal_tensor(residual, device=device) * sd
    return transformed, dict(transform="OLS_residual_RINT_original_sd", rint_offset=0.375,
                            sd_correction=1, original_sd=float(sd))


def prepare_phewas_model(phenotype, sample_ids, covariates, *, family="auto",
                          kinship=None, continuous_transform="paper",
                          device="cpu", association_mode="tf32", use_spa=True,
                          tol=1e-5, maxiter=500, max_block_size=2048,
                          fit_structure_cache=None):
    """Prepare one independent trait with optional bounded shared geometry.

    All leases release on successful return and on validation/fit failure.
    Each trait retains its own complete cases and numerical fitted state.
    """
    with ExitStack() as leases:
        return _prepare_phewas_model(phenotype, sample_ids, covariates,
            family=family, kinship=kinship, continuous_transform=continuous_transform,
            device=device, association_mode=association_mode, use_spa=use_spa,
            tol=tol, maxiter=maxiter, max_block_size=max_block_size,
            fit_structure_cache=fit_structure_cache, _structure_leases=leases)


def _prepare_phewas_model(phenotype, sample_ids, covariates, *, family="auto",
                          kinship=None, continuous_transform="paper",
                          device="cpu", association_mode="tf32", use_spa=True,
                          tol=1e-5, maxiter=500, max_block_size=2048,
                          fit_structure_cache=None, _structure_leases=None):
    """Prepare one independent trait, dropping only that trait's missing rows.

    ``covariates`` includes an explicit intercept. The caller supplies a shared
    ID axis; ``sample_indices`` maps this trait's complete cases to that axis.
    Covariate missingness also excludes a row. Binary cases/controls are never
    imputed. An explicit GRM is subset by ID and used for both families.

    ``model`` is the normal association state in ``association_mode``; the
    optional ``spa_model`` retains FP64 from the identical binary fit. The
    consumer must recalculate binary P<0.05 with that SPA state, rather than
    treating normal-only output as complete. No fitted arrays go into metadata.
    """
    mode = validate_mode(association_mode)
    y = np.asarray(phenotype, dtype=np.float64)
    ids = np.asarray(sample_ids, dtype=str)
    x = np.asarray(covariates, dtype=np.float64)
    if (y.ndim != 1 or ids.shape != y.shape or len(np.unique(ids)) != len(ids)
            or x.ndim != 2 or x.shape[0] != len(y) or x.shape[1] < 1
            or np.any(ids == "")):
        raise ValueError("phenotype, unique IDs and covariates must share one sample axis")
    if np.isinf(y).any() or np.isinf(x).any():
        raise ValueError("phenotype/covariates may be missing but cannot contain infinity")
    rows = np.flatnonzero(np.isfinite(y) & np.isfinite(x).all(axis=1))
    if len(rows) <= x.shape[1] + 1:
        raise ValueError("insufficient per-trait complete cases")
    y, ids, x = y[rows], ids[rows], x[rows]
    if int(np.linalg.matrix_rank(x)) != x.shape[1]:
        raise ValueError("per-trait covariates are rank deficient")
    family = infer_family(y) if family == "auto" else family
    if family == "binary":
        family = "binomial"
    if family not in ("gaussian", "binomial"):
        raise ValueError("family must be auto, gaussian or binomial")
    if continuous_transform not in ("paper", "none", "rint"):
        raise ValueError("continuous_transform must be paper, none or rint")
    if kinship is not None and not isinstance(kinship, SparseKinshipData):
        raise TypeError("kinship must be an explicitly ID-bound SparseKinshipData")
    if kinship is None:
        grm = None
    elif fit_structure_cache is None:
        grm = kinship.subset(ids)
    else:
        grm = _structure_leases.enter_context(
            fit_structure_cache.acquire_kinship_subset(kinship, ids))
    metadata = dict(family=family, n=len(rows), excluded_missing_count=len(phenotype)-len(rows),
                    has_kinship=grm is not None, null_fit_mode="fp64",
                    association_mode=mode, independent_trait=True,
                    shared_complete_case_required=False)
    spa_model = None
    if family == "gaussian":
        if continuous_transform == "paper":
            y_fit, transform = continuous_paper_transform(y, x, device=device)
            metadata.update(transform)
        elif continuous_transform == "rint":
            y_fit = rank_inverse_normal_tensor(y, device=device)
            metadata.update(transform="raw_RINT", rint_offset=0.375)
        else:
            y_fit = y
            metadata["transform"] = "none"
        parameters = {} if grm is None else grm.fit_parameters()
        model = fit_gaussian_null(y_fit, sample_ids=ids, covariates=x,
            device=device, matmul_mode="fp64", tol=tol, maxiter=maxiter,
            max_block_size=max_block_size, variance_normalization="unit",
            fit_structure_cache=fit_structure_cache, **parameters)
        model.set_matmul_mode(mode)
        # RINT keeps the original trait SD, so variance components can be much
        # smaller than an absolute AI tolerance. Fit in unit-variance units and
        # return every model field in the original units; the fixed-effect span,
        # GRM and association probabilities are unchanged by this rescaling.
        metadata.update(variance_normalization="unit",
                        variance_tolerance_scale="phenotype_variance",
                        gaussian_fit_phenotype_scale=float(model.phenotype_scale),
                        variance_components=model.theta.detach().cpu().tolist(),
                        residual_variance_boundary=bool(float(model.theta[0]) == 0),
                        model_fields_restored_to_original_phenotype_units=True)
        metadata["variant_set_method"] = "STAAR-O"
    else:
        if not np.all((y == 0) | (y == 1)) or not 0 < float(y.mean()) < 1:
            raise ValueError("binary phenotype requires both exact 0 and 1 labels")
        if grm is None:
            fitted = fit_logistic_null(y, x, sample_ids=ids, device=device,
                use_spa=use_spa, maxiter=maxiter, tol=min(tol, 1e-8))
            # Store fitted FP64 weights before association mode conversion;
            # recomputing mu*(1-mu) after rounding mu changes the supplied null.
            if getattr(fitted, "precision", None) is None:
                fitted.precision = fitted.fitted_probability * (1-fitted.fitted_probability)
                fitted.precision_x = fitted.xw.T
        else:
            fitted = fit_logistic_mixed_null(y, x, sample_ids=ids, device=device,
                use_spa=use_spa, maxiter=maxiter, tol=tol,
                max_block_size=max_block_size, fit_structure_cache=fit_structure_cache,
                **grm.fit_parameters())
        fitted.null_fit_source_sha256 = _fitted_binary_sha256(fitted)
        spa_model = fitted if use_spa else None
        # Keep the fitted FP64 SPA tensors distinct: a normal TF32 conversion
        # must not round and then upcast the projection used by SPA.
        model = copy.deepcopy(fitted) if use_spa else fitted
        model.use_spa = False
        model.set_matmul_mode(mode)
        metadata.update(transform="none", cases=int(y.sum()), controls=int(len(y)-y.sum()),
            variant_set_method="STAAR-Burden", spa_required=bool(use_spa),
            spa_p_filter=0.05 if use_spa else None, spa_mode="fp64" if use_spa else None,
            normal_only=not use_spa)
        metadata["null_fit_source_sha256"] = fitted.null_fit_source_sha256
        for diagnostic in ("binomial_link_method", "binomial_link_boundary_count",
                           "binomial_link_boundary_peak_count", "binomial_link_eta_abs_peak",
                           "variance_component_peak", "mixed_optimizer",
                           "mixed_ai_recovery_reason", "mixed_ai_iterations",
                           "mixed_brent_iterations", "mixed_optimizer_search_bounds",
                           "mixed_optimizer_search_regions"):
            if hasattr(fitted, diagnostic):
                metadata[diagnostic] = getattr(fitted, diagnostic)
        if hasattr(fitted, "theta"):
            metadata["variance_components"] = fitted.theta.detach().cpu().tolist()
            metadata["boundary_refit"] = fitted.boundary_refit
    metadata.update(iterations=model.iterations, converged=model.converged,
                    fit_method=getattr(model, "fit_method", "Gaussian_AI_REML"))
    return PreparedPheWASModel(model, spa_model, rows, metadata)
