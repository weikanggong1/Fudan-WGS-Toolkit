"""Gaussian WGS block-sparse REML: native FP32 or explicit FP64 controls.

SPDX-License-Identifier: GPL-3.0-only
AI initialization, steps, boundary refits and stopping rule follow GMMAT
R/glmmkin.R, https://github.com/hanchenphd/GMMAT.
"""
from __future__ import annotations
from ._cuda_memory import allocator_bytes

from dataclasses import dataclass
import numpy as np
import torch
from . import tf32 as _tf32
from .tf32 import matmul, validate_mode
from .rint import rank_inverse_normal_tensor
from .numerics import reference_crossprod, extended_variance, _extended_sum_pair
from .tensor_validation import tensor_all_finite, finite_check_scratch_bytes


def _guard_constant_precision_cache_allocation(sources, device):
    """Admit a first normalized state before cloning or arithmetic allocation.

    Existing model/cache tensors stay in the live allocator count. The new
    state owns the three FP32 operands and one scalar; finite checks and the
    three full Boolean zero-check operands run sequentially, so reserve their
    maximum scratch requirement. CPU component calculations have no CUDA cap.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return
    persistent = sum(value.numel() * value.element_size() for value in sources) + sources[0].element_size()
    scratch = max(max(finite_check_scratch_bytes(value.shape, element_size=value.element_size()),
                      3 * value.numel() + 2) for value in sources)
    required = persistent + scratch
    configuration = _tf32.execution_metadata()["tf32_memory_guard"]
    limit = configuration["process_allocated_limit_bytes"]
    reserve = configuration["live_cuda_reserve_bytes"]
    allocated, reserved = allocator_bytes(device)
    backend = _tf32._allocator_backend()
    if backend == "native" and _tf32._owned_cache_sufficient(allocated, reserved, required,
            limit=limit, reserve=reserve):
        return
    with torch.cuda.device(device):
        free, _ = torch.cuda.mem_get_info()
    snapshot = _tf32._product_workspace_availability(allocated=allocated, reserved=reserved,
        free=free, limit=limit, reserve=reserve)
    if required > snapshot["available_bytes"]:
        raise MemoryError(f"Constant precision covariance cache requires estimated {required} new bytes, "
            f"available {snapshot['available_bytes']}; allocated={allocated}, reserved={reserved}, "
            f"CUDAfree={free}, cap_available={snapshot['cap_available_bytes']}, "
            f"live_available={snapshot['live_available_bytes']}, binding={snapshot['binding']}; "
            "preflight is not an allocator guarantee; no precision fallback")


def _r_sum(values):
    """Compensated float64 tree for R's extended-precision sum reductions."""
    high, low = _extended_sum_pair(values)
    return high + low


def rank_inverse_normal(values, *, device="cpu"):
    """REGENIE --apply-rint: average tied ranks and (rank - 3/8) / (N + 1/4).

    Call this after selecting the analysis samples; missing values must have
    already been excluded. Returns float64, without additional rescaling.
    """
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("RINT requires at least two finite phenotype values")
    return rank_inverse_normal_tensor(y, device=device).cpu().numpy()


@dataclass
class KinshipSpectrum:
    """Diagonal entries plus disjoint small relatedness blocks.

    edge_rows/edge_cols are zero-based indices. Provide each off-diagonal
    edge once, or a symmetric COO with the same value in both directions.
    No cutoff is applied to an already sparse relationship matrix.
    """
    eigenvalues: torch.Tensor
    blocks: list[tuple[torch.Tensor, torch.Tensor]]

    @classmethod
    def from_sparse(cls, diagonal, edge_rows=(), edge_cols=(), edge_values=(), *, device="cpu", max_block_size=2048, dtype=torch.float64,
                    eigh_batch_bytes=64 * 1024**2):
        # Held error tracebacks must not own failed partial spectra or a
        # complete solver chunk after the constructor has unwound.
        completed = False
        blocks = None
        try:
            from ._fit_structure_cache import available_cuda_bytes, guard_cuda_workspace
            if type(eigh_batch_bytes) is not int or eigh_batch_bytes <= 0:
                raise ValueError("eigh_batch_bytes must be a positive integer")
            d = np.asarray(diagonal, dtype=np.float64)
            if d.ndim != 1 or not len(d) or not np.isfinite(d).all() or (d < 0).any():
                raise ValueError("kinship diagonal must be a finite nonnegative vector")
            rows = np.asarray(edge_rows, dtype=np.int64)
            cols = np.asarray(edge_cols, dtype=np.int64)
            vals = np.asarray(edge_values, dtype=np.float64)
            if rows.shape != cols.shape or rows.shape != vals.shape or rows.ndim != 1:
                raise ValueError("kinship edge arrays must have equal one-dimensional shapes")
            if ((rows < 0) | (rows >= len(d)) | (cols < 0) | (cols >= len(d))).any() or not np.isfinite(vals).all():
                raise ValueError("kinship edge indices or values are invalid")
            parent = {}
            def find(i):
                parent.setdefault(i, i)
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i
            edges = {}
            for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist()):
                if r == c:
                    raise ValueError("diagonal entries must be supplied separately")
                if v == 0:
                    continue
                key = (min(r, c), max(r, c))
                if key in edges and edges[key] != v:
                    raise ValueError("inconsistent symmetric kinship entries")
                edges[key] = v
                parent[find(r)] = find(c)
            groups = {}
            for i in parent:
                groups.setdefault(find(i), []).append(i)
            itemsize = torch.empty((), dtype=dtype).element_size()
            guard_cuda_workspace(len(d) * itemsize, device, context="GRM spectrum diagonal")
            eigenvalues = torch.as_tensor(d.copy(), dtype=dtype, device=device)
            grouped_edges = {}
            for (r, c), value in edges.items():
                grouped_edges.setdefault(find(r), []).append((r, c, value))
            grouped_sizes = {}
            ordered_groups = list(groups.items())
            for position, (root, group) in enumerate(ordered_groups):
                group.sort()
                if len(group) > max_block_size:
                    raise ValueError("relatedness block exceeds max_block_size; increase the limit only after checking memory")
                grouped_sizes.setdefault(len(group), []).append((position, root, group))
            blocks = [None] * len(ordered_groups)
            for size, members in grouped_sizes.items():
                # Input, output rotations, layout/solver allowance, eigenvalues and
                # indices are bounded per chunk. A large single family retains the
                # existing max_block_size route; never build a population N*N array.
                per_block = 6 * size * size * itemsize + size * (2 * itemsize + 8)
                offset = 0
                while offset < len(members):
                    available = available_cuda_bytes(device)
                    workspace = eigh_batch_bytes if available is None else min(eigh_batch_bytes, available)
                    width = max(1, workspace // per_block)
                    width = min(width, len(members) - offset)
                    guard_cuda_workspace(width * per_block, device, context="batched GRM block eigh")
                    chunk = members[offset:offset + width]
                    matrices = np.zeros((width, size, size), dtype=np.float64)
                    for batch, (_, root, group) in enumerate(chunk):
                        loc = {i: j for j, i in enumerate(group)}
                        matrices[batch, np.arange(size), np.arange(size)] = d[group]
                        for r, c, value in grouped_edges[root]:
                            matrices[batch, loc[r], loc[c]] = matrices[batch, loc[c], loc[r]] = value
                    tensor = torch.as_tensor(matrices, dtype=dtype, device=device)
                    # Preserve the scalar LAPACK/cuSOLVER route for a singleton;
                    # genuine same-size groups use the library batch dimension.
                    eigen, rotation = torch.linalg.eigh(tensor[0] if width == 1 else tensor)
                    if width == 1:
                        eigen, rotation = eigen.unsqueeze(0), rotation.unsqueeze(0)
                    if float(eigen.min()) < -1e-10:
                        raise ValueError("kinship matrix is not positive semidefinite")
                    for batch, (position, _, group) in enumerate(chunk):
                        idx = torch.as_tensor(group, dtype=torch.int64, device=device)
                        eigenvalues[idx] = eigen[batch].clamp_min(0)
                        # All slices in this chunk share only the batched rotation
                        # allocation; each disjoint block retains its original axis.
                        blocks[position] = (idx, rotation[batch])
                    del tensor, matrices, eigen, rotation
                    offset += width
            result = cls(eigenvalues, blocks)
            completed = True
            return result
        finally:
            if not completed and blocks is not None:
                blocks.clear()
            # Successful return owns the same tensors through result; setting
            # local references to None never mutates the returned spectrum.
            diagonal = edge_rows = edge_cols = edge_values = None
            d = rows = cols = vals = eigenvalues = blocks = None
            matrices = tensor = eigen = rotation = idx = result = None
            parent = edges = groups = grouped_edges = grouped_sizes = None
            ordered_groups = members = chunk = group = loc = find = None

    def rotate(self, values, *, inverse=False, matmul_mode="fp64"):
        dtype = torch.float32 if validate_mode(matmul_mode) == "tf32" else torch.float64
        value = torch.as_tensor(values, dtype=dtype, device=self.eigenvalues.device)
        if value.shape[0] != len(self.eigenvalues):
            raise ValueError("value rows do not match kinship samples")
        # Rotation callers consume this tensor without mutating it.
        if not self.blocks:
            return value
        result = value.clone()
        for idx, rotation in self.blocks:
            result[idx] = matmul((rotation if inverse else rotation.T).to(dtype=dtype), value[idx], mode=matmul_mode)
        return result


@dataclass
class GaussianNullModel:
    """Single Gaussian phenotype with fitted sparse mixed-model covariance."""
    sample_ids: np.ndarray
    x: torch.Tensor
    scaled_residuals: torch.Tensor
    coefficients: torch.Tensor
    theta: torch.Tensor
    precision_theta: torch.Tensor
    fixed_effect_covariance: torch.Tensor
    spectrum: KinshipSpectrum
    inverse_variance: torch.Tensor
    precision_x: torch.Tensor
    iterations: int
    converged: bool
    family: str = "gaussian"
    n_pheno: int = 1
    relatedness: bool = True
    use_spa: bool = False
    phenotype: torch.Tensor | None = None
    has_kinship: bool = True
    fitted_values: torch.Tensor | None = None
    working_phenotype: torch.Tensor | None = None
    matmul_mode: str = "fp64"
    variance_normalization: str = "none"
    phenotype_scale: float = 1.0

    @property
    def n(self):
        return len(self.sample_ids)

    @property
    def device(self):
        return self.x.device

    def set_matmul_mode(self, mode):
        """Convert the complete floating fitted state without refitting it.

        Legacy FP64 caches become FP32 for native execution; no precision
        control or arithmetic fallback is inferred from a cache's metadata.
        """
        mode = validate_mode(mode)
        dtype = torch.float32 if mode == "tf32" else torch.float64
        for name in ("x", "scaled_residuals", "coefficients", "theta", "precision_theta",
                     "fixed_effect_covariance", "inverse_variance", "precision_x",
                     "phenotype", "fitted_values", "working_phenotype"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(dtype=dtype))
        self.spectrum.eigenvalues = self.spectrum.eigenvalues.to(dtype=dtype)
        self.spectrum.blocks = [(rows, rotation.to(dtype=dtype)) for rows, rotation in self.spectrum.blocks]
        self.matmul_mode = mode
        self._centered_single_projection_cache = None
        self._constant_precision_covariance_cache = None
        return self

    def _covariance_product_state(self, matmul_mode=None):
        """Reuse an algebraically normalized constant diagonal TF32 precision.

        For actual constant fitted precision s, return phi/s, A/s and s*C;
        the caller restores covariance units by multiplying its final V by s.
        The score and saved fitted tensors remain unchanged, including their
        finite-iteration precision state. Nonconstant, FP64, blocked and other
        model states keep the original operands. Cached tensors are owned live
        CUDA allocations and participate in the normal allocator admission.
        """
        mode = validate_mode(self.matmul_mode if matmul_mode is None else matmul_mode)
        original = (self.inverse_variance, self.precision_x,
                    self.fixed_effect_covariance, None)
        if (mode != "tf32" or self.family != "gaussian" or self.n_pheno != 1
                or self.use_spa or self.spectrum.blocks):
            return original
        sources = original[:3]
        if any(value.dtype != torch.float32 or value.layout != torch.strided
               or value.device != self.device or value.requires_grad
               for value in sources):
            return original
        inverse, precision_x, fixed_cov = sources
        if (inverse.shape != (self.n,) or self.n < 1
                or precision_x.shape != self.x.shape
                or fixed_cov.shape != (self.x.shape[1], self.x.shape[1])):
            return original
        # Holding sources prevents Python identity reuse after replacement.
        # Inference tensors have no version counter: keep their old route.
        try:
            token = tuple((id(value), value._version) for value in sources)
        except RuntimeError:
            return original
        cached = getattr(self, "_constant_precision_covariance_cache", None)
        if cached is not None and cached[0] == token:
            return cached[2]
        self._constant_precision_covariance_cache = None
        del cached
        # Eligibility uses a scalar view; the owned scalar is cloned only
        # after first-cache storage and validation scratch pass admission.
        scale = inverse[0]
        state = original
        if (tensor_all_finite(scale) and bool(scale > 0)
                and torch.equal(inverse, scale.expand_as(inverse))):
            _guard_constant_precision_cache_allocation(sources, self.device)
            scale = scale.clone()
            normalized = (inverse / scale, precision_x / scale, fixed_cov * scale)
            # Extreme scales may overflow or flush nonzero state to zero.
            usable = all(tensor_all_finite(value) for value in normalized)
            if usable:
                usable = all(not bool(((value == 0) & (source != 0)).any())
                             for value, source in zip(normalized, sources))
            if usable:
                state = (*normalized, scale)
        self._constant_precision_covariance_cache = (token, sources, state)
        return state

    def score_covariance_sample_block(self, genotype, *, sample_block_size=8192, matmul_mode=None):
        """TF32 Gaussian score/covariance by streaming sample blocks.

        The host keeps the N-by-M imputed genotype matrix. A K-by-M sample
        block is copied to CUDA, and its score and weighted cross-product are
        accumulated into the complete M-by-M FP32 covariance. Projection
        cross-products are accumulated with the same TF32 products before the
        final dense projection. This route intentionally does not mirror
        upper-triangular blocks and is restricted to a single Gaussian null
        without SPA or kinship blocks.
        """
        mode = validate_mode(self.matmul_mode if matmul_mode is None else matmul_mode)
        if mode != "tf32":
            raise ValueError("sample-block covariance requires native TF32")
        if self.spectrum.blocks or self.n_pheno != 1 or self.use_spa or self.family != "gaussian":
            raise NotImplementedError("sample-block covariance requires a single Gaussian model without SPA or kinship blocks")
        if type(sample_block_size) is not int or sample_block_size < 1:
            raise ValueError("sample_block_size must be a positive integer")
        source = torch.as_tensor(genotype, dtype=torch.float32, device="cpu")
        if source.ndim != 2 or source.shape[0] != self.n:
            raise ValueError("genotype must be a finite samples-by-variants matrix")
        if not bool(torch.isfinite(source).all()):
            raise ValueError("genotype must be finite")
        n, m = map(int, source.shape)
        device = self.device
        inv, precision_x, fixed_cov, covariance_scale = self._covariance_product_state(mode)
        inv = inv.to(dtype=torch.float32, device=device)
        residual = self.scaled_residuals.to(dtype=torch.float32, device=device)
        precision_x = precision_x.to(dtype=torch.float32, device=device)
        fixed_cov = fixed_cov.to(dtype=torch.float32, device=device)
        score = torch.zeros((m,), dtype=torch.float32, device=device)
        covariance = torch.zeros((m, m), dtype=torch.float32, device=device)
        cross = torch.zeros((precision_x.shape[1], m), dtype=torch.float32, device=device)
        for start in range(0, n, sample_block_size):
            stop = min(start + sample_block_size, n)
            block = torch.as_tensor(source[start:stop], dtype=torch.float32, device=device)
            score.add_(matmul(block.T, residual[start:stop], mode="tf32"))
            covariance.add_(matmul(block.T, inv[start:stop, None] * block, mode="tf32"))
            cross.add_(matmul(precision_x[start:stop].T, block, mode="tf32"))
            del block
        projected = matmul(matmul(cross.T, fixed_cov, mode="tf32"), cross, mode="tf32")
        covariance.sub_(projected)
        covariance = (covariance + covariance.T) / 2
        if covariance_scale is not None:
            covariance.mul_(covariance_scale)
        del projected, cross, source, inv, residual, precision_x, fixed_cov
        return score, covariance


    def score_covariance_tiled(self, genotype, *, variant_tile_size=256, matmul_mode=None):
        """TF32 Gaussian covariance using bounded variant tiles.

        genotype remains on host memory. Each N-by-B tile is copied to the
        CUDA device, and only B-by-B products are resident while the complete
        M-by-M covariance is assembled in host FP32 storage. This preserves
        variant order, native TF32 products, and dense covariance/eigenspectrum
        semantics without changing rare-variant selection. This route is
        restricted to a single Gaussian model with diagonal precision.
        """
        mode = validate_mode(self.matmul_mode if matmul_mode is None else matmul_mode)
        if mode != "tf32":
            raise ValueError("tiled covariance requires native TF32")
        if self.spectrum.blocks or self.n_pheno != 1 or self.use_spa or self.family != "gaussian":
            raise NotImplementedError("tiled covariance requires a single Gaussian model without SPA or kinship blocks")
        if type(variant_tile_size) is not int or variant_tile_size < 1:
            raise ValueError("variant_tile_size must be a positive integer")
        source = torch.as_tensor(genotype, dtype=torch.float32, device="cpu")
        if source.ndim != 2 or source.shape[0] != self.n:
            raise ValueError("genotype must be a finite samples-by-variants matrix")
        if not bool(torch.isfinite(source).all()):
            raise ValueError("genotype must be finite")
        n, m = map(int, source.shape)
        if m < 1:
            return (torch.empty((0,), dtype=torch.float32, device=self.device),
                    torch.empty((0, 0), dtype=torch.float32, device=self.device))
        device = self.device
        inv, precision_x, fixed_cov, covariance_scale = self._covariance_product_state(mode)
        inv = inv.to(dtype=torch.float32, device=device)
        residual = self.scaled_residuals.to(dtype=torch.float32, device=device)
        precision_x = precision_x.to(dtype=torch.float32, device=device)
        fixed_cov = fixed_cov.to(dtype=torch.float32, device=device)
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
                         - matmul(matmul(left_cross.T, fixed_cov, mode="tf32"),
                                  right_cross, mode="tf32"))
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
        if covariance_scale is not None:
            covariance.mul_(covariance_scale)
        del covariance_host, source, inv, residual, precision_x, fixed_cov
        return score, covariance


    def score_covariance_cached(self, genotype, *, variant_tile_size=4096,
                                panel_variant_size=None, memory_limit_gib=40,
                                matmul_mode=None, profile=False, compute_score=True):
        """Return score, covariance and a bounded TF32 cache report.

        The host input is an already oriented/imputed FP32 [samples, variants]
        dosage matrix aligned to this model. Output score [variants] and dense
        covariance [variants, variants] remain CUDA FP32. Cache selection is
        automatic: retain original/weighted full genotypes if admitted, or
        stream two panels. Covariance tiles default to 4096 columns; score and
        covariate projection tiles remain 512. ``panel_variant_size`` is None
        or a positive multiple of ``variant_tile_size``. ``memory_limit_gib``
        bounds live allocated storage, including the resident model, and must
        respect live physical memory. ``profile`` adds CUDA stream timings with one final
        synchronization; stage intervals overlap and are not pure kernel time.
        ``compute_score=False`` returns None for score when it was already
        computed by the caller, preserving every covariance product/layout.

        Callers choose this backend explicitly. Gaussian covariance routes
        share constant-precision normalization; Single remains unchanged.
        No CPU, lower-precision or alternate statistical fallback is used.
        """
        mode = validate_mode(self.matmul_mode if matmul_mode is None else matmul_mode)
        from ._cached_covariance import score_covariance_cached
        return score_covariance_cached(
            self, genotype, variant_tile_size=variant_tile_size,
            panel_variant_size=panel_variant_size,
            memory_limit_gib=memory_limit_gib, matmul_mode=mode,
            symmetry="average", profile=profile, compute_score=compute_score)

    def score_covariance(self, genotype, *, reduction="blas", max_workspace_bytes=256 * 1024**2, matmul_mode=None):
        """U=G' scaled.residuals, V=G' Sigma_i G-X projection.

        The block eigensystem avoids materializing an N by N projector.
        Final precision and residuals preserve GMMAT's finite-tolerance
        return convention (precision is from the last pre-update step).
        reduction='reference_sparse' uses increasing sparse row sums for
        diagonal precision, with bounded temporary covariance pair blocks.
        Off-diagonal relatedness blocks require reduction='blas'.
        """
        mode = validate_mode(self.matmul_mode if matmul_mode is None else matmul_mode)
        dtype = torch.float32 if mode == "tf32" else torch.float64
        g = torch.as_tensor(genotype, dtype=dtype, device=self.device)
        if g.ndim != 2 or g.shape[0] != self.n or not tensor_all_finite(g):
            raise ValueError("genotype must be a finite samples-by-variants matrix")
        if mode != "fp64" and reduction == "reference_sparse":
            raise ValueError("reference_sparse is an FP64 control; forced TF32 requires reduction=blas")
        if reduction == "reference_sparse":
            if self.spectrum.blocks:
                raise NotImplementedError("ordered sparse reduction requires diagonal precision")
            from .sparse_numerics import reference_sparse_score_covariance
            return reference_sparse_score_covariance(
                g, self.scaled_residuals, self.inverse_variance, self.precision_x,
                self.fixed_effect_covariance, max_workspace_bytes=max_workspace_bytes)
        if reduction != "blas":
            raise ValueError("reduction must be 'blas' or 'reference_sparse'")
        inverse, precision_x, fixed_cov, covariance_scale = self._covariance_product_state(mode)
        rotated = self.spectrum.rotate(g, matmul_mode=mode)
        cross = matmul(precision_x.to(dtype=dtype).T, g, mode=mode)
        covariance = matmul(rotated.T, inverse.to(dtype=dtype)[:, None] * rotated, mode=mode) - matmul(matmul(cross.T, fixed_cov.to(dtype=dtype), mode=mode), cross, mode=mode)
        covariance = (covariance + covariance.T) / 2
        if covariance_scale is not None:
            covariance.mul_(covariance_scale)
        return matmul(g.T, self.scaled_residuals, mode=mode), covariance


    def _centered_single_projection_state(self):
        """FP32 constant-direction correction for this fitted diagonal model."""
        from .single_projection import centered_projection_state
        return centered_projection_state(
            self, self.inverse_variance, self.precision_x, product=matmul,
            blocked=bool(self.spectrum.blocks))

    def individual_score_variance(self, genotype, *, matmul_mode=None):
        """Single-variant scores and variances without an M by M matrix.

        Genotypes have the same oriented, imputed [samples, variants] layout
        as score_covariance. Storage and elementwise reductions follow the selected FP32/FP64 mode;
        matmul_mode selects explicit TF32 products or the FP64 control. For
        diagonal FP32 precision and an explicit intercept, compute the same P
        quadratic form on centered genotypes and retain its finite-state P1
        correction. Scores always use the original oriented genotypes.
        """
        mode = validate_mode(self.matmul_mode if matmul_mode is None else matmul_mode)
        dtype = torch.float32 if mode == "tf32" else torch.float64
        g = torch.as_tensor(genotype, dtype=dtype, device=self.device)
        if g.ndim != 2 or g.shape[0] != self.n or not tensor_all_finite(g):
            raise ValueError("genotype must be a finite samples-by-variants matrix")
        centered_state = (self._centered_single_projection_state()
                          if mode == "tf32" and not g.requires_grad else None)
        if centered_state is not None:
            score = matmul(g.T, self.scaled_residuals, mode=mode)
            mean = g.mean(dim=0)
            centered = g - mean[None, :]
            weighted = self.inverse_variance[:, None] * centered
            cross = matmul(self.precision_x.T, centered, mode=mode)
            projected = matmul(cross.T, self.fixed_effect_covariance, mode=mode)
            variance = ((centered * weighted).sum(dim=0)
                        - (projected * cross.T).sum(dim=1))
            del weighted, cross, projected
            p_one, p_one_sum = centered_state
            variance += (2 * mean * matmul(centered.T, p_one, mode=mode)
                         + mean.square() * p_one_sum)
            return score, variance
        rotated = self.spectrum.rotate(g, matmul_mode=mode)
        weighted = self.inverse_variance.to(dtype=dtype)[:, None] * rotated
        cross = matmul(self.precision_x.to(dtype=dtype).T, g, mode=mode)
        projected = matmul(cross.T, self.fixed_effect_covariance.to(dtype=dtype), mode=mode)
        variance = (rotated * weighted).sum(dim=0) - (projected * cross.T).sum(dim=1)
        return matmul(g.T, self.scaled_residuals, mode=mode), variance


def fit_gaussian_null(phenotype, *, sample_ids=None, covariates=None, kinship_diagonal=None,
                      edge_rows=(), edge_cols=(), edge_values=(), device="cpu", tol=1e-5,
                      maxiter=500, max_block_size=2048, trace_callback=None, matmul_mode="fp64",
                      variance_normalization="none", fit_structure_cache=None):
    """Fit one independent Gaussian model, optionally leasing exact GRM geometry.

    Cache reuse never includes phenotype-dependent variances, projections or
    convergence state. Null-fit arithmetic and stopping rules stay unchanged.
    """
    parameters = dict(sample_ids=sample_ids, covariates=covariates,
        kinship_diagonal=kinship_diagonal, edge_rows=edge_rows, edge_cols=edge_cols,
        edge_values=edge_values, device=device, tol=tol, maxiter=maxiter,
        max_block_size=max_block_size, trace_callback=trace_callback,
        matmul_mode=matmul_mode, variance_normalization=variance_normalization)
    if fit_structure_cache is None or kinship_diagonal is None:
        return _fit_gaussian_null(phenotype, **parameters)
    ids = np.arange(len(phenotype)).astype(str) if sample_ids is None else np.asarray(sample_ids).astype(str)
    dtype = torch.float32 if validate_mode(matmul_mode) == "tf32" else torch.float64
    with fit_structure_cache.acquire_spectrum(ids, kinship_diagonal, edge_rows,
            edge_cols, edge_values, device=device, dtype=dtype,
            max_block_size=max_block_size) as spectrum:
        return _fit_gaussian_null(phenotype, _kinship_spectrum=spectrum, **parameters)


def _fit_gaussian_null(phenotype, *, sample_ids=None, covariates=None, kinship_diagonal=None,
                      edge_rows=(), edge_cols=(), edge_values=(), device="cpu", tol=1e-5,
                      maxiter=500, max_block_size=2048, trace_callback=None, matmul_mode="fp64",
                      variance_normalization="none", _kinship_spectrum=None):
    """Fit intercept/covariate Gaussian null model using GMMAT AI REML.

    covariates must explicitly contain an intercept if one is wanted. None
    selects an intercept. All rows are already aligned; no implicit missing
    removal, phenotype transformation, or kinship threshold takes place.
    With no kinship, reproduces fit_nullmodel(kins=NULL)'s sparse Gaussian
    object and chi-square association calibration.

    ``variance_normalization="unit"`` fits the same covariance model after
    dividing y by its sample SD, so the AI variance-boundary tolerance is in
    relative units. Every returned fitted tensor retains the original y units.
    The default ``"none"`` preserves the established absolute-unit AI path.
    Trace callbacks explicitly report their units and the phenotype scale.
    """
    validate_mode(matmul_mode)
    def mm(a, b):
        return matmul(a, b, mode=matmul_mode)
    def cp(a, b):
        return reference_crossprod(a, b) if matmul_mode == "fp64" else mm(a.T, b)
    dtype = torch.float32 if matmul_mode == "tf32" else torch.float64
    rsum = _r_sum if matmul_mode == "fp64" else lambda values: values.sum()
    y = torch.as_tensor(phenotype, dtype=dtype, device=device)
    if y.ndim != 1 or not bool(torch.isfinite(y).all()):
        raise ValueError("phenotype must be a finite vector")
    n = len(y)
    if variance_normalization not in ("none", "unit"):
        raise ValueError("variance_normalization must be none or unit")
    original_y = y
    phenotype_scale = 1.0
    if variance_normalization == "unit":
        physical_variance = extended_variance(y) if matmul_mode == "fp64" else y.var(correction=1)
        if not bool(torch.isfinite(physical_variance)) or float(physical_variance) <= 0:
            raise ValueError("phenotype has zero or nonfinite variance")
        phenotype_scale = float(torch.sqrt(physical_variance))
        y = y / phenotype_scale

    def finish(model):
        model.variance_normalization = variance_normalization
        model.phenotype_scale = phenotype_scale
        if variance_normalization == "unit":
            squared_scale = phenotype_scale * phenotype_scale
            # This is a change of fitting coordinates, not a variance floor.
            model.scaled_residuals = model.scaled_residuals / phenotype_scale
            model.coefficients = model.coefficients * phenotype_scale
            model.theta = model.theta * squared_scale
            model.precision_theta = model.precision_theta * squared_scale
            model.fixed_effect_covariance = model.fixed_effect_covariance * squared_scale
            model.inverse_variance = model.inverse_variance / squared_scale
            model.precision_x = model.precision_x / squared_scale
            model.phenotype = original_y
            model.fitted_values = model.fitted_values * phenotype_scale
            model.working_phenotype = model.working_phenotype * phenotype_scale
        return model
    x = torch.ones((n, 1), dtype=y.dtype, device=device) if covariates is None else torch.as_tensor(covariates, dtype=y.dtype, device=device)
    if x.ndim != 2 or x.shape[0] != n or not bool(torch.isfinite(x).all()) or x.shape[1] >= n:
        raise ValueError("covariates must be a finite full-rank samples-by-columns matrix")
    if int(torch.linalg.matrix_rank(x)) != x.shape[1]:
        raise ValueError("covariates are rank deficient")
    ids = np.arange(n).astype(str) if sample_ids is None else np.asarray(sample_ids).astype(str)
    if ids.shape != (n,) or len(set(ids)) != n:
        raise ValueError("sample_ids must be unique and match phenotype rows")
    if tol <= 0 or maxiter < 2:
        raise ValueError("tol must be positive and maxiter at least 2")
    alpha0 = torch.linalg.solve(mm(x.T, x), mm(x.T, y))
    if kinship_diagonal is None:
        residual = y - mm(x, alpha0)
        dispersion = residual.square().sum() / (n - x.shape[1])
        if float(dispersion) <= 0:
            raise ValueError("phenotype has zero residual variance")
        kin = KinshipSpectrum.from_sparse(np.zeros(n), device=device, dtype=dtype)
        inv = torch.ones_like(y) / dispersion
        cov = torch.cholesky_inverse(torch.linalg.cholesky(mm(x.T, inv[:, None] * x)))
        return finish(GaussianNullModel(ids, x, residual / dispersion, alpha0, torch.stack((dispersion, dispersion * 0)),
                                 torch.stack((dispersion, dispersion * 0)), cov, kin, inv, inv[:, None] * x, 0, True, phenotype=y, has_kinship=False, fitted_values=mm(x, alpha0), working_phenotype=y, matmul_mode=matmul_mode))
    if len(kinship_diagonal) != n:
        raise ValueError("kinship diagonal does not match phenotype")
    kin = _kinship_spectrum if _kinship_spectrum is not None else KinshipSpectrum.from_sparse(
        kinship_diagonal, edge_rows, edge_cols, edge_values, device=device,
        max_block_size=max_block_size, dtype=dtype)
    eigen = kin.eigenvalues
    yr = kin.rotate(y, matmul_mode=matmul_mode)
    xr = kin.rotate(x, matmul_mode=matmul_mode)
    mean_diag = rsum(eigen) / n
    if mean_diag <= 0:
        raise ValueError("kinship has zero mean diagonal; omit it for an ordinary model")

    def state(tau):
        variance = tau[0] + tau[1] * eigen
        if bool((variance <= 0).any()):
            raise ValueError("mixed-model covariance is singular")
        inv = torch.reciprocal(torch.sqrt(variance)).square()
        sx = inv[:, None] * xr
        cov = torch.cholesky_inverse(torch.linalg.cholesky(cp(xr, sx)))
        alpha = mm(cov, cp(sx, yr))
        sx_cov = mm(sx, cov)
        py = inv * yr - mm(sx, cp(sx_cov, yr))
        return inv, sx, cov, alpha, py

    total_iterations = 0
    fixed = torch.zeros(2, dtype=torch.bool, device=device)
    converged = False
    for _refit in range(4):
        converged = False
        free = torch.nonzero(~fixed, as_tuple=True)[0]
        tau = torch.zeros(2, dtype=y.dtype, device=device)
        # R cov.c keeps centered products and the final division in extended
        # precision; rounding the squares before summation shifts the AI seed.
        variance_y = extended_variance(y) if matmul_mode == "fp64" else y.var(correction=1)
        tau[free] = variance_y / 2
        tau[1] /= mean_diag
        inv, sx, cov, alpha, py = state(tau)
        # GMMAT's single EM initialization, then average-information steps.
        diag_p = inv - torch.sum(sx * (mm(sx, cov)), dim=1)
        derivative = torch.stack((torch.ones_like(eigen), eigen))
        sx_cov = mm(sx, cov)
        apy_k = eigen * py
        papy_k = inv * apy_k - mm(sx, cp(sx_cov, apy_k))
        scores = torch.stack((rsum(py.square()) - rsum(diag_p),
            rsum(yr * papy_k) - (rsum(inv * eigen) - rsum(sx * (eigen[:, None] * sx_cov)))))
        tau[free] = torch.maximum(tau[free] + tau[free].square() * scores[free] / n, torch.zeros_like(tau[free]))
        alpha_prev = alpha0.clone()
        for iteration in range(1, maxiter + 1):
            total_iterations += 1
            old = tau.clone()
            old_y = yr.clone()
            inv, sx, cov, alpha, py = state(old)
            if len(free):
                sx_cov = mm(sx, cov)
                apy = derivative.T * py[:, None]
                papy = inv[:, None] * apy - mm(sx, cp(sx_cov, apy))
                diag_p = inv - torch.sum(sx * sx_cov, dim=1)
                all_scores = torch.stack((rsum(py.square()-diag_p),
                    rsum(yr*papy[:, 1]) - (rsum(inv*eigen)-rsum(sx*(eigen[:,None]*sx_cov)))))
                ai00 = rsum(py*(inv*py)) - rsum(cp(sx_cov, py)*cp(sx, py))
                ai01 = rsum(py*papy[:, 1])
                ai11 = rsum(py*(eigen*papy[:, 1]))
                all_ai = torch.stack((torch.stack((ai00, ai01)), torch.stack((ai01, ai11))))
                # R passes a column-major AI matrix to DGESV. cuSOLVER uses
                # a different transpose route for row-major input; preserving
                # column-major strides also preserves its scalar rounding.
                selected_ai = all_ai[free][:, free].T.contiguous().T
                delta = torch.linalg.solve(selected_ai, all_scores[free])
                for _step in range(2048):
                    tau[free] = old[free] + delta
                    tau[(tau < tol) & (old < tol)] = 0
                    if not bool((tau < 0).any()):
                        break
                    delta /= 2
                else:
                    raise ArithmeticError("AI step could not remain in the parameter space")
                tau[tau < tol] = 0
            eta_spectral = old_y - old[0] * (inv * old_y - mm(sx, alpha))
            eta = kin.rotate(eta_spectral, inverse=True, matmul_mode=matmul_mode)
            working_y = eta + (y - eta)
            yr = kin.rotate(working_y, matmul_mode=matmul_mode)
            if trace_callback is not None:
                trace_callback({"refit": _refit, "iteration": iteration, "tau_old": old,
                    "tau": tau.clone(), "Y_old": old_y, "Y": yr, "PY": py,
                    "inverse_variance": inv, "cov": cov, "alpha": alpha,
                    "score": all_scores if len(free) else None,
                    "AI": all_ai if len(free) else None, "eta": eta,
                    "variance_normalization": variance_normalization,
                    "phenotype_scale": phenotype_scale,
                    "trace_units": "unit_variance" if variance_normalization == "unit" else "original"})
            change = max(float((abs(alpha - alpha_prev) / (abs(alpha) + abs(alpha_prev) + tol)).max()),
                         float((abs(tau - old) / (abs(tau) + abs(old) + tol)).max())) * 2
            alpha_prev = alpha
            if change < tol:
                converged = iteration < maxiter
                break
            if float(abs(tau).max()) > tol ** -2:
                break
        new_fixed = tau < 1.01 * tol
        if bool((new_fixed == fixed).all()):
            break
        fixed = new_fixed
    else:
        raise ArithmeticError("variance boundary refits did not stabilize")
    if not converged:
        raise ArithmeticError("AI REML did not converge; no implicit alternative optimizer is used")
    # eta=y-old_dispersion*P_old*y; returned residual divides by final tau.
    # At a legitimate sigma=0 boundary with positive-definite tau*K, both
    # residual and sigma vanish. Their continuous limit is P*y, already
    # available from the same fitted precision; no ordinary-model refit or
    # positive variance clamp is needed. state() still rejects singular V.
    scaled = (kin.rotate(py, inverse=True, matmul_mode=matmul_mode)
              if float(tau[0]) == 0 else (y - eta) / tau[0])
    precision_x = kin.rotate(sx, inverse=True, matmul_mode=matmul_mode)
    return finish(GaussianNullModel(ids, x, scaled, alpha, tau, old, cov, kin, inv, precision_x, total_iterations, converged, phenotype=y, fitted_values=eta, working_phenotype=working_y, matmul_mode=matmul_mode))
