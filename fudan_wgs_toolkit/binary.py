"""PyTorch binary saddlepoint score tests.

SPDX-License-Identifier: GPL-3.0-only
Algorithm attribution is recorded in NOTICE.md.
Retains alternate cumulant formulas and failure diagnostics."""
from __future__ import annotations
from dataclasses import dataclass
import math
import warnings
import numpy as np
from contextlib import contextmanager
import torch
from .statistics import _double, _finite, _cct_tensor, _chi1_from_score, annotation_weights
from .precision_audit import explicit_binary_spa_fp64


@dataclass
class SPAResult:
    """Device-resident p values, branch flags and Newton iteration counts."""
    pvalues: torch.Tensor
    used_bisection: torch.Tensor
    failed: torch.Tensor
    iterations: torch.Tensor
    iteration_limit: torch.Tensor


def _cumulant(x, mu, g, *, alternate=False):
    z = g * x[None, :]
    if alternate:
        # Preserve the author's alternate expression, including its centering.
        return torch.log((1 - mu) * torch.exp(-z) + mu).sum(0)
    return (-z * mu + torch.log(1 - mu + mu * torch.exp(z))).sum(0)


def _first(x, mu, g, q, *, alternate=False):
    if alternate:
        e = torch.exp(g * x[None, :])
        return (-mu * g + mu * g * e / (mu * e + 1 - mu)).sum(0) - q
    e = torch.exp(-g * x[None, :])
    return (-mu * g + mu * g / (mu + (1 - mu) * e)).sum(0) - q


def _second(x, mu, g, *, alternate=False):
    if alternate:
        e = torch.exp(g * x[None, :])
        # This is the frozen author's expression, without adding exp(z).
        return (mu * (1 - mu) * g.square() / (mu * e + 1 - mu).square()).sum(0)
    e = torch.exp(-g * x[None, :])
    return (mu * (1 - mu) * g.square() * e / (mu + (1 - mu) * e).square()).sum(0)


def _first_finite(x, mu, g, q):
    value = _first(x, mu, g, q)
    bad = ~torch.isfinite(value)
    if bool(bad.any()):
        value = torch.where(bad, _first(x, mu, g, q, alternate=True), value)
    return value


def _second_finite(x, mu, g):
    value = _second(x, mu, g)
    bad = ~torch.isfinite(value)
    if bool(bad.any()):
        value = torch.where(bad, _second(x, mu, g, alternate=True), value)
    return value


def _newton_derivatives(x, mu, g, q):
    """Share the FP64 exponential without changing either reduction formula."""
    e = torch.exp(-g * x[None, :])
    first = (-mu * g + mu * g / (mu + (1 - mu) * e)).sum(0) - q
    second = (mu * (1 - mu) * g.square() * e / (mu + (1 - mu) * e).square()).sum(0)
    return first, second


def _newton(mu, g, q, tol, max_iter):
    current = torch.zeros_like(q)
    first, second = _newton_derivatives(current, mu, g, q)
    update = torch.where(first.abs() > tol, current - first / second, current)
    counts = torch.zeros_like(q, dtype=torch.int64)
    while True:
        first, second = _newton_derivatives(update, mu, g, q)
        active = (torch.isfinite(update) & ((update-current).abs() > tol)
                  & (first.abs() > tol) & (counts < max_iter))
        if not bool(active.any()):
            break
        counts = counts + active.to(torch.int64)
        current = torch.where(active, update, current)
        # Active current columns equal the update used above. Reuse their
        # first derivative and curvature, preserving the original alternate
        # formulas independently for each nonfinite derivative. Values on
        # inactive columns are discarded by the same update mask below.
        first_bad = active & ~torch.isfinite(first)
        second_bad = active & ~torch.isfinite(second)
        if bool((first_bad | second_bad).any()):
            first = torch.where(first_bad, _first(current, mu, g, q, alternate=True), first)
            second = torch.where(second_bad, _second(current, mu, g, alternate=True), second)
        candidate = current - first / second
        update = torch.where(active, candidate, update)
    # The original max-iteration guard evaluates its w at xhat=0, yielding
    # an undefined normal statistic; it leaves the last iterate unchanged.
    return torch.where(torch.isfinite(update), update, current), counts


def _same_sign(a, b):
    return (a >= 0) == (b >= 0)


def _bisection(mu, g, q, tol, max_iter):
    a, b = torch.full_like(q, -100.), torch.full_like(q, 100.)
    left, right = a.clone(), b.clone()
    fl, fr = _first_finite(left, mu, g, q), _first_finite(right, mu, g, q)
    phi = (1 + math.sqrt(5)) / 2
    for _ in range(max_iter):
        active = ((right-left).abs() > tol) & _same_sign(fl, fr)
        if not bool(active.any()):
            break
        right = torch.where(active, b - (b-a)/phi, right)
        left = torch.where(active, a + (b-a)/phi, left)
        fl, fr = _first_finite(left, mu, g, q), _first_finite(right, mu, g, q)
        b = torch.where(active & (fr < fl), left, b)
        a = torch.where(active & ~(fr < fl), right, a)
    lo, hi = torch.minimum(left, right), torch.maximum(left, right)
    fl, fr = _first_finite(lo, mu, g, q), _first_finite(hi, mu, g, q)
    valid = torch.isfinite(fl) & torch.isfinite(fr) & ~_same_sign(fl, fr)
    root, derivative = torch.zeros_like(q), torch.ones_like(q)
    # Width 200 / tolerance guarantees fewer than 1075 iterations even at
    # double precision's smallest positive tolerance.
    for _ in range(2048):
        active = valid & ((hi-lo).abs() > tol) & (derivative.abs() > tol)
        if not bool(active.any()):
            return root
        root = torch.where(active, (hi+lo)/2, root)
        derivative = _first_finite(root, mu, g, q)
        same = _same_sign(fl, derivative)
        lo = torch.where(active & same, root, lo)
        hi = torch.where(active & ~same, root, hi)
        fl = _first_finite(lo, mu, g, q)
    raise ArithmeticError("binary SPA bisection did not converge")


def _tail(root, mu, g, q, *, lower):
    w = torch.sqrt(2 * (root*q - _cumulant(root, mu, g)))
    bad = ~torch.isfinite(w)
    if bool(bad.any()):
        alternate = torch.sqrt(2 * (root*q - _cumulant(root, mu, g, alternate=True)))
        w = torch.where(bad, alternate, w)
    w = torch.where(root < 0, -w, w)
    ki = root * torch.sqrt(_second(root, mu, g))
    bad = ~torch.isfinite(ki)
    if bool(bad.any()):
        ki = torch.where(bad, root*torch.sqrt(_second(root, mu, g, alternate=True)), ki)
    z = w + torch.log(ki/w)/w
    probability = 0.5 * torch.erfc((-z if lower else z) / math.sqrt(2))
    return torch.where(root.abs() < 1e-4, 1., probability)


def binary_spa(score, projected_genotype, fitted_probability, *, tol=2**-13,
               max_iter=1000, warn_failures=True) -> SPAResult:
    """Original two-sided SPA for U and residualized genotype columns.

    ``projected_genotype`` is G-XXWX_inv@(XW@G), shape [n,p]; fitted
    probabilities have shape [n]. The null state must come from the same
    samples and design. Every numeric operation retains the input device.
    The frozen package uses Newton, then golden-section/bisection on failure.
    Its failure p=1 is reported by ``failed`` and an optional warning.
    """
    g = _double(projected_genotype)
    u, mu = _double(score, device=g.device), _double(fitted_probability, device=g.device)
    if g.ndim != 2 or not g.shape[0] or not g.shape[1] or u.shape != (g.shape[1],) or mu.shape != (g.shape[0],):
        raise ValueError("SPA requires projected_genotype[n,p], score[p], fitted_probability[n]")
    for value,name in ((g,'projected_genotype'),(u,'score'),(mu,'fitted_probability')):
        _finite(value,name)
    if bool(((mu <= 0) | (mu >= 1)).any()):
        raise ValueError("fitted probabilities must lie strictly between zero and one")
    if not math.isfinite(tol) or tol <= 0 or not isinstance(max_iter,int) or max_iter < 1:
        raise ValueError("tol must be positive and max_iter a positive integer")
    mu = mu[:,None]
    used = torch.zeros_like(u,dtype=torch.bool)
    iteration_limit = torch.zeros_like(used)
    tails, counts = [], torch.zeros_like(u,dtype=torch.int64)
    for lower, q in ((False,u.abs()),(True,-u.abs())):
        root, iterations = _newton(mu,g,q,tol,max_iter)
        counts += iterations
        iteration_limit |= iterations >= max_iter
        value = _tail(root,mu,g,q,lower=lower)
        retry = ~torch.isfinite(value) | (value == 1)
        if bool(retry.any()):
            alternate = _bisection(mu,g[:,retry],q[retry],tol,max_iter)
            value[retry] = _tail(alternate,mu,g[:,retry],q[retry],lower=lower)
            used |= retry
        tails.append(value)
    failed = ~torch.isfinite(tails[0]) | (tails[0]==1) | ~torch.isfinite(tails[1]) | (tails[1]==1)
    result = torch.where(failed, 1., torch.clamp(tails[0]+tails[1],max=1.))
    if warn_failures and bool(failed.any()):
        warnings.warn(f"original binary SPA failed for {int(failed.sum())} tests; returned p=1",RuntimeWarning,stacklevel=2)
    if warn_failures and bool(iteration_limit.any()):
        warnings.warn(f"original binary SPA reached the Newton iteration limit for {int(iteration_limit.sum())} tests",RuntimeWarning,stacklevel=2)
    return SPAResult(result,used,failed,counts,iteration_limit)


def individual_score_test_spa(genotype, residual, fitted_probability, xw,
                              projection_left, *, tol=2**-13,max_iter=1000,
                              normal_pvalues=None,p_filter_cutoff=.05,
                              return_diagnostics=False,_score=None):
    """Reproduce Individual_Score_Test_SPA, optionally applying its P filter.

    xw is [k,n], projection_left is original XXWX_inv [n,k] (or sparse
    mixed-model equivalent). residual is the appropriate original scaled
    residual for related samples, and y-mu for an ordinary binary model.
    ``normal_pvalues`` explicitly requests recalculation only below cutoff.
    """
    g = _double(genotype)
    r, xw = _double(residual,device=g.device), _double(xw,device=g.device)
    left = _double(projection_left,device=g.device)
    if g.ndim != 2 or r.shape != (g.shape[0],) or xw.ndim != 2 or xw.shape[1] != g.shape[0] or left.shape != (g.shape[0],xw.shape[0]):
        raise ValueError("incompatible genotype, residual, xw or projection_left shapes")
    for value,name in ((g,'genotype'),(r,'residual'),(xw,'xw'),(left,'projection_left')):
        _finite(value,name)
    mask = torch.ones(g.shape[1],dtype=torch.bool,device=g.device)
    p = torch.ones(g.shape[1],dtype=g.dtype,device=g.device)
    if normal_pvalues is not None:
        p = _double(normal_pvalues,device=g.device).clone()
        if p.shape != mask.shape or not bool(torch.isfinite(p).all()) or bool(((p<0)|(p>1)).any()) or not 0 < p_filter_cutoff <= 1:
            raise ValueError("normal_pvalues must be valid p values; cutoff must lie in (0,1]")
        mask = p < p_filter_cutoff
    used, failed = torch.zeros_like(mask), torch.zeros_like(mask)
    limit = torch.zeros_like(mask)
    counts = torch.zeros_like(mask,dtype=torch.int64)
    if bool(mask.any()):
        columns = g[:,mask]
        score = columns.T@r if _score is None else _double(_score,device=g.device)[mask]
        state = binary_spa(score,columns-left@(xw@columns),fitted_probability,tol=tol,max_iter=max_iter)
        p[mask],used[mask],failed[mask],counts[mask] = state.pvalues,state.used_bisection,state.failed,state.iterations
        limit[mask] = state.iteration_limit
    state = SPAResult(p,used,failed,counts,limit)
    return state if return_diagnostics else p


def _spa_combination(values):
    accepted = values[torch.isfinite(values) & (values < 1)]
    # Original wrapper deliberately omits failed ones and NA values. Its
    # sum(...)>0 check also returns one when all accepted values are zero.
    return _cct_tensor(accepted) if accepted.numel() and bool(accepted.sum()>0) else values.new_tensor(1.)


def association_binary_spa(genotype, maf, residual, fitted_probability, xw,
                     projection_left, annotations=None,names=None, *,
                     rare_maf_cutoff=.01,rv_num_cutoff=2,rv_num_cutoff_max=10**9,
                     tol=2**-13,max_iter=1000,spa_p_filter=False,
                     p_filter_cutoff=.05,covariance=None,return_diagnostics=False):
    """WGS_Binary_SPA_sp burden outputs, without an R runtime.

    Genotype is already minor-oriented and mean-imputed. The SPA method
    returns two Beta burden groups and their omnibus WGS-B, as the
    original does. Filtering requires an explicitly supplied score
    covariance from the same fitted null; it is never inferred or replaced.
    """
    g = _double(genotype); f = _double(maf,device=g.device)
    if g.ndim != 2 or f.shape != (g.shape[1],):
        raise ValueError("genotype must have shape [n,p] and maf shape [p]")
    _finite(f,'maf')
    if bool(((f<0)|(f>.5)).any()) or not 0 < rare_maf_cutoff <= .5 or not 1 <= rv_num_cutoff < rv_num_cutoff_max:
        raise ValueError("invalid MAF or variant cutoffs")
    mask = (f>0)&(f<rare_maf_cutoff); count = int(mask.sum())
    if count < rv_num_cutoff or count >= rv_num_cutoff_max:
        raise ValueError("rare-variant count is outside the allowed interval")
    annotation = None if annotations is None else _double(annotations,device=g.device)[mask]
    g,f = g[:,mask],f[mask]
    k = 0 if annotation is None else annotation.shape[1]
    labels = [f'annotation_{i+1}' for i in range(k)] if names is None else list(names)
    if len(labels)!=k or len(set(labels))!=k or any(not isinstance(s,str) or not s for s in labels):
        raise ValueError("names must be distinct nonempty annotation names")
    weights = annotation_weights(f,annotation)[0]
    burdens = g@weights
    normal = None
    if spa_p_filter:
        if covariance is None:
            raise ValueError("SPA filtering requires the fitted score covariance")
        v = _double(covariance,device=g.device)
        if v.shape != (mask.numel(),mask.numel()):
            raise ValueError("covariance must match the original variants")
        v = v[mask][:,mask]
        r = _double(residual,device=g.device)
        scores = weights.T@(g.T@r)
        normal = _chi1_from_score(scores,(weights.T@v@weights).diagonal())
    state = individual_score_test_spa(burdens,residual,fitted_probability,xw,projection_left,
        tol=tol,max_iter=max_iter,normal_pvalues=normal,p_filter_cutoff=p_filter_cutoff,return_diagnostics=True,
        _score=weights.T@(g.T@_double(residual,device=g.device)))
    result = {'num_variant':count,'cMAC':float(g.sum())}
    width = k+1
    for index,beta in enumerate(('1,25','1,1')):
        values = state.pvalues[index*width:(index+1)*width]
        result[f'Burden({beta})'] = float(values[0])
        for name,value in zip(labels,values[1:]):
            result[f'Burden({beta})-{name}'] = float(value)
        result[f'WGS-B({beta})'] = float(_spa_combination(values))
    result['WGS-B'] = float(_spa_combination(state.pvalues))
    return (result,state) if return_diagnostics else result


_BINARY_BURDEN_SEAL = object()


@dataclass(frozen=True)
class PreparedBinaryBurdens:
    """Immutable-use mask/cohort burdens; keep the source alive until reuse ends."""
    source: object
    source_tensor: torch.Tensor
    source_stamp: tuple
    sample_ids: np.ndarray
    maf: torch.Tensor
    annotations: torch.Tensor | None
    names: tuple
    columns: np.ndarray
    weights: torch.Tensor
    burdens: torch.Tensor
    cmac: float
    matmul_mode: str
    _seal: object


def _genotype_stamp(value):
    if isinstance(value, torch.Tensor):
        return (value.data_ptr(), tuple(value.shape), tuple(value.stride()),
                str(value.dtype), str(value.device), value._version)
    value = np.asarray(value)
    return (int(value.__array_interface__["data"][0]), value.shape, value.strides, str(value.dtype))


def prepare_binary_burdens(model, genotype, maf, annotations=None, names=None, *,
                            rare_maf_cutoff=.01, rv_num_cutoff=2,
                            rv_num_cutoff_max=10**9, variant_tile_size=512):
    """Compute the phenotype-independent burden matrix once for one cohort.

    A host genotype stays on the host and is transferred by variant tiles. A
    CUDA genotype already on the model's device uses one selected-mode product.
    Callers must not mutate NumPy source contents while this object is reused;
    Tensor mutations are checked by the storage/version stamp. A new mask,
    sample axis, MAF or annotation weights requires a new prepared object.
    """
    from .tf32 import matmul, validate_mode
    if getattr(model, "family", None) != "binomial" or getattr(model, "n_pheno", 1) != 1:
        raise ValueError("binary burdens require an independent binomial null")
    if type(variant_tile_size) is not int or variant_tile_size < 1:
        raise ValueError("variant_tile_size must be a positive integer")
    mode = validate_mode(getattr(model, "matmul_mode", "fp64"))
    dtype = torch.float32 if mode == "tf32" else torch.float64
    source = torch.as_tensor(genotype)
    f = _double(maf, device=model.device)
    if source.ndim != 2 or source.shape[0] != model.n or f.shape != (source.shape[1],):
        raise ValueError("binary genotype/MAF must align with the fitted model")
    if source.is_cuda and source.device != model.device:
        raise ValueError("CUDA genotype must be on the fitted model device")
    _finite(f, "maf")
    if (bool(((f < 0) | (f > .5)).any()) or not 0 < rare_maf_cutoff <= .5
            or not 1 <= rv_num_cutoff < rv_num_cutoff_max):
        raise ValueError("invalid binary MAF or variant cutoffs")
    selected = (f > 0) & (f < rare_maf_cutoff)
    count = int(selected.sum())
    if count < rv_num_cutoff or count >= rv_num_cutoff_max:
        raise ValueError("rare-variant count is outside the allowed interval")
    columns = torch.nonzero(selected, as_tuple=True)[0].cpu().numpy()
    annotation = None if annotations is None else _double(annotations, device=model.device)
    if annotation is not None:
        if annotation.ndim != 2 or annotation.shape[0] != len(f):
            raise ValueError("annotations must have shape variants-by-annotations")
        annotation = annotation[selected]
    f = f[selected]
    k = 0 if annotation is None else annotation.shape[1]
    labels = [f"annotation_{i+1}" for i in range(k)] if names is None else list(names)
    if len(labels) != k or len(set(labels)) != k or any(not isinstance(s, str) or not s for s in labels):
        raise ValueError("names must be distinct nonempty annotation names")
    weights = annotation_weights(f, annotation, dtype=dtype)[0]
    cmac = torch.zeros((), dtype=torch.float64, device=model.device)
    if source.is_cuda:
        _finite(source, "genotype")
        g = source.to(dtype=dtype) if count == source.shape[1] else source[:, selected].to(dtype=dtype)
        burdens = matmul(g, weights, mode=mode)
        cmac = g.sum(dtype=torch.float64)
    else:
        burdens = torch.zeros((model.n, weights.shape[1]), dtype=dtype, device=model.device)
        for start in range(0, count, variant_tile_size):
            stop = min(start+variant_tile_size, count)
            if count == source.shape[1]:
                piece = source[:, start:stop]
            else:
                piece = source[:, torch.as_tensor(columns[start:stop], dtype=torch.int64)]
            _finite(piece, "genotype")
            block = piece.to(device=model.device, dtype=dtype)
            burdens.add_(matmul(block, weights[start:stop], mode=mode))
            cmac += block.sum(dtype=torch.float64)
    ids = np.asarray(model.sample_ids, dtype=str).copy()
    ids.flags.writeable = False
    columns.flags.writeable = False
    return PreparedBinaryBurdens(genotype, source, _genotype_stamp(genotype), ids, f,
        annotation, tuple(labels), columns, weights, burdens, float(cmac), mode, _BINARY_BURDEN_SEAL)


def _validate_spa_sidecar(model, spa_model):
    if (getattr(spa_model, "family", None) != "binomial"
            or getattr(spa_model, "matmul_mode", None) != "fp64"
            or not getattr(spa_model, "use_spa", False)
            or spa_model.n != model.n or spa_model.device != model.device
            or (spa_model.sample_ids is not model.sample_ids
                and not np.array_equal(spa_model.sample_ids, model.sample_ids))):
        raise ValueError("SPA sidecar must be FP64 with the same fitted sample axis")
    left = getattr(model, "null_fit_source_sha256", None)
    right = getattr(spa_model, "null_fit_source_sha256", None)
    # Retained labels are a useful independent check, but matching labels do
    # not establish that covariates, GRM and fitted projection are identical.
    # Once either side carries a fitted-state binding, require both bindings.
    if left is not None or right is not None:
        if (not isinstance(left, str) or len(left) != 64 or left != right
                or any(ch not in "0123456789abcdef" for ch in left)):
            raise ValueError("binary normal/SPA state requires the same verified fitted-source SHA256")
    if model.phenotype is not None and spa_model.phenotype is not None:
        if not torch.equal(model.phenotype.double(), spa_model.phenotype.double()):
            raise ValueError("SPA sidecar phenotype differs from the normal model")
    else:
        # A thin normal bank may omit labels, but only a verified common fitted
        # source permits that. Missing labels alone never bypass same-fit checks.
        if (not isinstance(left, str) or len(left) != 64 or left != right
                or any(ch not in "0123456789abcdef" for ch in left)):
            raise ValueError("thin binary normal/SPA state requires the same verified fitted-source SHA256")


def _spa_requested(model, spa_model, spa_acquire):
    if spa_acquire is not None:
        if spa_model is not None or not callable(spa_acquire):
            raise ValueError("provide one SPA sidecar or a callable spa_acquire context factory")
        binding = getattr(model, "null_fit_source_sha256", None)
        if (not isinstance(binding, str) or len(binding) != 64
                or any(ch not in "0123456789abcdef" for ch in binding)):
            raise ValueError("lazy SPA requires the verified normal fitted-source SHA256")
    return spa_model is not None or spa_acquire is not None


@contextmanager
def _spa_context(model, spa_model, spa_acquire, selected):
    if spa_acquire is not None:
        # Unselected formal tests retain nominal P; a verified repository can
        # avoid loading/rebuilding an unused FP64 projection entirely.
        if not bool(selected.any()):
            yield None
        else:
            with spa_acquire() as sidecar:
                _validate_spa_sidecar(model, sidecar)
                with explicit_binary_spa_fp64():
                    yield sidecar
    else:
        if spa_model is not None:
            _validate_spa_sidecar(model, spa_model)
        if bool(selected.any()):
            with explicit_binary_spa_fp64():
                yield spa_model
        else:
            yield spa_model


def staar_binary_phewas(model, genotype, maf, annotations=None, names=None, *,
                         spa_model=None, spa_acquire=None, rare_maf_cutoff=.01, rv_num_cutoff=2,
                         rv_num_cutoff_max=10**9, p_filter_cutoff=.05,
                         tol=2**-13, max_iter=1000, spa_variant_tile_size=512,
                         prepared_burdens=None, variant_tile_size=512,
                         return_diagnostics=False):
    """Paper binary STAAR-Burden, with small covariance and selective SPA.

    A minor-oriented/imputed [N,M] genotype is multiplied by the two Beta
    families and each annotation weight. Only the resulting 2*(K+1) burdens
    enter the null projection, so no M-by-M covariance/eigendecomposition is
    needed. Native genotype products use the normal model's selected TF32 mode.

    If supplied, ``spa_model`` is the FP64 state from the *same* null fit. Only
    nominal burden P strictly below ``p_filter_cutoff`` are recalculated with
    the established binary SPA. Their original burdens and weights are rebuilt
    in FP64; an upcast rounded TF32 burden is not used for the SPA projection.
    The case-control-imbalance paper path requires that sidecar. Omitting it is
    explicitly normal-only and is reported in diagnostics. ``spa_acquire`` is
    an alternative context factory for a verified fitted repository; it is
    entered only if at least one nominal probability needs SPA.
    """
    from .tf32 import validate_mode
    if getattr(model, "family", None) != "binomial" or getattr(model, "n_pheno", 1) != 1:
        raise ValueError("binary PheWAS requires an independent binomial null")
    mode = validate_mode(getattr(model, "matmul_mode", "fp64"))
    use_spa = _spa_requested(model, spa_model, spa_acquire)
    if (not 0 < p_filter_cutoff <= 1
            or type(spa_variant_tile_size) is not int or spa_variant_tile_size < 1):
        raise ValueError("invalid binary MAF, variant or SPA cutoffs")
    prepared = prepared_burdens
    if prepared is None:
        prepared = prepare_binary_burdens(model, genotype, maf, annotations, names,
            rare_maf_cutoff=rare_maf_cutoff, rv_num_cutoff=rv_num_cutoff,
            rv_num_cutoff_max=rv_num_cutoff_max, variant_tile_size=variant_tile_size)
    else:
        if (not isinstance(prepared, PreparedBinaryBurdens) or prepared._seal is not _BINARY_BURDEN_SEAL
                or prepared.source is not genotype or prepared.source_stamp != _genotype_stamp(genotype)
                or not np.array_equal(prepared.sample_ids, model.sample_ids)
                or prepared.matmul_mode != mode or prepared.burdens.device != model.device):
            raise ValueError("prepared binary burdens have a different source, cohort or execution mode")
        full_f = _double(maf, device=model.device)
        selected = (full_f > 0) & (full_f < rare_maf_cutoff)
        index = torch.nonzero(selected, as_tuple=True)[0].cpu().numpy()
        given_a = None if annotations is None else _double(annotations, device=model.device)[selected]
        labels = tuple(f"annotation_{i+1}" for i in range(0 if given_a is None else given_a.shape[1])) if names is None else tuple(names)
        if (not np.array_equal(index, prepared.columns) or not torch.equal(full_f[selected], prepared.maf)
                or (given_a is None) != (prepared.annotations is None) or labels != prepared.names
                or (given_a is not None and not torch.equal(given_a, prepared.annotations))):
            raise ValueError("prepared binary burdens have different selected MAF/annotations")
    g, f, annotation = prepared.source_tensor, prepared.maf, prepared.annotations
    labels, count, weights, burdens = prepared.names, len(f), prepared.weights, prepared.burdens
    if count < rv_num_cutoff or count >= rv_num_cutoff_max:
        raise ValueError("rare-variant count is outside the allowed interval")
    k = len(labels)
    score, covariance = model.score_covariance(burdens)
    variance = covariance.diagonal()
    nominal = _chi1_from_score(score, variance)
    pvalues = nominal.clone()
    choose = nominal < p_filter_cutoff if use_spa else torch.zeros_like(nominal, dtype=torch.bool)
    flags = SPAResult(pvalues, torch.zeros_like(choose), torch.zeros_like(choose),
                      torch.zeros_like(choose, dtype=torch.int64), torch.zeros_like(choose))
    with _spa_context(model, spa_model, spa_acquire, choose) as spa_model:
        if bool(choose.any()):
            index = torch.nonzero(choose, as_tuple=True)[0]
            exact_weights = annotation_weights(f, annotation, dtype=torch.float64)[0][:, index]
            exact_burdens = torch.zeros((g.shape[0], len(index)), dtype=torch.float64, device=model.device)
            # Recompute selected columns in FP64 without doubling a long
            # samples-by-variants slab. Only one variant tile is converted.
            for start in range(0, count, spa_variant_tile_size):
                stop = min(start+spa_variant_tile_size, count)
                if count == g.shape[1]:
                    block = g[:, start:stop]
                else:
                    columns = torch.tensor(prepared.columns[start:stop], dtype=torch.int64, device=g.device)
                    block = g[:, columns]
                exact_burdens.add_(block.to(device=model.device, dtype=torch.float64) @ exact_weights[start:stop])
            exact_scores = exact_burdens.T @ spa_model.scaled_residuals
            projected = exact_burdens - spa_model.projection_left @ (spa_model.xw @ exact_burdens)
            corrected = binary_spa(exact_scores, projected, spa_model.fitted_probability,
                                   tol=tol, max_iter=max_iter)
            pvalues[index] = corrected.pvalues
            flags.used_bisection[index] = corrected.used_bisection
            flags.failed[index] = corrected.failed
            flags.iterations[index] = corrected.iterations
            flags.iteration_limit[index] = corrected.iteration_limit
            del exact_weights, exact_burdens, projected
    result = dict(num_variant=count, cMAC=prepared.cmac)
    width = k + 1
    for group, beta in enumerate(("1,25", "1,1")):
        values = pvalues[group*width:(group+1)*width]
        result[f"Burden({beta})"] = float(values[0])
        for name, value in zip(labels, values[1:]):
            result[f"Burden({beta})-{name}"] = float(value)
        result[f"STAAR-B({beta})"] = float(_spa_combination(values))
    result["STAAR-B"] = float(_spa_combination(pvalues))
    diagnostics = dict(method="binary_burden_nominal_then_selective_SPA",
        normal_only=not use_spa, num_burdens=len(pvalues),
        covariance_shape=list(covariance.shape), variant_covariance_constructed=False,
        nominal_pvalues=nominal, pvalues=pvalues, spa_selected=choose,
        spa=flags, matmul_mode=mode, spa_mode="fp64" if use_spa else None)
    return (result, diagnostics) if return_diagnostics else result


def binary_single_phewas(model, genotype, *, spa_model=None, spa_acquire=None,
                         normal_score=None, normal_variance=None,
                         normal_pvalues=None, p_filter_cutoff=.05,
                         spa_variant_tile_size=512, tol=2**-13, max_iter=1000):
    """Normal single-variant output with selective FP64 SPA replacing final P.

    The shared-reader callback may pass its already calculated normal score,
    variance and probability. Only selected P<0.05 columns are copied to the
    FP64 SPA state, at most 512 at once by default. Score/effect estimation stays
    normal-model based. ``pvalues`` and ``pvalue_log10`` are the corrected final
    columns; ``normal_pvalues`` and ``normal_pvalue_log10`` permit audit.

    Both log columns are -log10(P), consistent with the native Single output.
    A normal P that underflows to zero retains its score-based finite log tail.
    SPA P=0 has no separately saved log tail in the mature solver, so it is
    marked unverified rather than assigned a finite invented value. A verified
    repository may supply a lazy ``spa_acquire`` context factory instead of a
    sidecar; all-unselected blocks never enter that factory.
    """
    if getattr(model, "family", None) != "binomial" or getattr(model, "n_pheno", 1) != 1:
        raise ValueError("binary Single requires an independent binomial null")
    use_spa = _spa_requested(model, spa_model, spa_acquire)
    if (not 0 < p_filter_cutoff <= 1 or type(spa_variant_tile_size) is not int
            or not 1 <= spa_variant_tile_size <= 512):
        raise ValueError("binary Single SPA tile must be in [1,512] and cutoff in (0,1]")
    source = torch.as_tensor(genotype)
    if source.ndim != 2 or source.shape[0] != model.n:
        raise ValueError("binary Single genotype must align to the model")
    if (normal_score is None) != (normal_variance is None):
        raise ValueError("normal score and variance must be supplied together")
    if normal_score is None:
        dtype = torch.float32 if model.matmul_mode == "tf32" else torch.float64
        score, variance = model.individual_score_variance(source.to(device=model.device, dtype=dtype))
    else:
        score = torch.as_tensor(normal_score, device=model.device)
        variance = torch.as_tensor(normal_variance, device=model.device)
    if (score.shape != (source.shape[1],) or variance.shape != score.shape
            or not bool(torch.isfinite(score).all()) or not bool(torch.isfinite(variance).all())
            or bool((variance <= 0).any())):
        raise ValueError("binary Single normal statistics must be finite with positive variance")
    nominal = _chi1_from_score(score, variance)
    if normal_pvalues is not None:
        given = _double(normal_pvalues, device=model.device)
        if given.shape != score.shape or not bool(torch.isfinite(given).all()) or bool(((given < 0) | (given > 1)).any()):
            raise ValueError("normal Single probabilities must be aligned and in [0,1]")
        nominal = given
    # Scores preserve the finite chi-square log tail even if erfc underflows.
    normal_log10 = -(math.log(2.) + torch.special.log_ndtr(-score.double().abs()
                    / torch.sqrt(variance.double()))) / math.log(10.)
    values = nominal.clone()
    corrected_log10 = normal_log10.clone()
    selected = nominal < p_filter_cutoff if use_spa else torch.zeros_like(nominal, dtype=torch.bool)
    flags = SPAResult(values, torch.zeros_like(selected), torch.zeros_like(selected),
        torch.zeros_like(selected, dtype=torch.int64), torch.zeros_like(selected))
    zero_unverified = torch.zeros_like(selected)
    with _spa_context(model, spa_model, spa_acquire, selected) as spa_model:
        indices = torch.nonzero(selected, as_tuple=True)[0].cpu().numpy()
        for start in range(0, len(indices), spa_variant_tile_size):
            wanted = indices[start:start+spa_variant_tile_size]
            columns = torch.as_tensor(wanted, dtype=torch.int64, device=source.device)
            block = source[:, columns].to(device=model.device, dtype=torch.float64)
            _finite(block, "selected SPA genotype")
            state = individual_score_test_spa(block, spa_model.scaled_residuals,
                spa_model.fitted_probability, spa_model.xw, spa_model.projection_left,
                tol=tol, max_iter=max_iter, return_diagnostics=True)
            target = torch.as_tensor(wanted, dtype=torch.int64, device=model.device)
            values[target] = state.pvalues
            corrected_log10[target] = -torch.log10(state.pvalues)
            zero_unverified[target] = state.pvalues == 0
            flags.used_bisection[target] = state.used_bisection
            flags.failed[target] = state.failed
            flags.iterations[target] = state.iterations
            flags.iteration_limit[target] = state.iteration_limit
    return dict(score=score, variance=variance, normal_pvalues=nominal,
        normal_pvalue_log10=normal_log10, pvalues=values,
        pvalue_log10=corrected_log10, pvalue_log=corrected_log10*math.log(10.),
        spa_selected=selected, spa=flags, spa_zero_log_unverified=zero_unverified,
        normal_only=not use_spa, spa_mode="fp64" if use_spa else None)
