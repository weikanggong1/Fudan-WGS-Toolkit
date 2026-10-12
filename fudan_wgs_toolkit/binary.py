"""PyTorch binary saddlepoint score tests.

SPDX-License-Identifier: GPL-3.0-only
Algorithm attribution is recorded in NOTICE.md.
Uses stable equivalent Bernoulli cumulants and explicit failure diagnostics."""
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
    """Device-resident probabilities and actual solver/support diagnostics."""
    pvalues: torch.Tensor
    used_bisection: torch.Tensor
    failed: torch.Tensor
    iterations: torch.Tensor
    iteration_limit: torch.Tensor
    converged: torch.Tensor | None = None
    root_residual: torch.Tensor | None = None
    support_boundary: torch.Tensor | None = None
    outside_support: torch.Tensor | None = None
    bracket_iterations: torch.Tensor | None = None


@dataclass
class _SPARootResult:
    root: torch.Tensor
    converged: torch.Tensor
    residual: torch.Tensor
    iterations: torch.Tensor
    used_bisection: torch.Tensor
    iteration_limit: torch.Tensor
    bracket_iterations: torch.Tensor


def _unassessed_spa_result(pvalues):
    false = torch.zeros_like(pvalues, dtype=torch.bool)
    counts = torch.zeros_like(pvalues, dtype=torch.int64)
    return SPAResult(pvalues, false.clone(), false.clone(), counts.clone(), false.clone(),
        false.clone(), torch.full_like(pvalues, float('nan')), false.clone(), false.clone(), counts.clone())


def _copy_spa_result(target, index, source):
    for name in ('used_bisection', 'failed', 'iterations', 'iteration_limit', 'converged',
                 'root_residual', 'support_boundary', 'outside_support', 'bracket_iterations'):
        getattr(target, name)[index] = getattr(source, name)


def _score_support(mu, g):
    """Exact extrema of sum(g*(Bernoulli(mu)-mu)), including signed g."""
    lower = torch.where(g > 0, -mu*g, (1-mu)*g).sum(0)
    upper = torch.where(g > 0, (1-mu)*g, -mu*g).sum(0)
    return lower, upper


def _cumulant(x, mu, g, *, alternate=False):
    z = g * x[None, :]
    # Both official forms are the same centered CGF. Factor the larger
    # exponential before logaddexp; do not lose the alternate linear term.
    log_mu, log_control = torch.log(mu), torch.log1p(-mu)
    positive = ((1-mu)*z + torch.logaddexp(log_mu, log_control-z))
    negative = (-mu*z + torch.logaddexp(log_control, log_mu+z))
    ordinary = torch.where(z >= 0, positive, negative)
    # Preserve K(0)=0 and its quadratic leading term without subtracting two
    # nearly identical logs. expm1 only sees a bounded argument.
    small_z = z.clamp(-.5, .5)
    nearby = torch.log1p(mu*torch.expm1(small_z)) - mu*small_z
    return torch.where(z.abs() <= .5, nearby, ordinary).sum(0)


def _derivatives_from_log_odds(x, mu, g, q, log_odds):
    tilted = log_odds + g*x[None, :]
    probability = torch.sigmoid(tilted)
    # sigmoid(-a) retains the tiny control probability when sigmoid(a)
    # rounds to one. Their product is the stable official K'' expression,
    # including the exp(z) numerator missing from the old alternate port.
    variance = probability*torch.sigmoid(-tilted)
    first = ((probability-mu)*g).sum(0) - q
    second = (variance*g.square()).sum(0)
    return first, second


def _first(x, mu, g, q, *, alternate=False):
    return _newton_derivatives(x, mu, g, q)[0]


def _second(x, mu, g, *, alternate=False):
    return _newton_derivatives(x, mu, g, torch.zeros_like(x))[1]


def _first_finite(x, mu, g, q):
    return _first(x, mu, g, q)


def _second_finite(x, mu, g):
    return _second(x, mu, g)


def _newton_derivatives(x, mu, g, q):
    """Stable shared tilted-Bernoulli derivatives, all on the input device."""
    return _derivatives_from_log_odds(x, mu, g, q, torch.log(mu)-torch.log1p(-mu))


def _safeguarded_root(mu, g, q, tol, max_iter, *, newton=True):
    """Bracket the monotone K' and evaluate only still-active columns.

    Residuals certify convergence; a small/NaN Newton step does not. A single
    active-index extraction per round supplies the termination check and
    compacts all subsequent sample operations on device. Fixed +/-100 bounds
    and the non-monotone golden-section search are unnecessary.
    """
    zero = torch.zeros_like(q)
    flags = torch.zeros_like(q, dtype=torch.bool)
    counts = torch.zeros_like(q, dtype=torch.int64)
    result = _SPARootResult(zero.clone(), flags.clone(), torch.full_like(q, float('inf')),
        counts.clone(), flags.clone(), flags.clone(), counts.clone())
    log_odds = torch.log(mu)-torch.log1p(-mu)
    variance = (mu*(1-mu)*g.square()).sum(0)
    lower, upper = _score_support(mu, g)
    valid = (variance > 0) & torch.isfinite(variance) & (q > lower) & (q < upper)
    centered = valid & (q == 0)
    result.converged[centered] = True
    result.residual[centered] = 0
    # An accuracy target scaled to the statistic, independent of burden units.
    residual_tol = min(tol, 1e-10)*torch.maximum(variance.sqrt(), q.abs())
    step = (q/variance).abs().clamp(min=1e-3, max=8.)
    end = torch.where(q > 0, step, -step)
    pending = valid & ~centered
    bracketed = flags.clone()
    for _ in range(128):
        index = torch.nonzero(pending, as_tuple=True)[0]
        if not len(index):
            break
        first, _ = _derivatives_from_log_odds(end[index], mu, g[:, index], q[index], log_odds)
        result.bracket_iterations[index] += 1
        found = torch.isfinite(first) & torch.where(q[index] > 0, first >= 0, first <= 0)
        bracketed[index] = found
        pending[index] = ~found
        end[index] = torch.where(found, end[index], end[index]*2)
    lo, hi = torch.minimum(end, zero), torch.maximum(end, zero)
    current = q/variance
    ordinary = torch.isfinite(current) & (current > lo) & (current < hi)
    current = torch.where(ordinary, current, lo+(hi-lo)/2)
    result.used_bisection = bracketed & ~ordinary
    active = bracketed.clone()
    for _ in range(max_iter):
        index = torch.nonzero(active, as_tuple=True)[0]
        if not len(index):
            break
        first, curvature = _derivatives_from_log_odds(
            current[index], mu, g[:, index], q[index], log_odds)
        result.iterations[index] += 1
        result.root[index] = current[index]
        result.residual[index] = first.abs()
        done = torch.isfinite(first) & (first.abs() <= residual_tol[index])
        result.converged[index] = done
        lo[index] = torch.where(first < 0, current[index], lo[index])
        hi[index] = torch.where(first >= 0, current[index], hi[index])
        candidate = current[index]-first/curvature if newton else torch.full_like(first, float('nan'))
        take_newton = (torch.isfinite(candidate) & (curvature > 0)
                       & (candidate > lo[index]) & (candidate < hi[index]))
        midpoint = lo[index]+(hi[index]-lo[index])/2
        update = torch.where(take_newton, candidate, midpoint)
        result.used_bisection[index] |= ~take_newton & ~done
        # Floating point stagnation is a failure unless the residual passed.
        continues = ~done & torch.isfinite(first) & (update != current[index])
        active[index] = continues
        current[index] = update
    result.iteration_limit = active
    return result


def _newton(mu, g, q, tol, max_iter):
    state = _safeguarded_root(mu, g, q, tol, max_iter)
    return state.root, state.iterations


def _same_sign(a, b):
    return (a >= 0) == (b >= 0)


def _bisection(mu, g, q, tol, max_iter):
    state = _safeguarded_root(mu, g, q, tol, max_iter, newton=False)
    return torch.where(state.converged, state.root, float('nan'))


def _tail(root, mu, g, q, *, lower):
    cumulant = _cumulant(root, mu, g)
    action = root*q-cumulant
    roundoff = 32*torch.finfo(g.dtype).eps*(1+(root*q).abs()+cumulant.abs())
    action = torch.where((action < 0) & (action >= -roundoff), 0., action)
    w = torch.sign(root)*torch.sqrt(2*action)
    curvature = _second(root, mu, g)
    log_ratio = torch.log(root.abs())+.5*torch.log(curvature)-torch.log(w.abs())
    z = w+log_ratio/w
    probability = 0.5 * torch.erfc((-z if lower else z) / math.sqrt(2))
    variance = (mu*(1-mu)*g.square()).sum(0)
    standardized = q/variance.sqrt()
    central = standardized.abs() < 1e-4
    normal = .5*torch.erfc((-standardized if lower else standardized)/math.sqrt(2))
    return torch.where(central & (variance > 0), normal, probability)


def binary_spa(score, projected_genotype, fitted_probability, *, tol=2**-13,
               max_iter=1000, warn_failures=True) -> SPAResult:
    """Stable two-sided SPA for U and residualized genotype columns.

    ``projected_genotype`` is G-XXWX_inv@(XW@G), shape [n,p]; fitted
    probabilities have shape [n]. The null state must come from the same
    samples and design. Every numeric operation retains the input device.
    Centered Bernoulli cumulants and the original modified-normal tail formula
    are evaluated in FP64. Bracketed Newton only reports residual-certified
    roots. Reflected tails outside support are zero; support endpoints use
    their exact discrete mass, a boundary extension to the continuous SPA.
    A zero score with positive variance is valid p=1. Genuine failures retain
    p=1, ``failed`` and an optional warning, without using P itself as a flag.
    """
    g = _double(projected_genotype)
    u, mu = _double(score, device=g.device), _double(fitted_probability, device=g.device)
    if g.ndim != 2 or not g.shape[0] or not g.shape[1] or u.shape != (g.shape[1],) or mu.shape != (g.shape[0],):
        raise ValueError("SPA requires projected_genotype[n,p], score[p], fitted_probability[n]")
    for value,name in ((g,'projected_genotype'),(u,'score'),(mu,'fitted_probability')):
        _finite(value,name)
    if bool(((mu <= 0) | (mu >= 1)).any()):
        raise ValueError("fitted probabilities must lie strictly between zero and one")
    if not math.isfinite(tol) or tol <= 0 or type(max_iter) is not int or max_iter < 1:
        raise ValueError("tol must be positive and max_iter a positive integer")
    # t*G and t*q are invariant under a common positive column rescaling.
    # Avoid dependence on annotation/burden units and roots beyond +/-100.
    scale = torch.maximum(g.amax(0).abs(), g.amin(0).abs())
    scale = torch.where(scale > 0, scale, 1.)
    g, u = g/scale, u/scale
    mu = mu[:,None]
    lower_bound, upper_bound = _score_support(mu, g)
    support_tol = 16*torch.finfo(g.dtype).eps*g.abs().sum(0)
    variance = (mu*(1-mu)*g.square()).sum(0)
    outside = (u < lower_bound-support_tol) | (u > upper_bound+support_tol)
    valid = (variance > 0) & torch.isfinite(variance) & ~outside
    used = torch.zeros_like(u,dtype=torch.bool)
    iteration_limit = torch.zeros_like(used)
    converged = valid.clone()
    boundary = torch.zeros_like(used)
    residual = torch.where(valid, torch.zeros_like(u), float('nan'))
    bracket_counts = torch.zeros_like(u,dtype=torch.int64)
    tails, counts = [], torch.zeros_like(u,dtype=torch.int64)
    for lower, q in ((False,u.abs()),(True,-u.abs())):
        bound = lower_bound if lower else upper_bound
        endpoint = valid & ((q-bound).abs() <= support_tol)
        exterior = valid & ((q < bound-support_tol) if lower else (q > bound+support_tol))
        central = valid & (q == 0)
        value = torch.zeros_like(u)
        value[central] = .5
        # Enumerating only the extremal configuration is exact and avoids
        # the endpoint's infinite saddlepoint. Zero coefficients impose no Y.
        index = torch.nonzero(endpoint, as_tuple=True)[0]
        if len(index):
            columns = g[:,index]
            cases = (columns < 0) if lower else (columns > 0)
            log_mass = torch.where(columns == 0, 0.,
                torch.where(cases, torch.log(mu), torch.log1p(-mu))).sum(0)
            value[index] = log_mass.exp()
        boundary |= endpoint
        ordinary = valid & ~endpoint & ~exterior & ~central
        index = torch.nonzero(ordinary, as_tuple=True)[0]
        if len(index):
            state = _safeguarded_root(mu,g[:,index],q[index],tol,max_iter)
            probability = _tail(state.root,mu,g[:,index],q[index],lower=lower)
            tail_ok = state.converged & torch.isfinite(probability) & (probability >= 0) & (probability <= 1)
            value[index] = probability
            converged[index] &= tail_ok
            used[index] |= state.used_bisection
            counts[index] += state.iterations
            iteration_limit[index] |= state.iteration_limit
            residual[index] = torch.maximum(residual[index],state.residual)
            bracket_counts[index] += state.bracket_iterations
        tails.append(value)
    failed = ~converged
    result = torch.where(failed, 1., torch.clamp(tails[0]+tails[1],max=1.))
    if warn_failures and bool(failed.any()):
        warnings.warn(f"binary SPA failed for {int(failed.sum())} tests; returned p=1",RuntimeWarning,stacklevel=2)
    if warn_failures and bool(iteration_limit.any()):
        warnings.warn(f"binary SPA reached the solver iteration limit for {int(iteration_limit.sum())} tests",RuntimeWarning,stacklevel=2)
    return SPAResult(result,used,failed,counts,iteration_limit,converged,residual,boundary,
                     outside,bracket_counts)


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
    diagnostics = _unassessed_spa_result(p)
    if bool(mask.any()):
        columns = g[:,mask]
        score = columns.T@r if _score is None else _double(_score,device=g.device)[mask]
        state = binary_spa(score,columns-left@(xw@columns),fitted_probability,tol=tol,max_iter=max_iter)
        p[mask] = state.pvalues
        _copy_spa_result(diagnostics,mask,state)
    return diagnostics if return_diagnostics else p


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
    flags = _unassessed_spa_result(pvalues)
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
            _copy_spa_result(flags,index,corrected)
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
    flags = _unassessed_spa_result(values)
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
            _copy_spa_result(flags,target,state)
    return dict(score=score, variance=variance, normal_pvalues=nominal,
        normal_pvalue_log10=normal_log10, pvalues=values,
        pvalue_log10=corrected_log10, pvalue_log=corrected_log10*math.log(10.),
        spa_selected=selected, spa=flags, spa_zero_log_unverified=zero_unverified,
        normal_only=not use_spa, spa_mode="fp64" if use_spa else None)
