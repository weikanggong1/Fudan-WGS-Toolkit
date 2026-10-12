"""Anonymous SPA checks with independent SciPy/discrete oracles.

Component regressions, not a real-data WGS benchmark. The interior oracle
uses the centered Bernoulli CGF and modified-normal tail, solving with SciPy
Brent rather than the production tensor solver.
"""
import itertools
import math
import warnings

import numpy as np
import pytest
import torch
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import norm

from fudan_wgs_toolkit import binary


def interior_oracle(score, genotype, mu):
    """Same scientific expression, independent scalar root implementation."""
    score, g, mu = float(score), np.asarray(genotype), np.asarray(mu)
    minimum = np.where(g > 0, -mu*g, (1-mu)*g).sum()
    maximum = np.where(g > 0, (1-mu)*g, -mu*g).sum()
    log_odds = np.log(mu)-np.log1p(-mu)
    if score == 0:
        return 1.
    assert minimum < score < maximum
    result = 0.
    for lower, q in ((False, abs(score)), (True, -abs(score))):
        if (lower and q < minimum) or (not lower and q > maximum):
            continue
        derivative = lambda t: ((expit(log_odds+g*t)-mu)*g).sum()-q
        endpoint = math.copysign(1/max(abs(g)), q)
        for _ in range(128):
            if derivative(endpoint)*derivative(0) <= 0:
                break
            endpoint *= 2
        t = brentq(derivative, min(0., endpoint), max(0., endpoint), xtol=1e-13, rtol=2e-14)
        k = (np.logaddexp(np.log1p(-mu), np.log(mu)+g*t)-mu*g*t).sum()
        curvature = (expit(log_odds+g*t)*expit(-log_odds-g*t)*g*g).sum()
        w = math.copysign(math.sqrt(2*(t*q-k)), t)
        v = t*math.sqrt(curvature)
        z = w+math.log(v/w)/w
        result += norm.cdf(z) if lower else norm.sf(z)
    return min(result, 1.)


def discrete_two_sided(score, genotype, mu):
    """Enumerate Bernoulli configurations for exact endpoint probability."""
    total = 0.
    for outcome in itertools.product((0., 1.), repeat=len(mu)):
        outcome = np.asarray(outcome)
        observed = np.dot(genotype, outcome-np.asarray(mu))
        if abs(observed) >= abs(score)-1e-14:
            total += np.prod(np.where(outcome == 1, mu, 1-np.asarray(mu)))
    return total


@pytest.mark.parametrize("scale", [1e-12, 1., 1e12])
def test_regular_spa_matches_independent_tail_and_is_scale_invariant(scale):
    rng = np.random.default_rng(717)
    g = rng.normal(size=(256, 6))
    mu = np.linspace(.01, .4, len(g))
    g -= ((mu*(1-mu))[:, None]*g).sum(0)/(mu*(1-mu)).sum()
    q = np.asarray([.1, 2., -4., 7., -12., 20.])
    expected = np.asarray([interior_oracle(score, column, mu) for score, column in zip(q, g.T)])
    actual = binary.binary_spa(q*scale, g*scale, mu)
    assert actual.converged.all() and not actual.failed.any()
    assert not actual.iteration_limit.any()
    torch.testing.assert_close(-actual.pvalues.log10(), torch.as_tensor(-np.log10(expected)), atol=.001, rtol=0)
    assert actual.iterations.max() < 50


def test_reflected_tail_outside_support_is_zero_without_solver_failure():
    g, mu, q = np.ones((12, 1)), np.full(12, .01), np.asarray([3.])
    expected = interior_oracle(q[0], g[:, 0], mu)
    actual = binary.binary_spa(q, g, mu)
    assert actual.converged.all() and not actual.failed.any() and not actual.outside_support.any()
    torch.testing.assert_close(-actual.pvalues.log10(), torch.tensor([-math.log10(expected)], dtype=torch.float64), atol=.001, rtol=0)
    assert 0 < actual.pvalues[0] < .05


def test_low_variance_absolute_tolerance_does_not_certify_a_wrong_finite_tail():
    # A source-independent rare Bernoulli component exposes the old absolute
    # tolerance/alternate-curvature error. This is deliberately outside the
    # nominal P<.05 selection used by the production WGS wrapper.
    rng = np.random.default_rng(146)
    for frequency in (.0001, .001):
        mu = frequency*np.exp(rng.uniform(-.25, .25, 4096))
        raw = (rng.random((4096, 12)) < np.geomspace(.0005, .15, 12)).astype(float)
        weight = mu*(1-mu)
        genotype = raw-(raw*weight[:, None]).sum(0)/weight.sum()
        outcome = rng.binomial(1, mu)
    g = genotype[:, 0]
    score = np.dot(g, outcome-mu)
    variance = np.dot(g*g, weight)
    assert 2*norm.sf(abs(score)/math.sqrt(variance)) > .9
    # The old iterate passed its absolute 2**-13 test despite a substantial
    # root residual. Certify the corrected answer with independent Brent.
    old_iterate = -2340.6620210175115
    old_residual = abs(np.dot(expit(np.log(mu)-np.log1p(-mu)+g*old_iterate)-mu, g)+abs(score))
    assert 1e-5 < old_residual < 2**-13
    expected = interior_oracle(score, g, mu)
    actual = binary.binary_spa([score], g[:, None], mu)
    assert actual.converged.all() and not actual.failed.any()
    assert actual.root_residual[0] < 1e-10
    assert expected == pytest.approx(.04265570739858281, abs=2e-10)
    torch.testing.assert_close(-actual.pvalues.log10(), torch.tensor([-math.log10(expected)], dtype=torch.float64),
                               atol=.001, rtol=0)


@pytest.mark.parametrize("genotype,mu", [([1., -1., 0.], [.2, .2, .5]),
                                         ([1., 1., 1.], [.1, .1, .1])])
def test_support_endpoints_use_exact_discrete_mass(genotype, mu):
    g, mu = np.asarray(genotype), np.asarray(mu)
    maximum = np.where(g > 0, (1-mu)*g, -mu*g).sum()
    expected = discrete_two_sided(maximum, g, mu)
    actual = binary.binary_spa([maximum], g[:, None], mu)
    assert actual.converged.all() and actual.support_boundary.all()
    assert not actual.failed.any() and not actual.iteration_limit.any()
    assert not actual.iterations.any()
    torch.testing.assert_close(actual.pvalues, torch.tensor([expected], dtype=torch.float64), atol=1e-14, rtol=1e-14)


def test_zero_score_is_valid_probability_one_and_does_not_warn():
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        result = binary.binary_spa([0.], [[-1.], [1.], [0.]], [.1, .2, .3])
    assert observed == []
    assert result.converged.all() and not result.failed.any()
    assert not result.iterations.any() and not result.used_bisection.any()
    torch.testing.assert_close(result.pvalues, torch.ones(1, dtype=torch.float64))


def test_actual_score_outside_support_remains_explicit_failure():
    with pytest.warns(RuntimeWarning, match="returned p=1"):
        result = binary.binary_spa([4.], np.ones((3, 1)), [.1, .1, .1])
    assert result.failed.all() and result.outside_support.all()
    assert not result.converged.any() and not result.iterations.any()
    assert torch.isnan(result.root_residual).all()


def test_iteration_budget_cannot_return_a_finite_unconverged_p_as_success():
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        result = binary.binary_spa([3.], np.ones((12, 1)), np.full(12, .01), max_iter=1)
    assert result.iteration_limit.all() and result.failed.all() and not result.converged.any()
    assert result.iterations.tolist() == [1]
    assert result.root_residual[0] > 1e-8
    assert any("iteration limit" in str(item.message) for item in observed)


@pytest.mark.parametrize("x", [-1000., -.1, 0., .1, 1000.])
def test_both_cumulant_forms_match_the_centered_official_expression(x):
    mu = np.asarray([.0001, .2, .5, .999999])[:, None]
    g = np.asarray([[-1., .001], [.2, -2.], [.5, .1], [1., -1.]])
    expected = (np.logaddexp(np.log1p(-mu), np.log(mu)+g*x)-mu*g*x).sum(0)
    mu, g = torch.as_tensor(mu), torch.as_tensor(g)
    for alternate in (False, True):
        actual = binary._cumulant(torch.full((2,), x, dtype=torch.float64), mu, g, alternate=alternate)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, torch.as_tensor(expected), atol=2e-12, rtol=2e-13)


def test_selected_single_propagates_actual_convergence_diagnostics():
    g = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]], dtype=torch.float64)
    mu = torch.full((4,), .2, dtype=torch.float64)
    result = binary.individual_score_test_spa(g, torch.tensor([.1, -.1, .1, -.1]), mu,
        torch.full((1, 4), .16), torch.full((4, 1), 1/.64),
        normal_pvalues=[.01, .6], return_diagnostics=True)
    assert result.converged.tolist() == [True, False]
    assert torch.isfinite(result.root_residual[0]) and torch.isnan(result.root_residual[1])
    assert result.pvalues[1] == .6 and not result.failed.any()


def test_boolean_iteration_limit_is_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        binary.binary_spa([.1], [[-1.], [1.]], [.1, .2], max_iter=True)
