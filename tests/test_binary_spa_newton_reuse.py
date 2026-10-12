"""Frozen-solver differentials; synthetic inputs are component checks only."""
import warnings

import pytest
import torch

from fudan_wgs_toolkit import binary


def _frozen_newton(mu, g, q, tol, max_iter):
    """Pre-reuse implementation: keep its masks, counts and alternates exact."""
    current = torch.zeros_like(q)
    first = binary._first(current, mu, g, q)
    update = torch.where(first.abs() > tol,
                         current - first / binary._second(current, mu, g), current)
    counts = torch.zeros_like(q, dtype=torch.int64)
    while True:
        active = (torch.isfinite(update) & ((update-current).abs() > tol)
                  & (binary._first(update, mu, g, q).abs() > tol) & (counts < max_iter))
        if not bool(active.any()):
            break
        counts = counts + active.to(torch.int64)
        current = torch.where(active, update, current)
        candidate = (current - binary._first_finite(current, mu, g, q)
                     / binary._second_finite(current, mu, g))
        update = torch.where(active, candidate, update)
    return torch.where(torch.isfinite(update), update, current), counts


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA FP64 original SPA contractions")
    return request.param


def _inputs(device, extreme=False):
    generator = torch.Generator().manual_seed(905)
    genotype = torch.randn((47, 7), generator=generator, dtype=torch.float64)
    probability = torch.linspace(.015, .82, 47, dtype=torch.float64)
    genotype -= (genotype * (probability * (1-probability))[:, None]).sum(0) / (
        probability * (1-probability)).sum()
    if extreme:
        genotype[:, 0] = 0
        genotype[:, 1] *= 800
    # Different convergence steps, a zero root, both signs and a score outside
    # the cumulant support exercise inactive masks and original fallback.
    score = torch.tensor([0., 3., -2., 6., -5., .01, 1e6], dtype=torch.float64)
    if extreme:
        # The zero genotype has an infinite initial update and must remain
        # inactive while the other independent columns continue Newton.
        score[0] = 1.
    return probability[:, None].to(device), genotype.to(device), score.to(device)


@pytest.mark.parametrize("extreme", [False, True])
@pytest.mark.parametrize("max_iter", [1, 2, 40])
def test_newton_reuses_terms_with_exact_roots_and_counts(device, extreme, max_iter):
    mu, genotype, score = _inputs(device, extreme)
    expected = _frozen_newton(mu, genotype, score, 2**-13, max_iter)
    actual = binary._newton(mu, genotype, score, 2**-13, max_iter)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("overflow", [False, True])
def test_shared_derivatives_retain_each_expression_reduction(device, overflow):
    mu, genotype, score = _inputs(device)
    position = torch.tensor([0., .01, -.02, .2, -.3, 2., -2.], dtype=torch.float64, device=device)
    if overflow:
        position *= 1000
    first, second = binary._newton_derivatives(position, mu, genotype, score)
    torch.testing.assert_close(first, binary._first(position, mu, genotype, score),
                               atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(second, binary._second(position, mu, genotype),
                               atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("max_iter", [1, 40])
@pytest.mark.parametrize("extreme", [False, True])
def test_full_spa_preserves_both_tail_fallback_flags_and_warnings(
        device, max_iter, extreme, monkeypatch):
    mu, genotype, score = _inputs(device, extreme)
    optimized = binary._newton
    monkeypatch.setattr(binary, "_newton", _frozen_newton)
    with warnings.catch_warnings(record=True) as before:
        warnings.simplefilter("always")
        expected = binary.binary_spa(score, genotype, mu[:, 0], max_iter=max_iter)
    monkeypatch.setattr(binary, "_newton", optimized)
    with warnings.catch_warnings(record=True) as after:
        warnings.simplefilter("always")
        actual = binary.binary_spa(score, genotype, mu[:, 0], max_iter=max_iter)
    for field in ("pvalues", "used_bisection", "failed", "iterations", "iteration_limit"):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field),
                                   atol=0, rtol=0, equal_nan=True)
    assert [(str(item.message), item.category) for item in after] == [
        (str(item.message), item.category) for item in before]


def test_reuse_reduces_exponent_calls_without_expanding_sample_columns(monkeypatch):
    mu, genotype, score = _inputs("cpu")
    # Exclude the deliberately impossible score to measure ordinary iterations.
    mu, genotype, score = mu, genotype[:, :6], score[:6]
    original_exp = torch.exp
    calls = []
    def record(value):
        calls.append(tuple(value.shape))
        return original_exp(value)
    monkeypatch.setattr(torch, "exp", record)
    expected = _frozen_newton(mu, genotype, score, 2**-13, 40)
    old_calls = len(calls)
    calls.clear()
    actual = binary._newton(mu, genotype, score, 2**-13, 40)
    assert calls and len(calls) < old_calls
    assert set(calls) == {tuple(genotype.shape)}
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, atol=0, rtol=0)


def test_extreme_spa_tail_retains_existing_zero_probability_policy(device, monkeypatch):
    genotype = torch.ones((4096, 3), dtype=torch.float64, device=device)
    genotype[:2048] = -1
    probability = torch.full((4096,), .05, dtype=torch.float64, device=device)
    score = torch.tensor([500., 1200., 1800.], dtype=torch.float64, device=device)
    optimized = binary._newton
    monkeypatch.setattr(binary, "_newton", _frozen_newton)
    expected = binary.binary_spa(score, genotype, probability, warn_failures=False)
    monkeypatch.setattr(binary, "_newton", optimized)
    actual = binary.binary_spa(score, genotype, probability, warn_failures=False)
    for field in ("pvalues", "used_bisection", "failed", "iterations", "iteration_limit"):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field),
                                   atol=0, rtol=0, equal_nan=True)
    assert (expected.pvalues == 0).any()
    torch.testing.assert_close(-torch.log10(actual.pvalues), -torch.log10(expected.pvalues),
                               atol=0, rtol=0, equal_nan=True)
