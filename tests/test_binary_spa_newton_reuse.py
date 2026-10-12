"""Shared stable derivatives and active-column work; not a speed benchmark."""
import numpy as np
import pytest
import torch
from scipy.special import expit

from fudan_wgs_toolkit import binary


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA FP64 SPA contractions")
    return request.param


@pytest.mark.parametrize("overflow", [False, True])
def test_shared_derivatives_match_independent_logistic_formula(device, overflow):
    rng = np.random.default_rng(905)
    g = rng.normal(size=(47, 7))
    mu = np.linspace(.015, .82, 47)[:, None]
    q = np.linspace(-6, 6, 7)
    x = np.asarray([0., .01, -.02, .2, -.3, 2., -2.])
    if overflow:
        x *= 1000
    a = np.log(mu)-np.log1p(-mu)+g*x
    expected_first = ((expit(a)-mu)*g).sum(0)-q
    expected_second = (expit(a)*expit(-a)*g*g).sum(0)
    mu, g, q, x = [torch.as_tensor(value, dtype=torch.float64, device=device)
                   for value in (mu, g, q, x)]
    first, second = binary._newton_derivatives(x, mu, g, q)
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    torch.testing.assert_close(first, torch.as_tensor(expected_first, device=device), atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(second, torch.as_tensor(expected_second, device=device), atol=2e-13, rtol=2e-13)
    # Both entry points preserve the same centered CGF in each alternate form.
    for alternate in (False, True):
        torch.testing.assert_close(first, binary._first(x, mu, g, q, alternate=alternate), atol=0, rtol=0)
        torch.testing.assert_close(second, binary._second(x, mu, g, alternate=alternate), atol=0, rtol=0)


def test_solver_removes_converged_and_invalid_columns_from_sample_operations(monkeypatch):
    mu = torch.full((48, 1), .08, dtype=torch.float64)
    g = torch.ones((48, 5), dtype=torch.float64)
    g[:24] = -1
    q = torch.tensor([0., .01, 1., 7., 100.], dtype=torch.float64)
    operations = []
    original = binary._derivatives_from_log_odds

    def record(x, probabilities, columns, score, log_odds):
        operations.append((columns.shape[1], tuple(score.tolist())))
        return original(x, probabilities, columns, score, log_odds)

    monkeypatch.setattr(binary, "_derivatives_from_log_odds", record)
    state = binary._safeguarded_root(mu, g, q, 2**-13, 100)
    assert state.converged.tolist() == [True, True, True, True, False]
    assert operations and max(width for width, _ in operations) == 3
    assert min(width for width, _ in operations) == 1
    assert all(0. not in score and 100. not in score for _, score in operations)
    assert state.iterations[0] == state.iterations[-1] == 0


def test_pure_bisection_uses_adaptive_bracket_and_certifies_root(device):
    mu = torch.full((32, 1), .02, dtype=torch.float64, device=device)
    g = torch.full((32, 1), .001, dtype=torch.float64, device=device)
    q = torch.tensor([.003], dtype=torch.float64, device=device)
    root = binary._bisection(mu, g, q, 2**-13, 128)
    assert torch.isfinite(root).all() and root[0] > 100
    assert binary._first(root, mu, g, q).abs().max() < 1e-10
