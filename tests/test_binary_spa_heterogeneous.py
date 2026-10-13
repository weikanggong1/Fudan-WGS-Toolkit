"""Independent fitted probabilities and bounded tail solver contracts."""
import pytest
import torch

from fudan_wgs_toolkit import binary


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA FP64 SPA')
    return request.param


@pytest.mark.parametrize('combine_tails', [False, True])
def test_heterogeneous_probabilities_match_independent_scalar_tests(device, combine_tails):
    generator = torch.Generator().manual_seed(627)
    g = torch.randn((91, 7), generator=generator, dtype=torch.float64, device='cpu').to(device)
    mu = torch.linspace(.8, 1.2, 91, dtype=torch.float64, device=device)[:, None]
    mu = mu*torch.tensor([.001, .01, .06, .2, .4, .6, .75], dtype=torch.float64, device=device)
    weights = mu*(1-mu)
    g -= (g*weights).sum(0)/weights.sum(0)
    q = torch.tensor([0., .02, -1., 3., -7., 9., -11.], dtype=torch.float64, device=device)
    actual = binary.binary_spa(q, g, mu, combine_tails=combine_tails, warn_failures=False)
    expected = [binary.binary_spa(q[i:i+1], g[:, i:i+1], mu[:, i], warn_failures=False)
                for i in range(g.shape[1])]
    for name in ('pvalues', 'used_bisection', 'failed', 'iterations', 'iteration_limit',
                 'converged', 'root_residual', 'support_boundary', 'outside_support',
                 'bracket_iterations'):
        value = torch.cat([getattr(item, name) for item in expected])
        torch.testing.assert_close(getattr(actual, name), value, rtol=0, atol=2e-12, equal_nan=True)
    torch.testing.assert_close(-actual.pvalues.log10(),
        -torch.cat([item.pvalues for item in expected]).log10(), rtol=0, atol=1e-9)


def test_heterogeneous_mu_and_log_odds_follow_active_column_compaction(monkeypatch):
    n = 48
    mu = torch.tensor([.03, .07, .15, .4, .8], dtype=torch.float64)[None].expand(n, -1)
    g = torch.ones((n, 5), dtype=torch.float64)
    g[:24] = -1
    q = torch.tensor([0., .01, 1., 7., 100.], dtype=torch.float64)
    observed = []
    original = binary._derivatives_from_log_odds
    def derivatives(x, probabilities, columns, score, log_odds):
        assert probabilities.shape == log_odds.shape == columns.shape
        torch.testing.assert_close(log_odds, torch.log(probabilities)-torch.log1p(-probabilities), rtol=0, atol=0)
        observed.append(tuple(probabilities[0].tolist()))
        return original(x, probabilities, columns, score, log_odds)
    monkeypatch.setattr(binary, '_derivatives_from_log_odds', derivatives)
    state = binary._safeguarded_root(mu, g, q, 2**-13, 100)
    assert state.converged.tolist() == [True, True, True, True, False]
    assert observed and max(map(len, observed)) == 3 and min(map(len, observed)) == 1
    assert all(.03 not in values and .8 not in values for values in observed)


def test_combining_tails_uses_one_root_call_with_unchanged_diagnostics(monkeypatch):
    g = torch.ones((24, 3), dtype=torch.float64)
    g[:12] = -1
    mu = torch.tensor([.1, .2, .3], dtype=torch.float64)[None].expand(24, -1)
    q = torch.tensor([1., -2., 3.], dtype=torch.float64)
    original = binary._safeguarded_root
    calls = []
    def root(probabilities, columns, score, *args, **kwargs):
        calls.append(columns.shape[1])
        return original(probabilities, columns, score, *args, **kwargs)
    monkeypatch.setattr(binary, '_safeguarded_root', root)
    expected = binary.binary_spa(q, g, mu)
    assert calls == [3, 3]
    calls.clear()
    actual = binary.binary_spa(q, g, mu, combine_tails=True)
    assert calls == [6]
    for name in actual.__dataclass_fields__:
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=1e-13)


@pytest.mark.parametrize('combine_tails', [False, True])
def test_heterogeneous_support_endpoints_and_failures_match_scalar(combine_tails):
    g = torch.tensor([[1., 1., 0., 1.], [-1., 1., 0., 1.], [0., 1., 0., 1.]], dtype=torch.float64)
    mu = torch.tensor([[.2, .1, .3, .4], [.2, .2, .4, .5], [.5, .3, .5, .6]], dtype=torch.float64)
    q = torch.tensor([1., 2.4, 0., 10.], dtype=torch.float64)
    actual = binary.binary_spa(q, g, mu, combine_tails=combine_tails, warn_failures=False)
    expected = [binary.binary_spa(q[i:i+1], g[:, i:i+1], mu[:, i], warn_failures=False) for i in range(4)]
    for name in actual.__dataclass_fields__:
        torch.testing.assert_close(getattr(actual, name), torch.cat([getattr(item, name) for item in expected]),
                                   rtol=0, atol=1e-13, equal_nan=True)


@pytest.mark.parametrize('probability', [torch.ones((3, 1))*.1, torch.ones((2, 2))*.1])
def test_heterogeneous_probabilities_require_exact_ordered_test_dimensions(probability):
    with pytest.raises(ValueError, match='fitted_probability'):
        binary.binary_spa(torch.ones(2), torch.ones((3, 2)), probability)
