"""Binary SPA numerical rules; small matrices are not scientific benchmarks."""
import pytest
import torch
from fudan_wgs_toolkit.binary import binary_spa, individual_score_test_spa


def test_spa_degenerate_projection_explicitly_reports_failure_policy_without_root_search():
    genotype = torch.zeros((4, 2), dtype=torch.float64)
    probability = torch.full((4,), .1, dtype=torch.float64)
    with pytest.warns(RuntimeWarning, match="returned p=1"):
        result = binary_spa(torch.zeros(2), genotype, probability)
    assert result.failed.all()
    assert not result.used_bisection.any()
    assert not result.iterations.any()
    assert not result.converged.any()
    torch.testing.assert_close(result.pvalues, torch.ones(2, dtype=torch.float64))


def test_individual_spa_filter_preserves_values_above_strict_cutoff():
    genotype = torch.tensor([[0, 1], [1, 0], [0, 0], [0, 1]], dtype=torch.float64)
    intercept_weights = torch.full((1, 4), .09, dtype=torch.float64)
    left = torch.full((4, 1), 1 / .36, dtype=torch.float64)
    normal = torch.tensor([.05, .6], dtype=torch.float64)
    result = individual_score_test_spa(genotype, torch.zeros(4), torch.full((4,), .1),
        intercept_weights, left, normal_pvalues=normal, return_diagnostics=True)
    torch.testing.assert_close(result.pvalues, normal)
    assert not result.used_bisection.any()
    assert not result.failed.any()


def test_spa_rejects_invalid_fitted_probabilities():
    with pytest.raises(ValueError, match="strictly between"):
        binary_spa(torch.ones(1), torch.ones((3, 1)), [0, .1, .2])
