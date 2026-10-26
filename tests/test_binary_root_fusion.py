"""CUDA FP64 SPA fusion versus the retained Torch solver.

These bounded synthetic component regressions check roots, convergence
diagnostics and final SPA probabilities. They do not establish a real-data
chr21 benchmark or replace its independent result comparison.
"""
import pytest
import torch

from fudan_wgs_toolkit import binary


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FP64 SPA required")
TOLERANCE = 2**-13


def _strided_matrix(value):
    storage = torch.empty((2*value.shape[0], 2*value.shape[1]),
                          dtype=value.dtype, device=value.device)
    view = storage[::2, ::2]
    view.copy_(value)
    assert not view.is_contiguous()
    return view


def _known_root_fixture(shared_probability, strided):
    generator = torch.Generator(device="cpu").manual_seed(104729)
    genotype = torch.randn((129, 7), generator=generator, dtype=torch.float64).cuda()
    logits = torch.linspace(-2., 1.5, 129, dtype=torch.float64, device="cuda")[:, None]
    if not shared_probability:
        logits = logits + logits.new_tensor([-10., -4., -1., 0., 1., 4., 10.])[None, :]
    probability = torch.sigmoid(logits)
    weight = probability*(1-probability)
    genotype = genotype-(weight*genotype).sum(0)/weight.sum(0)
    known_root = genotype.new_tensor([-2., -.15, 0., .2, 1.5, 3., -3.])
    # Construct scores from a prescribed tilted Bernoulli mean. This gives
    # known roots without reproducing either iterative solver in the test.
    log_odds = torch.log(probability)-torch.log1p(-probability)
    score = ((torch.sigmoid(log_odds+genotype*known_root)-probability)*genotype).sum(0)
    # The centered score is mathematically zero; sigmoid(logit(mu)) need not
    # round back to exactly mu, so set this exact semantic case explicitly.
    score[2] = 0.
    if strided:
        genotype = _strided_matrix(genotype)
        probability = _strided_matrix(probability)
    return probability, genotype, score, known_root


def _assert_state_matches_reference(actual, expected):
    assert actual.__dataclass_fields__.keys() == expected.__dataclass_fields__.keys()
    for name in expected.__dataclass_fields__:
        observed, reference = getattr(actual, name), getattr(expected, name)
        assert observed.device == reference.device, name
        assert observed.dtype == reference.dtype, name
        if reference.is_floating_point():
            torch.testing.assert_close(observed, reference, rtol=2e-12, atol=2e-12,
                                       equal_nan=True, msg=lambda message: f"{name}: {message}")
        else:
            assert torch.equal(observed, reference), name


def _solve_both(probability, genotype, score, *, newton=True, max_iter=160):
    from fudan_wgs_toolkit._binary_root_control import safeguarded_root

    expected = binary._safeguarded_root_reference(
        probability, genotype, score, TOLERANCE, max_iter, newton=newton)
    actual = safeguarded_root(probability, genotype, score, TOLERANCE, max_iter,
                              newton=newton)
    _assert_state_matches_reference(actual, expected)
    return actual


@pytest.mark.parametrize("shared_probability", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("newton", [False, True])
def test_fused_root_matches_known_roots_and_all_reference_diagnostics(
        shared_probability, strided, newton):
    probability, genotype, score, known_root = _known_root_fixture(shared_probability, strided)
    actual = _solve_both(probability, genotype, score, newton=newton)
    assert actual.converged.all() and not actual.iteration_limit.any()
    torch.testing.assert_close(actual.root, known_root, rtol=0, atol=2e-7)
    log_odds = torch.log(probability)-torch.log1p(-probability)
    residual = (((torch.sigmoid(log_odds+genotype*actual.root)-probability)*genotype).sum(0)
                - score).abs()
    variance = (probability*(1-probability)*genotype.square()).sum(0)
    bound = min(TOLERANCE, 1e-10)*torch.maximum(variance.sqrt(), score.abs())
    assert torch.all(residual <= bound+32*torch.finfo(score.dtype).eps)
    assert actual.iterations[2] == actual.bracket_iterations[2] == 0
    assert actual.root[2] == actual.residual[2] == 0
    if not newton:
        assert actual.used_bisection[score != 0].all()


@pytest.mark.parametrize("newton", [False, True])
def test_fused_root_near_support_has_independent_binomial_roots(newton):
    n = 64
    probability = torch.tensor([.01, .2, .8, .99], dtype=torch.float64,
                               device="cuda")[None, :].expand(n, -1)
    tilted_probability = probability.new_tensor([1e-5, 1-1e-5, 1e-4, 1-1e-4])
    genotype = _strided_matrix(torch.ones((n, 4), dtype=torch.float64, device="cuda"))
    score = n*(tilted_probability-probability[0])
    # For a constant unit genotype, K'(t) = n*(sigmoid(logit(mu)+t)-mu).
    known_root = torch.logit(tilted_probability)-torch.logit(probability[0])
    actual = _solve_both(probability, genotype, score, newton=newton)
    assert actual.converged.all() and not actual.iteration_limit.any()
    torch.testing.assert_close(actual.root, known_root, rtol=0, atol=2e-5)


@pytest.mark.parametrize("newton", [False, True])
def test_fused_root_extreme_legal_logits_preserves_reference_outcomes(newton):
    logits = torch.tensor([-24., -12., -2., 0., 2., 12., 24.],
                          dtype=torch.float64, device="cuda")[None, :].expand(65, -1)
    probability = _strided_matrix(torch.sigmoid(logits))
    genotype = _strided_matrix(torch.ones_like(probability))
    target = logits.new_tensor([-2., .5, -3., 0., 3., -.5, 2.])
    log_odds = torch.log(probability)-torch.log1p(-probability)
    score = ((torch.sigmoid(log_odds+genotype*target)-probability)*genotype).sum(0)
    score[3] = 0.
    actual = _solve_both(probability, genotype, score, newton=newton)
    assert actual.converged.any()
    assert not actual.iteration_limit.any()
    assert torch.isfinite(actual.root).all()


@pytest.mark.parametrize("newton", [False, True])
def test_fused_root_preserves_support_rejection_and_centered_diagnostics(newton):
    probability = torch.full((48, 1), .2, dtype=torch.float64, device="cuda")
    genotype = torch.ones((48, 8), dtype=torch.float64, device="cuda")
    genotype[:24] = -1
    genotype[:, 0] = 0
    lower, upper = binary._score_support(probability, genotype)
    score = genotype.new_tensor([0., 0., 0., 0., 0., 0., 3., -3.])
    score[2], score[3] = lower[2], upper[3]
    score[4], score[5] = lower[4]-1, upper[5]+1
    actual = _solve_both(probability, genotype, score, newton=newton)
    assert actual.converged.tolist() == [False, True, False, False, False, False, True, True]
    assert not actual.iteration_limit.any()
    assert not actual.iterations[:6].any()
    assert not actual.bracket_iterations[:6].any()
    assert torch.isinf(actual.residual[[0, 2, 3, 4, 5]]).all()


@pytest.mark.parametrize("newton", [False, True])
def test_fused_root_iteration_limit_does_not_certify_unconverged_roots(newton):
    probability, genotype, score, _ = _known_root_fixture(False, True)
    actual = _solve_both(probability, genotype, score, newton=newton, max_iter=1)
    noncentral = score != 0
    assert torch.all(actual.iterations[noncentral] == 1)
    assert actual.iteration_limit[noncentral].any()
    assert not actual.converged[actual.iteration_limit].any()
    assert actual.converged[~noncentral].all()


def test_root_dispatch_uses_fusion_for_strided_matrices_but_not_strided_scores(monkeypatch):
    from fudan_wgs_toolkit import _binary_root_control

    probability, genotype, score, _ = _known_root_fixture(False, True)
    calls = []
    fused = _binary_root_control.safeguarded_root

    def observe_fused(*args, **kwargs):
        calls.append(1)
        return fused(*args, **kwargs)

    monkeypatch.setattr(_binary_root_control, "safeguarded_root", observe_fused)
    expected = binary._safeguarded_root_reference(probability, genotype, score, TOLERANCE, 160)
    actual = binary._safeguarded_root(probability, genotype, score, TOLERANCE, 160)
    assert len(calls) == 1
    _assert_state_matches_reference(actual, expected)
    score_storage = torch.empty(2*len(score), dtype=score.dtype, device=score.device)
    strided_score = score_storage[::2]
    strided_score.copy_(score)
    assert not strided_score.is_contiguous()
    fallback = binary._safeguarded_root(probability, genotype, strided_score, TOLERANCE, 160)
    assert len(calls) == 1
    _assert_state_matches_reference(fallback, expected)


@pytest.mark.parametrize("shared_probability", [False, True])
@pytest.mark.parametrize("combine_tails", [False, True])
def test_fused_spa_final_log_probabilities_match_retained_solver(
        shared_probability, combine_tails, monkeypatch):
    probability, genotype, score, _ = _known_root_fixture(shared_probability, True)
    public_probability = probability[:, 0] if shared_probability else probability
    actual = binary.binary_spa(score, genotype, public_probability,
                               combine_tails=combine_tails, warn_failures=False)
    monkeypatch.setattr(binary, "_safeguarded_root", binary._safeguarded_root_reference)
    expected = binary.binary_spa(score, genotype, public_probability,
                                 combine_tails=combine_tails, warn_failures=False)
    _assert_state_matches_reference(actual, expected)
    # Preserve each reference tail outcome, including explicit failures for
    # ill-conditioned tails, rather than demanding every synthetic tail pass.
    assert actual.converged[2] and expected.converged[2]
    assert actual.pvalues[2] == expected.pvalues[2] == 1.
    noncentral_success = ~expected.failed & (expected.pvalues > 0) & (expected.pvalues < 1)
    assert noncentral_success.any()
    assert torch.equal(actual.pvalues == 0, expected.pvalues == 0)
    assert torch.equal(actual.pvalues == 1, expected.pvalues == 1)
    positive = expected.pvalues > 0
    delta_logp = (actual.pvalues[positive].log10()-expected.pvalues[positive].log10()).abs()
    assert torch.all(delta_logp < .001)
