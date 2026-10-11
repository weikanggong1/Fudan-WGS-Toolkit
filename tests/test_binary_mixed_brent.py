"""Mixed-model recovery contracts; these are not real-cohort benchmarks."""
import math

import numpy as np
import pytest
import torch

import fudan_wgs_toolkit.binary_null as binary_null


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _fixture(device="cpu", n=16):
    diagonal = np.linspace(.8, 1.2, n)
    # Leave two singletons so both determinant paths are exercised.
    rows, cols = np.arange(0, n - 2, 2), np.arange(1, n - 2, 2)
    values = np.full(n // 2 - 1, .2)
    kinship = binary_null._BlockKinship(diagonal, rows, cols, values,
        device=device, max_block_size=8)
    dense = torch.diag(torch.as_tensor(diagonal, dtype=torch.float64,
                                     device=device))
    dense[rows, cols] = .2
    dense[cols, rows] = .2
    coordinate = torch.linspace(-1, 1, n, dtype=torch.float64, device=device)
    x = torch.stack((torch.ones_like(coordinate), coordinate, coordinate.square()), 1)
    working = torch.sin(3 * coordinate) + 2 * coordinate.square()
    weights = torch.linspace(.03, .25, n, dtype=torch.float64, device=device)
    return kinship, dense, x, working, weights


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tau", [0., 1e-5, .2, 100., 1e5])
def test_block_fixed_dispersion_objective_and_projection_match_dense(device, tau):
    kinship, dense, x, working, weights = _fixture(device)
    value, (inverse, sx, covariance, coefficients, py) = binary_null._block_binomial_reml(
        kinship, working, weights, x, tau)
    sigma = torch.diag(1 / weights) + tau * dense
    precision = torch.linalg.inv(sigma)
    gram = x.T @ precision @ x
    cov = torch.linalg.inv(gram)
    alpha = cov @ x.T @ precision @ working
    residual = working - x @ alpha
    quadratic = residual @ precision @ residual
    expected = torch.linalg.slogdet(sigma)[1] + torch.linalg.slogdet(gram)[1] + quadratic
    torch.testing.assert_close(value, expected, rtol=3e-11, atol=3e-11)
    torch.testing.assert_close(inverse.sparse_tensor().to_dense(), precision, rtol=3e-11, atol=3e-11)
    torch.testing.assert_close(sx, precision @ x, rtol=3e-11, atol=3e-11)
    torch.testing.assert_close(covariance, cov, rtol=3e-11, atol=3e-11)
    torch.testing.assert_close(coefficients, alpha, rtol=3e-11, atol=3e-11)
    torch.testing.assert_close(py, precision @ residual, rtol=3e-11, atol=3e-11)
    assert value.device.type == device
    # The profiled-dispersion Gaussian objective is a different model.
    profiled = (torch.linalg.slogdet(sigma)[1] + torch.linalg.slogdet(gram)[1]
                + (len(working) - x.shape[1]) * torch.log(quadratic))
    assert abs(float(value - profiled)) > 1


def test_block_objective_derivative_is_negative_mixed_score():
    kinship, dense, x, working, weights = _fixture()
    tau, step = .7, 1e-4
    low, _ = binary_null._block_binomial_reml(kinship, working, weights, x, tau - step)
    high, _ = binary_null._block_binomial_reml(kinship, working, weights, x, tau + step)
    _, (inverse, sx, cov, _, py) = binary_null._block_binomial_reml(
        kinship, working, weights, x, tau)
    precision = inverse.sparse_tensor().to_dense()
    p = precision - sx @ cov @ sx.T
    score = py @ dense @ py - torch.trace(p @ dense)
    torch.testing.assert_close((high - low) / (2 * step), -score, rtol=1e-8, atol=1e-9)


def test_objective_never_factorizes_a_samples_by_samples_matrix(monkeypatch):
    kinship, _, x, working, weights = _fixture(n=32)
    original = torch.linalg.cholesky
    seen = []

    def guarded(matrix, *args, **kwargs):
        seen.append(tuple(matrix.shape))
        assert matrix.shape[-2:] != (len(working), len(working))
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky", guarded)
    binary_null._block_binomial_reml(kinship, working, weights, x, .5)
    assert any(shape[-2:] == (2, 2) for shape in seen) and (3, 3) in seen


def test_brent_visits_official_ten_log_intervals_and_preserves_kinship(monkeypatch):
    import scipy.optimize
    kinship, _, x, working, weights = _fixture()
    calls = []
    original = scipy.optimize.minimize_scalar

    def observed(objective, **kwargs):
        calls.append(kwargs)
        return original(objective, **kwargs)

    monkeypatch.setattr(scipy.optimize, "minimize_scalar", observed)
    tau, value, fitted = binary_null._block_binomial_brent_step(
        kinship, working, weights, x, 1e-5)
    assert len(calls) == 10
    edges = np.linspace(math.log(1e-5), math.log(1e5), 11)
    for call, left, right in zip(calls, edges[:-1], edges[1:]):
        np.testing.assert_allclose(call["bounds"], [left, right], rtol=0, atol=1e-14)
        assert call["method"] == "bounded" and call["options"] == {"xatol": 1e-5}
    assert 1e-5 <= float(tau) <= 1e5 and torch.isfinite(value)
    assert fitted[0].kinship is kinship


def _mixed_inputs(n=64):
    coordinate = np.linspace(-1, 1, n)
    x = np.column_stack((np.ones(n), coordinate))
    y = np.tile([0., 1., 1., 0., 1., 0., 0., 1.], n // 8)
    parameters = dict(kinship_diagonal=np.ones(n),
        edge_rows=np.arange(0, n, 2), edge_cols=np.arange(1, n, 2),
        edge_values=np.full(n // 2, .2))
    return y, x, parameters


@pytest.mark.parametrize("device", DEVICES)
def test_ai_nonconvergence_restarts_same_mixed_model_and_explicit_zero_boundary(monkeypatch, device):
    y, x, parameters = _mixed_inputs()
    original_ordinary = binary_null.fit_logistic_null
    original_recovery = binary_null._fit_logistic_block_brent
    ordinary_fits, recovery_inputs, trace = [], [], []

    def ordinary(*args, **kwargs):
        fitted = original_ordinary(*args, **kwargs)
        ordinary_fits.append(fitted)
        return fitted

    def recovery(yy, xx, initial, kinship, **kwargs):
        recovery_inputs.append((yy.clone(), xx.clone(), initial, kinship))
        return original_recovery(yy, xx, initial, kinship, **kwargs)

    def force_failed_ai(self):
        raise binary_null._MixedAINonconvergence("forced AI trajectory")

    monkeypatch.setattr(binary_null, "fit_logistic_null", ordinary)
    monkeypatch.setattr(binary_null, "_fit_logistic_block_brent", recovery)
    monkeypatch.setattr(binary_null._BlockPrecision, "trace_kinship", force_failed_ai)
    fitted = binary_null.fit_logistic_mixed_null(y, x, device=device,
        trace_callback=trace.append, **parameters)
    assert len(ordinary_fits) == len(recovery_inputs) == 1
    yy, xx, initial, kinship = recovery_inputs[0]
    assert initial is ordinary_fits[0]
    torch.testing.assert_close(yy, initial.phenotype, atol=0, rtol=0)
    torch.testing.assert_close(xx, initial.x, atol=0, rtol=0)
    assert kinship.blocks and fitted.has_kinship and fitted.converged
    assert fitted.boundary_refit and fitted.theta.tolist() == [1., 0.]
    assert fitted.precision_theta.tolist() == [1., 0.]
    assert fitted.mixed_optimizer == "Brent"
    assert fitted.mixed_ai_recovery_reason == "forced AI trajectory"
    assert fitted.mixed_ai_iterations == 0 and fitted.mixed_brent_iterations > 1
    assert fitted.iterations == fitted.mixed_ai_iterations + fitted.mixed_brent_iterations
    assert fitted.mixed_optimizer_search_bounds == (1e-5, 1e5)
    assert fitted.mixed_optimizer_search_regions == 10
    assert fitted.fit_method == "logistic_block_sparse_GMMAT_PQL_Brent_REML"
    assert fitted.precision.layout == torch.sparse_coo
    assert fitted.device.type == device
    assert all(state["optimizer"] == "Brent" for state in trace)
    assert any(state["refit"] == 1 for state in trace)
    # The first recovery state resets to the original fixed-effects predictor.
    initial_mu, initial_derivative, initial_weights = binary_null._binomial_logit_state(
        initial.x @ initial.coefficients)
    expected_working = initial.x @ initial.coefficients + (initial.phenotype - initial_mu) / initial_derivative
    torch.testing.assert_close(trace[0]["Y_old"], expected_working, atol=0, rtol=0)
    torch.testing.assert_close(trace[0]["weights_old"], initial_weights, atol=0, rtol=0)


def test_failed_brent_remains_hard_failure(monkeypatch):
    y, x, parameters = _mixed_inputs()

    def force_failed_ai(self):
        raise binary_null._MixedAINonconvergence("forced AI trajectory")

    def force_failed_brent(*args, **kwargs):
        raise ArithmeticError("mixed logistic Brent PQL did not converge")

    monkeypatch.setattr(binary_null._BlockPrecision, "trace_kinship", force_failed_ai)
    monkeypatch.setattr(binary_null, "_fit_logistic_block_brent", force_failed_brent)
    with pytest.raises(ArithmeticError, match="Brent PQL did not converge"):
        binary_null.fit_logistic_mixed_null(y, x, device="cpu", **parameters)


def test_brent_search_failure_and_outer_iteration_limit_remain_errors(monkeypatch):
    import scipy.optimize
    from types import SimpleNamespace
    kinship, _, x, working, weights = _fixture()
    monkeypatch.setattr(scipy.optimize, "minimize_scalar",
        lambda *args, **kwargs: SimpleNamespace(success=False, fun=0., x=0.))
    with pytest.raises(ArithmeticError, match="Brent variance search did not converge"):
        binary_null._block_binomial_brent_step(kinship, working, weights, x, 1e-5)
    monkeypatch.undo()
    y, design, parameters = _mixed_inputs()
    ordinary = binary_null.fit_logistic_null(y, design, device="cpu")
    kinship = binary_null._BlockKinship(parameters["kinship_diagonal"],
        parameters["edge_rows"], parameters["edge_cols"], parameters["edge_values"],
        device="cpu", max_block_size=8)
    with pytest.raises(ArithmeticError, match="Brent PQL did not converge"):
        binary_null._fit_logistic_block_brent(ordinary.phenotype, ordinary.x,
            ordinary, kinship, maxiter=2, tol=1e-5)


def test_successful_ai_does_not_call_recovery(monkeypatch):
    y, x, parameters = _mixed_inputs()

    def forbidden_recovery(*args, **kwargs):
        raise AssertionError("successful AI must not use recovery")

    monkeypatch.setattr(binary_null, "_fit_logistic_block_brent", forbidden_recovery)
    fitted = binary_null.fit_logistic_mixed_null(y, x, device="cpu", **parameters)
    assert fitted.mixed_optimizer == "AI"
    assert fitted.mixed_ai_recovery_reason is None and fitted.mixed_brent_iterations == 0
    assert fitted.mixed_ai_iterations == fitted.iterations
