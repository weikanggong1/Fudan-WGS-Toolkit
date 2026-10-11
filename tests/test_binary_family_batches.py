"""Bounded FP64 family products; component tests are not cohort benchmarks."""
from contextlib import contextmanager
import weakref

import numpy as np
import pytest
import torch

import fudan_wgs_toolkit.binary_null as binary
import fudan_wgs_toolkit._fit_structure_cache as structure


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _kinship(device="cpu"):
    # Interleave sizes so regrouping cannot silently reorder trace/logdet.
    diagonal = np.linspace(.8, 1.2, 16)
    rows = np.array([0, 2, 3, 5, 7, 8, 10], dtype=np.int64)
    cols = rows + 1
    edges = np.full(len(rows), .12)
    model = binary._BlockKinship(diagonal, rows, cols, edges,
        device=device, max_block_size=3)
    dense = torch.diag(torch.as_tensor(diagonal, dtype=torch.float64, device=device))
    dense[rows, cols] = torch.as_tensor(edges, dtype=torch.float64, device=device)
    dense[cols, rows] = torch.as_tensor(edges, dtype=torch.float64, device=device)
    return model, dense


def _scalar_precision(kinship, weights, tau):
    diagonal = 1 / (1 / weights + tau * kinship.diagonal)
    dense = torch.diag(diagonal)
    logdet = -torch.log(diagonal[~kinship.grouped]).sum()
    trace = (diagonal[~kinship.grouped] * kinship.diagonal[~kinship.grouped]).sum()
    for rows, matrix in kinship.blocks:
        factor = torch.linalg.cholesky(torch.diag(1 / weights[rows]) + tau * matrix)
        inverse = torch.cholesky_inverse(factor)
        inverse = (inverse + inverse.T) / 2
        dense[rows[:, None], rows[None, :]] = inverse
        logdet = logdet + 2 * torch.log(factor.diagonal()).sum()
        trace = trace + (inverse * matrix.T).sum()
    return dense, logdet, trace


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("width", [1, 2, 4096])
@pytest.mark.parametrize("tau", [0., .43, 100.])
def test_precision_family_batches_match_scalar_and_dense(device, width, tau, monkeypatch):
    monkeypatch.setattr(binary, "_BINARY_FAMILY_BATCH_COUNT", width)
    kinship, dense_k = _kinship(device)
    weights = torch.linspace(.03, .25, 16, dtype=torch.float64, device=device)
    expected, expected_logdet, expected_trace = _scalar_precision(kinship, weights, tau)
    precision = binary._BlockPrecision(kinship, weights, tau, compute_logdet=True)
    torch.testing.assert_close(precision.sparse_tensor().to_dense(), expected, atol=3e-14, rtol=3e-14)
    torch.testing.assert_close(precision.logdet, expected_logdet, atol=3e-14, rtol=3e-14)
    torch.testing.assert_close(precision.trace_kinship(), expected_trace, atol=3e-14, rtol=3e-14)
    torch.testing.assert_close(expected, torch.linalg.inv(torch.diag(1 / weights) + tau * dense_k),
                               atol=3e-13, rtol=3e-13)
    # Both operators must accept original strided vectors/matrices.
    vector = torch.arange(32, dtype=torch.float64, device=device)[::2] / 7
    matrix = torch.arange(48, dtype=torch.float64, device=device).reshape(3, 16).T / 9
    for rhs in (vector, matrix):
        torch.testing.assert_close(precision.apply(rhs), expected @ rhs, atol=3e-13, rtol=3e-13)
        torch.testing.assert_close(kinship.apply(rhs), dense_k @ rhs, atol=3e-13, rtol=3e-13)
    assert [len(rows) for rows, _ in precision.blocks] == [len(rows) for rows, _ in kinship.blocks]
    assert precision.diagonal.dtype == torch.float64
    assert precision.apply(matrix[:, :0]).shape == (16, 0)


@pytest.mark.parametrize("device", DEVICES)
def test_one_factor_and_apply_call_per_same_size_batch(device, monkeypatch):
    kinship, _ = _kinship(device)
    weights = torch.linspace(.03, .25, 16, dtype=torch.float64, device=device)
    cholesky, bmm = torch.linalg.cholesky, torch.bmm
    factors, products = [], []

    def factor(matrix, *args, **kwargs):
        factors.append(tuple(matrix.shape))
        return cholesky(matrix, *args, **kwargs)

    def product(left, right):
        products.append((tuple(left.shape), tuple(right.shape)))
        return bmm(left, right)

    monkeypatch.setattr(torch.linalg, "cholesky", factor)
    precision = binary._BlockPrecision(kinship, weights, .43)
    assert sorted(factors) == [(2, 3, 3), (3, 2, 2)]
    monkeypatch.setattr(torch, "bmm", product)
    precision.apply(torch.ones((16, 4), dtype=torch.float64, device=device))
    assert len(products) == 2
    assert all(shape[0][-1] < 16 for shape in products)


def test_weights_and_tau_never_mutate_other_precision_or_shared_geometry():
    kinship, _ = _kinship()
    weights = torch.linspace(.03, .25, 16, dtype=torch.float64)
    first = binary._BlockPrecision(kinship, weights, .43)
    before = first.sparse_tensor().to_dense().clone()
    geometry = [matrix.clone() for _, matrix in kinship.blocks]
    second = binary._BlockPrecision(kinship, weights * 1.3, .72)
    torch.testing.assert_close(first.sparse_tensor().to_dense(), before, atol=0, rtol=0)
    assert not torch.equal(second.sparse_tensor().to_dense(), before)
    for (_, matrix), expected in zip(kinship.blocks, geometry):
        torch.testing.assert_close(matrix, expected, atol=0, rtol=0)


def test_family_batch_width_uses_live_available_bytes_and_single_block_guard(monkeypatch):
    # Fake byte availability tests admission arithmetic, not CUDA allocation.
    monkeypatch.setattr(structure, "available_cuda_bytes", lambda device: 250)
    assert binary._binary_family_batch_width("cpu", 20, 100, context="mock") == 2
    monkeypatch.setattr(structure, "available_cuda_bytes", lambda device: 99)
    with pytest.raises(MemoryError, match="mock requires estimated 100 new bytes"):
        binary._binary_family_batch_width("cpu", 20, 100, context="mock")
    monkeypatch.setattr(structure, "available_cuda_bytes", lambda device: None)
    monkeypatch.setattr(binary, "_BINARY_FAMILY_BATCH_BYTES", 1)
    assert binary._binary_family_batch_width("cpu", 20, 100, context="mock") == 1


def test_precision_failure_releases_partial_owned_batches(monkeypatch):
    kinship, _ = _kinship()
    weights = torch.linspace(.03, .25, 16, dtype=torch.float64)
    monkeypatch.setattr(binary, "_BINARY_FAMILY_BATCH_COUNT", 1)
    cholesky = torch.linalg.cholesky
    calls = 0
    retained_inverse = []

    def fail_second(matrix, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            retained_inverse.append(weakref.ref(precision._batches[0][2]))
            raise RuntimeError("factor failed")
        return cholesky(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky", fail_second)
    precision = object.__new__(binary._BlockPrecision)
    with pytest.raises(RuntimeError, match="factor failed") as captured_error:
        precision.__init__(kinship, weights, .43, compute_logdet=True)
    assert precision.blocks == precision._batches == []
    assert precision.diagonal is precision.logdet is None
    assert captured_error.value.__traceback__ is not None
    assert len(retained_inverse) == 1 and retained_inverse[0]() is None


def test_batch_retains_hard_nonpositive_definite_failure():
    kinship, _ = _kinship()
    with pytest.raises(RuntimeError):
        binary._BlockPrecision(kinship, torch.full((16,), -1., dtype=torch.float64), 0.)


def test_mixed_fit_cache_lease_covers_solver_and_releases_on_error(monkeypatch):
    cache = structure.FitStructureCache(max_bytes=2**20)
    samples = np.arange(16).astype(str)
    y = np.tile([0., 1., 0., 0.], 4)
    x = np.ones((16, 1))
    events = []
    acquire = cache.acquire_spectrum

    @contextmanager
    def track(*args, **kwargs):
        try:
            with acquire(*args, **kwargs) as spectrum:
                events.append("enter")
                yield spectrum
        finally:
            events.append("exit")

    def solver(ordinary, **kwargs):
        assert cache.stats()["active_leases"] == 1
        assert kwargs["spectrum"].eigenvalues.dtype == torch.float64
        assert ordinary.x.dtype == ordinary.fitted_probability.dtype == torch.float64
        raise ArithmeticError("solver failure")

    monkeypatch.setattr(cache, "acquire_spectrum", track)
    monkeypatch.setattr(binary, "_fit_logistic_mixed_from_ordinary", solver)
    with pytest.raises(ArithmeticError, match="solver failure"):
        binary.fit_logistic_mixed_null(y, x, sample_ids=samples,
            kinship_diagonal=np.ones(16), device="cpu", fit_structure_cache=cache)
    assert events == ["enter", "exit"]
    assert cache.stats()["active_leases"] == 0
    cache.close()


def test_complete_mixed_fit_matches_scalar_family_operator(monkeypatch):
    # A small fitted-state regression, not an original-software benchmark.
    n = 48
    rng = np.random.default_rng(916)
    x = np.column_stack((np.ones(n), np.linspace(-1, 1, n)))
    y = (rng.uniform(size=n) < .35).astype(float)
    rows = np.arange(0, n, 2)
    settings = dict(kinship_diagonal=np.ones(n), edge_rows=rows,
                    edge_cols=rows + 1, edge_values=np.full(len(rows), .12),
                    device="cpu", maxiter=100, tol=1e-5)
    fitted = binary.fit_logistic_mixed_null(y, x, **settings)
    cholesky, inverse = torch.linalg.cholesky, torch.cholesky_inverse

    def scalar_cholesky(matrix, *args, **kwargs):
        if matrix.ndim == 3:
            return torch.stack([cholesky(block, *args, **kwargs) for block in matrix])
        return cholesky(matrix, *args, **kwargs)

    def scalar_inverse(matrix, *args, **kwargs):
        if matrix.ndim == 3:
            return torch.stack([inverse(block, *args, **kwargs) for block in matrix])
        return inverse(matrix, *args, **kwargs)

    def scalar_product(left, right):
        return torch.stack([a @ b for a, b in zip(left, right)])

    monkeypatch.setattr(torch.linalg, "cholesky", scalar_cholesky)
    monkeypatch.setattr(torch, "cholesky_inverse", scalar_inverse)
    monkeypatch.setattr(torch, "bmm", scalar_product)
    reference = binary.fit_logistic_mixed_null(y, x, **settings)
    for field in ("coefficients", "fitted_probability", "scaled_residuals", "theta",
                  "precision_theta", "precision_x", "fixed_effect_covariance", "projection_left"):
        torch.testing.assert_close(getattr(fitted, field), getattr(reference, field), atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(fitted.precision.to_dense(), reference.precision.to_dense(), atol=1e-9, rtol=1e-9)
    for field in ("converged", "iterations", "boundary_refit", "mixed_optimizer",
                  "mixed_ai_iterations", "mixed_brent_iterations"):
        assert getattr(fitted, field) == getattr(reference, field)
