"""Explicit binary diagonal tiled/cache contracts; no fitting is performed."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.binary_null import BinaryNullModel, binary_prefitted_state
from fudan_wgs_toolkit._cached_covariance import score_covariance_cached


def fitted_state(*, device='cpu', mode='tf32', n=11):
    x = np.column_stack((np.ones(n), np.arange(n) / n, (np.arange(n) % 3) / 3))
    precision = .1 + (np.arange(n) % 5) / 25
    sx = precision[:, None] * x
    # Explicit finite-tolerance fitted C, with a small asymmetric component.
    # Neither bounded backend may solve a replacement C from X or probabilities.
    cov = np.linalg.inv(x.T @ sx) * .91
    cov[0, 1] += .001
    mu = .2 + (np.arange(n) % 4) / 10
    residual = ((np.arange(n) % 2) - mu) * .83
    return binary_prefitted_state(sample_ids=np.arange(n).astype(str), covariates=x,
        residual=residual, fitted_probability=mu, xw=sx.T, projection_left=x @ cov,
        fixed_effect_covariance=cov, precision=precision, precision_covariates=sx,
        has_kinship=True, use_spa=False, device=device, matmul_mode=mode)


def _cpu_product(a, b, *, mode):
    assert mode == 'tf32' and a.dtype == b.dtype == torch.float32
    return a @ b


@pytest.mark.parametrize('layout', ['C', 'F'])
@pytest.mark.parametrize('m', [0, 1, 2, 7])
@pytest.mark.parametrize('tile', [1, 3, 8])
def test_cpu_tiled_uses_supplied_binary_formula_and_preserves_order(monkeypatch, layout, m, tile):
    monkeypatch.setattr('fudan_wgs_toolkit.binary_null.matmul', _cpu_product)
    model = fitted_state()
    generator = np.random.default_rng(33)
    genotype = np.array(generator.integers(0, 3, size=(model.n, m)), dtype=np.float32, order=layout)
    if m:
        genotype[2, -1] = .173  # already imputed dosage stays unchanged
    before = genotype.copy(order=layout)
    fitted = {name: getattr(model, name).clone() for name in ('precision', 'precision_x', 'scaled_residuals', 'fixed_effect_covariance')}
    u, v = model.score_covariance_tiled(genotype, variant_tile_size=tile)
    g = torch.from_numpy(genotype)
    cross = model.precision_x.T @ g
    directed = g.T @ (model.precision[:, None] * g) - (cross.T @ model.fixed_effect_covariance) @ cross
    expected_v = (directed + directed.T) / 2
    torch.testing.assert_close(u, g.T @ model.scaled_residuals, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(v, expected_v, rtol=3e-6, atol=3e-6)
    assert u.dtype == v.dtype == torch.float32
    assert tuple(u.shape) == (m,) and tuple(v.shape) == (m, m)
    np.testing.assert_array_equal(genotype, before)
    for name, original in fitted.items():
        torch.testing.assert_close(getattr(model, name), original, rtol=0, atol=0)
    assert not hasattr(model, 'spectrum')


@pytest.mark.parametrize('method', ['score_covariance_tiled', 'score_covariance_cached'])
@pytest.mark.parametrize('change,exception', [
    ('spa', NotImplementedError), ('joint', NotImplementedError), ('family', NotImplementedError),
    ('no_precision', NotImplementedError), ('no_sx', NotImplementedError),
    ('dense_precision', ValueError), ('sparse_precision', ValueError),
    ('precision_fp64', ValueError), ('sx_wrong_shape', ValueError), ('residual_wrong_shape', ValueError),
    ('cov_wrong_shape', ValueError), ('nonfinite_precision', ValueError), ('nonfinite_sx', ValueError),
    ('nonfinite_residual', ValueError), ('nonfinite_cov', ValueError), ('nonpositive_precision', ValueError),
    ('model_grad', ValueError), ('genotype_grad', ValueError), ('fp64_mode', ValueError),
])
def test_unsupported_binary_state_rejected_before_backend_allocation(method, change, exception):
    model = fitted_state()
    genotype = torch.ones((model.n, 3))
    if change == 'spa': model.use_spa = True
    elif change == 'joint': model.n_pheno = 2
    elif change == 'family': model.family = 'gaussian'
    elif change == 'no_precision': model.precision = None
    elif change == 'no_sx': model.precision_x = None
    elif change == 'dense_precision': model.precision = torch.diag(model.precision)
    elif change == 'sparse_precision': model.precision = torch.diag(model.precision).to_sparse_coo()
    elif change == 'precision_fp64': model.precision = model.precision.double()
    elif change == 'sx_wrong_shape': model.precision_x = model.precision_x[:-1]
    elif change == 'residual_wrong_shape': model.scaled_residuals = model.scaled_residuals[:, None]
    elif change == 'cov_wrong_shape': model.fixed_effect_covariance = model.fixed_effect_covariance[:-1]
    elif change == 'nonfinite_precision': model.precision[0] = float('nan')
    elif change == 'nonfinite_sx': model.precision_x[0, 0] = float('inf')
    elif change == 'nonfinite_residual': model.scaled_residuals[0] = float('nan')
    elif change == 'nonfinite_cov': model.fixed_effect_covariance[0, 0] = float('inf')
    elif change == 'nonpositive_precision': model.precision[0] = 0
    elif change == 'model_grad': model.precision_x.requires_grad_()
    elif change == 'genotype_grad': genotype.requires_grad_()
    elif change == 'fp64_mode': model.set_matmul_mode('fp64')
    with pytest.raises(exception):
        getattr(model, method)(genotype)


@pytest.mark.parametrize('tile', [True, 0, -1, 1.5])
def test_binary_tiled_rejects_invalid_tile_size(tile):
    with pytest.raises(ValueError, match='positive integer'):
        fitted_state().score_covariance_tiled(np.ones((11, 2)), variant_tile_size=tile)


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_binary_tiled_checks_last_host_column(monkeypatch, value):
    monkeypatch.setattr('fudan_wgs_toolkit.binary_null.matmul', _cpu_product)
    genotype = np.ones((11, 7), dtype=np.float32)
    genotype[-1, -1] = value
    with pytest.raises(ValueError, match='genotype must be finite'):
        fitted_state().score_covariance_tiled(genotype, variant_tile_size=3)


def test_binary_tiled_rejects_axis_mismatch():
    with pytest.raises(ValueError, match='aligned to the model'):
        fitted_state().score_covariance_tiled(np.ones((10, 2), dtype=np.float32))


def test_diagonal_protocol_returns_original_fitted_tensors():
    model = fitted_state()
    state = model._diagonal_tf32_covariance_state()
    assert all(actual is getattr(model, name) for actual, name in zip(state,
               ('precision', 'precision_x', 'scaled_residuals', 'fixed_effect_covariance')))


def test_cached_wrapper_uses_mature_backend_and_keeps_original_model(monkeypatch):
    model = fitted_state()
    calls = []
    expected = (object(), object(), object())
    def backend(actual, genotype, **kwargs):
        calls.append((actual, genotype, kwargs))
        return expected
    monkeypatch.setattr('fudan_wgs_toolkit._cached_covariance.score_covariance_cached', backend)
    genotype = np.ones((11, 2), dtype=np.float32)
    actual = model.score_covariance_cached(genotype, variant_tile_size=512, panel_variant_size=1024,
                                            memory_limit_gib=20, profile=True)
    assert actual is expected and calls[0][0] is model and calls[0][1] is genotype
    assert calls[0][2] == dict(variant_tile_size=512, panel_variant_size=1024, memory_limit_gib=20,
                             matmul_mode='tf32', symmetry='average', profile=True, compute_score=True)


def test_cached_backend_rejects_cpu_and_untyped_binary_protocol():
    with pytest.raises(ValueError, match='CUDA model'):
        fitted_state().score_covariance_cached(np.ones((11, 2), dtype=np.float32))
    fake = SimpleNamespace(family='binomial', n_pheno=1, use_spa=False, matmul_mode='tf32')
    with pytest.raises(NotImplementedError):
        score_covariance_cached(fake, np.ones((11, 2), dtype=np.float32))


def test_pipeline_long_binary_uses_explicit_diagonal_state_without_spectrum(monkeypatch):
    from fudan_wgs_toolkit.pipeline import PheWASPipeline, AnalysisOptions
    from fudan_wgs_toolkit.profiling import StageProfiler
    model = fitted_state()
    pipeline = PheWASPipeline.__new__(PheWASPipeline)
    pipeline.options = AnalysisOptions(long_mask_threshold=2, memory_limit_gib=20)
    pipeline.local_mask_reuse_counters = {}
    pipeline.profiler = StageProfiler('cpu', enabled=False)
    expected = (object(), object())
    called = []
    def bounded(self, actual, host):
        called.append((actual, host))
        return expected
    monkeypatch.setattr(PheWASPipeline, '_long_mask_products', bounded)
    host = np.ones((model.n, 3), dtype=np.float32)
    assert pipeline._hybrid_host_products(model, host) is expected
    assert called == [(model, host)] and not hasattr(model, 'spectrum')
    assert pipeline.local_mask_reuse_counters['hybrid_long_host_masks'] == 1


@pytest.mark.parametrize('change', ['spa', 'fp64', 'missing_precision'])
def test_pipeline_rejects_unsupported_binary_before_long_backend(monkeypatch, change):
    from fudan_wgs_toolkit.pipeline import PheWASPipeline, AnalysisOptions
    model = fitted_state()
    if change == 'spa':model.use_spa = True
    elif change == 'fp64':model.matmul_mode = 'fp64'
    else:model.precision = None
    pipeline = PheWASPipeline.__new__(PheWASPipeline)
    pipeline.options = AnalysisOptions(long_mask_threshold=2, memory_limit_gib=20)
    monkeypatch.setattr(PheWASPipeline, '_long_mask_products', lambda *args: pytest.fail('unsupported backend'))
    with pytest.raises(NotImplementedError):
        pipeline._hybrid_host_products(model, np.ones((model.n, 3), dtype=np.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA component comparison')
@pytest.mark.parametrize('layout', ['C', 'F'])
@pytest.mark.parametrize('m', [513, 4097])
def test_cuda_binary_cached_matches_mature_tiled_geometry(layout, m):
    if torch.cuda.get_device_capability(0)[0] < 8:
        pytest.skip('native TF32 requires Ampere or newer')
    pytest.importorskip('triton')
    from fudan_wgs_toolkit import tf32
    tf32.configure_tf32(memory_limit_gib=20, split_k=0)
    model = fitted_state(device='cuda:0', n=257)
    genotype = np.array(np.random.default_rng(67).integers(0, 3, size=(model.n, m)), dtype=np.float32, order=layout)
    u0, v0 = model.score_covariance_tiled(genotype, variant_tile_size=512)
    for panel in (None, 4096):
        u, v, report = model.score_covariance_cached(genotype, panel_variant_size=panel,
            memory_limit_gib=20, profile=True)
        assert torch.equal(u, u0) and torch.equal(v, v0)
        assert report['matmul_module'] == 'fudan_wgs_toolkit.binary_null'
        assert report['symmetry'] == 'average' and report['preparation_tile_size'] == 512
        assert report['covariance_d2h_bytes'] == 0
        assert report['max_observed_allocated_bytes'] <= 20 * 2**30
        del u, v
