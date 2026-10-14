"""Resident preparation CPU/actual-CUDA contracts, not performance claims."""
from contextlib import contextmanager
import weakref

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit import null_model, tf32
from fudan_wgs_toolkit._gaussian_resident_shared import shared_gaussian_resident


class Broker:
    def __init__(self): self.callbacks, self.guards = [], []
    def _guard(self, required): self.guards.append(required)
    @contextmanager
    def optional_device_cache(self, release):
        self.callbacks.append(release)
        try: yield
        finally: self.callbacks.remove(release)


def model(n=37, q=3, *, shift=0., device='cpu', diagonal=False):
    generator = torch.Generator().manual_seed(1729)
    x = torch.randn(n, q, generator=generator, dtype=torch.float32).to(device)
    x[:, 0] = 1
    x[:, -1] += shift
    inverse = torch.linspace(.7, 1.3+shift, n, device=device)
    spectrum = null_model.KinshipSpectrum(torch.ones(n, device=device),
        [] if diagonal else [(torch.tensor([2, n-2], device=device),
                             torch.tensor([[.8, -.6], [.6, .8]], device=device))])
    px = x*inverse[:, None]
    return null_model.GaussianNullModel(np.arange(n).astype(str), x,
        torch.ones(n, device=device), torch.zeros(q, device=device),
        torch.tensor([1., 2.], device=device), torch.tensor([1., 2.], device=device),
        torch.linalg.inv(x.T@px), spectrum, inverse, px, 1, True, matmul_mode='tf32')


def genotype(n=37, m=7, *, device='cpu', transposed=False):
    generator = torch.Generator().manual_seed(314159)
    value = torch.randn((m, n) if transposed else (n, m), generator=generator).to(device)
    return value.T if transposed else value


def original_covariance(fitted, g, *, rotated=None, cross=None):
    rotated = fitted.spectrum.rotate(g, matmul_mode='tf32') if rotated is None else rotated
    cross = tf32.matmul(fitted.precision_x.T, g, mode='tf32') if cross is None else cross
    value = (tf32.matmul(rotated.T, fitted.inverse_variance[:, None]*rotated, mode='tf32')
             -tf32.matmul(tf32.matmul(cross.T, fitted.fixed_effect_covariance, mode='tf32'), cross, mode='tf32'))
    return (value+value.T)/2


@pytest.fixture
def cpu_products(monkeypatch):
    calls = []
    def product(a, b, *, mode):
        assert mode == 'tf32' and a.device.type == b.device.type == 'cpu'
        calls.append((a.clone(), b.clone(), tf32._native_route(a.shape, b.shape)))
        # Per-output ordered FP32 CPU oracle. It is not a CUDA RNA/MMA claim.
        result = torch.zeros((a.shape[0], b.shape[1]))
        for start in range(0, a.shape[1], 32):
            block = torch.zeros_like(result)
            for k in range(start, min(start+32, a.shape[1])):
                block += a[:, k, None]*b[k, None, :]
            result += block
        return result
    monkeypatch.setattr(tf32, 'matmul', product)
    monkeypatch.setattr(null_model, 'matmul', product)
    return calls


@pytest.mark.parametrize('transposed', [False, True])
def test_shared_rotation_and_exact_input_rows_keep_independent_covariance(cpu_products, transposed):
    models, g, broker = [model(), model(shift=.2), model(q=4, shift=.4)], genotype(transposed=transposed), Broker()
    controls = [original_covariance(fitted, g) for fitted in models]
    before = [(fitted.precision_x.clone(), fitted.inverse_variance.clone(),
               fitted.fixed_effect_covariance.clone()) for fitted in models]
    report = {}; cpu_products.clear()
    with shared_gaussian_resident(models[0], g, broker, capacity_bytes=2**20, report=report) as prep:
        crosses = prep.batched_cross(models)
        assert len(cpu_products) == 1
        packed, right, route = cpu_products[0]
        assert route == 'tf32_mma' and packed.shape == (10, 37)
        assert torch.equal(packed, torch.cat(tuple(fitted.precision_x.T for fitted in models)))
        assert torch.equal(right, g)
        with prep.pin():
            rotated = [prep.rotated_for(fitted) for fitted in models]
            assert all(value is rotated[0] for value in rotated)
            for fitted, cross, expected in zip(models, crosses, controls):
                torch.testing.assert_close(original_covariance(fitted, g, rotated=rotated[0], cross=cross),
                                           expected, rtol=0, atol=0)
        assert prep.storage_bytes <= 2**20
    metrics = report['gene_shared_resident_preparation']
    assert metrics['rotation_preparations'] == metrics['cross_batches'] == 1
    assert metrics['rotation_hits'] == 2 and metrics['cross_columns'] == 3
    assert metrics['storage_bytes_after_close'] == metrics['pinned_after_close'] == 0
    assert not metrics['covariance_shared'] and not metrics['precision_shared']
    assert not metrics['collection_forced'] and not broker.callbacks
    for fitted, states in zip(models, before):
        for actual, expected in zip((fitted.precision_x, fitted.inverse_variance,
                                     fitted.fixed_effect_covariance), states):
            assert torch.equal(actual, expected)


@pytest.mark.parametrize('n,q,m', [(37, 3, 1), (37, 1, 7), (1, 3, 7)])
def test_original_singleton_routes_do_not_become_tf32_matrix_products(cpu_products, n, q, m):
    # q>n cannot provide an invertible fitted C; a valid stored state with
    # independent small fixed C suffices to test the existing product route.
    if n == 1:
        models = [model(37, q=q, diagonal=True), model(37, q=q, shift=.2, diagonal=True)]
        for fitted in models:
            fitted.sample_ids = fitted.sample_ids[:1]
            fitted.x = fitted.x[:1]
            fitted.precision_x = fitted.precision_x[:1]
            fitted.inverse_variance = fitted.inverse_variance[:1]
            fitted.spectrum.eigenvalues = fitted.spectrum.eigenvalues[:1]
    else: models = [model(n, q=q), model(n, q=q, shift=.2)]
    g, broker, report = genotype(n, m), Broker(), {}
    expected = [tf32.matmul(fitted.precision_x.T, g, mode='tf32') for fitted in models]
    cpu_products.clear()
    with shared_gaussian_resident(models[0], g, broker, capacity_bytes=2**20, report=report) as prep:
        actual = prep.batched_cross(models)
        assert len(cpu_products) == 2
        assert all(route != 'tf32_mma' for _, _, route in cpu_products)
        for value, control in zip(actual, expected): torch.testing.assert_close(value, control, rtol=0, atol=0)
    assert report['gene_shared_resident_preparation']['cross_batches'] == 0
    assert report['gene_shared_resident_preparation']['cross_mature_products'] == 2


def test_stack_admission_is_before_cat_and_falls_back_without_math_change(cpu_products, monkeypatch):
    models, g, broker = [model(), model(shift=.2)], genotype(), Broker()
    report = {}
    broker._guard = lambda required: (_ for _ in ()).throw(MemoryError('no extra stack space'))
    monkeypatch.setattr(torch, 'cat', lambda *a, **k: pytest.fail('stack allocated before admission'))
    with shared_gaussian_resident(models[0], g, broker, capacity_bytes=2**20, report=report) as prep:
        assert prep.batched_cross(models) == (None, None)
        assert prep.rotated_for(models[0]) is None
    assert report['gene_shared_resident_preparation']['cross_admission_fallbacks'] == 1
    assert not cpu_products


def test_capacity_refuses_optional_owners_before_allocation(cpu_products):
    models, g, broker = [model(), model(shift=.2)], genotype(), Broker()
    with shared_gaussian_resident(models[0], g, broker, capacity_bytes=0) as prep:
        assert prep.rotated_for(models[0]) is None
        assert prep.batched_cross(models) == (None, None)
        assert prep.storage_bytes == 0 and not cpu_products


def test_diagonal_rotation_aliases_borrowed_genotype_without_cache_credit(cpu_products):
    fitted, g, broker = model(diagonal=True), genotype(), Broker()
    with shared_gaussian_resident(fitted, g, broker, capacity_bytes=0) as prep:
        assert prep.rotated_for(fitted) is g
        assert prep.storage_bytes == 0
        assert prep.release_unpinned() == 0
        assert torch.equal(g, genotype())


def test_pin_preserves_active_objects_and_pressure_recomputes_afterward(cpu_products):
    models, g, broker = [model(), model(shift=.2)], genotype(), Broker()
    with shared_gaussian_resident(models[0], g, broker, capacity_bytes=2**20) as prep:
        crosses = prep.batched_cross(models)
        with prep.pin():
            rotated = prep.rotated_for(models[0])
            before = rotated.clone()
            assert broker.callbacks[0]() == 0
            assert torch.equal(rotated, before)
            with pytest.raises(RuntimeError, match='pinned'): prep.close()
        assert broker.callbacks[0]() > 0 and prep.storage_bytes == 0
        with prep.pin():
            new = prep.rotated_for(models[1])
            torch.testing.assert_close(new, before, rtol=0, atol=0)
        assert all(value is not None for value in crosses)  # Real borrowed tensor storage remains valid.
    assert not broker.callbacks


@pytest.mark.parametrize('change', ['axis', 'rotation', 'genotype', 'precision', 'mode', 'inference', 'gradient'])
def test_changed_or_uncertified_state_selects_mature_fallback(cpu_products, change):
    models, g, broker = [model(), model(shift=.2)], genotype(), Broker()
    with shared_gaussian_resident(models[0], g, broker, capacity_bytes=2**20) as prep:
        assert prep.supports(models[1])
        if change == 'axis': models[1].sample_ids = models[1].sample_ids[::-1].copy()
        elif change == 'rotation': models[1].spectrum.blocks[0][1].add_(.01)
        elif change == 'genotype': g.add_(.01)
        elif change == 'precision': models[1].precision_x = models[1].precision_x.double()
        elif change == 'mode': models[1].matmul_mode = 'fp64'
        elif change == 'gradient': models[1].precision_x.requires_grad_(True)
        else:
            with torch.inference_mode(): models[1].inverse_variance = models[1].inverse_variance.clone()
        assert not prep.supports(models[1])
        assert prep.batched_cross(models) == (None, None)
        assert prep.rotated_for(models[1]) is None


def test_weak_model_retirement_and_close_remove_all_optional_owned_refs(cpu_products):
    first, second, g, broker = model(), model(shift=.2), genotype(), Broker()
    model_ref, state_ref = weakref.ref(first), weakref.ref(first.precision_x)
    g_ref, broker_ref = weakref.ref(g), weakref.ref(broker)
    prep = shared_gaussian_resident(first, g, broker, capacity_bytes=2**20)
    prep.batched_cross((first, second))
    rotated = prep.rotated_for(first); rotated_ref = weakref.ref(rotated); del rotated
    del first
    assert model_ref() is None and state_ref() is None
    assert prep.supports(second)
    del g, broker
    assert g_ref() is not None and broker_ref() is not None
    prep.close()
    assert g_ref() is None and broker_ref() is None and rotated_ref() is None
    assert prep.storage_bytes == 0 and prep._spectrum is prep._axis is None


def test_no_unknown_runtime_error_is_converted_to_fallback(cpu_products, monkeypatch):
    models, g, broker = [model(), model(shift=.2)], genotype(), Broker()
    retained = None
    report = {}
    monkeypatch.setattr(tf32, 'matmul', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('real kernel failure')))
    with shared_gaussian_resident(models[0], g, broker, capacity_bytes=2**20, report=report) as prep:
        try: prep.batched_cross(models)
        except RuntimeError as error: retained = error
        assert retained is not None
        assert prep._cross_output is None
    assert report['gene_shared_resident_preparation']['storage_bytes_after_close'] == 0
    assert not broker.callbacks


def test_actual_shape_workspace_and_bk32_contract(monkeypatch):
    models = [model(1033, q=16, shift=i*.01) for i in range(16)]
    g, broker, report = genotype(1033, 257), Broker(), {}
    observed = []
    def mock_product(a, b, *, mode):
        observed.append((tuple(a.shape), tuple(b.shape), mode))
        return torch.zeros((a.shape[0], b.shape[1]))
    monkeypatch.setattr(tf32, 'matmul', mock_product)
    with shared_gaussian_resident(models[0], g, broker, capacity_bytes=16*2**20, report=report) as prep:
        result = prep.batched_cross(models)
        assert observed == [((256, 1033), (1033, 257), 'tf32')]
        assert broker.guards[0] == 4*1033*256+4*256*257
        assert all(value.shape == (16, 257) for value in result)
    assert tf32._native_geometry(256, 257, 1033)[2] == tf32._native_geometry(16, 257, 1033)[2] == 32
    assert report['gene_shared_resident_preparation']['cross_geometry_changes'] == 16


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
                   reason='native TF32 product comparison requires Ampere CUDA')
@pytest.mark.parametrize('transposed', [False, True])
@pytest.mark.parametrize('m', [1, 513])
def test_actual_cuda_preparation_preserves_independent_full_covariance(transposed, m):
    models = [model(1536, q=16, shift=i*.01, device='cuda:0') for i in range(16)]
    g, broker, report = genotype(1536, m, device='cuda:0', transposed=transposed), Broker(), {}
    expected_cross = [tf32.matmul(fitted.precision_x.T, g, mode='tf32') for fitted in models]
    with shared_gaussian_resident(models[0], g, broker, capacity_bytes=64*2**20, report=report) as prep:
        crosses = prep.batched_cross(models)
        with prep.pin():
            for fitted, cross, expected in zip(models, crosses, expected_cross):
                torch.testing.assert_close(cross, expected, rtol=0, atol=0)
                actual = original_covariance(fitted, g, rotated=prep.rotated_for(fitted), cross=cross)
                control = original_covariance(fitted, g)
                torch.testing.assert_close(actual, control, rtol=0, atol=0)
    assert report['gene_shared_resident_preparation']['rotation_preparations'] == 1
    assert report['gene_shared_resident_preparation']['storage_bytes_after_close'] == 0
    assert not broker.callbacks
