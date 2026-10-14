"""Shared preparation contracts; fixtures are not real-data performance proof."""
from contextlib import contextmanager
import weakref

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit import null_model, tf32
from fudan_wgs_toolkit._blocked_covariance import blocked_gaussian_covariance
from fudan_wgs_toolkit._gaussian_covariance_shared import shared_gaussian_panels


class Broker:
    def __init__(self):
        self.callbacks = []
        self.guards = []

    def _guard(self, size):
        self.guards.append(size)

    @contextmanager
    def optional_device_cache(self, callback):
        self.callbacks.append(callback)
        try:
            yield
        finally:
            self.callbacks.remove(callback)


def model(n=23, *, shift=0., device='cpu'):
    row = torch.arange(n, dtype=torch.float32, device=device)
    x = torch.stack((torch.ones_like(row), row/n,
                     torch.sin(row*.17)+shift*torch.cos(row*.29)), 1)
    spectrum = null_model.KinshipSpectrum(torch.ones_like(row),
        [(torch.tensor([3, n-2], device=device),
          torch.tensor([[.8, -.6], [.6, .8]], device=device))])
    inverse = .7+row/n+shift*.2
    # Arbitrary independent actual fitted precision: no constant singleton
    # or common precision hypothesis enters the shared preparation helper.
    px = x*inverse[:, None]
    return null_model.GaussianNullModel(
        np.arange(n).astype(str), x, torch.sin(row*.31), torch.zeros(3, device=device),
        torch.tensor([1., 2.], device=device), torch.tensor([2., 3.], device=device),
        torch.linalg.inv(x.T@px), spectrum, inverse, px, 2, True, matmul_mode='tf32')


def genotype(n=23, m=10, *, fortran=False):
    host = np.random.default_rng(1729).uniform(0, 2, (n, m)).astype(np.float32)
    return np.asfortranarray(host) if fortran else host


@pytest.fixture
def cpu_products(monkeypatch):
    def product(a, b, *, mode):
        assert mode == 'tf32' and a.device.type == b.device.type == 'cpu'
        return a@b
    monkeypatch.setattr(tf32, 'matmul', product)
    monkeypatch.setattr(null_model, 'matmul', product)


def covariance(fitted, host, broker, bank=None, *, capacity=2**20, width=3, report=None):
    kwargs = {} if bank is None else {'panel_cache_factory': bank.panel_cache_factory}
    return blocked_gaussian_covariance(fitted, host, broker, variant_tile_size=width,
        capacity_bytes=capacity, report=report, **kwargs)


@pytest.mark.parametrize('fortran', [False, True])
def test_different_nonconstant_precisions_preserve_every_mature_product(cpu_products, fortran):
    models, host, broker = [model(), model(shift=.7)], genotype(fortran=fortran), Broker()
    control = [covariance(m, host, broker) for m in models]
    assert not torch.equal(control[0], control[1])
    report = {}
    with shared_gaussian_panels(models[0], host, broker, variant_tile_size=3,
                                capacity_bytes=2**20, report=report) as bank:
        for fitted, expected in zip(models, control):
            model_report = {}
            actual = covariance(fitted, host, broker, bank, report=model_report)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            metrics = model_report['gene_panel_preparation_cache']
            assert metrics['projection_preparations'] == 4
            assert metrics['storage_bytes_after_close'] == 0  # View-owned only.
            assert metrics['shared_raw_rotated_storage_bytes_at_model_end'] > 0
            assert not bank._cache.projections and bank._cache.model is None
            assert broker.callbacks == [bank.release_unpinned]
        assert bank._cache.metrics['panel_preparations'] == 4
        assert bank.storage_bytes > 0
    metrics = report['gene_shared_panel_preparation_cache']
    assert metrics['model_scopes'] == 2 and metrics['panel_preparations'] == 4
    assert metrics['projection_preparations'] == 8
    assert metrics['storage_bytes_after_close'] == metrics['pinned_panels_after_close'] == 0
    assert metrics['peak_storage_bytes'] <= metrics['capacity_bytes']
    assert not metrics['covariance_shared'] and not metrics['precision_shared']
    assert not metrics['model_projection_shared'] and not metrics['collection_forced']
    assert not broker.callbacks


@pytest.mark.parametrize('capacity', [0, 1, 200, 1024, 2**20])
def test_identical_capacity_bounds_never_skip_covariance(cpu_products, capacity):
    first, second, host, broker = model(), model(shift=.2), genotype(), Broker()
    report = {}
    with shared_gaussian_panels(first, host, broker, variant_tile_size=3,
                                capacity_bytes=capacity, report=report) as bank:
        for fitted in (first, second):
            actual = covariance(fitted, host, broker, bank, capacity=capacity)
            expected = covariance(fitted, host, broker, capacity=capacity)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert bank.storage_bytes <= capacity
    metrics = report['gene_shared_panel_preparation_cache']
    assert metrics['peak_storage_bytes'] <= capacity
    if capacity == 0:
        assert metrics['panel_hits'] == 0
        assert metrics['panel_preparations'] == 20


def test_cache_can_shrink_and_recover_with_same_live_guard(cpu_products):
    first, host, broker = model(), genotype(), Broker()
    report = {}
    with shared_gaussian_panels(first, host, broker, variant_tile_size=3,
                                capacity_bytes=2**20, report=report) as bank:
        for capacity in (2**20, 0, 1024):
            actual = covariance(first, host, broker, bank, capacity=capacity)
            torch.testing.assert_close(actual, covariance(first, host, broker, capacity=capacity),
                                       rtol=0, atol=0)
            assert bank.storage_bytes <= capacity
    assert report['gene_shared_panel_preparation_cache']['peak_storage_bytes'] <= 2**20


def test_partial_bank_preserves_products_without_increasing_capacity(cpu_products):
    models, host, broker = [model(), model(shift=.2)], genotype(m=19), Broker()
    capacity = 1000  # Deliberately too small for the whole mask's panels.
    baseline_preparations = 0
    expected = []
    for fitted in models:
        report = {}
        expected.append(covariance(fitted, host, broker, capacity=capacity, report=report))
        baseline_preparations += report['gene_panel_preparation_cache']['panel_preparations']
    report = {}
    with shared_gaussian_panels(models[0], host, broker, variant_tile_size=3,
                                capacity_bytes=capacity, report=report) as bank:
        for fitted, control in zip(models, expected):
            torch.testing.assert_close(covariance(fitted, host, broker, bank, capacity=capacity),
                                       control, rtol=0, atol=0)
            assert 0 < bank.storage_bytes <= capacity
    metrics = report['gene_shared_panel_preparation_cache']
    # A tight partial bank can evict all useful retained panels while making
    # room for the next model's independent projection. No benefit is assumed.
    assert metrics['panel_preparations'] <= baseline_preparations
    assert metrics['capacity_bytes'] == capacity
    assert metrics['peak_storage_bytes'] <= capacity


def test_pressure_release_never_retires_active_borrowed_panels(cpu_products):
    fitted, host, broker = model(), genotype(), Broker()
    with shared_gaussian_panels(fitted, host, broker, variant_tile_size=3) as bank:
        view = bank.panel_cache_factory(fitted, host, broker, 3, 2**20)
        with view.borrow(6, 9):
            pass
        with view.borrow(0, 3) as active:
            before = tuple(value.clone() for value in active)
            assert view.release_unpinned() > 0
            assert set(bank._cache.panels) == {0} and set(bank._cache.projections) == {0}
            for actual, expected in zip(active, before):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert view.pins == {0: 1}
            with pytest.raises(RuntimeError, match='borrowed panels'):
                view.close()
        view.close()
        assert bank._cache.projection_bytes == 0
        assert bank.release_unpinned() > 0 and bank.storage_bytes == 0


def test_mask_pressure_callback_remains_during_statistical_tail(cpu_products):
    first, second, host, broker = model(), model(shift=.2), genotype(), Broker()
    with shared_gaussian_panels(first, host, broker, variant_tile_size=3) as bank:
        covariance(first, host, broker, bank)
        assert bank._cache.model is None and bank.storage_bytes > 0
        # This is between covariance models, when the original model-local
        # callback has already ended and the tail can need a large workspace.
        assert len(broker.callbacks) == 1
        assert broker.callbacks[0]() > 0 and bank.storage_bytes == 0
        actual = covariance(second, host, broker, bank)
        torch.testing.assert_close(actual, covariance(second, host, broker), rtol=0, atol=0)
    assert not broker.callbacks


@pytest.mark.parametrize('change', ['axis', 'rows', 'rotation', 'eigenvalues', 'mode', 'dtype', 'spa'])
def test_exact_certificate_refuses_incompatible_state_and_factory_falls_back(cpu_products, change):
    first, second, host, broker = model(), model(shift=.2), genotype(), Broker()
    if change == 'axis': second.sample_ids = second.sample_ids[::-1].copy()
    elif change == 'rows': second.spectrum.blocks[0][0].add_(1)
    elif change == 'rotation': second.spectrum.blocks[0][1].add_(.001)
    elif change == 'eigenvalues': second.spectrum.eigenvalues[0] += .1
    elif change == 'mode': second.matmul_mode = 'fp64'
    elif change == 'dtype': second.precision_x = second.precision_x.double()
    else: second.use_spa = True
    with shared_gaussian_panels(first, host, broker, variant_tile_size=3) as bank:
        assert not bank.supports(second, host)
        cache = bank.panel_cache_factory(second, host, broker, 3, 2**20)
        assert cache.model is second  # Mature unrelated-model fallback.
        cache.close()
        assert bank._fallback_scopes == 1


def test_certificate_compares_separate_arrays_once_and_uses_shared_identity(cpu_products, monkeypatch):
    first, second, host, broker = model(), model(shift=.2), genotype(), Broker()
    calls = []
    original_equal = np.array_equal
    monkeypatch.setattr(np, 'array_equal', lambda a, b: calls.append(1) or original_equal(a, b))
    with shared_gaussian_panels(first, host, broker, variant_tile_size=3) as bank:
        assert bank.supports(second, host)
        assert bank.supports(second, host)
        assert len(calls) == 1
        second.sample_ids = first.sample_ids
        assert bank.supports(second, host)
        assert len(calls) == 1
        assert not bank.supports(second, host.copy())
        second.spectrum.blocks[0][1].add_(.1)
        assert not bank.supports(second, host)


@pytest.mark.parametrize('target', ['basis', 'projection', 'inverse'])
def test_tensor_mutation_rejects_stale_view_and_releases_with_retained_exception(cpu_products, target):
    fitted, host, broker = model(), genotype(), Broker()
    fitted_ref, x_ref = weakref.ref(fitted), weakref.ref(fitted.x)
    retained = None
    with shared_gaussian_panels(fitted, host, broker, variant_tile_size=3) as bank:
        view = bank.panel_cache_factory(fitted, host, broker, 3, 2**20)
        with view.borrow(0, 3): pass
        if target == 'basis': fitted.spectrum.blocks[0][1].add_(.01)
        elif target == 'projection': fitted.precision_x.add_(.01)
        else: fitted.inverse_variance.add_(.01)
        try:
            view._guard(1)
        except ValueError as error:
            retained = error
        assert retained is not None
        view.close()
        del fitted
        assert fitted_ref() is None and x_ref() is None
    assert bank.storage_bytes == 0


def test_anchor_and_model_state_are_not_retained_between_views(cpu_products):
    first, host, broker = model(), genotype(), Broker()
    second = model(shift=.2)
    first_ref, x_ref, host_ref, broker_ref = map(weakref.ref, (first, first.x, host, broker))
    bank = shared_gaussian_panels(first, host, broker, variant_tile_size=3)
    covariance(first, host, broker, bank)
    del first
    assert first_ref() is None and x_ref() is None
    assert bank.supports(second, host)
    covariance(second, host, broker, bank)
    del host, broker
    assert host_ref() is not None and broker_ref() is not None
    bank.close()
    assert host_ref() is None and broker_ref() is None
    assert bank._spectrum is bank._axis is bank._cache.model is None


def test_inference_state_without_mutation_counter_uses_mature_fallback(cpu_products):
    fitted, host, broker = model(), genotype(), Broker()
    with torch.inference_mode():
        fitted.inverse_variance = fitted.inverse_variance.clone()
    with shared_gaussian_panels(fitted, host, broker, variant_tile_size=3) as bank:
        assert not bank.supports(fitted, host)
        cache = bank.panel_cache_factory(fitted, host, broker, 3, 0)
        assert cache.model is fitted
        cache.close()


def test_failed_allocation_closes_model_view_and_mask_owners(cpu_products):
    fitted, host, broker = model(), genotype(), Broker()
    report = {}; retained = None
    with shared_gaussian_panels(fitted, host, broker, variant_tile_size=3, report=report) as bank:
        def reject(size):
            if bank._active_view is not None:
                raise MemoryError('no legal live workspace')
        broker._guard = reject
        try:
            covariance(fitted, host, broker, bank)
        except MemoryError as error:
            retained = error
        assert retained is not None
        assert bank._active_view is None and not bank._cache.projections
        assert bank._cache.model is None and broker.callbacks == [bank.release_unpinned]
    assert report['gene_shared_panel_preparation_cache']['storage_bytes_after_close'] == 0


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
                   reason='native TF32 contract requires an Ampere CUDA device')
@pytest.mark.parametrize('fortran', [False, True])
@pytest.mark.parametrize('m', [513, 734])
def test_native_cuda_bitexact_different_weights_and_original_tail_shapes(fortran, m):
    models = [model(1536, device='cuda:0'), model(1536, shift=.7, device='cuda:0')]
    host, broker = genotype(1536, m, fortran=fortran), Broker()
    report = {}
    control = [covariance(fitted, host, broker, width=512, capacity=32*2**20) for fitted in models]
    with shared_gaussian_panels(models[0], host, broker, capacity_bytes=32*2**20,
                                report=report) as bank:
        for fitted, expected in zip(models, control):
            actual = covariance(fitted, host, broker, bank, width=512, capacity=32*2**20)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    metrics = report['gene_shared_panel_preparation_cache']
    assert metrics['panel_preparations'] == 2 and metrics['projection_preparations'] == 4
    assert metrics['storage_bytes_after_close'] == 0 and not broker.callbacks
