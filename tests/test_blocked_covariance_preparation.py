"""Preparation reuse contracts; synthetic fixtures are not performance evidence."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit import null_model, tf32
from fudan_wgs_toolkit._blocked_covariance import (
    _PanelPreparation, _storage_bytes, blocked_gaussian_covariance,
)


class Broker:
    def __init__(self):
        self.guards, self.callbacks = [], []

    def _guard(self, required):
        self.guards.append(required)

    @contextmanager
    def optional_device_cache(self, release):
        self.callbacks.append(release)
        try:
            yield
        finally:
            self.callbacks.remove(release)


def fitted_model(n, device='cpu', multiplier=1.):
    generator = torch.Generator().manual_seed(1729)
    x = torch.randn(n, 3, generator=generator).to(device)
    inverse = torch.linspace(.7, 1.3, n, device=device)*multiplier
    rotation = torch.tensor([[.8, -.6], [.6, .8]], device=device)
    spectrum = null_model.KinshipSpectrum(torch.ones(n, device=device),
        [(torch.tensor([3, n-2], device=device), rotation)])
    return SimpleNamespace(device=torch.device(device), x=x,
        inverse_variance=inverse, spectrum=spectrum,
        precision_x=x*inverse[:, None],
        fixed_effect_covariance=torch.linalg.inv(x.T@(x*inverse[:, None])))


def genotype(n, m):
    generator = np.random.default_rng(1729)
    return np.asfortranarray(generator.uniform(0., 2., (n, m)).astype(np.float32))


def original_pair_products(model, host, width):
    """Frozen directed-panel formula, including its original tail shapes."""
    m = host.shape[1]
    output = torch.empty((m, m), device=model.device)
    for begin in range(0, m, width):
        end = min(begin+width, m)
        left = torch.as_tensor(host[:, begin:end], device=model.device)
        rotated_left = model.spectrum.rotate(left, matmul_mode='tf32')
        cross_left = tf32.matmul(model.precision_x.T, left, mode='tf32')
        projected_left = tf32.matmul(cross_left.T, model.fixed_effect_covariance, mode='tf32')
        for other in range(begin, m, width):
            stop = min(other+width, m)
            if begin == other:
                right, rotated_right, cross_right = left, rotated_left, cross_left
            else:
                right = torch.as_tensor(host[:, other:stop], device=model.device)
                rotated_right = model.spectrum.rotate(right, matmul_mode='tf32')
                cross_right = tf32.matmul(model.precision_x.T, right, mode='tf32')
            block = (tf32.matmul(rotated_left.T, model.inverse_variance[:, None]*rotated_right, mode='tf32')
                     - tf32.matmul(projected_left, cross_right, mode='tf32'))
            if begin == other:
                block = (block+block.T)/2
            else:
                reverse = (tf32.matmul(rotated_right.T, model.inverse_variance[:, None]*rotated_left, mode='tf32')
                           - tf32.matmul(tf32.matmul(cross_right.T, model.fixed_effect_covariance, mode='tf32'),
                                         cross_left, mode='tf32'))
                block = (block+reverse.T)/2
            output[begin:end, other:stop] = block
            if begin != other:
                output[other:stop, begin:end] = block.T
    return output


@pytest.fixture
def cpu_oracle(monkeypatch):
    def product(a, b, *, mode='tf32'):
        assert mode == 'tf32' and a.device.type == b.device.type == 'cpu'
        return a@b
    monkeypatch.setattr(tf32, 'matmul', product)
    monkeypatch.setattr(null_model, 'matmul', product)


def test_cached_preparation_preserves_relatedness_and_independent_precision(cpu_oracle):
    host, model, broker = genotype(23, 9), fitted_model(23), Broker()
    control = original_pair_products(model, host, 3)
    reports = []
    for capacity in (0, 2**20):
        report = {}
        actual = blocked_gaussian_covariance(model, host, broker,
            variant_tile_size=3, capacity_bytes=capacity, report=report)
        torch.testing.assert_close(actual, control, rtol=0, atol=0)
        reports.append(report['gene_panel_preparation_cache'])
        assert not broker.callbacks
    assert reports[0]['panel_preparations'] == 6
    assert reports[1]['panel_preparations'] == 3
    assert reports[1]['projection_preparations'] == 3
    assert reports[1]['panel_hits'] == 3
    assert reports[1]['storage_bytes_after_close'] == 0
    assert reports[1]['pinned_panels_after_close'] == 0
    assert reports[1]['peak_storage_bytes'] <= reports[1]['capacity_bytes']
    second_model = fitted_model(23, multiplier=1.7)
    second = blocked_gaussian_covariance(second_model, host, broker,
        variant_tile_size=3, capacity_bytes=2**20)
    torch.testing.assert_close(second, original_pair_products(second_model, host, 3), rtol=0, atol=0)
    assert not torch.equal(second, control)
    model.spectrum.blocks = []
    assert not torch.allclose(control, original_pair_products(model, host, 3))


def test_pressure_retirement_keeps_borrowed_panel_until_scope_exit(cpu_oracle):
    cache = _PanelPreparation(fitted_model(23), genotype(23, 9), Broker(), 3, 2**20)
    with cache.borrow(6, 9) as cold:
        pass
    with cache.borrow(0, 3) as active:
        expected = tuple(t.clone() for t in active)
        assert cache.release_unpinned() > 0
        assert set(cache.panels) == {0} and set(cache.projections) == {0}
        for value, reference in zip(active, expected):
            torch.testing.assert_close(value, reference)
        assert cache.pins == {0: 1}
    assert cache.release_unpinned() > 0 and not cache.panels and not cache.projections
    cache.close()
    assert cache.model is cache.host is cache.broker is None


def test_true_owned_storage_not_view_logical_sizes():
    storage = torch.ones(100)
    assert _storage_bytes((storage[:3], storage[80:], storage)) == 400
    assert _storage_bytes((storage, storage.clone())) == 800


def test_guard_failure_closes_mask_owners_with_retained_traceback(cpu_oracle, monkeypatch):
    instances = []
    original_init = _PanelPreparation.__init__
    def record(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        instances.append(self)
    monkeypatch.setattr(_PanelPreparation, '__init__', record)
    broker = Broker()
    def reject(required):
        if instances:
            raise MemoryError('no legal live workspace')
    broker._guard = reject
    retained = None
    try:
        blocked_gaussian_covariance(fitted_model(23), genotype(23, 9), broker,
            variant_tile_size=3, capacity_bytes=2**20)
    except MemoryError as error:
        retained = error
    assert retained is not None
    cache = instances[0]
    assert not cache.panels and not cache.projections and not cache.pins
    assert cache.panel_bytes == cache.projection_bytes == 0
    assert cache.model is cache.host is cache.broker is None
    assert not broker.callbacks


@pytest.mark.parametrize('capacity', [1, 200, 1024])
def test_small_cache_falls_back_without_skipping_products(cpu_oracle, capacity):
    model, host, report = fitted_model(23), genotype(23, 9), {}
    result = blocked_gaussian_covariance(model, host, Broker(), variant_tile_size=3,
        capacity_bytes=capacity, report=report)
    torch.testing.assert_close(result, original_pair_products(model, host, 3), rtol=0, atol=0)
    metrics = report['gene_panel_preparation_cache']
    assert metrics['peak_storage_bytes'] <= capacity
    assert metrics['storage_bytes_after_close'] == 0


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
                   reason='native TF32 comparison requires an Ampere CUDA device')
@pytest.mark.parametrize('m', [513, 734, 767, 1536])
def test_native_cuda_keeps_original_directed_product_and_tail_geometry(m):
    model, host, broker = fitted_model(1536, 'cuda:0'), genotype(1536, m), Broker()
    control = original_pair_products(model, host, 512)
    for capacity in (0, 32*2**20):
        report = {}
        actual = blocked_gaussian_covariance(model, host, broker,
            capacity_bytes=capacity, report=report)
        torch.testing.assert_close(actual, control, rtol=0, atol=0)
        assert report['gene_panel_preparation_cache']['storage_bytes_after_close'] == 0
        assert not broker.callbacks
