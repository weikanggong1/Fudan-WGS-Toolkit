"""Single SPA allocation/admission contracts; small fixtures are not benchmarks."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from fudan_wgs_toolkit.binary import binary_single_phewas
from fudan_wgs_toolkit.phewas_runtime import runtime
from fudan_wgs_toolkit.phewas_storage import ModelRepository
from test_binary_phewas_burden import fixture
from test_phewas_workspace import entry, simulated_cuda


@pytest.mark.parametrize('m', [1, 11, 512, 1024, 5000])
def test_statistical_spa_budget_covers_full_selected_tile_without_scaling_past_512(m):
    n, q, width = 61, 5, min(m, 512)
    # Original selected gather plus conservative independent peak slabs:
    # FP64 source/projection/scaling, active gathers, CGF derivatives/tails.
    tensors = [torch.empty((n, width), dtype=torch.float32)]
    tensors.extend(torch.empty((n, width), dtype=torch.float64) for _ in range(24))
    tensors.extend(torch.empty((q, width), dtype=torch.float64) for _ in range(4))
    actual = sum(value.untyped_storage().nbytes() for value in tensors)
    budget = runtime._binary_single_spa_workspace_new_bytes(n, m, q)
    assert budget >= actual + 256*2**20
    if m > 512:
        assert budget == runtime._binary_single_spa_workspace_new_bytes(n, 512, q)


def test_full_sample_single_spa_budget_fits_76_gib_with_native_pinned_tile():
    # Shape arithmetic only; no large allocation or GPU timing is performed.
    required = runtime._binary_single_spa_workspace_new_bytes(343034, 5000, 16)
    assert 32*2**30 < required < 34*2**30


def test_spa_context_evicts_cold_storage_then_reserves_fitted_and_statistical_state(monkeypatch):
    gib = 2**30
    repository = ModelRepository([], workspace_bytes=0)
    repository.models[(0, 'cuda:0')] = entry(10*gib)
    repository.models[(1, 'cuda:0')] = entry(35*gib)
    repository.pinned[(0, 'cuda:0')] = 1
    state = simulated_cuda(monkeypatch, repository, total=79*gib)
    state['external'] = 10*gib  # Already-live raw G and core storage.
    acquisitions = []
    @contextmanager
    def acquire_spa(index, device):
        acquisitions.append((index, device))
        assert repository._workspace_for(torch.device(device))[0] >= 32*gib
        repository._admit(device, fitted_bytes=5*gib, phase='test_fitted_source')
        state['external'] += 5*gib
        try:
            yield 'verified-fitted-sidecar'
        finally:
            state['external'] -= 5*gib
    repository.acquire_spa = acquire_spa
    releases = []
    broker = SimpleNamespace(device=torch.device('cuda:0'), _reserve=2*gib,
        release_device_cache=lambda: releases.append(True))
    @contextmanager
    def workspace_guard(admission):
        broker.active = admission
        try:
            yield
        finally:
            broker.active = None
    broker.workspace_guard = workspace_guard
    with repository.memory_policy('cuda:0', limit_bytes=76*gib, reserve_bytes=2*gib):
        with runtime._binary_single_spa_acquire(repository, broker, 7,
                n=343034, m=5000, q=16) as sidecar:
            assert sidecar == 'verified-fitted-sidecar'
            assert list(repository.models) == [(0, 'cuda:0')]
            assert repository.pinned[(0, 'cuda:0')] == 1
            assert callable(broker.active)
        assert repository.summary()['active_workspace_reservations'] == 0
        assert broker.active is None
    assert acquisitions == [(7, 'cuda:0')]
    assert repository.summary()['workspace_phase_counts']['single_binary_spa'] == 1


@pytest.mark.parametrize('failure', ['load', 'compute'])
def test_selected_spa_releases_workspace_and_owner_hook_on_exception(failure):
    events = []
    class Repository:
        @contextmanager
        def reserve_workspace(self, device, required, **kwargs):
            events.append('workspace-enter')
            try:
                yield SimpleNamespace(ensure_free=lambda **kwargs: None)
            finally:
                events.append('workspace-exit')
        @contextmanager
        def acquire_spa(self, index, device):
            events.append('source-enter')
            try:
                if failure == 'load':
                    raise RuntimeError('load failure')
                yield object()
            finally:
                events.append('source-exit')
    class Broker:
        device, _reserve = torch.device('cpu'), 0
        release_device_cache = staticmethod(lambda: None)
        @contextmanager
        def workspace_guard(self, admission):
            self.active = admission
            try:
                yield
            finally:
                self.active = None
    broker = Broker()
    with pytest.raises(RuntimeError, match=failure+' failure'):
        with runtime._binary_single_spa_acquire(Repository(), broker, 0, n=64, m=9, q=2):
            raise RuntimeError('compute failure')
    assert events == ['workspace-enter', 'source-enter', 'source-exit', 'workspace-exit']
    assert broker.active is None


@pytest.mark.parametrize('cutoff', [1e-12, 1.])
def test_actual_binary_single_keeps_probabilities_and_lazy_selection(cutoff):
    prepared, g, _, _ = fixture()
    expected = binary_single_phewas(prepared.model, g, spa_model=prepared.spa_model,
        p_filter_cutoff=cutoff)
    events = []
    class Repository:
        @contextmanager
        def acquire_spa(self, index, device):
            events.append('source')
            yield prepared.spa_model
    broker = SimpleNamespace(device=torch.device('cpu'), _reserve=0,
        _guard=lambda required: events.append(required))
    # Compatibility owner still checks the full future bytes before source load.
    # A real bank uses the eviction/pin boundary checked in the CUDA simulation.
    from unittest.mock import patch
    @contextmanager
    def workspace_scope(repository, broker, required):
        events.append(required)
        yield SimpleNamespace(ensure_free=lambda **kwargs: events.append('compute'))
    with patch.object(runtime, '_gene_workspace_scope', workspace_scope):
        actual = binary_single_phewas(prepared.model, g,
            spa_acquire=lambda: runtime._binary_single_spa_acquire(Repository(), broker, 0,
                n=prepared.model.n, m=g.shape[1], q=prepared.model.x.shape[1]),
            p_filter_cutoff=cutoff)
    for field in ('pvalues', 'pvalue_log10', 'normal_pvalues', 'spa_selected'):
        torch.testing.assert_close(actual[field], expected[field], rtol=0, atol=0)
    if expected['spa_selected'].any():
        assert events[1:] == ['source', 'compute']
    else:
        assert events == []
