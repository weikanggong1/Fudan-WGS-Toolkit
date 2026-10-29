"""Selected Single SPA admission; bounded fixtures are not WGS benchmarks."""
from contextlib import contextmanager
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.binary import binary_single_phewas
from fudan_wgs_toolkit.phewas_models import prepare_phewas_model
from fudan_wgs_toolkit.phewas_runtime import runtime
from fudan_wgs_toolkit.phewas_storage import ModelRepository
from test_phewas_workspace import entry, simulated_cuda


@pytest.fixture(params=['cpu', 'cuda'])
def single_fit(request):
    device = request.param
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA native FP32/TF32 normal and FP64 SPA products')
    n = 64
    x = np.column_stack((np.ones(n), np.linspace(-1, 1, n)))
    y = np.tile([0., 1., 1., 0., 1., 0., 0., 1.], 8)
    prepared = prepare_phewas_model(y, np.arange(n).astype(str), x,
        device=device, association_mode='tf32' if device == 'cuda' else 'fp64')
    g = torch.zeros((n, 9), dtype=prepared.model.x.dtype, device=device)
    for j in range(9):
        g[(j*5+np.arange(6)) % n, j] = 1
    return prepared, g


def supplied_normal(g, selected):
    scores = torch.linspace(.2, 1.8, g.shape[1], dtype=g.dtype, device=g.device)
    variance = torch.ones_like(scores)
    # Equality is deliberately not selected: the public predicate remains P<.05.
    nominal = torch.full((g.shape[1],), .05, dtype=torch.float64, device=g.device)
    if selected:
        nominal[list(selected)] = .01
    return dict(normal_score=scores, normal_variance=variance,
        normal_pvalues=nominal)


def assert_same_output(actual, expected):
    assert set(actual) == set(expected)
    for name, value in expected.items():
        if name == 'spa':
            for field in fields(value):
                torch.testing.assert_close(getattr(actual[name], field.name),
                    getattr(value, field.name), rtol=0, atol=0, equal_nan=True)
        elif isinstance(value, torch.Tensor):
            assert actual[name].dtype == value.dtype and actual[name].device == value.device
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0, equal_nan=True)
        else:
            assert actual[name] == value


@pytest.mark.parametrize('selected', [(), (2,), (0, 3, 7), tuple(range(9))])
def test_selected_context_receives_strict_actual_count_and_preserves_every_output(single_fit, selected):
    prepared, g = single_fit
    normal = supplied_normal(g, selected)
    expected = binary_single_phewas(prepared.model, g, spa_model=prepared.spa_model, **normal)
    events = []
    @contextmanager
    def acquire(count):
        assert type(count) is int
        events.append(('enter', count))
        try:
            yield prepared.spa_model
        finally:
            events.append(('exit', count))
    actual = binary_single_phewas(prepared.model, g, spa_acquire_selected=acquire, **normal)
    assert events == ([('enter', len(selected)), ('exit', len(selected))] if selected else [])
    assert actual['spa_selected'].nonzero(as_tuple=True)[0].tolist() == list(selected)
    assert_same_output(actual, expected)


def test_original_zero_argument_callback_remains_unmodified(single_fit):
    prepared, g = single_fit
    normal = supplied_normal(g, (1, 6))
    calls = []
    @contextmanager
    def acquire():
        calls.append('enter')
        try:
            yield prepared.spa_model
        finally:
            calls.append('exit')
    expected = binary_single_phewas(prepared.model, g, spa_model=prepared.spa_model, **normal)
    actual = binary_single_phewas(prepared.model, g, spa_acquire=acquire, **normal)
    assert calls == ['enter', 'exit']
    assert_same_output(actual, expected)


@pytest.mark.parametrize('protocol', ['legacy', 'selected'])
def test_callback_typeerror_propagates_once_without_signature_retry(single_fit, protocol):
    prepared, g = single_fit
    error = TypeError('real callback failure must not become an arity probe')
    calls = []
    @contextmanager
    def context():
        calls.append('enter')
        raise error
        yield prepared.spa_model
    def selected_context(count):
        assert count == 1
        return context()
    kwargs = ({'spa_acquire': context} if protocol == 'legacy'
              else {'spa_acquire_selected': selected_context})
    with pytest.raises(TypeError) as caught:
        binary_single_phewas(prepared.model, g, **kwargs, **supplied_normal(g, (4,)))
    assert caught.value is error and calls == ['enter']


@pytest.mark.parametrize('conflict', ['sidecar', 'legacy', 'not_callable'])
def test_selected_protocol_is_explicit_and_rejects_ambiguous_factories(single_fit, conflict):
    prepared, g = single_fit
    called = []
    @contextmanager
    def context(*args):
        called.append(args)
        yield prepared.spa_model
    kwargs = {'spa_acquire_selected': context}
    if conflict == 'sidecar':
        kwargs['spa_model'] = prepared.spa_model
    elif conflict == 'legacy':
        kwargs['spa_acquire'] = context
    else:
        kwargs['spa_acquire_selected'] = 7
    with pytest.raises(ValueError, match='one SPA sidecar or one callable SPA context factory'):
        binary_single_phewas(prepared.model, g, **kwargs, **supplied_normal(g, ()))
    assert called == []


@pytest.mark.parametrize('selected', [(), (2,), tuple(range(9))])
def test_actual_selection_drives_existing_workspace_estimate_not_solver_tile(single_fit, selected, monkeypatch):
    prepared, g = single_fit
    events = []
    class Repository:
        @contextmanager
        def acquire_spa(self, index, device):
            events.append('source')
            yield prepared.spa_model
    broker = SimpleNamespace(device=g.device)
    @contextmanager
    def workspace(repository, owner, required):
        assert owner is broker
        events.append(required)
        yield SimpleNamespace(ensure_free=lambda **kwargs: events.append('compute'))
    monkeypatch.setattr(runtime, '_gene_workspace_scope', workspace)
    normal = supplied_normal(g, selected)
    expected = binary_single_phewas(prepared.model, g, spa_model=prepared.spa_model, **normal)
    actual = binary_single_phewas(prepared.model, g,
        spa_acquire_selected=lambda count: runtime._binary_single_spa_acquire(
            Repository(), broker, 0, n=prepared.model.n, m=count, q=prepared.model.x.shape[1]),
        **normal)
    assert_same_output(actual, expected)
    if not selected:
        assert events == []
        return
    required = runtime._binary_single_spa_workspace_new_bytes(prepared.model.n,
        len(selected), prepared.model.x.shape[1])
    full = runtime._binary_single_spa_workspace_new_bytes(prepared.model.n,
        g.shape[1], prepared.model.x.shape[1])
    assert events == [required, 'source', 'compute']
    assert (required < full) if len(selected) < g.shape[1] else (required == full)


def test_sparse_selection_admits_real_repository_guard_while_full_bound_still_rejects(monkeypatch):
    gib = 2**30
    repository = ModelRepository([], workspace_bytes=0)
    repository.models[(0, 'cuda:0')] = entry(gib//4)
    repository.pinned[(0, 'cuda:0')] = 1
    state = simulated_cuda(monkeypatch, repository, total=3*gib)
    state['external'] = gib//4  # Simulated still-live current G/core; never evicted.
    admissions = []
    @contextmanager
    def acquire_spa(index, device):
        repository._admit(device, fitted_bytes=gib//4, phase='test_fitted_source')
        admissions.append(index)
        state['external'] += gib//4
        try:
            yield 'verified-sidecar'
        finally:
            state['external'] -= gib//4
    repository.acquire_spa = acquire_spa
    broker = SimpleNamespace(device=torch.device('cuda:0'), _reserve=gib//2,
        release_device_cache=lambda: None, active=None)
    @contextmanager
    def workspace_guard(callback):
        previous = broker.active
        broker.active = callback
        try:
            yield
        finally:
            broker.active = previous
    broker.workspace_guard = workspace_guard
    with repository.memory_policy('cuda:0', limit_bytes=3*gib, reserve_bytes=gib//2):
        with runtime._binary_single_spa_acquire(repository, broker, 7, n=300000, m=1, q=16):
            assert callable(broker.active)
        assert broker.active is None and not repository._workspace_reservations
        assert list(repository.models) == [(0, 'cuda:0')]  # Pinned model is retained.
        with pytest.raises(MemoryError, match='pinned states are retained'):
            with runtime._binary_single_spa_acquire(repository, broker, 7, n=300000, m=256, q=16):
                pytest.fail('all-selected full bound was incorrectly reduced')
    assert admissions == [7]
    assert state['external'] == gib//4
    assert broker.active is None and not repository._workspace_reservations


def test_selected_estimate_retains_repository_static_floor(monkeypatch):
    gib = 2**30
    repository = ModelRepository([], workspace_bytes=8*gib)
    simulated_cuda(monkeypatch, repository, total=7*gib)
    with repository.memory_policy('cuda:0', limit_bytes=7*gib, reserve_bytes=0):
        with pytest.raises(MemoryError, match='pinned states are retained'):
            with repository.reserve_workspace('cuda:0',
                    runtime._binary_single_spa_workspace_new_bytes(64, 1, 2), reserve_bytes=0):
                pytest.fail('sparse selection bypassed original bank headroom')
    assert not repository._workspace_reservations
