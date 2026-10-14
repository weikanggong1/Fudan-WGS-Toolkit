"""Numerical and ownership contracts; synthetic CPU fixtures are not benchmarks."""
import gc
import weakref

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.binary import binary_spa
from fudan_wgs_toolkit.binary_null import ImmutableFittedBinarySource
from fudan_wgs_toolkit.phewas_models import prepare_phewas_model
from fudan_wgs_toolkit.phewas_storage import ModelRepository, save_model_store


def repository_fixture(tmp_path, *, cache_limit_bytes=None):
    n = 64
    fitted = prepare_phewas_model((np.arange(n) % 7 == 0).astype(float),
        np.arange(n).astype(str), np.column_stack((np.ones(n), np.linspace(-1, 1, n))),
        association_mode='fp64')
    entry = save_model_store(fitted.model, tmp_path/'model', sample_rows=np.arange(n),
        fit_metadata={}, spa_model=fitted.spa_model)
    return ModelRepository([entry], cache_limit_bytes=cache_limit_bytes), fitted.spa_model


def test_repeated_spa_acquire_reuses_exact_fixed_projections_and_accounts_storage(tmp_path, monkeypatch):
    repository, original = repository_fixture(tmp_path)
    calls = []
    projection = ImmutableFittedBinarySource._projection
    def count(source):
        calls.append(id(source))
        return projection(source)
    monkeypatch.setattr(ImmutableFittedBinarySource, '_projection', count)
    pointers, pvalues = [], []
    for _ in range(3):
        with repository.acquire_spa(0, 'cpu') as model:
            torch.testing.assert_close(model.xw, original.xw, rtol=0, atol=0)
            torch.testing.assert_close(model.projection_left, original.projection_left, rtol=0, atol=0)
            pointers.append((model.precision_x.data_ptr(), model.projection_left.data_ptr()))
            g = torch.tensor(np.column_stack((np.arange(64) % 5 == 0,
                np.arange(64) % 9 == 0)), dtype=torch.float64)
            z = g-model.projection_left@(model.xw@g)
            pvalues.append(binary_spa(g.T@model.scaled_residuals, z, model.fitted_probability).pvalues)
            source = repository.spa_sources[(0, 'cpu')]
            stored = ModelRepository._owned_storages(source)
            assert repository._entry_storages[('spa', 0, 'cpu')] == stored
            assert repository._cache_bytes(torch.device('cpu')) == sum(stored.values())
            observed = repository.resident_storage_summary()['resident_storage_bytes_by_device']['cpu']
            assert observed == sum(stored.values())
    assert len(calls) == 1 and len(set(pointers)) == 1
    for actual in pvalues[1:]:
        torch.testing.assert_close(actual, pvalues[0], rtol=0, atol=0)
    assert repository.summary()['spa_projection_cache_builds'] == 1
    assert repository.summary()['spa_projection_cache_hits'] == 2


@pytest.mark.parametrize('mutation', ['source', 'derived', 'binding', 'geometry', 'device'])
def test_projection_cache_rejects_changed_source_or_derived_storage(tmp_path, mutation):
    repository, _ = repository_fixture(tmp_path)
    with repository.acquire_spa(0, 'cpu'):
        pass
    source = repository.spa_sources[(0, 'cpu')]
    cache = source._spa_projection_cache
    if mutation == 'source':
        source._tensors[1].add_(1)
    elif mutation == 'derived':
        cache['values'][0].add_(1)
    elif mutation == 'binding':
        cache['binding'] = 'f'*64
    elif mutation == 'geometry':
        cache['values'] = (cache['values'][0][:1], cache['values'][1])
    else:
        cache['values'] = (cache['values'][0].to('meta'), cache['values'][1])
    with pytest.raises(ValueError, match='mutated|rebound|geometry|device'):
        with repository.acquire_spa(0, 'cpu'):
            pytest.fail('changed projection was yielded')


def test_projection_cache_follows_source_lru_budget_and_releases_tensor_owners(tmp_path):
    repository, original = repository_fixture(tmp_path, cache_limit_bytes=0)
    for _ in range(2):
        with repository.acquire_spa(0, 'cpu') as model:
            sx = weakref.ref(model.precision_x)
            left = weakref.ref(model.projection_left)
            torch.testing.assert_close(model.projection_left, original.projection_left, rtol=0, atol=0)
            assert repository._cache_bytes() > 0  # Pinned until this context ends.
        del model
        gc.collect()
        assert sx() is None and left() is None
        assert repository._cache_bytes() == 0 and not repository.spa_sources
    assert repository.summary()['spa_projection_cache_builds'] == 2
    assert repository.summary()['spa_projection_cache_hits'] == 0
    assert repository.summary()['cache_budget_evictions'] >= 2


def test_projection_admission_precedes_first_cached_allocation(tmp_path, monkeypatch):
    repository, _ = repository_fixture(tmp_path)
    with repository._acquire_binary_source(0, 'cpu') as source:
        assert not source.spa_projection_cached()
    before = repository._cache_bytes()
    original = repository._admit
    def admit(device, **kwargs):
        if kwargs.get('phase') == 'binary_spa_derivation':
            assert kwargs['fitted_bytes'] == 2*64*2*8
            raise MemoryError('fixed projection denied before allocation')
        return original(device, **kwargs)
    monkeypatch.setattr(repository, '_admit', admit)
    with pytest.raises(MemoryError, match='before allocation'):
        with repository.acquire_spa(0, 'cpu'):
            pytest.fail('denied projection yielded a model')
    assert not source.spa_projection_cached()
    assert repository._cache_bytes() == before
    assert repository.spa_pinned[(0, 'cpu')] == 0
