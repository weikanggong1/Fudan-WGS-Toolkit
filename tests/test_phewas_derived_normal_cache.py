"""Bounded cache contracts for normals derived from original binary fits.

These CPU tests check state, lifetimes and admission. They do not benchmark
native TF32 throughput or substitute for real binary association validation.
"""
import gc
import weakref

import numpy as np
import pytest
import torch

import fudan_wgs_toolkit.binary_null as binary_null
import fudan_wgs_toolkit.phewas_storage as storage
from fudan_wgs_toolkit.phewas_models import prepare_phewas_model


def _entry(tmp_path, monkeypatch, *, sparse=False):
    # Formula/storage oracle; the association state still has its native FP32
    # tensors. CPU execution is not presented as an actual TF32 benchmark.
    monkeypatch.setattr(binary_null, 'matmul', lambda left, right, mode: left @ right)
    n = 48
    design = np.column_stack((np.ones(n), np.linspace(-1, 1, n)))
    fitted = prepare_phewas_model((np.arange(n) % 3 == 0).astype(float),
        np.arange(n).astype(str), design, family='binomial', device='cpu')
    if sparse:
        for model in (fitted.model, fitted.spa_model):
            model.precision = torch.diag(model.precision).to_sparse().coalesce()
            model.has_kinship = True
    entry = storage.save_model_store(fitted.model, tmp_path/'model',
        sample_rows=np.arange(n), fit_metadata={}, spa_model=fitted.spa_model)
    return entry, fitted


def test_warm_binary_normal_avoids_projection_and_cast_but_keeps_spa_lazy(tmp_path, monkeypatch):
    entry, fitted = _entry(tmp_path, monkeypatch)
    repository = storage.ModelRepository([entry])
    original_normal = binary_null.ImmutableFittedBinarySource.normal
    original_projection = binary_null.ImmutableFittedBinarySource._projection
    calls = {'normal': 0, 'projection': 0}
    def normal(source, **kwargs):
        calls['normal'] += 1
        return original_normal(source, **kwargs)
    def projection(source):
        calls['projection'] += 1
        return original_projection(source)
    monkeypatch.setattr(binary_null.ImmutableFittedBinarySource, 'normal', normal)
    monkeypatch.setattr(binary_null.ImmutableFittedBinarySource, '_projection', projection)
    genotype = torch.tensor((np.arange(48) % 2)[:, None], dtype=torch.float32)
    with repository.acquire([0], 'cpu') as models:
        first = models[0]
        assert first.projection_left is None and not first.use_spa
        score, covariance = first.score_covariance(genotype)
        for actual, expected in zip((score, covariance), fitted.model.score_covariance(genotype)):
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
        with repository.acquire([0], 'cpu') as nested:
            assert nested[0] is first and repository.pinned[(0, 'cpu')] == 2
    for _ in range(5):
        with repository.acquire([0], 'cpu') as models:
            assert models[0] is first
            for actual, expected in zip(models[0].score_covariance(genotype), (score, covariance)):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == {'normal': 1, 'projection': 1}
    with repository.acquire_spa(0, 'cpu') as spa:
        assert spa.x.dtype == torch.float64 and spa.use_spa
        torch.testing.assert_close(spa.projection_left, fitted.spa_model.projection_left,
            rtol=1e-12, atol=1e-12)
    assert calls == {'normal': 1, 'projection': 2}
    summary = repository.summary()
    assert summary['binary_normal_derivations'] == 1
    assert summary['binary_normal_cache_hits'] == 6
    assert summary['resident_derived_binary_normals'] == 1
    assert repository.pinned[(0, 'cpu')] == repository.spa_pinned[(0, 'cpu')] == 0
    repository.close()
    assert repository.summary()['resident_derived_binary_normals'] == 0


@pytest.mark.parametrize('mutation', [
    'residual', 'design', 'precision', 'projection', 'sample_ids', 'source_sha', 'mode'])
def test_cached_normal_rejects_caller_mutation(tmp_path, monkeypatch, mutation):
    entry, _ = _entry(tmp_path, monkeypatch)
    repository = storage.ModelRepository([entry])
    with repository.acquire([0], 'cpu') as models:
        model = models[0]
    if mutation == 'sample_ids':
        model.sample_ids = np.frombuffer(model.sample_ids.tobytes(), dtype=model.sample_ids.dtype)
    elif mutation == 'source_sha':
        model.null_fit_source_sha256 = 'f'*64
    elif mutation == 'mode':
        model.matmul_mode = 'fp64'
    else:
        field = {'residual': 'scaled_residuals', 'design': 'x',
                 'precision': 'precision', 'projection': 'precision_x'}[mutation]
        getattr(model, field).add_(1)
    with pytest.raises(ValueError, match='cached derived binary normal fitted state was mutated'):
        with repository.acquire([0], 'cpu'):
            pytest.fail('mutated normal was admitted')
    assert repository.pinned[(0, 'cpu')] == 0


def test_normal_cache_does_not_keep_retired_fp64_source_alive(tmp_path, monkeypatch):
    entry, _ = _entry(tmp_path, monkeypatch)
    repository = storage.ModelRepository([entry])
    with repository.acquire([0], 'cpu'):
        source_ref = weakref.ref(repository.spa_sources[(0, 'cpu')])
    repository._retire_entry(('spa', 0, 'cpu'))
    gc.collect()
    assert source_ref() is None
    assert repository._derived_normal_guards[(0, 'cpu')]['source']() is None
    with repository.acquire([0], 'cpu') as models:
        assert models[0].x.dtype == torch.float32
    assert repository.summary()['binary_normal_derivations'] == 1
    assert repository.summary()['binary_normal_cache_hits'] == 1
    assert repository.summary()['resident_spa_sources'] == 0


@pytest.mark.parametrize('mutation', ['tensor', 'source_sha'])
def test_external_source_is_guarded_after_leaving_bank(tmp_path, monkeypatch, mutation):
    entry, _ = _entry(tmp_path, monkeypatch)
    repository = storage.ModelRepository([entry])
    with repository.acquire([0], 'cpu'):
        external_source = repository.spa_sources[(0, 'cpu')]
    repository._retire_entry(('spa', 0, 'cpu'))
    if mutation == 'tensor':
        external_source._tensors[1].add_(1)
    else:
        # Constructor sealing rejects normal assignment; bypass only to
        # demonstrate that even externally corrupted private state is caught.
        object.__setattr__(external_source, '_binding', 'f'*64)
    with pytest.raises(ValueError, match='source was mutated|source identity changed'):
        with repository.acquire([0], 'cpu'):
            pytest.fail('normal whose still-live source changed was admitted')
    assert repository.pinned[(0, 'cpu')] == 0


def test_cached_normal_hit_releases_pin_when_admission_fails(tmp_path, monkeypatch):
    entry, _ = _entry(tmp_path, monkeypatch)
    repository = storage.ModelRepository([entry])
    with repository.acquire([0], 'cpu') as models:
        first = models[0]
        original_admit = repository._admit
        def reject_hit(device, **kwargs):
            if kwargs.get('phase') == 'binary_normal_cache_hit':
                assert repository.pinned[(0, 'cpu')] == 2
                assert kwargs['protect'] == (('normal', 0, 'cpu'),)
                repository._evict_one(device=torch.device('cpu'), protect=kwargs['protect'])
                assert (0, 'cpu') in repository.models
                raise MemoryError('injected warm admission failure')
            return original_admit(device, **kwargs)
        monkeypatch.setattr(repository, '_admit', reject_hit)
        with pytest.raises(MemoryError, match='warm admission failure'):
            with repository.acquire([0], 'cpu'):
                pytest.fail('admission failure must propagate')
        assert repository.pinned[(0, 'cpu')] == 1
        assert repository.models[(0, 'cpu')] is first
    assert repository.pinned[(0, 'cpu')] == 0
    assert all(value == 0 for value in repository.spa_pinned.values())


def test_zero_budget_only_keeps_active_normal_and_drops_fp64_source_before_yield(tmp_path, monkeypatch):
    entry, _ = _entry(tmp_path, monkeypatch)
    source_refs = []
    original_normal = binary_null.ImmutableFittedBinarySource.normal
    def normal(source, **kwargs):
        source_refs.append(weakref.ref(source))
        return original_normal(source, **kwargs)
    monkeypatch.setattr(binary_null.ImmutableFittedBinarySource, 'normal', normal)
    repository = storage.ModelRepository([entry], cache_limit_bytes=0)
    for _ in range(2):
        with repository.acquire([0], 'cpu') as models:
            gc.collect()
            assert source_refs[-1]() is None
            assert repository.spa_sources == {}
            assert repository.pinned[(0, 'cpu')] == 1
            assert repository._cache_bytes('cpu') > 0
            assert repository._cache_bytes('cpu') == repository.resident_storage_summary()[
                'resident_storage_bytes_by_device']['cpu']
            assert models[0].x.dtype == torch.float32
        assert repository._cache_bytes('cpu') == 0
        assert repository.models == {} and repository._derived_normal_guards == {}
    assert repository.summary()['binary_normal_derivations'] == 2


def test_positive_shared_budget_preserves_pinned_normal_and_evicts_inactive_normals(tmp_path, monkeypatch):
    entry, _ = _entry(tmp_path, monkeypatch)
    repository = storage.ModelRepository([entry, entry.copy()])
    with repository.acquire([0], 'cpu'):
        pass
    repository._retire_entry(('spa', 0, 'cpu'))
    one_normal_bytes = repository._cache_bytes('cpu')
    assert one_normal_bytes > 0
    repository.cache_limit_bytes = one_normal_bytes
    with repository.acquire([0], 'cpu') as outer:
        retained = outer[0]
        with repository.acquire([1], 'cpu'):
            assert repository.pinned[(0, 'cpu')] == 1
            assert repository.pinned[(1, 'cpu')] == 1
            # Two concurrently owned states may exceed an otherwise bounded
            # budget; an active caller must not observe a retired fitted state.
            assert repository._cache_bytes('cpu') > one_normal_bytes
        assert repository.models[(0, 'cpu')] is retained
        assert (1, 'cpu') not in repository.models
        assert repository._cache_bytes('cpu') <= one_normal_bytes
    with repository.acquire([1], 'cpu'):
        assert (0, 'cpu') not in repository.models
    assert repository._cache_bytes('cpu') <= one_normal_bytes
    assert repository.summary()['resident_derived_binary_normals'] == 1
    assert repository.summary()['cache_budget_active_overflows'] > 0
    assert all(value == 0 for value in repository.pinned.values())
    assert all(value == 0 for value in repository.spa_pinned.values())


def test_sparse_derived_normal_accounts_indices_and_aliased_projection_once(tmp_path, monkeypatch):
    entry, _ = _entry(tmp_path, monkeypatch, sparse=True)
    repository = storage.ModelRepository([entry])
    with repository.acquire([0], 'cpu') as models:
        source = repository.spa_sources[(0, 'cpu')]
        normal = models[0]
        assert normal.precision.layout == torch.sparse_coo
        assert normal.precision.dtype == torch.float32
        assert source._tensors[4].dtype == torch.float64
        # Casting COO may copy its index array as well as values. Both arrays
        # belong in the physical ledger; the xw transpose must count once.
        for precision in (normal.precision, source._tensors[4]):
            for tensor in (precision.indices(), precision.values()):
                pointer = tensor.untyped_storage().data_ptr()
                assert ('cpu', pointer) in repository._storage_owners
        assert normal.xw.untyped_storage().data_ptr() == normal.precision_x.untyped_storage().data_ptr()
        assert repository._cache_bytes('cpu') == repository.resident_storage_summary()[
            'resident_storage_bytes_by_device']['cpu']
        before = repository._cache_bytes('cpu')
    repository._retire_entry(('spa', 0, 'cpu'))
    assert 0 < repository._cache_bytes('cpu') < before
    with repository.acquire([0], 'cpu') as models:
        assert models[0] is normal
    repository._retire_entry(('normal', 0, 'cpu'))
    assert repository._cache_bytes('cpu') == 0 and not repository._storage_owners


def test_reloaded_spa_source_must_match_cached_normal_binding_and_failed_load_is_retired(tmp_path, monkeypatch):
    entry, _ = _entry(tmp_path, monkeypatch)
    repository = storage.ModelRepository([entry])
    with repository.acquire([0], 'cpu'):
        pass
    repository._retire_entry(('spa', 0, 'cpu'))
    original_load = storage.load_spa_source
    def different_binding(*args, **kwargs):
        arrays = original_load(*args, **kwargs)
        arrays['null_fit_source_sha256'] = arrays['verified_fit_source_sha256'] = 'f'*64
        return arrays
    monkeypatch.setattr(storage, 'load_spa_source', different_binding)
    with pytest.raises(ValueError, match='normal and SPA fitted-source bindings differ'):
        with repository.acquire_spa(0, 'cpu'):
            pytest.fail('SPA source with a different fitted identity was admitted')
    assert (0, 'cpu') not in repository.spa_sources
    assert repository._cache_bytes('cpu') == repository.resident_storage_summary()[
        'resident_storage_bytes_by_device']['cpu']
    assert all(value == 0 for value in repository.spa_pinned.values())
    with repository.acquire([0], 'cpu'):
        pass
