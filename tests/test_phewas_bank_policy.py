"""Ownership/LRU contracts; these small fixtures are not throughput evidence."""
import gc
import errno
import json
from pathlib import Path
from types import SimpleNamespace
import weakref
from collections import OrderedDict

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.phewas_storage import ModelRepository, digest_file


class _ExplicitStorageEntry(SimpleNamespace):
    """Compatible bank owner exposing all of this fixture's retained tensors."""
    def explicit_storage_owners(self):
        return vars(self)


def tensor_entry(value, *, shared=None):
    return _ExplicitStorageEntry(x=value if shared is None else shared, residual=value)


def insert(repository, namespace, index, value, *, device='cpu'):
    key = (index, device)
    bank = repository.models if namespace == 'normal' else repository.spa_sources
    bank[key] = value
    repository._touch_entry(namespace, key)
    return key


def test_unified_lru_retires_cold_spa_before_recent_normal():
    repository = ModelRepository([], workspace_bytes=0)
    cold = insert(repository, 'spa', 0, tensor_entry(torch.ones(11)))
    hot = insert(repository, 'normal', 1, tensor_entry(torch.ones(13)))
    assert repository._evict_one(device=torch.device('cpu'))
    assert cold not in repository.spa_sources and hot in repository.models
    assert repository.summary()['fitted_cache_unique_storage_bytes'] == 13*4


def test_namespace_protection_does_not_pin_other_state_of_same_trait():
    repository = ModelRepository([], workspace_bytes=0)
    key = insert(repository, 'normal', 0, tensor_entry(torch.ones(11)))
    insert(repository, 'spa', 0, tensor_entry(torch.ones(13)))
    assert repository._evict_one(protect=(('spa', *key),))
    assert key not in repository.models and key in repository.spa_sources
    assert not repository._evict_one(protect=(key,))


def test_byte_budget_deduplicates_shared_design_and_lazy_projection():
    shared = torch.ones(20)
    repository = ModelRepository([], workspace_bytes=0, cache_limit_bytes=120)
    first = insert(repository, 'normal', 0, tensor_entry(torch.ones(5), shared=shared))
    second = insert(repository, 'normal', 1, tensor_entry(torch.ones(5), shared=shared))
    assert repository._cache_bytes('cpu') == 120  # X counted once, each residual once.
    repository._trim_cache('cpu')
    assert len(repository.models) == 2
    repository.models[first]._centered_single_projection_cache = (None, torch.ones(8))
    repository._refresh_entry('normal', first)
    assert repository._cache_bytes('cpu') == 152
    repository.pinned[first] = 1
    repository._trim_cache('cpu')
    assert first in repository.models and second not in repository.models
    assert repository._cache_bytes('cpu') == 132  # Shared X remains alive in first.
    repository.pinned[first] = 0
    repository._trim_cache('cpu')
    assert repository._cache_bytes('cpu') == 0 and not repository.models
    assert repository.summary()['cache_budget_evictions'] == 2


def test_hot_byte_ledger_does_not_iterate_banks_and_direct_insert_reconciles_once():
    class ObservedBank(OrderedDict):
        iterations = 0
        def __iter__(self):
            self.iterations += 1
            return super().__iter__()
    repository = ModelRepository([], workspace_bytes=0, cache_limit_bytes=4096)
    for i in range(40):
        insert(repository, 'normal', i, tensor_entry(torch.ones(8)))
    repository.models = ObservedBank(repository.models)
    repository.spa_sources = ObservedBank(repository.spa_sources)
    for _ in range(100):
        assert repository._cache_bytes('cpu') == 40*8*4
        repository._trim_cache('cpu')
    assert repository.models.iterations == repository.spa_sources.iterations == 0
    # Legacy/test callers can directly insert a compatible object. The one
    # reconciliation updates its ledger; subsequent guards remain constant-time.
    repository.models[(40, 'cpu')] = tensor_entry(torch.ones(3))
    assert repository._cache_bytes('cpu') == (40*8+3)*4
    first = (repository.models.iterations, repository.spa_sources.iterations)
    assert first == (1, 1)
    for _ in range(100):
        assert repository._cache_bytes('cpu') == (40*8+3)*4
    assert (repository.models.iterations, repository.spa_sources.iterations) == first


def test_live_cache_fraction_counts_owned_storage_once_and_restores_nested_limit(monkeypatch):
    repository = ModelRepository([], workspace_bytes=0)
    insert(repository, 'normal', 0, tensor_entry(torch.ones(25)), device='cuda:0')
    # The compatible bank fixture's tensor is CPU; bind it to the simulated
    # device in the accounting ledger without allocating CUDA.
    token = ('normal', 0, 'cuda:0')
    storage = next(iter(repository._entry_storages[token]))
    owners = repository._storage_owners.pop(storage)
    repository._storage_owners[('cuda:0', storage[1])] = owners
    repository._entry_storages[token] = {('cuda:0', storage[1]): 100}
    repository._cache_bytes_by_device = {'cuda:0': 100}
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (700, 1000))
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 300)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 400)
    with repository.memory_policy('cuda:0', limit_bytes=900, reserve_bytes=100,
                                  cache_live_fraction=.5):
        # min(700+100-100,900-300-100)+100 already-owned =600.
        assert repository._cache_limits[torch.device('cuda:0')] == 300
        with repository.memory_policy('cuda:0', cache_limit_bytes=250):
            assert repository._cache_limits[torch.device('cuda:0')] == 250
        assert repository._cache_limits[torch.device('cuda:0')] == 300
    assert repository._cache_limits == {}


def axis_entries(directory, axes):
    entries = []
    for index, rows in enumerate(axes):
        path = directory/str(index)
        path.mkdir(parents=True)
        np.save(path/'sample_rows.npy', np.asarray(rows, dtype=np.int64))
        (path/'model.private.json').write_text(json.dumps(dict(n=len(rows),
            sample_rows_sha256=digest_file(path/'sample_rows.npy'))))
        entries.append(dict(path=str(path), family='gaussian',
                            sha256=digest_file(path/'model.private.json')))
    return entries


def test_authenticated_mmap_axis_alias_survives_repository_close(tmp_path):
    entries = axis_entries(tmp_path, [[3, 1, 8], [3, 1, 8]])
    repository = ModelRepository(entries)
    rows = repository.descriptors()[0]['ordered_rows']
    assert isinstance(rows, np.memmap) and rows.mode == 'r'
    assert repository.descriptors()[1]['ordered_rows'] is rows
    with pytest.raises(ValueError):
        rows.setflags(write=True)
    pointer = weakref.ref(rows)
    repository.close()
    gc.collect()
    np.testing.assert_array_equal(rows, [3, 1, 8])
    assert pointer() is rows  # Never forcibly close a mapping used externally.
    del rows
    gc.collect()
    assert pointer() is None
    assert (Path(entries[0]['path'])/'sample_rows.npy').is_file()


def test_axis_mapping_fd_budget_falls_back_to_immutable_bytes(tmp_path):
    entries = axis_entries(tmp_path, [[3, 1, 8], [8, 1, 3], [3, 1, 8]])
    repository = ModelRepository(entries, sample_axis_mmap_limit=1)
    first, second, duplicate = [item['ordered_rows'] for item in repository.descriptors()]
    assert isinstance(first, np.memmap) and not isinstance(second, np.memmap)
    assert duplicate is first
    with pytest.raises(ValueError):
        second.setflags(write=True)
    report = repository.summary()
    assert report['sample_axis_open_mmaps'] == 1
    assert report['sample_axis_ram_bytes'] == second.nbytes
    assert report['sample_axis_mapped_logical_bytes'] == first.nbytes
    assert report['sample_axis_mapping_bytes_are_resident_memory'] is False


def test_active_bank_can_exceed_cache_budget_then_retires_after_consumer_error(tmp_path, monkeypatch):
    from fudan_wgs_toolkit import phewas_storage
    entries = axis_entries(tmp_path, [[0, 2, 3], [3, 2, 0]])
    repository = ModelRepository(entries, cache_limit_bytes=0)
    monkeypatch.setattr(phewas_storage, 'load_model_store',
                        lambda *args, **kwargs: tensor_entry(torch.ones(7)))
    with pytest.raises(RuntimeError, match='consumer'):
        with repository.acquire([0, 1], 'cpu') as models:
            assert len(models) == len(repository.models) == 2
            assert all(repository.pinned[key] == 1 for key in repository.models)
            assert repository._cache_bytes('cpu') == 56
            raise RuntimeError('consumer')
    assert not repository.models and repository._cache_bytes('cpu') == 0
    assert not any(repository.pinned.values())
    repository.close()


def test_spa_pin_is_released_when_cache_bookkeeping_rejects_before_yield(monkeypatch):
    repository = ModelRepository([], workspace_bytes=0)
    repository._descriptors = [dict(spa_available=True)]
    repository.spa_sources[(0, 'cpu')] = tensor_entry(torch.ones(7))
    def reject(*args):
        raise RuntimeError('bookkeeping')
    monkeypatch.setattr(repository, '_touch_entry', reject)
    with pytest.raises(RuntimeError, match='bookkeeping'):
        with repository._acquire_binary_source(0, 'cpu'):
            pytest.fail('failed cache setup yielded its state')
    assert repository.spa_pinned[(0, 'cpu')] == 0
    repository.close()


def test_axis_mapping_leaves_half_fd_budget_for_models_and_metadata(monkeypatch):
    monkeypatch.setattr(ModelRepository, '_file_descriptor_headroom', staticmethod(lambda: 264))
    assert ModelRepository._sample_axis_mapping_budget() == 100


def test_fd_admission_retires_only_cold_normal_and_preserves_active_maps(monkeypatch):
    repository = ModelRepository([], workspace_bytes=0)
    source = insert(repository, 'spa', 0, tensor_entry(torch.ones(5)))
    pinned = insert(repository, 'normal', 1, tensor_entry(torch.ones(7)))
    cold = insert(repository, 'normal', 2, tensor_entry(torch.ones(9)))
    repository.pinned[pinned] = 1
    monkeypatch.setattr(repository, '_file_descriptor_headroom',
                        lambda: 7 if cold in repository.models else 8)
    repository._admit_file_descriptors('cpu')
    assert source in repository.spa_sources and pinned in repository.models
    assert cold not in repository.models and repository.pinned[pinned] == 1
    assert repository.summary()['file_descriptor_evictions'] == 1


def test_emfile_retry_has_a_distinct_owner_eviction_and_never_spins(monkeypatch):
    repository = ModelRepository([], workspace_bytes=0)
    insert(repository, 'normal', 0, tensor_entry(torch.ones(7)))
    monkeypatch.setattr(repository, '_file_descriptor_headroom', lambda: 100)
    attempts = []
    def loader():
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError(errno.EMFILE, 'temporary file-descriptor race')
        return 'loaded'
    assert repository._load_with_file_descriptor_admission('cpu', loader) == 'loaded'
    assert len(attempts) == 2 and not repository.models
    assert repository.summary()['file_descriptor_retries'] == 1
    def exhausted():
        attempts.append(True)
        raise OSError(errno.EMFILE, 'external descriptor pressure')
    with pytest.raises(OSError) as error:
        repository._load_with_file_descriptor_admission('cpu', exhausted)
    assert error.value.errno == errno.EMFILE and len(attempts) == 3


def test_fd_pressure_with_only_pinned_states_is_an_explicit_error(monkeypatch):
    repository = ModelRepository([], workspace_bytes=0)
    key = insert(repository, 'normal', 0, tensor_entry(torch.ones(7)))
    repository.pinned[key] = 1
    monkeypatch.setattr(repository, '_file_descriptor_headroom', lambda: 1)
    with pytest.raises(OSError, match='active model and external mappings') as error:
        repository._admit_file_descriptors('cpu')
    assert error.value.errno == errno.EMFILE and key in repository.models


@pytest.mark.parametrize('option,value', [('cache_limit_bytes', True), ('cache_limit_bytes', -1),
    ('sample_axis_storage', 'writable'), ('sample_axis_mmap_limit', -1)])
def test_invalid_bank_budget_and_axis_options_are_rejected(option, value):
    with pytest.raises(ValueError):
        ModelRepository([], **{option: value})
