"""Content checks and identity-axis ownership across cold fitted-bank reloads."""
import gc
import hashlib
import json
import os
from pathlib import Path
import weakref

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit import phewas_storage as storage
from fudan_wgs_toolkit.null_model import fit_gaussian_null


def _entries(tmp_path, count=2):
    n = 24
    entries = []
    for index in range(count):
        model = fit_gaussian_null(np.sin(np.arange(n)+index),
            sample_ids=np.arange(n).astype(str), covariates=np.ones((n, 1)),
            device='cpu', matmul_mode='fp64')
        entries.append(storage.save_model_store(model, tmp_path/f'model{index}',
            sample_rows=np.arange(n), fit_metadata={}))
    return entries


def _binary_entries(tmp_path, count=2):
    import copy
    from fudan_wgs_toolkit.binary_null import fit_logistic_null
    n = 24
    entries = []
    indices = torch.stack((torch.arange(n), torch.arange(n)))
    for index in range(count):
        full = fit_logistic_null((np.arange(n) % (3+index) == 0).astype(float),
            np.column_stack((np.ones(n), np.linspace(-1, 1, n))),
            sample_ids=np.arange(n).astype(str), device='cpu', use_spa=True)
        full.precision = torch.sparse_coo_tensor(indices,
            torch.linspace(0.5+index, 1+index, n, dtype=torch.float64), (n, n)).coalesce()
        full.precision_x = torch.sparse.mm(full.precision, full.x)
        full.xw = full.precision_x.T
        full.fixed_effect_covariance = torch.linalg.inv(full.x.T @ full.precision_x)
        full.projection_left = full.x @ full.fixed_effect_covariance
        full.null_fit_source_sha256 = 'a'*64
        normal = copy.deepcopy(full)
        normal.use_spa = False
        normal.projection_left = None
        entries.append(storage.save_model_store(normal, tmp_path/f'binary{index}',
            sample_rows=np.arange(n), fit_metadata={}, spa_model=full))
    return entries


def test_verification_hashes_once_after_repeated_identity_checks(tmp_path, monkeypatch):
    path = tmp_path/'immutable.npy'
    path.write_bytes(b'original fitted state')
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    calls = []
    original = storage.digest_file
    monkeypatch.setattr(storage, 'digest_file', lambda p: (calls.append(Path(p)), original(p))[1])
    first = storage.verify_file(path, expected)
    for _ in range(5):
        assert storage.verify_file(path, expected) == first
    assert calls == [path]
    with pytest.raises(ValueError, match='identity changed'):
        storage.verify_file(path, '0'*64)
    assert calls == [path]


@pytest.mark.parametrize('change', ['rewrite', 'replace', 'touch'])
def test_verification_rejects_changed_identity_even_if_contents_match(tmp_path, change):
    path = tmp_path/'immutable.npy'
    payload = b'same contents do not grant a new identity'
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    original = path.stat()
    storage.verify_file(path, expected)
    if change == 'rewrite':
        path.write_bytes(payload)
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    elif change == 'replace':
        replacement = tmp_path/'replacement'
        replacement.write_bytes(payload)
        replacement.replace(path)
    else:
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns+1_000_000))
    with pytest.raises(ValueError, match='identity changed'):
        storage.verify_file(path, expected)


def test_first_verification_rejects_write_during_scan(tmp_path, monkeypatch):
    path = tmp_path/'immutable.npy'
    path.write_bytes(b'initial bytes')
    expected = storage.digest_file(path)
    original = storage.digest_file
    def changed_during_scan(p):
        digest = original(p)
        Path(p).write_bytes(b'changed bytes of a different length')
        return digest
    monkeypatch.setattr(storage, 'digest_file', changed_during_scan)
    with pytest.raises(ValueError, match='during verification'):
        storage.verify_file(path, expected)
    assert str(path.resolve()) not in storage._VERIFIED_FILES


def test_first_verification_rejects_wrong_contents_and_does_not_admit(tmp_path):
    path = tmp_path/'immutable.npy'
    path.write_bytes(b'not the receipt-bound bytes')
    with pytest.raises(ValueError, match='checksum differs'):
        storage.verify_file(path, '0'*64)
    assert str(path.resolve()) not in storage._VERIFIED_FILES


def test_identical_id_axes_load_once_and_release_with_last_owner(tmp_path, monkeypatch):
    entries = _entries(tmp_path)
    repository = storage.ModelRepository(entries)
    original, id_reads = storage.readonly_npy_mmap, []
    def counted_load(path, *args, **kwargs):
        if Path(path).name == 'sample_ids.npy':
            id_reads.append(Path(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(storage, 'readonly_npy_mmap', counted_load)
    with repository.acquire([0, 1], 'cpu') as models:
        ids = models[0].sample_ids
        assert ids is models[1].sample_ids
        with pytest.raises(ValueError):
            ids.setflags(write=True)
        retained = weakref.ref(ids)
        assert len(id_reads) == 1
        summary = repository.summary()
        assert summary['live_deduplicated_sample_id_axes'] == 1
        assert summary['deduplicated_sample_id_bytes'] == ids.nbytes
    del ids, models
    repository.close()
    gc.collect()
    assert retained() is None
    assert repository.summary()['live_deduplicated_sample_id_axes'] == 0


def test_eviction_reloads_tensor_without_rehashing_unchanged_files(tmp_path, monkeypatch):
    entries = _entries(tmp_path, 1)
    repository = storage.ModelRepository(entries, cache_limit_bytes=0)
    original, hashes = storage.digest_file, []
    monkeypatch.setattr(storage, 'digest_file', lambda p: (hashes.append(str(p)), original(p))[1])
    with repository.acquire([0], 'cpu') as models:
        score = models[0].scaled_residuals.clone()
    del models
    first_hashes = list(hashes)
    assert first_hashes
    assert not repository.models
    with repository.acquire([0], 'cpu') as models:
        torch.testing.assert_close(models[0].scaled_residuals, score, rtol=0, atol=0)
    del models
    assert hashes == first_hashes
    repository.close()
    reopened = storage.ModelRepository(entries)
    with reopened.acquire([0], 'cpu') as models:
        torch.testing.assert_close(models[0].scaled_residuals, score, rtol=0, atol=0)
    assert hashes == first_hashes
    assert reopened.summary()['file_verification_process_delta']['content_hashes'] == 0
    reopened.close()


def test_reload_rejects_array_change_after_eviction(tmp_path):
    entries = _entries(tmp_path, 1)
    repository = storage.ModelRepository(entries, cache_limit_bytes=0)
    with repository.acquire([0], 'cpu'):
        pass
    path = Path(entries[0]['path'])/'normal'/'scaled_residuals.npy'
    values = np.load(path, allow_pickle=False)
    np.save(path, values+1)
    with pytest.raises(ValueError, match='identity changed'):
        with repository.acquire([0], 'cpu'):
            pass
    repository.close()


def test_cached_design_still_verifies_a_second_source_path(tmp_path):
    entries = _entries(tmp_path)
    repository = storage.ModelRepository(entries)
    with repository.acquire([0], 'cpu'):
        pass
    path = Path(entries[1]['path'])/'normal'/'x.npy'
    values = np.load(path, allow_pickle=False)
    np.save(path, values+1)
    with pytest.raises(ValueError, match='checksum differs'):
        with repository.acquire([1], 'cpu'):
            pass
    repository.close()


def test_cached_id_axis_still_verifies_a_second_source_path(tmp_path):
    entries = _entries(tmp_path)
    repository = storage.ModelRepository(entries)
    with repository.acquire([0], 'cpu'):
        pass
    path = Path(entries[1]['path'])/'normal'/'sample_ids.npy'
    values = np.load(path, allow_pickle=False)
    values[0] = 'a'
    np.save(path, values)
    with pytest.raises(ValueError, match='checksum differs'):
        with repository.acquire([1], 'cpu'):
            pass
    repository.close()


def test_reload_rejects_root_metadata_change(tmp_path):
    entries = _entries(tmp_path, 1)
    repository = storage.ModelRepository(entries, cache_limit_bytes=0)
    with repository.acquire([0], 'cpu'):
        pass
    path = Path(entries[0]['path'])/'model.private.json'
    value = json.loads(path.read_text())
    path.write_text(json.dumps(value, indent=4))
    with pytest.raises(ValueError, match='identity changed'):
        with repository.acquire([0], 'cpu'):
            pass
    repository.close()


def test_normal_and_spa_share_verified_sparse_structure_and_keep_trait_values(tmp_path):
    entries = _binary_entries(tmp_path)
    repository = storage.ModelRepository(entries, binary_normal_cache='stored_native')
    with repository.acquire([0, 1], 'cpu') as models:
        first, second = (model.precision for model in models)
        assert first.indices().data_ptr() == second.indices().data_ptr()
        assert first.values().data_ptr() != second.values().data_ptr()
        assert not torch.equal(first.values(), second.values())
        pattern = weakref.ref(first._immutable_shared_indices)
        with repository.acquire_spa(0, 'cpu') as spa:
            assert spa.precision.indices().data_ptr() == first.indices().data_ptr()
            torch.testing.assert_close(spa.precision.values(), first.values(), rtol=0, atol=0)
            assert repository.summary()['live_deduplicated_sparse_index_tensors'] == 1
        del spa
    del first, second, models
    repository.close()
    gc.collect()
    assert pattern() is None


def test_sparse_pattern_cache_still_checks_each_source_path(tmp_path):
    entries = _binary_entries(tmp_path)
    repository = storage.ModelRepository(entries, binary_normal_cache='stored_native')
    with repository.acquire([0], 'cpu'):
        pass
    path = Path(entries[1]['path'])/'normal'/'precision_indices.npy'
    values = np.load(path, allow_pickle=False)
    values[0, 0] = 1
    np.save(path, values)
    with pytest.raises(ValueError, match='checksum differs'):
        with repository.acquire([1], 'cpu'):
            pass
    repository.close()


@pytest.mark.parametrize('indices,shareable', [
    ([[0, 1, 2], [0, 1, 2]], True),
    ([[2, 0, 1], [2, 0, 1]], False),
    ([[0, 0, 1], [0, 0, 1]], False),
])
def test_sparse_pattern_sharing_preserves_coalesce_order_and_duplicates(tmp_path, indices, shareable):
    directory = tmp_path/'state'
    directory.mkdir()
    arrays = {}
    source = dict(precision_indices=np.asarray(indices, dtype=np.int64),
                  precision_values=np.asarray([1, 2, 3], dtype=np.float64))
    for name, value in source.items():
        file = directory/(name+'.npy')
        np.save(file, value)
        arrays[name] = dict(file=file.name, sha256=storage.digest_file(file),
                           shape=list(value.shape), dtype=value.dtype.str)
    meta = dict(arrays=arrays, precision_shape=[3, 3])
    memo = weakref.WeakValueDictionary()
    def tensor(name):
        return torch.tensor(np.load(directory/arrays[name]['file'], allow_pickle=False))
    expected = torch.sparse_coo_tensor(torch.tensor(source['precision_indices']),
        torch.tensor(source['precision_values']), (3, 3)).coalesce()
    first = storage._load_sparse_tensor('precision', meta, directory, 'cpu', memo, tensor)
    second = storage._load_sparse_tensor('precision', meta, directory, 'cpu', memo, tensor)
    torch.testing.assert_close(second.indices(), expected.indices(), rtol=0, atol=0)
    torch.testing.assert_close(second.values(), expected.values(), rtol=0, atol=0)
    assert bool(len(memo)) == shareable
    assert (first.indices().data_ptr() == second.indices().data_ptr()) == shareable


def test_bank_detects_cross_alias_sparse_index_write_at_acquire_exit(tmp_path):
    entries = _binary_entries(tmp_path)
    repository = storage.ModelRepository(entries, binary_normal_cache='stored_native')
    with pytest.raises(ValueError, match='shared sparse owner was mutated'):
        with repository.acquire([0, 1], 'cpu') as models:
            first, second = (model.precision for model in models)
            original_first_version = first.indices()._version
            # Independent COO wrappers do not share version counters, despite
            # aliasing the index storage. The pool's owner guard is necessary.
            second.indices()[0, 0] = 1
            assert first.indices()[0, 0] == 1
            assert first.indices()._version == original_first_version
    assert all(value == 0 for value in repository.pinned.values())
    repository.close()


def test_bank_detects_cross_alias_sparse_index_write_before_cached_spa_use(tmp_path):
    entries = _binary_entries(tmp_path)
    repository = storage.ModelRepository(entries, binary_normal_cache='stored_native')
    with repository.acquire([0, 1], 'cpu') as models:
        second = models[1].precision
        with repository.acquire_spa(0, 'cpu'):
            pass
    second.indices()[0, 0] = 1
    with pytest.raises(ValueError, match='shared sparse owner was mutated'):
        with repository.acquire_spa(0, 'cpu'):
            pytest.fail('mutated shared pattern reached a new SPA calculation')
    repository.close()


def test_nested_acquires_validate_pattern_pool_once_per_outer_boundary(tmp_path, monkeypatch):
    entries = _binary_entries(tmp_path)
    repository = storage.ModelRepository(entries, binary_normal_cache='stored_native')
    calls, original = [], repository._validate_shared_sparse_patterns
    def checked():
        calls.append(True)
        original()
    monkeypatch.setattr(repository, '_validate_shared_sparse_patterns', checked)
    with repository.acquire([0, 1], 'cpu'):
        with repository.acquire_spa(0, 'cpu'):
            with repository.acquire_spa(1, 'cpu'):
                pass
        assert len(calls) == 1
    assert len(calls) == 2
    assert repository._fitted_context_depth == 0
    repository.close()


def test_retired_shared_sparse_owner_does_not_pin_gpu_or_cpu_state(tmp_path):
    entries = _binary_entries(tmp_path)
    repository = storage.ModelRepository(entries, binary_normal_cache='stored_native')
    with repository.acquire([0, 1], 'cpu') as models:
        pattern = models[0].precision._immutable_shared_pattern
        first = weakref.ref(models[0].precision)
        second = weakref.ref(models[1].precision)
        assert len(pattern._owners) == 2
    del models
    assert repository._evict_one(device=torch.device('cpu'), protect=((1, 'cpu'),))
    gc.collect()
    assert first() is None
    assert len(pattern._owners) == 1
    assert second() is not None
    del pattern
    repository.close()
    gc.collect()
    assert second() is None


def test_model_array_map_is_bracketed_by_authenticated_file_identity(tmp_path, monkeypatch):
    entries = _entries(tmp_path, 1)
    repository = storage.ModelRepository(entries)
    original = storage.readonly_npy_mmap
    def replaced_after_map(path):
        value = original(path)
        if Path(path).name == 'scaled_residuals.npy':
            replacement = Path(path).with_suffix('.replacement.npy')
            np.save(replacement, np.asarray(value)+1)
            replacement.replace(path)
        return value
    monkeypatch.setattr(storage, 'readonly_npy_mmap', replaced_after_map)
    with pytest.raises(ValueError, match='identity changed'):
        with repository.acquire([0], 'cpu'):
            pytest.fail('a replaced source escaped its manifest identity guard')
    repository.close()
