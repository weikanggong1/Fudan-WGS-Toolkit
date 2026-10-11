"""Metadata/accounting contracts; tiny fitted fixtures are not benchmarks."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.binary_null import fit_logistic_null, ImmutableFittedBinarySource
from fudan_wgs_toolkit.phewas_model_footprint import (
    estimate_fitted_model_storage, estimate_fitted_subset_storage)
from fudan_wgs_toolkit.phewas_storage import ModelRepository, digest_file, save_model_store
from test_phewas_batched_runtime import model_for
from test_phewas_bank_policy import axis_entries


def fitted_entries(path):
    n = 32
    entries = [save_model_store(model_for(np.arange(n), .7+i), path/str(i),
        sample_rows=np.arange(n), fit_metadata={}) for i in range(2)]
    x = np.column_stack((np.ones(n), np.linspace(-1, 1, n)))
    full = fit_logistic_null((np.arange(n)%3==0).astype(float), x,
        sample_ids=np.asarray([f'sample{i}' for i in range(n)]), device='cpu')
    full.precision = full.fitted_probability*(1-full.fitted_probability)
    full.precision_x = full.xw.T
    full.null_fit_source_sha256 = 'a'*64
    source = ImmutableFittedBinarySource(sample_ids=full.sample_ids, covariates=full.x,
        residual=full.scaled_residuals, fitted_probability=full.fitted_probability,
        fixed_effect_covariance=full.fixed_effect_covariance, precision=full.precision,
        null_fit_source_sha256='a'*64, verified_fit_source_sha256='a'*64,
        coefficients=full.coefficients, device='cpu')
    entries.append(save_model_store(source.normal(), path/'2',
        sample_rows=np.arange(n), fit_metadata={}, spa_model=full))
    return entries


def test_metadata_loaded_payloads_match_actual_normal_and_minimal_spa_owners(tmp_path):
    entries = fitted_entries(tmp_path)
    estimate = estimate_fitted_model_storage(entries)
    bank = ModelRepository(entries, binary_normal_cache='stored_native', workspace_bytes=0)
    with bank.acquire([0, 1, 2], 'cpu'):
        with bank._acquire_binary_source(2, 'cpu'):
            actual = bank.resident_storage_summary()['resident_storage_bytes_by_device']['cpu']
            assert estimate['persistent_loaded_bytes'] == actual
    assert estimate['shared_x_bytes_saved'] == 32*4  # Two Gaussian models share X only.
    binary = estimate['per_trait'][2]
    spa = json.loads((Path(entries[2]['path'])/'spa'/'state.private.json').read_text())['arrays']
    ignored = {'xw', 'projection_left', 'phenotype', 'working_phenotype'}
    assert ignored.intersection(spa)
    assert not ignored.intersection(payload['identity'][-1]
        for payload in binary['storage_payloads'] if payload['identity'][0]=='spa')
    assert binary['active_spa_projection_bytes_upper_bound'] == 16*32*2
    assert binary['lazy_single_projection_bytes_upper_bound'] == 4*33
    assert estimate['array_contents_read'] is False and estimate['array_checksums_verified'] is False
    assert estimate['metadata_checksums_verified'] is True
    bank.close()


def test_metadata_estimator_does_not_read_npy_and_preserves_global_ids_ordered_axis(tmp_path, monkeypatch):
    entries = fitted_entries(tmp_path)
    entries[0]['trait_index'], entries[1]['trait_index'], entries[2]['trait_index'] = 7, 2, 11
    original = Path.read_bytes
    def metadata_only(path):
        assert path.suffix == '.json'
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', metadata_only)
    monkeypatch.setattr(np, 'load', lambda *a, **kw: pytest.fail('metadata estimation loaded NPY'))
    estimate = estimate_fitted_model_storage(entries)
    assert [r['trait_index'] for r in estimate['per_trait']] == [7, 2, 11]
    assert all(r['ordered_sample_rows_sha256'] == json.loads(
        (Path(entries[i]['path'])/'model.private.json').read_text())['sample_rows_sha256']
        for i, r in enumerate(estimate['per_trait']))
    subset = estimate_fitted_subset_storage(estimate, [2, 7])
    assert subset['spa_source_loaded_bytes'] == 0 and subset['shared_x_bytes_saved'] == 32*4
    assert subset['normal_loaded_bytes'] == sum(r['normal_loaded_bytes'] for r in estimate['per_trait'][:2])-32*4
    assert estimate_fitted_subset_storage(estimate, [])['persistent_loaded_bytes'] == 0
    with pytest.raises(ValueError, match='absent'):
        estimate_fitted_subset_storage(estimate, [1])


def test_metadata_estimator_rejects_changed_small_manifest_without_array_rehash(tmp_path):
    entries = fitted_entries(tmp_path)
    state = Path(entries[0]['path'])/'normal'/'state.private.json'
    state.write_text(state.read_text()+'\n')
    with pytest.raises(ValueError, match='metadata checksum'):
        estimate_fitted_model_storage(entries)


def test_derived_binary_policy_does_not_count_temporary_normal_as_persistent(tmp_path):
    entries = fitted_entries(tmp_path)
    native = estimate_fitted_model_storage(entries)
    derived = estimate_fitted_model_storage(entries, binary_normal_cache='derived_fp64', include_spa=False)
    record = derived['per_trait'][2]
    assert record['normal_loaded_bytes'] == 0 and record['spa_source_loaded_bytes'] > 0
    assert record['normal_derived_active_bytes_upper_bound'] == native['per_trait'][2]['normal_loaded_bytes']
    assert derived['persistent_loaded_bytes'] < native['persistent_loaded_bytes']


def test_subset_model_bank_keeps_global_ids_and_only_maps_assigned_axes(tmp_path, monkeypatch):
    entries = fitted_entries(tmp_path)
    bank = ModelRepository(entries, trait_indices=[2, 0], binary_normal_cache='stored_native')
    assert [d['trait_index'] for d in bank.descriptors()] == [2, 0]
    with bank.acquire([2, 0], 'cpu') as models:
        assert [m.family for m in models] == ['binomial', 'gaussian']
        assert (2, 'cpu') in bank.models and (0, 'cpu') in bank.models
    with bank.acquire_spa(2, 'cpu') as source:
        assert source.null_fit_source_sha256 == 'a'*64
    with pytest.raises(ValueError, match='not assigned'):
        with bank.acquire([1], 'cpu'):
            pass
    assert bank.summary()['global_inventory_model_count'] == 3
    assert bank.summary()['assigned_model_count'] == 2
    # The unassigned path can be unavailable: no descriptor/model read is made.
    unavailable = dict(entries[1], path=str(tmp_path/'absent'))
    second = ModelRepository([entries[0], unavailable, entries[2]], trait_indices=[2])
    assert [d['trait_index'] for d in second.descriptors()] == [2]
    assert second.summary()['unique_sample_axes'] == 1
    second.close()
    bank.close()


@pytest.mark.parametrize('indices', [[True], [-1], [3], [1, 1], [1.0]])
def test_subset_model_bank_rejects_invalid_global_assignment(tmp_path, indices):
    entries = axis_entries(tmp_path, [[0, 1], [1, 0], [2, 0]])
    with pytest.raises(ValueError, match='distinct global'):
        ModelRepository(entries, trait_indices=indices)
