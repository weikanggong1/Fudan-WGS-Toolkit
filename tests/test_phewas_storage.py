import csv
import gzip
import json
import numpy as np
import pytest
import torch
from fudan_wgs_toolkit.binary_null import fit_logistic_null
from fudan_wgs_toolkit.null_model import fit_gaussian_null
from fudan_wgs_toolkit.phewas_storage import (
    ModelRepository, StreamingCSVWriter, digest_file, load_model_store, save_model_store)


def test_binary_fitted_state_and_spa_survive_thin_store(tmp_path):
    n = 40
    x = np.column_stack((np.ones(n), np.linspace(-1, 1, n)))
    y = (np.arange(n) % 3 == 0).astype(float)
    full = fit_logistic_null(y, x, sample_ids=np.arange(n).astype(str), device='cpu', use_spa=True)
    full.precision = full.fitted_probability*(1-full.fitted_probability)
    full.precision_x = full.xw.T
    full.null_fit_source_sha256 = 'a'*64
    import copy
    normal = copy.deepcopy(full)
    normal.use_spa = False
    destination = tmp_path/'model'
    save_model_store(normal, destination, sample_rows=np.arange(n), fit_metadata={}, spa_model=full)
    loaded = load_model_store(destination)
    restored = load_model_store(destination, spa=True)
    g = torch.tensor(np.column_stack((np.arange(n) % 2, np.arange(n) % 5 == 0)), dtype=torch.float64)
    expected = normal.score_covariance(g)
    actual = loaded.score_covariance(g)
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert loaded.projection_left is None and loaded.phenotype is None
    assert restored.use_spa and not loaded.use_spa
    assert loaded.null_fit_source_sha256 == restored.null_fit_source_sha256 == 'a'*64
    torch.testing.assert_close(restored.projection_left, full.projection_left, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        save_model_store(normal, destination, sample_rows=np.arange(n), fit_metadata={})


def test_gaussian_store_and_nested_repository_preserve_projection(tmp_path):
    n = 20
    x = np.column_stack((np.ones(n), np.linspace(-1, 1, n)))
    model = fit_gaussian_null(np.sin(np.arange(n)), sample_ids=np.arange(n).astype(str),
                              covariates=x, device='cpu', matmul_mode='fp64')
    destination = tmp_path/'model'
    entry = save_model_store(model, destination, sample_rows=np.arange(n), fit_metadata={})
    repository = ModelRepository([entry])
    g = torch.tensor((np.arange(n) % 3)[:, None], dtype=torch.float64)
    with repository.acquire([0], 'cpu') as a:
        with repository.acquire([0], 'cpu') as b:
            assert a[0] is b[0]
            for actual, expected in zip(a[0].score_covariance(g), model.score_covariance(g)):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert repository.summary()['model_loads'] == 1
    assert repository.pinned[(0, 'cpu')] == 0


def test_interrupted_model_store_retries_in_same_process_without_overwriting_partial_or_complete(tmp_path, monkeypatch):
    """Inject interruption during array publication, then retry the same PID."""
    import os
    import fudan_wgs_toolkit.phewas_storage as storage
    n = 20
    model = fit_gaussian_null(np.sin(np.arange(n)), sample_ids=np.arange(n).astype(str),
        covariates=np.column_stack((np.ones(n), np.linspace(-1, 1, n))),
        device='cpu', matmul_mode='fp64')
    destination = tmp_path/'model'
    original_save, calls, pid = storage.np.save, 0, os.getpid()
    def interrupted_save(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('injected array-write interruption')
        return original_save(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(storage.np, 'save', interrupted_save)
        with pytest.raises(OSError, match='injected array-write interruption'):
            save_model_store(model, destination, sample_rows=np.arange(n), fit_metadata={})
    assert not destination.exists()
    partials = list(tmp_path.glob(f'model.{pid}.*.partial'))
    assert len(partials) == 1
    interrupted_bytes = {p.relative_to(partials[0]): p.read_bytes()
                         for p in partials[0].rglob('*') if p.is_file()}
    assert interrupted_bytes
    entry = save_model_store(model, destination, sample_rows=np.arange(n), fit_metadata={})
    assert os.getpid() == pid
    assert {p.relative_to(partials[0]): p.read_bytes()
            for p in partials[0].rglob('*') if p.is_file()} == interrupted_bytes
    restored = load_model_store(destination)
    g = torch.tensor((np.arange(n) % 3)[:, None], dtype=torch.float64)
    for actual, expected in zip(restored.score_covariance(g), model.score_covariance(g)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    before = {p.relative_to(destination): digest_file(p)
              for p in destination.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError, match='immutable fitted model store'):
        save_model_store(model, destination, sample_rows=np.arange(n), fit_metadata={'retry': True})
    assert {p.relative_to(destination): digest_file(p)
            for p in destination.rglob('*') if p.is_file()} == before
    assert entry['sha256'] == digest_file(destination/'model.private.json')


def test_model_axes_are_immutable_and_descriptor_groups_alias_authenticated_rows(tmp_path):
    from fudan_wgs_toolkit.phewas_runtime.runtime import _descriptor_groups
    n = 20
    model = fit_gaussian_null(np.sin(np.arange(n)), sample_ids=np.arange(n).astype(str),
        covariates=np.ones((n, 1)), device='cpu', matmul_mode='fp64')
    entry = save_model_store(model, tmp_path/'model', sample_rows=np.arange(n), fit_metadata={})
    repository = ModelRepository([entry, entry.copy()])
    rows = repository.descriptors()[0]['ordered_rows']
    with pytest.raises(ValueError):
        rows.flags.writeable = True
    descriptors, groups = _descriptor_groups(repository)
    assert len(groups) == 1 and groups[0]['rows'] is rows
    assert all(descriptor['ordered_rows'] is rows for descriptor in descriptors)
    assert repository.summary()['sample_axis_ram_bytes'] == 0
    assert repository.summary()['sample_axis_mapped_logical_bytes'] == rows.nbytes
    assert repository.summary()['sample_axis_open_mmaps'] == 1


def test_design_dedup_survives_one_bank_eviction_and_releases_last_owner(tmp_path):
    import gc
    import weakref
    n = 24
    x = np.column_stack((np.ones(n), np.linspace(-1, 1, n)))
    entries = []
    for i in range(2):
        model = fit_gaussian_null(np.sin(np.arange(n)+i), sample_ids=np.arange(n).astype(str),
            covariates=x, device='cpu', matmul_mode='fp64')
        entries.append(save_model_store(model, tmp_path/f'model{i}',
            sample_rows=np.arange(n), fit_metadata={}))
    repository = ModelRepository(entries)
    with repository.acquire([0, 1], 'cpu') as loaded:
        assert loaded[0].x is loaded[1].x
        design = weakref.ref(loaded[1].x)
    del loaded
    assert repository._evict_one(device=torch.device('cpu'), protect=((1, 'cpu'),))
    with repository.acquire([0], 'cpu') as loaded:
        assert loaded[0].x is repository.models[(1, 'cpu')].x is design()
    del loaded
    while repository._evict_one():
        pass
    gc.collect()
    assert design() is None
    assert repository.summary()['live_deduplicated_design_tensors'] == 0


def test_bank_close_releases_owned_arrays_and_refuses_active_contexts(tmp_path):
    import gc
    import weakref
    n = 20
    model = fit_gaussian_null(np.sin(np.arange(n)), sample_ids=np.arange(n).astype(str),
        covariates=np.ones((n, 1)), device='cpu', matmul_mode='fp64')
    destination = tmp_path/'model'
    entry = save_model_store(model, destination, sample_rows=np.arange(n), fit_metadata={})
    repository = ModelRepository([entry])
    axis = weakref.ref(repository.descriptors()[0]['ordered_rows'])
    with repository.acquire([0], 'cpu') as loaded:
        design = weakref.ref(loaded[0].x)
        with pytest.raises(RuntimeError, match='active model/workspace'):
            repository.close()
        storage = repository.resident_storage_summary()
        assert storage['resident_storage_bytes_by_device']['cpu'] >= loaded[0].x.numel()*8
        assert storage['unique_cuda_storage_bytes'] == 0
    del loaded
    with repository.reserve_workspace('cpu', 0):
        with pytest.raises(RuntimeError, match='active model/workspace'):
            repository.close()
    repository.close()
    repository.close()
    gc.collect()
    assert design() is None and axis() is None
    assert repository.summary()['closed'] is True
    assert repository.summary()['resident_storage_bytes_by_device'] == {}
    assert (destination/'normal'/'x.npy').exists()
    with pytest.raises(RuntimeError, match='closed'):
        with repository.acquire([0], 'cpu'):
            pass



def test_csv_shards_cover_all_traits_fields_and_significance_levels(tmp_path):
    writer = StreamingCSVWriter(tmp_path, max_rows_per_shard=3, minimum_free_bytes=0)
    for trait in range(5):
        writer.write('individual', trait, '1', [dict(CHR=1, POS=trait+1, pvalue=.6),
                      dict(CHR=1, POS=trait+10, pvalue=1e-10)], job_index=trait)
    report = writer.finish()
    rows = []
    for receipt in report['receipts']:
        file = tmp_path/receipt['csv_file']
        assert digest_file(file) == receipt['sha256']
        with gzip.open(file, 'rt', newline='') as f:
            rows.extend(csv.DictReader(f))
        assert receipt['rows'] <= 3
    assert len(rows) == 10 and {int(r['trait_index']) for r in rows} == set(range(5))
    assert sum(float(r['pvalue']) < 1e-5 for r in rows) == 5
    assert sum(float(r['pvalue']) > .05 for r in rows) == 5
    assert not list(tmp_path.rglob('*.partial'))
    assert len(report['receipts']) == 4


def test_binary_repository_reuses_verified_source_for_normal_and_spa(tmp_path, monkeypatch):
    import fudan_wgs_toolkit.binary_null as binary_null
    # CPU formula/storage oracle; actual native TF32 is checked on CUDA.
    monkeypatch.setattr(binary_null, 'matmul', lambda left,right,mode: left@right)
    from fudan_wgs_toolkit.phewas_models import prepare_phewas_model
    n = 45
    x = np.column_stack((np.ones(n), np.linspace(-1, 1, n)))
    fitted = prepare_phewas_model((np.arange(n)%3 == 0).astype(float),
        np.arange(n).astype(str), x, family='binomial', device='cpu')
    entry = save_model_store(fitted.model, tmp_path/'model', sample_rows=np.arange(n),
                             fit_metadata={}, spa_model=fitted.spa_model)
    repository = ModelRepository([entry])
    g = torch.tensor((np.arange(n)%2)[:,None], dtype=torch.float32)
    for _ in range(2):
        with repository.acquire([0], 'cpu') as models:
            for actual, expected in zip(models[0].score_covariance(g), fitted.model.score_covariance(g)):
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
            with repository.acquire_spa(0, 'cpu') as spa:
                assert spa.sample_ids is models[0].sample_ids
                torch.testing.assert_close(spa.projection_left, fitted.spa_model.projection_left,
                                           rtol=1e-12, atol=1e-12)
    assert repository.summary()['spa_source_loads'] == 1
    assert repository.summary()['spa_source_hits'] == 2
    assert repository.summary()['binary_normal_derivations'] == 1
    assert repository.summary()['binary_normal_cache_hits'] == 1
    assert repository.spa_pinned[(0, 'cpu')] == 0


def test_optional_stored_native_binary_bank_preserves_fitted_normal_and_original_spa(tmp_path, monkeypatch):
    import fudan_wgs_toolkit.binary_null as binary_null
    monkeypatch.setattr(binary_null, 'matmul', lambda left, right, mode: left@right)
    from fudan_wgs_toolkit.phewas_models import prepare_phewas_model
    n = 45
    x = np.column_stack((np.ones(n), np.linspace(-1, 1, n)))
    fitted = prepare_phewas_model((np.arange(n)%3 == 0).astype(float),
        np.arange(n).astype(str), x, family='binomial', device='cpu')
    entry = save_model_store(fitted.model, tmp_path/'model', sample_rows=np.arange(n),
        fit_metadata={}, spa_model=fitted.spa_model)
    repository = ModelRepository([entry], binary_normal_cache='stored_native')
    g = torch.tensor(np.column_stack((np.arange(n)%2, np.arange(n)%5 == 0)), dtype=torch.float32)
    with repository.acquire([0], 'cpu') as normal:
        first = normal[0]
        assert not first.use_spa and repository.summary()['spa_source_loads'] == 0
        for name in ('x', 'scaled_residuals', 'fitted_probability', 'precision',
                     'precision_x', 'fixed_effect_covariance', 'coefficients'):
            torch.testing.assert_close(getattr(first, name), getattr(fitted.model, name), rtol=0, atol=0)
        for actual, expected in zip(first.score_covariance(g), fitted.model.score_covariance(g)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with repository.acquire_spa(0, 'cpu') as spa:
            assert spa.use_spa and spa.x.dtype == torch.float64
            assert first.null_fit_source_sha256 == spa.null_fit_source_sha256
            np.testing.assert_array_equal(first.sample_ids, spa.sample_ids)
            torch.testing.assert_close(spa.x, fitted.spa_model.x, rtol=0, atol=0)
            torch.testing.assert_close(spa.projection_left, fitted.spa_model.projection_left,
                rtol=1e-12, atol=1e-12)
            for actual, expected in zip(spa.score_covariance(g.double()), fitted.spa_model.score_covariance(g.double())):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with repository.acquire([0], 'cpu') as normal:
        assert normal[0] is first
    summary = repository.summary()
    assert summary['model_loads'] == 1 and summary['spa_source_loads'] == 1
    assert summary['binary_normal_from_same_fp64_source'] is False
    assert summary['binary_spa_state_provenance'] == 'authenticated_original_fp64_source'
    with pytest.raises(ValueError, match='binary_normal_cache'):
        ModelRepository([], binary_normal_cache='rounded_spa')


def test_interrupted_writer_keeps_partial_and_cannot_claim_complete(tmp_path):
    writer = StreamingCSVWriter(tmp_path, minimum_free_bytes=0)
    writer.write('individual', 0, '1', [dict(CHR=1,POS=1,pvalue=.2)])
    writer.abort()
    assert list(tmp_path.rglob('*.partial'))
    assert not (tmp_path/'worker_0.complete.private.json').exists()
    report = json.loads((tmp_path/'worker_0.interrupted.private.json').read_text())
    assert report['complete'] is False and report['closed_partial_streams'][0]['rows'] == 1


def test_1342_sample_descriptors_do_not_require_1342_file_descriptors(tmp_path):
    """Reproduce the production scale failure under a low per-process limit.

    This verifies descriptor ownership, not scientific accuracy or throughput.
    """
    import os
    from pathlib import Path
    import subprocess
    import sys
    directory = tmp_path/'descriptor'
    directory.mkdir()
    np.save(directory/'sample_rows.npy', np.array([0, 2, 7], dtype=np.int64))
    metadata = directory/'model.private.json'
    metadata.write_text(json.dumps(dict(n=3, sample_rows_sha256=digest_file(directory/'sample_rows.npy'))))
    code = '''
import json,resource,sys
from pathlib import Path
from fudan_wgs_toolkit.phewas_storage import ModelRepository,digest_file
directory=Path(sys.argv[1])
before=len(list(Path('/proc/self/fd').iterdir()))
soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE,(128,hard))
entry=dict(path=str(directory),sha256=digest_file(directory/'model.private.json'),family='gaussian')
bank=ModelRepository([entry.copy() for _ in range(1342)])
axes=bank.descriptors()
assert len(axes)==1342 and all(axis['ordered_rows'].tolist()==[0,2,7] for axis in axes)
assert all(axis['ordered_rows'] is axes[0]['ordered_rows'] for axis in axes)
assert not axes[0]['ordered_rows'].flags.writeable
assert bank.summary()['unique_sample_axes']==1 and bank.summary()['sample_axis_open_mmaps']==1
assert len(list(Path('/proc/self/fd').iterdir()))<=before+2
'''
    result = subprocess.run([sys.executable, '-c', code, str(directory)],
        capture_output=True, text=True, env=dict(os.environ), timeout=60)
    assert result.returncode == 0, result.stderr
