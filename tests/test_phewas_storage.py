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
    assert repository.summary()['spa_source_hits'] == 3
    assert repository.spa_pinned[(0, 'cpu')] == 0


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
assert bank.summary()['unique_sample_axes']==1 and bank.summary()['sample_axis_open_mmaps']==0
assert len(list(Path('/proc/self/fd').iterdir()))<=before+2
'''
    result = subprocess.run([sys.executable, '-c', code, str(directory)],
        capture_output=True, text=True, env=dict(os.environ), timeout=60)
    assert result.returncode == 0, result.stderr
