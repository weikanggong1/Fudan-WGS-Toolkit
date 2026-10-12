"""Prepared schema-2 model/checkpoint contracts, with association injected.

Real tiny PLINK preparation and actual CPU null models verify identity and
immutable state proofs. Injected probabilities are not a CUDA benchmark.
"""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from fudan_wgs_toolkit import prepared_association as prepared
from fudan_wgs_toolkit.identity import sample_keys, pairs_from_keys
from fudan_wgs_toolkit.phewas_models import prepare_phewas_model
from fudan_wgs_toolkit.run import run_WGS_all_binanry


def fixture(tmp_path):
    from test_plink_prepare import source_fixture
    from fudan_wgs_toolkit.prepare import prepare_WGS_data
    source, annotations, pairs, _ = source_fixture(tmp_path)
    root = tmp_path/'prepared'
    prepare_WGS_data(source, root, annotation_directory=annotations, cpu_threads=1, chunk_size=2)
    phenotype, covariate = tmp_path/'phenotypes.csv', tmp_path/'covariates.csv'
    with phenotype.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['FID', 'IID', 'binary_a', 'binary'])
        for i, pair in enumerate(pairs):
            writer.writerow([*pair, ['NA', 0, 1, 0, 1, 0][i], [0, 'NA', 1, 0, 1, 0][i]])
    with covariate.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['FID', 'IID', 'age'])
        for i, pair in reversed(list(enumerate(pairs))):
            writer.writerow([*pair, ['NA', 1, 2, 3, 4, 5][i]])
    return root, phenotype, covariate, pairs


def injected_execution(monkeypatch):
    from fudan_wgs_toolkit import run
    from fudan_wgs_toolkit.phewas_runtime import runtime
    calls = {'fits': [], 'computed': [], 'runner': []}
    monkeypatch.setattr(run.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(run.torch.cuda, 'device_count', lambda: 2)
    from fudan_wgs_toolkit import phewas_partition
    monkeypatch.setattr(phewas_partition, "gpu_model_cache_capacity",
                        lambda gpu_ids, **policy: 8*2**30)
    def fit(*args, **kwargs):
        assert prepared.torch.get_num_threads() == 1
        calls['fits'].append(kwargs.copy())
        kwargs['device'] = 'cpu'
        return prepare_phewas_model(*args, **kwargs)
    monkeypatch.setattr(prepared, 'prepare_phewas_model', fit)
    def associate(configuration, *, model_repository, reader_factory, writer, **kwargs):
        assert prepared.torch.get_num_threads() == kwargs['cpu_threads']
        calls['runner'].append(dict(configuration=configuration, resources=kwargs))
        descriptors = model_repository.descriptors()
        expected_rows = ([1, 2, 3, 4, 5], [0, 2, 3, 4, 5])
        for index, descriptor in enumerate(descriptors):
            assert descriptor['ordered_rows'].tolist() == expected_rows[index]
            with model_repository.acquire([index], 'cpu') as models:
                assert len(models[0].sample_ids) == 5
        assert descriptors[1]['spa_available'] is True
        rows = dict.fromkeys(('individual', 'coding', 'noncoding', 'ncrna'), 0)
        for chromosome_index, reference in enumerate(configuration['chromosomes']):
            with reader_factory(reference) as resources:
                np.testing.assert_array_equal(resources['metadata'].sample_ids(), sample_keys(
                    pairs_from_keys(resources['metadata'].sample_ids())))
                assert resources['portable_axis'] is True
            for local_job_index, job in enumerate(reference['jobs']):
                kind, index = job['kind'], job['checkpoint_job_index']
                key = dict(block_index=0, block_identity='contract-block', ordinal_before=0)
                reused = writer.checkpoint_tile(kind, reference['name'], index, [0, 1], **key)
                if reused is None:
                    calls['computed'].append(index)
                    record = writer.write_tile(kind, reference['name'], [(i, [dict(
                        pvalue=.25, pvalue_log10=0.6020599913279624)]) for i in (0, 1)],
                        job_index=index, ordinal_after=1, **key)
                else:
                    record = reused
                rows[kind] += record['rows']
                kwargs['emit'](dict(event='finished', chromosome_index=chromosome_index,
                                    job_index=local_job_index, kind=kind))
        return dict(association_rows=rows, job_groups=sum(len(e['jobs']) for e in configuration['chromosomes']),
                    output=writer.finish())
    monkeypatch.setattr(runtime, 'run_batched_configuration', associate)
    return calls


def call(root, phenotype, covariate, output, **kwargs):
    kwargs.setdefault('impute_missing', False)
    return run_WGS_all_binanry(phenotype, covariate, root, output_directory=output,
        analyses=('individual',), grm='none', cpu_threads=1, single_region_size=10,
        phenotype_covariates={'binary_a': ['age'], 'binary': []}, **kwargs)


def test_modern_shared_models_identity_and_verified_resume(tmp_path, monkeypatch):
    root, phenotype, covariate, pairs = fixture(tmp_path)
    calls = injected_execution(monkeypatch)
    output = tmp_path/'results'
    first = call(root, phenotype, covariate, output)
    assert first['completed'] and first['association_rows']['individual'] == 6
    assert [entry['family'] for entry in calls['fits']] == ['binomial', 'binomial']
    assert all(entry['kinship'] is None for entry in calls['fits'])
    assert calls['computed'] == [0, 1, 2]
    assert calls['runner'][0]['resources']['memory_limit_gib'] is None
    assert calls['runner'][0]['resources']['memory_reserve_bytes'] == 2**30
    second = call(root, phenotype, covariate, output, resume=True)
    assert second['completed'] and second['association_rows'] == first['association_rows']
    assert len(calls['fits']) == 2 and calls['computed'] == [0, 1, 2]
    assert second['workers'][0]['output']['reused_blocks'] == 3
    index = list(csv.DictReader((output/'trait_index.private.csv').open()))
    assert [entry['family'] for entry in index] == ['binomial', 'binomial']
    with pytest.raises(FileExistsError, match='fresh output'):
        call(root, phenotype, covariate, output)


def test_failed_resume_cannot_leave_old_success_as_current_attempt(tmp_path, monkeypatch):
    root, phenotype, covariate, _ = fixture(tmp_path)
    injected_execution(monkeypatch)
    output = tmp_path/'results'
    first = call(root, phenotype, covariate, output)
    with pytest.raises(ValueError, match='source/input/schedule'):
        call(root, phenotype, covariate, output, resume=True, impute_missing=True)
    assert not (output/'report.private.json').exists()
    archived = list(output.glob('report.previous.*.private.json'))
    assert len(archived)==1 and json.loads(archived[0].read_text())==first
    status = json.loads((output/'status.private.json').read_text())
    assert status['status']=='failed' and status['attempt_id']!=first['attempt_id']


def test_resume_into_existing_empty_directory_computes_every_block(tmp_path, monkeypatch):
    root, phenotype, covariate, _ = fixture(tmp_path)
    calls = injected_execution(monkeypatch)
    output = tmp_path/'empty_results';output.mkdir()
    report = call(root, phenotype, covariate, output, resume=True)
    assert report['completed'] and len(calls['fits'])==2 and calls['computed']==[0,1,2]


def test_completion_publication_failure_invalidates_current_success(tmp_path, monkeypatch):
    from fudan_wgs_toolkit import run
    root, phenotype, covariate, _ = fixture(tmp_path)
    injected_execution(monkeypatch)
    output = tmp_path/'results'
    write = run._json_write
    def fail_complete(path, value):
        if Path(path).name=='status.private.json' and value.get('status')=='complete':
            raise OSError('injected completion publication failure')
        return write(path,value)
    monkeypatch.setattr(run,'_json_write',fail_complete)
    with pytest.raises(OSError,match='publication'):
        call(root,phenotype,covariate,output)
    assert not (output/'report.private.json').exists()
    assert len(list(output.glob('report.failed.*.private.json')))==1
    assert json.loads((output/'status.private.json').read_text())['status']=='failed'


def test_intercept_only_trait_does_not_require_unselected_covariate_rows(tmp_path):
    from fudan_wgs_toolkit import run
    pairs = np.asarray([['0','001'],['f','001'],['0','003']],dtype=str)
    phenotype = run.NumericCSV(pairs,('trait',),np.asarray([[2.],[4.],[1.]]))
    covariate = run.NumericCSV(pairs[:1],('unused',),np.asarray([[np.nan]]))
    ids,y,x,names,counts = run.aligned_phenotype(phenotype,covariate,pairs,'trait',
        covariate_names=[],reduce_redundant=True)
    np.testing.assert_array_equal(ids,sample_keys(pairs))
    np.testing.assert_array_equal(y,[2.,4.,1.])
    np.testing.assert_array_equal(x,np.ones((3,1)))
    assert counts['analysis_samples']==3 and names==('Intercept',)


@pytest.mark.parametrize("option", ["continuous_transform", "transform", "phenotype_families",
                                  "null_fit_mode", "long_mask_method", "long_mask_threshold",
                                  "long_mask_rank", "gpu_ids"])
def test_removed_generic_options_fail_before_models_or_results(tmp_path, monkeypatch, option):
    root, phenotype, covariate, _ = fixture(tmp_path)
    calls = injected_execution(monkeypatch)
    output = tmp_path / "results"
    with pytest.raises(TypeError, match="unexpected keyword"):
        call(root, phenotype, covariate, output, **{option: "removed"})
    assert not output.exists() and calls["fits"] == []


def test_binary_public_fits_keep_no_continuous_transform(tmp_path, monkeypatch):
    root, phenotype, covariate, _ = fixture(tmp_path)
    calls = injected_execution(monkeypatch)
    output = tmp_path / "results"
    assert call(root, phenotype, covariate, output)["completed"]
    assert [entry["family"] for entry in calls["fits"]] == ["binomial", "binomial"]
    assert all(entry["continuous_transform"] == "none" for entry in calls["fits"])
    assert json.loads((output / "plan.private.json").read_text())["transform"] == "none"


def test_portable_grm_exact_family_identity_and_reordered_submatrix(tmp_path):
    # Two families share an IID, and another pair differs only by leading
    # zeros. All four are distinct IDs; an IID-only or numeric match is wrong.
    pairs = np.asarray([['0', '001'], ['family_a', 'same'],
                        ['family_b', 'same'], ['0', '1']], dtype=str)
    keys = sample_keys(pairs)
    source_order = np.asarray([2, 0, 3, 1])
    diagonal = np.asarray([1.2, 1.3, 1.4, 1.5])
    row, col = np.asarray([0, 0, 1, 2]), np.asarray([1, 3, 2, 3])
    value = np.asarray([.05, .02, -.01, .03])
    dense = np.diag(diagonal)
    dense[row, col] = value
    dense[col, row] = value
    path = tmp_path/'kinship.npz'
    np.savez(path, sample_ids=keys[source_order], diagonal=diagonal,
             edge_rows=row, edge_cols=col, edge_values=value)
    loaded = prepared.SparseKinshipData.load_npz(path)
    np.testing.assert_array_equal(loaded.sample_ids, keys[source_order])
    requested = np.asarray([1, 3, 0])
    selected = loaded.subset(keys[requested])
    source_positions = np.asarray([np.flatnonzero(source_order == index)[0] for index in requested])
    expected = dense[np.ix_(source_positions, source_positions)]
    actual = np.diag(selected.diagonal)
    actual[selected.edge_rows, selected.edge_cols] = selected.edge_values
    actual[selected.edge_cols, selected.edge_rows] = selected.edge_values
    np.testing.assert_array_equal(selected.sample_ids, keys[requested])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(loaded.diagonal, diagonal)
    assert not loaded.diagonal.flags.writeable
    assert not selected.sample_ids.flags.writeable
    # Even a unique IID vector does not identify a FID/IID cohort.
    with pytest.raises(ValueError, match='every analysis sample'):
        loaded.subset(np.asarray(['001', '1']))
    with pytest.raises(ValueError, match='unique vector'):
        loaded.subset(np.asarray([keys[0], keys[0]]))


def test_corrupt_committed_trait_csv_blocks_resume_without_overwriting_evidence(tmp_path, monkeypatch):
    from fudan_wgs_toolkit.phewas_trait_csv import CommittedCSVCorruption
    root, phenotype, covariate, _ = fixture(tmp_path)
    calls = injected_execution(monkeypatch)
    output = tmp_path/'results'
    report = call(root, phenotype, covariate, output)
    filename = output/'csv'/report['workers'][0]['output']['receipts'][0]['csv_file']
    filename.write_bytes(b'corrupt gzip')
    with pytest.raises(CommittedCSVCorruption, match='truncated'):
        call(root, phenotype, covariate, output, resume=True)
    assert filename.read_bytes() == b'corrupt gzip'
    assert len(calls['fits']) == 2 and calls['computed'] == [0, 1, 2]
    assert not (output/'report.private.json').exists()
    assert json.loads((output/'status.private.json').read_text())['status'] == 'failed'
    assert json.loads(next(output.glob('report.previous.*.private.json')).read_text()) == report


def test_corrupt_model_array_cannot_resume_or_refit_silently(tmp_path, monkeypatch):
    root, phenotype, covariate, _ = fixture(tmp_path)
    calls = injected_execution(monkeypatch)
    output = tmp_path/'results'
    call(root, phenotype, covariate, output)
    path = output/'models/trait_0000/normal/scaled_residuals.npy'
    data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    with pytest.raises(ValueError, match='fitted array checksum'):
        call(root, phenotype, covariate, output, resume=True)
    assert len(calls['fits']) == 2


@pytest.mark.parametrize('damage', [None, 'array', 'binding'])
def test_atomic_model_missing_receipt_recovers_only_verified_state(tmp_path, monkeypatch, damage):
    root, phenotype, covariate, _ = fixture(tmp_path)
    calls = injected_execution(monkeypatch)
    output = tmp_path/'results'
    first = call(root, phenotype, covariate, output)
    receipt = output/'models/trait_0001.complete.private.json'
    original = json.loads(receipt.read_text())
    receipt.unlink()
    if damage == 'array':
        path = output/'models/trait_0001/spa/fitted_probability.npy'
        data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    elif damage == 'binding':
        path = output/'models/trait_0001/model.private.json'
        meta = json.loads(path.read_text())
        meta['fit']['prepared_association_binding']['sample_ids_sha256'] = '0'*64
        path.write_text(json.dumps(meta))
    if damage is None:
        resumed = call(root, phenotype, covariate, output, resume=True)
        assert resumed['completed'] and resumed['association_rows'] == first['association_rows']
        assert json.loads(receipt.read_text()) == original
        assert resumed['workers'][0]['output']['reused_blocks'] == 3
    else:
        with pytest.raises(ValueError, match='checksum|binding'):
            call(root, phenotype, covariate, output, resume=True)
        assert not receipt.exists()
    assert len(calls['fits']) == 2


def test_model_and_schedule_binding_rejects_changed_covariates(tmp_path, monkeypatch):
    root, phenotype, covariate, _ = fixture(tmp_path)
    calls = injected_execution(monkeypatch)
    output = tmp_path/'results'
    call(root, phenotype, covariate, output)
    covariate.write_text(covariate.read_text().replace(',5\n', ',6\n'))
    with pytest.raises(ValueError, match='source/input/schedule'):
        call(root, phenotype, covariate, output, resume=True)
    assert len(calls['fits']) == 2


def test_missing_grm_and_incomplete_id_coverage_reject_before_dispatch(tmp_path, monkeypatch):
    root, phenotype, covariate, pairs = fixture(tmp_path)
    injected_execution(monkeypatch)
    output = tmp_path/'results'
    with pytest.raises(ValueError, match='sparse GRM requires'):
        run_WGS_all_binanry(phenotype, covariate, root, output_directory=output, analyses=('individual',))
    assert not output.exists()
    path = tmp_path/'grm.npz'
    np.savez(path, sample_ids=sample_keys(pairs[:2]), diagonal=np.ones(2),
        edge_rows=np.asarray([], dtype=np.int64), edge_cols=np.asarray([], dtype=np.int64), edge_values=np.asarray([]))
    with pytest.raises(ValueError, match='every analysis sample'):
        run_WGS_all_binanry(phenotype, covariate, root, output_directory=output, analyses=('individual',),
            grm_file=path, phenotype_covariates={'binary': []}, cpu_threads=1)


def test_default_covariates_reduce_constant_and_collinear_columns_per_trait(tmp_path, monkeypatch):
    root, phenotype, covariate, pairs = fixture(tmp_path)
    calls = injected_execution(monkeypatch)
    with covariate.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['FID', 'IID', 'sex', 'age', 'age_twice'])
        writer.writerows([*pair, 1, i, 2*i] for i, pair in reversed(list(enumerate(pairs))))
    inputs = prepared.run.read_csv_inputs(phenotype, covariate)
    # Existing direct callers retain their strict default. The public shared
    # pipeline reduces the same fixed-effect span after per-trait missingness.
    with pytest.raises(ValueError, match='rank[- ]deficient'):
        prepared.run.aligned_phenotype(*inputs, pairs, 'binary_a')
    selected = prepared.run.aligned_phenotype(*inputs, pairs, 'binary_a', reduce_redundant=True)
    assert selected[3] == ('Intercept', 'age')
    design = selected[-1]['covariate_design']
    assert design['dropped_covariate_names'] == ['sex', 'age_twice']
    assert design['center'] == [1., 3., 6.]
    raw = np.column_stack((np.ones(5), np.ones(5), np.arange(1., 6.), np.arange(2., 12., 2.)))
    np.testing.assert_allclose(selected[2]@np.linalg.pinv(selected[2]), raw@np.linalg.pinv(raw), atol=1e-12)
    output = tmp_path/'results'
    first = run_WGS_all_binanry(phenotype, covariate, root, output_directory=output,
        analyses=('individual',), grm='none', cpu_threads=1, single_region_size=10,
        phenotype_covariates={'binary': []}, impute_missing=False)
    assert first['completed'] and len(calls['fits']) == 2
    meta = json.loads((output/'models/trait_0000/model.private.json').read_text())
    assert meta['fit']['covariate_names'] == ['Intercept', 'age']
    assert meta['fit']['covariate_design'] == design
    assert meta['fit']['prepared_association_binding']['covariate_design'] == design
    resumed = run_WGS_all_binanry(phenotype, covariate, root, output_directory=output,
        analyses=('individual',), grm='none', cpu_threads=1, single_region_size=10,
        phenotype_covariates={'binary': []}, impute_missing=False, resume=True)
    assert resumed['completed'] and len(calls['fits']) == 2


def test_genetic_partition_keeps_fixed_checkpoint_ordinals_when_card_count_changes():
    original = dict(chromosomes=[dict(name='1', jobs=[dict(checkpoint_job_index=i) for i in range(3)]),
                               dict(name='2', jobs=[dict(checkpoint_job_index=i) for i in range(3, 7)])])
    for count in (1, 2, 4, 9):
        partitions = [prepared._partition(original, i, count) for i in range(count)]
        assigned = [job['checkpoint_job_index'] for partition in partitions
                    for chromosome in partition['chromosomes'] for job in chromosome['jobs']]
        assert sorted(assigned) == list(range(7)) and len(assigned) == len(set(assigned))
    assert original['chromosomes'][0]['jobs'][0] == {'checkpoint_job_index': 0}


def test_worker_cpu_pool_obeys_shared_quota_and_does_not_assign_visible_128(monkeypatch):
    monkeypatch.setattr(prepared.os, 'sched_getaffinity', lambda pid: set(range(128)))
    monkeypatch.setattr(prepared, 'detect_cpu_capacity', lambda: dict(effective_cores=40))
    resources = dict(prepared._cpu_resources(8), cpu_threads=8)
    assert resources['cpu_pool'] == list(range(40))
    for worker_id in range(9):
        selected = prepared._worker_cpu_selection(resources, worker_id)
        assert len(selected) == 8 and len(set(selected)) == 8 and set(selected) <= set(range(40))
    with pytest.raises(ValueError, match='effective affinity/cgroup CPU quota'):
        prepared._cpu_resources(41)
    monkeypatch.setattr(prepared, 'detect_cpu_capacity', lambda: dict(effective_cores=4))
    with pytest.raises(ValueError, match='worker affinity/cgroup CPU quota'):
        prepared._worker_cpu_selection(resources, 0)


@pytest.mark.parametrize('fail', [False, True])
def test_inline_cpu_policy_restores_existing_and_new_helper_threads(monkeypatch, fail):
    caller_main = set(range(128))
    affinities = {0: caller_main.copy(), 100: caller_main.copy(), 101: {10, 11}}
    thread_ids = [100, 101]
    torch_threads = [24]
    monkeypatch.setattr(prepared.os, 'sched_getaffinity', lambda pid: affinities[pid].copy())
    monkeypatch.setattr(prepared.os, 'sched_setaffinity', lambda pid, cpus: affinities.__setitem__(pid, set(cpus)))
    monkeypatch.setattr(prepared, '_thread_ids', lambda: thread_ids.copy())
    monkeypatch.setattr(prepared, 'detect_cpu_capacity', lambda: dict(effective_cores=40))
    monkeypatch.setattr(prepared.torch, 'get_num_threads', lambda: torch_threads[0])
    monkeypatch.setattr(prepared.torch, 'set_num_threads', lambda count: torch_threads.__setitem__(0, count))
    resources = dict(prepared._cpu_resources(8), cpu_threads=8)
    def invoke():
        with prepared._inline_cpu_policy(resources):
            assert torch_threads[0] == 8
            assert all(cpus == set(range(8)) for cpus in affinities.values())
            thread_ids.append(102)
            affinities[102] = set(range(8))
            if fail:
                raise RuntimeError('injected invocation failure')
    if fail:
        with pytest.raises(RuntimeError, match='injected invocation failure'):
            invoke()
    else:
        invoke()
    assert torch_threads[0] == 24
    assert affinities[0] == caller_main and affinities[100] == caller_main
    assert affinities[101] == {10, 11} and affinities[102] == caller_main


def test_modern_host_budget_uses_shared_gate_and_can_change_on_resume(tmp_path,monkeypatch):
    from fudan_wgs_toolkit import phewas_resources
    root,phenotype,covariate,_=fixture(tmp_path)
    calls=injected_execution(monkeypatch)
    policies=[]
    class Gate:
        def __init__(self,directory,**policy):policies.append((Path(directory),policy))
        def lease_factory(self,*args):pass
    monkeypatch.setattr(phewas_resources,'HostMemoryGate',Gate)
    output=tmp_path/'results'
    assert call(root,phenotype,covariate,output,host_memory_limit_gib=12)['completed']
    first=json.loads((output/'plan.private.json').read_text())
    assert first['host_memory_limit_gib']==12
    assert call(root,phenotype,covariate,output,host_memory_limit_gib=8,resume=True)['completed']
    second=json.loads((output/'plan.private.json').read_text())
    assert second['scientific_binding_sha256']==first['scientific_binding_sha256']
    assert second['host_memory_limit_gib']==8
    assert policies==[(output,dict(host_limit_gib=12,reserve_gib=None)),
                      (output,dict(host_limit_gib=8,reserve_gib=None))]
    assert all(kwargs['resources']['host_memory_lease_factory'] is not None for kwargs in calls['runner'])
    assert len(calls['fits'])==2


@pytest.mark.parametrize('failed',[False,True])
def test_modern_worker_closes_model_mappings_and_preserves_original_failure(tmp_path,monkeypatch,failed):
    from fudan_wgs_toolkit.chromosome_csv import ChromosomeCheckpointCSVWriter
    from fudan_wgs_toolkit.phewas_runtime import runtime
    root,phenotype,covariate,_=fixture(tmp_path)
    injected_execution(monkeypatch)
    bank_type=prepared.ModelRepository
    actions=[]
    class Repository(bank_type):
        def close(self):
            super().close()
            actions.append('close')
            if failed:raise MemoryError('secondary close failure')
    monkeypatch.setattr(prepared,'ModelRepository',Repository)
    original_abort=ChromosomeCheckpointCSVWriter.abort
    def abort(writer):
        actions.append('abort')
        original_abort(writer)
        if failed:raise MemoryError('secondary abort failure')
    monkeypatch.setattr(ChromosomeCheckpointCSVWriter,'abort',abort)
    if failed:
        def association(*args,**kwargs):raise ArithmeticError('primary association failure')
        monkeypatch.setattr(runtime,'run_batched_configuration',association)
        with pytest.raises(ArithmeticError,match='primary association'):
            call(root,phenotype,covariate,tmp_path/'results')
        assert actions==['abort','close']
        receipts=list((tmp_path/'results').glob('worker_*.exit.anonymous.json'))
        assert receipts and json.loads(receipts[0].read_text())['exit_code']==1
    else:
        assert call(root,phenotype,covariate,tmp_path/'results')['completed']
        assert actions==['close']
