"""Small anonymous API contracts; no synthetic fixture is a benchmark."""
import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from fudan_wgs_toolkit import phewas_run
from fudan_wgs_toolkit.phewas_inputs import PhewasInputs, prepare_phewas_inputs


def csv_table(path, rows):
    with path.open('w', newline='') as stream:
        csv.writer(stream).writerows(rows)


@pytest.fixture
def api_inputs(tmp_path, monkeypatch):
    root = tmp_path/'cache'
    root.mkdir()
    axis = np.arange(301, 307, dtype=np.int64)
    np.save(root/'sample_ids.npy', axis, allow_pickle=False)
    gene_catalog = root/'genes.json'
    gene_catalog.write_text(json.dumps([
        dict(kind='coding', gene_name='example_gene', start=2, end=8),
        dict(kind='noncoding', gene_name='example_gene'),
        dict(kind='ncrna', gene_name='example_rna', start=3, end=7)]))
    entry = dict(name='21', container_directory=str(root/'chr21'),
        metadata_directory=str(root/'metadata'), gene_catalog=str(gene_catalog), _maximum_position=25)
    container = Path(entry['container_directory'])
    container.mkdir()
    for name in ('data.bin', 'headers.bin', 'counts.bin'):
        (container/name).write_bytes(b'01')
    # Deliberately declare a missing profile file: the public API must not
    # consume it or infer covariates/families from the cache sidecar.
    manifest = dict(sample_ids='sample_ids.npy', annotation_catalog={},
                    phewas=dict(covariate_profiles='ignored.profiles.csv'))
    monkeypatch.setattr(phewas_run, '_dataset', lambda *args: (
        root, root/'cache_dataset.json', manifest, [entry], axis))
    phenotype, covariate = tmp_path/'p.csv', tmp_path/'c.csv'
    csv_table(phenotype, [['sample_id', 'trait_a', 'trait_b', 'trait_c'],
        [301, 0, 0, 0], [302, 1, 1, 1], [303, '', 2, 0],
        [304, 1, 3, 1], [305, 0, '', 0], [306, 1, 4, 1]])
    csv_table(covariate, [['sample_id', 'age', 'extra'], [301, 30, ''], [302, 31, 1],
        [303, 32, 2], [304, 33, 3], [305, 34, 4], [306, 35, 5]])
    # A neighboring profile file is also deliberately unrelated.
    csv_table(covariate.with_suffix('.profiles.csv'), [['wrong', 'schema']])
    return dict(phenotype_csv=phenotype, covariate_csv=covariate,
                cache_directory=root, output_directory=tmp_path/'out', grm='none', rm_nan=False)


def test_api_per_trait_family_override_and_covariate_missingness(api_inputs):
    plan = phewas_run.prepare_phewas_run(**api_inputs, analyses=('coding',),
        phenotype_families={'trait_a': 'gaussian'}, phenotype_covariates={'trait_a': ['age']})
    with_inputs = PhewasInputs(plan['prepared_inputs_directory'])
    try:
        a, b, c = [with_inputs.trait(index) for index in range(3)]
        assert [a['family'], b['family'], c['family']] == ['gaussian', 'gaussian', 'binomial']
        assert a['sample_indices'].tolist() == [0, 1, 3, 4, 5]
        assert b['sample_indices'].tolist() == [1, 2, 3, 5]
        assert c['sample_indices'].tolist() == [1, 2, 3, 4, 5]
        assert a['covariate_names'] == ['age']
        assert c['covariate_names'] == ['age', 'extra']
        assert 'profiles' not in with_inputs.manifest['sources']
    finally:
        with_inputs.close()
    assert plan['kinship_npz'] is None and plan['kinship_sha256'] is None
    assert plan['configuration']['analysis_options']['variant_type'] == 'SNV'
    assert [job['kind'] for job in plan['configuration']['chromosomes'][0]['jobs']] == ['coding']


def test_older_nonresumable_deployment_observes_actual_streams(api_inputs, monkeypatch):
    plan = dict(output_directory=str(api_inputs['output_directory']),
                configuration={'chromosomes':[{'container_directory':str(api_inputs['cache_directory']/'chr21')}]})
    expected={'scope':'this invocation only'}
    monkeypatch.setattr(phewas_run,'_execute_worker',lambda *args,**kwargs:expected)
    assert phewas_run.execute_worker(plan)==expected
    assert json.loads((Path(plan['output_directory'])/'worker_0.exit.anonymous.json').read_text())['exit_code']==0
    with pytest.raises(ValueError,match='persistent stream bindings'):
        phewas_run.execute_worker(dict(plan,resume=True))


def test_sparse_grm_is_default_and_explicit_none_is_required(api_inputs):
    del api_inputs['grm']
    with pytest.raises(ValueError, match='sparse GRM'):
        phewas_run.prepare_phewas_run(**api_inputs)
    assert not Path(api_inputs['output_directory']).exists()


def test_resume_requires_same_trait_options_and_scientific_plan(api_inputs):
    first = phewas_run.prepare_phewas_run(**api_inputs, analyses=('coding',),
        phenotype_covariates={'trait_a': ['age']})
    with pytest.raises(FileExistsError, match='resume=True'):
        phewas_run.prepare_phewas_run(**api_inputs, analyses=('coding',))
    again = phewas_run.prepare_phewas_run(**api_inputs, analyses=('coding',), resume=True,
        phenotype_covariates={'trait_a': ['age']})
    assert first['prepared_inputs_manifest_sha256'] == again['prepared_inputs_manifest_sha256']
    assert list(Path(again['output_directory']).glob('plan.previous.*.private.json'))
    with pytest.raises(ValueError, match='family/covariate'):
        phewas_run.prepare_phewas_run(**api_inputs, analyses=('coding',), resume=True)
    with pytest.raises(ValueError, match='scientific/source'):
        phewas_run.prepare_phewas_run(**api_inputs, analyses=('coding',), resume=True,
            phenotype_covariates={'trait_a': ['age']}, gene_variant_type='Indel')


@pytest.mark.parametrize('stream', ['data.bin', 'headers.bin', 'counts.bin'])
def test_resume_rejects_equal_size_stream_change_without_rehashing(api_inputs, stream):
    from fudan_wgs_toolkit.phewas_cache import _verify_stream_bindings
    first = phewas_run.prepare_phewas_run(**api_inputs, analyses=('coding',))
    path = Path(api_inputs['cache_directory'])/'chr21'/stream
    before = path.stat()
    path.write_bytes(b'10')
    # Restoring the timestamp and keeping the size cannot restore the inode's
    # change time, so a completed-block-only attempt must still reject reuse.
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ValueError, match='genotype stream changed'):
        _verify_stream_bindings(first['genotype_streams'])
    with pytest.raises(ValueError, match='scientific/source bindings'):
        phewas_run.prepare_phewas_run(**api_inputs, analyses=('coding',), resume=True)
    assert json.loads((Path(api_inputs['output_directory'])/'plan.private.json').read_text()) == first


@pytest.mark.parametrize('stream', ['data.bin', 'headers.bin', 'counts.bin'])
def test_missing_stream_rejects_before_prepared_inputs_publication(api_inputs, stream):
    path = Path(api_inputs['cache_directory'])/'chr21'/stream
    path.unlink()
    with pytest.raises(ValueError, match='genotype stream is missing'):
        phewas_run.prepare_phewas_run(**api_inputs, analyses=('coding',))
    assert not Path(api_inputs['output_directory']).exists()


def test_stream_stat_binding_rejects_replacement_and_partial_or_invalid_proof(tmp_path):
    from fudan_wgs_toolkit.phewas_cache import _stream_bindings, _verify_stream_bindings
    container = tmp_path/'chr1'
    container.mkdir()
    for name in ('data.bin', 'headers.bin', 'counts.bin'):
        (container/name).write_bytes(b'01')
    binding = _stream_bindings([dict(container_directory=str(container))])
    assert len(binding) == 3
    assert all(set(fields) == {'dev', 'ino', 'size', 'mtime_ns', 'ctime_ns'}
               for fields in binding.values())
    _verify_stream_bindings(binding)
    path = container/'data.bin'
    old = path.stat()
    replacement = container/'replacement.tmp'
    replacement.write_bytes(b'01')
    os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns))
    replacement.replace(path)
    assert path.stat().st_size == old.st_size
    assert path.stat().st_mtime_ns == old.st_mtime_ns
    with pytest.raises(ValueError, match='genotype stream changed'):
        _verify_stream_bindings(binding)
    current = _stream_bindings([dict(container_directory=str(container))])
    current.pop(str(path.absolute()))
    with pytest.raises(ValueError, match='all three'):
        _verify_stream_bindings(current)
    with pytest.raises(ValueError, match='bindings are missing'):
        _verify_stream_bindings(None)


@pytest.mark.parametrize('change_before_call', [False, True])
def test_worker_checks_streams_before_and_after_even_if_all_tiles_reused(
        tmp_path, monkeypatch, change_before_call):
    from fudan_wgs_toolkit.phewas_cache import _stream_bindings
    container = tmp_path/'chr1'
    container.mkdir()
    for name in ('data.bin', 'headers.bin', 'counts.bin'):
        (container/name).write_bytes(b'01')
    plan = dict(output_directory=str(tmp_path/'result'),
                genotype_streams=_stream_bindings([dict(container_directory=str(container))]))
    path = container/'data.bin'
    calls = []
    if change_before_call:
        path.write_bytes(b'10')
        # Some shared filesystems coarsen timestamps; make the intended
        # changed identity explicit rather than depending on a clock tick.
        timestamp = path.stat()
        os.utime(path, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns+1_000_000_000))
    def reused(*args, **kwargs):
        calls.append(True)
        path.write_bytes(b'10')
        timestamp = path.stat()
        os.utime(path, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns+1_000_000_000))
        return dict(reused_blocks=1)
    monkeypatch.setattr(phewas_run, '_execute_worker', reused)
    with pytest.raises(ValueError, match='genotype stream changed'):
        phewas_run.execute_worker(plan)
    assert len(calls) == (0 if change_before_call else 1)
    exit_record = json.loads((Path(plan['output_directory'])/'worker_0.exit.anonymous.json').read_text())
    assert exit_record['exit_code'] == 1


def test_single_chromosome_jobs_partition_once_across_gpus(api_inputs):
    config = phewas_run.build_configuration(api_inputs['cache_directory'], chromosomes=[21],
        single_region_size=10, single_mac_cutoff=4, single_group_variants=7,
        gene_variant_type='Indel', covariance_block_size=64, long_mask_rank=32,
        memory_limit_gib=20)
    jobs = config['chromosomes'][0]['jobs']
    singles = [job for job in jobs if job['kind'] == 'individual']
    assert [(job['arguments']['start'], job['arguments']['end']) for job in singles] == [(1, 10), (11, 20), (21, 25)]
    assert singles[0]['arguments']['mac_cutoff'] == 4
    assert singles[0]['arguments']['subset_variants_num'] == 7
    assert singles[0]['arguments']['variant_type'] == 'variant'
    assert config['analysis_options']['variant_type'] == 'Indel'
    assert config['analysis_options']['cached_variant_tile_size'] == 64
    assert config['analysis_options']['long_mask_rank'] == 32
    shards = [phewas_run._partition_configuration(config, index, 3) for index in range(3)]
    assigned = [job['checkpoint_job_index'] for shard in shards
                for chromosome in shard['chromosomes'] for job in chromosome['jobs']]
    assert len(assigned) == len(set(assigned)) == len(jobs)
    assert set(assigned) == set(range(len(jobs)))
    assert all(shard['chromosomes'] for shard in shards)
    idle = phewas_run._partition_configuration(config, len(jobs), len(jobs)+1)
    assert idle['chromosomes'] == []


def test_single_gpu_uses_this_call_and_restores_thread_policy(tmp_path, monkeypatch):
    from fudan_wgs_toolkit import cpu_budget
    calls, thread_updates = [], []
    from fudan_wgs_toolkit import phewas_partition
    budgets = []
    monkeypatch.setattr(phewas_partition, 'gpu_model_cache_capacity',
        lambda gpu_ids, **options: budgets.append((gpu_ids, options)) or 8*2**30)
    monkeypatch.setattr(cpu_budget, 'detect_cpu_capacity', lambda: dict(effective_cores=8))
    monkeypatch.setattr(phewas_run.os, 'sched_getaffinity', lambda pid: set(range(8)))
    monkeypatch.setattr(phewas_run.os, 'sched_setaffinity', lambda pid, cpus: None)
    monkeypatch.setattr(phewas_run.torch, 'get_num_threads', lambda: 3)
    monkeypatch.setattr(phewas_run.torch, 'set_num_threads', thread_updates.append)
    monkeypatch.setattr(phewas_run, 'prepare_phewas_run', lambda *args, **kwargs: dict(
        output_directory=str(tmp_path), attempt_id='example'))
    def execute(plan, **kwargs):
        calls.append((plan.copy(), kwargs))
        configuration = dict(chromosomes=[dict(name='21', jobs=[dict(kind='individual')])])
        dispatch = phewas_partition.build_trait_dispatch([dict(trait_index=0, sha256='a'*64)],
            1, configuration, source_binding='inline-thread-contract', strategy='genetic_jobs')
        phewas_partition.publish_trait_dispatch(tmp_path, dispatch, plan['attempt_id'])
        return dict(worker_id=0, association_rows=dict.fromkeys(('individual', 'coding', 'noncoding', 'ncrna'), 0),
            assigned_trait_indices=[0], association_dispatch_sha256=dispatch['dispatch_binding_sha256'],
            assigned_genetic_schedule_sha256=dispatch['workers'][0]['genetic_schedule_sha256'],
            job_groups=1, trait_rows={0:dict.fromkeys(('individual', 'coding', 'noncoding', 'ncrna'), 0)})
    monkeypatch.setattr(phewas_run, 'execute_worker', execute)
    result = phewas_run.run_phewas('p.csv', 'c.csv', 'cache', output_directory=tmp_path,
                                 gpu_ids=(2,), cpu_threads=8)
    assert len(result) == 1 and len(calls) == 1
    assert calls[0][1] == dict(worker_id=0, worker_count=1, device='cuda:2')
    assert calls[0][0]['inline_worker'] and calls[0][0]['cpu_pool'] == list(range(8))
    assert thread_updates == [3]
    assert budgets[0][0] == (2,) and budgets[0][1]['memory_limit_gib'] is None


def test_resume_completed_tiles_do_not_hide_corrupt_null_arrays(api_inputs):
    from fudan_wgs_toolkit.phewas_storage import digest_file
    plan = phewas_run.prepare_phewas_run(**api_inputs, analyses=('coding',),
        phenotype_families={'trait_a': 'gaussian'})
    loaded = PhewasInputs(plan['prepared_inputs_directory'])
    try:
        rows = loaded.trait(0)['sample_indices']
        ids = loaded.sample_ids[rows].astype(str)
    finally:
        loaded.close()
    target = Path(plan['output_directory'])/'models/trait_0000'
    state_root = target/'normal'
    state_root.mkdir(parents=True)
    np.save(target/'sample_rows.npy', rows.astype(np.int64), allow_pickle=False)
    arrays = {}
    for name, values in dict(sample_ids=ids, x=np.ones((len(ids), 1)),
            scaled_residuals=np.zeros(len(ids)), precision_x=np.ones((len(ids), 1))).items():
        path = state_root/(name+'.npy')
        np.save(path, values, allow_pickle=False)
        arrays[name] = dict(file=path.name, shape=list(values.shape), dtype=values.dtype.str,
                            sha256=digest_file(path))
    state = state_root/'state.private.json'
    state.write_text(json.dumps(dict(family='gaussian', arrays=arrays)))
    metadata = dict(family='gaussian', n=len(ids), normal_state='normal', spa_state=None,
        sample_rows_sha256=digest_file(target/'sample_rows.npy'), state_sha256=digest_file(state),
        fit=dict(input_manifest_sha256=plan['prepared_inputs_manifest_sha256'], **phewas_run._fit_binding(plan)))
    (target/'model.private.json').write_text(json.dumps(metadata))
    # The model barrier must verify saved state even when every association
    # block has a complete marker and therefore no GPU acquire is necessary.
    checkpoint = Path(plan['output_directory'])/'csv/.checkpoints/blocks/dummy.complete.private.json'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps(dict(complete=True)))
    accepted = phewas_run.fit_models(plan, device='cpu', trait_indices=[0])
    assert accepted['errors'] == [] and accepted['models'][0]['model_arrays_materialized'] is False
    path = state_root/'x.npy'
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match='immutable fitted-state identity|checksum'):
        phewas_run.fit_models(plan, device='cpu', trait_indices=[0])


@pytest.mark.parametrize('families,covariates', [({'unknown': 'gaussian'}, None),
    ({'trait_a': 'poisson'}, None), ('gaussian', None),
    ('auto', {'trait_a': ['unknown']}), ('auto', {'trait_a': ['age', 'age']})])
def test_invalid_trait_options_are_rejected_before_input_publication(api_inputs, families, covariates):
    with pytest.raises(ValueError):
        phewas_run.prepare_phewas_run(**api_inputs, phenotype_families=families,
                                      phenotype_covariates=covariates)
    assert not (Path(api_inputs['output_directory'])/'inputs/manifest.private.json').exists()


def test_explicit_binomial_checks_every_source_value_even_outside_cache(api_inputs):
    with Path(api_inputs['phenotype_csv']).open('a', newline='') as stream:
        csv.writer(stream).writerow([999, 0, 0, 2])
    with pytest.raises(ValueError, match='Binary phenotype'):
        phewas_run.prepare_phewas_run(**api_inputs, phenotype_families={'trait_c': 'binomial'})


def test_explicit_empty_covariate_list_keeps_an_intercept_only_profile(api_inputs):
    plan = phewas_run.prepare_phewas_run(**api_inputs, phenotype_covariates={'trait_a': []})
    loaded = PhewasInputs(plan['prepared_inputs_directory'])
    try:
        trait = loaded.trait(0)
        assert trait['covariates'].shape == (5, 0)
        assert trait['sample_indices'].tolist() == [0, 1, 3, 4, 5]
        design, metadata = phewas_run._design(trait['covariates'], [])
        assert design.shape == (5, 1) and np.all(design == 1)
        assert metadata['covariate_names'] == ['Intercept']
    finally:
        loaded.close()


@pytest.mark.parametrize('options', [dict(cpu_threads=0), dict(memory_limit_gib=-1),
    dict(memory_reserve_gib=float('nan')), dict(memory_limit_gib=1, memory_reserve_gib=2),
    dict(device_cache_bytes=-1), dict(resume=1), dict(analyses=('coding', 'coding'))])
def test_invalid_public_resource_options_do_not_prepare_inputs(api_inputs, options):
    with pytest.raises(ValueError):
        phewas_run.prepare_phewas_run(**api_inputs, **options)
    assert not Path(api_inputs['output_directory']).exists()


def test_public_host_budget_is_resource_only_and_recorded_in_plan(api_inputs):
    first=phewas_run.prepare_phewas_run(**api_inputs,analyses=('coding',),host_memory_limit_gib=12)
    assert first['host_memory_limit_gib']==12
    again=phewas_run.prepare_phewas_run(**api_inputs,analyses=('coding',),resume=True,host_memory_limit_gib=8)
    assert again['host_memory_limit_gib']==8
    assert again['configuration']==first['configuration']
    assert again['prepared_inputs_manifest_sha256']==first['prepared_inputs_manifest_sha256']
    assert phewas_run._host_memory_policy(dict(again,host_limit_gib=200,host_reserve_gib=20,
        host_limit_gib_by_worker={'0':40,'1':80}))==dict(host_limit_gib=8,reserve_gib=None)
    assert phewas_run._host_memory_policy(dict(again,host_memory_limit_gib=None,
        host_limit_gib=200))==dict(host_limit_gib=None,reserve_gib=None)


def test_legacy_host_policy_keeps_explicit_budget_and_uses_one_shared_limit():
    assert phewas_run._host_memory_policy(dict(host_limit_gib=200,host_reserve_gib=20))==dict(
        host_limit_gib=200,reserve_gib=20)
    plan=dict(host_limit_gib=200,host_reserve_gib=20,host_limit_gib_by_worker={'0':80,'1':120})
    assert phewas_run._host_memory_policy(plan)==dict(host_limit_gib=80,reserve_gib=20)
    assert phewas_run._host_memory_policy({})==dict(host_limit_gib=None,reserve_gib=None)


@pytest.mark.parametrize('limit',[0,-1,True,float('nan'),float('inf'),'12'])
def test_host_budget_rejected_before_preparing_or_creating_output(api_inputs,limit):
    with pytest.raises(ValueError,match='host_memory_limit_gib'):
        phewas_run.prepare_phewas_run(**api_inputs,host_memory_limit_gib=limit)
    assert not Path(api_inputs['output_directory']).exists()


def test_unified_and_phewas_cli_forward_shared_host_budget(monkeypatch,tmp_path,capsys):
    from fudan_wgs_toolkit import run as public
    calls=[]
    def unified(**options):
        calls.append(options)
        return {'completed':True}
    monkeypatch.setattr(public,'run_WGS_all_binanry',unified)
    public.main(['phenotypes.csv','covariates.csv','prepared','--output-directory','output',
                 '--host-memory-limit-gib','64'])
    assert calls[-1]['host_memory_limit_gib']==64
    def phewas(*args,**options):calls.append(options);return []
    monkeypatch.setattr(phewas_run,'run_phewas',phewas)
    phewas_run.main(['--phenotype-csv','phenotypes.csv','--covariate-csv','covariates.csv',
        '--cache-directory','prepared','--output-directory','output','--host-memory-limit-gib','32'])
    assert calls[-1]['host_memory_limit_gib']==32
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'host_limit_gib':200}))
    monkeypatch.setattr(phewas_run,'execute_worker',lambda plan,**options:calls.append(plan))
    phewas_run.main(['--plan',str(plan),'--host-memory-limit-gib','16'])
    assert calls[-1]['host_memory_limit_gib']==16 and calls[-1]['host_limit_gib']==200
