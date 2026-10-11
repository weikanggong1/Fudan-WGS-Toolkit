"""Association-only receipt gates; anonymous fixtures are not benchmarks."""
import json
from pathlib import Path

import numpy as np
import pytest

from fudan_wgs_toolkit import phewas_run as run
from fudan_wgs_toolkit.phewas_inputs import prepare_phewas_inputs
from fudan_wgs_toolkit.phewas_storage import digest_file

SOURCE = 'a'*64


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return digest_file(path)


@pytest.fixture
def completed(tmp_path, monkeypatch):
    phenotypes, covariates = tmp_path/'phenotypes.csv', tmp_path/'covariates.csv'
    phenotypes.write_text('FID,IID,trait_a,trait_b\n0,11,1,0\n0,12,2,1\n0,13,3,0\n0,14,4,1\n')
    covariates.write_text('FID,IID,age\n0,11,20\n0,12,30\n0,13,40\n0,14,50\n')
    sample_ids = tmp_path/'sample_ids.npy'
    np.save(sample_ids, np.arange(11, 15, dtype=np.int64))
    prepared = tmp_path/'inputs'
    prepare_phewas_inputs(phenotypes=phenotypes, covariates=covariates,
        sample_ids=sample_ids, output=prepared, rm_nan=False)
    output = tmp_path/'output';output.mkdir()
    container = tmp_path/'container';container.mkdir()
    for name in ('data.bin','headers.bin','counts.bin'):
        (container/name).write_bytes(b'fixture stream')
    streams = run._stream_bindings([dict(container_directory=str(container))])
    cohort = tmp_path/'cohort.npy';np.save(cohort, np.arange(4, dtype=np.int64))
    plan = dict(output_directory=str(output), prepared_inputs_directory=str(prepared),
        prepared_inputs_manifest_sha256=digest_file(prepared/'manifest.private.json'),
        source_implementation_sha256=SOURCE, trait_count=2, rm_nan=False,
        kinship_npz=None, kinship_sha256=None, cohort_rows=str(cohort),
        cohort_sha256=digest_file(cohort), continuous_transform='paper',
        configuration=dict(chromosomes=[]), genotype_streams=streams, cpu_threads=2,
        host_memory_limit_gib=8, inline_worker=True)
    monkeypatch.setattr(run, '_implementation_sha256', lambda: SOURCE)
    traits = json.loads((prepared/'traits.private.json').read_text())['traits']
    models=[]
    for index, trait in enumerate(traits):
        target=output/'models'/f'trait_{index:04d}';target.mkdir(parents=True)
        np.save(target/'sample_rows.npy', np.arange(4, dtype=np.int64))
        hashes={}
        for label in ('normal','spa') if trait['family']=='binomial' else ('normal',):
            state=target/label;state.mkdir()
            arrays={}
            for name, array in dict(sample_ids=np.asarray(['11','12','13','14']),
                x=np.ones((4,1)), scaled_residuals=np.ones(4), precision_x=np.ones((4,1))).items():
                file=state/(name+'.npy');np.save(file,array)
                arrays[name]=dict(file=file.name,shape=list(array.shape),dtype=array.dtype.str,
                                  sha256=digest_file(file))
            hashes[label]=write(state/'state.private.json',dict(family=trait['family'],converged=True,
                has_kinship=False,matmul_mode='fp64' if label=='spa' else 'tf32',arrays=arrays))
        fit=dict(input_manifest_sha256=plan['prepared_inputs_manifest_sha256'],
            kinship_sha256=None,cohort_sha256=plan['cohort_sha256'],
            source_implementation_sha256=SOURCE,continuous_transform='paper',rm_nan=False,
            trait_index=index,phenotype=trait['name'],profile=trait['profile'])
        sha=write(target/'model.private.json',dict(n=4,family=trait['family'],fit=fit,
            normal_state='normal',spa_state='spa' if trait['family']=='binomial' else None,
            sample_rows_sha256=digest_file(target/'sample_rows.npy'),
            state_sha256=hashes['normal'],spa_state_sha256=hashes.get('spa')))
        models.append(dict(trait_index=index,path=str(target),n=4,family=trait['family'],sha256=sha))
    receipt=output/'fit_worker_0.complete.private.json'
    write(receipt,dict(worker_id=0,assigned=2,models=models,errors=[],mode='fresh_fit',
        model_fit_source_implementation_sha256=SOURCE,consumer_source_implementation_sha256=SOURCE,
        null_fit_compatibility_sha256=None,model_import_sha256=None))
    return plan, receipt, models


def rebind_metadata(receipt, index, callback):
    value=json.loads(receipt.read_text());entry=value['models'][index]
    path=Path(entry['path'])/'model.private.json'
    metadata=json.loads(path.read_text());callback(metadata)
    entry['sha256']=write(path,metadata);write(receipt,value)


def test_association_only_verifies_small_metadata_without_rehashing_model_or_input_arrays(completed,monkeypatch):
    plan, receipt, models=completed
    original=run.digest_file;reads=[]
    def checked(path):
        path=Path(path);reads.append(path)
        if path.suffix=='.npy' and path.name!='cohort.npy':
            pytest.fail('association-only must not rehash the complete stored array bank')
        return original(path)
    monkeypatch.setattr(run,'digest_file',checked)
    monkeypatch.setattr(run,'_verify_stored_model',lambda *a,**k:pytest.fail('bulk NPY verification'))
    before=receipt.read_bytes()
    assert run._all_models(plan,1,association_only=True)==models
    assert receipt.read_bytes()==before
    assert all(path.name!='phenotypes.npy' for path in reads)


def test_missing_fit_receipt_fails_immediately_without_waiting_or_refitting(completed,monkeypatch):
    plan, receipt, _=completed;receipt.unlink()
    monkeypatch.setattr(run.time,'sleep',lambda *a:pytest.fail('association-only cannot wait for fits'))
    monkeypatch.setattr(run,'fit_models',lambda *a,**k:pytest.fail('association-only cannot refit'))
    with pytest.raises(FileNotFoundError,match='every existing completed fit receipt'):
        run._all_models(plan,1,association_only=True)


@pytest.mark.parametrize('problem',['receipt_source','receipt_coverage','receipt_error','input_sha',
    'input_status','input_trait_count','policy','cohort','source','model_source',
    'model_trait','model_count','model_policy','state_convergence','state_sha','missing_spa','input_completion',
    'input_sample_count','receipt_assigned','missing_array'])
def test_association_only_rejects_shifted_bindings_and_incomplete_models(completed,problem):
    plan, receipt, models=completed
    if problem.startswith('receipt_'):
        value=json.loads(receipt.read_text())
        if problem=='receipt_source':value['consumer_source_implementation_sha256']='b'*64
        elif problem=='receipt_coverage':value['models'].pop()
        elif problem=='receipt_assigned':value['assigned']=1
        else:value['errors']=[{'error_type':'ArithmeticError'}]
        write(receipt,value)
    elif problem=='input_sha':plan['prepared_inputs_manifest_sha256']='b'*64
    elif problem=='input_completion':
        path=Path(plan['prepared_inputs_directory'])/'status.anonymous.json'
        write(path,dict(status='preparing',manifest_sha256=plan['prepared_inputs_manifest_sha256']))
    elif problem in ('input_status','input_trait_count','input_sample_count'):
        manifest=Path(plan['prepared_inputs_directory'])/'manifest.private.json'
        value=json.loads(manifest.read_text())
        if problem=='input_status':value['status']='preparing'
        elif problem=='input_sample_count':value['sample_count']=5
        else:value['trait_count']=1
        plan['prepared_inputs_manifest_sha256']=write(manifest,value)
    elif problem=='policy':plan['rm_nan']=True
    elif problem=='cohort':Path(plan['cohort_rows']).write_bytes(b'changed cohort')
    elif problem=='source':plan['source_implementation_sha256']='b'*64
    elif problem in ('model_source','model_trait','model_policy','model_count'):
        def change(meta):
            if problem=='model_source':meta['fit']['source_implementation_sha256']='b'*64
            elif problem=='model_trait':meta['fit']['trait_index']=1
            elif problem=='model_policy':meta['fit']['rm_nan']=True
            else:meta['n']=5
        rebind_metadata(receipt,0,change)
    elif problem=='state_convergence':
        state=Path(models[0]['path'])/'normal/state.private.json'
        value=json.loads(state.read_text());value['converged']=False
        sha=write(state,value)
        rebind_metadata(receipt,0,lambda meta:meta.update(state_sha256=sha))
    elif problem=='state_sha':
        (Path(models[0]['path'])/'normal/state.private.json').write_text('{}')
    elif problem=='missing_array':(Path(models[0]['path'])/'normal/x.npy').unlink()
    else:(Path(models[1]['path'])/'spa/state.private.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError,RuntimeError)):
        run._all_models(plan,1,association_only=True)


def test_association_only_success_never_calls_fit_or_rewrites_completed_fit_receipt(completed,monkeypatch):
    from fudan_wgs_toolkit import phewas_resources
    plan, receipt, _=completed
    before=receipt.read_bytes();stat=receipt.stat()
    affinities={}
    monkeypatch.setattr(run.os,'sched_getaffinity',lambda pid:affinities.get(pid,{0,1}))
    monkeypatch.setattr(run.os,'sched_setaffinity',lambda pid,cpus:affinities.__setitem__(pid,set(cpus)))
    monkeypatch.setattr(run.torch,'set_num_threads',lambda value:None)
    monkeypatch.setattr(run.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(run.torch.cuda,'set_device',lambda value:None)
    monkeypatch.setattr(run,'fit_models',lambda *a,**k:pytest.fail('fit receipt must remain immutable'))
    class Repository:
        def __init__(self,*a,**k):pass
        def summary(self):return {}
        def close(self):pass
    class Writer:
        def __init__(self,*a,**k):pass
        def finish(self):return {'complete':True}
    class Gate:
        def __init__(self,*a,**k):pass
        def lease_factory(self,*a,**k):pass
    monkeypatch.setattr(run,'ModelRepository',Repository)
    monkeypatch.setattr(run,'StreamingCSVWriter',Writer)
    monkeypatch.setattr(phewas_resources,'HostMemoryGate',Gate)
    report=run.execute_worker(plan,association_only=True)
    assert report['association_only'] is True and report['models_refitted']==0
    assert report['existing_fit_receipts_preserved'] is True
    assert receipt.read_bytes()==before
    assert (receipt.stat().st_mtime_ns,receipt.stat().st_ctime_ns)==(stat.st_mtime_ns,stat.st_ctime_ns)


@pytest.mark.parametrize('entry',[run.execute_worker,run._execute_worker])
def test_fit_only_and_association_only_are_mutually_exclusive_before_any_work(entry,tmp_path):
    with pytest.raises(ValueError,match='mutually exclusive'):
        entry(dict(output_directory=str(tmp_path/'untouched')),fit_only=True,association_only=True)
    assert not (tmp_path/'untouched').exists()


def test_cli_association_only_requires_existing_plan_and_excludes_fit_only(tmp_path,monkeypatch):
    with pytest.raises(SystemExit):run.main(['--association-only'])
    with pytest.raises(SystemExit):run.main(['--fit-only','--association-only'])
    path=tmp_path/'plan.json';path.write_text('{}')
    calls=[]
    monkeypatch.setattr(run,'execute_worker',lambda plan,**kw:calls.append(kw))
    run.main(['--plan',str(path),'--association-only'])
    assert calls[-1]['association_only'] is True and calls[-1]['fit_only'] is False
    run.main(['--plan',str(path)])
    assert calls[-1]['association_only'] is False and calls[-1]['fit_only'] is False
