"""Legacy absent/False plan encoding and real checkpoint proof contracts."""
import copy
import hashlib
import json
import math

import pytest

from fudan_wgs_toolkit import continuous_run,phewas_run,prepared_association
from fudan_wgs_toolkit.phewas_trait_csv import PerTraitCheckpointCSVWriter
from test_phewas_public_options import api_inputs


def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(value).hexdigest()


@pytest.mark.parametrize('policy',[None,0,1,'false'])
def test_normalization_never_treats_non_booleans_as_legacy_false(policy):
    with pytest.raises(ValueError,match='single_cross_frame_batches'):
        phewas_run._legacy_single_batch_configuration({'single_cross_frame_batches':policy})


def test_known_false_normalization_is_narrow_and_does_not_mutate_inputs():
    nested={'mac_cutoff':20};missing={'other':nested}
    explicit=dict(missing,single_cross_frame_batches=False)
    true=dict(missing,single_cross_frame_batches=True)
    normalize=phewas_run._legacy_single_batch_configuration
    assert normalize(missing)==normalize(explicit)==missing
    assert normalize(explicit)['other'] is nested
    assert 'single_cross_frame_batches' in explicit
    assert normalize(true)!=missing and normalize(true)['single_cross_frame_batches'] is True
    assert normalize(dict(explicit,other={'mac_cutoff':21}))!=missing


@pytest.mark.parametrize('representation',['missing','explicit_false'])
def test_population_resume_retains_old_raw_configuration_sha_and_completed_receipt(api_inputs,monkeypatch,representation):
    monkeypatch.setattr(phewas_run,'_implementation_sha256',lambda:'fixed-legacy-policy-source')
    original=phewas_run.build_configuration
    def historical(*args,**kwargs):
        config=original(*args,**kwargs)
        if representation=='explicit_false':config['single_cross_frame_batches']=False
        else:config.pop('single_cross_frame_batches',None)
        return config
    with monkeypatch.context() as temporary:
        temporary.setattr(phewas_run,'build_configuration',historical)
        first=phewas_run.prepare_phewas_run(**api_inputs,analyses=('individual',))
    configuration=copy.deepcopy(first['configuration']);old_sha=digest(encoded(configuration))
    output=api_inputs['output_directory']/'csv'
    binding=dict(source_implementation_sha256=first['source_implementation_sha256'],
        input_manifest_sha256=first['prepared_inputs_manifest_sha256'],configuration_sha256=old_sha,
        models={'0':'anonymous-contract-model'})
    writer=PerTraitCheckpointCSVWriter(output,binding=binding,minimum_free_bytes=0)
    row=dict(CHR='21',POS=7,REF='A',ALT='C',ALT_AF=.2,MAF=.2,N=6,pvalue=.5,
        pvalue_log10=-math.log10(.5),Score=.5,Score_se=1.,Est=.5,Est_se=1.)
    proof=writer.write_tile('individual','21',[(0,[row])],job_index=0,block_identity='legacy-request-hash',
        ordinal_before=0,ordinal_after=1)
    final=writer.finish()
    csv_hashes={receipt['csv_file']:digest((output/receipt['csv_file']).read_bytes()) for receipt in final['receipts']}
    standalone={receipt['csv_file']:(output/receipt['csv_file']).with_name(
        (output/receipt['csv_file']).name+'.receipt.json').read_bytes() for receipt in final['receipts']}
    resumed=phewas_run.prepare_phewas_run(**api_inputs,analyses=('individual',),resume=True,single_cross_frame_batches=False)
    assert resumed['configuration']==configuration
    assert digest(encoded(resumed['configuration']))==old_sha
    assert ('single_cross_frame_batches' in resumed['configuration'])==(representation=='explicit_false')
    for receipt in final['receipts']:
        path=output/receipt['csv_file']
        assert digest(path.read_bytes())==csv_hashes[receipt['csv_file']]
        assert path.with_name(path.name+'.receipt.json').read_bytes()==standalone[receipt['csv_file']]
    reopened=PerTraitCheckpointCSVWriter(output,binding=dict(binding,configuration_sha256=digest(encoded(resumed['configuration']))),
        resume=True,worker_id='1',minimum_free_bytes=0)
    cached=reopened.checkpoint_tile('individual','21',0,[0],block_identity='legacy-request-hash',ordinal_before=0)
    assert cached['receipt_sha256']==proof['receipt_sha256']
    assert reopened.finish()['reused_blocks']==1
    with pytest.raises(ValueError,match='resume plan scientific/source bindings differ'):
        phewas_run.prepare_phewas_run(**api_inputs,analyses=('individual',),resume=True,single_cross_frame_batches=True)


@pytest.mark.parametrize('change',['source','input','other_configuration'])
def test_population_false_compatibility_does_not_relax_other_bindings(api_inputs,monkeypatch,change):
    monkeypatch.setattr(phewas_run,'_implementation_sha256',lambda:'fixed-legacy-policy-source')
    plan=phewas_run.prepare_phewas_run(**api_inputs,analyses=('individual',))
    path=api_inputs['output_directory']/'plan.private.json'
    plan['configuration'].pop('single_cross_frame_batches',None)
    if change=='source':plan['source_implementation_sha256']='different-source'
    elif change=='input':plan['prepared_inputs_manifest_sha256']='different-input'
    else:plan['configuration']['analysis_options']['rv_num_cutoff']=987
    path.write_text(json.dumps(plan));before=path.read_bytes()
    with pytest.raises(ValueError,match='resume plan scientific/source bindings differ'):
        phewas_run.prepare_phewas_run(**api_inputs,analyses=('individual',),resume=True,single_cross_frame_batches=False)
    assert path.read_bytes()==before


@pytest.mark.parametrize('representation',['missing','explicit_false'])
def test_modern_resume_keeps_original_configuration_scientific_sha_and_csv(tmp_path,monkeypatch,representation):
    from test_continuous_prepared import continuous_fixture,cpu_single_control,call
    root,phenotype,covariate,_=continuous_fixture(tmp_path);cpu_single_control(monkeypatch)
    original=prepared_association._configuration
    def historical(*args,**kwargs):
        config=original(*args,**kwargs)
        if representation=='explicit_false':config['single_cross_frame_batches']=False
        else:config.pop('single_cross_frame_batches',None)
        return config
    output=tmp_path/'out'
    with monkeypatch.context() as temporary:
        temporary.setattr(prepared_association,'_configuration',historical)
        first=call(continuous_run.run_WGS_continuous,root,phenotype,covariate,output,rm_nan=False)
    plan=json.loads((output/'plan.private.json').read_text());configuration=copy.deepcopy(plan['configuration'])
    old_configuration_sha=digest(encoded(configuration));old_scientific_sha=plan['scientific_binding_sha256']
    csv_hashes={str(path.relative_to(output)):digest(path.read_bytes()) for path in (output/'csv').rglob('*.csv.gz')}
    resumed=call(continuous_run.run_WGS_continuous,root,phenotype,covariate,output,rm_nan=False,
        resume=True,single_cross_frame_batches=False)
    saved=json.loads((output/'plan.private.json').read_text())
    assert first['completed'] and resumed['completed']
    assert saved['configuration']==configuration and digest(encoded(saved['configuration']))==old_configuration_sha
    assert saved['scientific_binding_sha256']==old_scientific_sha
    assert resumed['workers'][0]['output']['reused_blocks']==2
    assert csv_hashes=={str(path.relative_to(output)):digest(path.read_bytes()) for path in (output/'csv').rglob('*.csv.gz')}
    with pytest.raises(ValueError,match='resume prepared association source/input/schedule binding differs'):
        call(continuous_run.run_WGS_continuous,root,phenotype,covariate,output,rm_nan=False,
            resume=True,single_cross_frame_batches=True)


@pytest.mark.parametrize('change',['source','input','other_configuration','scientific_proof'])
def test_modern_false_compatibility_rejects_other_binding_changes(tmp_path,monkeypatch,change):
    from test_continuous_prepared import continuous_fixture,cpu_single_control,call
    root,phenotype,covariate,_=continuous_fixture(tmp_path);cpu_single_control(monkeypatch)
    output=tmp_path/'out';call(continuous_run.run_WGS_continuous,root,phenotype,covariate,output,rm_nan=False)
    file=output/'plan.private.json';plan=json.loads(file.read_text())
    if change=='source':plan['source_implementation_sha256']='different-source'
    elif change=='input':plan['input_binding']['dataset_sha256']='different-input'
    elif change=='other_configuration':plan['configuration']['analysis_options']['rv_num_cutoff']=987
    else:plan['scientific_binding_sha256']='different-proof'
    file.write_text(json.dumps(plan));before=file.read_bytes()
    with pytest.raises(ValueError,match='resume prepared association source/input/schedule binding differs'):
        call(continuous_run.run_WGS_continuous,root,phenotype,covariate,output,rm_nan=False,
            resume=True,single_cross_frame_batches=False)
    assert file.read_bytes()==before


def test_old_missing_flag_plan_cli_keeps_original_bytes_and_refuses_true(tmp_path,monkeypatch):
    file=tmp_path/'plan.json';plan=dict(configuration=dict(chromosomes=[]),rm_nan=False)
    file.write_bytes(json.dumps(plan,indent=3).encode()+b'\n');original=file.read_bytes();seen=[]
    monkeypatch.setattr(phewas_run,'execute_worker',lambda prepared,**kwargs:seen.append(prepared))
    phewas_run.main(['--plan',str(file),'--association-only'])
    phewas_run.main(['--plan',str(file),'--no-single-cross-frame-batches','--association-only'])
    assert seen==[plan,plan] and file.read_bytes()==original
    with pytest.raises(SystemExit) as caught:
        phewas_run.main(['--plan',str(file),'--single-cross-frame-batches','--association-only'])
    assert caught.value.code==2 and len(seen)==2 and file.read_bytes()==original
