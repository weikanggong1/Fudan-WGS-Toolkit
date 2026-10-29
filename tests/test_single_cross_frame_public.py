"""Public policy forwarding and immutable plans; no CUDA performance claims."""
import inspect
import json

import pytest

from fudan_wgs_toolkit import continuous_run, phewas_run, prepared_association, run
from fudan_wgs_toolkit.phewas_runtime import runtime
from test_phewas_public_options import api_inputs


@pytest.mark.parametrize('value',[False,True])
def test_population_builder_and_actual_plan_bind_explicit_policy(api_inputs,monkeypatch,value):
    monkeypatch.setattr(phewas_run,'_implementation_sha256',lambda:'fixed-public-policy-source')
    config=phewas_run.build_configuration(api_inputs['cache_directory'],analyses=('individual',),
        single_cross_frame_batches=value)
    plan=phewas_run.prepare_phewas_run(**api_inputs,analyses=('individual',),single_cross_frame_batches=value)
    assert config.get('single_cross_frame_batches',False) is value
    assert plan['configuration'].get('single_cross_frame_batches',False) is value
    assert json.loads((api_inputs['output_directory']/'plan.private.json').read_text())['configuration']==plan['configuration']
    resumed=phewas_run.prepare_phewas_run(**api_inputs,analyses=('individual',),single_cross_frame_batches=value,resume=True)
    assert resumed['configuration']==plan['configuration']
    with pytest.raises(ValueError,match='resume plan scientific/source bindings differ'):
        phewas_run.prepare_phewas_run(**api_inputs,analyses=('individual',),single_cross_frame_batches=not value,resume=True)


def test_public_and_builder_default_remains_legacy_false():
    for entry in (run.run_WGS_all_binanry,continuous_run.run_WGS_continuous,
                  phewas_run.build_configuration,phewas_run.prepare_phewas_run,phewas_run.run_phewas):
        assert inspect.signature(entry).parameters['single_cross_frame_batches'].default is False


@pytest.mark.parametrize('value',[None,0,1,'true',[],{}])
@pytest.mark.parametrize('entry',['binary','continuous','builder','population_plan','prepared'])
def test_invalid_policy_rejected_before_inputs_or_cuda(tmp_path,monkeypatch,value,entry):
    def no_inputs(*args,**kwargs):pytest.fail('invalid opt-in reached scientific inputs')
    monkeypatch.setattr(phewas_run,'_dataset',no_inputs)
    monkeypatch.setattr(run,'_dataset',no_inputs)
    monkeypatch.setattr(run,'_normalise_devices',no_inputs)
    with pytest.raises(ValueError,match='single_cross_frame_batches'):
        if entry=='builder':phewas_run.build_configuration('missing',single_cross_frame_batches=value)
        elif entry=='population_plan':phewas_run.prepare_phewas_run('p','c','g',output_directory=tmp_path/'out',single_cross_frame_batches=value)
        elif entry=='prepared':prepared_association.run_prepared_association('p','c','g',output_directory=tmp_path/'out',gpu_ids=(0,),single_cross_frame_batches=value)
        else:
            function=run.run_WGS_all_binanry if entry=='binary' else continuous_run.run_WGS_continuous
            function('p','c','g',output_directory=tmp_path/'out',single_cross_frame_batches=value)
    assert not (tmp_path/'out').exists()


class ExecutorReached(Exception):pass


@pytest.mark.parametrize('family',['binary','continuous'])
@pytest.mark.parametrize('population',[False,True])
@pytest.mark.parametrize('value',[False,True])
def test_public_policy_reaches_both_cache_format_executors(tmp_path,monkeypatch,family,population,value):
    prepared=tmp_path/'prepared';prepared.mkdir()
    (prepared/('cache_dataset.json' if population else 'dataset.json')).write_text('{}')
    monkeypatch.setattr(run,'_normalise_devices',lambda *args:(0,))
    received=[]
    def stop(*args,**kwargs):received.append(kwargs);raise ExecutorReached
    target=phewas_run if population else prepared_association
    monkeypatch.setattr(target,'run_phewas' if population else 'run_prepared_association',stop)
    function=run.run_WGS_all_binanry if family=='binary' else continuous_run.run_WGS_continuous
    with pytest.raises(ExecutorReached):
        function('p','c',prepared,output_directory=tmp_path/'out',grm='none',
            single_cross_frame_batches=value)
    assert len(received)==1 and received[0]['single_cross_frame_batches'] is value


@pytest.mark.parametrize('value',[False,True])
def test_modern_public_plan_reaches_shared_runtime_and_rejects_changed_resume(tmp_path,monkeypatch,value):
    from test_continuous_prepared import continuous_fixture,cpu_single_control,call
    root,phenotype,covariate,_=continuous_fixture(tmp_path)
    cpu_single_control(monkeypatch)
    original=runtime.run_batched_configuration;seen=[]
    def observe(configuration,**kwargs):
        seen.append(configuration.get('single_cross_frame_batches',False))
        return original(configuration,**kwargs)
    monkeypatch.setattr(runtime,'run_batched_configuration',observe)
    output=tmp_path/'out'
    report=call(continuous_run.run_WGS_continuous,root,phenotype,covariate,output,
        rm_nan=False,single_cross_frame_batches=value)
    assert report['completed'] and seen==[value]
    saved=json.loads((output/'plan.private.json').read_text())
    assert saved['configuration'].get('single_cross_frame_batches',False) is value
    with pytest.raises(ValueError,match='resume prepared association source/input/schedule binding differs'):
        call(continuous_run.run_WGS_continuous,root,phenotype,covariate,output,
            rm_nan=False,single_cross_frame_batches=not value,resume=True)
    assert seen==[value]


@pytest.mark.parametrize('module,name',[(run,'run_WGS_all_binanry'),(continuous_run,'run_WGS_continuous')])
@pytest.mark.parametrize('option,value',[(None,False),('--single-cross-frame-batches',True),('--no-single-cross-frame-batches',False)])
def test_public_cli_boolean_switch_reaches_public_entry(monkeypatch,module,name,option,value):
    seen=[]
    monkeypatch.setattr(module,name,lambda **kwargs:seen.append(kwargs) or {})
    argv=['p','c','g','--output-directory','out']+([] if option is None else [option])
    module.main(argv)
    assert seen[0]['single_cross_frame_batches'] is value


@pytest.mark.parametrize('option,value',[(None,False),('--single-cross-frame-batches',True),('--no-single-cross-frame-batches',False)])
def test_population_cli_new_plan_policy_is_explicit(monkeypatch,option,value):
    seen=[]
    monkeypatch.setattr(phewas_run,'run_phewas',lambda *args,**kwargs:seen.append(kwargs) or [])
    argv=['--phenotype-csv','p','--covariate-csv','c','--cache-directory','g','--output-directory','out']
    phewas_run.main(argv+([] if option is None else [option]))
    assert seen[0]['single_cross_frame_batches'] is value


@pytest.mark.parametrize('stored',[False,True])
def test_population_plan_cli_preserves_and_never_overrides_policy(tmp_path,monkeypatch,stored):
    file=tmp_path/'plan.json';plan=dict(configuration=dict(single_cross_frame_batches=stored),rm_nan=False)
    file.write_text(json.dumps(plan));seen=[]
    monkeypatch.setattr(phewas_run,'execute_worker',lambda prepared,**kwargs:seen.append(prepared))
    phewas_run.main(['--plan',str(file),'--association-only'])
    same='--single-cross-frame-batches' if stored else '--no-single-cross-frame-batches'
    phewas_run.main(['--plan',str(file),same,'--association-only'])
    assert seen==[plan,plan] and json.loads(file.read_text())==plan
    different='--no-single-cross-frame-batches' if stored else '--single-cross-frame-batches'
    with pytest.raises(SystemExit) as caught:phewas_run.main(['--plan',str(file),different,'--association-only'])
    assert caught.value.code==2 and len(seen)==2


@pytest.mark.parametrize('effective_bound',[128,256,512,1024])
def test_actual_public_plan_binds_effective_width_and_rejects_changed_resume(api_inputs,monkeypatch,effective_bound):
    monkeypatch.setattr(phewas_run,'_implementation_sha256',lambda:'fixed-effective-width-source')
    options=dict(analyses=('individual',),single_cross_frame_batches=True,
                 individual_effective_block_size=effective_bound)
    plan=phewas_run.prepare_phewas_run(**api_inputs,**options)
    assert plan['configuration']['individual_genotype_block_size']==1024
    assert plan['configuration']['individual_effective_block_size']==effective_bound
    plan_file=api_inputs['output_directory']/'plan.private.json'
    before=plan_file.read_bytes()
    resumed=phewas_run.prepare_phewas_run(**api_inputs,resume=True,**options)
    assert resumed['configuration']==plan['configuration']
    before=plan_file.read_bytes()
    changed=dict(options,individual_effective_block_size=256 if effective_bound==128 else 128)
    with pytest.raises(ValueError,match='resume plan scientific/source bindings differ'):
        phewas_run.prepare_phewas_run(**api_inputs,resume=True,**changed)
    assert plan_file.read_bytes()==before
