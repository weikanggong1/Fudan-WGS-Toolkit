"""One association API for both prepared dataset identity formats."""
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
from fudan_wgs_toolkit import run as module


@pytest.fixture
def cuda(monkeypatch):
    monkeypatch.setattr(module.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(module.torch.cuda, 'device_count', lambda: 8)


def worker_rows(count=1):
    return dict(association_rows=dict(individual=count,coding=count,noncoding=count,ncrna=count))


def test_existing_cache_accepts_same_trait_resource_and_resume_options(tmp_path, monkeypatch, cuda):
    from fudan_wgs_toolkit import phewas_run
    (tmp_path/'cache_dataset.json').write_text('{}')
    calls=[]
    def run(*args, **kwargs):
        calls.append((args,kwargs)); return [worker_rows() for _ in kwargs['gpu_ids']]
    monkeypatch.setattr(phewas_run,'run_phewas',run)
    result=module.run_WGS_all('phenotypes.csv','covariates.csv',tmp_path,output_directory=tmp_path/'out',
        device=8,chromosomes=[1,21],memory_limit_gib=64,memory_reserve_gib=2,
        phenotype_families={'trait_a':'gaussian'},phenotype_covariates={'trait_a':['age']},resume=True)
    assert result['completed'] and result['eligible_association_tests']==32
    assert calls[0][0]==('phenotypes.csv','covariates.csv',tmp_path)
    requested=calls[0][1]
    assert requested['gpu_ids']==tuple(range(8)) and requested['chromosomes']==['1','21']
    assert requested['memory_limit_gib']==64 and requested['memory_reserve_gib']==2
    assert requested['phenotype_families']=={'trait_a':'gaussian'}
    assert requested['phenotype_covariates']=={'trait_a':['age']} and requested['resume'] is True
    assert requested['gene_variant_type']=='SNV' and requested['grm']=='sparse'
    assert (tmp_path/'out/report.private.json').is_file()


def test_defaults_are_auto_full_genome_one_gpu_with_reserved_live_memory(tmp_path, monkeypatch, cuda):
    from fudan_wgs_toolkit import phewas_run
    (tmp_path/'cache_dataset.json').write_text('{}')
    calls=[]
    monkeypatch.setattr(phewas_run,'run_phewas',lambda *a,**k: calls.append(k) or [worker_rows()])
    module.run_WGS_all('p.csv','c.csv',tmp_path,output_directory=tmp_path/'out')
    options=calls[0]
    assert options['memory_limit_gib'] is None and options['memory_reserve_gib']==1
    assert options['phenotype_families']=='auto' and options['phenotype_covariates'] is None
    assert options['chromosomes'] is None and options['gpu_ids']==(0,) and options['resume'] is False
    signature=inspect.signature(module.run_WGS_all)
    assert signature.parameters['gene_variant_type'].default=='SNV'


@pytest.mark.parametrize('options,expected', [({}, 'none'),
    ({'continuous_transform':'paper'}, 'paper'),
    ({'continuous_transform':'none'}, 'none'),
    ({'continuous_transform':'rint'}, 'rint'), ({'transform':'rint'}, 'rint')])
def test_modern_cache_preserves_default_and_forwards_explicit_transformation(tmp_path, monkeypatch, cuda, options, expected):
    from fudan_wgs_toolkit import prepared_association
    calls=[]
    monkeypatch.setattr(prepared_association, 'run_prepared_association',
        lambda *a,**k: calls.append(k) or [worker_rows()])
    module.run_WGS_all('p.csv','c.csv',tmp_path,output_directory=tmp_path/'out',**options)
    assert calls[0]['transform']==expected


@pytest.mark.parametrize('value,expected',[('all',None),(['all'],None),([1,21],['1','21']),(['21'],['21']),(['chr21'],['21'])])
def test_chromosome_selection(value,expected):
    assert module._normalise_chromosomes(value)==expected


@pytest.mark.parametrize('value',[[],[0],[23],[1,1],[True],[1.0],['all',21]])
def test_invalid_chromosome_selection(value):
    with pytest.raises(ValueError): module._normalise_chromosomes(value)


@pytest.mark.parametrize('value,expected',[(1,(0,)),(3,(0,1,2)),(['cuda:3'],(3,)),(['cuda:1','cuda:0'],(1,0)),('cuda:0',(0,))])
def test_gpu_selection(value,expected,cuda):
    assert module._normalise_devices(value)==expected


@pytest.mark.parametrize('value',[0,-1,True,[],['cpu'],['cuda:8'],['cuda:0','cuda:0'],[0]])
def test_invalid_gpu_selection(value,cuda):
    with pytest.raises(ValueError): module._normalise_devices(value)


def test_family_auto_is_column_wise_and_explicit_overrides_remain_possible():
    y=np.asarray([[0.,2.,np.nan],[1.,3.,np.nan],[np.nan,0.,np.nan]])
    assert module._family_selection('auto',('a','b','c'),y)=={'a':'binomial','b':'gaussian','c':'gaussian'}
    assert module._family_selection({'a':'gaussian'},('a','b','c'),y)['a']=='gaussian'
    with pytest.raises(ValueError):module._family_selection({'b':'binomial'},('a','b','c'),y)


def test_two_dataset_roots_cannot_silently_change_identity(tmp_path,cuda):
    (tmp_path/'cache_dataset.json').write_text('{}');(tmp_path/'dataset.json').write_text('{}')
    with pytest.raises(ValueError,match='ambiguous'):
        module.run_WGS_all('p.csv','c.csv',tmp_path,output_directory=tmp_path/'out')


@pytest.mark.parametrize('kwargs',[{'memory_limit_gib':-1},{'memory_limit_gib':float('inf')},
    {'memory_limit_gib':1,'memory_reserve_gib':1},{'cpu_threads':0},{'resume':1},
    {'grm':'none','grm_file':'grm.npz'},{'analyses':('other',)},{'trait_batch_size':0}])
def test_invalid_options_fail_before_work(tmp_path,kwargs,cuda):
    with pytest.raises(ValueError):module.run_WGS_all('p.csv','c.csv',tmp_path,output_directory=tmp_path/'out',**kwargs)


def test_cli_accepts_gpu_count_all_chromosomes_and_family_auto(monkeypatch,capsys):
    calls=[]
    monkeypatch.setattr(module,'run_WGS_all',lambda **kw:calls.append(kw) or dict(completed=True))
    module.main(['p.csv','c.csv','prepared','--output-directory','out','--device','2','--chromosomes','all','--grm','none','--resume'])
    assert calls[0]['device']==2 and calls[0]['phenotype_families']=='auto'
    assert calls[0]['resume'] is True and calls[0]['grm']=='none'
    assert json.loads(capsys.readouterr().out)['completed'] is True
