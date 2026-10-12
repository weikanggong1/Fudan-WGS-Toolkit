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
    result=module.run_WGS_all_binanry('phenotypes.csv','covariates.csv',tmp_path,output_directory=tmp_path/'out',
        device=8,chromosomes=[1,21],memory_limit_gib=64,memory_reserve_gib=2,
        impute_missing=False,phenotype_covariates={'trait_a':['age']},resume=True)
    assert result['completed'] and result['eligible_association_tests']==32
    assert calls[0][0]==('phenotypes.csv','covariates.csv',tmp_path)
    requested=calls[0][1]
    assert requested['gpu_ids']==tuple(range(8)) and requested['chromosomes']==['1','21']
    assert requested['memory_limit_gib']==64 and requested['memory_reserve_gib']==2
    assert requested['binary_only'] is True and requested['rm_nan'] is False
    assert requested['association_partition_strategy']=='whole_chromosomes'
    assert requested['phenotype_covariates']=={'trait_a':['age']} and requested['resume'] is True
    assert requested['gene_variant_type']=='SNV' and requested['grm']=='sparse'
    assert (tmp_path/'out/report.private.json').is_file()


def test_defaults_are_auto_full_genome_one_gpu_with_reserved_live_memory(tmp_path, monkeypatch, cuda):
    from fudan_wgs_toolkit import phewas_run
    (tmp_path/'cache_dataset.json').write_text('{}')
    calls=[]
    monkeypatch.setattr(phewas_run,'run_phewas',lambda *a,**k: calls.append(k) or [worker_rows()])
    module.run_WGS_all_binanry('p.csv','c.csv',tmp_path,output_directory=tmp_path/'out')
    options=calls[0]
    assert options['memory_limit_gib'] is None and options['memory_reserve_gib']==1
    assert options['binary_only'] is True and options['phenotype_covariates'] is None
    assert options['rm_nan'] is True and options['continuous_transform']=='none'
    assert options['association_partition_strategy']=='whole_chromosomes'
    assert options['chromosomes'] is None and options['gpu_ids']==(0,) and options['resume'] is False
    signature=inspect.signature(module.run_WGS_all_binanry)
    assert signature.parameters['gene_variant_type'].default=='SNV'


def test_modern_cache_routes_binary_only_and_fixed_null_fit(tmp_path, monkeypatch, cuda):
    from fudan_wgs_toolkit import prepared_association
    calls = []
    monkeypatch.setattr(prepared_association, "run_prepared_association",
        lambda *a, **k: calls.append(k) or [worker_rows()])
    module.run_WGS_all_binanry("p.csv", "c.csv", tmp_path, output_directory=tmp_path/"out")
    assert calls[0]["transform"] == "none" and calls[0]["null_fit_mode"] == "fp64"
    assert calls[0]["binary_only"] is True and calls[0]["rm_nan"] is True
    assert calls[0]["association_partition_strategy"] == "whole_chromosomes"


@pytest.mark.parametrize("option", ["phenotype_families", "continuous_transform", "transform",
                                  "rm_nan", "null_fit_mode", "long_mask_method", "gpu_ids"])
def test_generic_public_options_are_not_accepted(tmp_path, option):
    with pytest.raises(TypeError, match="unexpected keyword"):
        module.run_WGS_all_binanry("missing.csv", "missing.csv", "missing",
            output_directory=tmp_path/"out", **{option: "removed"})
    assert not (tmp_path/"out").exists()


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


def test_binary_validator_preserves_nan_and_rejects_continuous_values():
    pairs = np.asarray([["0", str(index)] for index in range(3)])
    values = np.asarray([[0., 1.], [1., 0.], [np.nan, np.nan]])
    table = module.NumericCSV(pairs, ("a", "b"), values)
    assert module._validate_binary_phenotypes(table) == {"a": "binomial", "b": "binomial"}
    assert np.isnan(table.values[-1]).all()
    bad = values.copy()
    bad[0, 1] = 2.
    with pytest.raises(ValueError, match="only zero, one or missing"):
        module._validate_binary_phenotypes(module.NumericCSV(pairs, ("a", "b"), bad))
    with pytest.raises(ValueError, match="at least two"):
        module._validate_binary_phenotypes(module.NumericCSV(pairs, ("a",), values[:, :1]))


def test_two_dataset_roots_cannot_silently_change_identity(tmp_path,cuda):
    (tmp_path/'cache_dataset.json').write_text('{}');(tmp_path/'dataset.json').write_text('{}')
    with pytest.raises(ValueError,match='ambiguous'):
        module.run_WGS_all_binanry('p.csv','c.csv',tmp_path,output_directory=tmp_path/'out')


@pytest.mark.parametrize('kwargs',[{'memory_limit_gib':-1},{'memory_limit_gib':float('inf')},
    {'memory_limit_gib':1,'memory_reserve_gib':1},{'cpu_threads':0},{'resume':1},
    {'grm':'none','grm_file':'grm.npz'},{'analyses':('other',)},{'trait_batch_size':0}])
def test_invalid_options_fail_before_work(tmp_path,kwargs,cuda):
    with pytest.raises(ValueError):module.run_WGS_all_binanry('p.csv','c.csv',tmp_path,output_directory=tmp_path/'out',**kwargs)


def test_cli_accepts_gpu_count_all_chromosomes_and_binary_missing_policy(monkeypatch,capsys):
    calls=[]
    monkeypatch.setattr(module,'run_WGS_all_binanry',lambda **kw:calls.append(kw) or dict(completed=True))
    module.main(['p.csv','c.csv','prepared','--output-directory','out','--device','2','--chromosomes','all','--grm','none','--resume'])
    assert calls[0]['device']==2 and calls[0]['impute_missing'] is True
    assert 'phenotype_families' not in calls[0]
    assert calls[0]['resume'] is True and calls[0]['grm']=='none'
    assert json.loads(capsys.readouterr().out)['completed'] is True
