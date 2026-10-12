"""Public phenotype-policy and immutable resume contracts, not benchmarks."""
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from fudan_wgs_toolkit import run, phewas_run, prepared_association
from fudan_wgs_toolkit.identity import sample_keys
from fudan_wgs_toolkit.phewas_inputs import PhewasInputs
from test_phewas_public_options import api_inputs


def tables():
    pairs=np.asarray([['0',str(i)] for i in range(1,7)])
    phenotypes=run.NumericCSV(pairs,('continuous','binary'),np.asarray([
        [2.,1.],[np.nan,np.nan],[8.,7.],[100.,0.],[np.nan,1.],[6.,0.]]))
    covariates=run.NumericCSV(pairs,('chosen','unused'),np.asarray([
        [1.,np.nan],[2.,np.nan],[3.,0.],[np.nan,1.],[5.,2.],[6.,3.]]))
    return pairs,phenotypes,covariates


def test_modern_mean_and_binary_fill_share_covariate_mother_population():
    pairs,p,c=tables()
    ids,y,x,names,counts=run.aligned_phenotype(p,c,pairs,'continuous',
        covariate_names=['chosen'],rm_nan=True,family='gaussian')
    mean=(2+8+6)/3
    np.testing.assert_allclose(y,[2,mean,8,mean,6])
    assert ids.tolist()==sample_keys(pairs[[0,1,2,4,5]]).tolist()
    assert counts['available_n']==3 and counts['mother_n']==5
    assert counts['imputed_n']==2 and counts['imputation_mean']==pytest.approx(mean)
    ids,y,x,names,counts=run.aligned_phenotype(p,c,pairs,'binary',
        covariate_names=['chosen'],rm_nan=True,family='binomial')
    np.testing.assert_array_equal(y,[1,0,0,1,0])
    assert counts['available_n']==4 and counts['mother_n']==5
    assert counts['imputation_mean'] is None
    # The covariate-invalid phenotype value 100 is never included in the mean.
    assert len(y)==len(x)==5


def test_available_alignment_and_empty_profile_keep_the_original_cases():
    pairs,p,c=tables()
    _,y,_,_,counts=run.aligned_phenotype(p,c,pairs,'continuous',
        covariate_names=['chosen'],rm_nan=False,family='gaussian')
    np.testing.assert_array_equal(y,[2,8,6])
    assert counts['available_n']==3 and counts['analysis_samples']==3 and counts['imputed_n']==0
    # An intercept-only trait does not require an unused covariate CSV row.
    shorter=run.NumericCSV(c.sample_pairs[1:],c.names,c.values[1:])
    _,y,_,_,counts=run.aligned_phenotype(p,shorter,pairs,'continuous',
        covariate_names=[],rm_nan=True,family='gaussian')
    assert len(y)==6 and counts['mother_n']==6
    assert counts['imputation_mean']==pytest.approx((2+8+100+6)/4)
    with pytest.raises(ValueError,match='[Bb]inary|binomial'):
        run.aligned_phenotype(p,c,pairs,'binary',covariate_names=[],rm_nan=False,family='binomial')


def test_rm_nan_public_defaults_and_numeric_plan_identity(api_inputs):
    assert inspect.signature(run.run_WGS_all_binanry).parameters['impute_missing'].default is True
    for function in (phewas_run.prepare_phewas_run,phewas_run.run_phewas):
        assert inspect.signature(function).parameters['rm_nan'].default is True
    options=dict(api_inputs,rm_nan=True)
    plan=phewas_run.prepare_phewas_run(**options,analyses=('coding',),
        phenotype_covariates={'trait_a':['age']})
    assert plan['rm_nan'] is True
    inputs=PhewasInputs(plan['prepared_inputs_directory'])
    try:
        assert inputs.manifest['rm_nan'] is True and inputs.manifest['trait_options']['rm_nan'] is True
        assert len(inputs.trait(0)['y_raw'])==6
        assert len(inputs.trait(1)['y_raw'])==5
    finally:
        inputs.close()
    before=(Path(plan['output_directory'])/'plan.private.json').read_bytes()
    with pytest.raises(ValueError,match='selection|policy|binding'):
        phewas_run.prepare_phewas_run(**api_inputs,resume=True,analyses=('coding',),
            phenotype_covariates={'trait_a':['age']})
    assert (Path(plan['output_directory'])/'plan.private.json').read_bytes()==before


def test_modern_rm_nan_is_scientific_and_cannot_resume_old_nulls(tmp_path,monkeypatch):
    from test_prepared_association import fixture
    root,p,c,_=fixture(tmp_path)
    calls=[]
    monkeypatch.setattr(prepared_association,'_fit_models',lambda plan,*args: calls.append(plan) or [])
    monkeypatch.setattr(prepared_association,'_dispatch',lambda *args: [])
    options=dict(cpu_threads=1,memory_limit_gib=20,memory_reserve_gib=1,
        phenotype_families='auto',phenotype_covariates={'binary':[]},grm='none',grm_file=None,
        analyses=('individual',),gene_variant_type='SNV',covariance_block_size=64,
        long_mask_threshold=5000,long_mask_rank=32,seed=1729,single_mac_cutoff=20,
        single_group_variants=5000,single_region_size=10,individual_effective_block_size=16,
        device_cache_bytes=0,compact_cache_bytes=0,metadata_cache_bytes=2**20,
        trait_batch_size=2,resume=False,rm_nan=True)
    out=tmp_path/'results'
    prepared_association.run_prepared_association(p,c,root,output_directory=out,gpu_ids=(0,),**options)
    first=json.loads((out/'plan.private.json').read_text())
    assert first['rm_nan'] is True and calls[0]['rm_nan'] is True
    options.update(resume=True,rm_nan=False)
    with pytest.raises(ValueError,match='source/input/schedule'):
        prepared_association.run_prepared_association(p,c,root,output_directory=out,gpu_ids=(0,),**options)
    assert len(calls)==1


@pytest.mark.parametrize('bad',[None,0,1,'False',[],{}])
def test_rm_nan_is_strict_boolean_before_inputs_or_output(tmp_path,bad):
    with pytest.raises(ValueError,match='impute_missing'):
        run.run_WGS_all_binanry('missing.csv','missing.csv','missing',output_directory=tmp_path/'run',impute_missing=bad)
    with pytest.raises(ValueError,match='rm_nan'):
        phewas_run.prepare_phewas_run('missing.csv','missing.csv','missing',
            output_directory=tmp_path/'plan',rm_nan=bad)
    assert not (tmp_path/'run').exists() and not (tmp_path/'plan').exists()


def test_cli_modes_forward_true_default_and_explicit_available(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(run,'run_WGS_all_binanry',lambda **options: calls.append(options) or {'completed':True})
    arguments=['phenotype.csv','covariate.csv','prepared','--output-directory','results']
    run.main(arguments)
    assert calls[-1]['impute_missing'] is True
    run.main(arguments+['--no-impute-missing'])
    assert calls[-1]['impute_missing'] is False
    monkeypatch.setattr(phewas_run,'run_phewas',lambda *args,**options: calls.append(options) or [])
    arguments=['--phenotype-csv','p.csv','--covariate-csv','c.csv',
        '--cache-directory','prepared','--output-directory','results']
    phewas_run.main(arguments)
    assert calls[-1]['rm_nan'] is True
    phewas_run.main(arguments+['--no-rm-nan'])
    assert calls[-1]['rm_nan'] is False
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'rm_nan':False}))
    monkeypatch.setattr(phewas_run,'execute_worker',lambda *args,**kwargs: pytest.fail('changed policy dispatched'))
    with pytest.raises(SystemExit):
        phewas_run.main(['--plan',str(plan),'--rm-nan'])
