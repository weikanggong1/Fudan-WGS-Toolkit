"""Immutable input identities and constant full-hash work across binary fits."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fudan_wgs_toolkit import prepared_association as prepared
from fudan_wgs_toolkit import run
from fudan_wgs_toolkit.phewas_cache import _stream_bindings


def binding(tmp_path):
    root=tmp_path/'prepared'; root.mkdir()
    manifest=root/'dataset.json'; manifest.write_text('{"schema_version":2}')
    manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
    (root/'COMPLETE').write_text(manifest_sha+'\n')
    phenotype,covariate=tmp_path/'phenotypes.csv',tmp_path/'covariates.csv'
    phenotype.write_text('FID,IID,disease_a,disease_b\n0,1,0,1\n')
    covariate.write_text('FID,IID,age\n0,1,30\n')
    container=root/'container'; container.mkdir()
    for name in ('data.bin','headers.bin','counts.bin'):
        (container/name).write_bytes(b'01')
    files={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
           for path in (phenotype,covariate,manifest)}
    return dict(prepared_directory=str(root),dataset_sha256=manifest_sha,files=files,
        genotype_streams=_stream_bindings([dict(container_directory=str(container))])),phenotype


def test_one_hundred_stat_checks_do_not_rehash_large_csv_and_final_hash_runs(tmp_path,monkeypatch):
    source,phenotype=binding(tmp_path)
    original=prepared.digest_file
    hashes=[]
    def digest(path):
        hashes.append(str(path)); return original(path)
    monkeypatch.setattr(prepared,'digest_file',digest)
    verifier=prepared._BinaryInputVerification(source)
    for _ in range(100):
        verifier.check()
    assert hashes.count(str(phenotype))==1
    verifier.finish()
    assert hashes.count(str(phenotype))==2
    assert verifier.summary()['full_hash_passes']==2
    assert verifier.summary()['stat_check_passes']==100
    assert verifier.summary()['final_full_hash_verified'] is True


@pytest.mark.parametrize('change',['same_size_content','same_bytes_replacement','stream_replacement','completion_replacement'])
def test_same_length_edits_or_inode_replacements_fail_closed(tmp_path,change):
    source,phenotype=binding(tmp_path)
    verifier=prepared._BinaryInputVerification(source)
    target=phenotype
    if change=='stream_replacement':
        target=Path(next(iter(source['genotype_streams'])))
    elif change=='completion_replacement':
        target=Path(source['prepared_directory'])/'COMPLETE'
    original=target.stat(); payload=target.read_bytes()
    if change=='same_size_content':
        target.write_bytes(payload.replace(b'disease_a',b'disease_x'))
        assert target.stat().st_size==original.st_size
    else:
        replacement=target.with_name(target.name+'.new')
        replacement.write_bytes(payload)
        os.utime(replacement,ns=(original.st_atime_ns,original.st_mtime_ns))
        replacement.replace(target)
        assert target.stat().st_ino!=original.st_ino
    os.utime(target,ns=(original.st_atime_ns,original.st_mtime_ns))
    with pytest.raises(ValueError,match='changed'):
        verifier.check()


def test_source_change_between_matrix_read_and_initial_hash_is_rejected(tmp_path):
    source,phenotype=binding(tmp_path)
    loaded={str(phenotype):prepared._input_file_identity(phenotype)}
    phenotype.write_bytes(phenotype.read_bytes())
    with pytest.raises(ValueError,match='matrices were read'):
        prepared._BinaryInputVerification(source,loaded_input_stats=loaded)


@pytest.mark.parametrize('boundary',['starting','final'])
def test_full_hash_pass_brackets_same_bytes_rewrite_with_stat_proof(tmp_path,monkeypatch,boundary):
    source,phenotype=binding(tmp_path)
    original=prepared.digest_file
    verifier=None
    if boundary=='final':
        verifier=prepared._BinaryInputVerification(source)
    def digest(path):
        value=original(path)
        if str(path)==str(phenotype):
            # Even a same-byte rewrite invalidates the identity around hashing.
            phenotype.write_bytes(phenotype.read_bytes())
        return value
    monkeypatch.setattr(prepared,'digest_file',digest)
    with pytest.raises(ValueError,match=f'during the {boundary}'):
        if boundary=='starting':
            prepared._BinaryInputVerification(source)
        else:
            verifier.finish()


def test_final_checksum_is_independent_of_quick_stat_information(tmp_path,monkeypatch):
    source,phenotype=binding(tmp_path)
    verifier=prepared._BinaryInputVerification(source)
    baseline={path:prepared._input_file_identity(path) for path in verifier.paths}
    phenotype.write_bytes(phenotype.read_bytes().replace(b'disease_a',b'disease_x'))
    # Simulate stale filesystem metadata: the final hash must still reject it.
    monkeypatch.setattr(prepared,'_input_file_identity',lambda path:baseline[str(path)])
    with pytest.raises(ValueError,match='checksum changed'):
        verifier.finish()


def test_real_fit_loop_one_hundred_traits_has_two_boundary_hashes(tmp_path,monkeypatch):
    source,phenotype=binding(tmp_path)
    output=tmp_path/'output'; (output/'models').mkdir(parents=True)
    pairs=np.asarray([['0',str(i)] for i in range(1,7)])
    names=tuple(f'disease_{i}' for i in range(100))
    values=np.tile(np.asarray([0.,1.,0.,1.,0.,1.])[:,None],(1,100))
    phenotypes=run.NumericCSV(pairs,names,values)
    covariates=run.NumericCSV(pairs,('age',),np.arange(6,dtype=float)[:,None])
    plan=dict(output_directory=str(output),input_binding=source,dataset={},binary_only=True,
        families=dict.fromkeys(names,'binomial'),covariate_selection={},rm_nan=False,
        scientific_binding_sha256='scientific',source_implementation_sha256='source',transform='none')
    fit_calls=[]
    def fit(y,ids,x,**options):
        fit_calls.append((len(y),options['fit_structure_cache']))
        return SimpleNamespace(model=SimpleNamespace(converged=True,n=len(y)),
            spa_model=None,sample_indices=np.arange(len(y)),metadata={})
    def store(model,target,*,sample_rows,fit_metadata,spa_model):
        target.mkdir()
        path=target/'model.private.json'
        path.write_text(json.dumps(dict(family='binomial',n=model.n,fit=fit_metadata)))
        return dict(path=str(target),n=model.n,family='binomial',
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(prepared,'prepare_phewas_model',fit)
    monkeypatch.setattr(prepared,'save_model_store',store)
    monkeypatch.setattr(prepared.gc,'collect',lambda:None)
    original=prepared.digest_file
    hashes=[]
    def digest(path):
        hashes.append(str(path)); return original(path)
    monkeypatch.setattr(prepared,'digest_file',digest)
    models=prepared._fit_models(plan,phenotypes,covariates,pairs,None,{'resume':False},'cpu')
    assert len(models)==len(fit_calls)==100
    assert len({id(cache) for _,cache in fit_calls})==1
    assert hashes.count(str(phenotype))==2
    summary=json.loads((output/'model_fit.runtime.private.json').read_text())
    assert summary['complete'] is True
    assert summary['input_verification']['full_hash_passes']==2
    assert summary['input_verification']['stat_check_passes']==200
    assert summary['alignment_cache']['design_builds']==1
    assert summary['fit_structure_cache']['active_leases']==0
