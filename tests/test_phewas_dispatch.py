"""Scheduling coverage contracts; these fixtures are not scientific benchmarks."""
import json
import numpy as np
import pytest
from fudan_wgs_toolkit.phewas_run import build_configuration

def test_all_trait_schedule_restores_published_ptv_categories_from_legacy_catalog(tmp_path,monkeypatch):
    catalog=tmp_path/'genes.json'
    catalog.write_text(json.dumps([
        dict(kind='coding',gene_name='GENE_EXAMPLE',start=1,end=10,
             category='all_categories',include_ptv=False),
        dict(kind='noncoding',gene_name='GENE_EXAMPLE',category='all_categories'),
        dict(kind='ncrna',gene_name='GENE_RNA',start=2,end=9)]))
    entry=dict(name='1',gene_catalog=str(catalog),container_directory='cache',metadata_directory='metadata',
               _maximum_position=10)
    monkeypatch.setattr('fudan_wgs_toolkit.phewas_run._dataset',lambda *args:
        (tmp_path,tmp_path/'cache_dataset.json',dict(annotation_catalog={}),[entry],np.arange(4)))
    config=build_configuration(tmp_path)
    jobs=config['chromosomes'][0]['jobs']
    assert [j['kind'] for j in jobs]==['coding','noncoding','ncrna','individual']
    assert jobs[0]['arguments']['category']=='all_categories_incl_ptv'
    assert jobs[0]['arguments']['include_ptv'] is True
    assert config['analysis_options']['memory_limit_gib'] is None

def test_portable_promoter_json_and_existing_text_have_the_same_ordered_intervals(tmp_path):
    from fudan_wgs_toolkit.cli import _read_promoter_intervals
    rows=[['1',10,20],['1',25,40],['2',5,15]]
    a,b=tmp_path/'promoters.json',tmp_path/'promoters.bed'
    a.write_text(json.dumps(rows))
    b.write_text('chromosome start end\n'+'\n'.join(' '.join(map(str,r)) for r in rows))
    assert _read_promoter_intervals(a)==_read_promoter_intervals(b)==[tuple(r) for r in rows]
    a.write_text(json.dumps([['1',20,10]]))
    with pytest.raises(ValueError,match='inclusive integer coordinates'):
        _read_promoter_intervals(a)

def test_model_barrier_rejects_mixed_prepared_inputs_even_with_valid_receipt_hash(tmp_path):
    from fudan_wgs_toolkit.phewas_run import _all_models
    from fudan_wgs_toolkit.phewas_storage import digest_file
    inputs=tmp_path/'inputs';inputs.mkdir()
    manifest=inputs/'manifest.private.json';manifest.write_text('{}')
    model=tmp_path/'trait_0000';model.mkdir()
    fit=dict(input_manifest_sha256=digest_file(manifest),kinship_sha256='kinship',
             source_implementation_sha256='source',cohort_sha256='cohort',continuous_transform='paper')
    metadata=model/'model.private.json';metadata.write_text(json.dumps(dict(fit=fit)))
    receipt=tmp_path/'fit_worker_0.complete.private.json'
    def bind_receipt():
        receipt.write_text(json.dumps(dict(errors=[],models=[dict(trait_index=0,path=str(model),
            sha256=digest_file(metadata))])))
    bind_receipt()
    plan=dict(output_directory=str(tmp_path),prepared_inputs_directory=str(inputs),trait_count=1,
        prepared_inputs_manifest_sha256=digest_file(manifest),kinship_sha256='kinship',
        source_implementation_sha256='source',cohort_sha256='cohort')
    assert len(_all_models(plan,1))==1
    fit['input_manifest_sha256']='another-input'
    metadata.write_text(json.dumps(dict(fit=fit)));bind_receipt()
    with pytest.raises(ValueError,match='do not share'):_all_models(plan,1)
