"""Gene serialization/durability contracts, not scientific speed benchmarks."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import pytest

import fudan_wgs_toolkit.phewas_trait_csv as module
from fudan_wgs_toolkit.phewas_trait_csv import PerTraitCheckpointCSVWriter

BINDING=dict(source_implementation_sha256='fixture-source',input_manifest_sha256='fixture-input',
             configuration_sha256='fixture-ordered-schedule',models={'0':'fixture0','1':'fixture1'})

class _OriginalGeneWriter(PerTraitCheckpointCSVWriter):
    """Use the original trait-prefixed DictWriter serialization route."""
    def _append_member(self,info,start,rows,*,trait_index=None):
        if trait_index is not None:
            rows=(dict(trait_index=trait_index,**source) for source in rows)
        return super()._append_member(info,start,rows)

def row(value,**extra):
    return dict(Chr='21',**{'Gene name':'fixture, quote" newline\nα','Category':'fixture','#SNV':7,'cMAC':21,'STAAR-O':value},
                _chunk=0,_common=False,_base_group=2,missing=None,**extra)

def scientific_receipts(output):
    return [{k:v for k,v in r.items() if k not in ('observed_at',)} for r in output['receipts']]

@pytest.mark.parametrize('kind',['coding','noncoding','ncrna'])
def test_native_gene_members_and_checkpoint_receipts_match_original_bytes(tmp_path,kind):
    values=[row(-0.,flag=True),row(5e-324,flag=False),row(1.23456789012345e-100,flag=True)]
    binary=[row(.2,spa_selected=True,spa_failed=False),row(.3,spa_selected=True,spa_failed=True)]
    reordered=[{key:value for key,value in reversed(tuple(binary[0].items()))}]
    records=[];reports=[];roots=[tmp_path/'old',tmp_path/'new']
    for cls,root in zip((_OriginalGeneWriter,PerTraitCheckpointCSVWriter),roots):
        writer=cls(root,binding=BINDING,max_open_streams=1)
        records.append(writer.write_tile(kind,'21',[(0,values),(1,dict(common=binary,rare=reordered+binary))],job_index=0))
        writer.abort()
        resumed=cls(root,binding=BINDING,resume=True,max_open_streams=1)
        assert resumed.checkpoint_tile(kind,'21',0,[0,1])==records[-1]
        resumed.write_tile(kind,'21',[(0,[row(.4,flag=False)]),(1,[])],job_index=1)
        reports.append(resumed.finish())
    assert records[0]==records[1]
    assert scientific_receipts(reports[0])==scientific_receipts(reports[1])
    for receipt in reports[0]['receipts']:
        relative=receipt['csv_file'];old=(roots[0]/relative).read_bytes();new=(roots[1]/relative).read_bytes()
        assert old==new and hashlib.sha256(new).hexdigest()==receipt['sha256']
        assert gzip.decompress(old)==gzip.decompress(new)
    assert reports[1]['computed_blocks']==1 and reports[1]['reused_blocks']==1

def test_gene_checked_scalar_stream_keeps_original_row_owners(tmp_path,monkeypatch):
    originals=[row(.25),row(.5),row(.75)];seen=[];actual=module._write_bound_trait_rows
    def observe(writer,rows,trait,columns):
        def stream():
            for source in rows:
                seen.append(source)
                yield source
        return actual(writer,stream(),trait,columns)
    monkeypatch.setattr(module,'_write_bound_trait_rows',observe)
    writer=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    writer.write_tile('coding','21',[(0,originals)],job_index=0)
    result=writer.finish()
    assert len(seen)==3 and all(a is b for a,b in zip(seen,originals))
    assert all('trait_index' not in source for source in originals)
    assert result['rows']['coding']==3

def test_gene_serialization_failure_archives_tail_without_committing_then_resumes(tmp_path):
    class Failure:
        def __str__(self):raise ValueError('injected scalar serialization failure')
    writer=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,max_open_streams=1)
    with pytest.raises(ValueError,match='scalar serialization'):
        writer.write_tile('coding','21',[(0,[row(.1)]),(1,[row(.2),row(Failure())])],job_index=0)
    assert writer.db.execute('SELECT COUNT(*) FROM blocks').fetchone()[0]==0
    assert not writer._dirty_streams
    assert len(list((tmp_path/'.trait_checkpoints/uncommitted').rglob('tail.gz')))==2
    assert all(path.stat().st_size==0 for path in tmp_path.rglob('*.partial'))
    writer.abort()
    resumed=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,resume=True)
    assert resumed.checkpoint_tile('coding','21',0,[0,1]) is None
    resumed.write_tile('coding','21',[(0,[row(.3)]),(1,[row(.4)])],job_index=0)
    output=resumed.finish()
    assert output['rows']['coding']==2
    for receipt in output['receipts']:
        with gzip.open(tmp_path/receipt['csv_file'],'rt',newline='') as stream:
            actual=list(csv.DictReader(stream))
        assert len(actual)==1 and actual[0]['STAAR-O'] in ('0.3','0.4')

def test_gene_dictionary_subclass_keeps_original_plain_copy_semantics(tmp_path):
    class AlteredGet(dict):
        def get(self,key,default=None):return 'overridden get'
    values=[AlteredGet(row(.2,spa_selected=False,spa_failed=False))]
    outputs=[]
    for cls,name in ((_OriginalGeneWriter,'old'),(PerTraitCheckpointCSVWriter,'new')):
        writer=cls(tmp_path/name,binding=BINDING)
        output=writer.write_tile('noncoding','21',[(0,values)],job_index=0)
        assert output['binary_spa_selected']==output['binary_spa_failed']==0
        reports=writer.finish();outputs.append((tmp_path/name/reports['receipts'][0]['csv_file']).read_bytes())
    assert outputs[0]==outputs[1]

