"""CPU orchestration/checkpoint equivalence, not TF32 or real-data benchmarks."""
from contextlib import contextmanager
from types import SimpleNamespace
import csv
import gzip
import hashlib
import json

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.cache_runtime import fast_container, sparse_codec_fast, store
from fudan_wgs_toolkit.phewas_runtime import runtime
from fudan_wgs_toolkit.phewas_runtime.metadata import SharedMetadataReader
from fudan_wgs_toolkit.phewas_runtime.shared_state import SharedStateBroker
from fudan_wgs_toolkit.phewas_trait_csv import PerTraitCheckpointCSVWriter
from fudan_wgs_toolkit.pipeline import AnalysisOptions
from test_phewas_batched_runtime import Repository, descriptor, model_for, cpu_product_oracle
from test_phewas_shared_state import MetadataReader


def states_fixture(n=16,m=37,*,empty=False):
    states=np.zeros((m,n),dtype=np.uint8)
    if not empty:
        for variant in range(1,m,4):
            states[variant,(np.arange(4)+variant//4)%n]=1
    return states


def make_small_frame_cache(path,states,frame_width=4):
    m,n=states.shape;samples=np.arange(n,dtype=np.int64);binding={'fixture':'cross-frame-no-scientific-data'}
    writer=store.Writer(path,binding,samples,m,source_bytes=100*2**20,chunk=frame_width)
    for start in range(0,m,frame_width):
        part=states[start:start+frame_width]
        counts=sparse_codec_fast.integer_counts(*sparse_codec_fast.compact(part),*part.shape)
        writer.append(part,counts)
    writer.finish();return fast_container.Container(path,binding,samples)


class Reader(MetadataReader):
    def read_field(self,name,indices=None):
        if name=='annotation/filter':value=np.full(self.n_variants,'PASS')
        elif name=='chromosome':value=np.full(self.n_variants,'21')
        else:value=np.arange(self.n_variants)
        return value if indices is None else value[indices]
    def read_ref_alt(self,indices=None):
        count=self.n_variants if indices is None else len(indices)
        return np.full(count,'A'),np.full(count,'C')


class CaptureWriter:
    checkpoint_tiles=True
    def __init__(self,*,fail_at=None):self.tiles=[];self.rows={0:[],1:[],2:[]};self.fail_at=fail_at;self.checks=[]
    def checkpoint_tile(self,kind,chromosome,job,traits,**options):
        self.checks.append(dict(kind=kind,traits=list(traits),**options));return None
    def write_tile(self,kind,chromosome,values,**options):
        if self.fail_at is not None and len(self.tiles)+1==self.fail_at:raise OSError('injected writer failure')
        self.tiles.append(dict(kind=kind,**options))
        for trait,rows in values:self.rows[trait].extend(rows)


def fitted_models(rows):
    result=[]
    for i,axis in enumerate(rows):
        model=model_for(axis)
        # Dyadic CPU oracle values make changed column geometry exact. This
        # tests orchestration and row/axis semantics, not CUDA reduction order.
        model.scaled_residuals=torch.arange(len(axis),dtype=torch.float32)/32
        model.inverse_variance=torch.ones(len(axis),dtype=torch.float32)
        model.precision_x=model.x.clone()
        model.fixed_effect_covariance=torch.tensor([[1/len(axis)]],dtype=torch.float32)
        result.append(model)
    return result


def execute(path,cross_frame,*,writer=None,empty=False,selected_empty=False,fail_read=False,trait_batch_size=2,
            genotype_block_size=8,effective_block_size=None,variant_count=37,frame_width=4):
    raw=states_fixture(m=variant_count,empty=empty);m,n=raw.shape
    axes=[np.arange(n-1,-1,-1),np.arange(n-1,-1,-1),np.arange(7,-1,-1)]
    descriptors=[descriptor(i,axis) for i,axis in enumerate(axes)]
    models=fitted_models(axes);repository=Repository(descriptors,models)
    _,groups=runtime._descriptor_groups(repository)
    options=AnalysisOptions(wrapper_semantics='base',memory_limit_gib=80)
    config=dict(individual_genotype_block_size=genotype_block_size,single_progress_frame_interval=1000,
        single_progress_seconds=10**9)
    if effective_block_size is not None:config['individual_effective_block_size']=effective_block_size
    if cross_frame is not None:config['single_cross_frame_batches']=cross_frame
    report=dict(association_rows={'individual':0},trait_rows={i:{'individual':0} for i in range(3)},
        binary_spa_selected=0,binary_spa_failed=0)
    writer=CaptureWriter() if writer is None else writer
    cache=make_small_frame_cache(path/'cache',raw,frame_width)
    requests=[]
    with SharedStateBroker(SharedMetadataReader(Reader(n,m)),cache,device='cpu') as broker:
        original=broker.read_states
        def read(columns,minimum_mac_bound=None):
            requests.append(np.asarray(columns).copy())
            assert len(columns)<=genotype_block_size
            if fail_read:raise OSError('injected raw failure')
            return original(columns,minimum_mac_bound=minimum_mac_bound)
        broker.read_states=read
        axis=broker.reader_view(axes[0]);initial_pins=axis._axis.pins
        structural=runtime._thin_pipeline(axis,models[0],axis._axis.samples,options,config)
        structural._base_mask=lambda *args:np.zeros(m,dtype=bool) if selected_empty else np.ones(m,dtype=bool)
        try:
            runtime._batched_single(groups,repository,broker,broker._reader,options,config,
                {'mac_cutoff':2,'variant_type':'variant','subset_variants_num':3},writer,report,set(),'21',5,trait_batch_size,structural)
        finally:
            assert repository.active==0
            assert axis._axis.pins==initial_pins
            axis.close()
        return writer,report,requests,repository.acquired


def assert_rows_exact(first,second):
    assert set(first)==set(second)
    for trait in first:
        assert len(first[trait])==len(second[trait])
        for a,b in zip(first[trait],second[trait]):
            assert list(a)==list(b)
            for key in a:
                if isinstance(a[key],float):
                    assert a[key].hex()==b[key].hex(),key
                else:assert a[key]==b[key],key


def test_enabled_cross_frame_same_rows_models_ordinals_and_reduced_bounded_requests(tmp_path,cpu_product_oracle):
    legacy,a,old_requests,_=execute(tmp_path/'legacy',None)
    candidate,b,new_requests,_=execute(tmp_path/'candidate',True)
    assert_rows_exact(legacy.rows,candidate.rows)
    assert a['association_rows']==b['association_rows']
    assert len(new_requests)<len(old_requests)
    assert max(map(len,new_requests))==8
    assert [int(x) for request in new_requests for x in request]==list(range(1,37,4))
    assert [len(candidate.rows[i]) for i in range(3)]==[9,9,7]
    assert max(tile['ordinal_after'] for tile in candidate.tiles)==9
    assert all(tile['block_identity'].startswith('source-upper-MAC-bounded-requests-v1:') for tile in candidate.tiles)
    assert 'single_request_batching' not in a
    assert b['single_request_batching']['counters']['physical_requests_examined']==10


def test_default_false_retains_original_requests_and_exact_checkpoint_keys(tmp_path,cpu_product_oracle):
    implicit,a,requests_a,_=execute(tmp_path/'implicit',None)
    explicit,b,requests_b,_=execute(tmp_path/'explicit',False)
    assert_rows_exact(implicit.rows,explicit.rows)
    assert implicit.tiles==explicit.tiles and len(requests_a)==len(requests_b)
    for first,second in zip(requests_a,requests_b):np.testing.assert_array_equal(first,second)
    assert all(':' not in tile['block_identity'] for tile in explicit.tiles)


@pytest.mark.parametrize('effective_bound',[128,256,512,1024])
def test_public_effective_width_bounds_cross_frame_requests_and_preserves_rows(tmp_path,cpu_product_oracle,effective_bound):
    # 1,025 eligible columns span 65 physical frames: each bound must be
    # reached, followed by a tail, rather than passing on an underfilled input.
    geometry=dict(genotype_block_size=1024,effective_block_size=effective_bound,
                  variant_count=4101,frame_width=64)
    legacy,a,old_requests,_=execute(tmp_path/'legacy',False,**geometry)
    candidate,b,new_requests,_=execute(tmp_path/'candidate',True,**geometry)
    assert_rows_exact(legacy.rows,candidate.rows)
    assert a['association_rows']==b['association_rows']
    assert max(map(len,new_requests))==effective_bound
    assert all(len(request)<=effective_bound for request in new_requests)
    expected=list(range(1,4101,4))
    assert [int(x) for request in new_requests for x in request]==expected
    assert len(new_requests)==(len(expected)+effective_bound-1)//effective_bound
    assert b['single_request_batching']['maximum_request_columns']==effective_bound
    assert b['single_request_batching']['counters']['maximum_request_columns']==effective_bound
    assert all(tile['block_identity'].startswith('source-upper-MAC-bounded-requests-v1:')
               for tile in candidate.tiles)
    assert len(new_requests)<len(old_requests)


@pytest.mark.parametrize('effective_bound',[128,256,512,1024])
def test_false_uses_legacy_genotype_bound_and_keys_independent_of_effective_width(tmp_path,cpu_product_oracle,effective_bound):
    geometry=dict(genotype_block_size=1024,variant_count=613,frame_width=257)
    implicit,a,old_requests,_=execute(tmp_path/'implicit',None,**geometry)
    explicit,b,new_requests,_=execute(tmp_path/'explicit',False,
        effective_block_size=effective_bound,**geometry)
    assert_rows_exact(implicit.rows,explicit.rows)
    assert implicit.tiles==explicit.tiles
    assert len(old_requests)==len(new_requests)==3
    assert max(map(len,new_requests))==257
    for first,second in zip(old_requests,new_requests):np.testing.assert_array_equal(first,second)
    assert 'single_request_batching' not in a and 'single_request_batching' not in b
    assert all(':' not in tile['block_identity'] for tile in explicit.tiles)


def test_effective_width_never_exceeds_existing_genotype_bound(tmp_path,cpu_product_oracle):
    _,report,requests,_=execute(tmp_path,True,genotype_block_size=128,
        effective_block_size=512,variant_count=613,frame_width=64)
    assert max(map(len,requests))==128
    assert report['single_request_batching']['maximum_request_columns']==128


def test_effective_width_is_bound_in_checkpoint_identity_even_when_underfilled(tmp_path,cpu_product_oracle):
    first,a,request_a,_=execute(tmp_path/'small',True,genotype_block_size=1024,effective_block_size=128)
    second,b,request_b,_=execute(tmp_path/'large',True,genotype_block_size=1024,effective_block_size=256)
    assert_rows_exact(first.rows,second.rows)
    assert len(request_a)==len(request_b)==1
    np.testing.assert_array_equal(request_a[0],request_b[0])
    assert a['single_request_batching']['maximum_request_columns']==128
    assert b['single_request_batching']['maximum_request_columns']==256
    assert [tile['block_identity'] for tile in first.tiles]!=[tile['block_identity'] for tile in second.tiles]


def test_changed_effective_width_cannot_reuse_underfilled_committed_tile(tmp_path,cpu_product_oracle):
    from fudan_wgs_toolkit.phewas_trait_csv import CommittedCSVCorruption
    output=tmp_path/'output'
    # Hold the outer binding fixed to isolate the additional request-identity
    # guard. Actual public plans also reject changing the configuration SHA.
    binding=dict(writer_binding(),configuration_sha256='fixed-to-test-request-identity')
    writer=PerTraitCheckpointCSVWriter(output,binding=binding,minimum_free_bytes=0)
    execute(tmp_path/'first',True,writer=writer,genotype_block_size=1024,effective_block_size=128)
    published=writer.finish()
    original={receipt['csv_file']:(output/receipt['csv_file']).read_bytes() for receipt in published['receipts']}
    resumed=PerTraitCheckpointCSVWriter(output,binding=binding,resume=True,worker_id='1',minimum_free_bytes=0)
    try:
        with pytest.raises(CommittedCSVCorruption,match='tile binding'):
            execute(tmp_path/'second',True,writer=resumed,genotype_block_size=1024,effective_block_size=256)
    finally:
        resumed.abort()
    assert all((output/file).read_bytes()==data for file,data in original.items())


@pytest.mark.parametrize('selected_empty',[False,True])
def test_empty_source_or_empty_selected_axis_keeps_zero_rows_and_ordinals(tmp_path,cpu_product_oracle,selected_empty):
    writer,report,requests,_=execute(tmp_path,True,empty=True,selected_empty=selected_empty)
    assert report['association_rows']['individual']==0 and all(not rows for rows in writer.rows.values())
    if selected_empty:assert not requests and not writer.tiles
    else:
        assert len(requests)==1 and not len(requests[0])
        assert all(tile['ordinal_before']==tile['ordinal_after']==0 for tile in writer.tiles)


def test_source_and_writer_errors_propagate_without_model_or_axis_leaks(tmp_path,cpu_product_oracle):
    with pytest.raises(OSError,match='injected raw failure'):execute(tmp_path/'read',True,fail_read=True)
    with pytest.raises(OSError,match='injected writer failure'):
        execute(tmp_path/'write',True,writer=CaptureWriter(fail_at=2),trait_batch_size=1)


@pytest.mark.parametrize('flag',[0,1,None,'true',[],{}])
def test_non_boolean_opt_in_rejected_before_source_or_prefetch(flag):
    structural=SimpleNamespace(_base_mask=lambda *args:pytest.fail('invalid switch touched metadata'))
    with pytest.raises(ValueError,match='single_cross_frame_batches'):
        runtime._batched_single([],None,None,None,None,{'single_cross_frame_batches':flag},{},None,{},set(),'21',0,1,structural)


def writer_binding():
    return dict(source_implementation_sha256='cpu-contract-source',input_manifest_sha256='cpu-contract-input',
        configuration_sha256='cross-frame-upper-MAC-v1-B8',models={str(i):'model-'+str(i) for i in range(3)})


def read_csv_rows(directory,report):
    result={0:[],1:[],2:[]}
    for receipt in report['receipts']:
        with gzip.open(directory/receipt['csv_file'],'rt',newline='') as stream:
            for row in csv.DictReader(stream):result[int(row['trait_index'])].append(row)
    return result


def test_partial_trait_tile_resume_reuses_only_bound_merged_requests(tmp_path,cpu_product_oracle):
    class Interrupted(PerTraitCheckpointCSVWriter):
        calls=0
        def write_tile(self,*args,**kwargs):
            self.calls+=1
            if self.calls==2:raise OSError('stop after first trait tile')
            return super().write_tile(*args,**kwargs)
    output=tmp_path/'resumed';writer=Interrupted(output,binding=writer_binding(),minimum_free_bytes=0)
    with pytest.raises(OSError,match='stop after first trait tile'):
        execute(tmp_path/'interrupted',True,writer=writer,trait_batch_size=1)
    writer.abort()
    recovered=PerTraitCheckpointCSVWriter(output,binding=writer_binding(),resume=True,worker_id='1',minimum_free_bytes=0)
    _,report,_,_=execute(tmp_path/'resume-input',True,writer=recovered,trait_batch_size=1)
    final=recovered.finish()
    reference=tmp_path/'reference';fresh=PerTraitCheckpointCSVWriter(reference,binding=writer_binding(),minimum_free_bytes=0)
    _,expected,_,_=execute(tmp_path/'reference-input',True,writer=fresh,trait_batch_size=1)
    wanted=fresh.finish()
    assert report['association_rows']==expected['association_rows']
    assert final['reused_blocks']>=1
    assert read_csv_rows(output,final)==read_csv_rows(reference,wanted)
