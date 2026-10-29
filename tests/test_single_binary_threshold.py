"""Strict SPA-boundary replay contracts; synthetic fixtures are not benchmarks."""
from contextlib import nullcontext
from types import SimpleNamespace
import gc
import weakref

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.cache_runtime.single_batches import _physical_requests
from fudan_wgs_toolkit.phewas_runtime import single, single_threshold as threshold
from fudan_wgs_toolkit.statistics import _chi1_from_score


def planner_broker():
    return SimpleNamespace(_reader=SimpleNamespace(n_variants=40),
        _starts=np.array([0,9,17,29]),_sizes=np.array([9,8,12,11]))


@pytest.mark.parametrize('block_size',[1,3,8,1024])
@pytest.mark.parametrize('selected',[
    [0,2,7,8,10,12,14,17,22,28,29,31,39],
    [22,0,39,2,17,9,10,28,7,38],
])
def test_direct_plan_exact_original_requests_and_order_without_reading_prefix(selected,block_size):
    broker=planner_broker();axis=np.asarray(selected,dtype=np.int64)
    wanted=axis[1::3]
    expected=[request for request in _physical_requests(broker,axis,block_size)
              if np.isin(request,wanted).any()]
    actual=list(threshold.LegacySingleRequestPlan(broker,axis,block_size).requests_for(wanted))
    assert len(actual)==len(expected)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
    assert not hasattr(broker,'read_states')  # Planning owns no reader IO.


def test_plan_respects_full_job_selection_and_raw_bound_not_effective_or_entire_frame():
    broker=planner_broker();selected=np.array([10,12,14,17,22],dtype=np.int64)
    plan=threshold.LegacySingleRequestPlan(broker,selected,2)
    result=list(plan.requests_for(np.array([14,22])))
    assert [part.tolist() for part in result]==[[14],[17,22]]
    assert list(plan.requests_for(np.array([],dtype=np.int64)))==[]
    with pytest.raises(ValueError,match='outside the original selected axis'):
        list(plan.requests_for(np.array([11])))


@pytest.mark.parametrize('bad',[0,-1,True,1.5])
def test_invalid_original_raw_bound_is_rejected(bad):
    with pytest.raises(ValueError,match='legacy genotype block size'):
        threshold.LegacySingleRequestPlan(planner_broker(),np.array([1,2]),bad)


class Broker:
    def __init__(self,*,error=None):
        self._reader=SimpleNamespace(n_variants=32)
        self._starts=np.arange(0,32,8);self._sizes=np.full(4,8)
        self.reads=[];self.guards=[];self.error=error
    def _guard(self,required):
        self.guards.append(required)
        if self.error=='guard':raise MemoryError('injected replay budget')
    def read_states(self,request,*,minimum_mac_bound):
        if self.error=='read':raise OSError('injected replay read')
        self.reads.append((np.asarray(request).copy(),minimum_mac_bound))
        return np.asarray(request).copy()
    def trait_block(self,raw,rows,*,minimum_mac):
        assert rows.tolist()==[3,1,2,0] and minimum_mac==20
        return raw


class Pipeline:
    def __init__(self,index,*,family='binomial',empty=False):
        self.models=[SimpleNamespace(family=family,n=4,scaled_residuals=torch.full((4,),float(index)),
                                    x=torch.ones((4,1),dtype=torch.float32),device='cpu',
                                    n_pheno=1,use_spa=False,matmul_mode='tf32')]
        self.trait_rows=[np.arange(4)];self.options=SimpleNamespace(wrapper_semantics='base',imputation='mean')
        self.union_rows=np.array([3,1,2,0]);self.profiler=SimpleNamespace(measure=lambda *a,**k:nullcontext())
        self.preparations=[];self.empty=empty;self.genotype_references=[]
    def _workspace_estimate(self,model,width,*,individual):
        assert individual and model is self.models[0]
        return 16*model.n*width
    def _prepare_individual_block(self,block,ordinal,*,mac_cutoff):
        assert mac_cutoff==20
        self.preparations.append((block.copy(),ordinal))
        return (None if self.empty else block),999999
    def _prepare_individual_trait(self,state,trait,*,mac_cutoff):
        assert trait==0 and mac_cutoff==20
        g=torch.zeros((4,len(state)),dtype=torch.float32);g[0]=torch.as_tensor(state,dtype=torch.float32)
        self.genotype_references.append(weakref.ref(g))
        return dict(genotype=g,selected=state.copy(),trait_ordinals=np.arange(len(state)))


def cut_scores():
    high=torch.tensor(1.959963984540054,dtype=torch.float32)
    low=torch.nextafter(high,torch.tensor(float('-inf'),dtype=torch.float32))
    assert float(_chi1_from_score(low[None],torch.ones(1))[0])>.05
    assert float(_chi1_from_score(high[None],torch.ones(1))[0])<.05
    return low,high


def recheck_fixture(*,traits=16,error=None,second_variant=6):
    broker=Broker(error=error);pipelines=[Pipeline(i,family='gaussian' if i==1 else 'binomial') for i in range(traits)]
    selected=np.array([0,2,4,6,8,10,12,18,22,26],dtype=np.int64)
    current=np.array([2,second_variant,10,26],dtype=np.int64)
    low,high=cut_scores()
    scores=torch.full((len(current),traits),.2,dtype=torch.float32)
    scores[0,0]=high;scores[1,-1]=low;scores[0,1]=high
    variances=[torch.ones(len(current),dtype=torch.float32) for _ in pipelines]
    callback=threshold.LegacySingleThresholdRecheck(broker,selected,
        genotype_block_size=8,mac_cutoff=20,p_filter_cutoff=.05)
    return broker,pipelines,selected,current,scores,variances,callback


@pytest.mark.parametrize('perturbation',['score','variance'])
@pytest.mark.parametrize('second_variant',[6,18])
def test_only_near_binary_cells_restore_both_directions_from_original_full_tile(monkeypatch,perturbation,second_variant):
    broker,pipelines,axis,current,scores,variances,callback=recheck_fixture(second_variant=second_variant)
    low,high=cut_scores();native_calls=[]
    if perturbation=='variance':
        scores[0,0]=high;scores[1,-1]=high
        variances[-1][1]=torch.nextafter(torch.tensor(1.),torch.tensor(float('inf')))
    before_score=scores.clone();before_var=[v.clone() for v in variances]
    def original_core(tile,g):
        native_calls.append((g.shape,len(tile),g.dtype,g[0].clone()))
        assert tile==pipelines
        values=g[0].to(torch.int64).tolist()
        result=torch.full((len(values),len(tile)),.2,dtype=torch.float32)
        vv=[torch.ones(len(values)) for _ in tile]
        if 2 in values:
            result[values.index(2),0]=low if perturbation=='score' else high
            if perturbation=='variance':vv[0][values.index(2)]=torch.nextafter(torch.tensor(1.),torch.tensor(float('inf')))
        if second_variant in values:result[values.index(second_variant),-1]=high
        return result,vv
    monkeypatch.setattr(single,'_shared_single_normal_statistics',original_core)
    callback(pipelines,{'selected':current},scores,variances)
    assert callback.metrics['near_boundary_cells']==callback.metrics['restored_cells']==2
    assert callback.metrics['legacy_requests_replayed']==(1 if second_variant==6 else 2)
    assert native_calls and all(t==16 and dtype==torch.float32 for _,t,dtype,_ in native_calls)
    assert broker.reads[0][0].tolist()==[0,2,4,6]
    assert all(mac==20 for _,mac in broker.reads)
    assert all(ordinal==0 for _,ordinal in pipelines[0].preparations)
    assert callback.metrics['maximum_legacy_request_columns']==4
    after=_chi1_from_score(scores.T,torch.stack(variances))
    assert not bool(after[0,0]<.05) and bool(after[-1,1]<.05)
    assert torch.equal(scores[:,1],before_score[:,1])  # Gaussian near cell untouched.
    for trait in range(16):
        for column in range(len(current)):
            if (trait,column) not in ((0,0),(15,1)):
                assert scores[column,trait].item()==before_score[column,trait].item()
                assert variances[trait][column].item()==before_var[trait][column].item()
    assert all(p.preparations==[] for p in pipelines[1:])
    assert list(callback.plan.selected)==list(axis)
    gc.collect();assert all(reference() is None for reference in pipelines[0].genotype_references)


def test_no_near_cells_or_no_binary_traits_never_rereads_or_invokes_core(monkeypatch):
    broker,pipelines,_,current,scores,variances,callback=recheck_fixture()
    monkeypatch.setattr(single,'_shared_single_normal_statistics',lambda *a:pytest.fail('unexpected replay'))
    scores.fill_(.2);callback(pipelines,{'selected':current},scores,variances)
    assert not broker.reads and callback.metrics['near_boundary_cells']==0
    for pipeline in pipelines:pipeline.models[0].family='gaussian'
    before=callback.metrics.copy();callback(pipelines,{'selected':current},scores,variances)
    assert callback.metrics==before and not broker.reads


@pytest.mark.parametrize('kind',['guard','read','core','empty'])
def test_replay_failure_propagates_and_retains_no_dense_replay_owner(monkeypatch,kind):
    broker,pipelines,_,current,scores,variances,callback=recheck_fixture(error=kind)
    if kind=='empty':pipelines[0].empty=True
    def core(*args):raise ArithmeticError('injected replay kernel')
    monkeypatch.setattr(single,'_shared_single_normal_statistics',core)
    expected=MemoryError if kind=='guard' else OSError if kind=='read' else ValueError if kind=='empty' else ArithmeticError
    try:callback(pipelines,{'selected':current},scores,variances)
    except expected:pass
    else:pytest.fail('replay failure was concealed')
    assert callback.metrics['restored_cells']==0
    if kind=='guard':assert not broker.reads
    assert all(not isinstance(value,torch.Tensor) for value in vars(callback).values())
    gc.collect();assert all(reference() is None for reference in pipelines[0].genotype_references)


def test_replay_keeps_whole_legacy_trait_extreme_tail_vector_fallback(monkeypatch):
    broker,pipelines,_,current,scores,variances,callback=recheck_fixture()
    # Keep one target; an extreme NON-target legacy column must still force
    # the entire original trait's GEMV, including that near-boundary target.
    scores[1,-1]=.2;low,high=cut_scores();calls=[]
    counters=dict(fp32_simt_products=0,fp32_vector_products=0,simt_admission_fallbacks=0)
    monkeypatch.setattr(single,'score_execution_metadata',lambda:dict(counters))
    def score_columns(g,residuals):
        calls.append((tuple(g.shape),len(residuals),g[0].clone()))
        variants=g[0].to(torch.int64).tolist()
        if len(residuals)>1:
            counters['fp32_simt_products']+=1
            value=torch.full((len(variants),len(residuals)),.2)
            value[variants.index(0),0]=100.;value[variants.index(2),0]=high
            return value
        counters['fp32_vector_products']+=1
        value=torch.full((len(variants),1),.2)
        value[variants.index(2),0]=low
        return value
    monkeypatch.setattr(single,'fp32_score_columns',score_columns)
    monkeypatch.setattr(single,'_variance_from_shared_genotype',lambda model,g,*a,**k:torch.ones(g.shape[1]))
    callback(pipelines,{'selected':current},scores,variances)
    assert [(shape,traits) for shape,traits,_ in calls]==[((4,4),16),((4,4),1)]
    assert scores[0,0].item()==low.item()
    assert callback.metrics['restored_cells']==1


@pytest.mark.parametrize('bound,traits',[(8,1),(1024,16),(1024,32)])
def test_host_scratch_is_bounded_by_original_raw_width_and_complete_trait_tile(bound,traits):
    scratch=threshold.threshold_host_scratch_bytes(bound,traits)
    # Admission is the implemented linear forecast, not an arbitrary MiB cap.
    # It includes per-variant metadata plus per-trait masks/index bookkeeping.
    assert scratch>=bound*traits*8
    assert threshold.threshold_host_scratch_bytes(2*bound,traits)==2*scratch
    assert (threshold.threshold_host_scratch_bytes(bound,traits+1)-scratch)==256*bound
    # Exercise the real existing output-lease path: the replay forecast is
    # additional to ordinary rows/transfer, admitted once, then retired.
    from fudan_wgs_toolkit.phewas_resources import host_output_lease_scope,single_output_workspace_bytes
    calls=[]
    class Lease:
        def __enter__(self):calls.append(('enter',));return self
        def set_future(self,value,*,phase):calls.append(('future',value,phase))
        def release(self):calls.append(('release',))
    lease=Lease()
    with host_output_lease_scope(lambda *args:lease,rows=3,traits=traits,
            chromosome_index=0,job_index=0,scratch_bytes=64*1024+scratch) as reservation:
        assert calls==[('enter',),('future',single_output_workspace_bytes(3,traits)+scratch,
                                  'single_output_rows_and_transfer')]
        reservation.materialized()
    assert calls[-2:]==[('future',0,'single_output_materialized'),('release',)]
    assert sum(item[0]=='enter' for item in calls)==1


def test_cutoff_and_planner_validation_never_silently_changes_scientific_filter():
    broker=planner_broker()
    for cutoff in (0,1.1,float('nan'),True):
        with pytest.raises(ValueError,match='SPA P cutoff'):
            threshold.LegacySingleThresholdRecheck(broker,np.array([1,2]),genotype_block_size=8,
                mac_cutoff=20,p_filter_cutoff=cutoff)
    with pytest.raises(ValueError,match='duplicate'):
        threshold.LegacySingleRequestPlan(broker,np.array([1,1]),8)


@pytest.mark.parametrize('callback',['recheck','correction'])
def test_private_core_capture_rejects_callbacks_before_calculation(callback,monkeypatch):
    pipeline=Pipeline(0)
    monkeypatch.setattr(single,'fp32_score_columns',lambda *a,**k:pytest.fail('invalid capture reached score'))
    options={'normal_recheck' if callback=='recheck' else 'binary_correction':lambda *a:None}
    with pytest.raises(ValueError,match='capture cannot use'):
        single.process_single_shared_genotype([pipeline],None,0,None,
            prepared_state=({'genotype':torch.ones((4,3))},0),_return_normal_core=True,**options)


@pytest.mark.parametrize('flag',[0,1,None,'true'])
def test_private_core_capture_requires_actual_boolean(flag):
    with pytest.raises(TypeError,match='_return_normal_core'):
        single.process_single_shared_genotype([],None,0,None,_return_normal_core=flag)


def test_thin_capture_uses_only_original_producer_and_returns_owned_vectors_without_copy(monkeypatch):
    wrapper=single._shared_single_normal_statistics;pipelines=[Pipeline(0),Pipeline(1)]
    original_score=torch.tensor([[.1,.2],[.3,.4]],dtype=torch.float32)
    original_variance=[torch.tensor([1.,2.]),torch.tensor([3.,4.])]
    monkeypatch.setattr(single,'_shared_single_normal_statistics',lambda *a:pytest.fail('normal producer recursively called wrapper'))
    monkeypatch.setattr(single,'fp32_score_columns',lambda *a,**k:original_score)
    monkeypatch.setattr(single,'_variance_from_shared_genotype',lambda model,*a,**k:original_variance[int(model.scaled_residuals[0])])
    actual_score,actual_variance=wrapper(pipelines,torch.ones((4,2),dtype=torch.float32))
    assert actual_score is original_score
    assert all(a is b for a,b in zip(actual_variance,original_variance))


@pytest.mark.parametrize('stage',['score','variance'])
def test_held_capture_exception_traceback_cannot_retain_second_genotype(monkeypatch,stage):
    pipeline=Pipeline(0);references=[]
    def score(g,residuals):
        if stage=='score':
            temporary=g[:, :]
            raise ArithmeticError('held capture score failure')
        return torch.ones((g.shape[1],len(residuals)),dtype=torch.float32)
    def variance(model,g,centered,mean,**options):
        temporary=centered[:, :]
        raise ArithmeticError('held capture variance failure')
    monkeypatch.setattr(single,'fp32_score_columns',score)
    monkeypatch.setattr(single,'_variance_from_shared_genotype',variance)
    def capture():
        genotype=torch.ones((4,7),dtype=torch.float32);references.append(weakref.ref(genotype))
        try:single._shared_single_normal_statistics([pipeline],genotype)
        except ArithmeticError as error:return error
        finally:genotype=None
    held=capture()
    assert isinstance(held,ArithmeticError) and held.__traceback__ is not None
    assert str(held)=='held capture '+stage+' failure'
    # Keep both exception AND its complete traceback alive during the check.
    traceback=held.__traceback__;assert traceback is held.__traceback__
    gc.collect();assert references[0]() is None


def test_ordinary_normal_failure_does_not_invoke_private_capture_cleanup(monkeypatch):
    pipeline=Pipeline(0)
    def fail(*args,**kwargs):raise ArithmeticError('ordinary failure')
    monkeypatch.setattr(single,'fp32_score_columns',fail)
    monkeypatch.setattr(single,'_clear_normal_core_error',lambda *a:pytest.fail('ordinary path cleared error frames'))
    with pytest.raises(ArithmeticError,match='ordinary failure'):
        single.process_single_shared_genotype([pipeline],None,0,'21',
            prepared_state=({'genotype':torch.ones((4,3))},0))
