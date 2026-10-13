"""Owned native allocator cache preflight; no timing/GPU acceptance claims."""
from contextlib import nullcontext
from types import SimpleNamespace
import pytest
import torch
from fudan_wgs_toolkit import tf32


@pytest.fixture(autouse=True)
def reset_guard(monkeypatch):
    previous_limit=tf32._memory_limit_bytes
    tf32.configure_tf32();tf32.execution_metadata(reset=True)
    monkeypatch.setattr(torch.cuda,'get_allocator_backend',lambda:'native',raising=False)
    monkeypatch.setattr(torch.cuda,'device',lambda device:nullcontext())
    yield
    tf32._memory_limit_bytes=previous_limit
    tf32.execution_metadata(reset=True)


def operands(left_dtype=torch.float32,right_dtype=torch.float32):
    return (SimpleNamespace(shape=(2,4),dtype=left_dtype,device='cuda:0'),
            SimpleNamespace(shape=(4,3),dtype=right_dtype,device='cuda:0'))


def snapshots(monkeypatch,allocated,reserved,free=30*2**30):
    calls=[]
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda device:(calls.append(('allocated',device)) or allocated))
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda device:(calls.append(('reserved',device)) or reserved))
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(calls.append(('free','cuda:0')) or free,40*2**30))
    return calls


@pytest.mark.parametrize('cache_delta,cap_delta,owned,rejected',[
    (0,0,True,False),(1,0,True,False),(-1,0,False,False),
    (0,-1,False,True),(1,-1,False,True),
])
def test_byte_boundaries_keep_cap_and_fallback(monkeypatch,cache_delta,cap_delta,owned,rejected):
    required=4*2*3;allocated=100
    tf32._memory_limit_bytes=allocated+required+tf32._memory_reserve_bytes+cap_delta
    calls=snapshots(monkeypatch,allocated,
        allocated+required+tf32._memory_reserve_bytes+cache_delta)
    if rejected:
        with pytest.raises(MemoryError,match='process_cap.*no precision fallback'):
            tf32._guard_product_workspace(*operands(),mode='tf32')
    else:tf32._guard_product_workspace(*operands(),mode='tf32')
    report=tf32.execution_metadata()['tf32_memory_guard']
    assert report['total_checks']==1
    assert report['owned_cache_passes']==int(owned)
    assert report['cuda_free_queries']==int(not owned)
    assert report['workspace_rejections']==int(rejected)
    assert calls[:2]==[('allocated','cuda:0'),('reserved','cuda:0')]
    if owned:
        assert report['last_owned_cache_pass']['cuda_free_bytes'] is None
        assert report['last_owned_cache_pass']['available_owned_cache_lower_bound_bytes']>=required
        assert report['cuda_free_query_host_wall_seconds']==0
    assert report['allocator_backend_check_counts']=={'native':1}


@pytest.mark.parametrize('backend',['cudaMallocAsync','unknown','custom'])
def test_non_native_backend_always_queries_fresh_free(monkeypatch,backend):
    monkeypatch.setattr(torch.cuda,'get_allocator_backend',lambda:backend)
    calls=snapshots(monkeypatch,0,2**30)
    tf32._guard_product_workspace(*operands(),mode='tf32')
    report=tf32.execution_metadata()['tf32_memory_guard']
    assert calls[-1]==('free','cuda:0')
    assert report['owned_cache_passes']==0 and report['cuda_free_queries']==1
    assert report['allocator_backend_check_counts']=={
        backend if backend in ('native','cudaMallocAsync') else 'unknown':1}


def test_missing_or_failed_backend_api_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda,'get_allocator_backend',None)
    monkeypatch.setattr(torch.cuda.memory,'get_allocator_backend',None,raising=False)
    calls=snapshots(monkeypatch,0,2**30)
    tf32._guard_product_workspace(*operands(),mode='tf32')
    def failing():raise RuntimeError('unsupported backend query')
    monkeypatch.setattr(torch.cuda,'get_allocator_backend',failing)
    tf32._guard_product_workspace(*operands(),mode='tf32')
    report=tf32.execution_metadata()['tf32_memory_guard']
    assert sum(kind=='free' for kind,_ in calls)==2
    assert report['allocator_backend_check_counts']=={'unknown':2}
    assert report['owned_cache_passes']==0


def test_allocator_state_is_fresh_after_an_owned_cache_pass(monkeypatch):
    allocated=iter([0,0]);reserved=iter([2**30,0]);calls=[]
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda device:(calls.append('allocated') or next(allocated)))
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda device:(calls.append('reserved') or next(reserved)))
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(calls.append('free') or 30*2**30,40*2**30))
    tf32._guard_product_workspace(*operands(),mode='tf32')
    tf32._guard_product_workspace(*operands(),mode='tf32')
    assert calls==['allocated','reserved','allocated','reserved','free']
    report=tf32.execution_metadata()['tf32_memory_guard']
    assert report['total_checks']==2 and report['owned_cache_passes']==report['cuda_free_queries']==1


def test_live_free_snapshots_are_never_reused(monkeypatch):
    calls=snapshots(monkeypatch,100,90)
    tf32._guard_product_workspace(*operands(),mode='tf32')
    tf32._guard_product_workspace(*operands(),mode='tf32')
    assert sum(kind=='free' for kind,_ in calls)==2
    assert tf32.execution_metadata()['tf32_memory_guard']['owned_cache_passes']==0


def test_output_and_casts_are_both_covered_by_owned_cache(monkeypatch):
    # Cache covers only the output; FP64-storage input needs an FP32 cast too.
    calls=snapshots(monkeypatch,0,tf32._memory_reserve_bytes+4*2*3,free=0)
    with pytest.raises(MemoryError):
        tf32._guard_product_workspace(*operands(torch.float64),mode='tf32')
    report=tf32.execution_metadata()['tf32_memory_guard']
    assert calls[-1][0]=='free'
    assert report['last_rejection']['required_bytes']==4*(2*4+2*3)
    assert report['last_rejection']['component_workspace_bytes']==0
    assert report['owned_cache_passes']==0


def test_no_workspace_check_for_zero_output_and_metrics_reset(monkeypatch):
    calls=snapshots(monkeypatch,0,2**30)
    a,b=operands();a.shape=(0,4)
    tf32._guard_product_workspace(a,b,mode='tf32')
    assert not calls
    tf32._guard_product_workspace(*operands(),mode='tf32')
    report=tf32.execution_metadata(reset=True)['tf32_memory_guard']
    assert report['total_checks']==report['owned_cache_passes']==1
    assert report['guard_host_wall_seconds']>=report['allocator_snapshot_host_wall_seconds']>=0
    after=tf32.execution_metadata()['tf32_memory_guard']
    assert after['total_checks']==after['owned_cache_passes']==after['cuda_free_queries']==0
    assert after['last_owned_cache_pass'] is None and after['allocator_backend_check_counts']=={}


@pytest.mark.parametrize('required',[1,24,1048576])
def test_owned_condition_implies_the_original_guard_for_any_nonnegative_free(required):
    reserve=256*2**20;allocated=10*2**20
    reserved=allocated+required+reserve;limit=allocated+required+reserve
    assert tf32._owned_cache_sufficient(allocated,reserved,required,limit=limit,reserve=reserve)
    for free in (0,1,128*2**20,10*2**30):
        original=tf32._product_workspace_availability(allocated=allocated,reserved=reserved,
            free=free,limit=limit,reserve=reserve)
        assert original['available_bytes']>=required


@pytest.mark.parametrize('reserve',[1,256*2**20,2*2**30])
def test_owned_cache_cannot_consume_process_reserve(reserve):
    required=24;allocated=100
    reserved=allocated+required+reserve+2**30
    assert not tf32._owned_cache_sufficient(allocated,reserved,required,
        limit=allocated+required+reserve-1,reserve=reserve)
    assert tf32._owned_cache_sufficient(allocated,reserved,required,
        limit=allocated+required+reserve,reserve=reserve)
