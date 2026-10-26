"""CPU memory contracts; no synthetic performance or CUDA execution claims."""
from contextlib import nullcontext
from types import SimpleNamespace
import pytest
import torch
from fudan_wgs_toolkit import tf32


@pytest.mark.parametrize('ld,rd',[(torch.float32,torch.float32),(torch.float64,torch.float32),(torch.float32,torch.float64),(torch.float64,torch.float64)])
def test_native_matrix_workspace_has_only_casts_and_fp32_output(ld,rd):
    d=tf32._estimated_product_workspace((11,1000),(1000,17),mode='tf32',left_dtype=ld,right_dtype=rd,breakdown=True)
    casts=4*((11000 if ld==torch.float64 else 0)+(17000 if rd==torch.float64 else 0))
    assert d['cast_bytes']==casts and d['output_bytes']==4*11*17
    assert d['required_bytes']==casts+4*11*17
    assert d['layout_copy_allowance_bytes']==d['component_workspace_bytes']==0


def test_vector_layout_and_zero_k_estimates():
    d=tf32._estimated_product_workspace((1,1000),(1000,7),mode='tf32',left_dtype=torch.float64,right_dtype=torch.float32,breakdown=True)
    assert d['route']=='gemv' and d['required_bytes']==4000+4*8000+4*7
    e=tf32._estimated_product_workspace((3,0),(0,4),mode='tf32',left_dtype=torch.float64,right_dtype=torch.float64,breakdown=True)
    assert e['required_bytes']==48 and e['cast_bytes']==e['layout_copy_allowance_bytes']==0
    assert tf32._estimated_product_workspace((0,3),(3,4),mode='tf32',left_dtype=torch.float32,right_dtype=torch.float32)==0


@pytest.mark.parametrize('cap,live,binding',[(50,60,'process_cap'),(70,60,'live_free'),(60,60,'both'),(-1,60,'process_cap'),(60,-1,'live_free')])
def test_live_shared_device_and_process_budget_bind_independently(cap,live,binding):
    d=tf32._product_workspace_availability(allocated=100,reserved=120,free=live-10,limit=100+cap+10,reserve=10)
    assert d['binding']==binding and d['available_bytes']==max(0,min(cap,live))
    assert d['cap_available_bytes']==cap and d['live_available_bytes']==live


def test_guard_rejects_before_math_with_one_query_and_anonymous_snapshot(monkeypatch):
    tf32.configure_tf32();tf32.execution_metadata(reset=True);queries=[]
    monkeypatch.setattr(tf32,'allocator_bytes',lambda device:(100,120))
    monkeypatch.setattr(torch.cuda,'device',lambda device:nullcontext())
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(queries.append(1) or 0,80*2**30))
    left=SimpleNamespace(shape=(2,100),dtype=torch.float64,device='cuda:0')
    right=SimpleNamespace(shape=(100,3),dtype=torch.float32,device='cuda:0')
    with pytest.raises(MemoryError,match='binding=live_free.*no precision fallback'):
        tf32._guard_product_workspace(left,right,mode='tf32')
    m=tf32.execution_metadata(reset=True);d=m['tf32_memory_guard']['last_rejection']
    assert len(queries)==m['tf32_memory_guard']['cuda_free_queries']==1
    assert m['tf32_gemm_call_count']==m['fp32_vector_product_count']==0
    assert d['allocated_bytes']==100 and d['reserved_bytes']==120 and d['cuda_free_bytes']==0
    assert d['required_bytes']==4*200+4*6 and d['component_workspace_bytes']==0
    assert tf32.execution_metadata()['tf32_memory_guard']['last_rejection'] is None


@pytest.mark.parametrize('limit',[True,0,-1,float('inf'),float('nan'),'20'])
def test_invalid_budget_rejected(limit):
    with pytest.raises(ValueError,match='memory_limit_gib'):tf32.configure_tf32(memory_limit_gib=limit)


@pytest.mark.parametrize('reserve',[True,-1,1.5,None])
def test_invalid_live_reserve_rejected(reserve):
    with pytest.raises(ValueError,match='memory_reserve_bytes'):
        tf32.configure_tf32(memory_reserve_bytes=reserve)


def test_configured_reserve_reaches_both_live_and_process_guard(monkeypatch):
    monkeypatch.setattr(tf32, '_memory_limit_bytes', tf32._memory_limit_bytes)
    monkeypatch.setattr(tf32, '_memory_reserve_bytes', tf32._memory_reserve_bytes)
    configured=tf32.configure_tf32(memory_limit_gib=80,memory_reserve_bytes=512)
    monkeypatch.setattr(tf32,'allocator_bytes',lambda device:(100,100))
    monkeypatch.setattr(torch.cuda,'device',lambda device:nullcontext())
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(520,80*2**30))
    left=SimpleNamespace(shape=(2,2),dtype=torch.float32,device='cuda:0')
    right=SimpleNamespace(shape=(2,2),dtype=torch.float32,device='cuda:0')
    with pytest.raises(MemoryError,match='binding=live_free'):
        tf32._guard_product_workspace(left,right,mode='tf32')
    snapshot=tf32.execution_metadata()['tf32_memory_guard']['last_rejection']
    assert configured['tf32_memory_reserve_bytes']==snapshot['reserve_bytes']==512
    assert snapshot['cap_available_bytes']==80*2**30-100-512


@pytest.mark.parametrize('unused_cache',[0,10000])
def test_process_reserve_is_preserved_even_with_idle_device_and_owned_cache(monkeypatch,unused_cache):
    # Output fits the raw cap, but would leave only 15 bytes of a 16-byte
    # reserve. An idle device or a large reusable cache cannot override it.
    monkeypatch.setattr(tf32,'_memory_limit_bytes',100+16+16-1)
    monkeypatch.setattr(tf32,'_memory_reserve_bytes',16)
    monkeypatch.setattr(tf32,'allocator_bytes',lambda device:(100,100+unused_cache))
    monkeypatch.setattr(torch.cuda,'get_allocator_backend',lambda:'native',raising=False)
    monkeypatch.setattr(torch.cuda,'device',lambda device:nullcontext())
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(10**9,80*2**30))
    left=SimpleNamespace(shape=(2,2),dtype=torch.float32,device='cuda:0')
    right=SimpleNamespace(shape=(2,2),dtype=torch.float32,device='cuda:0')
    with pytest.raises(MemoryError,match='binding=process_cap'):
        tf32._guard_product_workspace(left,right,mode='tf32')
    snapshot=tf32.execution_metadata()['tf32_memory_guard']['last_rejection']
    assert snapshot['required_bytes']==16 and snapshot['available_bytes']==15


def test_available_ignores_negative_unused_reservation():
    assert tf32._available_product_workspace(allocated=100,reserved=90,free=50,limit=1000,reserve=10)==40


@pytest.mark.parametrize('ld,rd',[(torch.float32,torch.float32),(torch.float64,torch.float32),(torch.float32,torch.float64)])
def test_outer_broadcast_has_no_operand_materialization_or_component_workspace(ld,rd):
    d=tf32._estimated_product_workspace((11,1),(1,17),mode='tf32',left_dtype=ld,right_dtype=rd,breakdown=True)
    casts=4*((11 if ld==torch.float64 else 0)+(17 if rd==torch.float64 else 0))
    assert d['route']=='outer' and d['required_bytes']==casts+4*11*17
    assert d['layout_copy_allowance_bytes']==d['component_workspace_bytes']==0


def test_live_memory_queries_are_not_cached_even_for_repeated_product(monkeypatch):
    tf32.configure_tf32();tf32.execution_metadata(reset=True);queries=[]
    monkeypatch.setattr(tf32,'allocator_bytes',lambda device:(0,0))
    monkeypatch.setattr(torch.cuda,'device',lambda device:nullcontext())
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(queries.append(1) or 30*2**30,40*2**30))
    a=SimpleNamespace(shape=(2,1),dtype=torch.float32,device='cuda:0')
    b=SimpleNamespace(shape=(1,3),dtype=torch.float32,device='cuda:0')
    tf32._guard_product_workspace(a,b,mode='tf32');tf32._guard_product_workspace(a,b,mode='tf32')
    assert len(queries)==tf32.execution_metadata(reset=True)['tf32_memory_guard']['cuda_free_queries']==2
