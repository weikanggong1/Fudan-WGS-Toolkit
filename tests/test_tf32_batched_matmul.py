"""Trait batching formulas and memory contracts, plus explicit CUDA gates.

CPU/mock checks are not GPU speed or association-accuracy measurements.
"""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from fudan_wgs_toolkit import tf32


@pytest.fixture(autouse=True)
def reset_metadata():
    tf32.execution_metadata(reset=True)
    yield
    tf32.execution_metadata(reset=True)


def strided_matrices():
    a0=torch.arange(60,dtype=torch.float64).reshape(10,6)[1:9:2,::2].T
    a1=torch.arange(48,dtype=torch.float64).reshape(4,12)[:,1:7:2].T
    b0=torch.arange(40,dtype=torch.float64).reshape(4,10)[:,1:7:2]
    b1=torch.arange(40,dtype=torch.float64).reshape(10,4)[2:8:2].T
    return (a0,a1),(b0,b1)


def test_fp64_sequence_control_is_independent_and_preserves_offset_views():
    lefts,rights=strided_matrices()
    stamps=[(v.data_ptr(),v.stride(),v._version,v.clone()) for v in (*lefts,*rights)]
    actual=tf32.batched_matmul(lefts,rights,mode='fp64')
    expected=torch.stack([a@b for a,b in zip(lefts,rights)])
    assert actual.shape==(2,3,3) and actual.dtype==torch.float64
    assert torch.equal(actual,expected)
    for value,(pointer,strides,version,before) in zip((*lefts,*rights),stamps):
        assert (value.data_ptr(),value.stride(),value._version)==(pointer,strides,version)
        assert torch.equal(value,before)
    assert tf32.execution_metadata()['tf32_actual_kernel_launch_count']==0


@pytest.mark.parametrize('shared_side',['left','right','neither','both'])
def test_trait_axis_and_shared_matrix_fp64_control(shared_side):
    left=torch.arange(72,dtype=torch.float64).reshape(3,6,4).transpose(1,2)
    right=torch.arange(54,dtype=torch.float64).reshape(3,3,6).transpose(1,2)
    a=left[1] if shared_side in ('left','both') else left
    b=right[2] if shared_side in ('right','both') else right
    actual=tf32.batched_matmul(a,b,mode='fp64')
    count=1 if shared_side=='both' else 3
    expected=torch.stack([(a if a.ndim==2 else a[t])@(b if b.ndim==2 else b[t]) for t in range(count)])
    assert torch.equal(actual,expected) and actual.shape==(count,4,3)


def test_normalisation_does_not_stack_or_copy_trait_model_storage():
    left=torch.arange(120.).reshape(3,5,8).transpose(1,2)
    right=torch.arange(70.).reshape(5,14)[:,1:8:2]
    lefts,rights=tf32._normalise_batched_operands(left,right,maximum_batch_size=32)
    assert len(lefts)==len(rights)==3
    assert all(value.untyped_storage().data_ptr()==left.untyped_storage().data_ptr() for value in lefts)
    assert [value.data_ptr() for value in lefts]==[left[t].data_ptr() for t in range(3)]
    assert [value.stride() for value in lefts]==[left[t].stride() for t in range(3)]
    assert all(value is right for value in rights)


@pytest.mark.parametrize('limit',[True,0,33,-1,1.5,None])
def test_maximum_batch_is_bounded_before_allocations(limit):
    with pytest.raises(ValueError,match='maximum_batch_size'):
        tf32.batched_matmul(torch.ones(2,3),torch.ones(3,2),maximum_batch_size=limit)


def test_inadmissible_trait_axis_is_rejected_before_unbind_host_view_allocation(monkeypatch):
    giant_empty_axis=torch.empty((1000000,0,0))
    def forbidden(*args):raise AssertionError('batch limit must precede view allocation')
    monkeypatch.setattr(torch.Tensor,'unbind',forbidden)
    with pytest.raises(ValueError,match='batch size'):
        tf32.batched_matmul(giant_empty_axis,torch.empty(0,0))


def test_invalid_batch_shapes_and_broadcasts_are_rejected():
    a=torch.ones(2,3);b=torch.ones(3,2)
    cases=[([],b,'nonempty'),(torch.empty(0,2,3),b,'equal trait counts'),
           ([a]*33,b,'batch size'),([a]*2,[b],'equal trait counts'),
           ([a,torch.ones(3,3)],b,'same matrix dimensions'),
           ([a]*2,[b,torch.ones(3,4)],'same matrix dimensions'),
           (torch.ones(2,3,4,5),b,'2D shared or 3D'),
           (a,torch.ones(4,2),'Invalid native product shapes')]
    for left,right,message in cases:
        with pytest.raises(ValueError,match=message):
            tf32.batched_matmul(left,right,mode='fp64')
    with pytest.raises(ValueError,match='same device'):
        tf32.batched_matmul(a,torch.empty(3,2,device='meta'),mode='fp64')


def test_tf32_cpu_and_unsupported_dtype_have_no_fallback():
    with pytest.raises(ValueError,match='CUDA'):
        tf32.batched_matmul([torch.ones(3,4)]*2,torch.ones(4,5))
    for dtype in (torch.int64,torch.float16,torch.bfloat16):
        with pytest.raises(ValueError,match='float32/float64'):
            tf32.batched_matmul(torch.ones(3,4,dtype=dtype),torch.ones(4,5))
    assert tf32.execution_metadata()['fp64_gemm_fallback_count']==0
    for sparse in (torch.ones(3,4).to_sparse(),torch.ones(2,3,4).to_sparse()):
        with pytest.raises(ValueError,match='dense'):
            tf32.batched_matmul(sparse,torch.ones(4,5))


def test_pointer_table_uses_original_offsets_and_all_four_int64_strides():
    lefts,rights=strided_matrices()
    rows=tf32._batch_descriptor_rows(lefts,rights)
    assert len(rows)==6 and all(len(row)==2 for row in rows)
    assert rows[0]==[v.data_ptr() for v in lefts]
    assert rows[1]==[v.data_ptr() for v in rights]
    assert rows[2:]==[[v.stride(axis) for v in values]
                     for values in (lefts,rights) for axis in (0,1)]
    descriptor=torch.tensor(rows,dtype=torch.int64)
    assert descriptor.numel()*descriptor.element_size()==48*len(lefts)


@pytest.mark.parametrize('shared_side',['left','right','neither'])
def test_workspace_includes_batch_output_pointer_table_and_unique_fp64_casts(shared_side):
    lefts=[torch.ones(3,4,dtype=torch.float64) for _ in range(3)]
    rights=[torch.ones(4,5,dtype=torch.float64) for _ in range(3)]
    if shared_side=='left':lefts=[lefts[0]]*3
    if shared_side=='right':rights=[rights[0]]*3
    d=tf32._estimated_batched_workspace(lefts,rights,breakdown=True)
    cast=4*((12 if shared_side=='left' else 36)+(20 if shared_side=='right' else 60))
    assert d['cast_bytes']==cast and d['pointer_table_bytes']==48*3
    assert d['output_bytes']==4*3*3*5
    assert d['required_bytes']==cast+48*3+4*3*3*5
    assert d['layout_copy_allowance_bytes']==d['component_workspace_bytes']==0
    assert d['temporary_output_bytes']==0 and d['route']=='batched_tf32_mma'


def test_fp64_duplicate_equal_views_cast_once_and_do_not_mutate_source():
    original=torch.arange(40,dtype=torch.float64).reshape(5,8)
    lefts=(original.T,original.T)
    right=torch.ones(5,3,dtype=torch.float64)
    rights=(right,right)
    a,b=tf32._cast_batched_operands(lefts,rights)
    assert a[0] is a[1] and b[0] is b[1]
    assert a[0].dtype==b[0].dtype==torch.float32
    assert torch.equal(a[0],original.T.float())
    d=tf32._estimated_batched_workspace(lefts,rights,breakdown=True)
    assert d['cast_bytes']==4*(40+15)


def test_large_cohort_projection_admission_excludes_resident_model_and_genotype():
    class Geometry:
        dtype=torch.float32
        device=torch.device('cuda:0')
        def __init__(self,shape,pointer,strides):self.shape,self.pointer,self.strides=shape,pointer,strides
        def data_ptr(self):return self.pointer
        def stride(self,axis=None):return self.strides if axis is None else self.strides[axis]
        def numel(self):return self.shape[0]*self.shape[1]
    # No cohort-sized data is allocated by this geometry-only budget check.
    traits=32;samples=343034;covariates=23;variants=5000
    lefts=[Geometry((covariates,samples),10**12+t*10**10,(1,covariates)) for t in range(traits)]
    right=Geometry((samples,variants),10**14,(1,samples))
    d=tf32._estimated_batched_workspace(lefts,[right]*traits,breakdown=True)
    assert d['required_bytes']==4*traits*covariates*variants+48*traits
    assert d['cast_bytes']==d['layout_copy_allowance_bytes']==0
    rows=tf32._batch_descriptor_rows(lefts,[right]*traits)
    table=torch.tensor(rows,dtype=torch.int64)
    assert torch.equal(table[0],torch.tensor([v.pointer for v in lefts],dtype=torch.int64))
    assert table[0].min()>2**31 and table[1].min()>2**31


def test_failed_ptx_audit_releases_new_cast_output_and_pointer_owners(monkeypatch):
    import gc
    import weakref
    references=[]
    class Kernel:
        def __getitem__(self,grid):
            def launch(*args,**kwargs):
                references.extend(weakref.ref(value) for value in args[:2])
                return SimpleNamespace(asm={'ptx':''})
            return launch
    original_cast=tf32._cast_batched_operands
    def track_casts(*args):
        lefts,rights=original_cast(*args)
        references.extend(weakref.ref(value) for value in (*lefts,*rights))
        return lefts,rights
    monkeypatch.setattr(torch.Tensor,'is_cuda',property(lambda self:True))
    monkeypatch.setattr(tf32,'_guard_workspace',lambda *args,**kwargs:None)
    monkeypatch.setattr(tf32,'_cast_batched_operands',track_casts)
    monkeypatch.setattr(tf32,'triton',SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b))
    monkeypatch.setattr(tf32,'_batched_gemm',Kernel(),raising=False)
    monkeypatch.setattr(tf32,'_device_capability',lambda device:(8,0))
    monkeypatch.setattr(torch.cuda,'device',lambda device:nullcontext())
    lefts=[torch.ones(3,7,dtype=torch.float64) for _ in range(3)]
    right=torch.ones(7,5,dtype=torch.float64)
    with pytest.raises(RuntimeError,match='no TF32') as failure:
        tf32.batched_matmul(lefts,right)
    # Keep the exception/traceback alive deliberately, as an OOM retry owner
    # may do. New allocation owners in our frames must already be released.
    assert failure.value.__traceback__ is not None
    gc.collect()
    assert references and all(reference() is None for reference in references)
    assert tf32.execution_metadata()['tf32_actual_kernel_launch_count']==0


@pytest.mark.parametrize('left_shape,right_shape,route',[
    ((1,8),(8,3),'gemv'),((3,8),(8,1),'gemv'),((1,8),(8,1),'dot'),
    ((3,1),(1,5),'outer'),((3,0),(0,5),'empty'),((0,8),(8,5),'empty')])
def test_singleton_and_empty_batch_preserve_independent_fp32_formulas(monkeypatch,left_shape,right_shape,route):
    monkeypatch.setattr(torch.Tensor,'is_cuda',property(lambda self:True))
    monkeypatch.setattr(tf32,'_guard_workspace',lambda *args,**kwargs:None)
    a=[torch.arange(torch.tensor(left_shape).prod().item(),dtype=torch.float64).reshape(left_shape)+i for i in range(3)]
    b=torch.arange(torch.tensor(right_shape).prod().item(),dtype=torch.float64).reshape(right_shape)
    actual=tf32.batched_matmul(a,b)
    expected=torch.stack([left.float()@b.float() for left in a])
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    report=tf32.execution_metadata()
    assert report['logical_product_count']==report['trait_batched_product_count']==3
    assert report['tf32_gemm_call_count']==report['tf32_actual_kernel_launch_count']==0
    assert report['trait_batched_call_count']==1 and report['largest_trait_batch']==3
    assert report['fp32_outer_product_count']==(3 if route=='outer' else 0)
    assert report['empty_product_count']==(3 if route=='empty' else 0)
    assert report['fp32_vector_product_count']==(3 if route in ('gemv','dot') else 0)
    estimate=tf32._estimated_batched_workspace(a,[b]*3,breakdown=True)
    assert estimate['pointer_table_bytes']==0


def test_workspace_guard_rejects_before_cast_or_kernel_and_keeps_process_reserve(monkeypatch):
    monkeypatch.setattr(torch.Tensor,'is_cuda',property(lambda self:True))
    monkeypatch.setattr(tf32,'_memory_limit_bytes',100+16+240-1)
    monkeypatch.setattr(tf32,'_memory_reserve_bytes',16)
    monkeypatch.setattr(tf32,'allocator_bytes',lambda device:(100,10000))
    monkeypatch.setattr(torch.cuda,'get_allocator_backend',lambda:'native',raising=False)
    monkeypatch.setattr(torch.cuda,'device',lambda device:nullcontext())
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(10**9,80*2**30))
    def forbidden(*args):raise AssertionError('rejected batch must not cast or launch')
    monkeypatch.setattr(tf32,'_cast_batched_operands',forbidden)
    monkeypatch.setattr(tf32,'_batched_matrix_product',forbidden)
    a=[torch.ones(2,3)]*3;b=torch.ones(3,4)
    # 3*2*4*4 output bytes + 3*48 pointer bytes = 240 bytes.
    with pytest.raises(MemoryError,match='process_cap.*no precision fallback'):
        tf32.batched_matmul(a,b)
    report=tf32.execution_metadata()
    d=report['tf32_memory_guard']['last_rejection']
    assert d['required_bytes']==240 and d['available_bytes']==239
    assert d['pointer_table_bytes']==144 and d['output_bytes']==96
    assert d['reserve_bytes']==16 and d['trait_batch_size']==3
    assert report['tf32_actual_kernel_launch_count']==report['trait_batched_call_count']==0


@pytest.mark.parametrize('m,n,k',[(3,5,7),(256,257,1025)])
def test_matrix_batch_launch_uses_one_grid_and_runtime_strides(monkeypatch,m,n,k):
    launches=[]
    mma='mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 %x;'
    rna='cvt.rna.tf32.f32 %a, %b;'
    class Kernel:
        def __getitem__(self,grid):
            def launch(*args,**kwargs):
                launches.append((grid,args,kwargs))
                return SimpleNamespace(asm={'ptx':mma+'\n'+rna+'\n'+rna})
            return launch
    monkeypatch.setattr(tf32,'triton',SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b))
    monkeypatch.setattr(tf32,'_batched_gemm',Kernel(),raising=False)
    monkeypatch.setattr(tf32,'_device_capability',lambda device:(8,0))
    monkeypatch.setattr(torch.cuda,'device',lambda device:nullcontext())
    lefts=[torch.empty(k,m).T for _ in range(3)]
    right=torch.empty(n,k).T
    out=tf32._batched_matrix_product(lefts,[right]*3)
    grid,args,config=launches[0]
    bm,bn,bk,warps,stages=tf32._native_geometry(m,n,k)
    assert grid==((m+bm-1)//bm,(n+bn-1)//bn,3)
    assert args[2:]==(3,m,n,k)
    assert torch.equal(args[0],torch.tensor(tf32._batch_descriptor_rows(lefts,[right]*3),dtype=torch.int64))
    assert config==dict(BM=bm,BN=bn,BK=32,num_warps=4,num_stages=3)
    assert out.shape==(3,m,n) and out.dtype==torch.float32
    report=tf32.execution_metadata()
    assert report['tf32_gemm_call_count']==report['ptx_verified_tf32_gemm_count']==3
    assert report['tf32_actual_kernel_launch_count']==1
    assert report['output_tile_padding_elements']==3*(grid[0]*bm*grid[1]*bn-m*n)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable: native batch gate not executed')
@pytest.mark.parametrize('m,n,k',[(3,5,7),(35,67,33),(256,257,1025)])
@pytest.mark.parametrize('shared_side',['left','right','neither'])
def test_cuda_native_batch_matches_independent_ordered_rna_products(m,n,k,shared_side):
    if tf32.triton is None or torch.cuda.get_device_capability()[0]<8:
        pytest.skip('Triton and Ampere or newer CUDA hardware required')
    device=torch.device('cuda',torch.cuda.current_device())
    generator=torch.Generator(device=device).manual_seed(90213)
    lefts=[torch.randn(k,m+1,device=device,generator=generator).T[1:] for _ in range(3)]
    rights=[torch.randn(n,k+1,device=device,generator=generator).T[1:] for _ in range(3)]
    a=lefts[0] if shared_side=='left' else lefts
    b=rights[0] if shared_side=='right' else rights
    normalised_lefts,normalised_rights=tf32._normalise_batched_operands(a,b,maximum_batch_size=32)
    expected=torch.stack([tf32.matmul(x,y,mode='tf32') for x,y in zip(normalised_lefts,normalised_rights)])
    tf32.execution_metadata(reset=True)
    actual=tf32.batched_matmul(a,b)
    torch.cuda.synchronize(device)
    assert torch.equal(actual.view(torch.int32),expected.view(torch.int32))
    report=tf32.execution_metadata()
    assert report['tf32_actual_kernel_launch_count']==1
    assert report['ptx_verified_tf32_gemm_count']==3
    assert report['ptx_specialization_inspection_count']+report['ptx_verified_code_cache_hits']==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable: native RNA batch gate not executed')
def test_cuda_batched_rna_ties_are_independent_fp32_two_term_oracle():
    if tf32.triton is None or torch.cuda.get_device_capability()[0]<8:
        pytest.skip('Triton and Ampere or newer CUDA hardware required')
    device=torch.device('cuda',torch.cuda.current_device())
    source=torch.tensor([[1+2**-11,-1-2**-11],[2**-10,3+2**-11],
                         [-3,1+3*2**-11]],dtype=torch.float32)
    rounded=torch.tensor([[1+2**-10,-1-2**-10],[2**-10,3],
                          [-3,1+2**-9]],dtype=torch.float32)
    right=torch.tensor([[1.25,-2.75,3.5],[-.875,4,1.125]],dtype=torch.float32)
    lefts=[];expected=[]
    for scale in (1.,2.,-1.):
        storage=torch.empty((3,4),dtype=torch.float32,device=device)
        view=storage[:,::2];view.copy_(source*scale);lefts.append(view)
        a=rounded*scale
        expected.append(a[:,0,None]*right[0]+a[:,1,None]*right[1])
    right_storage=torch.empty((2,7),dtype=torch.float32,device=device)
    right_view=right_storage[:,1:7:2];right_view.copy_(right)
    actual=tf32.batched_matmul(lefts,right_view)
    torch.cuda.synchronize(device)
    assert torch.equal(actual.cpu().view(torch.int32),torch.stack(expected).view(torch.int32))
    assert tf32.execution_metadata()['tf32_actual_kernel_launch_count']==1
