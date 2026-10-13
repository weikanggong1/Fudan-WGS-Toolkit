"""Native TF32 matrix products with FP32 accumulation and output.

SPDX-License-Identifier: GPL-3.0-only
Matrices use one explicit Triton TF32 MMA product. Vector products use CUDA
FP32 GEMV/dot or K=1 outer products and are reported separately. No component reconstruction,
FP64 chunk accumulation, alternate statistical formula or precision fallback.
"""
from __future__ import annotations
import math
import re
import time
import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    triton=None

_MODES=('fp64','tf32')
_calls=_products=_ptx_verified=_vector_calls=_empty_calls=_outer_calls=0
_ptx_inspections=_ptx_cache_hits=0
_capability_cache={}
_verified_compiled_cache={}
_padded_output_elements=0
_modes={}
_vector_routes={}
_memory_limit_bytes=20*2**30
_memory_reserve_bytes=256*2**20
_memory_guard_queries=_memory_guard_rejections=_memory_guard_max_workspace_bytes=0
_memory_guard_query_seconds=0.
_memory_guard_last_rejection=None
_memory_guard_checks=_memory_guard_owned_cache_passes=0
_memory_guard_total_seconds=_memory_guard_allocator_seconds=0.
_memory_guard_backend_counts={}
_memory_guard_last_owned_cache_pass=None

if triton is not None:
    @triton.jit
    def _gemm(A,B,C,M,N,K,AS0,AS1,BS0,BS1,
              BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
        rows=(tl.program_id(0)*BM+tl.arange(0,BM)).to(tl.int64)
        cols=(tl.program_id(1)*BN+tl.arange(0,BN)).to(tl.int64)
        inner=tl.arange(0,BK).to(tl.int64)
        acc=tl.full((BM,BN),0,tl.float32)
        # Dimensions/strides are runtime scalars: shape changes do not require
        # a separate fully specialized kernel for every variant-set size.
        for start in range(tl.cdiv(K,BK)):
            kk=start*BK+inner
            a=tl.load(A+rows[:,None]*AS0+kk[None,:]*AS1,
                      (rows[:,None]<M)&(kk[None,:]<K),other=0)
            b=tl.load(B+kk[:,None]*BS0+cols[None,:]*BS1,
                      (kk[:,None]<K)&(cols[None,:]<N),other=0)
            # Explicit nearest conversion prevents unrounded F32 truncation.
            # RNA is supported on all admitted devices (sm_80 and newer).
            a=tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", "=r,r",
                                       [a],dtype=tl.float32,is_pure=True,pack=1)
            b=tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", "=r,r",
                                       [b],dtype=tl.float32,is_pure=True,pack=1)
            acc=tl.dot(a,b,acc,input_precision='tf32')
        tl.store(C+rows[:,None]*N+cols[None,:],acc,
                 (rows[:,None]<M)&(cols[None,:]<N))


def validate_mode(mode):
    if mode not in _MODES:
        raise ValueError("matmul_mode must be 'fp64' or native 'tf32'; component modes are obsolete")
    return mode


def validate_split_k(split_k):
    if isinstance(split_k,bool) or not isinstance(split_k,int) or split_k!=0:
        raise ValueError('Native TF32 requires split_k=0; split/chunk reconstruction is obsolete')
    return split_k


def configure_tf32(*,memory_limit_gib=20,memory_reserve_bytes=256*2**20,split_k=0,**obsolete):
    """Configure native products; preflight is not an allocator guarantee.

    memory_limit_gib bounds estimated process allocations including the
    configured reserve. New products must leave that reserve inside both the
    process cap and the live device capacity. Only split_k=0 is accepted. Old
    component, fusion and tile controls are intentionally rejected.
    """
    global _memory_limit_bytes,_memory_reserve_bytes
    if obsolete:raise TypeError('Obsolete TF32 component/fusion/tile settings are unsupported')
    validate_split_k(split_k)
    if isinstance(memory_limit_gib,bool) or not isinstance(memory_limit_gib,(int,float)) or not math.isfinite(memory_limit_gib) or memory_limit_gib<=0:
        raise ValueError('memory_limit_gib must be a positive finite number')
    if type(memory_reserve_bytes) is not int or memory_reserve_bytes<0:
        raise ValueError('memory_reserve_bytes must be a nonnegative integer')
    _memory_limit_bytes=int(memory_limit_gib*2**30)
    _memory_reserve_bytes=memory_reserve_bytes
    return {'raw_tf32_default_split_k':0,'tf32_memory_limit_bytes':_memory_limit_bytes,
            'tf32_memory_reserve_bytes':_memory_reserve_bytes,
            'matrix_backend':'triton_explicit_native_tf32','vector_backend':'CUDA FP32 GEMV/dot'}


def _native_route(left_shape,right_shape):
    m,k=left_shape;rk,n=right_shape
    if k!=rk or min(m,k,n)<0:raise ValueError('Invalid native product shapes')
    if not m or not k or not n:return 'empty'
    if m==1 and n==1:return 'dot'
    if m==1 or n==1:return 'gemv'
    if k==1:return 'outer'
    return 'tf32_mma'


def _estimated_product_workspace(left_shape,right_shape,*,mode,left_dtype,right_dtype,breakdown=False):
    """New FP32 output/casts and conservative GEMV layout-copy allowance.

    Resident inputs are excluded. Triton accepts arbitrary strided matrices;
    CUDA vector routes reserve space for possible contiguous operand copies.
    No digit planes, FP64 residuals or component partial matrices exist.
    """
    if mode!='tf32':raise ValueError('Workspace estimate requires native tf32')
    route=_native_route(left_shape,right_shape)
    m,k=map(int,left_shape);n=int(right_shape[1]);left=m*k;right=k*n;output=m*n
    casts=0 if route=='empty' else 4*((left if left_dtype==torch.float64 else 0)+(right if right_dtype==torch.float64 else 0))
    layout=4*(left+right) if route in ('gemv','dot') else 0
    required=casts+layout+4*output
    d=dict(required_bytes=required,cast_bytes=casts,layout_copy_allowance_bytes=layout,
           output_bytes=4*output,component_workspace_bytes=0,route=route)
    return d if breakdown else required


def _product_workspace_availability(*,allocated,reserved,free,limit,reserve):
    """Leave reserve inside both the process budget and live device capacity.

    Cached unused blocks count as reusable capacity, never as live allocation
    credit. Even a physically idle device must not let a product consume the
    reserve within a smaller configured process cap.
    """
    reusable=max(int(reserved)-int(allocated),0)
    cap=int(limit)-int(allocated)-int(reserve);live=int(free)+reusable-int(reserve)
    return dict(allocated_bytes=int(allocated),reserved_bytes=int(reserved),cuda_free_bytes=int(free),
        reusable_reserved_bytes=reusable,process_limit_bytes=int(limit),reserve_bytes=int(reserve),
        cap_available_bytes=cap,live_available_bytes=live,
        binding='both' if cap==live else 'process_cap' if cap<live else 'live_free',
        available_bytes=max(0,min(cap,live)))


def _available_product_workspace(**kwargs):
    return _product_workspace_availability(**kwargs)['available_bytes']


def _allocator_backend():
    """Unknown/non-native allocators must use the fresh CUDA-free route."""
    getter=getattr(torch.cuda,'get_allocator_backend',None)
    if not callable(getter):
        getter=getattr(getattr(torch.cuda,'memory',None),'get_allocator_backend',None)
    if not callable(getter):return 'unknown'
    try:
        backend=getter()
    except Exception:
        return 'unknown'
    return backend if backend in ('native','cudaMallocAsync') else 'unknown'


def _owned_cache_sufficient(allocated,reserved,required,*,limit,reserve):
    """Byte-valued sufficient condition for the reserve-aware live/cap guard.

    For any nonnegative fresh CUDA free value, owned unused cache minus reserve
    already covers required. This is a preflight, not a fragmentation or
    concurrent-allocation guarantee; actual allocation errors still propagate.
    """
    unused=max(int(reserved)-int(allocated),0)
    return (unused>=int(required)+int(reserve)
            and int(allocated)+int(required)+int(reserve)<=int(limit))


def _guard_product_workspace(left,right,*,mode):
    global _memory_guard_queries,_memory_guard_rejections,_memory_guard_max_workspace_bytes,_memory_guard_query_seconds,_memory_guard_last_rejection
    global _memory_guard_checks,_memory_guard_owned_cache_passes,_memory_guard_total_seconds,_memory_guard_allocator_seconds,_memory_guard_last_owned_cache_pass
    estimate=_estimated_product_workspace(left.shape,right.shape,mode=mode,left_dtype=left.dtype,right_dtype=right.dtype,breakdown=True)
    required=estimate['required_bytes']
    if not required:return
    start=time.perf_counter()
    _memory_guard_checks+=1
    _memory_guard_max_workspace_bytes=max(_memory_guard_max_workspace_bytes,required)
    allocator_start=time.perf_counter()
    allocated=torch.cuda.memory_allocated(left.device);reserved=torch.cuda.memory_reserved(left.device)
    backend=_allocator_backend()
    _memory_guard_allocator_seconds+=time.perf_counter()-allocator_start
    _memory_guard_backend_counts[backend]=_memory_guard_backend_counts.get(backend,0)+1
    if backend=='native' and _owned_cache_sufficient(allocated,reserved,required,
            limit=_memory_limit_bytes,reserve=_memory_reserve_bytes):
        _memory_guard_owned_cache_passes+=1
        unused=max(int(reserved)-int(allocated),0)
        _memory_guard_last_owned_cache_pass=dict(allocated_bytes=int(allocated),reserved_bytes=int(reserved),
            required_bytes=required,reusable_reserved_bytes=unused,
            available_owned_cache_lower_bound_bytes=unused-_memory_reserve_bytes,
            process_limit_bytes=_memory_limit_bytes,reserve_bytes=_memory_reserve_bytes,
            cuda_free_bytes=None,allocator_backend=backend)
        _memory_guard_total_seconds+=time.perf_counter()-start
        return
    query_start=time.perf_counter()
    _memory_guard_queries+=1
    try:
        with torch.cuda.device(left.device):free,_=torch.cuda.mem_get_info()
    finally:
        _memory_guard_query_seconds+=time.perf_counter()-query_start
        _memory_guard_total_seconds+=time.perf_counter()-start
    snapshot=_product_workspace_availability(allocated=allocated,reserved=reserved,free=free,limit=_memory_limit_bytes,reserve=_memory_reserve_bytes)
    if required>snapshot['available_bytes']:
        _memory_guard_rejections+=1
        _memory_guard_last_rejection=dict(snapshot,**estimate,allocator_backend=backend,left_shape=list(left.shape),right_shape=list(right.shape))
        raise MemoryError(f'Native TF32 workspace requires estimated {required} new bytes, available {snapshot["available_bytes"]}; '
            f'allocated={allocated}, reserved={reserved}, CUDAfree={free}, cap_available={snapshot["cap_available_bytes"]}, '
            f'live_available={snapshot["live_available_bytes"]}, binding={snapshot["binding"]}, casts={estimate["cast_bytes"]}, '
            f'layout={estimate["layout_copy_allowance_bytes"]}, output={estimate["output_bytes"]}; '
            'preflight is not an allocator guarantee; no precision fallback')


def _verified_tf32_ptx(ptx):
    ptx=re.sub(r'/\*.*?\*/|//[^\n]*','',ptx,flags=re.S)
    if not re.search(r'mma[^;\n]*\.tf32\.tf32',ptx):
        raise RuntimeError('Native TF32 kernel has no TF32 Tensor Core MMA instruction')
    if re.search(r'mma[^;\n]*\.(?:bf16|f16)(?:\.|;)',ptx):
        raise RuntimeError('Native TF32 kernel contains an unsupported reduced-precision MMA')
    if len(re.findall(r'\bcvt\.rna\.tf32\.f32\s',ptx))<2:
        raise RuntimeError('Native TF32 kernel lacks explicit RNA conversion for both operands')


def _device_capability(device):
    # Device properties are immutable within this process. No tensors are held.
    index=device.index if device.index is not None else torch.cuda.current_device()
    if index not in _capability_cache:
        _capability_cache[index]=torch.cuda.get_device_capability(index)
    return _capability_cache[index]


def _verify_compiled_tf32(compiled):
    global _ptx_inspections,_ptx_cache_hits
    identity=id(compiled)
    if _verified_compiled_cache.get(identity) is compiled:
        _ptx_cache_hits+=1
        return
    _verified_tf32_ptx(compiled.asm.get('ptx',''))
    # Retaining the code object prevents id reuse. No operand/result is cached.
    _verified_compiled_cache[identity]=compiled
    _ptx_inspections+=1


def _native_geometry(m,n,k):
    """Fixed large-product tile validated against the original native tile.

    Small outputs/short reductions retain the original geometry. This policy
    changes only launch geometry, never K order, rounding or accumulation.
    """
    return (64,128,32,4,3) if m>=256 and n>=256 and k>=1024 else (32,64,32,4,3)


def _matrix_product(left,right):
    global _calls,_ptx_verified,_padded_output_elements
    if triton is None:raise RuntimeError('Native TF32 matrices require Triton >=3 and CUDA')
    if _device_capability(left.device)[0]<8:raise RuntimeError('Native TF32 requires Ampere or newer CUDA hardware')
    m,k=left.shape;n=right.shape[1];bm,bn,bk,warps,stages=_native_geometry(m,n,k)
    result=torch.empty((m,n),dtype=torch.float32,device=left.device)
    with torch.cuda.device(left.device):
        compiled=_gemm[(triton.cdiv(m,bm),triton.cdiv(n,bn))](left,right,result,m,n,k,*left.stride(),*right.stride(),BM=bm,BN=bn,BK=bk,num_warps=warps,num_stages=stages)
    _verify_compiled_tf32(compiled)
    _calls+=1;_ptx_verified+=1
    _padded_output_elements+=triton.cdiv(m,bm)*bm*triton.cdiv(n,bn)*bn-m*n
    return result



def _vector_product(left,right):
    """FP32 GEMV/dot formula; runtime callers supply CUDA FP32 operands."""
    if left.dtype!=torch.float32 or right.dtype!=torch.float32:
        raise ValueError('Native vector products require float32 operands')
    if left.shape[0]==1 and right.shape[1]==1:
        return torch.dot(left[0],right[:,0]).reshape(1,1)
    if right.shape[1]==1:return torch.mv(left,right[:,0])[:,None]
    return torch.mv(right.T,left[0])[None,:]


def _outer_product(left,right):
    """K=1 formula as CUDA FP32 elementwise products; no MMA claim."""
    if left.dtype!=torch.float32 or right.dtype!=torch.float32:
        raise ValueError('Native outer products require float32 operands')
    if left.shape[1]!=1 or right.shape[0]!=1:
        raise ValueError('Outer products require inner dimension one')
    return left[:,0,None]*right[None,0,:]


def matmul(left,right,*,mode='fp64'):
    """Native matrix/vector product, with an explicit historical FP64 control.

    tf32 accepts same-device CUDA float32/float64 storage; it casts inputs once
    to float32 and returns float32. GEMM is one actual TF32 MMA product with
    FP32 accumulation. Singleton outputs use ordinary CUDA FP32 GEMV/dot;
    those calls and K=1 FP32 outer products are not counted as TF32 MMA.
    Matrix inputs are explicitly rounded to TF32 nearest, ties away from zero.
    """
    global _products,_vector_calls,_empty_calls,_outer_calls
    validate_mode(mode)
    if mode=='fp64':return left@right
    if left.layout!=torch.strided or right.layout!=torch.strided:raise ValueError('Native TF32 requires dense operands')
    if left.dtype not in (torch.float32,torch.float64) or right.dtype not in (torch.float32,torch.float64):raise ValueError('Native TF32 accepts only float32/float64 input storage')
    if left.ndim not in (1,2) or right.ndim not in (1,2):raise ValueError('Native TF32 supports matrix/vector operands')
    lv,rv=left.ndim==1,right.ndim==1
    a=left[None,:] if lv else left;b=right[:,None] if rv else right
    route=_native_route(a.shape,b.shape)
    if not a.is_cuda or b.device!=a.device:raise ValueError('Native TF32 requires inputs on the same CUDA device')
    _guard_product_workspace(a,b,mode=mode)
    if route=='empty':
        result=torch.zeros((a.shape[0],b.shape[1]),dtype=torch.float32,device=a.device);_empty_calls+=1
    else:
        a=a.to(torch.float32);b=b.to(torch.float32)
        if route=='tf32_mma':result=_matrix_product(a,b)
        elif route=='outer':
            result=_outer_product(a,b);_outer_calls+=1
        else:result=_vector_product(a,b)
        if route in ('gemv','dot'):
            _vector_calls+=1;_vector_routes[route]=_vector_routes.get(route,0)+1
    _products+=1;_modes[mode]=_modes.get(mode,0)+1
    if lv:result=result.squeeze(0)
    if rv:result=result.squeeze(-1)
    return result


def execution_metadata(*,reset=False):
    """Report observed execution; configuration alone is never MMA evidence."""
    global _calls,_products,_ptx_verified,_vector_calls,_empty_calls,_outer_calls,_padded_output_elements
    global _ptx_inspections,_ptx_cache_hits
    global _memory_guard_queries,_memory_guard_rejections,_memory_guard_max_workspace_bytes,_memory_guard_query_seconds,_memory_guard_last_rejection
    global _memory_guard_checks,_memory_guard_owned_cache_passes,_memory_guard_total_seconds,_memory_guard_allocator_seconds,_memory_guard_last_owned_cache_pass
    report={'backend':'triton_explicit_native_tf32' if _calls else 'not_used',
        'logical_product_count':_products,'logical_product_modes':dict(_modes),
        'tf32_gemm_call_count':_calls,'tf32_actual_kernel_launch_count':_calls,
        'ptx_verified_tf32_gemm_count':_ptx_verified,'fp32_vector_product_count':_vector_calls,
        'fp32_vector_routes':dict(_vector_routes),'fp32_outer_product_count':_outer_calls,
        'empty_product_count':_empty_calls,
        'tf32_input_rounding':'RNA nearest, ties away from zero; explicit cvt.rna.tf32.f32 on both operands',
        'ptx_verified_rna_gemm_count':_ptx_verified,
        'ptx_specialization_inspection_count':_ptx_inspections,'ptx_verified_code_cache_hits':_ptx_cache_hits,
        'kernel_input_dtype':'float32','kernel_accumulator_dtype':'float32',
        'kernel_output_dtype':'float32','result_dtype':'float32',
        'raw_tf32_default_split_k':0,'split_k_workspace_policy':'single_native_product_no_chunk_reconstruction',
        'component_reconstruction_product_count':0,'tf32_total_logical_component_product_count':0,
        'output_tile_padding_elements':_padded_output_elements,'fp64_gemm_fallback_count':0,
        'vector_execution_policy':'CUDA FP32 GEMV/dot; no claim of TF32 Tensor Core execution',
        'outer_execution_policy':'K=1, nonsingleton output: CUDA FP32 broadcast multiply; no TF32 MMA',
        'tf32_memory_guard':{'process_allocated_limit_bytes':_memory_limit_bytes,'live_cuda_reserve_bytes':_memory_reserve_bytes,
            'total_checks':_memory_guard_checks,'owned_cache_passes':_memory_guard_owned_cache_passes,
            'cuda_free_queries':_memory_guard_queries,'workspace_rejections':_memory_guard_rejections,
            'max_estimated_new_workspace_bytes':_memory_guard_max_workspace_bytes,'query_host_wall_seconds':_memory_guard_query_seconds,
            'cuda_free_query_host_wall_seconds':_memory_guard_query_seconds,
            'allocator_snapshot_host_wall_seconds':_memory_guard_allocator_seconds,
            'guard_host_wall_seconds':_memory_guard_total_seconds,
            'allocator_backend_check_counts':dict(_memory_guard_backend_counts),
            'last_owned_cache_pass':_memory_guard_last_owned_cache_pass,
            'last_rejection':_memory_guard_last_rejection,'component_workspace_bytes':0,
            'policy':'Fresh native owned cache >= required + reserve AND allocated + required + reserve <= process cap; otherwise min(process cap minus allocated minus reserve, fresh CUDA free plus unused reserved minus reserve)',
            'counter_scope':'nonzero estimated workspace checks; owned passes + CUDA free queries = total checks after completed checks',
            'timing_scope':'query timers measure only fresh mem_get_info route, including device-context entry/exit; allocator timer measures fresh allocated/reserved/backend; guard timer includes both routes. Nested observations, not additive to end-to-end wall.',
            'limitation':'Concurrent allocations/fragmentation can still OOM; preflight is not a reservation'}}
    if reset:
        _calls=_products=_ptx_verified=_vector_calls=_empty_calls=_outer_calls=_padded_output_elements=0
        _ptx_inspections=_ptx_cache_hits=0
        _modes.clear();_vector_routes.clear()
        _memory_guard_queries=_memory_guard_rejections=_memory_guard_max_workspace_bytes=0
        _memory_guard_query_seconds=0.;_memory_guard_last_rejection=None
        _memory_guard_checks=_memory_guard_owned_cache_passes=0
        _memory_guard_total_seconds=_memory_guard_allocator_seconds=0.
        _memory_guard_last_owned_cache_pass=None;_memory_guard_backend_counts.clear()
    return report
