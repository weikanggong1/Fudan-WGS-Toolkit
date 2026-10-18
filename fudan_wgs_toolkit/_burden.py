"""Native IEEE FP32 Burden products with a bounded CUDA workspace.

Score/covariance products keep native TF32. No component reconstruction or
FP64 dense multiplication is performed.
"""
from ._cuda_memory import allocator_bytes
from contextlib import contextmanager
from copy import deepcopy
import math
import os
import threading
import time
import torch

_LOCK=threading.RLock()
_METRICS=dict(selected_fp32_mm_calls=0,live_workspace_queries=0,workspace_rejections=0,
              max_estimated_new_workspace_bytes=0,max_output_bytes=0,query_host_seconds=0.,
              last_rejection=None)


def estimate_workspace(left_shape,right_shape):
    """Output plus conservative potential copies of both FP32 operands."""
    if len(left_shape)!=2 or len(right_shape)!=2 or left_shape[1]!=right_shape[0]:raise ValueError('Burden product dimensions do not match')
    if any(d<0 for d in (*left_shape,*right_shape)):raise ValueError('invalid product dimension')
    m,k=left_shape;_,n=right_shape
    return dict(output_bytes=4*m*n,layout_copy_allowance_bytes=4*(m*k+k*n),
                required_bytes=4*(m*n+m*k+k*n),cast_bytes=0,fp64_dense_bytes=0)


def guard_workspace(left,right,*,limit_bytes=None,reserve_bytes=None):
    """Inherit the configured native-product budget and live-memory reserve.

    An explicit byte limit can only tighten the configured process limit.
    The operand-copy estimate remains conservative and unchanged.
    """
    from . import tf32
    if limit_bytes is not None and (isinstance(limit_bytes,bool) or not isinstance(limit_bytes,int) or limit_bytes<0):
        raise ValueError('limit_bytes must be nonnegative integer bytes or None')
    if reserve_bytes is not None and (isinstance(reserve_bytes,bool) or not isinstance(reserve_bytes,int) or reserve_bytes<0):
        raise ValueError('reserve_bytes must be nonnegative integer bytes or None')
    limit_bytes=tf32._memory_limit_bytes if limit_bytes is None else min(limit_bytes,tf32._memory_limit_bytes)
    reserve_bytes=tf32._memory_reserve_bytes if reserve_bytes is None else reserve_bytes
    estimate=estimate_workspace(left.shape,right.shape)
    _METRICS['max_estimated_new_workspace_bytes']=max(_METRICS['max_estimated_new_workspace_bytes'],estimate['required_bytes'])
    _METRICS['max_output_bytes']=max(_METRICS['max_output_bytes'],estimate['output_bytes'])
    if not estimate['required_bytes']:return
    start=time.perf_counter()
    allocated, reserved = allocator_bytes(left.device)
    with torch.cuda.device(left.device):free,_=torch.cuda.mem_get_info()
    _METRICS['query_host_seconds']+=time.perf_counter()-start
    _METRICS['live_workspace_queries']+=1
    snapshot=tf32._product_workspace_availability(allocated=allocated,reserved=reserved,free=free,
                                                 limit=limit_bytes,reserve=reserve_bytes)
    if estimate['required_bytes']>snapshot['available_bytes']:
        _METRICS['workspace_rejections']+=1
        _METRICS['last_rejection']=dict(snapshot,**estimate,
            left_shape=list(left.shape),right_shape=list(right.shape))
        raise MemoryError('IEEE FP32 Burden workspace exceeds live/process budget; '
            f'required={estimate["required_bytes"]}, available={snapshot["available_bytes"]}, '
            f'allocated={allocated}, reserved={reserved}, CUDAfree={free}, '
            f'limit={limit_bytes}, reserve={reserve_bytes}, '
            f'cap_available={snapshot["cap_available_bytes"]}, live_available={snapshot["live_available_bytes"]}, '
            f'binding={snapshot["binding"]}, layout={estimate["layout_copy_allowance_bytes"]}, '
            f'output={estimate["output_bytes"]}; no precision/chunk fallback')


@contextmanager
def _ieee_flags():
    precision=torch.get_float32_matmul_precision();allow=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
        yield
    finally:
        torch.set_float32_matmul_precision(precision);torch.backends.cuda.matmul.allow_tf32=allow


def ieee_burden_product(left,right,*,require_cuda=True,memory_limit_gib=None):
    """Use IEEE FP32 with the shared budget; a positive explicit cap tightens it."""
    if not isinstance(left,torch.Tensor) or not isinstance(right,torch.Tensor):raise TypeError('Burden operands must be tensors')
    if left.dtype!=torch.float32 or right.dtype!=torch.float32:raise ValueError('Burden control requires float32 tensors; no dtype fallback')
    if left.layout!=torch.strided or right.layout!=torch.strided or left.ndim!=2 or right.ndim!=2:raise ValueError('Burden control requires dense matrices')
    if left.device!=right.device or (require_cuda and not left.is_cuda):raise ValueError('Burden matrices must be on the same CUDA device')
    estimate_workspace(left.shape,right.shape)
    if memory_limit_gib is not None and (isinstance(memory_limit_gib,bool)
            or not isinstance(memory_limit_gib,(int,float))
            or not math.isfinite(memory_limit_gib) or memory_limit_gib<=0):
        raise ValueError('memory_limit_gib must be a positive finite number or None')
    if left.is_cuda and os.getenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise ValueError("unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE: native Burden requires IEEE FP32")
    if left.is_cuda:
        from . import tf32
        limit_bytes=(None if memory_limit_gib is None else
                     int(min(memory_limit_gib,tf32._memory_limit_bytes/2**30)*2**30))
        guard_workspace(left,right,limit_bytes=limit_bytes)
    # Source association_test has already checked finite score/covariance/metadata;
    # keep its validation priority rather than adding per-product host checks.
    # Frozen CLI sets allow_tf32=True after an outer scope starts. Protect
    # every selected product itself so that setter cannot weaken this control.
    with _LOCK, _ieee_flags():result=torch.mm(left,right)
    _METRICS['selected_fp32_mm_calls']+=1
    return result


def execution_metadata():
    return dict(deepcopy(_METRICS), selected_backend="IEEE FP32", fp64_dense_product_count=0, tf32_component_reconstruction=False)
