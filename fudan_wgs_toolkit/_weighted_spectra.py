"""Complete FP32 weighted spectra and exact proportional-weight reuse.

Original FP32 weighted matrix, exact proportional proof and full eigvalsh stay
unchanged. Real chr21 significant-P validation is described in the benchmark;
CPU contracts check interfaces separately.
"""
from contextlib import contextmanager,nullcontext
import copy
import math
import threading
import time
import torch

WEIGHTED_BATCH_LIMIT_BYTES=512*2**20
_PROFILE_HOOK=None
_SOLVER_LOCK=threading.RLock()
_SOLVER_CONTEXT_ACTIVE=False
_SOLVER=None
_LAST_SOLVER_CONTEXT=None
_BATCH_SAMPLE_LIMIT=64
_METRICS=dict(calls=0,batch_sizes=[],batch_size_counts={},eigen_api_calls=0,complete_representative_spectra=0,
              exact_relation_reused_spectra=0,live_workspace_queries=0,workspace_rejections=0,
              max_estimated_batch_workspace_bytes=0,max_weighted_tensor_bytes=0,
              full_sequential_calls=0,full_sequential_spectra=0,max_full_new_workspace_bytes=0,
              workspace_query_host_seconds=0.,actual_backend_calls={},actual_backend_matrices={})

def _record_batch_size(size):
    """Keep exact lifetime counts and only the latest bounded batch samples."""
    counts=_METRICS['batch_size_counts']
    counts[size]=counts.get(size,0)+1
    recent=_METRICS['batch_sizes']
    recent.append(size)
    del recent[:-_BATCH_SAMPLE_LIMIT]

def _make_solver(memory_limit):
    # Library discovery stays lazy until the first selected CUDA tensor call.
    from .cuda_eigen import FP32SmallSpectrumSolver
    return FP32SmallSpectrumSolver(torch_module=torch,memory_limit=memory_limit)

@contextmanager
def eigensolver_context(mode='torch',*,memory_limit=40*2**30):
    """Own one serialized weighted-spectrum solver for one pipeline run.

    This module never replaces Torch functions. Its selector may temporarily
    prefer MAGMA only after an eligible Torch FP32 convergence error, restoring
    the preference before return. The caller resolves auto/control/CPU policy.
    """
    global _SOLVER_CONTEXT_ACTIVE,_SOLVER,_LAST_SOLVER_CONTEXT
    if mode not in ('torch','cusolver_batched'):
        raise ValueError('weighted eigensolver context requires torch or cusolver_batched')
    if not _SOLVER_LOCK.acquire(blocking=False):
        raise RuntimeError('weighted eigensolver context is already running on another thread')
    if _SOLVER_CONTEXT_ACTIVE:
        _SOLVER_LOCK.release()
        raise RuntimeError('nested weighted eigensolver contexts are unsupported')
    _SOLVER_CONTEXT_ACTIVE=True
    state=dict(effective=mode,selector=None,context_setup_seconds=0.,context_cleanup_seconds=0.)
    solver=None
    try:
        execution_metadata(reset=True)
        started=time.perf_counter()
        if mode=='cusolver_batched':solver=_make_solver(memory_limit)
        _SOLVER=solver
        state['context_setup_seconds']=time.perf_counter()-started
        yield state
    finally:
        started=time.perf_counter()
        try:
            if solver is not None:solver.close()
        finally:
            try:
                state['context_cleanup_seconds']=time.perf_counter()-started
                state['selector']=None if solver is None else solver.report()
                _LAST_SOLVER_CONTEXT=copy.deepcopy(state)
            finally:
                _SOLVER=None;_SOLVER_CONTEXT_ACTIVE=False
                _SOLVER_LOCK.release()

def _complete_eigvalsh(matrices):
    acquired = False
    try:
        if not _SOLVER_LOCK.acquire(blocking=False):
            raise RuntimeError('weighted eigensolver belongs to another active pipeline thread')
        acquired = True
        if _SOLVER is None or not matrices.is_cuda:
            spectra=torch.linalg.eigvalsh(matrices,UPLO='U')
            backend='torch.linalg.eigvalsh'
        else:
            spectra=_SOLVER.eigvalsh(matrices,UPLO='U')
            backend=_SOLVER.last_execution_backend
            if backend not in ('torch.linalg.eigvalsh','torch.linalg.eigvalsh:magma_retry',
                               'cusolverDnSsyevjBatched'):
                raise RuntimeError('successful weighted solve lacks its actual API route')
        _METRICS['actual_backend_calls'][backend]=_METRICS['actual_backend_calls'].get(backend,0)+1
        count=1 if matrices.ndim==2 else matrices.shape[0]
        _METRICS['actual_backend_matrices'][backend]=_METRICS['actual_backend_matrices'].get(backend,0)+count
        return spectra,backend
    finally:
        # A caller may keep the exception/traceback for its private audit.
        # Drop our argument and output owners before that traceback escapes.
        matrices = spectra = None
        if acquired:
            _SOLVER_LOCK.release()

def _pinned_statistics():
    from . import statistics
    return statistics

def _profile(label):
    return _PROFILE_HOOK(label) if _PROFILE_HOOK is not None else nullcontext()

@contextmanager
def profile_stages(hook):
    global _PROFILE_HOOK
    old=_PROFILE_HOOK;_PROFILE_HOOK=hook
    try:yield
    finally:_PROFILE_HOOK=old

def workspace_unit_bytes(covariance):
    # Three weighted-matrix-sized buffers plus one full FP32 spectrum output.
    # CUDA allocator cap/live margin remain final controls for solver scratch.
    return 3*covariance.numel()*covariance.element_size()+4*covariance.shape[0]

def _batch_capacity(covariance,remaining):
    try:
        budget=WEIGHTED_BATCH_LIMIT_BYTES
        if covariance.is_cuda:
            from fudan_wgs_toolkit import tf32
            start=time.perf_counter()
            allocated=torch.cuda.memory_allocated(covariance.device)
            reserved=torch.cuda.memory_reserved(covariance.device)
            with torch.cuda.device(covariance.device):free,_=torch.cuda.mem_get_info()
            _METRICS['workspace_query_host_seconds']+=time.perf_counter()-start
            _METRICS['live_workspace_queries']+=1
            available=tf32._product_workspace_availability(allocated=allocated,reserved=reserved,free=free,
                limit=tf32._memory_limit_bytes,reserve=tf32._memory_reserve_bytes)['available_bytes']
            budget=min(budget,available)
        unit=workspace_unit_bytes(covariance)
        capacity=min(remaining,budget//max(unit,1))
        if capacity<1:
            _METRICS['workspace_rejections']+=1
            raise MemoryError('full FP32 weighted eigensolve batch cannot fit fresh live/process workspace; no truncation/precision retry')
        _METRICS['max_estimated_batch_workspace_bytes']=max(_METRICS['max_estimated_batch_workspace_bytes'],capacity*unit)
        _METRICS['max_weighted_tensor_bytes']=max(_METRICS['max_weighted_tensor_bytes'],capacity*covariance.numel()*covariance.element_size())
        return capacity
    finally:
        covariance = None

def scale_exact_spectra(solved,mapping,exact,pivots):
    """Vectorize original independent FP64 scalars; no changed reductions."""
    representative_rows=torch.tensor(mapping,dtype=torch.int64,device=exact.device)
    representative_pivots=pivots[representative_rows]
    numerators=exact.gather(1,representative_pivots[:,None])[:,0]
    denominators=exact[representative_rows,representative_pivots]
    scales=numerators/denominators
    full_fp32=torch.stack([solved[i] for i in mapping])
    return full_fp32.double()*scales.square()[:,None]

def full_workspace_unit_bytes(number_variants):
    """Estimated NEW sequential H32/eigensolver buffers, not a GPU cap.

    Four FP32 matrix units conservatively cover left-weighting, its final H,
    the solver input copy and library workspace. Spectrum output/conversion
    adds one FP32 and one FP64 vector. C and the final all-weight spectrum
    output are already resident when this per-weight allowance is admitted.
    Library scratch is estimated; real allocation/OOM remains a hard failure.
    """
    m = int(number_variants)
    if m < 1:
        raise ValueError('complete spectrum requires a positive variant count')
    return 16*m*m + 12*m


def sequential_complete_spectra(covariance, weights):
    """Independent complete H32 spectra, admitted against the whole GPU.

    Each actual D column uses (C*d[:,None])*d[None,:], then the established
    Torch eigvalsh UPLO=U. No proportional scaling, truncation, full H64 or
    alternative eigensolver is used. The old 512 MiB weighted batch preference
    is not a cap on one legal full matrix. All J-by-M spectrum storage is
    counted before allocation, then live admission is repeated per weight.
    """
    result = weight = weighted = spectrum = converted = None
    try:
        if (not isinstance(covariance, torch.Tensor) or not isinstance(weights, torch.Tensor)
                or covariance.dtype != torch.float32 or weights.dtype != torch.float32
                or covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]
                or covariance.shape[0] < 1 or weights.ndim != 2
                or weights.shape[0] != covariance.shape[0] or weights.shape[1] < 1
                or covariance.device != weights.device):
            raise ValueError('complete spectra require aligned square FP32 C and actual D columns')
        if not _SOLVER_LOCK.acquire(blocking=False):
            raise RuntimeError('weighted spectra belong to another active pipeline thread')
        _SOLVER_LOCK.release()
        from ._fit_structure_cache import guard_cuda_workspace
        m, j = weights.shape
        unit = full_workspace_unit_bytes(m)
        output_bytes = 8*j*m
        if covariance.is_cuda:
            guard_cuda_workspace(output_bytes+unit, covariance.device, context='complete STAAR output/workspace')
        _METRICS['max_full_new_workspace_bytes'] = max(
            _METRICS['max_full_new_workspace_bytes'], output_bytes+unit)
        _METRICS['full_sequential_calls'] += 1
        original = _pinned_statistics()
        with torch.no_grad():
            result = torch.empty((j, m), dtype=torch.float64, device=covariance.device)
            for index in range(j):
                if covariance.is_cuda:
                    guard_cuda_workspace(unit, covariance.device, context='complete STAAR weight workspace')
                weight = weights[:, index]
                with _profile('weighted_matrix_build'):
                    weighted = (covariance*weight[:, None])*weight[None, :]
                with _profile('eigvalsh'):
                    spectrum = torch.linalg.eigvalsh(weighted, UPLO='U')
                original.record_gpu_eigen_route(weighted)
                converted = spectrum.to(dtype=torch.float64)
                result[index].copy_(converted)
                _record_batch_size(1)
                _METRICS['eigen_api_calls'] += 1
                _METRICS['complete_representative_spectra'] += 1
                _METRICS['full_sequential_spectra'] += 1
                route = 'torch.linalg.eigvalsh'
                _METRICS['actual_backend_calls'][route] = _METRICS['actual_backend_calls'].get(route, 0)+1
                _METRICS['actual_backend_matrices'][route] = _METRICS['actual_backend_matrices'].get(route, 0)+1
                original._STATISTICS_METADATA['native_complete_spectra'] += 1
                weight = weighted = spectrum = converted = None
        return result
    except BaseException as error:
        # A retained Torch/context traceback must not pin complete H tensors.
        from ._lobpcg_stream_pool import clear_exception_frames
        clear_exception_frames(error)
        raise
    finally:
        covariance = weights = result = weight = weighted = spectrum = converted = None


def native_weighted_spectra(covariance, weights):
    """Complete FP32 spectra; reuse only exact proportional input weights.

    Products of two FP32 numbers are represented exactly in FP64 (at most
    48 significand bits). Equal cross-products therefore prove proportionality,
    without a tolerance or annotation grouping approximation. Dense weighted
    matrices and eigvalsh remain FP32. Only spectrum scalar scaling uses FP64.
    """
    solved = {}
    try:
        if not _SOLVER_LOCK.acquire(blocking=False):
            raise RuntimeError('weighted spectra belong to another active pipeline thread')
        _SOLVER_LOCK.release()
        original=_pinned_statistics()
        _native_weight_relations=original._native_weight_relations
        _STATISTICS_METADATA=original._STATISTICS_METADATA
        record_gpu_eigen_route=original.record_gpu_eigen_route
        _METRICS['calls']+=1
        rows = weights.T.contiguous()
        with _profile('weight_relations'):
            exact, pivots, related = _native_weight_relations(rows)
        representatives, mapping = [], []
        for j in range(len(rows)):
            found = next((i for i in representatives if related[i][j]), None)
            if found is None:
                representatives.append(j); found = j
            mapping.append(found)
        # Bound only weighted-matrix materialization; eigensolver workspace is
        # additionally covered by the pipeline's process allocation limit.
        if covariance.dtype!=torch.float32 or covariance.ndim!=2 or covariance.shape[0]!=covariance.shape[1] or covariance.shape[0]!=rows.shape[1] or covariance.device!=weights.device:
            raise ValueError('weighted spectrum candidate requires aligned square FP32 covariance')
        per_matrix = covariance.numel() * covariance.element_size()
        start=0
        while start<len(representatives):
            batch_size=_batch_capacity(covariance,len(representatives)-start)
            indices = representatives[start:start + batch_size]
            w = rows[indices]
            with _profile('weighted_matrix_build'):
                matrices = covariance[None, :, :] * w[:, :, None] * w[:, None, :]
            with _profile('eigvalsh'):
                spectra,backend = _complete_eigvalsh(matrices)
            if backend in ('torch.linalg.eigvalsh','torch.linalg.eigvalsh:magma_retry'):
                record_gpu_eigen_route(matrices)
            _record_batch_size(len(indices))
            _METRICS['eigen_api_calls']+=1
            for index, spectrum in zip(indices, spectra):
                solved[index] = spectrum
            start+=len(indices)
        _STATISTICS_METADATA["native_complete_spectra"] += len(representatives)
        _STATISTICS_METADATA["native_spectra_reused"] += len(rows) - len(representatives)
        _METRICS['complete_representative_spectra']+=len(representatives)
        _METRICS['exact_relation_reused_spectra']+=len(rows)-len(representatives)
        with _profile('spectrum_scaling'):
            return scale_exact_spectra(solved,mapping,exact,pivots)
    finally:
        # Keep audited exceptions without retaining this caller's inputs or
        # complete weighted matrices after the failed stage has unwound.
        solved.clear()
        covariance = weights = rows = exact = pivots = related = None
        w = matrices = spectra = spectrum = None


def execution_metadata(*,reset=False):
    global _LAST_SOLVER_CONTEXT
    report=dict(copy.deepcopy(_METRICS),
        batch_sizes_scope=f'last {_BATCH_SAMPLE_LIMIT} weighted-matrix batches; exact lifetime counts in batch_size_counts',
        batch_limit_bytes=WEIGHTED_BATCH_LIMIT_BYTES,
        batch_limit_scope='legacy proportional weighted batching preference; full sequential uses live/process GPU cap',
        full_sequential_policy='independent complete native H32/eigvalsh; no spectrum scaling or M threshold',
        weighted_matrix_dtype='float32',eigensolver='explicit complete FP32 eigvalsh APIs, UPLO=U',
        eigensolver_context=copy.deepcopy(_LAST_SOLVER_CONTEXT),
        spectrum_scalar_scale_dtype='float64',exact_proportional_relations=True,
        eigenvalues_truncated=False,fp64_dense_matrix_products=False,
        scalar_scale_reductions_changed=False,
        workspace_policy='three matrix buffers + full spectrum; fresh live/process cap before each batch',
        workspace_limitation='estimate is not CUDA reservation or measured solver scratch bound; OOM propagates',
        GPU_batch_equivalence='external original-software real-data run report; not implied by API success')
    if reset:
        for key,value in _METRICS.items():_METRICS[key]=[] if isinstance(value,list) else {} if isinstance(value,dict) else 0. if isinstance(value,float) else 0
        _LAST_SOLVER_CONTEXT=None
    return report
