"""Complete weighted spectra, retaining the original FP32 matrix products.

The sequential full path retries only a failed FP32 convergence solve, using
the same represented H32 upper triangle converted exactly to FP64. Full
spectra remain unmodified; a retry is bounded and explicitly reported.
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
              full_requested_weights=0,full_fp32_solver_calls=0,full_exact_duplicate_reused_spectra=0,
              full_fp32_solver_matrices=0,full_batch_api_calls=0,full_batch_matrices=0,
              full_batch_convergence_failures=0,full_batch_sequential_fallbacks=0,
              full_selected_solver_calls=0,full_selected_solver_matrices=0,full_selected_solver_failures=0,
              max_full_batch_new_workspace_bytes=0,full_scope_reused_spectra=0,
              full_fp32_convergence_failures=0,full_retry_attempts=0,full_retry_solver_calls=0,
              full_retry_successes=0,full_retry_failures=0,
              full_retry_workspace_rejections=0,full_retry_allocation_failures=0,
              full_nonfinite_rejections=0,
              max_full_retry_new_workspace_bytes=0,full_retry_reports=[],
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


def full_retry_workspace_unit_bytes(number_variants):
    """NEW H64, solver copy/two scratch matrices, vectors and diagnostics.

    The original H32, covariance and all-weight output are already resident.
    Four FP64 matrix units cover the converted input and a conservative dense
    solve allowance. Library scratch is estimated rather than reserved: an
    actual allocation failure still terminates the test without another retry.
    """
    m = int(number_variants)
    if m < 1:
        raise ValueError('complete spectrum retry requires a positive variant count')
    return 32*m*m + 64*m


def _mark_full_failure(error, status):
    error.weighted_spectrum_failure = status
    if hasattr(error, 'add_note'):
        error.add_note('complete weighted spectrum status: '+status)
    return error


def _is_convergence_error(error):
    return (isinstance(error, torch.linalg.LinAlgError)
            and 'failed to converge' in str(error).lower())


def _validate_retry_spectrum(matrix, spectrum, zero_modes, report):
    """Check the complete raw spectrum; never floor negatives or drop zeros.

    H32 can contain ordinary FP32 rounding error around a PSD covariance.
    The PSD/rank diagnostic tolerance is 64 eps32 times spectral scale. The
    trace check uses FP64 solver arithmetic, independent of that tolerance.
    Both tolerances are recorded; the returned eigenvalues are unchanged.
    """
    try:
        m = matrix.shape[0]
        if spectrum.ndim != 1 or spectrum.shape[0] != m:
            raise ArithmeticError('complete spectrum retry returned an incomplete spectrum')
        if not bool(torch.isfinite(spectrum).all()):
            raise ArithmeticError('complete spectrum retry returned non-finite eigenvalues')
        if bool((spectrum[1:] < spectrum[:-1]).any()):
            raise ArithmeticError('complete spectrum retry returned an unordered spectrum')
        scale = float(spectrum.abs().max())
        psd_tolerance = 64*torch.finfo(torch.float32).eps*scale
        minimum = float(spectrum[0])
        trace = float(matrix.diagonal().to(dtype=torch.float64).sum())
        trace_scale = max(float(matrix.diagonal().to(dtype=torch.float64).abs().sum()), scale)
        trace_error = float(spectrum.sum())-trace
        trace_tolerance = 64*torch.finfo(torch.float64).eps*m*trace_scale
        near_zero = int((spectrum.abs() <= psd_tolerance).sum())
        report.update(output_length=m, minimum_eigenvalue=minimum,
            psd_tolerance=psd_tolerance, trace_error=trace_error,
            trace_tolerance=trace_tolerance, exact_zero_eigenvalues=int((spectrum == 0).sum()),
            structural_zero_modes=zero_modes, numerical_zero_eigenvalues=near_zero,
            numerical_zero_tolerance=psd_tolerance, numerical_rank=m-near_zero,
            negative_eigenvalues=int((spectrum < 0).sum()))
        if minimum < -psd_tolerance:
            raise ArithmeticError('complete spectrum retry covariance is materially non-PSD')
        if abs(trace_error) > trace_tolerance:
            raise ArithmeticError('complete spectrum retry failed its trace check')
        if near_zero < zero_modes:
            raise ArithmeticError('complete spectrum retry lost structural zero modes')
    finally:
        matrix = spectrum = None


def _retry_complete_spectrum(matrix, *, workspace_limit_bytes, zero_modes):
    matrix64 = spectrum = None
    m = matrix.shape[0]
    required = full_retry_workspace_unit_bytes(m)
    if workspace_limit_bytes is None:
        from . import tf32
        workspace_limit_bytes = tf32._memory_limit_bytes
    report = dict(status='retry_workspace_check', matrix_size=m,
        required_new_workspace_bytes=required, workspace_limit_bytes=workspace_limit_bytes,
        route='torch.linalg.eigvalsh:fp64_convergence_retry',
        input_dtype='float32', retry_dtype='float64', same_H32_upper_triangle=True,
        regularization=False, eigenvalues_truncated=False)
    _METRICS['full_retry_attempts'] += 1
    recent = _METRICS['full_retry_reports']
    recent.append(report)
    del recent[:-_BATCH_SAMPLE_LIMIT]
    _METRICS['max_full_retry_new_workspace_bytes'] = max(
        _METRICS['max_full_retry_new_workspace_bytes'], required)
    try:
        # Admission precedes the dense FP64 copy. A failed live/process guard
        # propagates as MemoryError, retaining an explicit incomplete status.
        try:
            if required > workspace_limit_bytes:
                raise MemoryError('complete spectrum FP64 convergence retry requires estimated '
                    f'{required} new bytes, explicit retry budget {workspace_limit_bytes}')
            if matrix.is_cuda:
                from ._fit_structure_cache import guard_cuda_workspace
                guard_cuda_workspace(required, matrix.device, context='complete STAAR convergence retry')
        except MemoryError as error:
            _METRICS['full_retry_workspace_rejections'] += 1
            raise _mark_full_failure(error, 'retry_workspace_rejected')
        report['status'] = 'retry_copying'
        # Conversion preserves every represented H32 value exactly. UPLO=U
        # preserves the original selected triangle, including weighting order.
        matrix64 = matrix.to(dtype=torch.float64)
        report['status'] = 'retry_solving'
        _METRICS['full_retry_solver_calls'] += 1
        spectrum = torch.linalg.eigvalsh(matrix64, UPLO='U')
        report['status'] = 'retry_validating'
        _validate_retry_spectrum(matrix, spectrum, zero_modes, report)
        _pinned_statistics().record_gpu_eigen_route(matrix64)
        report['status'] = 'retry_complete'
        _METRICS['full_retry_successes'] += 1
        return spectrum
    except BaseException as error:
        if (isinstance(error, (MemoryError, torch.OutOfMemoryError)) and
                not hasattr(error, 'weighted_spectrum_failure')):
            _METRICS['full_retry_allocation_failures'] += 1
            status = 'retry_allocation_failed'
        else:
            status = getattr(error, 'weighted_spectrum_failure',
                'retry_validation_failed' if report['status'] == 'retry_validating' else 'retry_solve_failed')
        report['status'] = status
        report['failure_type'] = type(error).__name__
        _METRICS['full_retry_failures'] += 1
        _mark_full_failure(error, status)
        raise
    finally:
        matrix = matrix64 = spectrum = None


def _record_full_solve(route, count=1, *, record_backend=True):
    original = _pinned_statistics()
    _METRICS['eigen_api_calls'] += 1
    _METRICS['complete_representative_spectra'] += count
    _METRICS['full_sequential_spectra'] += count
    if record_backend:
        _METRICS['actual_backend_calls'][route] = _METRICS['actual_backend_calls'].get(route, 0)+1
        _METRICS['actual_backend_matrices'][route] = _METRICS['actual_backend_matrices'].get(route, 0)+count
    original._STATISTICS_METADATA['native_complete_spectra'] += count


def _solve_full_weight(covariance, weight, fallback_workspace_limit_bytes):
    weighted = spectrum = converted = None
    try:
        with _profile('weighted_matrix_build'):
            weighted = (covariance*weight[:, None])*weight[None, :]
        with _profile('eigvalsh'):
            try:
                _METRICS['full_fp32_solver_calls'] += 1
                _METRICS['full_fp32_solver_matrices'] += 1
                spectrum = torch.linalg.eigvalsh(weighted, UPLO='U')
            except torch.linalg.LinAlgError as error:
                if not _is_convergence_error(error):
                    raise
                _METRICS['full_fp32_convergence_failures'] += 1
                if not bool(torch.isfinite(weighted).all()):
                    _METRICS['full_nonfinite_rejections'] += 1
                    raise _mark_full_failure(ValueError('weighted H32 is non-finite; retry rejected'),
                                             'nonfinite_weighted_matrix_rejected') from error
                from ._lobpcg_stream_pool import clear_exception_frames
                clear_exception_frames(error)
                spectrum = _retry_complete_spectrum(weighted,
                    workspace_limit_bytes=fallback_workspace_limit_bytes,
                    zero_modes=int((weight == 0).sum()))
                route = 'torch.linalg.eigvalsh:fp64_convergence_retry'
            else:
                _pinned_statistics().record_gpu_eigen_route(weighted)
                route = 'torch.linalg.eigvalsh'
        if spectrum.ndim != 1 or spectrum.shape[0] != covariance.shape[0] or not bool(torch.isfinite(spectrum).all()):
            raise _mark_full_failure(ArithmeticError('complete eigensolver returned an invalid full spectrum'),
                                     'invalid_complete_spectrum')
        converted = spectrum.to(dtype=torch.float64)
        _record_batch_size(1)
        _record_full_solve(route)
        return converted
    finally:
        covariance = weight = weighted = spectrum = converted = None


def _full_batch_capacity(covariance, remaining, maximum):
    """Fresh whole-process admission, including gathered FP32 D rows."""
    m = covariance.shape[0]
    unit = full_workspace_unit_bytes(m)+4*m
    capacity = min(remaining, maximum)
    if covariance.is_cuda:
        from . import tf32
        start = time.perf_counter()
        allocated = torch.cuda.memory_allocated(covariance.device)
        reserved = torch.cuda.memory_reserved(covariance.device)
        with torch.cuda.device(covariance.device):
            free, _ = torch.cuda.mem_get_info()
        available = tf32._product_workspace_availability(allocated=allocated,
            reserved=reserved, free=free, limit=tf32._memory_limit_bytes,
            reserve=tf32._memory_reserve_bytes)['available_bytes']
        _METRICS['workspace_query_host_seconds'] += time.perf_counter()-start
        _METRICS['live_workspace_queries'] += 1
        capacity = max(1, min(capacity, available//unit))
    required = (unit*capacity if capacity > 1 else full_workspace_unit_bytes(m))
    if covariance.is_cuda:
        from ._fit_structure_cache import guard_cuda_workspace
        guard_cuda_workspace(required, covariance.device, context='complete STAAR weight batch workspace')
    _METRICS['max_full_batch_new_workspace_bytes'] = max(
        _METRICS['max_full_batch_new_workspace_bytes'], required)
    return capacity


def _fill_complete_batches(result, covariance, weights, policy, fallback_workspace_limit_bytes):
    """Complete H32 batches; an uncertain convergence batch falls back in order."""
    from ._complete_spectrum_scope import exact_weight_mapping
    rows = weighted = spectra = converted = None
    try:
        relation = exact_weight_mapping(weights, policy['relation_workspace_bytes'])
        if relation is None:
            return False
        representatives, mapping = relation
        m = covariance.shape[0]
        begin = 0
        with torch.no_grad():
            while begin < len(representatives):
                capacity = _full_batch_capacity(covariance, len(representatives)-begin,
                                                policy['weight_batch_size'])
                indices = representatives[begin:begin+capacity]
                if capacity == 1:
                    result[indices[0]].copy_(_solve_full_weight(covariance, weights[:, indices[0]],
                                                               fallback_workspace_limit_bytes))
                    begin += 1
                    continue
                rows = weights[:, indices].T
                with _profile('weighted_matrix_build'):
                    weighted = (covariance[None, :, :]*rows[:, :, None])*rows[:, None, :]
                failed = False
                selected = bool(weighted.is_cuda and 33 <= m <= 128 and _SOLVER is not None)
                backend = 'torch.linalg.eigvalsh:complete_weight_batch'
                with _profile('eigvalsh'):
                    try:
                        _METRICS['full_fp32_solver_calls'] += 1
                        _METRICS['full_fp32_solver_matrices'] += capacity
                        _METRICS['full_batch_api_calls'] += 1
                        _METRICS['full_batch_matrices'] += capacity
                        if selected:
                            _METRICS['full_selected_solver_calls'] += 1
                            _METRICS['full_selected_solver_matrices'] += capacity
                            # The established selector owns the same complete
                            # FP32 H32/UPLO=U solve and its fresh workspace checks.
                            # Only this explicit batch policy enters the route.
                            spectra, backend = _complete_eigvalsh(weighted)
                        else:
                            spectra = torch.linalg.eigvalsh(weighted, UPLO='U')
                    except torch.linalg.LinAlgError as error:
                        _METRICS['full_selected_solver_failures'] += int(selected)
                        if not _is_convergence_error(error):
                            raise
                        _METRICS['full_fp32_convergence_failures'] += 1
                        _METRICS['full_batch_convergence_failures'] += 1
                        if not bool(torch.isfinite(weighted).all()):
                            _METRICS['full_nonfinite_rejections'] += 1
                            raise _mark_full_failure(ValueError('weighted H32 is non-finite; retry rejected'),
                                                     'nonfinite_weighted_matrix_rejected') from error
                        from ._lobpcg_stream_pool import clear_exception_frames
                        clear_exception_frames(error)
                        failed = True
                    except BaseException as error:
                        if selected:
                            _METRICS['full_selected_solver_failures'] += 1
                            _mark_full_failure(error, 'selected_complete_fp32_solve_failed')
                        # Budget/API/info/nonfinite failures are not proof of a
                        # pure convergence failure and cannot select a retry.
                        raise
                if failed:
                    rows = weighted = spectra = None
                    _METRICS['full_batch_sequential_fallbacks'] += capacity
                    for index in indices:
                        if covariance.is_cuda:
                            from ._fit_structure_cache import guard_cuda_workspace
                            guard_cuda_workspace(full_workspace_unit_bytes(m), covariance.device,
                                                 context='complete STAAR weight workspace')
                        result[index].copy_(_solve_full_weight(covariance, weights[:, index],
                                                               fallback_workspace_limit_bytes))
                else:
                    if spectra.shape != (capacity, m) or not bool(torch.isfinite(spectra).all()):
                        raise _mark_full_failure(ArithmeticError('complete eigensolver returned an invalid full spectrum'),
                                                 'invalid_complete_spectrum')
                    if backend.startswith('torch.linalg.eigvalsh'):
                        _pinned_statistics().record_gpu_eigen_route(weighted)
                    converted = spectra.to(dtype=torch.float64)
                    for position, index in enumerate(indices):
                        result[index].copy_(converted[position])
                    _record_batch_size(capacity)
                    _record_full_solve(backend, capacity, record_backend=not selected)
                rows = weighted = spectra = converted = None
                begin += capacity
            for index, representative in enumerate(mapping):
                if index != representative:
                    result[index].copy_(result[representative])
                    _METRICS['full_exact_duplicate_reused_spectra'] += 1
                    _METRICS['full_sequential_spectra'] += 1
                    _pinned_statistics()._STATISTICS_METADATA['native_spectra_reused'] += 1
        return True
    finally:
        result = covariance = weights = rows = weighted = spectra = converted = None


def sequential_complete_spectra(covariance, weights, *, fallback_workspace_limit_bytes=None):
    """Independent complete H32 spectra, admitted against the whole GPU.

    Each distinct actual D column uses (C*d[:,None])*d[None,:], then the
    established Torch FP32 eigvalsh UPLO=U. Bit-identical FP32 columns reuse
    their first complete spectrum within this call, without proportional
    scaling or cross-covariance caching. Only a convergence failure admits one FP64
    complete solve of that same H32, with a NEW workspace budget and fresh
    CUDA admission. The default inherits the active process memory limit;
    an explicit byte limit can only add an admission bound. No scaling, jitter or
    matrix-product precision change is used. The old weighted batch preference
    is not a cap on one legal full matrix. All J-by-M spectrum storage is
    counted before allocation, then live admission is repeated per weight.
    """
    result = weight = weighted = spectrum = converted = weight_bits = None
    representatives = []
    try:
        if (not isinstance(covariance, torch.Tensor) or not isinstance(weights, torch.Tensor)
                or covariance.dtype != torch.float32 or weights.dtype != torch.float32
                or covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]
                or covariance.shape[0] < 1 or weights.ndim != 2
                or weights.shape[0] != covariance.shape[0] or weights.shape[1] < 1
                or covariance.device != weights.device):
            raise ValueError('complete spectra require aligned square FP32 C and actual D columns')
        if (fallback_workspace_limit_bytes is not None and
                (type(fallback_workspace_limit_bytes) is not int or fallback_workspace_limit_bytes < 0)):
            raise ValueError('complete spectrum fallback workspace limit must be None or nonnegative integer bytes')
        if not _SOLVER_LOCK.acquire(blocking=False):
            raise RuntimeError('weighted spectra belong to another active pipeline thread')
        _SOLVER_LOCK.release()
        from ._fit_structure_cache import guard_cuda_workspace
        from ._complete_spectrum_scope import current_reuse_scope, batch_policy
        m, j = weights.shape
        scope = current_reuse_scope()
        if scope is not None:
            cached = scope.lookup(covariance, weights)
            if cached is not None:
                _METRICS['full_sequential_calls'] += 1
                _METRICS['full_requested_weights'] += j
                _METRICS['full_sequential_spectra'] += j
                _METRICS['full_scope_reused_spectra'] += j
                _pinned_statistics()._STATISTICS_METADATA['native_spectra_reused'] += j
                return cached
        unit = full_workspace_unit_bytes(m)
        output_bytes = 8*j*m
        if covariance.is_cuda:
            guard_cuda_workspace(output_bytes+unit, covariance.device, context='complete STAAR output/workspace')
        if not bool(torch.isfinite(covariance).all()) or not bool(torch.isfinite(weights).all()):
            _METRICS['full_nonfinite_rejections'] += 1
            raise _mark_full_failure(ValueError('complete spectra require finite covariance and weights'),
                                     'nonfinite_input_rejected')
        _METRICS['max_full_new_workspace_bytes'] = max(
            _METRICS['max_full_new_workspace_bytes'], output_bytes+unit)
        _METRICS['full_sequential_calls'] += 1
        _METRICS['full_requested_weights'] += j
        original = _pinned_statistics()
        with torch.no_grad():
            result = torch.empty((j, m), dtype=torch.float64, device=covariance.device)
            policy = batch_policy(m)
            if policy is not None and _fill_complete_batches(result, covariance, weights,
                                                             policy, fallback_workspace_limit_bytes):
                if scope is not None:
                    scope.capture(covariance, weights, result)
                return result
            # Reinterpretation owns no additional tensor storage. Comparisons
            # return scalar booleans, not an M-by-J-by-J broadcast allocation.
            # Comparing bits also distinguishes signed zeros and 1-ULP changes.
            weight_bits = weights.view(torch.int32)
            for index in range(j):
                duplicate = next((previous for previous in representatives
                    if torch.equal(weight_bits[:, index], weight_bits[:, previous])), None)
                if duplicate is not None:
                    result[index].copy_(result[duplicate])
                    _METRICS['full_exact_duplicate_reused_spectra'] += 1
                    _METRICS['full_sequential_spectra'] += 1
                    original._STATISTICS_METADATA['native_spectra_reused'] += 1
                    continue
                if covariance.is_cuda:
                    guard_cuda_workspace(unit, covariance.device, context='complete STAAR weight workspace')
                weight = weights[:, index]
                converted = _solve_full_weight(covariance, weight, fallback_workspace_limit_bytes)
                result[index].copy_(converted)
                representatives.append(index)
                weight = weighted = spectrum = converted = None
        if scope is not None:
            scope.capture(covariance, weights, result)
        return result
    except BaseException as error:
        # A retained Torch/context traceback must not pin complete H tensors.
        from ._lobpcg_stream_pool import clear_exception_frames
        clear_exception_frames(error)
        raise
    finally:
        representatives.clear()
        covariance = weights = result = weight = weighted = spectrum = converted = weight_bits = cached = scope = None


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
    from ._complete_spectrum_scope import execution_metadata as scope_metadata
    report=dict(copy.deepcopy(_METRICS),
        batch_sizes_scope=f'last {_BATCH_SAMPLE_LIMIT} weighted-matrix batches; exact lifetime counts in batch_size_counts',
        batch_limit_bytes=WEIGHTED_BATCH_LIMIT_BYTES,
        batch_limit_scope='legacy proportional weighted batching preference; full sequential uses live/process GPU cap',
        full_sequential_policy='complete H32/eigvalsh per bit-distinct D; explicit bounded weight batch and immutable covariance scope candidates; ordinary default sequential; bounded convergence-only FP64 retry; no scaling or mask skip',
        full_duplicate_reuse_scope='same covariance and call; FP32 weight bits exactly equal; no proportional scaling, fingerprints or persistent spectrum cache',
        full_spectrum_coverage_scope='full_requested_weights counts all requested D columns; full_sequential_spectra counts solved or copied complete rows',
        full_solver_call_scope='full_fp32_solver_calls includes failed FP32 API attempts; full_retry_solver_calls counts actual FP64 API attempts',
        full_solver_matrix_scope='full_fp32_solver_matrices counts actual complete H32 inputs, including failed batch inputs; requested/spectrum rows include exact copies',
        full_selected_solver_scope='explicit FULL weight batch only, CUDA FP32 M33..128 with owned selector; above128/CPU/no selector retain Torch; backend counters count successful actual API routes once',
        full_retry_default_limit='active process allocation limit; no independent matrix-size ceiling',
        full_retry_reports_scope=f'last {_BATCH_SAMPLE_LIMIT} failed FP32 convergence solves; exact lifetime counters retained',
        full_retry_precision_scope='same represented H32 upper triangle converted to FP64 only after convergence failure; ordinary products/solves stay FP32',
        full_retry_workspace_policy='four NEW FP64 matrix units plus vectors; explicit retry budget and fresh live/process CUDA cap',
        full_retry_spectrum_policy='complete unmodified spectrum; finite/order/PSD/trace/structural-zero checks; no floor or rank truncation',
        weighted_matrix_dtype='float32',eigensolver='complete eigvalsh UPLO=U; normal FP32, convergence-only bounded FP64 retry',
        eigensolver_context=copy.deepcopy(_LAST_SOLVER_CONTEXT),
        spectrum_scalar_scale_dtype='float64',exact_proportional_relations=True,
        eigenvalues_truncated=False,fp64_dense_matrix_products=False,
        scalar_scale_reductions_changed=False,
        workspace_policy='three matrix buffers + full spectrum; fresh live/process cap before each batch',
        workspace_limitation='estimate is not CUDA reservation or measured solver scratch bound; OOM propagates',
        GPU_batch_equivalence='external original-software real-data run report; not implied by API success')
    report['complete_spectrum_optimization'] = scope_metadata(reset=reset)
    if reset:
        for key,value in _METRICS.items():_METRICS[key]=[] if isinstance(value,list) else {} if isinstance(value,dict) else 0. if isinstance(value,float) else 0
        _LAST_SOLVER_CONTEXT=None
    return report
