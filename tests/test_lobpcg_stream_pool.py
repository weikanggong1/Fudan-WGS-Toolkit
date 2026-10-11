"""Independent-matrix algorithm/component contracts, not real WGS benchmarks."""
import gc
import importlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import weakref

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit import _lobpcg_stream_pool as candidate

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')


def run_scoped(*args, **kwargs):
    try:
        if kwargs.get('worker_pool') is not None or not torch.cuda.is_available() or not args[0][0].is_cuda:
            return candidate.independent_lobpcg(*args, **kwargs)
        with candidate.LOBPCGWorkerPool(2) as pool:
            return candidate.independent_lobpcg(*args, **kwargs, worker_pool=pool)
    finally:
        args = kwargs = pool = None



def matrices(device, *, m=193, k=8, traits=2):
    g = torch.Generator(device=device).manual_seed(123)
    matrices, starts = [], []
    for index in range(traits):
        u = torch.randn((m, 48), generator=g, device=device)
        matrix = (u@u.T)/m
        matrix.diagonal().add_(torch.linspace(.1, .2, m, device=device))
        matrices.append(matrix)
        starts.append(torch.randn((m, k), generator=g, device=device))
    return matrices, starts


def histories(width):
    records = [[] for _ in range(width)]
    def tracker(index):
        def call(worker):
            records[index].append((worker.ivars['istep'], worker.ivars['converged_count'],
                worker.ivars['iterations_left']))
        return call
    return records, [tracker(i) for i in range(width)]


def test_budget_does_not_allocate_full_operator_batch():
    result = candidate.workspace_estimate(9950, 512, 2)
    assert result['whole_operator_stack_bytes'] == 0
    assert result['maximum_projected_size'] == 1536
    assert result['required_bytes'] == 2*(48*9950*512+400*512**2+16*512)
    for params in ((1, 2, 2), (100, True, 2), (100, 2, -1)):
        with pytest.raises(ValueError):
            candidate.workspace_estimate(*params)


def test_cpu_falls_back_to_original_public_api_exactly():
    a, x = matrices('cpu')
    expected = [torch.lobpcg(matrix, k=8, X=start, niter=7, tol=1e-5)
                for matrix, start in zip(a, x)]
    actual, meta = run_scoped(a, x, k=8, niter=7)
    for left, right in zip(actual, expected):
        for l, r in zip(left, right):
            torch.testing.assert_close(l, r, rtol=0, atol=0)
    assert meta['actual_routes'] == ['original_torch_sequential']*2


@CUDA
@pytest.mark.parametrize('traits', [2, 4])
@pytest.mark.parametrize('niter', [1, 9, 25])
def test_installed_worker_matrices_subspaces_and_stopping_match(traits, niter):
    a, x = matrices('cuda', traits=traits)
    original, original_trackers = histories(traits)
    expected = [torch.lobpcg(matrix, k=8, X=start, niter=niter, tol=1e-5,
                           tracker=tracker)
                for matrix, start, tracker in zip(a, x, original_trackers)]
    traced, trackers = histories(traits)
    torch_module = importlib.import_module('torch._lobpcg')
    original_symeig = torch_module._utils.symeig
    original_class_tracker = torch_module.LOBPCG.call_tracker
    before = [start.clone() for start in x]
    actual, meta = run_scoped(a, x, k=8, niter=niter, trackers=trackers)
    assert meta['no_torch_global_patch']
    assert torch_module._utils.symeig is original_symeig
    assert torch_module.LOBPCG.call_tracker is original_class_tracker
    assert traced == original
    assert sum(tile.get('grouped_eigh_api_calls', 0) for tile in meta['tiles']) == 0
    for matrix, (e, q), (old_e, old_q), before_x, after_x in zip(a, actual, expected, before, x):
        torch.testing.assert_close(after_x, before_x, rtol=0, atol=0)
        torch.testing.assert_close(e, old_e, rtol=1e-4, atol=3e-5)
        # Subspace signs may differ; compare actual projectors and eigenpairs.
        torch.testing.assert_close(q@q.T, old_q@old_q.T, rtol=1e-3, atol=2e-4)
        residual = torch.linalg.vector_norm(matrix@q-q*e)/torch.linalg.vector_norm(matrix)
        old_residual = torch.linalg.vector_norm(matrix@old_q-old_q*old_e)/torch.linalg.vector_norm(matrix)
        assert abs(float(residual-old_residual)) < 1e-4


@CUDA
def test_precision_uses_original_fp64_refinement_and_final_log_probability():
    from fudan_wgs_toolkit._fastskat_numerics import refine_spectrum_and_moments
    from fudan_wgs_toolkit.precision_audit import explicit_fp64_spectral_refinement
    from fudan_wgs_toolkit.statistics import _fastskat_hybrid_sf_tensor
    a, x = matrices('cuda', m=769, k=32, traits=2)
    expected = [torch.lobpcg(matrix, k=32, X=start, niter=40, tol=1e-5)
                for matrix, start in zip(a, x)]
    actual, meta = run_scoped(a, x, k=32, niter=40)
    assert meta['original_projected_eigh_per_worker']
    for matrix, pair, old_pair in zip(a, actual, expected):
        with explicit_fp64_spectral_refinement():
            new = refine_spectrum_and_moments(matrix, pair[1])
            old = refine_spectrum_and_moments(matrix, old_pair[1])
        torch.testing.assert_close(new['top'], old['top'], rtol=1e-5, atol=2e-6)
        for multiplier in (.7, 1., 2., 5.):
            statistic = old['trace']*multiplier
            new_diag, old_diag = {}, {}
            p = _fastskat_hybrid_sf_tensor(statistic, new['top'], new['residual_mean'],
                new['residual_second'], diagnostics=new_diag)
            reference = _fastskat_hybrid_sf_tensor(statistic, old['top'], old['residual_mean'],
                old['residual_second'], diagnostics=old_diag)
            if bool((p > 0) & (reference > 0)):
                assert abs(float(-p.log10()+reference.log10())) < .001
            else:
                assert 'stable_log_probability' in new_diag
                assert abs(new_diag['stable_log_probability']-old_diag['stable_log_probability'])/np.log(10) < .001


@CUDA
def test_pressure_runs_original_sequential_without_operator_stack(monkeypatch):
    a, x = matrices('cuda')
    monkeypatch.setattr(candidate, 'available_cuda_bytes', lambda device: 0)
    actual, meta = run_scoped(a, x, k=8, niter=3)
    assert meta['actual_routes'] == ['original_torch_sequential']*2
    assert all(pair[0].shape == (8,) for pair in actual)


@CUDA
def test_twenty_four_warm_tiles_release_outputs_workers_and_scratch():
    a, x = matrices('cuda', m=97, k=4)
    with candidate.LOBPCGWorkerPool(2) as pool:
        for _ in range(3):
            out, _ = run_scoped(a, x, k=4, niter=3, worker_pool=pool)
            del out
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        for _ in range(24):
            out, meta = run_scoped(a, x, k=4, niter=3, worker_pool=pool)
            assert len(out) == 2
            assert meta['worker_pool']['persistent_threads']
            assert meta['worker_pool']['owns_cuda_streams']
            assert meta['worker_pool']['stream_count'] == 2
            assert meta['worker_pool']['observed_thread_count'] == 2
            del out
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
        assert sum(t.name.startswith('wgs-lobpcg') for t in threading.enumerate()) == 2
    assert pool.metadata()['closed']
    assert not any(t.name.startswith('wgs-lobpcg') for t in threading.enumerate())
    with pytest.raises(RuntimeError, match='closed'):
        pool.submit_slot(0, lambda: None)


@CUDA
@pytest.mark.parametrize('k,m', [(64, 1537), (128, 3073)])
def test_large_projected_batch_preserves_refined_spectrum_and_tail(k, m):
    from fudan_wgs_toolkit._fastskat_numerics import refine_spectrum_and_moments
    from fudan_wgs_toolkit.precision_audit import explicit_fp64_spectral_refinement
    from fudan_wgs_toolkit.statistics import _fastskat_hybrid_sf_tensor
    a, x = matrices('cuda', m=m, k=k)
    expected = [torch.lobpcg(matrix, k=k, X=start, niter=25, tol=1e-5)
                for matrix, start in zip(a, x)]
    actual, meta = run_scoped(a, x, k=k, niter=25)
    assert meta['original_projected_eigh_per_worker']
    for matrix, pair, reference_pair in zip(a, actual, expected):
        with explicit_fp64_spectral_refinement():
            new = refine_spectrum_and_moments(matrix, pair[1])
            old = refine_spectrum_and_moments(matrix, reference_pair[1])
        for multiplier in (.7, 1., 2., 5.):
            q = old['trace'] * multiplier
            p = _fastskat_hybrid_sf_tensor(q, new['top'], new['residual_mean'], new['residual_second'])
            old_p = _fastskat_hybrid_sf_tensor(q, old['top'], old['residual_mean'], old['residual_second'])
            assert bool((p > 0) & (old_p > 0))
            assert abs(float(-p.log10()+old_p.log10())) < .001


@CUDA
def test_held_future_exception_releases_matrix_owners_and_pool_threads():
    a, x = matrices('cuda')
    refs = [weakref.ref(tensor) for tensor in [*a, *x]]
    pool = candidate.LOBPCGWorkerPool(2)
    held = None
    def failing_owner(matrices, starts):
        try:
            run_scoped(matrices, starts, k=8, niter=3, worker_pool=pool,
                trackers=[lambda worker: (_ for _ in ()).throw(RuntimeError('injected tracker')), None])
        finally:
            matrices = starts = None
    with ThreadPoolExecutor(max_workers=1) as outer:
        held = outer.submit(failing_owner, a, x)
        assert isinstance(held.exception(timeout=30), RuntimeError)
    pool.close()
    del a, x
    gc.collect()
    # The completed Future and its traceback remain deliberately held.
    assert isinstance(held.exception(), RuntimeError)
    assert all(ref() is None for ref in refs)


@CUDA
def test_worker_preserves_grad_inference_and_contextvar_state():
    context = ContextVar('lobpcg-scoped-context', default=None)
    seen = []
    token = context.set('owning-pipeline')
    try:
        with torch.inference_mode(), torch.no_grad():
            a, x = matrices('cuda')
            expected = [torch.lobpcg(matrix, k=8, X=start, niter=3, tol=1e-5)
                        for matrix, start in zip(a, x)]
            def tracker(worker):
                seen.append((context.get(), torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
            actual, _ = run_scoped(a, x, k=8, niter=3,
                                                   trackers=[tracker, tracker])
            for new, old in zip(actual, expected):
                for tensor, reference in zip(new, old):
                    torch.testing.assert_close(tensor, reference, rtol=0, atol=0)
        assert seen and all(value == ('owning-pipeline', False, True) for value in seen)
    finally:
        context.reset(token)


@CUDA
def test_held_eigh_exception_releases_projected_and_worker_owners(monkeypatch):
    a, x = matrices('cuda')
    refs = [weakref.ref(tensor) for tensor in [*a, *x]]
    original = torch.linalg.eigh
    def failure(matrix, *args, **kwargs):
        raise RuntimeError('injected projected eigensolve failure')
    monkeypatch.setattr(torch.linalg, 'eigh', failure)
    held = None
    try:
        run_scoped(a, x, k=8, niter=3)
    except RuntimeError as error:
        held = error
    del a, x
    gc.collect()
    assert held is not None
    assert all(ref() is None for ref in refs)


@CUDA
def test_unscoped_call_never_creates_cuda_streams_or_control_threads(monkeypatch):
    a, x = matrices('cuda')
    expected = [torch.lobpcg(matrix, k=8, X=start, niter=3, tol=1e-5)
                for matrix, start in zip(a, x)]
    def forbidden(*args, **kwargs):
        raise AssertionError('unscoped factor created a new CUDA stream')
    monkeypatch.setattr(torch.cuda, 'Stream', forbidden)
    actual, meta = candidate.independent_lobpcg(a, x, k=8, niter=3)
    assert meta['actual_routes'] == ['original_torch_sequential']*2
    for pair, old in zip(actual, expected):
        for tensor, reference in zip(pair, old):
            torch.testing.assert_close(tensor, reference, rtol=0, atol=0)


@CUDA
def test_first_native_handles_are_new_and_fixed_streams_reused():
    a, x = matrices('cuda', m=97, k=4)
    with candidate.LOBPCGWorkerPool(2) as pool:
        assert pool.initial_native_workspace_estimate(a[0].device) == 64 << 20
        before = torch.cuda.memory_allocated()
        first, meta = candidate.independent_lobpcg(a, x, k=4, niter=3, worker_pool=pool)
        del first
        torch.cuda.synchronize()
        identities = tuple(stream.cuda_stream for stream in pool.streams(a[0].device))
        assert pool.initial_native_workspace_estimate(a[0].device) == 0
        for _ in range(3):
            out, _ = candidate.independent_lobpcg(a, x, k=4, niter=3, worker_pool=pool)
            del out
        assert tuple(stream.cuda_stream for stream in pool.streams(a[0].device)) == identities
        assert torch.cuda.memory_allocated()-before < 64 << 20
    assert pool.metadata()['stream_count'] == 0
    assert pool.metadata()['native_handle_cache_may_survive_owner_close']
