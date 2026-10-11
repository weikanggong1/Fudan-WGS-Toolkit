"""Weight lookahead, ownership and scientific control, not WGS benchmarks."""
import ast
import gc
import os
from pathlib import Path
import weakref

import pytest
import torch

from fudan_wgs_toolkit import _fastskat_lobpcg_tiles as tiles
from fudan_wgs_toolkit import statistics
from fudan_wgs_toolkit._lobpcg_stream_pool import LOBPCGWorkerPool

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')


def original_function():
    source = Path(os.environ['WGS_LOBPCG_ORIGINAL_STATS']).read_text()
    tree = ast.parse(source)
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_fastskat_hybrid_pvalues')
    namespace = dict(statistics.__dict__)
    exec(compile(ast.Module(body=[original], type_ignores=[]), '<bound original FastSKAT>', 'exec'), namespace)
    return namespace['_fastskat_hybrid_pvalues']


def inputs(m=193, k=8, width=2):
    rng = torch.Generator(device='cuda').manual_seed(125)
    u = torch.randn((m, 48), generator=rng, device='cuda')
    a = u@u.T/m
    a.diagonal().add_(torch.linspace(.1, .2, m, device='cuda'))
    w = torch.rand((m, width), generator=rng, device='cuda')*.2+.9
    q = torch.stack([(a.diagonal()*w[:, i].square()).sum()*2 for i in range(width)])
    return a, w, q


def test_budget_covers_lookahead_creation_before_resident_solver_call():
    value = tiles.tile_peak_estimate(9950, 512, 2, native_new_bytes=64<<20)
    assert value['creation_new_bytes'] == 3*4*9950**2+2*4*9950*512
    assert value['required_new_bytes'] >= value['creation_new_bytes']+(64<<20)
    assert value['whole_operator_stack_bytes'] == 0
    assert value['separate_dense_weighted_operators'] == 2
    extra = tiles.additional_workspace_estimate(9950, 512, 2, native_new_bytes=64<<20)
    assert extra['additional_new_bytes'] == max(0, value['required_new_bytes']-(8*9950**2+24*9950*512))


def test_context_scopes_are_lazy_nested_and_restore_after_exception():
    assert tiles.current_lobpcg_context() == (None, None)
    with tiles.lobpcg_pool_scope(max_workers=2) as outer:
        assert outer.pool is None
        assert tiles.current_lobpcg_context()[0] is outer
        with pytest.raises(RuntimeError, match='injected scope'):
            with tiles.lobpcg_pool_scope(max_workers=1) as inner:
                with tiles.lobpcg_workspace_scope(maximum_new_bytes=123, callback=lambda *a, **kw: False) as settings:
                    assert tiles.current_lobpcg_context() == (inner, settings)
                    raise RuntimeError('injected scope')
        assert inner.closed and inner.pool is None
        assert settings.callback is None
        assert tiles.current_lobpcg_context() == (outer, None)
    assert outer.closed and tiles.current_lobpcg_context() == (None, None)


@CUDA
def test_scoped_pool_is_reused_and_scoped_bank_rejection_falls_back():
    a, w, q = inputs()
    old = original_function()(q, a, w, rank=8, threshold=100)
    with tiles.lobpcg_pool_scope(max_workers=2) as owner:
        first = statistics._fastskat_hybrid_pvalues(q, a, w, rank=8, threshold=100)
        pool = owner.pool
        assert pool is not None
        with tiles.lobpcg_workspace_scope(callback=lambda *a, **kw: False):
            second = statistics._fastskat_hybrid_pvalues(q, a, w, rank=8, threshold=100)
        assert owner.pool is pool and not pool.metadata()['closed']
        torch.testing.assert_close(second, old, rtol=0, atol=0)
        torch.testing.assert_close(first, old, rtol=0, atol=0)
    assert pool.metadata()['closed']
    assert tiles.current_lobpcg_context() == (None, None)


@CUDA
@pytest.mark.parametrize('m,k', [(193, 8), (769, 32)])
def test_final_native_refinement_tail_matches_original_function(m, k):
    a, w, q = inputs(m, k)
    original = original_function()
    old = original(q, a, w, rank=k, threshold=100)
    with LOBPCGWorkerPool(2) as pool:
        new = statistics._fastskat_hybrid_pvalues(q, a, w, rank=k, threshold=100,
                                                _lobpcg_worker_pool=pool)
    assert bool(((new > 0) & (old > 0)).all())
    assert float((-new.log10()+old.log10()).abs().max()) < .001


@CUDA
def test_start_vectors_follow_original_generator_weight_order():
    a, w, q = inputs(width=4)
    expected_rng = torch.Generator(device='cuda').manual_seed(1729)
    expected = [torch.randn((193, 8), device='cuda', generator=expected_rng) for _ in range(4)]
    actual_rng = torch.Generator(device='cuda').manual_seed(1729)
    owner = tiles.FastSKATPairTiles(a, w, actual_rng, 8)
    try:
        for i in range(4):
            entry = owner.take(i)
            torch.testing.assert_close(entry[2], expected[i], rtol=0, atol=0)
            del entry
        torch.testing.assert_close(actual_rng.get_state(), expected_rng.get_state(), rtol=0, atol=0)
    finally:
        owner.close()


@CUDA
def test_pressure_disables_lookahead_and_preserves_original_result(monkeypatch):
    a, w, q = inputs()
    old = original_function()(q, a, w, rank=8, threshold=100)
    monkeypatch.setattr(tiles, 'available_cuda_bytes', lambda device: 0)
    new = statistics._fastskat_hybrid_pvalues(q, a, w, rank=8, threshold=100)
    torch.testing.assert_close(new, old, rtol=0, atol=0)


@CUDA
def test_batch_failure_replays_original_order_and_held_refinement_error_releases_inputs(monkeypatch):
    from fudan_wgs_toolkit import _fastskat_numerics
    a, w, q = inputs()
    refs = [weakref.ref(value) for value in (a, w, q)]
    def batch_failure(*args, **kwargs):
        raise RuntimeError('injected later batch solve failure')
    def refine_failure(weighted, basis, **kwargs):
        raise ArithmeticError('original first-weight refinement failure')
    monkeypatch.setattr(tiles, 'independent_lobpcg', batch_failure)
    monkeypatch.setattr(_fastskat_numerics, 'refine_spectrum_and_moments', refine_failure)
    held = None
    try:
        with LOBPCGWorkerPool(2) as pool:
            statistics._fastskat_hybrid_pvalues(q, a, w, rank=8, threshold=100,
                _lobpcg_worker_pool=pool)
    except ArithmeticError as error:
        held = error
    assert held is not None
    assert 'weight 0: FP64 spectral refinement failed' in str(held)
    failure = statistics.statistics_execution_metadata()['fastskat_failure_records'][-1]
    assert failure['weight_index'] == 0
    assert failure['error'] == 'original first-weight refinement failure'
    del a, w, q
    gc.collect()
    assert all(ref() is None for ref in refs)


@CUDA
def test_caller_lease_rejection_keeps_original_scientific_path():
    a, w, q = inputs()
    old = original_function()(q, a, w, rank=8, threshold=100)
    requested = []
    def reject(extra, *, phase):
        requested.append((extra, phase))
        raise MemoryError('bounded caller tile allowance rejected')
    with LOBPCGWorkerPool(2) as pool:
        new = statistics._fastskat_hybrid_pvalues(q, a, w, rank=8, threshold=100,
            _lobpcg_worker_pool=pool, _lobpcg_workspace_callback=reject)
    assert requested and all(phase == 'fastskat_lookahead_new_peak' for _, phase in requested)
    torch.testing.assert_close(new, old, rtol=0, atol=0)


@CUDA
def test_full_tail_twenty_four_warm_calls_enter_allocated_plateau():
    a, w, q = inputs()
    with LOBPCGWorkerPool(2) as pool:
        for _ in range(3):
            out = statistics._fastskat_hybrid_pvalues(q, a, w, rank=8, threshold=100,
                                                   _lobpcg_worker_pool=pool)
            del out
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        for _ in range(24):
            out = statistics._fastskat_hybrid_pvalues(q, a, w, rank=8, threshold=100,
                                                   _lobpcg_worker_pool=pool)
            del out
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated


@CUDA
def test_unscoped_fastskat_never_creates_per_mask_stream_pool(monkeypatch):
    a, w, q = inputs()
    expected = original_function()(q, a, w, rank=8, threshold=100)
    def forbidden(*args, **kwargs):
        raise AssertionError('unscoped FastSKAT created pool')
    monkeypatch.setattr(tiles, 'LOBPCGWorkerPool', forbidden)
    actual = statistics._fastskat_hybrid_pvalues(q, a, w, rank=8, threshold=100)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
