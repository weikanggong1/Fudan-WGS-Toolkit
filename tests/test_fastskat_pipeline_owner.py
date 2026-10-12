"""Pipeline/control ownership contracts; these are not WGS benchmarks."""
from types import SimpleNamespace
from contextlib import ExitStack
import pytest
import torch

from fudan_wgs_toolkit.pipeline import PheWASPipeline
from fudan_wgs_toolkit import _fastskat_lobpcg_tiles as tiles
from fudan_wgs_toolkit import _lobpcg_stream_pool as streams

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')


def test_pipeline_owned_pool_is_lazy_and_close_is_idempotent():
    pipeline = object.__new__(PheWASPipeline)
    owner = pipeline._enable_fastskat_lobpcg()
    assert owner.pool is None and not owner.closed
    assert pipeline._enable_fastskat_lobpcg() is owner
    pipeline.close()
    pipeline.close()
    assert owner.closed and pipeline._association_lobpcg_owner is None
    assert owner.metadata()['owner_count'] == 0
    assert owner.metadata()['pool_created_count'] == 0


def test_pipeline_borrowed_pool_survives_thin_pipeline_close():
    pipeline = object.__new__(PheWASPipeline)
    with tiles.lobpcg_pool_scope() as owner:
        pipeline._enable_fastskat_lobpcg(owner=owner)
        with pipeline._fastskat_pool_context():
            assert tiles.current_lobpcg_context()[0] is owner
        pipeline.close()
        assert not owner.closed and tiles.current_lobpcg_context()[0] is owner
    assert owner.closed and tiles.current_lobpcg_context() == (None, None)
    assert owner.metadata()['active_evaluation_workspace_scopes'] == 0


def test_pipeline_evaluation_restores_scope_after_retained_exception():
    pipeline = object.__new__(PheWASPipeline)
    owner = pipeline._enable_fastskat_lobpcg()
    def failure(*args):
        assert tiles.current_lobpcg_context()[0] is owner
        raise ArithmeticError('original tail error')
    pipeline._evaluate_prepared_core = failure
    with pytest.raises(ArithmeticError, match='original tail error'):
        pipeline._evaluate_prepared({}, SimpleNamespace())
    assert tiles.current_lobpcg_context() == (None, None)
    pipeline.close()
    assert owner.closed


def test_borrowed_owner_rejects_closed_or_different_pipeline_binding():
    pipeline = object.__new__(PheWASPipeline)
    first = pipeline._enable_fastskat_lobpcg()
    with tiles.lobpcg_pool_scope() as other:
        with pytest.raises(RuntimeError, match='different'):
            pipeline._enable_fastskat_lobpcg(owner=other)
    pipeline.close()
    with pytest.raises(ValueError, match='open'):
        pipeline._enable_fastskat_lobpcg(owner=first)


@CUDA
def test_first_native_reservation_survives_failure_before_native_work(monkeypatch):
    matrices = [torch.eye(97, device='cuda') for _ in range(2)]
    starts = [torch.ones((97, 4), device='cuda') for _ in range(2)]
    def failure(tile):
        raise RuntimeError('before first native operation')
    monkeypatch.setattr(streams._IndependentTile, 'run', failure)
    with streams.LOBPCGWorkerPool(2) as pool:
        with pytest.raises(RuntimeError, match='first native'):
            streams.independent_lobpcg(matrices, starts, k=4, niter=3, worker_pool=pool)
        assert pool.initial_native_workspace_estimate(matrices[0].device) == 64 << 20
        assert pool.metadata()['first_tile_allocator_warmup'] is None
        assert pool.metadata()['active_solver_workspace_leases'] == 0
        assert pool.metadata()['active_worker_count'] == 0
    assert pool.metadata()['live_control_thread_count'] == 0


@CUDA
def test_bank_callback_rejects_extra_before_lookahead_allocation(monkeypatch):
    covariance = torch.eye(193, device='cuda')
    weights = torch.ones((193, 2), device='cuda')
    generator = torch.Generator(device='cuda').manual_seed(1729)
    allocated = torch.cuda.memory_allocated()
    requests = []
    def reject(additional, *, phase):
        assert torch.cuda.memory_allocated() == allocated
        assert additional >= 0 and phase == 'fastskat_lookahead_new_peak'
        requests.append(additional)
        raise MemoryError('whole-STAAR future budget rejected')
    def forbidden(*args, **kwargs):
        raise AssertionError('rejected lookahead materialized operators')
    with tiles.lobpcg_pool_scope() as scope:
        with tiles.lobpcg_workspace_scope(callback=reject):
            owner = tiles.FastSKATPairTiles(covariance, weights, generator, 8)
            monkeypatch.setattr(owner, '_populate', forbidden)
            try:
                width, estimate = owner._width(0)
                assert width == 1 and estimate is None and requests
                assert scope.pool.metadata()['stream_count'] == 0
            finally:
                owner.close()
    assert scope.metadata()['pool_created_count'] == 1
    assert scope.metadata()['active_evaluation_workspace_scopes'] == 0


def test_many_thin_pipelines_borrow_one_worker_owner_and_workspace_lifetime():
    with tiles.lobpcg_pool_scope() as owner:
        for _ in range(24):
            pipeline = object.__new__(PheWASPipeline)
            assert pipeline._enable_fastskat_lobpcg(owner=owner) is owner
            with pipeline._fastskat_pool_context():
                with tiles.lobpcg_workspace_scope(callback=lambda *args, **kwargs: True):
                    assert owner.metadata()['active_evaluation_workspace_scopes'] == 1
            pipeline.close()
            assert not owner.closed
            assert owner.metadata()['active_evaluation_workspace_scopes'] == 0
        assert owner.metadata()['pool_created_count'] == 0
    assert owner.metadata()['owner_count'] == 0


def _failing_streams(pool, monkeypatch):
    calls = []
    class FailedDeviceStream:
        device = 'cuda:0'
        def synchronize(self):
            calls.append(1)
            raise RuntimeError('closed device synchronization failed')
    pool._streams = [FailedDeviceStream(), FailedDeviceStream()]
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda *args: 0)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda *args: 0)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda *args: (1, 2))
    return calls


def test_close_device_error_attempts_both_streams_and_releases_owners(monkeypatch):
    pool = streams.LOBPCGWorkerPool(2)
    calls = _failing_streams(pool, monkeypatch)
    with pytest.raises(RuntimeError, match='closed device'):
        pool.close()
    assert len(calls) == 2
    metadata = pool.metadata()
    assert metadata['closed'] and metadata['stream_count'] == 0
    assert metadata['live_control_thread_count'] == 0
    assert metadata['active_solver_workspace_leases'] == 0
    assert metadata['cleanup_error_types'] == ['RuntimeError', 'RuntimeError']
    pool.close()


def test_pool_exit_retains_computation_error_over_cleanup_error(monkeypatch):
    pool = streams.LOBPCGWorkerPool(2)
    calls = _failing_streams(pool, monkeypatch)
    with pytest.raises(ArithmeticError, match='original precision failure') as retained:
        with pool:
            raise ArithmeticError('original precision failure')
    assert len(calls) == 2 and pool.metadata()['stream_count'] == 0
    # Native exception notes are available from Python 3.11; error priority
    # and complete cleanup remain mandatory on supported Python 3.10.
    if hasattr(retained.value, 'add_note'):
        assert any('cleanup also failed: RuntimeError' in note for note in retained.value.__notes__)


def test_pipeline_scope_error_releases_lazy_owner_and_restores_context(monkeypatch):
    with pytest.raises(ArithmeticError, match='original tail failure'):
        with tiles.lobpcg_pool_scope() as owner:
            calls = _failing_streams(owner.acquire(), monkeypatch)
            raise ArithmeticError('original tail failure')
    assert len(calls) == 2 and owner.closed and owner.pool is None
    assert owner.metadata()['pool']['stream_count'] == 0
    assert tiles.current_lobpcg_context() == (None, None)


def test_pipeline_close_error_cannot_retain_pool_or_pipeline_owner(monkeypatch):
    pipeline = object.__new__(PheWASPipeline)
    owner = pipeline._enable_fastskat_lobpcg()
    calls = _failing_streams(owner.acquire(), monkeypatch)
    with pytest.raises(RuntimeError, match='closed device'):
        pipeline.close()
    assert len(calls) == 2 and owner.pool is None
    assert pipeline._association_lobpcg_owner is None


def test_manual_exitstack_secondary_error_preserves_primary_and_finishes_cleanup():
    primary = ArithmeticError('original worker failure')
    closed = []
    stack = ExitStack()
    stack.callback(lambda: closed.append('second'))
    def failure():
        closed.append('first')
        raise RuntimeError('cleanup device failed')
    stack.callback(failure)
    streams.close_preserving_error(stack.close, primary)
    assert closed == ['first', 'second']
    if hasattr(primary, 'add_note'):
        assert any('cleanup also failed: RuntimeError' in note for note in primary.__notes__)
