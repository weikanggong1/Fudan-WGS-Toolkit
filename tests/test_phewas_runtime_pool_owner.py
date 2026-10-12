"""Worker-scope ownership contracts; no GPU arithmetic or WGS benchmark."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from fudan_wgs_toolkit import _fastskat_lobpcg_tiles as tiles
from fudan_wgs_toolkit.phewas_runtime import runtime


@pytest.fixture
def initialized_runtime(monkeypatch):
    """Reach real owner lifetime without allocating a CUDA tensor."""
    before = torch.get_num_threads()
    aborted, observed = [], []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'init', lambda: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *args: None)
    monkeypatch.setattr(runtime, '_resolved_budget', lambda *args: 20.)
    native_scope = tiles.lobpcg_pool_scope

    @contextmanager
    def recorded_scope(**kwargs):
        with native_scope(**kwargs) as owner:
            observed.append(owner)
            yield owner

    monkeypatch.setattr(tiles, 'lobpcg_pool_scope', recorded_scope)
    primary = ArithmeticError('injected runtime stage failure')

    def stage_failure(*args, **kwargs):
        assert tiles.current_lobpcg_context()[0] is observed[-1]
        raise primary

    monkeypatch.setattr(runtime, 'StageProfiler', stage_failure)
    writer = SimpleNamespace(write=lambda *args: None, abort=lambda: aborted.append(True))

    def invoke(configuration):
        return runtime.run_batched_configuration(
            configuration, model_repository=None, reader_factory=lambda *args: None,
            writer=writer, device='cuda:0', cpu_threads=1)

    yield SimpleNamespace(invoke=invoke, observed=observed, aborted=aborted,
                          primary=primary, previous_threads=before)
    assert torch.get_num_threads() == before
    assert runtime._LOCK.acquire(blocking=False)
    runtime._LOCK.release()


def test_runtime_default_owner_closes_after_original_stage_exception(initialized_runtime):
    case = initialized_runtime
    with pytest.raises(ArithmeticError) as raised:
        case.invoke({})
    assert raised.value is case.primary
    assert len(case.observed) == 1 and case.observed[0].closed
    assert case.observed[0].metadata()['owner_count'] == 0
    assert case.observed[0].metadata()['pool_created_count'] == 0
    assert case.aborted == [True]
    assert tiles.current_lobpcg_context() == (None, None)


def test_runtime_borrowed_owner_survives_original_stage_exception(initialized_runtime):
    case = initialized_runtime
    owner = tiles._LazyPoolOwner(2)
    configuration = {'_association_lobpcg_owner': owner}
    try:
        with pytest.raises(ArithmeticError) as raised:
            case.invoke(configuration)
        assert raised.value is case.primary and case.observed == [owner]
        assert not owner.closed and owner.pool is None
        assert configuration == {'_association_lobpcg_owner': owner}
        assert case.aborted == [True]
        assert tiles.current_lobpcg_context() == (None, None)
    finally:
        owner.close()


def test_runtime_borrowed_owner_restores_distinct_outer_context(initialized_runtime):
    case = initialized_runtime
    outer, borrowed = tiles._LazyPoolOwner(1), tiles._LazyPoolOwner(2)
    try:
        with tiles.lobpcg_pool_scope(owner=outer):
            with pytest.raises(ArithmeticError):
                case.invoke({'_association_lobpcg_owner': borrowed})
            assert tiles.current_lobpcg_context()[0] is outer
            assert not borrowed.closed and not outer.closed
        assert not outer.closed and tiles.current_lobpcg_context() == (None, None)
    finally:
        borrowed.close()
        outer.close()


@pytest.mark.parametrize('wrong_owner', [False, True])
def test_runtime_rejects_closed_or_untyped_owner_before_stage(initialized_runtime, wrong_owner):
    case = initialized_runtime
    owner = object() if wrong_owner else tiles._LazyPoolOwner(2)
    if not wrong_owner:
        owner.close()
    with pytest.raises(ValueError, match='open and explicitly scoped'):
        case.invoke({'_association_lobpcg_owner': owner})
    assert case.observed == [] and case.aborted == [True]
    assert tiles.current_lobpcg_context() == (None, None)


def test_runtime_owns_cleanup_without_masking_original_error(initialized_runtime, monkeypatch):
    case = initialized_runtime
    native_close = tiles._LazyPoolOwner.close

    def secondary_close(owner):
        native_close(owner)
        raise RuntimeError('injected secondary close failure')

    monkeypatch.setattr(tiles._LazyPoolOwner, 'close', secondary_close)
    with pytest.raises(ArithmeticError) as raised:
        case.invoke({})
    assert raised.value is case.primary and case.observed[0].closed
    if hasattr(raised.value, 'add_note'):
        assert any('RuntimeError' in note for note in raised.value.__notes__)
    assert case.aborted == [True]
    assert tiles.current_lobpcg_context() == (None, None)
