"""CPU startup/environment contracts; no scientific or resource benchmark."""
import importlib.util
import multiprocessing
import os
from pathlib import Path
import sys
import threading

import pytest

_PATH = Path(__file__).resolve().parents[1]/'fudan_wgs_toolkit/_continuous_spawn_allocator.py'
_SPEC = importlib.util.spec_from_file_location('continuous_spawn_allocator_contract', _PATH)
allocator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(allocator)


def absent(monkeypatch):
    for name in allocator._DEFAULTS:
        monkeypatch.delenv(name, raising=False)


def scope(**extra):
    return allocator.continuous_spawn_allocator_scope(input_contract='continuous',
        families=['gaussian', 'gaussian'], worker_count=2, **extra)


def _spawn_child(connection):
    connection.send(dict(environment={name: os.environ.get(name) for name in allocator._DEFAULTS},
        implementation=sys.implementation.name, ignore_environment=sys.flags.ignore_environment))
    connection.close()


def test_actual_spawn_child_observes_allocator_at_interpreter_start_and_parent_restores(monkeypatch):
    absent(monkeypatch)
    context = multiprocessing.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_spawn_child, args=(send,))
    with scope() as receipt:
        process.start()
    send.close()
    try:
        assert receive.poll(20), 'actual owned CPU spawn did not produce startup receipt'
        result = receive.recv()
    finally:
        receive.close()
        process.join(timeout=20)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0
    assert result['implementation'] == 'cpython' and result['ignore_environment'] == 0
    assert result['environment'] == allocator._DEFAULTS
    assert receipt['parent_environment_restored'] is True
    assert receipt['parent_allocator_modified'] is False
    assert receipt['startup_allocator_runtime_verified'] is False
    assert all(name not in os.environ for name in allocator._DEFAULTS)


@pytest.mark.parametrize('contract,families,count,enabled', [
    ('mixed', ['gaussian', 'gaussian'], 2, True),
    ('continuous', ['gaussian', 'binomial'], 2, True),
    ('continuous', ['binomial', 'binomial'], 2, True),
    ('continuous', [], 2, True),
    ('continuous', None, 2, True),
    ('continuous', ['gaussian', 'gaussian'], 1, True),
    ('continuous', ['gaussian', 'gaussian'], 2, False),
])
def test_ineligible_paths_leave_startup_environment_unchanged(monkeypatch, contract, families, count, enabled):
    absent(monkeypatch)
    with allocator.continuous_spawn_allocator_scope(input_contract=contract, families=families,
            worker_count=count, enabled=enabled) as receipt:
        assert receipt['active'] is False
        assert all(name not in os.environ for name in allocator._DEFAULTS)
    assert receipt['parent_environment_restored'] is True


def test_explicit_startup_settings_and_exact_parent_state_win_on_exception(monkeypatch):
    absent(monkeypatch)
    for explicit in ('pymalloc_debug', ''):
        monkeypatch.setenv('PYTHONMALLOC', explicit)
        before = {name: os.environ.get(name) for name in allocator._DEFAULTS}
        with pytest.raises(RuntimeError, match='child_start_failed'):
            with scope() as receipt:
                assert os.environ['PYTHONMALLOC'] == explicit
                assert receipt['default_environment_added'] == {}
                assert receipt['explicit_environment_preserved'] == {'PYTHONMALLOC': explicit}
                raise RuntimeError('child_start_failed')
        assert {name: os.environ.get(name) for name in allocator._DEFAULTS} == before
        assert receipt['parent_environment_restored'] is True


def test_other_allocator_environment_is_never_modified_or_restored(monkeypatch):
    absent(monkeypatch)
    other = {'MALLOC_ARENA_MAX': '', 'MALLOC_MMAP_THRESHOLD_': '524288',
             'MALLOC_TRIM_THRESHOLD_': '262144', 'GLIBC_TUNABLES': 'glibc.malloc.arena_max=7'}
    for name, value in other.items():
        monkeypatch.setenv(name, value)
    with scope() as receipt:
        assert receipt['default_environment_added'] == {'PYTHONMALLOC': 'malloc'}
        assert {name: os.environ[name] for name in other} == other
        # An unrelated deliberate caller change must not be rolled back by
        # this scope: it owns only the PYTHONMALLOC environment entry.
        os.environ['MALLOC_ARENA_MAX'] = '9'
    assert os.environ['MALLOC_ARENA_MAX'] == '9'
    assert {name: os.environ[name] for name in other if name != 'MALLOC_ARENA_MAX'} == {
        name: value for name, value in other.items() if name != 'MALLOC_ARENA_MAX'}


def test_nested_scope_preserves_outer_values_then_exactly_restores(monkeypatch):
    absent(monkeypatch)
    with scope() as outer:
        with scope() as inner:
            assert inner['default_environment_added'] == {}
            assert inner['effective_startup_environment'] == allocator._DEFAULTS
        assert inner['parent_environment_restored'] is True
        assert outer['effective_startup_environment'] == allocator._DEFAULTS
    assert outer['parent_environment_restored'] is True
    assert all(name not in os.environ for name in allocator._DEFAULTS)


def test_cooperating_mixed_spawn_waits_until_candidate_environment_is_restored(monkeypatch):
    absent(monkeypatch)
    entered, released = threading.Event(), threading.Event()
    observed = []
    def inactive():
        entered.set()
        with allocator.continuous_spawn_allocator_scope(input_contract='mixed', families=['binomial'], worker_count=2):
            observed.append({name: os.environ.get(name) for name in allocator._DEFAULTS})
        released.set()
    with scope():
        thread = threading.Thread(target=inactive)
        thread.start()
        assert entered.wait(5)
        assert not released.wait(.1)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert observed == [dict.fromkeys(allocator._DEFAULTS)]


@pytest.mark.parametrize('count,enabled', [(0, True), (True, True), (2, 1)])
def test_invalid_startup_contract_rejected_before_mutation(monkeypatch, count, enabled):
    absent(monkeypatch)
    with pytest.raises(ValueError):
        with allocator.continuous_spawn_allocator_scope(input_contract='continuous',
                families=['gaussian'], worker_count=count, enabled=enabled):
            raise AssertionError('invalid scope must not enter')
    assert all(name not in os.environ for name in allocator._DEFAULTS)
