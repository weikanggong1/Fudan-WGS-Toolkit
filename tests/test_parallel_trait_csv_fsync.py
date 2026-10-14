"""CPU durability/thread contracts; injected waits are not speed benchmarks."""
import csv
import gzip
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

import pytest

import fudan_wgs_toolkit.phewas_trait_csv as module
from fudan_wgs_toolkit.phewas_trait_csv import PerTraitCheckpointCSVWriter

BINDING = dict(source_implementation_sha256='fixture-source', input_manifest_sha256='fixture-input',
               configuration_sha256='fixture-schedule', models={'0': 'fixture-model'})


def row(value=.2):
    return dict(Chr='21', **{'Gene name': 'fixture', '#SNV': 8, 'cMAC': 20, 'STAAR-O': value})


def values(n):
    return [(index, [row(.2), row(.3)]) for index in range(n)]


def read_csv(root, receipt):
    with gzip.open(root/receipt['csv_file'], 'rt', newline='') as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize('bound', [1, 2, 48])
@pytest.mark.parametrize('kind', ['coding', 'noncoding', 'ncrna', 'individual'])
def test_parallel_and_default_serial_preserve_all_gzip_bytes_and_resume_receipts(tmp_path, bound, kind):
    results = []
    roots = [tmp_path/'serial', tmp_path/'parallel']
    for mode, root in zip((False, True), roots):
        writer = PerTraitCheckpointCSVWriter(root, binding=BINDING, parallel_fsync=mode, max_open_streams=bound)
        kwargs = dict(ordinal_before=0, ordinal_after=2) if kind == 'individual' else {}
        first = writer.write_tile(kind, '21', values(6), job_index=0, **kwargs)
        writer.abort()
        assert not writer._pending_fsyncs and writer._fsync_executor is None
        writer = PerTraitCheckpointCSVWriter(root, binding=BINDING, parallel_fsync=mode,
                                             max_open_streams=bound, resume=True)
        assert writer.checkpoint_tile(kind, '21', 0, list(range(6)),
            **({'ordinal_before': 0} if kind == 'individual' else {})) == first
        kwargs = dict(ordinal_before=2, ordinal_after=4) if kind == 'individual' else {}
        writer.write_tile(kind, '21', values(6), job_index=1, **kwargs)
        results.append(writer.finish())
        assert not writer._dirty_streams and not writer._pending_fsyncs and writer._fsync_executor is None
        assert writer._closed and not writer._owners and not writer.streams
    assert results[0]['receipts'] == results[1]['receipts']
    assert results[0]['committed_blocks_binding_sha256'] == results[1]['committed_blocks_binding_sha256']
    assert results[0]['rows'] == results[1]['rows']
    for receipt in results[0]['receipts']:
        assert (roots[0]/receipt['csv_file']).read_bytes() == (roots[1]/receipt['csv_file']).read_bytes()


def test_default_is_serial_and_never_constructs_an_executor(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('default may not create fsync threads')
    monkeypatch.setattr(module, 'ThreadPoolExecutor', forbidden)
    writer = PerTraitCheckpointCSVWriter(tmp_path, binding=BINDING)
    writer.write_tile('coding', '21', values(3), job_index=0)
    profile = writer.finish()['writer_io_profile']
    assert profile['parallel_fsync'] is False and profile['fsync_calls'] == 3
    assert profile['fsync_future_waits'] == profile['maximum_pending_fsyncs'] == 0


@pytest.mark.parametrize('bad', [0, 1, None, 'true'])
def test_parallel_parameter_requires_an_explicit_boolean(tmp_path, bad):
    with pytest.raises(ValueError, match='parallel_fsync must be boolean'):
        PerTraitCheckpointCSVWriter(tmp_path, binding=BINDING, parallel_fsync=bad)


def test_real_future_barrier_precedes_sqlite_commit_and_bounds_four_pending_fds(tmp_path, monkeypatch):
    writer = PerTraitCheckpointCSVWriter(tmp_path, binding=BINDING, parallel_fsync=True)
    original = os.fsync; lock = threading.Lock(); entered = threading.Event(); release = threading.Event()
    state = dict(active=0, maximum=0, finished=0); commits = []; failures = []
    main_thread = threading.get_ident()
    def wait_fsync(fd):
        if threading.get_ident() == main_thread:
            return original(fd)
        with lock:
            state['active'] += 1; state['maximum'] = max(state['maximum'], state['active'])
            if state['active'] == 4: entered.set()
        try:
            assert release.wait(5), 'fixture did not release actual fsync futures'
            os.fstat(fd)  # FD remains open while a real executor owns it.
            original(fd)
        finally:
            with lock:
                state['active'] -= 1; state['finished'] += 1
    monkeypatch.setattr(module.os, 'fsync', wait_fsync)
    original_commit = writer._commit_tile
    def commit(*args):
        assert threading.get_ident() == main_thread
        assert state['finished'] == 9 and state['active'] == 0
        assert not writer._pending_fsyncs and not writer._dirty_streams
        commits.append(True); return original_commit(*args)
    monkeypatch.setattr(writer, '_commit_tile', commit)
    def observe():
        try:
            assert entered.wait(5)
            assert not commits and state['active'] == 4
            for raw in writer.streams.values(): os.fstat(raw.fileno())
        except BaseException as error:
            failures.append(error)
        finally:
            release.set()
    observer = threading.Thread(target=observe); observer.start()
    writer.write_tile('coding', '21', values(9), job_index=0)
    observer.join(5); assert not observer.is_alive() and not failures
    pool_threads = list(writer._fsync_executor._threads)
    profile = writer.finish()['writer_io_profile']
    assert commits == [True] and state['maximum'] == 4
    assert profile['fsync_calls'] == profile['fsync_future_waits'] == 9
    assert profile['maximum_pending_fsyncs'] == 4 and profile['fsync_barrier_calls'] == 1
    assert profile['gzip_members'] == profile['raw_flush_calls'] == 9
    assert profile['initialization_sqlite_commit_calls'] == profile['schema_sqlite_commit_calls'] == profile['tile_sqlite_commit_calls'] == 1
    assert all(not thread.is_alive() for thread in pool_threads)
    assert writer._fsync_executor is None


def test_one_failed_fsync_waits_for_other_futures_before_archiving_and_can_resume(tmp_path, monkeypatch):
    writer = PerTraitCheckpointCSVWriter(tmp_path, binding=BINDING, parallel_fsync=True)
    original = os.fsync; main_thread = threading.get_ident(); lock = threading.Lock()
    entered = threading.Event(); release = threading.Event(); finished = []
    failure_once = []; calls = []; monitor_errors = []
    def fail_one(fd):
        if threading.get_ident() == main_thread:
            return original(fd)
        with lock:
            ordinal = len(calls); calls.append(fd)
            if len(calls) == 3: entered.set()
        try:
            if ordinal == 0 and not failure_once:
                failure_once.append(True); raise OSError('injected fsync failure')
            assert release.wait(5)
            os.fstat(fd); original(fd)
        finally:
            with lock: finished.append(ordinal)
    monkeypatch.setattr(module.os, 'fsync', fail_one)
    original_recover = writer._recover_tail; recovered = []
    def recover(info):
        assert len(finished) == 3 and not writer._pending_fsyncs
        assert writer._fsync_executor is None
        recovered.append(info['id']); return original_recover(info)
    monkeypatch.setattr(writer, '_recover_tail', recover)
    def observe():
        try:
            assert entered.wait(5)
            assert len(finished) == 1 and not recovered
            for raw in writer.streams.values(): os.fstat(raw.fileno())
        except BaseException as error:
            monitor_errors.append(error)
        finally: release.set()
    observer = threading.Thread(target=observe); observer.start()
    with pytest.raises(OSError, match='injected fsync failure'):
        writer.write_tile('coding', '21', values(3), job_index=0)
    observer.join(5); assert not observer.is_alive() and not monitor_errors
    assert len(recovered) == 3 and not writer._pending_fsyncs
    assert writer.db.execute('SELECT COUNT(*) FROM blocks').fetchone()[0] == 0
    assert len(list((tmp_path/'.trait_checkpoints/uncommitted').rglob('tail.gz'))) == 3
    assert all(p.stat().st_size == 0 for p in tmp_path.rglob('*.partial'))
    writer.abort()
    monkeypatch.setattr(module.os, 'fsync', original)
    writer = PerTraitCheckpointCSVWriter(tmp_path, binding=BINDING, parallel_fsync=True, resume=True)
    assert writer.checkpoint_tile('coding', '21', 0, [0, 1, 2]) is None
    writer.write_tile('coding', '21', values(3), job_index=0)
    output = writer.finish()
    assert output['rows']['coding'] == 6 and output['computed_blocks'] == 1
    assert sum(len(read_csv(tmp_path, r)) for r in output['receipts']) == 6


def test_failed_submission_drains_accepted_future_before_fd_recovery(tmp_path, monkeypatch):
    accepted = threading.Event(); release = threading.Event(); finished = threading.Event(); records = []
    original = os.fsync; main_thread = threading.get_ident(); errors = []
    class SubmitFailure(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            if records:
                accepted.set(); raise OSError('injected executor submission failure')
            future = super().submit(*args, **kwargs); records.append(future); return future
    monkeypatch.setattr(module, 'ThreadPoolExecutor', SubmitFailure)
    def delay(fd):
        if threading.get_ident() != main_thread:
            assert release.wait(5); os.fstat(fd); original(fd); finished.set()
        else: original(fd)
    monkeypatch.setattr(module.os, 'fsync', delay)
    writer = PerTraitCheckpointCSVWriter(tmp_path, binding=BINDING, parallel_fsync=True)
    original_recover = writer._recover_tail
    def recover(info):
        assert finished.is_set() and records[0].done() and not writer._pending_fsyncs
        return original_recover(info)
    monkeypatch.setattr(writer, '_recover_tail', recover)
    def observe():
        try:
            assert accepted.wait(5) and not finished.is_set()
            for raw in writer.streams.values(): os.fstat(raw.fileno())
        except BaseException as error: errors.append(error)
        finally: release.set()
    observer = threading.Thread(target=observe); observer.start()
    with pytest.raises(OSError, match='submission failure'):
        writer.write_tile('coding', '21', values(2), job_index=0)
    observer.join(5); assert not observer.is_alive() and not errors
    assert writer.db.execute('SELECT COUNT(*) FROM blocks').fetchone()[0] == 0
    assert writer.io_profile['fsync_future_waits'] == 1 and writer._fsync_executor is None
    writer.abort()


def test_interrupted_future_wait_retries_before_releasing_fd(tmp_path, monkeypatch):
    release = threading.Event(); entered = threading.Event(); finished = threading.Event()
    original = os.fsync; main_thread = threading.get_ident(); actual_futures = []
    class InterruptedFuture:
        calls = 0
        def __init__(self, value): self.value = value
        def result(self):
            self.calls += 1
            if self.calls == 1:
                assert entered.wait(5); release.set(); raise KeyboardInterrupt('injected caller wait interrupt')
            return self.value.result()
        def done(self): return self.value.done()
    class InterruptedExecutor(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs); actual_futures.append(future)
            return InterruptedFuture(future)
    monkeypatch.setattr(module, 'ThreadPoolExecutor', InterruptedExecutor)
    def delay(fd):
        if threading.get_ident() != main_thread:
            entered.set(); assert release.wait(5)
            os.fstat(fd); original(fd); finished.set()
        else: original(fd)
    monkeypatch.setattr(module.os, 'fsync', delay)
    writer = PerTraitCheckpointCSVWriter(tmp_path, binding=BINDING, parallel_fsync=True)
    original_recover = writer._recover_tail
    def recover(info):
        assert finished.is_set() and actual_futures[0].done()
        return original_recover(info)
    monkeypatch.setattr(writer, '_recover_tail', recover)
    with pytest.raises(KeyboardInterrupt, match='caller wait interrupt'):
        writer.write_tile('coding', '21', values(1), job_index=0)
    assert not writer._pending_fsyncs and writer._fsync_executor is None
    assert writer.db.execute('SELECT COUNT(*) FROM blocks').fetchone()[0] == 0
    writer.abort()


def test_empty_parallel_tile_still_commits_without_a_pool_or_gzip_member(tmp_path):
    writer = PerTraitCheckpointCSVWriter(tmp_path, binding=BINDING, parallel_fsync=True)
    writer.write_tile('coding', '21', [(0, [])], job_index=0)
    result = writer.finish(); profile = result['writer_io_profile']
    assert result['committed_blocks'] == 1 and result['files'] == 0
    assert profile['tile_sqlite_commit_calls'] == 1 and profile['gzip_members'] == profile['fsync_calls'] == 0
    assert writer._fsync_executor is None


def test_actual_owned_child_crash_during_parallel_fsync_preserves_prior_commit_and_resume(tmp_path):
    code = '''
import os,sys,threading
from pathlib import Path
import fudan_wgs_toolkit.phewas_trait_csv as module
from fudan_wgs_toolkit.phewas_trait_csv import PerTraitCheckpointCSVWriter
binding=dict(source_implementation_sha256='fixture-source',input_manifest_sha256='fixture-input',configuration_sha256='fixture-schedule',models={'0':'fixture-model'})
def rows(p):return [(i,[dict(Chr='21',**{'Gene name':'fixture','STAAR-O':p})]) for i in range(4)]
writer=PerTraitCheckpointCSVWriter(Path(sys.argv[1]),binding=binding,parallel_fsync=True)
writer.write_tile('coding','21',rows(.2),job_index=0)
original=os.fsync;main_thread=threading.get_ident()
def crash(fd):
 if threading.get_ident()!=main_thread:os._exit(77)
 return original(fd)
module.os.fsync=crash
writer.write_tile('coding','21',rows(.9),job_index=1)
raise AssertionError('fixture crash was not exercised')
'''
    repo = Path(module.__file__).resolve().parent.parent
    child = subprocess.Popen([sys.executable, '-c', code, str(tmp_path)],
        env=dict(os.environ, PYTHONPATH=str(repo)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = child.communicate(timeout=20)
    assert child.wait() == 77, stderr.decode()
    writer = PerTraitCheckpointCSVWriter(tmp_path, binding=BINDING, parallel_fsync=True,
                                         resume=True, worker_id='recovered')
    assert writer.checkpoint_tile('coding', '21', 0, [0, 1, 2, 3]) is not None
    assert writer.checkpoint_tile('coding', '21', 1, [0, 1, 2, 3]) is None
    writer.write_tile('coding', '21', [(i, [row(.4)]) for i in range(4)], job_index=1)
    result = writer.finish()
    assert result['committed_blocks'] == 2 and result['reused_blocks'] == result['computed_blocks'] == 1
    assert sorted(r['STAAR-O'] for receipt in result['receipts'] for r in read_csv(tmp_path, receipt)) == ['0.2']*4+['0.4']*4
    assert len(list((tmp_path/'.trait_checkpoints/uncommitted').rglob('tail.gz'))) == 4
