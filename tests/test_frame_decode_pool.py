"""CPU multiprocessing lifecycle and byte parity; no GPU performance claims."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from fudan_wgs_toolkit.cache_runtime import fast_container, sparse_codec_fast, store


def _cache(tmp_path, encoding='absolute', large=False):
    # All six states, empty exception rows and high sample IDs are retained.
    n = 70001 if large else 91
    states = np.zeros((41, n), dtype=np.uint8)
    for i in range(len(states)):
        if i % 7:
            states[i, i::103 if large else 3] = (np.arange(len(states[i, i::103 if large else 3])) % 5)+1
    path = tmp_path/'cache'
    writer = store.Writer(path, {'fixture':'process-frame-byte-parity'}, np.arange(n,dtype=np.int64),
        len(states), source_bytes=1024*2**20, chunk=4, sample_encoding=encoding)
    for start in range(0,len(states),4):
        part=states[start:start+4]
        writer.append(part, sparse_codec_fast.integer_counts(*sparse_codec_fast.compact(part),*part.shape))
    writer.finish()
    return path


def _assert_frame_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value,np.ndarray):
            assert actual[key].dtype == value.dtype
            assert actual[key].shape == value.shape
            assert actual[key].tobytes() == value.tobytes()
            assert actual[key].flags.writeable is False
        else:
            assert actual[key] == value


@pytest.mark.parametrize('encoding,large', [('absolute',False),('row-delta',False),('row-delta',True)])
def test_parallel_frames_are_byte_exact_in_request_order_and_repeated_reads(tmp_path,encoding,large):
    path = _cache(tmp_path,encoding,large)
    original=store.Container(path)
    reader=fast_container.Container(path)
    reader.enable_process_prefetch(2,64*2**20,tmp_path)
    sequence=[7,0,9,2,4,1,3]
    reader.prefetch_frames(sequence+sequence)
    assert reader.prefetch_metrics['submitted'] >= 2
    assert reader.prefetch_metrics['active_processes'] == 2
    assert reader.prefetch_metrics['pending_frames'] <= 4
    for frame_id in sequence:
        _assert_frame_equal(reader.read_frame(frame_id),original.read_frame(frame_id))
    reader.prefetch_frames([0,0])
    _assert_frame_equal(reader.read_frame(0),original.read_frame(0))
    before=reader.prefetch_metrics
    identities=[record['identity'] for record in before['workers']]
    assert len({identity['pid'] for identity in identities}) == 2
    for record in before['workers']:
        assert record['cuda_initialized'] is False
        assert record['startup_verified'] is True
        assert record['identity']['parent'] == os.getpid()
        assert record['identity']['effective_affinity'] == sorted(os.sched_getaffinity(0))
        assert set(record['single_thread_environment'].values()) == {'1'}
    assert before['duplicate_requests'] >= len(sequence)
    registry_path=Path(before['registry_path'])
    reader.close()
    registry=json.loads(registry_path.read_text())
    assert registry['pool_shutdown'] is True
    assert all(record['actual_exit_code']==0 for record in registry['workers'])
    assert not (registry_path.parent/'handoffs').exists()
    for identity in identities:
        assert not Path('/proc/%d'%identity['pid']).exists()
    assert reader.prefetch_metrics['charged_bytes'] == 0


def test_pending_scratch_handoff_admission_never_exceeds_cap(tmp_path):
    path=_cache(tmp_path)
    with fast_container.Container(path) as reader:
        reader.enable_process_prefetch(2,64*2**20,tmp_path)
        pool=reader._decode_pool
        estimate=max(pool._estimate(i) for i in range(len(reader.index)))
        pool.memory_bytes=estimate*2
        reader.prefetch_frames(range(len(reader.index)))
        assert reader.prefetch_metrics['pending_frames'] == 2
        for frame_id in range(len(reader.index)):
            reader.read_frame(frame_id)
            metrics=reader.prefetch_metrics
            assert metrics['charged_bytes'] <= pool.memory_bytes
            assert metrics['peak_charged_bytes'] <= pool.memory_bytes
            assert metrics['pending_frames'] <= 2


def test_too_small_cap_rejects_before_spawning_children(tmp_path):
    with fast_container.Container(_cache(tmp_path)) as reader:
        reader.enable_process_prefetch(2,1,tmp_path)
        with pytest.raises(MemoryError,match='estimate exceeds'):
            reader.prefetch_frames([0,1])
        assert reader.prefetch_metrics['active_processes'] == 0
        assert reader.prefetch_metrics['charged_bytes'] == 0


def test_prefetched_corrupt_source_error_is_not_silently_retried(tmp_path):
    path=_cache(tmp_path)
    # Mutation before opening preserves the existing source checksum failure,
    # rather than relying on the separate immutable-open/close identity guard.
    with (path/'data.bin').open('r+b') as stream:
        old=stream.read(1)
        stream.seek(0)
        stream.write(bytes([old[0]^1]))
    reader=fast_container.Container(path)
    reader.enable_process_prefetch(2,64*2**20,tmp_path)
    reader.prefetch_frames([0,1])
    identities=[record['identity'] for record in reader.prefetch_metrics['workers']]
    with pytest.raises(ValueError,match='SHA mismatch'):
        reader.read_frame(0)
    with pytest.raises((ValueError,RuntimeError)):
        reader.close()
    assert all(not Path('/proc/%d'%identity['pid']).exists() for identity in identities)


def test_child_actual_exit_propagates_and_all_children_are_reaped(tmp_path):
    reader=fast_container.Container(_cache(tmp_path))
    reader.enable_process_prefetch(2,64*2**20,tmp_path)
    reader.prefetch_frames([0,1])
    pool=reader._decode_pool
    processes=[worker['process'] for worker in pool._workers]
    processes[0].terminate()
    processes[0].wait()
    with pytest.raises((ValueError,RuntimeError)):
        reader.close()
    assert all(process.poll() is not None for process in processes)
    assert json.loads(pool._registry_path.read_text())['pool_shutdown'] is True


def test_close_surfaces_unconsumed_prefetch_failure(tmp_path):
    path=_cache(tmp_path)
    row=np.load(path/'index.npy',allow_pickle=False)[3]
    with (path/'data.bin').open('r+b') as stream:
        stream.seek(int(row['offset']))
        byte=stream.read(1)
        stream.seek(int(row['offset']))
        stream.write(bytes([byte[0]^1]))
    reader=fast_container.Container(path)
    reader.enable_process_prefetch(2,64*2**20,tmp_path)
    reader.prefetch_frames([0,3])
    _assert_frame_equal(reader.read_frame(0),store.Container(path).read_frame(0))
    with pytest.raises((ValueError,RuntimeError)):
        reader.close()


def test_full_queue_accepts_arbitrary_demand_order_with_byte_parity(tmp_path):
    path=_cache(tmp_path)
    original=store.Container(path)
    with fast_container.Container(path) as reader:
        reader.enable_process_prefetch(2,64*2**20,tmp_path)
        pool=reader._decode_pool
        pool.memory_bytes=2*max(pool._estimate(i) for i in range(len(reader.index)))
        reader.prefetch_frames(range(len(reader.index)))
        for frame_id in [9,4,0,7,1,10,2,6,3,8,5]:
            _assert_frame_equal(reader.read_frame(frame_id),original.read_frame(frame_id))
            assert reader.prefetch_metrics['charged_bytes'] <= pool.memory_bytes
        assert reader.prefetch_metrics['discarded'] > 0


def test_consumer_cache_hits_can_skip_prefetched_prefix(tmp_path):
    path=_cache(tmp_path)
    original=store.Container(path)
    with fast_container.Container(path) as reader:
        reader.enable_process_prefetch(2,64*2**20,tmp_path)
        pool=reader._decode_pool
        pool.memory_bytes=2*max(pool._estimate(i) for i in range(len(reader.index)))
        reader.prefetch_frames(range(len(reader.index)))
        # Frames 0..3 are supplied by an existing consumer LRU. The transport
        # must permit the next actual request without requiring their consumption.
        for frame_id in range(4,len(reader.index)):
            _assert_frame_equal(reader.read_frame(frame_id),original.read_frame(frame_id))
        assert reader.prefetch_metrics['discarded'] >= 2


def test_partial_protocol_line_times_out_and_child_is_reaped(tmp_path):
    from fudan_wgs_toolkit.cache_runtime.frame_decode_pool import FrameDecodeProcessPool, _child_identity
    with fast_container.Container(_cache(tmp_path)) as container:
        pool=FrameDecodeProcessPool(container,1,64*2**20,tmp_path,worker_timeout_seconds=.1)
        stderr=(tmp_path/'blocked-child.stderr').open('w')
        process=subprocess.Popen([sys.executable,'-u','-c',
            "import sys,time;sys.stdout.write('{');sys.stdout.flush();time.sleep(10)"],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True)
        identity=_child_identity(process.pid)
        worker=dict(process=process,stderr=stderr,identity=identity)
        pool._workers.append(worker)
        pool._registry['workers'].append(dict(role='cpu_decode',identity=identity,actual_exit_code=None))
        started=time.monotonic()
        with pytest.raises(TimeoutError,match='bounded timeout'):
            pool._exchange(worker,dict(operation='decode'))
        assert time.monotonic()-started < 2
        with pytest.raises(RuntimeError):
            pool.close()
        assert process.poll() is not None
        assert json.loads(pool._registry_path.read_text())['pool_shutdown'] is True


def test_actual_cpu_completion_is_counted_before_consumption_and_on_discard(tmp_path):
    reader=fast_container.Container(_cache(tmp_path))
    reader.enable_process_prefetch(2,64*2**20,tmp_path)
    reader.prefetch_frames([0,1])
    pool=reader._decode_pool
    for future,_,_ in list(pool._pending.values()):
        future.result()
    before=reader.prefetch_metrics
    assert before['actual_completed_tasks'] == 2
    assert before['consumed'] == 0
    assert sum(record['completed_tasks'] for record in before['workers']) == 2
    assert before['child_cpu_seconds'] > 0
    reader.read_frame(0)
    reader.close()
    after=reader.prefetch_metrics
    assert after['actual_completed_tasks'] == 2
    assert after['consumed'] == 1
    assert after['discarded_completed_tasks'] == 1
    assert after['child_cpu_seconds'] == before['child_cpu_seconds']


@pytest.mark.parametrize('processes,memory', [(0,2**20),(33,2**20),(True,2**20),(1,0)])
def test_invalid_parallel_limits_fail_without_processes(tmp_path,processes,memory):
    with fast_container.Container(_cache(tmp_path)) as reader:
        with pytest.raises(ValueError):
            reader.enable_process_prefetch(processes,memory,tmp_path)
        assert reader._decode_pool is None
