"""Runtime progress reflects durable ACKs, including chromosome transitions."""
import gzip
import sqlite3
import threading

import numpy as np
import pytest

from fudan_wgs_toolkit.chromosome_csv import ChromosomeCheckpointCSVWriter
from fudan_wgs_toolkit.phewas_async_writer import AsyncPerTraitCheckpointCSVWriter
from fudan_wgs_toolkit.phewas_runtime.runtime import _durable_event_emitter
from fudan_wgs_toolkit.phewas_trait_csv import PerTraitCheckpointCSVWriter


BINDING = dict(source_implementation_sha256='anonymous-source',
    input_manifest_sha256='anonymous-input', configuration_sha256='anonymous-config',
    models={'0':'anonymous-model'})


def tile(job, chromosome='21'):
    return dict(kind='coding', chromosome=chromosome, job_index=job,
        trait_results=[(0,[{'Gene name':'anonymous', 'Chr':chromosome,
                           'Category':'mask', '#SNV':3, 'P':.05+job*.001}])])


def finished(job, chromosome=0):
    return dict(event='finished', kind='coding', chromosome_index=chromosome,
                job_index=job, wall_seconds=.01)


def test_finished_waits_for_real_batch_commit_and_callbacks_stay_on_producer(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class Blocked(PerTraitCheckpointCSVWriter):
        def write_tiles(self, values):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('test did not release output commit')
            return super().write_tiles(values)
    writer = AsyncPerTraitCheckpointCSVWriter(tmp_path, binding=BINDING,
        batch_tiles=32, flush_interval_seconds=30, _writer_factory=Blocked)
    records, threads, errors = [], [], []
    producer = threading.get_ident()
    emit, drain = _durable_event_emitter(writer,
        lambda record: (records.append(record), threads.append(threading.get_ident())))
    emit({'event':'started', 'kind':'coding', 'job_index':0})
    ticket = writer.write_tile(**tile(0))
    emit(finished(0))
    def flush():
        try:
            writer.flush()
        except BaseException as error:
            errors.append(error)
    waiter = threading.Thread(target=flush)
    waiter.start()
    try:
        assert entered.wait(5)
        drain()
        assert [r['event'] for r in records] == ['started']
        assert not ticket.done()
        release.set(); waiter.join(5)
        assert not waiter.is_alive() and not errors
        drain()
        assert [r['event'] for r in records] == ['started', 'finished']
        assert threads == [producer, producer]
        assert ticket.done() and ticket.result()['complete']
        # An independent connection verifies that progress follows a committed
        # ledger entry, rather than merely a producer queue acknowledgment.
        connection = sqlite3.connect(tmp_path/'.trait_checkpoints/ranges.sqlite')
        try:
            assert connection.execute('SELECT COUNT(*) FROM blocks').fetchone()[0] == 1
        finally:
            connection.close()
        result = writer.finish(); drain()
        assert result['committed_blocks'] == 1 and len(records) == 2
    finally:
        release.set(); waiter.join(5)
        if not writer._closed:
            writer.abort()


def test_full_event_queue_flushes_and_drains_before_retrying_completion(tmp_path, monkeypatch):
    writer = AsyncPerTraitCheckpointCSVWriter(tmp_path, binding=BINDING,
        max_deferred_events=1, batch_tiles=32, flush_interval_seconds=30)
    received, flushes = [], []
    original_flush = writer.flush
    def counted_flush():
        flushes.append(True)
        return original_flush()
    monkeypatch.setattr(writer, 'flush', counted_flush)
    emit, drain = _durable_event_emitter(writer, received.append)
    try:
        writer.write_tile(**tile(0)); emit(finished(0))
        writer.write_tile(**tile(1)); emit(finished(1))
        assert flushes, 'event BufferError must cause a FIFO durability barrier'
        writer.finish(); drain()
        assert [r['job_index'] for r in received] == [0, 1]
        assert writer.profile()['pending_owned_bytes'] == 0
        assert writer.profile()['pending_durable_events'] == 0
    finally:
        if not writer._closed:
            writer.abort()


def test_final_finish_then_drain_delivers_last_completion_without_another_started_event(tmp_path):
    writer = AsyncPerTraitCheckpointCSVWriter(tmp_path, binding=BINDING,
        batch_tiles=32, flush_interval_seconds=30)
    received = []
    emit, drain = _durable_event_emitter(writer, received.append)
    try:
        writer.write_tile(**tile(0)); emit(finished(0))
        assert not received
        output = writer.finish()
        drain(); drain()
        assert [r['job_index'] for r in received] == [0]
        assert output['complete'] and output['committed_blocks'] == 1
        assert writer.profile()['pending_owned_bytes'] == 0
    finally:
        if not writer._closed:
            writer.abort()


def test_chromosome_switch_keeps_old_ack_and_final_drain_keeps_new_ack(tmp_path):
    writer = ChromosomeCheckpointCSVWriter(tmp_path, binding=BINDING,
        chromosomes=['21','22'], asynchronous=True, batch_tiles=32,
        flush_interval_seconds=30)
    received = []
    emit, drain = _durable_event_emitter(writer, received.append)
    try:
        writer.write_tile(**tile(0,'21')); emit(finished(0,0))
        assert not received
        # Transition forces old backend finish/readback and retains its ACK;
        # creating a new current backend must not overwrite that notification.
        writer.write_tile(**tile(0,'22'))
        drain()
        assert [(r['chromosome_index'],r['job_index']) for r in received] == [(0,0)]
        emit(finished(0,1))
        result = writer.finish(); drain(); drain()
        assert [(r['chromosome_index'],r['job_index']) for r in received] == [(0,0),(1,0)]
        assert result['committed_blocks'] == 2 and result['assigned_chromosomes'] == ['21','22']
        assert result['committed_range_checksums_verified']
        texts = [gzip.decompress((tmp_path/r['csv_file']).read_bytes()).decode()
                 for r in result['receipts']]
        assert len(texts) == 2 and all('Gene name,Chr,Category,#SNV,P' in text for text in texts)
    finally:
        if not writer._closed:
            writer.abort()


def test_consumer_failure_never_emits_a_finished_success(tmp_path):
    class Failed(PerTraitCheckpointCSVWriter):
        def _sync_batch_files(self):
            raise OSError('injected durability failure')
    writer = AsyncPerTraitCheckpointCSVWriter(tmp_path, binding=BINDING,
        batch_tiles=32, flush_interval_seconds=30, _writer_factory=Failed)
    received = []
    emit, drain = _durable_event_emitter(writer, received.append)
    writer.write_tile(**tile(0)); emit(finished(0))
    with pytest.raises(OSError, match='durability failure'):
        writer.finish()
    with pytest.raises(OSError, match='durability failure'):
        drain()
    assert not received and not (tmp_path/'worker_0.complete.private.json').exists()


def test_synchronous_writer_completion_remains_immediate():
    received = []
    emit, drain = _durable_event_emitter(object(), received.append)
    event = finished(0)
    emit(event); drain()
    assert received == [event]


@pytest.mark.parametrize('body_failure', [False, True])
def test_portable_string_reference_prefetch_and_failed_metadata_cleanup_close_container(
        tmp_path, monkeypatch, body_failure):
    from fudan_wgs_toolkit import phewas_run
    from fudan_wgs_toolkit.cache_runtime import portable, fast_container
    closed, limits = [], []
    class Metadata:
        manifest = {'source_sample_rows':'rows'}
        def __init__(self, *args, **kwargs):
            assert kwargs['verify_checksums'] is True
        def _array(self, path):
            return np.array([1,2,3])
        def close(self):
            closed.append('metadata')
            raise OSError('metadata cleanup failure')
    class Container:
        samples = np.array([1,2,3])
        def __init__(self, *args, **kwargs):
            pass
        def enable_prefetch(self, limit):
            limits.append(limit)
        def close(self):
            closed.append('container')
    monkeypatch.setattr(portable, 'PortableMetadataReader', Metadata)
    monkeypatch.setattr(fast_container, 'Container', Container)
    failure = ValueError if body_failure else OSError
    message = 'original analysis failure' if body_failure else 'metadata cleanup failure'
    with pytest.raises(failure, match=message):
        with phewas_run.portable_reader(str(tmp_path/'cache'),
                {'metadata_directory':str(tmp_path/'metadata'), 'frame_prefetch_bytes':65536}):
            if body_failure:
                raise ValueError('original analysis failure')
    assert limits == [65536]
    assert closed == ['metadata','container']
