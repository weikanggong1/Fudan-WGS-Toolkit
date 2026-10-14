"""Crash/acknowledgment contracts for bounded durable batch CSV output."""
from pathlib import Path
import csv,gzip,json,multiprocessing,os,sqlite3,threading,time,weakref
import numpy as np
import pytest
from fudan_wgs_toolkit.phewas_trait_csv import PerTraitCheckpointCSVWriter, CommittedCSVCorruption
from fudan_wgs_toolkit.phewas_async_writer import AsyncPerTraitCheckpointCSVWriter

BINDING=dict(source_implementation_sha256='source',input_manifest_sha256='input',
             configuration_sha256='configuration',models={'0':'model0','1':'model1'})


def row(value=.1,**extra):
    return dict({'Gene name':'g','Chr':21,'Category':'mask','#SNV':3,'cMAC':10.,'STAAR-B':value},**extra)


def tile(job,value=.1,*,traits=(0,1)):
    return dict(kind='coding',chromosome='21',trait_results=[(i,[row(value+i)]) for i in traits],job_index=job)


def csv_bytes(root,result):
    return {r['csv_file']:gzip.decompress((Path(root)/r['csv_file']).read_bytes()) for r in result['receipts']}


def test_batch_same_bytes_ranges_and_checksums_with_one_sync_per_file(tmp_path):
    a,b=tmp_path/'sync',tmp_path/'batch'
    first=PerTraitCheckpointCSVWriter(a,binding=BINDING)
    expected=[first.write_tile(**tile(j,.01*j)) for j in range(8)];ar=first.finish()
    second=PerTraitCheckpointCSVWriter(b,binding=BINDING,max_open_streams=1)
    trace=[];second.db.set_trace_callback(trace.append)
    observed=second.write_tiles([tile(j,.01*j) for j in range(8)])
    assert expected==observed
    assert second.durability_profile['data_fsync_calls']==2
    assert second.durability_profile['tile_commit_transactions']==1
    assert second.durability_profile['registration_transactions']==0
    assert sum(statement=='COMMIT' for statement in trace)==1
    br=second.finish()
    assert csv_bytes(a,ar)==csv_bytes(b,br)
    assert br['committed_blocks']==8 and br['rows']['coding']==16
    assert {r['csv_file']:r['sha256'] for r in ar['receipts']}=={r['csv_file']:r['sha256'] for r in br['receipts']}


def test_batch_empty_tiles_have_durable_individual_blocks(tmp_path):
    w=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    records=w.write_tiles([dict(kind='ncrna',chromosome='21',trait_results=[(0,[]),(1,[])],job_index=j) for j in range(4)])
    assert len(records)==4 and all(r['complete'] and r['rows']==0 for r in records)
    assert w.durability_profile['tile_commit_transactions']==1
    result=w.finish();assert result['committed_blocks']==4 and result['files']==0


def _batch_crash(root,where):
    w=PerTraitCheckpointCSVWriter(root,binding=BINDING)
    if where=='existing':w.write_tile(**tile(0,.01))
    if where=='partial_append':
        original=w._append_member;count=0
        def append(*a,**kw):
            nonlocal count
            out=original(*a,**kw);count+=1
            if count==3:os._exit(73)
            return out
        w._append_member=append
    else:
        def commit(*a,**kw):os._exit(73)
        w._commit_tiles=commit
    w.write_tiles([tile(j,.1*j) for j in (1,2,3)])


@pytest.mark.parametrize('where',['fresh','existing','partial_append'])
def test_process_crash_does_not_commit_any_batch_tile_and_resume_replays_all(tmp_path,where):
    process=multiprocessing.get_context('fork').Process(target=_batch_crash,args=(str(tmp_path),where))
    process.start();process.join(20);assert process.exitcode==73
    w=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,resume=True,worker_id='new')
    if where=='existing':assert w.checkpoint_tile('coding','21',0,[0,1]) is not None
    for j in (1,2,3):assert w.checkpoint_tile('coding','21',j,[0,1]) is None
    w.write_tiles([tile(j,.1*j) for j in (1,2,3)])
    result=w.finish()
    assert result['committed_blocks']==(4 if where=='existing' else 3)
    assert result['rows']['coding']==(8 if where=='existing' else 6)
    assert list((tmp_path/'.trait_checkpoints/uncommitted').rglob('tail.gz'))


def test_commit_return_failure_never_truncates_successfully_committed_batch(tmp_path,monkeypatch):
    w=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING);original=w._commit_tiles
    def committed_then_failed(*args,**kwargs):
        original(*args,**kwargs);raise OSError('after successful commit')
    monkeypatch.setattr(w,'_commit_tiles',committed_then_failed)
    with pytest.raises(OSError,match='successful commit'):w.write_tiles([tile(0),tile(1)])
    w.abort();r=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,resume=True,worker_id='recovered')
    assert all(r.checkpoint_tile('coding','21',j,[0,1]) for j in (0,1))
    assert r.finish()['rows']['coding']==4


def test_fsync_failure_preserves_prior_prefix_and_replays_failed_batch(tmp_path,monkeypatch):
    import fudan_wgs_toolkit.phewas_trait_csv as module
    w=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING);w.write_tile(**tile(0,.01))
    original=module.os.fsync;once=True
    def fail_once(fd):
        nonlocal once
        if once:once=False;raise OSError('injected data fsync failure')
        return original(fd)
    monkeypatch.setattr(module.os,'fsync',fail_once)
    with pytest.raises(OSError,match='data fsync'):w.write_tiles([tile(1),tile(2)])
    assert w.db.execute('SELECT count(*) FROM blocks').fetchone()[0]==1
    w.write_tiles([tile(1),tile(2)])
    assert w.finish()['rows']['coding']==6


def test_batch_corrupt_committed_ranges_still_fail_resume(tmp_path):
    w=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING);w.write_tiles([tile(0),tile(1)])
    result=w.finish();path=tmp_path/result['receipts'][0]['csv_file'];raw=bytearray(path.read_bytes());raw[15]^=1;path.write_bytes(raw)
    r=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,resume=True,worker_id='new')
    with pytest.raises(CommittedCSVCorruption):r.checkpoint_tile('coding','21',0,[0,1])
    r.abort()


def test_async_snapshot_drops_source_owners_and_preserves_exact_scalar_csv(tmp_path):
    class Row(dict):pass
    source=Row(row(np.float32(.1),optional=None,boolean=False))
    ref=weakref.ref(source)
    expected=PerTraitCheckpointCSVWriter(tmp_path/'expected',binding=BINDING)
    expected.write_tile('coding','21',[(0,[source])],job_index=0)
    er=expected.finish()
    w=AsyncPerTraitCheckpointCSVWriter(tmp_path/'async',binding=BINDING,flush_interval_seconds=30)
    ticket=w.write_tile('coding','21',[(0,[source])],job_index=0)
    source['STAAR-B']=.9;del source
    assert ref() is None and not ticket.done()
    w.defer_completion(dict(event='finished',kind='coding',job_index=0))
    assert w.poll_durable_events()==[]
    result=w.finish()
    assert ticket.done() and ticket.result()['rows']==1
    assert w.poll_durable_events()==[dict(event='finished',kind='coding',job_index=0)]
    assert csv_bytes(tmp_path/'async',result)==csv_bytes(tmp_path/'expected',er)
    assert result['asynchronous_output']['pending_owned_bytes_peak']<=64*2**20


def test_async_batches_ack_events_only_after_commit_on_caller_poll(tmp_path):
    entered=threading.Event();release=threading.Event()
    class Blocking(PerTraitCheckpointCSVWriter):
        def _commit_tiles(self,*args,**kwargs):
            entered.set();assert release.wait(10);return super()._commit_tiles(*args,**kwargs)
    w=AsyncPerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,batch_tiles=3,flush_interval_seconds=30,_writer_factory=Blocking)
    futures=[]
    try:
        for j in range(3):
            futures.append(w.write_tile(**tile(j)))
            w.defer_completion(dict(event='finished',job_index=j))
        assert entered.wait(5)
        assert all(not f.done() for f in futures) and w.poll_durable_events()==[]
        release.set();w.flush()
        assert all(f.result()['complete'] for f in futures)
        assert w.poll_durable_events()==[dict(event='finished',job_index=j) for j in range(3)]
        r=w.finish();assert r['durability_profile']['tile_commit_transactions']==1
        assert r['durability_profile']['data_fsync_calls']==2
    finally:
        release.set()
        if not w._closed:w.abort()


def test_fresh_checkpoint_miss_does_not_force_existing_batch_or_block(tmp_path):
    w=AsyncPerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,batch_tiles=32,flush_interval_seconds=30)
    ticket=w.write_tile(**tile(0))
    assert w.checkpoint_tile('coding','21',1,[0,1]) is None
    assert not ticket.done()
    assert w.finish()['rows']['coding']==2


def test_pending_duplicate_checkpoint_waits_for_actual_durable_ack(tmp_path):
    w=AsyncPerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,flush_interval_seconds=30)
    ticket=w.write_tile(**tile(0));receipt=w.checkpoint_tile('coding','21',0,[0,1])
    assert ticket.done() and receipt==ticket.result()
    assert w.finish()['rows']['coding']==2


def test_async_timed_flush_commits_without_more_gpu_production(tmp_path):
    w=AsyncPerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,flush_interval_seconds=.02)
    ticket=w.write_tile(**tile(0));deadline=time.monotonic()+5
    while not ticket.done() and time.monotonic()<deadline:time.sleep(.01)
    assert ticket.done()
    assert w.finish()['rows']['coding']==2


def test_async_oversize_tile_uses_synchronous_source_without_pending_copy(tmp_path):
    w=AsyncPerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,pending_bytes_limit=16384)
    record=row();record['text']='x'*40000
    future=w.write_tile('coding','21',[(0,[record])],job_index=0)
    assert future.done() and future.result()['rows']==1
    result=w.finish();assert result['asynchronous_output']['oversize_synchronous_tiles']==1
    assert result['asynchronous_output']['pending_owned_bytes_peak']<=16384


def test_async_consumer_failure_reaches_every_ticket_and_finish(tmp_path):
    class Failed(PerTraitCheckpointCSVWriter):
        def _sync_batch_files(self):raise OSError('consumer data sync failed')
    w=AsyncPerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,batch_tiles=3,_writer_factory=Failed)
    tickets=[w.write_tile(**tile(j)) for j in range(3)]
    for ticket in tickets:
        with pytest.raises(OSError,match='consumer data sync'):ticket.result(timeout=5)
    with pytest.raises(OSError,match='consumer data sync'):w.finish()
    assert not (tmp_path/'worker_0.complete.private.json').exists()
    assert w.profile()['pending_owned_bytes']==0


def test_async_abort_drops_uncommitted_tiles_without_success_events(tmp_path):
    w=AsyncPerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,flush_interval_seconds=30)
    ticket=w.write_tile(**tile(0));event=w.defer_completion(dict(event='finished',job_index=0));w.abort()
    with pytest.raises(RuntimeError,match='aborted'):ticket.result(timeout=1)
    with pytest.raises(RuntimeError,match='aborted'):event.result(timeout=1)
    assert w.poll_durable_events()==[] and w.profile()['pending_owned_bytes']==0
    assert not (tmp_path/'worker_0.complete.private.json').exists()
    r=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,resume=True)
    assert r.checkpoint_tile('coding','21',0,[0,1]) is None
    r.write_tile(**tile(0));assert r.finish()['rows']['coding']==2


def test_async_rejects_tensor_owners_instead_of_retaining_cuda_or_lazy_results(tmp_path):
    import torch
    w=AsyncPerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    try:
        with pytest.raises(ValueError,match='CPU scalars'):w.write_tile('coding','21',[(0,[row(torch.tensor(.2))])],job_index=0)
        assert w.profile()['pending_owned_bytes']==0
    finally:w.abort()


def test_async_queue_backpressure_bounds_actual_owned_data(tmp_path):
    w=AsyncPerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,pending_bytes_limit=65536,batch_tiles=32,flush_interval_seconds=30)
    tickets=[]
    for j in range(20):
        value=row();value['text']='x'*6000
        tickets.append(w.write_tile('coding','21',[(0,[value])],job_index=j))
    result=w.finish()
    assert all(t.done() for t in tickets) and result['rows']['coding']==20
    assert result['asynchronous_output']['pending_owned_bytes_peak']<=65536
    assert result['asynchronous_output']['durable_batches']>1


def test_fresh_committed_checkpoint_reuses_ledger_without_keeping_unbounded_tokens(tmp_path):
    w=AsyncPerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,flush_interval_seconds=30)
    ticket=w.write_tile(**tile(0));w.flush()
    assert not w._by_token
    assert w.checkpoint_tile('coding','21',0,[0,1])==ticket.result()
    assert w.finish()['rows']['coding']==2


def test_pending_duplicate_checks_scientific_identity_and_cannot_cancel_ack(tmp_path):
    w=AsyncPerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,flush_interval_seconds=30)
    future=w.write_tile(**tile(0))
    assert not future.cancel()
    with pytest.raises(ValueError,match='scientific identity'):
        w.checkpoint_tile('coding','21',0,[0,1],block_identity={'different':1})
    with pytest.raises(ValueError,match='scientific identity'):
        w.write_tile(**dict(tile(0),block_identity={'different':1}))
    assert w.finish()['rows']['coding']==2


def test_ticket_mapping_read_synchronizes_legacy_receipt_consumers(tmp_path):
    from collections.abc import Mapping
    w=AsyncPerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,flush_interval_seconds=30)
    ticket=w.write_tile(**tile(0))
    assert isinstance(ticket,Mapping) and not ticket.done()
    assert ticket['rows']==2 and ticket.done()
    assert ticket.get('kind')=='coding' and 'rows' in list(ticket) and len(ticket)>0
    assert w.finish()['rows']['coding']==2


def test_single_original_boolean_spa_counts_survive_text_snapshot(tmp_path):
    records=[dict(CHR=21,POS=1,pvalue=.1,spa_selected=True,spa_failed=False),
             dict(CHR=21,POS=2,pvalue=.2,spa_selected=True,spa_failed=True),
             dict(CHR=21,POS=3,pvalue=.3,spa_selected=False,spa_failed=False)]
    w=AsyncPerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    ticket=w.write_tile('individual','21',[(0,records)],job_index=0,
                        ordinal_before=0,ordinal_after=3)
    receipt=ticket.result(timeout=5)
    assert receipt['binary_spa_selected']==2 and receipt['binary_spa_failed']==1
    result=w.finish()
    with gzip.open(tmp_path/result['receipts'][0]['csv_file'],'rt',newline='') as source:
        observed=list(csv.DictReader(source))
    assert [r['spa_failed'] for r in observed]==['False','True','False']


def test_mixed_schema_batches_and_single_ordinals_keep_exact_csv_and_tile_receipts(tmp_path):
    sync=PerTraitCheckpointCSVWriter(tmp_path/'sync',binding=BINDING)
    batch=PerTraitCheckpointCSVWriter(tmp_path/'batch',binding=BINDING,max_open_streams=1)
    inputs=[dict(kind='coding',chromosome='21',job_index=0,
                 trait_results=[(0,[row(.1),row(.2,extra='comma,quote"\nline')]),(1,[row(.3)])]),
            dict(kind='coding',chromosome='21',job_index=1,trait_results=[(0,[row(.4)]),(1,[])]),
            dict(kind='individual',chromosome='21',job_index=2,block_index=0,ordinal_before=0,ordinal_after=2,
                 trait_results=[(0,[dict(CHR=21,POS=1,pvalue=.5),dict(CHR=21,POS=2,pvalue=.6)])])]
    expected=[sync.write_tile(**value) for value in inputs]
    actual=batch.write_tiles(inputs)
    assert actual==expected
    assert csv_bytes(tmp_path/'sync',sync.finish())==csv_bytes(tmp_path/'batch',batch.finish())


def test_failed_second_tile_sql_insert_rolls_back_first_tile_and_all_registrations(tmp_path):
    w=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    w.db.execute("CREATE TRIGGER fail_second_tile BEFORE INSERT ON blocks WHEN (SELECT count(*) FROM blocks)>=1 BEGIN SELECT RAISE(ABORT,'injected second tile'); END")
    with pytest.raises(sqlite3.IntegrityError,match='second tile'):w.write_tiles([tile(0),tile(1)])
    assert w.db.execute('SELECT count(*) FROM blocks').fetchone()[0]==0
    assert w.db.execute('SELECT count(*) FROM files').fetchone()[0]==0
    assert w.db.execute('SELECT count(*) FROM ranges').fetchone()[0]==0
    w.db.execute('DROP TRIGGER fail_second_tile')
    w.write_tiles([tile(0),tile(1)])
    assert w.finish()['rows']['coding']==4


def test_resume_true_checkpoint_rpc_does_not_flush_new_unrelated_pending_tile(tmp_path):
    initial=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    expected=initial.write_tile(**tile(0));initial.abort()
    w=AsyncPerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,resume=True,flush_interval_seconds=30)
    assert w.checkpoint_tile('coding','21',0,[0,1])==expected
    pending=w.write_tile(**tile(1))
    assert w.checkpoint_tile('coding','21',2,[0,1]) is None
    assert not pending.done()
    assert w.finish()['rows']['coding']==4


def test_rejected_source_not_kept_alive_by_retained_snapshot_exception(tmp_path):
    import torch
    class Row(dict):pass
    record=Row(row(torch.tensor(.2)));reference=weakref.ref(record)
    w=AsyncPerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    failure=None
    try:
        try:w.write_tile('coding','21',[(0,[record])],job_index=0)
        except ValueError as error:failure=error
        assert failure is not None
        del record
        assert reference() is None
    finally:w.abort()


def test_async_fresh_checkpoint_claims_before_compute_and_rejects_peer(tmp_path):
    first=AsyncPerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    second=AsyncPerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,worker_id='other')
    try:
        assert first.checkpoint_tile('coding','21',0,[0,1]) is None
        assert len(first._backend._owners)==2
        assert first.profile()['claim_rpcs']==1
        assert first.checkpoint_tile('coding','21',1,[0,1]) is None
        assert first.profile()['claim_rpcs']==1
        with pytest.raises(RuntimeError,match='active writer owner'):
            second.checkpoint_tile('coding','21',0,[0,1])
    finally:
        first.abort()
        try:second.abort()
        except RuntimeError:pass


def test_async_claim_cache_bounded_and_observer_runs_on_consumer(tmp_path):
    seen=[]
    def observer():seen.append(threading.get_ident())
    w=AsyncPerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,claim_cache_size=1,consumer_observer=observer)
    main=threading.get_ident()
    w.checkpoint_tile('coding','21',0,[0,1])
    assert len(w._claim_cache)<=1 and seen and all(t!=main for t in seen)
    before=len(seen);w.write_tile(**tile(0));result=w.finish()
    assert len(seen)>before and result['rows']['coding']==2


def test_consumer_observer_admission_failure_propagates_before_false_ack(tmp_path):
    observed=0
    def observer():
        nonlocal observed
        observed+=1
        if observed>=2:raise OSError('observer file descriptor admission denied')
    w=AsyncPerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,consumer_observer=observer)
    with pytest.raises(OSError,match='descriptor admission'):
        w.checkpoint_tile('coding','21',0,[0])
    with pytest.raises(OSError,match='descriptor admission'):w.finish()
    w._thread.join(5)
    assert not w._backend._owners and not (tmp_path/'worker_0.complete.private.json').exists()

@pytest.mark.parametrize('bound',[1,2,48])
@pytest.mark.parametrize('kind',['coding','individual'])
def test_merged_parallel_batch_preserves_positional_bytes_and_single_transaction(tmp_path,bound,kind):
    """Cross the upstream optional fsync path with actual multi-tile batches."""
    values=[]
    for job in range(4):
        item=dict(kind=kind,chromosome='21',job_index=job,
            trait_results=[(i,[row(.1*job,flag=False,label='comma,quote"\nα'),row(np.float32(.2),flag=True)]) for i in range(6)])
        if kind=='individual':item.update(ordinal_before=job*2,ordinal_after=(job+1)*2)
        values.append(item)
    sync=PerTraitCheckpointCSVWriter(tmp_path/'sync',binding=BINDING,max_open_streams=bound)
    expected=[sync.write_tile(**item) for item in values];a=sync.finish()
    batch=PerTraitCheckpointCSVWriter(tmp_path/'batch',binding=BINDING,max_open_streams=bound,parallel_fsync=True)
    actual=batch.write_tiles(values)
    assert actual==expected
    result=batch.finish();profile=result['writer_io_profile']
    assert csv_bytes(tmp_path/'sync',a)==csv_bytes(tmp_path/'batch',result)
    assert a['receipts']==result['receipts']
    # Two different ordered schemas per trait produce twelve actual files.
    assert result['files']==12
    assert profile['fsync_calls']==12 and result['durability_profile']['data_fsync_calls']==12
    assert profile['tile_commit_calls']==profile['tile_sqlite_commit_calls']==1
    assert profile['schema_sqlite_commit_calls']==0 and profile['gzip_members']==48
    assert profile['maximum_pending_fsyncs']<=4
    assert not batch._pending_fsyncs and batch._fsync_executor is None


def test_merged_async_native_positional_serialization_and_pool_remain_consumer_owned(tmp_path):
    class AlteredGet(dict):
        def get(self,key,default=None):return 'not_plain_dict_semantics'
    raw=[AlteredGet(row(.2,spa_selected=False,spa_failed=False,label='quote"\nα')),
         row(np.float32(1e-20),spa_selected=True,spa_failed=False,label=None)]
    direct=PerTraitCheckpointCSVWriter(tmp_path/'direct',binding=BINDING,parallel_fsync=True)
    expected=[direct.write_tile('noncoding','21',[(0,raw),(1,raw)],job_index=i) for i in range(3)]
    a=direct.finish();producer=threading.get_ident();calls=[]
    class Observed(PerTraitCheckpointCSVWriter):
        def _commit_tiles(self,*args,**kwargs):
            calls.append(threading.get_ident())
            assert threading.get_ident()!=producer
            assert not self._pending_fsyncs and not self._dirty_streams
            return super()._commit_tiles(*args,**kwargs)
    async_writer=AsyncPerTraitCheckpointCSVWriter(tmp_path/'async',binding=BINDING,parallel_fsync=True,
        _writer_factory=Observed,batch_tiles=3,flush_interval_seconds=60)
    tickets=[async_writer.write_tile('noncoding','21',[(0,raw),(1,raw)],job_index=i) for i in range(3)]
    b=async_writer.finish()
    assert [t.result() for t in tickets]==expected
    assert csv_bytes(tmp_path/'direct',a)==csv_bytes(tmp_path/'async',b)
    assert a['receipts']==b['receipts']
    assert all(t['binary_spa_selected']==2 and t['binary_spa_failed']==0 for t in expected)
    assert len(calls)==1 and b['writer_io_profile']['parallel_fsync'] is True
    assert b['durability_profile']['tile_commit_transactions']==1


def test_merged_parallel_batch_failure_rolls_back_all_tiles_after_futures_drain(tmp_path,monkeypatch):
    writer=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,parallel_fsync=True)
    original=writer._fsync_call;main=threading.get_ident();workers=[];failed=False
    def failure(fd):
        nonlocal failed
        assert threading.get_ident()!=main
        workers.append(threading.get_ident())
        if not failed:
            failed=True;return 0.,OSError('injected merged parallel batch fsync failure')
        return original(fd)
    monkeypatch.setattr(writer,'_fsync_call',failure)
    with pytest.raises(OSError,match='merged parallel batch'):
        writer.write_tiles([tile(0),tile(1),tile(2)])
    assert workers and not writer._pending_fsyncs and writer._fsync_executor is None
    assert writer.db.execute('SELECT COUNT(*) FROM blocks').fetchone()[0]==0
    assert writer.db.execute('SELECT COUNT(*) FROM files').fetchone()[0]==0
    assert all(path.stat().st_size==0 for path in tmp_path.rglob('*.partial'))
    monkeypatch.setattr(writer,'_fsync_call',original)
    writer.write_tiles([tile(0),tile(1),tile(2)])
    assert writer.finish()['rows']['coding']==6
