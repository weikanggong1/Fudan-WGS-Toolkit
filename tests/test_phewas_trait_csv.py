"""Durable canonical CSV output contracts, not association benchmarks."""
import csv
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sqlite3
import weakref

import pytest

from fudan_wgs_toolkit.phewas_trait_csv import (PerTraitCheckpointCSVWriter, CommittedCSVCorruption,
                                              association_output_lease, invocation_lease_for_spawn,
                                              held_invocation_lease, close_pending_invocation_lease)
from fudan_wgs_toolkit.phewas_storage import StreamingCSVWriter

BINDING = dict(source_implementation_sha256='source',input_manifest_sha256='inputs',
               configuration_sha256='ordered-schedule-and-fixed-tile',models={'0':'model0','1':'model1'})


def row(value=.25, **extra):
    return dict(Chr='21', **{'Gene name':'fixture'}, **{'STAAR-O':value}, **extra)


def rows(directory, report):
    result = []
    for receipt in report['receipts']:
        with gzip.open(Path(directory)/receipt['csv_file'],'rt',newline='') as handle:
            result.extend(csv.DictReader(handle))
    return result


def test_per_trait_real_directories_and_original_scalar_column_order(tmp_path):
    values = [row(-0.,flag=True,text='quote, newline\nα'),row(1e-100,flag=False,text='plain')]
    old = StreamingCSVWriter(tmp_path/'old')
    new = PerTraitCheckpointCSVWriter(tmp_path/'new',binding=BINDING)
    for trait in (0,1):
        old.write('coding',trait,'21',values,job_index=0)
    original = old.finish()
    new.write_tile('coding','21',[(0,values),(1,values)],job_index=0)
    actual = new.finish()
    assert rows(tmp_path/'new',actual)==rows(tmp_path/'old',original)
    assert actual['rows']['coding']==4 and actual['files']==2
    for receipt in actual['receipts']:
        path = Path(receipt['csv_file'])
        assert path.parts[:4]==('traits',f"trait_{receipt['trait_index']:04d}",'coding','chr21')
        assert {r['trait_index'] for r in rows(tmp_path/'new',{'receipts':[receipt]})}=={str(receipt['trait_index'])}
        assert receipt['columns']==['trait_index','Chr','Gene name','STAAR-O','flag','text']
        assert receipt['sha256']==hashlib.sha256((tmp_path/'new'/path).read_bytes()).hexdigest()
    assert not list((tmp_path/'new').rglob('*.partial'))
    assert not new.streams and not new._owners and new._closed


def test_many_jobs_reopen_same_files_with_bounded_stream_and_verification_lru(tmp_path):
    writer = PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,max_open_streams=1,verification_cache_size=2)
    for job in range(60):
        writer.write_tile('coding','21',[(0,[row(job)]),(1,[row(job+100)])],job_index=job)
        assert len(writer.streams)<=1 and len(writer._verified)<=2
    report = writer.finish()
    assert report['files']==2 and report['committed_blocks']==60
    assert len(list(tmp_path.rglob('*.csv.gz')))==2
    assert [float(r['STAAR-O']) for r in rows(tmp_path,report) if r['trait_index']=='0']==list(range(60))
    assert [float(r['STAAR-O']) for r in rows(tmp_path,report) if r['trait_index']=='1']==list(range(100,160))
    for receipt in report['receipts']:
        with gzip.open(tmp_path/receipt['csv_file'],'rt',newline='') as handle:
            assert sum(r==receipt['columns'] for r in csv.reader(handle))==1


def test_resume_reassigns_worker_and_reuses_source_bound_ranges_without_duplicates(tmp_path):
    writer = PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,worker_id='0')
    first = writer.write_tile('coding','21',[(0,[row()]),(1,[row(.5)])],job_index=7,
                              binary_counts={'selected':2,'failed':0})
    writer.abort()
    assert not writer.streams and not writer._owners
    recovered = PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,resume=True,worker_id='new-worker')
    assert recovered.checkpoint_tile('coding','21',7,[0,1])==first
    assert recovered.write_tile('coding','21',[(0,[row(.9)]),(1,[row(.9)])],job_index=7)==first
    recovered.write_tile('coding','21',[(0,[row(.75)]),(1,[])],job_index=8)
    result = recovered.finish()
    assert result['reused_blocks']==1 and result['computed_blocks']==1
    assert result['rows']['coding']==3 and result['committed_blocks']==2
    assert [r['STAAR-O'] for r in rows(tmp_path,result)]==['0.25','0.75','0.5']
    assert result['trait_rows']['0']['coding']==2 and result['trait_rows']['1']['coding']==1


def test_same_invocation_retry_is_idempotent_without_enabling_prior_run_resume(tmp_path):
    writer = PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    first = writer.write_tile('coding','21',[(0,[row()])],job_index=0)
    assert writer.checkpoint_tile('coding','21',0,[0])==first
    assert writer.write_tile('coding','21',[(0,[row(.99)])],job_index=0)==first
    report=writer.finish()
    assert report['computed_blocks']==1 and report['reused_blocks']==0 and report['rows']['coding']==1
    with pytest.raises(FileExistsError,match='resume=True'):
        PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    other=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,worker_id='different')
    with pytest.raises(FileExistsError,match='resume=True'):
        other.write_tile('coding','21',[(0,[row(.99)])],job_index=0)
    other.abort()


def test_final_complete_replay_preserves_whole_csv_bytes_and_block_binding(tmp_path):
    writer=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    for job in range(3):writer.write_tile('ncrna','21',[(0,[row(job)])],job_index=job)
    original=writer.finish()
    recovered=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,resume=True,worker_id='7')
    for job in range(3):assert recovered.checkpoint_tile('ncrna','21',job,[0]) is not None
    result=recovered.finish()
    assert result['receipts']==original['receipts']
    assert result['committed_blocks_binding_sha256']==original['committed_blocks_binding_sha256']
    assert result['computed_blocks']==0 and result['reused_blocks']==3


def test_empty_tile_and_single_ordinals_are_durable_even_without_a_csv_file(tmp_path):
    writer=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    original=writer.write_tile('individual','21',[(0,[]),(1,None)],job_index=0,block_index=4,
        block_identity='physical-request-hash',ordinal_before=11,ordinal_after=11,
        binary_counts={'selected':0,'failed':0})
    writer.abort()
    recovered=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,resume=True,worker_id='3')
    assert recovered.checkpoint_tile('individual','21',0,[0,1],block_index=4,
        block_identity='physical-request-hash',ordinal_before=11)==original
    result=recovered.finish()
    assert result['files']==0 and result['committed_blocks']==1 and not any(result['rows'].values())
    assert not list(tmp_path.rglob('*.csv.gz'))


def test_columns_change_and_order_are_distinct_canonical_schemas(tmp_path):
    writer=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    first=row(extra=1)
    reordered={key:first[key] for key in reversed(first)}
    writer.write_tile('coding','21',[(0,[first,reordered,first])],job_index=0)
    result=writer.finish()
    assert result['files']==2 and result['rows']['coding']==3
    assert {tuple(r['columns']) for r in result['receipts']}=={
        ('trait_index',*first),('trait_index',*reordered)}


def test_row_override_or_mid_tile_error_does_not_commit_any_trait_prefix(tmp_path):
    writer=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    with pytest.raises(ValueError,match='override'):
        writer.write_tile('coding','21',[(0,[row()]),(1,[dict(row(),trait_index=0)])],job_index=0)
    assert writer.row_counts=={}
    assert writer.db.execute('SELECT COUNT(*) FROM blocks').fetchone()[0]==0
    assert all(path.stat().st_size==0 for path in tmp_path.rglob('*.partial'))
    assert list((tmp_path/'.trait_checkpoints/uncommitted').rglob('tail.gz'))
    writer.write_tile('coding','21',[(0,[row(.7)]),(1,[])],job_index=0)
    result=writer.finish()
    assert result['rows']['coding']==1 and rows(tmp_path,result)[0]['STAAR-O']=='0.7'


def test_failure_before_database_commit_preserves_tail_and_computes_only_missing_block(tmp_path,monkeypatch):
    writer=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    writer.write_tile('coding','21',[(0,[row(.2)]),(1,[row(.3)])],job_index=0)
    original=writer._commit_tile
    def fail(*args):raise OSError('injected durable commit failure')
    monkeypatch.setattr(writer,'_commit_tile',fail)
    with pytest.raises(OSError,match='durable commit'):
        writer.write_tile('coding','21',[(0,[row(.9)]),(1,[row(.8)])],job_index=1)
    monkeypatch.setattr(writer,'_commit_tile',original)
    writer.abort()
    recovered=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,resume=True,worker_id='2')
    assert recovered.checkpoint_tile('coding','21',0,[0,1]) is not None
    assert recovered.checkpoint_tile('coding','21',1,[0,1]) is None
    recovered.write_tile('coding','21',[(0,[row(.4)]),(1,[row(.5)])],job_index=1)
    result=recovered.finish()
    assert result['rows']['coding']==4 and result['computed_blocks']==result['reused_blocks']==1
    assert sorted(r['STAAR-O'] for r in rows(tmp_path,result))==['0.2','0.3','0.4','0.5']


def _crash_between_append_and_commit(directory):
    writer=PerTraitCheckpointCSVWriter(directory,binding=BINDING)
    writer.write_tile('coding','21',[(0,[row(.2)]),(1,[row(.3)])],job_index=0)
    writer._commit_tile=lambda *args:os._exit(73)
    writer.write_tile('coding','21',[(0,[row(.9)]),(1,[row(.8)])],job_index=1)


def test_abrupt_process_exit_releases_owners_and_uncommitted_ranges_never_skip(tmp_path):
    process=multiprocessing.get_context('fork').Process(target=_crash_between_append_and_commit,args=(str(tmp_path),))
    process.start();process.join(30)
    assert process.exitcode==73
    recovered=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,resume=True,worker_id='4')
    assert recovered.checkpoint_tile('coding','21',0,[0,1]) is not None
    assert recovered.checkpoint_tile('coding','21',1,[0,1]) is None
    recovered.write_tile('coding','21',[(0,[row(.4)]),(1,[row(.5)])],job_index=1)
    result=recovered.finish()
    assert sorted(r['STAAR-O'] for r in rows(tmp_path,result))==['0.2','0.3','0.4','0.5']
    assert len(list((tmp_path/'.trait_checkpoints/uncommitted').rglob('tail.gz')))==2


def _parallel_trait(directory,trait,ready,go):
    writer=PerTraitCheckpointCSVWriter(directory,binding=BINDING,worker_id=str(trait))
    writer.checkpoint_tile('coding','21',0,[trait])
    ready.set();assert go.wait(20)
    for job in range(8):writer.write_tile('coding','21',[(trait,[row(trait+job)])],job_index=job)
    writer.finish()


def test_concurrent_distinct_traits_have_short_shared_commits_and_independent_files(tmp_path):
    context=multiprocessing.get_context('fork');go=context.Event();ready=[context.Event(),context.Event()]
    processes=[context.Process(target=_parallel_trait,args=(str(tmp_path),trait,ready[trait],go)) for trait in (0,1)]
    for process in processes:process.start()
    try:
        assert all(event.wait(20) for event in ready)
        go.set()
        for process in processes:process.join(30)
        assert [process.exitcode for process in processes]==[0,0]
        reports=[json.loads((tmp_path/f'worker_{trait}.complete.private.json').read_text()) for trait in (0,1)]
        assert [r['committed_blocks'] for r in reports]==[8,8]
        assert [len(rows(tmp_path,r)) for r in reports]==[8,8]
        assert len(list(tmp_path.rglob('*.csv.gz')))==2
    finally:
        go.set()
        for process in processes:
            if process.is_alive():process.terminate()
            process.join(10)


def test_active_owner_conflict_is_explicit_instead_of_deadlock_or_duplicate_rows(tmp_path):
    first=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    first.write_tile('ncrna','21',[(0,[])],job_index=0)
    second=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,resume=True,worker_id='other')
    with pytest.raises(RuntimeError,match='active writer owner'):
        second.checkpoint_tile('ncrna','21',0,[0])
    first.abort()
    assert second.checkpoint_tile('ncrna','21',0,[0]) is not None
    second.finish()


def _parallel_chromosome(directory,chromosome,ready,go):
    writer=PerTraitCheckpointCSVWriter(directory,binding=BINDING,worker_id=chromosome,
        ownership_scope='chromosome',max_open_streams=1)
    writer.checkpoint_tile('coding',chromosome,0,[0,1])
    assert len(writer._owners)==1
    ready.set();assert go.wait(20)
    for job in range(8):
        writer.write_tile('coding',chromosome,[(0,[row(job,chromosome=chromosome)]),
            (1,[row(job+100,chromosome=chromosome)])],job_index=job)
        assert len(writer.streams)<=1 and len(writer._owners)==1
    writer.finish()


def test_parallel_chromosomes_share_traits_without_recovering_or_finishing_peer_files(tmp_path):
    context=multiprocessing.get_context('fork');go=context.Event();ready=[context.Event(),context.Event()]
    processes=[context.Process(target=_parallel_chromosome,args=(str(tmp_path),chromosome,ready[index],go))
        for index,chromosome in enumerate(('21','22'))]
    for process in processes:process.start()
    try:
        assert all(event.wait(20) for event in ready);go.set()
        for process in processes:process.join(30)
        assert [process.exitcode for process in processes]==[0,0]
        reports=[json.loads((tmp_path/f'worker_{chromosome}.complete.private.json').read_text())
            for chromosome in ('21','22')]
        assert [report['committed_blocks'] for report in reports]==[8,8]
        assert [len(rows(tmp_path,report)) for report in reports]==[16,16]
        assert all(set(report['trait_rows'])=={'0','1'} for report in reports)
        assert [{receipt['chromosome'] for receipt in report['receipts']} for report in reports]==[{'21'},{'22'}]
        assert len(list(tmp_path.rglob('*.csv.gz')))==4
        assert not list(tmp_path.rglob('*.partial'))
    finally:
        go.set()
        for process in processes:
            if process.is_alive():process.terminate()
            process.join(10)


def test_chromosome_resume_card_redispatch_cannot_steal_active_owner(tmp_path):
    first=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,ownership_scope='chromosome')
    block=first.write_tile('coding','21',[(0,[row()]),(1,[row(.5)])],job_index=4)
    resumed=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,resume=True,worker_id='new-card',
        ownership_scope='chromosome')
    with pytest.raises(RuntimeError,match='active writer owner'):
        resumed.checkpoint_tile('coding','21',4,[0,1])
    first.abort()
    assert resumed.checkpoint_tile('coding','21',4,[0,1])==block
    resumed.write_tile('coding','21',[(0,[row(.75)]),(1,[])],job_index=5)
    result=resumed.finish()
    assert result['rows']['coding']==3 and result['reused_blocks']==1
    assert result['trait_rows']['0']['coding']==2
    assert len(rows(tmp_path,result))==3


def test_chromosome_claim_keeps_other_chromosome_uncommitted_tail_untouched(tmp_path):
    first=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,ownership_scope='chromosome')
    first.write_tile('coding','21',[(0,[row()])],job_index=0);first.abort()
    partial=next(tmp_path.rglob('*.partial'))
    with partial.open('ab') as handle:handle.write(b'preserved-peer-tail')
    before=partial.read_bytes()
    second=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,worker_id='1',ownership_scope='chromosome')
    second.write_tile('coding','22',[(0,[row(.9)])],job_index=0)
    result=second.finish()
    assert partial.read_bytes()==before and result['rows']['coding']==1
    assert [receipt['chromosome'] for receipt in result['receipts']]==['22']
    resumed=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,resume=True,worker_id='2',ownership_scope='chromosome')
    assert resumed.checkpoint_tile('coding','21',0,[0]) is not None
    resumed.finish()
    assert list((tmp_path/'.trait_checkpoints/uncommitted').rglob('tail.gz'))


def test_chromosome_and_legacy_trait_writer_modes_cannot_share_ledger(tmp_path):
    writer=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,ownership_scope='chromosome')
    with pytest.raises(ValueError,match='binding differs'):
        PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,worker_id='legacy')
    writer.abort()


def _hold_invocation_lease(directory,ready,release,crash):
    with association_output_lease(directory):
        assert not Path(directory).exists()
        ready.set();assert release.wait(20)
        if crash:os._exit(73)


@pytest.mark.parametrize('crash',[False,True])
def test_public_invocation_lease_rejects_active_parent_and_recovers_after_exit(tmp_path,crash):
    context=multiprocessing.get_context('fork');ready=context.Event();release=context.Event()
    output=tmp_path/'output';different=tmp_path/'independent'
    process=context.Process(target=_hold_invocation_lease,args=(str(output),ready,release,crash))
    process.start()
    try:
        assert ready.wait(20)
        with pytest.raises(RuntimeError,match='active public invocation'):
            with association_output_lease(output):raise AssertionError('active output acquired')
        with association_output_lease(different):assert not different.exists()
        release.set();process.join(30)
        assert process.exitcode==(73 if crash else 0)
        with association_output_lease(output):assert not output.exists()
        assert (tmp_path/'.output.wgs-association.lock').stat().st_size==0
    finally:
        release.set()
        if process.is_alive():process.terminate()
        process.join(10)


def test_invocation_lease_checks_symlink_lock_before_any_output_change(tmp_path):
    preserved=tmp_path/'preserved';preserved.write_bytes(b'preserved')
    (tmp_path/'.output.wgs-association.lock').symlink_to(preserved)
    with pytest.raises(ValueError,match='symlink'):
        with association_output_lease(tmp_path/'output'):pass
    assert preserved.read_bytes()==b'preserved' and not (tmp_path/'output').exists()


def _alive_spawn_lease_child(handle,ready,release,done):
    with held_invocation_lease(handle):
        ready.set();assert release.wait(45)
    done.set()


def _crash_parent_with_spawned_lease_children(directory,ready,release,done):
    with association_output_lease(directory):
        context=multiprocessing.get_context('spawn')
        children=[]
        for index in range(2):
            child=context.Process(target=_alive_spawn_lease_child,
                args=(invocation_lease_for_spawn(),ready[index],release[index],done[index]))
            child.start();children.append(child)
        assert all(event.wait(45) for event in ready)
        # Deliberately bypass cleanup after both real spawn children have
        # detached their descriptors and entered their owned work scopes.
        os._exit(73)


def test_parent_crash_does_not_release_invocation_until_last_spawn_worker_ends(tmp_path):
    context=multiprocessing.get_context('spawn')
    ready=[context.Event(),context.Event()];release=[context.Event(),context.Event()]
    done=[context.Event(),context.Event()];output=tmp_path/'output'
    parent=context.Process(target=_crash_parent_with_spawned_lease_children,
        args=(str(output),ready,release,done))
    parent.start()
    try:
        assert all(event.wait(45) for event in ready)
        parent.join(30);assert parent.exitcode==73
        with pytest.raises(RuntimeError,match='active public invocation'):
            with association_output_lease(output):raise AssertionError('orphan workers lost ownership')
        release[0].set();assert done[0].wait(45)
        with pytest.raises(RuntimeError,match='active public invocation'):
            with association_output_lease(output):raise AssertionError('last live worker lost ownership')
        release[1].set();assert done[1].wait(45)
        with association_output_lease(output):assert not output.exists()
    finally:
        for event in release:event.set()
        if parent.is_alive():parent.terminate()
        parent.join(10)
        for event in done:event.wait(10)


def test_unstarted_spawn_transfer_is_retired_without_leaving_invocation_locked(tmp_path):
    output=tmp_path/'output'
    with association_output_lease(output):
        close_pending_invocation_lease(invocation_lease_for_spawn())
        with pytest.raises(RuntimeError,match='active public invocation'):
            with association_output_lease(output):pass
    with association_output_lease(output):assert not output.exists()


def _spawn_child_before_lease_detach(handle,ready,release):
    # The OS-transferred descriptor must hold ownership before scientific
    # worker entry/detach as well as through its normal context manager.
    ready.set();assert release.wait(45)
    with held_invocation_lease(handle):pass


def test_spawn_killed_before_detach_has_no_resource_sharer_lock_leak(tmp_path,monkeypatch):
    import multiprocessing.resource_sharer as resource_sharer
    def unexpected_sharer(*args,**kwargs):
        raise AssertionError('invocation lease registered with resource_sharer')
    monkeypatch.setattr(resource_sharer,'DupFd',unexpected_sharer)
    context=multiprocessing.get_context('spawn')
    ready=context.Event();release=context.Event();output=tmp_path/'output';child=None
    try:
        with association_output_lease(output):
            child=context.Process(target=_spawn_child_before_lease_detach,
                args=(invocation_lease_for_spawn(),ready,release))
            child.start();assert ready.wait(45)
        with pytest.raises(RuntimeError,match='active public invocation'):
            with association_output_lease(output):raise AssertionError('undetached child lost ownership')
        child.terminate();child.join(30)
        assert child.exitcode is not None and child.exitcode<0
        with association_output_lease(output):assert not output.exists()
    finally:
        # A terminated Event waiter cannot acknowledge Condition.notify_all.
        # Do not set that Event during cleanup: this test deliberately kills
        # the exact child before detach and needs only terminate/join cleanup.
        if child is not None:
            if child.is_alive():child.terminate()
            child.join(10)


def test_invocation_transfer_refuses_unbound_serialization_and_retired_reuse(tmp_path):
    with association_output_lease(tmp_path/'output'):
        handle=invocation_lease_for_spawn()
        with pytest.raises(RuntimeError,match='Process.start'):
            pickle.dumps(handle)
        close_pending_invocation_lease(handle)
        with pytest.raises(ValueError,match='retired'):
            pickle.dumps(handle)
        with pytest.raises(ValueError,match='retired'):
            with held_invocation_lease(handle):pass


def test_spawn_worker_exception_releases_only_its_duplicate(tmp_path):
    output=tmp_path/'output'
    with association_output_lease(output):
        with pytest.raises(ArithmeticError,match='owned worker failure'):
            with held_invocation_lease(invocation_lease_for_spawn()):
                raise ArithmeticError('owned worker failure')
        with pytest.raises(RuntimeError,match='active public invocation'):
            with association_output_lease(output):pass
    with association_output_lease(output):assert not output.exists()


@pytest.mark.parametrize('damage',['byte','truncate','database'])
def test_corrupt_committed_range_fails_closed_and_preserves_evidence(tmp_path,damage):
    writer=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    writer.write_tile('coding','21',[(0,[row()])],job_index=0)
    result=writer.finish();path=tmp_path/result['receipts'][0]['csv_file']
    if damage=='byte':
        content=bytearray(path.read_bytes());content[0]^=1;path.write_bytes(content)
    elif damage=='truncate':path.write_bytes(path.read_bytes()[:-1])
    else:
        with sqlite3.connect(tmp_path/'.trait_checkpoints/ranges.sqlite') as database:
            database.execute('UPDATE ranges SET rows=rows+1')
    before=path.read_bytes()
    recovered=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,resume=True,worker_id='9')
    with pytest.raises(CommittedCSVCorruption):recovered.checkpoint_tile('coding','21',0,[0])
    recovered.abort()
    assert path.read_bytes()==before


def test_binding_mismatch_is_checked_before_any_uncommitted_tail_repair(tmp_path):
    writer=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    writer.write_tile('coding','21',[(0,[row()])],job_index=0)
    writer.abort();partial=next(tmp_path.rglob('*.partial'))
    with partial.open('ab') as stream:stream.write(b'preserved-tail')
    before=partial.read_bytes()
    with pytest.raises(ValueError,match='binding differs'):
        PerTraitCheckpointCSVWriter(tmp_path,binding=dict(BINDING,models={'0':'changed'}),resume=True)
    assert partial.read_bytes()==before


def test_omitted_committed_block_cannot_be_claimed_as_a_complete_file(tmp_path):
    writer=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    for job in range(2):writer.write_tile('coding','21',[(0,[row(job)])],job_index=job)
    writer.abort()
    recovered=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,resume=True)
    recovered.checkpoint_tile('coding','21',0,[0])
    with pytest.raises(RuntimeError,match='omitted'):
        recovered.finish()
    recovered.abort()
    assert not (tmp_path/'worker_0.complete.private.json').exists()


def test_source_rows_are_not_retained_in_lru_or_checkpoint_metadata(tmp_path):
    class Row(dict):pass
    value=Row(row());reference=weakref.ref(value)
    writer=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    writer.write_tile('coding','21',[(0,[value])],job_index=0)
    del value
    assert reference() is None
    assert len(writer._verified)==0  # durable writes do not claim readback
    writer.abort()


def test_fresh_write_defers_readback_until_final_or_resume(tmp_path,monkeypatch):
    writer=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    def unexpected(*args):raise AssertionError('fresh data read back twice')
    monkeypatch.setattr(writer,'_read_range',unexpected)
    writer.write_tile('coding','21',[(0,[row()])],job_index=0)
    assert writer.finish()['rows']['coding']==1


def test_live_disk_reserve_is_rechecked_for_an_existing_open_stream(tmp_path,monkeypatch):
    writer=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,minimum_free_bytes=100)
    original=writer.write_tile('coding','21',[(0,[row()])],job_index=0)
    partial=next(tmp_path.rglob('*.partial'));before=partial.read_bytes()
    assert len(writer.streams)==1
    from types import SimpleNamespace
    import fudan_wgs_toolkit.phewas_trait_csv as module
    monkeypatch.setattr(module.shutil,'disk_usage',lambda path:SimpleNamespace(free=99))
    # Reusing committed ranges does not need new disk space, but a new tile does.
    assert writer.checkpoint_tile('coding','21',0,[0])==original
    with pytest.raises(OSError,match='free-space reserve'):
        writer.write_tile('coding','21',[(0,[row(.9)])],job_index=1)
    assert partial.read_bytes()==before
    assert writer.db.execute('SELECT COUNT(*) FROM blocks').fetchone()[0]==1
    assert writer.finish()['rows']['coding']==1


def test_primary_commit_error_survives_recovery_failure_and_abort_closes_owners(tmp_path,monkeypatch):
    writer=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    def fail(*args):raise OSError('primary durable commit failure')
    def cleanup(*args):raise RuntimeError('secondary tail cleanup failure')
    monkeypatch.setattr(writer,'_commit_tile',fail)
    monkeypatch.setattr(writer,'_recover_tail',cleanup)
    with pytest.raises(OSError,match='primary durable') as error:
        writer.write_tile('coding','21',[(0,[row()])],job_index=0)
    assert any('secondary tail cleanup failure' in note for note in error.value.__notes__)
    writer.abort()
    assert not writer._owners and not writer.streams and writer._closed


def test_close_releases_all_owners_even_if_one_stream_flush_fails(tmp_path,monkeypatch):
    writer=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    writer.write_tile('coding','21',[(0,[row()]),(1,[row()])],job_index=0)
    real=writer._close_stream
    first=True
    def fail_one(file_id):
        nonlocal first
        real(file_id)
        if first:
            first=False
            raise OSError('injected close failure')
    monkeypatch.setattr(writer,'_close_stream',fail_one)
    with pytest.raises(OSError,match='close failure'):writer.close()
    assert not writer._owners and not writer.streams and writer._closed
    with pytest.raises(sqlite3.ProgrammingError):writer.db.execute('SELECT 1')
    recovered=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,resume=True)
    assert recovered.checkpoint_tile('coding','21',0,[0,1]) is not None
    recovered.finish()


def test_synchronized_members_do_not_repeat_fsync_on_eviction_or_close(tmp_path, monkeypatch):
    writer = PerTraitCheckpointCSVWriter(tmp_path, binding=BINDING, max_open_streams=1)
    original = os.fsync
    synchronized = []
    def observed(descriptor):
        target = os.readlink(f'/proc/self/fd/{descriptor}')
        if target.endswith('results.csv.gz.partial'):
            synchronized.append(target)
            assert not writer.db.in_transaction
        return original(descriptor)
    monkeypatch.setattr(os, 'fsync', observed)
    for job in range(3):
        writer.write_tile('coding', '21', [(0, [row(job)]), (1, [row(job)])], job_index=job)
        assert not writer._dirty_streams
    # Six complete members are durable before their scientific tile commit.
    assert len(synchronized) == 6
    writer.close()
    assert len(synchronized) == 6
    assert not writer.streams and not writer._dirty_streams and not writer._owners
    recovered = PerTraitCheckpointCSVWriter(tmp_path, binding=BINDING, resume=True)
    for job in range(3):
        assert recovered.checkpoint_tile('coding', '21', job, [0, 1]) is not None
    assert recovered.finish()['rows']['coding'] == 6


def test_failed_member_sync_remains_dirty_until_recovery_and_never_commits(tmp_path, monkeypatch):
    writer = PerTraitCheckpointCSVWriter(tmp_path, binding=BINDING)
    original = os.fsync
    attempts = []
    def fail_first(descriptor):
        target = os.readlink(f'/proc/self/fd/{descriptor}')
        if target.endswith('results.csv.gz.partial'):
            attempts.append(bool(writer._dirty_streams))
            if len(attempts) == 1:
                raise OSError('injected member durability failure')
        return original(descriptor)
    monkeypatch.setattr(os, 'fsync', fail_first)
    with pytest.raises(OSError, match='member durability'):
        writer.write_tile('coding', '21', [(0, [row()])], job_index=0)
    assert attempts[:2] == [True, True]  # failed append and dirty close retry
    assert not writer.streams and not writer._dirty_streams
    assert writer.db.execute('SELECT COUNT(*) FROM blocks').fetchone()[0] == 0
    assert next(tmp_path.rglob('*.partial')).stat().st_size == 0
    assert len(list((tmp_path/'.trait_checkpoints/uncommitted').rglob('tail.gz'))) == 1
    writer.write_tile('coding', '21', [(0, [row(.4)])], job_index=0)
    result = writer.finish()
    assert result['rows']['coding'] == 1 and rows(tmp_path, result)[0]['STAAR-O'] == '0.4'


def test_initial_file_metadata_has_one_short_transaction_and_existing_files_are_read_only(tmp_path, monkeypatch):
    writer = PerTraitCheckpointCSVWriter(tmp_path, binding=BINDING, max_open_streams=1)
    statements = []
    writer.db.set_trace_callback(statements.append)
    original = writer._append_member
    def append(*args, **kwargs):
        assert not writer.db.in_transaction
        return original(*args, **kwargs)
    monkeypatch.setattr(writer, '_append_member', append)
    writer.write_tile('coding', '21', [(trait, [row(trait)]) for trait in range(32)], job_index=0)
    assert statements.count('BEGIN IMMEDIATE') == statements.count('COMMIT') == 2
    assert sum('INSERT OR IGNORE INTO files' in statement for statement in statements) == 32
    statements.clear()
    writer.write_tile('coding', '21', [(trait, [row(trait+1)]) for trait in range(32)], job_index=1)
    assert statements.count('BEGIN IMMEDIATE') == statements.count('COMMIT') == 1
    assert not any('INSERT OR IGNORE INTO files' in statement for statement in statements)
    assert writer.finish()['rows']['coding'] == 64


def test_initial_file_registration_failure_rolls_back_all_metadata_and_can_retry(tmp_path):
    writer = PerTraitCheckpointCSVWriter(tmp_path, binding=BINDING)
    writer.db.executescript('''CREATE TRIGGER reject_second BEFORE INSERT ON files
        WHEN NEW.trait=1 BEGIN SELECT RAISE(ABORT,'injected file registration failure'); END;''')
    with pytest.raises(sqlite3.IntegrityError, match='file registration'):
        writer.write_tile('coding', '21', [(0, [row()]), (1, [row()])], job_index=0)
    assert not writer.db.in_transaction
    assert writer.db.execute('SELECT COUNT(*) FROM files').fetchone()[0] == 0
    assert writer.db.execute('SELECT COUNT(*) FROM blocks').fetchone()[0] == 0
    assert not list(tmp_path.rglob('*.partial'))
    writer.db.execute('DROP TRIGGER reject_second')
    writer.write_tile('coding', '21', [(0, [row()]), (1, [row()])], job_index=0)
    assert writer.finish()['rows']['coding'] == 2


def _crash_after_file_registration(directory):
    writer = PerTraitCheckpointCSVWriter(directory, binding=BINDING)
    writer._append_member = lambda *args, **kwargs: os._exit(74)
    writer.write_tile('coding', '21', [(0, [row()]), (1, [row()])], job_index=0)


def test_registration_without_csv_after_crash_is_missing_work_not_a_checkpoint(tmp_path):
    process = multiprocessing.get_context('fork').Process(
        target=_crash_after_file_registration, args=(str(tmp_path),))
    process.start(); process.join(30)
    assert process.exitcode == 74
    recovered = PerTraitCheckpointCSVWriter(tmp_path, binding=BINDING, resume=True)
    assert recovered.db.execute('SELECT COUNT(*) FROM files').fetchone()[0] == 2
    assert recovered.checkpoint_tile('coding', '21', 0, [0, 1]) is None
    recovered.write_tile('coding', '21', [(0, [row()]), (1, [row()])], job_index=0)
    assert recovered.finish()['rows']['coding'] == 2


def test_abort_receipt_failure_still_releases_database_and_trait_owner(tmp_path,monkeypatch):
    writer=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    writer.write_tile('coding','21',[(0,[row()])],job_index=0)
    import fudan_wgs_toolkit.phewas_trait_csv as module
    original=module.atomic_json
    def fail(*args):raise OSError('receipt filesystem failure')
    monkeypatch.setattr(module,'atomic_json',fail)
    with pytest.raises(OSError,match='receipt filesystem'):writer.abort()
    assert not writer._owners and not writer.streams and writer._closed
    monkeypatch.setattr(module,'atomic_json',original)
    recovered=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,resume=True)
    assert recovered.checkpoint_tile('coding','21',0,[0]) is not None
    recovered.finish()


@pytest.mark.parametrize('options',[{'binding':{}},{'max_open_streams':0},{'verification_cache_size':0},{'minimum_free_bytes':-1}])
def test_invalid_writer_configuration_rejected(tmp_path,options):
    with pytest.raises(ValueError):PerTraitCheckpointCSVWriter(tmp_path,**dict({'binding':BINDING},**options))


def test_single_identity_and_ordinals_cannot_be_rebound(tmp_path):
    writer=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING)
    value=dict(CHR='21',POS=7,REF='A',ALT='C',pvalue=.3)
    writer.write_tile('individual','21',[(0,[value])],job_index=0,block_index=2,
                     block_identity='first',ordinal_before=3,ordinal_after=4)
    writer.abort()
    recovered=PerTraitCheckpointCSVWriter(tmp_path,binding=BINDING,resume=True)
    with pytest.raises(CommittedCSVCorruption):
        recovered.checkpoint_tile('individual','21',0,[0],block_index=2,block_identity='other',ordinal_before=3)
    recovered.abort()
