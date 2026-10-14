"""Fresh whole-task admission and crash-safe local IPC; no GPU benchmark."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import pytest

from fudan_wgs_toolkit import phewas_resources as r


G = r.GiB


@pytest.fixture
def scoped_task(monkeypatch, tmp_path):
    parent = os.getpid()
    identities = {pid: dict(pid=pid, start_ticks=pid, boot_id='local-task-test')
                  for pid in (parent, 90001, 90002)}
    pss = {parent: 50*G, 90001: 20*G, 90002: 10*G}
    children = {parent: [90001, 90002]}
    calls, snapshots = [], []
    node = dict(limit_bytes=200*G, current_bytes=100*G,
                effective_current_bytes=100*G, inactive_file_bytes=0, scope='cgroup_v2')
    monkeypatch.setattr(r, 'host_scope', lambda: ('a'*64, {'anonymous_node': True}))
    monkeypatch.setattr(r, 'process_identity', lambda pid: identities.get(pid))
    monkeypatch.setattr(r, '_verified_children', lambda pid, **kwargs: children.get(pid, []))
    def counters(pid, **kwargs):
        calls.append(pid)
        return dict(pss_bytes=pss[pid], rss_bytes=pss[pid]*2)
    monkeypatch.setattr(r, '_process_memory_counters', counters)
    monkeypatch.setattr(r, 'read_host_memory', lambda: dict(node, constraints=[dict(node)]))
    def shared_pressure(task):
        snapshots.append(dict(task))
        # The private adapter's real-node policy uses exactly this already
        # sampled task endpoint; changing this local copy cannot alter gate PSS.
        task['pss_bytes'] = 0
        return r.read_host_memory()
    gate = r.HostMemoryGate(tmp_path/'scientific-output', host_limit_gib=100, reserve_gib=10,
        task_root_identity=identities[parent], host_memory_sampler=shared_pressure)
    return dict(gate=gate, identities=identities, pss=pss, children=children,
                calls=calls, snapshots=snapshots, parent=parent, node=node)


def test_controller_and_unregistered_children_count_once_per_fresh_growth(scoped_task):
    task = scoped_task
    gate = task['gate']
    with gate.lease(8*G) as lease:
        assert lease.last_admission['task_memory']['pss_bytes'] == 80*G
        assert lease.last_admission['task_memory']['observed_process_count'] == 3
        assert sorted(task['calls']) == sorted(task['identities'])
        assert len(task['snapshots']) == 1
        # Every later increase samples again: an endpoint from entry cannot
        # authorize allocations after other task buffers have materialized.
        task['pss'][90002] = 25*G
        with pytest.raises(r.HostMemoryDeferred):
            lease.set_future(9*G)
        assert task['snapshots'][-1]['pss_bytes'] == 95*G
        assert len(task['calls']) == 6
        assert gate.metadata()['pss_sampling_count'] == 2


def test_root_traversal_and_registered_peer_deduplicate_same_live_child(scoped_task, monkeypatch, tmp_path):
    task = scoped_task
    # Register a peer that root traversal already discovers. Its PSS must not
    # be counted twice, and its cached allocations count without a live lease.
    with monkeypatch.context() as child:
        child.setattr(r.os, 'getpid', lambda: 90001)
        peer = r.HostMemoryGate(tmp_path/'scientific-output', host_limit_gib=100, reserve_gib=10,
            task_root_identity=task['identities'][task['parent']],
            host_memory_sampler=lambda _: r.read_host_memory())
    with task['gate'].lease(10*G) as lease:
        assert lease.last_admission['task_memory']['pss_bytes'] == 80*G
        assert len(task['calls']) == 3
    peer.close()


def test_root_orphan_stays_owned_and_live_ancestor_headroom_still_constrains(scoped_task):
    task = scoped_task
    gate = task['gate']
    with gate.lease(G):
        pass
    task['children'][task['parent']] = []  # Exact observed orphan identity persists.
    with gate.lease(G) as lease:
        assert lease.last_admission['task_memory']['pss_bytes'] == 80*G
    task['node'].update(current_bytes=190*G, effective_current_bytes=190*G)
    with pytest.raises(r.HostMemoryDeferred):
        with gate.lease(G):
            pytest.fail('whole-task PSS headroom cannot replace shared-node headroom')


def test_changed_root_identity_never_admits_using_earlier_snapshot(scoped_task):
    task = scoped_task
    with task['gate'].lease(G) as lease:
        task['identities'][task['parent']] = dict(task['identities'][task['parent']], start_ticks=0)
        with pytest.raises(RuntimeError, match='owner identity changed'):
            lease.set_future(2*G)


def test_worker_growth_rejects_changed_controller_identity(scoped_task, monkeypatch, tmp_path):
    task = scoped_task
    with monkeypatch.context() as child:
        child.setattr(r.os, 'getpid', lambda: 90001)
        gate = r.HostMemoryGate(tmp_path/'worker-output', host_limit_gib=100, reserve_gib=10,
            task_root_identity=task['identities'][task['parent']],
            host_memory_sampler=lambda _: r.read_host_memory())
        with gate.lease(G) as lease:
            task['identities'][task['parent']] = dict(task['identities'][task['parent']], start_ticks=0)
            with pytest.raises(r.HostMemoryDeferred, match='task_pss_coverage'):
                lease.set_future(2*G)


def test_controller_cannot_change_while_other_process_pss_is_sampled(scoped_task, monkeypatch):
    task = scoped_task
    old_counters = r._process_memory_counters
    def changes_mid_scan(pid, **kwargs):
        value = old_counters(pid, **kwargs)
        if pid == 90002:
            task['identities'][task['parent']] = dict(task['identities'][task['parent']], start_ticks=0)
        return value
    monkeypatch.setattr(r, '_process_memory_counters', changes_mid_scan)
    with pytest.raises(r.HostMemoryDeferred, match='task_pss_coverage'):
        with task['gate'].lease(G):
            pytest.fail('the task root changed during the full scan')


def test_controller_cannot_change_during_host_policy_callback(scoped_task):
    task = scoped_task
    def changes_during_node_read(snapshot):
        task['identities'][task['parent']] = dict(task['identities'][task['parent']], start_ticks=0)
        return r.read_host_memory()
    task['gate'].host_memory_sampler = changes_during_node_read
    with pytest.raises(RuntimeError, match='root identity changed during host-pressure'):
        with task['gate'].lease(G):
            pytest.fail('a callback must not admit after root identity changes')


def test_dead_root_rebinding_waits_for_all_remembered_orphans(scoped_task, tmp_path):
    task = scoped_task
    with task['gate'].lease(G):
        pass  # Persists the exact descendant identities in the old scope.
    parent = task['parent']
    task['identities'][parent] = dict(task['identities'][parent], start_ticks=parent+1)
    def replacement():
        return r.HostMemoryGate(tmp_path/'scientific-output', host_limit_gib=100, reserve_gib=10,
            task_root_identity=task['identities'][parent],
            host_memory_sampler=lambda _: r.read_host_memory())
    with pytest.raises(ValueError, match='one task root identity'):
        replacement()
    task['identities'].pop(90001); task['identities'].pop(90002)
    recovered = replacement()
    with recovered.lease(G) as lease:
        assert lease.last_admission['task_memory']['observed_process_count'] == 1
        assert lease.last_admission['task_memory']['pss_bytes'] == 50*G
    recovered.close()


def test_local_ipc_is_atomic_without_fsync_and_equal_zero_updates_are_coalesced(monkeypatch, tmp_path):
    # Real current-process PSS and real mount verification; only count fsync.
    syncs = []
    monkeypatch.setattr(r.os, 'fsync', lambda fd: syncs.append(fd))
    local = tmp_path/'local-ledgers'
    gate = r.HostMemoryGate(tmp_path/'output', host_limit_gib=2, reserve_gib=.125,
        ledger_directory=local, ledger_durable=False, task_root_identity=r.process_identity(os.getpid()))
    assert gate.path.parent.parent == local
    with gate.lease(2**20) as lease:
        lease.set_future(0, phase='materialized')
        after_materialization = gate.metadata()['ledger_writes']
        for phase in ('mask_finished', 'job_finished', 'cleanup_finished'):
            lease.set_future(0, phase=phase)
            state = json.loads(gate.path.read_text())
            assert state['leases'][lease.token]['future_bytes'] == 0
            assert lease.last_admission['phase'] == phase
            assert lease.last_admission['ledger_phase_coalesced']
        assert gate.metadata()['ledger_writes'] == after_materialization
        assert gate.metadata()['ledger_noop_publications'] == 3
        assert not list(gate.directory.glob('*.tmp'))
    gate.close()
    assert not syncs and gate.metadata()['ledger_fsyncs'] == 0
    assert not json.loads(gate.path.read_text())['leases']


def test_same_local_ipc_root_cannot_merge_different_scientific_tasks(tmp_path):
    local = tmp_path/'ipc'
    first = r.HostMemoryGate(tmp_path/'one', host_limit_gib=2, reserve_gib=.125,
        ledger_directory=local, ledger_durable=False)
    second = r.HostMemoryGate(tmp_path/'two', host_limit_gib=2, reserve_gib=.125,
        ledger_directory=local, ledger_durable=False)
    assert first.path != second.path
    with first.lease(2**20), second.lease(2**20):
        assert len(json.loads(first.path.read_text())['leases']) == 1
        assert len(json.loads(second.path.read_text())['leases']) == 1
    first.close(); second.close()


@pytest.mark.parametrize('filesystem', ['nfs', 'lustre', 'fuse.cpfs', '9p', 'unknown'])
def test_local_mount_verification_rejects_distributed_or_unknown_type(monkeypatch, tmp_path, filesystem):
    info = tmp_path/'mountinfo'
    info.write_text('1 0 0:1 / / rw - ext4 /dev/test rw\n'
                    f'2 1 0:2 / {tmp_path} rw - {filesystem} anonymous rw\n')
    original = Path
    monkeypatch.setattr(r, 'Path', lambda value: info if str(value) == '/proc/self/mountinfo' else original(value))
    with pytest.raises(ValueError, match='verified node-local'):
        r._local_ledger_filesystem(tmp_path/'ledger')


@pytest.mark.parametrize('kwargs', [dict(ledger_durable=False), dict(ledger_durable='false'),
    dict(host_memory_sampler=False), dict(task_root_identity={'pid': 1})])
def test_invalid_explicit_ipc_and_snapshot_contracts_fail_closed(tmp_path, kwargs):
    with pytest.raises((ValueError, TypeError)):
        r.HostMemoryGate(tmp_path, host_limit_gib=2, reserve_gib=.125, **kwargs)


def _local_contender(logical, ipc, send, hold):
    r.read_host_memory = lambda: dict(limit_bytes=2*G, current_bytes=0,
        effective_current_bytes=0, inactive_file_bytes=0, scope='cgroup_v2')
    gate = r.HostMemoryGate(logical, host_limit_gib=2, reserve_gib=.125,
                           ledger_directory=ipc, ledger_durable=False)
    try:
        with gate.lease(int(1.2*G)):
            send.send('admitted')
            if hold:
                time.sleep(30)
    except r.HostMemoryDeferred:
        send.send('deferred')
    finally:
        send.close()


@pytest.mark.skipif(not Path('/proc/self/stat').exists(), reason='Linux identity and flock required')
def test_real_local_competitors_cannot_double_reserve_and_crashed_owner_is_reaped(tmp_path):
    # Spawn avoids inheriting imported CPU pools/CUDA. Keep these processes
    # entirely local and disposable; no analysis worker or external task exists.
    context = mp.get_context('spawn')
    pipes = [context.Pipe(duplex=False) for _ in range(3)]
    logical, ipc = str(tmp_path/'output'), str(tmp_path/'ipc')
    children = [context.Process(target=_local_contender, args=(logical, ipc, tx, i == 0))
                for i, (_, tx) in enumerate(pipes)]
    try:
        children[0].start()
        assert pipes[0][0].poll(60) and pipes[0][0].recv() == 'admitted'
        children[1].start(); children[1].join(60)
        assert children[1].exitcode == 0
        assert pipes[1][0].poll(2) and pipes[1][0].recv() == 'deferred'
        # The local ledger survives this process crash and exact PID/start/boot
        # reaping permits a new owner; disabling fsync does not weaken flock.
        children[0].kill(); children[0].join(10)
        children[2].start(); children[2].join(60)
        assert children[2].exitcode == 0
        assert pipes[2][0].poll(2) and pipes[2][0].recv() == 'admitted'
    finally:
        for child in children:
            if child.pid:
                if child.is_alive():
                    child.kill()
                child.join(10)
        for pipe in pipes:
            for end in pipe:
                end.close()
