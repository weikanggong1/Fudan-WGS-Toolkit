"""Task PSS admission, ownership and live shared-node limits; no benchmark."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from fudan_wgs_toolkit import phewas_resources as r
from fudan_wgs_toolkit.phewas_run import _close_association_resources

G = r.GiB


@pytest.fixture
def task(monkeypatch,tmp_path):
    pid = os.getpid()
    identities = {pid:dict(pid=pid,start_ticks=1,boot_id='anonymous-task')}
    pss,rss,children,calls = {pid:20*G},{pid:180*G},{},[]
    node = dict(current_bytes=128*G,effective_current_bytes=128*G,
        limit_bytes=200*G,inactive_file_bytes=0,scope='cgroup_v2')
    constraints = [node,dict(node,current_bytes=300*G,effective_current_bytes=300*G,
        limit_bytes=512*G,scope='system_available_memory')]
    monkeypatch.setattr(r,'host_scope',lambda:('a'*64,{'anonymous_node':True}))
    monkeypatch.setattr(r,'process_identity',lambda value:identities.get(value))
    monkeypatch.setattr(r,'_verified_children',lambda value,**kwargs:children.get(value,[]))
    monkeypatch.setattr(r,'read_host_memory',lambda:dict(node,constraints=constraints))
    def counters(value,**kwargs):
        calls.append(value)
        return dict(pss_bytes=pss.get(value),rss_bytes=rss.get(value))
    monkeypatch.setattr(r,'_process_memory_counters',counters)
    gate = r.HostMemoryGate(tmp_path,host_limit_gib=100,reserve_gib=10)
    def register(value,amount):
        identities.setdefault(value,dict(pid=value,start_ticks=value,boot_id='anonymous-task'))
        pss[value],rss[value] = amount,180*G
        with monkeypatch.context() as scoped:
            scoped.setattr(r.os,'getpid',lambda:value)
            return r.HostMemoryGate(tmp_path,host_limit_gib=100,reserve_gib=10)
    return SimpleNamespace(gate=gate,pid=pid,identities=identities,pss=pss,rss=rss,
        children=children,calls=calls,node=node,constraints=constraints,register=register)


def test_shared_128_of_200_is_distinct_from_task_20_plus_future_20(task):
    with task.gate.lease(20*G) as lease:
        audit = lease.last_admission
        assert audit['task_memory']['pss_bytes']==20*G
        assert audit['task_memory']['rss_bytes']==180*G
        assert audit['pressure_constraints'][0]['capacity_bytes']==190*G
        assert audit['pressure_constraints'][0]['projected_bytes']==148*G
        assert audit['task_memory']['pss_coverage_complete']
        assert audit['task_memory']['sample_host_seconds']>=0
        assert audit['task_budget_is_cgroup_hard_limit'] is False


def test_task_95_of_100_rejects_new_bytes_despite_node_headroom(task):
    task.pss[task.pid]=95*G
    with pytest.raises(r.HostMemoryDeferred) as caught:
        with task.gate.lease(G):
            pytest.fail('task exceeded its 90 GiB usable budget')
    assert caught.value.available_bytes==0
    state=json.loads(task.gate.path.read_text())
    assert not state['leases'] and len(state['owners'])==1


def test_cached_peer_without_lease_is_counted_and_shared_mapping_rss_is_not(task):
    peer=task.register(90001,60*G)
    assert not json.loads(peer.path.read_text())['leases']
    with task.gate.lease(10*G) as lease:
        assert lease.last_admission['task_memory']['pss_bytes']==80*G
        assert lease.last_admission['task_memory']['rss_bytes']==360*G
    with pytest.raises(r.HostMemoryDeferred):
        with task.gate.lease(11*G):pass


def test_every_actual_ancestor_and_system_headroom_remains_live(task):
    task.constraints.append(dict(task.node,limit_bytes=250*G,
        current_bytes=239*G,effective_current_bytes=239*G))
    with pytest.raises(r.HostMemoryDeferred):
        with task.gate.lease(2*G):pass
    task.constraints.pop()
    task.constraints[1].update(current_bytes=503*G,effective_current_bytes=503*G)
    with pytest.raises(r.HostMemoryDeferred):
        with task.gate.lease(G):pass


def test_registered_worker_and_descendant_are_sampled_once(task):
    task.register(90001,10*G)
    task.children[task.pid]=[90001]
    with task.gate.lease(G) as lease:
        assert lease.last_admission['task_memory']['pss_bytes']==30*G
        assert lease.last_admission['task_memory']['observed_process_count']==2
    assert sorted(task.calls)==sorted([task.pid,90001])


def test_discovered_child_stays_owned_after_registered_parent_exits(task):
    task.register(90001,10*G)
    task.identities[90002]=dict(pid=90002,start_ticks=9,boot_id='anonymous-task')
    task.pss[90002],task.rss[90002]=15*G,50*G
    task.children[90001]=[90002]
    with task.gate.lease(G):pass
    task.identities.pop(90001)
    task.children.pop(90001)
    with task.gate.lease(G) as lease:
        assert lease.last_admission['task_memory']['pss_bytes']==35*G
        assert lease.last_admission['task_memory']['observed_process_count']==2


@pytest.mark.parametrize('change',['dead','start_ticks','boot_id'])
def test_recycled_or_dead_owner_cannot_keep_budget_or_count_unrelated_process(task,change):
    task.register(90001,80*G)
    with pytest.raises(r.HostMemoryDeferred):
        with task.gate.lease(G):pass
    if change=='dead':task.identities.pop(90001)
    else:task.identities[90001]=dict(task.identities[90001],**{change:99 if change=='start_ticks' else 'other-boot'})
    task.pss[90001]=900*G
    with task.gate.lease(G) as lease:
        assert lease.last_admission['task_memory']['pss_bytes']==20*G


@pytest.mark.parametrize('value',[None,-1,True])
def test_missing_or_invalid_pss_is_unverified_and_fails_closed(task,value):
    task.pss[task.pid]=value
    with pytest.raises(r.HostMemoryDeferred) as caught:
        with task.gate.lease(G):pass
    assert caught.value.reason=='task_pss_coverage_not_verified'
    assert not json.loads(task.gate.path.read_text())['leases']


def test_identity_change_while_pss_is_read_fails_closed(task,monkeypatch):
    def counters(pid,**kwargs):
        task.identities[pid]=dict(task.identities[pid],start_ticks=99)
        return dict(pss_bytes=0,rss_bytes=0)
    monkeypatch.setattr(r,'_process_memory_counters',counters)
    with pytest.raises(r.HostMemoryDeferred,match='task_pss_coverage'):
        with task.gate.lease(G):pass


def test_zero_shrinking_and_release_do_not_resample_or_wait_under_pressure(task,monkeypatch):
    with task.gate.lease(20*G) as lease:
        assert len(task.calls)==1
        task.pss[task.pid]=None
        task.node.update(current_bytes=199*G,effective_current_bytes=199*G)
        monkeypatch.setattr(r,'read_host_memory',lambda:(_ for _ in ()).throw(RuntimeError('unreadable node counters')))
        lease.set_future(10*G,phase='materialized')
        lease.set_future(0,phase='released')
        assert len(task.calls)==1
        assert lease.last_admission['task_memory'] is None
    assert len(task.calls)==1 and not json.loads(task.gate.path.read_text())['leases']
    assert task.gate.metadata()['pss_sampling_count']==1
    assert task.gate.metadata()['pss_sampling_host_seconds']>=0


def test_close_is_registration_specific_and_refuses_its_active_lease(task):
    peer=task.register(task.pid,20*G)
    with task.gate.lease(G):
        with pytest.raises(RuntimeError,match='active leases'):task.gate.close()
        peer.close()
        assert len(json.loads(peer.path.read_text())['owners'])==1
    task.gate.close();task.gate.close()
    assert not json.loads(task.gate.path.read_text())['owners']
    with pytest.raises(RuntimeError,match='closed'):task.gate.lease()


def test_loader_children_on_nonleader_thread_are_verified(tmp_path,monkeypatch):
    proc=tmp_path/'proc'
    for thread,child in ((1,2),(7,3)):
        folder=proc/'1/task'/str(thread);folder.mkdir(parents=True)
        (folder/'children').write_text(str(child))
        folder=proc/str(child);folder.mkdir(exist_ok=True)
        (folder/'stat').write_text(f'{child} (anonymous) S 1 0 0\n')
    real_path=Path
    monkeypatch.setattr(r,'Path',lambda value:real_path(str(value).replace('/proc',str(proc),1)))
    assert sorted(r._verified_children(1,strict=True))==[2,3]


def test_unreadable_system_memory_is_not_replaced_with_only_cgroup_limit(tmp_path,monkeypatch):
    group=tmp_path/'cgroup';group.mkdir()
    (group/'memory.max').write_text(str(200*G))
    (group/'memory.current').write_text(str(128*G))
    (group/'memory.stat').write_text('inactive_file 0\n')
    monkeypatch.setattr(r,'_memory_directories',lambda:[(2,group,group)])
    real_path=Path
    monkeypatch.setattr(r,'Path',lambda value:tmp_path/'missing_meminfo' if str(value)=='/proc/meminfo' else real_path(value))
    with pytest.raises(RuntimeError,match='physical available memory'):
        r.read_host_memory()


def test_close_after_model_cleanup_preserves_original_failure_and_owner_if_cleanup_fails():
    calls=[]
    writer=SimpleNamespace(abort=lambda:calls.append('abort'))
    repository=SimpleNamespace(close=lambda:calls.append('models_closed'))
    gate=SimpleNamespace(close=lambda:calls.append('gate_closed'))
    _close_association_resources(writer,repository,failed=True,gate=gate)
    assert calls==['abort','models_closed','gate_closed']
    calls.clear()
    def bad_close():
        calls.append('model_close_failed')
        raise OSError('still owns mappings')
    repository.close=bad_close
    _close_association_resources(writer,repository,failed=True,gate=gate)
    assert calls==['abort','model_close_failed']
