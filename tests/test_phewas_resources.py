"""Host admission contracts; these do not benchmark the population workload."""
import json
import fcntl
import multiprocessing as mp
import os
from pathlib import Path
import time

import pytest
from fudan_wgs_toolkit import phewas_resources as resources


G = resources.GiB


@pytest.fixture
def host(monkeypatch,tmp_path):
    owner = dict(pid=os.getpid(),start_ticks=1,boot_id="anonymous-test-boot")
    identities,rss = {owner["pid"]:owner},{owner["pid"]:2*G}
    memory = dict(current_bytes=40*G,limit_bytes=200*G,inactive_file_bytes=0,
                  effective_current_bytes=40*G,scope="cgroup_v2")
    monkeypatch.setattr(resources,"host_scope",lambda:("f"*64,dict(node="anonymous",boot_id="b",cgroups=[])))
    monkeypatch.setattr(resources,"process_identity",lambda pid:identities.get(pid))
    monkeypatch.setattr(resources,"process_tree_rss",lambda identity:rss[identity["pid"]])
    monkeypatch.setattr(resources,"read_host_memory",lambda:dict(memory))
    gate = resources.HostMemoryGate(tmp_path,host_limit_gib=200,reserve_gib=20)
    return gate,memory,identities,rss


def add_foreign_lease(gate,identities,rss,*,pid=987654,tree_gib=4,total_gib=14,start_ticks=5):
    identity=dict(pid=pid,start_ticks=start_ticks,boot_id="anonymous-test-boot")
    identities[pid],rss[pid]=identity,tree_gib*G
    with gate._locked(0):
        state=gate._load()
        state["leases"]["foreign"] = dict(identity=identity,total_bytes=total_gib*G,
                                          baseline_rss_bytes=tree_gib*G,
                                          workspace_bytes=(total_gib-tree_gib)*G)
        gate._write(state)
    return identity


def test_future_leases_do_not_treat_reader_rss_growth_as_materialized_genotype(host):
    gate,memory,identities,rss=host
    with gate.lease(64*G) as lease:
        assert lease.last_admission["capacity_bytes"]==180*G
        assert lease.last_admission["future_bytes"]==64*G
        rss[os.getpid()]=52*G  # Its cause is unknown: reader/model or genotype.
        memory.update(current_bytes=90*G,effective_current_bytes=90*G)
        lease.grow(90*G)
        assert lease.last_admission["future_bytes"]==90*G
        assert lease.last_admission["materialized_credit_bytes"]==0
        assert lease.total_bytes==92*G
        lease.grow(30*G)  # A smaller mask cannot shrink a still-live owner lease.
        assert lease.required_bytes==90*G
    assert json.loads(gate.path.read_text())["leases"]=={}


def test_growth_defers_without_retaining_the_small_lease(host):
    gate,memory,identities,rss=host
    memory.update(current_bytes=165*G,effective_current_bytes=165*G)
    add_foreign_lease(gate,identities,rss)
    with pytest.raises(resources.HostMemoryDeferred) as caught:
        with gate.lease() as lease:
            lease.grow(15*G)
    assert caught.value.required_bytes==15*G and lease.closed
    assert set(json.loads(gate.path.read_text())["leases"])=={"foreign"}
    lease.release()  # Runtime finally is allowed to release again.
    with pytest.raises(RuntimeError,match="not active"):
        lease.grow(0)


def test_impossible_capacity_is_not_an_infinite_temporary_retry(host):
    gate,*_=host
    with pytest.raises(resources.HostMemoryRequestTooLarge) as caught:
        with gate.lease(179*G):
            pytest.fail("owner RSS plus requested peak cannot fit")
    assert caught.value.capacity_bytes==180*G
    assert json.loads(gate.path.read_text())["leases"]=={}


def test_current_rss_is_independently_guarded_and_nested_owners_rejected(host):
    gate,memory,identities,rss=host
    rss[os.getpid()]=150*G
    with pytest.raises(resources.HostMemoryRequestTooLarge):
        with gate.lease(31*G):
            pass
    rss[os.getpid()]=2*G
    with gate.lease(1*G):
        with pytest.raises(RuntimeError,match="nested"):
            with gate.lease():
                pass


@pytest.mark.parametrize("reuse", ["start_ticks","boot_id","dead"])
def test_stale_owner_pid_boot_or_exit_does_not_keep_a_future_budget(host,reuse):
    gate,memory,identities,rss=host
    original=add_foreign_lease(gate,identities,rss,total_gib=170)
    if reuse=="dead":
        identities.pop(original["pid"])
    else:
        identities[original["pid"]]=dict(original,**{reuse:99 if reuse=="start_ticks" else "new-boot"})
    with gate.lease(50*G) as lease:
        assert lease.last_admission["active_leases"]==1


def test_private_ledger_scopes_and_limit_policy_are_bound(host,tmp_path,monkeypatch):
    gate,*_=host
    with gate.lease(1*G):
        other=resources.HostMemoryGate(tmp_path,host_limit_gib=250,reserve_gib=20)
        with pytest.raises(ValueError,match="same limit/reserve"):
            with other.lease():
                pass
        monkeypatch.setattr(resources,"host_scope",lambda:("e"*64,dict(node="other-node",boot_id="other",cgroups=[])))
        other=resources.HostMemoryGate(tmp_path,host_limit_gib=250,reserve_gib=20)
        with other.lease(1*G):
            assert other.path!=gate.path
        state=json.loads(gate.path.read_text())
        state["scope_key"]="wrong"
        gate.path.write_text(json.dumps(state))
        with pytest.raises(ValueError,match="scope"):
            gate._load()
        # Restore the private fixture before its outer finally.
        state["scope_key"]=gate.key
        gate.path.write_text(json.dumps(state))


def test_cgroup_pressure_excludes_inactive_file_and_obeys_parent(tmp_path,monkeypatch):
    mount=tmp_path/"memory";leaf=mount/"child";leaf.mkdir(parents=True)
    for directory,limit,current,inactive in ((mount,250,220,0),(leaf,200,180,50)):
        (directory/"memory.max").write_text(str(limit*G))
        (directory/"memory.current").write_text(str(current*G))
        (directory/"memory.stat").write_text(f"inactive_file {inactive*G}\n")
    monkeypatch.setattr(resources,"_memory_directories",lambda:[(2,leaf,mount)])
    report=resources.read_host_memory()
    cgroups=[c for c in report["constraints"] if c["scope"]=="cgroup_v2"]
    assert [c["effective_current_bytes"] for c in cgroups]==[130*G,220*G]
    assert [c["limit_bytes"] for c in cgroups]==[200*G,250*G]
    v1=tmp_path/"v1";v1.mkdir()
    (v1/"memory.limit_in_bytes").write_text(str(200*G))
    (v1/"memory.usage_in_bytes").write_text(str(180*G))
    (v1/"memory.stat").write_text(f"inactive_file {G}\ntotal_inactive_file {40*G}\n")
    assert resources._memory_constraint(1,v1)["effective_current_bytes"]==140*G


def test_physical_ram_usage_is_not_compared_with_the_project_limit(host,monkeypatch):
    gate,memory,*_=host
    system=dict(current_bytes=800*G,limit_bytes=1024*G,inactive_file_bytes=0,
                effective_current_bytes=800*G,scope="system_available_memory")
    monkeypatch.setattr(resources,"read_host_memory",lambda:dict(memory,constraints=[dict(memory),system]))
    with gate.lease(10*G):
        pass
    parent=dict(memory,limit_bytes=250*G,effective_current_bytes=240*G,current_bytes=240*G)
    monkeypatch.setattr(resources,"read_host_memory",lambda:dict(memory,constraints=[dict(memory),parent,system]))
    with pytest.raises(resources.HostMemoryDeferred):
        with gate.lease(10*G):
            pass


def test_gene_workspace_contract_and_exception_unwind(host):
    gate,*_=host
    expected=5*1000*3000+4*1000*128+512*3000+256*2**20
    assert resources.shared_gene_workspace_bytes(1000,3000)==expected
    with pytest.raises(RuntimeError,match="consumer"):
        with gate.lease_factory(kind="coding",chromosome_index=0,job_index=3) as lease:
            lease.guard(n=1000,m=3000,phase="prepare")
            assert lease.required_bytes==expected
            raise RuntimeError("consumer failed")
    assert not json.loads(gate.path.read_text())["leases"]


def test_lock_timeout_unwinds_immediately_and_retry_retires_only_its_own_token(host):
    gate,*_=host
    gate.lock_timeout_seconds=.05
    with pytest.raises(resources.HostMemoryDeferred,match="lock_busy"):
        with gate.lease(G) as first:
            fd=os.open(gate.lock_path,os.O_RDWR)
            try:
                fcntl.flock(fd,fcntl.LOCK_EX)
                first.grow(2*G)
            finally:
                fcntl.flock(fd,fcntl.LOCK_UN)
                os.close(fd)
    assert first.closed and first.token in gate._retired_tokens
    with gate.lease(G) as second:
        state=json.loads(gate.path.read_text())
        assert set(state["leases"])=={second.token}
    assert not json.loads(gate.path.read_text())["leases"]


@pytest.mark.parametrize("field,value", [("workspace_bytes",-1),("workspace_bytes",True),
    ("workspace_bytes","100"),("baseline_rss_bytes",-1),("total_bytes",0),
    ("identity",dict(pid=True,start_ticks=1,boot_id="test"))])
def test_corrupt_ledger_cannot_reduce_the_shared_future_budget(host,field,value):
    gate,_,identities,rss=host
    add_foreign_lease(gate,identities,rss)
    state=json.loads(gate.path.read_text())
    state["leases"]["foreign"][field]=value
    gate.path.write_text(json.dumps(state))
    with pytest.raises(ValueError,match="ledger"):
        with gate.lease(G):
            pytest.fail("corrupt ledger must fail before admission")


def _competing_child(directory,connection,hold):
    # Spawned contract is CPU-only and uses real PID/start/boot/fcntl ownership.
    resources.read_host_memory=lambda:dict(current_bytes=0,limit_bytes=2*G,
        inactive_file_bytes=0,effective_current_bytes=0,scope="cgroup_v2")
    gate=resources.HostMemoryGate(directory,host_limit_gib=2,reserve_gib=.125)
    try:
        with gate.lease(int(1.2*G)):
            connection.send("admitted")
            if hold:
                time.sleep(30)
    except resources.HostMemoryDeferred:
        connection.send("deferred")
    finally:
        connection.close()


@pytest.mark.skipif(not hasattr(os,"fork"),reason="Linux process ownership and flock required")
def test_real_cross_process_lease_cannot_double_admit_and_reaps_dead_owner(tmp_path):
    context=mp.get_context("spawn")
    first_rx,first_tx=context.Pipe(duplex=False)
    second_rx,second_tx=context.Pipe(duplex=False)
    third_rx,third_tx=context.Pipe(duplex=False)
    first=context.Process(target=_competing_child,args=(str(tmp_path),first_tx,True))
    second=context.Process(target=_competing_child,args=(str(tmp_path),second_tx,False))
    try:
        first.start()
        # Spawn imports the numerical runtime on a shared filesystem.
        assert first_rx.poll(60)
        assert first_rx.recv()=="admitted"
        second.start();second.join(60)
        assert not second.is_alive() and second.exitcode==0
        assert second_rx.poll(2) and second_rx.recv()=="deferred"
        # Kill only the disposable child created by this test; its old lease
        # remains on disk, then is reaped using exact process identity.
        first.kill();first.join(5)
        third=context.Process(target=_competing_child,args=(str(tmp_path),third_tx,False))
        third.start();third.join(60)
        assert not third.is_alive() and third.exitcode==0
        assert third_rx.poll(2) and third_rx.recv()=="admitted"
    finally:
        for child in (first,second,locals().get("third")):
            if child is not None and child.pid:
                if child.is_alive():
                    child.kill()
                child.join(5)
        for connection in (first_rx,first_tx,second_rx,second_tx,third_rx,third_tx):
            connection.close()


@pytest.mark.parametrize("kwargs", [dict(host_limit_gib=0),dict(reserve_gib=200),
    dict(host_limit_gib=float("nan")),dict(lock_timeout_seconds=0)])
def test_invalid_policy_is_rejected(tmp_path,kwargs):
    with pytest.raises(ValueError):
        resources.HostMemoryGate(tmp_path,**kwargs)
