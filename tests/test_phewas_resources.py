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
        state["leases"]["foreign"] = dict(identity=identity,
            future_bytes=(total_gib-tree_gib)*G,peak_future_bytes=(total_gib-tree_gib)*G,
            phase="test")
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
        assert lease.total_bytes==90*G
        lease.set_future(30*G,phase="next_cohort")
        assert lease.required_bytes==30*G
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
        with gate.lease(181*G):
            pytest.fail("future allocation exceeds the immutable capacity")
    assert caught.value.capacity_bytes==180*G
    assert json.loads(gate.path.read_text())["leases"]=={}


def test_shared_process_rss_is_diagnostic_and_nested_owners_rejected(host):
    gate,memory,identities,rss=host
    rss[os.getpid()]=150*G
    # Shared file mappings can make summed RSS much larger than physical
    # cgroup use. This must not reject a reservation that fits live memory.
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


def test_unreadable_finite_cgroup_does_not_fall_back_to_a_wider_limit(tmp_path,monkeypatch):
    group=tmp_path/"finite";group.mkdir()
    (group/"memory.max").write_text(str(200*G))
    (group/"memory.stat").write_text("inactive_file 0\n")
    monkeypatch.setattr(resources,"_memory_directories",lambda:[(2,group,group)])
    # The omitted current counter must fail closed rather than silently
    # substitute an unconstrained machine's much larger physical limit.
    with pytest.raises(RuntimeError,match="finite cgroup"):
        resources.read_host_memory()
    (group/"memory.max").write_text("max")
    assert resources._memory_constraint(2,group) is None


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
    expected=5*1000*3000+7*1000*128+512*3000+256*2**20
    assert resources.shared_gene_workspace_bytes(1000,3000)==expected
    with pytest.raises(RuntimeError,match="consumer"):
        with gate.lease_factory(kind="coding",chromosome_index=0,job_index=3) as lease:
            lease.guard(n=1000,m=3000,phase="prepare")
            assert lease.required_bytes==expected
            raise RuntimeError("consumer failed")
    assert not json.loads(gate.path.read_text())["leases"]


def test_materialized_arrays_are_not_reserved_twice_and_zero_restores_capacity(host):
    gate,memory,*_=host
    # A 140 GiB unmaterialized allocation fits 40+140=180. Once 80 GiB is
    # initialized its remaining future must be 60, leaving the same total.
    with gate.lease(140*G) as lease:
        memory.update(current_bytes=120*G,effective_current_bytes=120*G)
        lease.set_future(60*G,phase="initialized_parts")
        assert lease.last_admission["future_bytes"]==60*G
        assert not lease.last_admission["pressure_check_performed"]
        assert lease.last_admission["pressure_constraints"]==[]
        assert lease.last_admission["live_effective_bytes"] is None
        memory.update(current_bytes=179*G,effective_current_bytes=179*G)
        # Releasing a cohort reservation cannot be deferred by unrelated
        # pressure, and the next mask does not inherit the old 140 GiB peak.
        lease.set_future(0,phase="cohort_released")
        assert lease.required_bytes==0
        memory.update(current_bytes=40*G,effective_current_bytes=40*G)
        lease.set_future(10*G,phase="next_cohort")
        assert lease.last_admission["future_bytes"]==10*G


def test_zero_future_does_not_claim_capacity_but_next_allocation_checks_pressure(host):
    gate,memory,*_=host
    memory.update(current_bytes=190*G,effective_current_bytes=190*G)
    with pytest.raises(resources.HostMemoryDeferred):
        with gate.lease() as lease:
            assert lease.required_bytes==0
            lease.set_future(G,phase="before_copy")
    assert not json.loads(gate.path.read_text())["leases"]


def test_zero_future_skips_live_sampling_but_keeps_durable_ledger(host,monkeypatch):
    gate,*_=host
    sync_calls=[]
    original_sync=resources.os.fsync
    def sync(fd):
        sync_calls.append(fd)
        return original_sync(fd)
    def unavailable():
        pytest.fail("a zero budget must not read live pressure or task PSS")
    monkeypatch.setattr(resources,"read_host_memory",unavailable)
    monkeypatch.setattr(resources.os,"fsync",sync)
    with gate.lease() as lease:
        for phase in ("gene_host_ready","gene_mask_released"):
            lease.set_future(0,phase=phase)
            sample=lease.last_admission
            assert sample["pressure_check_performed"] is False
            assert sample["pressure_check_skipped_reason"]=="zero_or_nonincreasing_future_budget"
            assert sample["capacity_bytes"] is None and sample["live_effective_bytes"] is None
            assert sample["pressure_constraints"]==[] and sample["future_bytes"]==0
            assert json.loads(gate.path.read_text())["leases"][lease.token]["phase"]==phase
        assert len(sync_calls)==3  # Initial zero plus both phase updates.
    assert len(sync_calls)==4 and not json.loads(gate.path.read_text())["leases"]


def test_decrease_and_equal_future_skip_sampling_but_growth_requires_fresh_pressure(host,monkeypatch):
    gate,memory,*_=host
    calls=[]
    def sampled():
        calls.append(dict(memory))
        return dict(memory)
    monkeypatch.setattr(resources,"read_host_memory",sampled)
    with gate.lease(10*G) as lease:
        assert len(calls)==1 and lease.last_admission["pressure_check_performed"]
        memory.update(current_bytes=190*G,effective_current_bytes=190*G)
        lease.set_future(5*G,phase="partly_initialized")
        lease.set_future(5*G,phase="same_future")
        lease.set_future(0,phase="released")
        assert len(calls)==1
        assert lease.last_admission["live_effective_bytes"] is None
        with pytest.raises(resources.HostMemoryDeferred):
            lease.set_future(G,phase="new_allocation")
        assert len(calls)==2 and calls[-1]["effective_current_bytes"]==190*G
    assert not json.loads(gate.path.read_text())["leases"]


def test_zero_budget_reaps_only_dead_foreign_identity_and_retains_live_future(host,monkeypatch):
    gate,_,identities,rss=host
    dead=add_foreign_lease(gate,identities,rss,pid=987650,total_gib=14)
    state=gate._load()
    state["leases"]["dead"]=state["leases"].pop("foreign")
    gate._write(state)
    identities.pop(dead["pid"])
    live=add_foreign_lease(gate,identities,rss,pid=987651,total_gib=24)
    monkeypatch.setattr(resources,"read_host_memory",lambda:pytest.fail("zero admission must not sample"))
    with gate.lease() as lease:
        state=json.loads(gate.path.read_text())
        assert set(state["leases"])=={"foreign",lease.token}
        assert state["leases"]["foreign"]["identity"]==live
        assert lease.last_admission["future_bytes"]==20*G
    assert set(json.loads(gate.path.read_text())["leases"])=={"foreign"}


def test_shrinking_keeps_own_identity_guard_and_original_ledger_on_failure(host,monkeypatch):
    gate,_,identities,_=host
    with gate.lease(G) as lease:
        original=identities[os.getpid()]
        identities[os.getpid()]=dict(original,start_ticks=original["start_ticks"]+1)
        monkeypatch.setattr(resources,"read_host_memory",lambda:pytest.fail("owner fails before sampling"))
        try:
            with pytest.raises(RuntimeError,match="identity changed"):
                lease.set_future(0,phase="release")
            assert json.loads(gate.path.read_text())["leases"][lease.token]["future_bytes"]==G
        finally:
            identities[os.getpid()]=original


def test_zero_admission_cannot_take_another_owners_token(host,monkeypatch):
    gate,_,identities,rss=host
    foreign=add_foreign_lease(gate,identities,rss,total_gib=14)
    lease=gate.lease()
    lease.token="foreign"
    monkeypatch.setattr(resources,"read_host_memory",lambda:pytest.fail("token owner fails before sampling"))
    with pytest.raises(RuntimeError,match="another owner"):
        lease.__enter__()
    record=json.loads(gate.path.read_text())["leases"]["foreign"]
    assert record["identity"]==foreign and record["future_bytes"]==10*G


def test_zero_admission_still_obeys_shared_ledger_lock(host,monkeypatch):
    gate,*_=host
    gate.lock_timeout_seconds=.05
    fd=os.open(gate.lock_path,os.O_CREAT|os.O_RDWR,0o600)
    monkeypatch.setattr(resources,"read_host_memory",lambda:pytest.fail("busy lock fails before sampling"))
    try:
        fcntl.flock(fd,fcntl.LOCK_EX)
        with pytest.raises(resources.HostMemoryDeferred,match="lock_busy"):
            with gate.lease():
                pytest.fail("zero budget must not bypass peer ledger lock")
    finally:
        fcntl.flock(fd,fcntl.LOCK_UN)
        os.close(fd)


def test_only_explicit_initialized_bytes_reduce_preparation_future(host):
    gate,memory,_,rss=host
    peak=resources.shared_gene_workspace_bytes(1000,3000)
    with gate.lease() as lease:
        # A reserved np.empty or growing reader RSS grants no automatic
        # credit. The caller has not completed any initialized allocation.
        rss[os.getpid()]=70*G
        lease.guard(n=1000,m=3000,materialized_bytes=0)
        assert lease.required_bytes==peak
        initialized_compact=1000*3000
        lease.guard(n=1000,m=3000,materialized_bytes=initialized_compact)
        assert lease.required_bytes==peak-initialized_compact
        assert lease.last_admission["materialized_credit_bytes"]==initialized_compact
        with pytest.raises(ValueError,match="exceed"):
            lease.guard(n=1000,m=3000,materialized_bytes=peak+1)


def test_legacy_full_workspace_ledger_cannot_be_misread_as_future_only(host):
    gate,*_=host
    state=gate._load()
    state["schema_version"]=1
    gate.path.write_text(json.dumps(state))
    with pytest.raises(ValueError,match="scope"):
        with gate.lease(G):
            pass


def test_shrinking_future_under_pressure_does_not_regrant_the_original_peak(host):
    gate,memory,*_=host
    with gate.lease(100*G) as lease:
        memory.update(current_bytes=190*G,effective_current_bytes=190*G)
        lease.set_future(50*G,phase="copy_finished")
        assert lease.required_bytes==50*G
        with pytest.raises(resources.HostMemoryDeferred):
            lease.set_future(51*G,phase="new_allocation")
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


@pytest.mark.parametrize("field,value", [("future_bytes",-1),("future_bytes",True),
    ("future_bytes","100"),("peak_future_bytes",-1),("peak_future_bytes",0),("phase",None),
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
    # A clean child uses real PID/start/boot/flock ownership without inheriting
    # initialized CUDA or threads from earlier GPU tests in the same suite.
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


@pytest.mark.skipif(not Path("/proc/self/stat").exists(),reason="Linux process ownership and flock required")
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


@pytest.mark.parametrize("kwargs", [dict(host_limit_gib=0),dict(host_limit_gib=200,reserve_gib=200),
    dict(host_limit_gib=float("nan")),dict(lock_timeout_seconds=0)])
def test_invalid_policy_is_rejected(tmp_path,kwargs):
    with pytest.raises(ValueError):
        resources.HostMemoryGate(tmp_path,**kwargs)
