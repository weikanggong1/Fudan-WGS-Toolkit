"""Single host-output admission and one-write error contracts, not benchmarks."""
import json
import os

import pytest

from fudan_wgs_toolkit import phewas_resources as resources


@pytest.fixture
def output_gate(monkeypatch,tmp_path):
    identity = dict(pid=os.getpid(),start_ticks=7,boot_id="component-test-boot")
    memory = dict(current_bytes=0,limit_bytes=resources.GiB,inactive_file_bytes=0,
                  effective_current_bytes=0,scope="cgroup_v2")
    monkeypatch.setattr(resources,"host_scope",lambda:("c"*64,dict(node="test",cgroups=[])))
    monkeypatch.setattr(resources,"process_identity",lambda pid:identity if pid==identity["pid"] else None)
    monkeypatch.setattr(resources,"_process_memory_counters",lambda pid,**kwargs:dict(pss_bytes=0,rss_bytes=0))
    monkeypatch.setattr(resources,"read_host_memory",lambda:dict(memory))
    gate = resources.HostMemoryGate(tmp_path,host_limit_gib=1,reserve_gib=0)
    return gate,memory


def test_output_forecast_is_bounded_tile_rows_not_population_genotype():
    assert resources.single_output_workspace_bytes(1024,16)==1024*16*(2048+256)+64*1024
    assert resources.single_output_workspace_bytes(0,16)==0
    assert resources.single_output_workspace_bytes(3,2,bytes_per_row=100,
        packed_bytes_per_row=8,scratch_bytes=10)==658


def test_materialized_rows_zero_future_without_rss_double_accounting(output_gate):
    gate,memory = output_gate
    with resources.host_output_lease_scope(gate.lease_factory,1024,16,
            chromosome_index=0,job_index=2) as reservation:
        required = resources.single_output_workspace_bytes(1024,16)
        assert reservation.lease.required_bytes==required
        memory.update(current_bytes=required,effective_current_bytes=required)
        reservation.materialized()
        assert reservation.lease.required_bytes==0
        reservation.materialized()  # Idempotent, never writes the ledger twice.
        assert reservation.rows_materialized
    assert json.loads(gate.path.read_text())["leases"]=={}


def test_admission_retries_before_outputs_and_writer_execute_once(output_gate,monkeypatch):
    gate,memory = output_gate
    memory.update(current_bytes=resources.GiB-1,effective_current_bytes=resources.GiB-1)
    waits,owners,writes = [],[],[]
    def factory(*args):
        lease = gate.lease_factory(*args)
        owners.append(lease)
        return lease
    def sleep(seconds):
        assert json.loads(gate.path.read_text())["leases"]=={}
        assert not writes
        memory.update(current_bytes=0,effective_current_bytes=0)
    monkeypatch.setattr(resources.time,"sleep",sleep)
    with resources.host_output_lease_scope(factory,1024,16,chromosome_index=0,
            job_index=2,on_wait=waits.append) as reservation:
        reservation.materialized()
        writes.append("one accepted tile")
    assert len(owners)==2 and owners[0].closed and owners[1].closed
    assert len(waits)==1 and waits[0]["event"]=="single_host_output_wait"
    assert writes==["one accepted tile"]


def test_deferred_calculation_is_not_retried_after_scope_entry(output_gate):
    gate,_ = output_gate
    writes = []
    with pytest.raises(resources.HostMemoryDeferred,match="component_injected"):
        with resources.host_output_lease_scope(gate.lease_factory,4,1,
                chromosome_index=0,job_index=0):
            writes.append("prefix published once")
            raise resources.HostMemoryDeferred(1,reason="component_injected")
    assert writes==["prefix published once"]
    assert json.loads(gate.path.read_text())["leases"]=={}


def test_writer_failure_is_not_masked_by_cleanup_failure(output_gate,monkeypatch):
    gate,_ = output_gate
    with pytest.raises(ArithmeticError,match="writer primary"):
        with resources.host_output_lease_scope(gate.lease_factory,2,1,
                chromosome_index=0,job_index=0) as reservation:
            monkeypatch.setattr(reservation.lease,"release",lambda:(_ for _ in ()).throw(
                RuntimeError("secondary cleanup")))
            raise ArithmeticError("writer primary")


def test_impossible_tile_fails_before_write_without_retry(output_gate,monkeypatch):
    gate,_ = output_gate
    monkeypatch.setattr(resources.time,"sleep",lambda seconds:pytest.fail("capacity errors must not retry"))
    with pytest.raises(resources.HostMemoryRequestTooLarge) as caught:
        with resources.host_output_lease_scope(gate.lease_factory,1024*1024,16,
                chromosome_index=0,job_index=0):
            pytest.fail("an impossible output must not run")
    assert caught.value.reason=="single_output_tile_exceeds_host_capacity"
    assert json.loads(gate.path.read_text())["leases"]=={}


def test_explicit_zero_wait_reports_temporary_pressure_without_writing(output_gate):
    gate,memory = output_gate
    memory.update(current_bytes=resources.GiB,effective_current_bytes=resources.GiB)
    with pytest.raises(resources.HostMemoryDeferred) as caught:
        with resources.host_output_lease_scope(gate.lease_factory,2,1,
                chromosome_index=0,job_index=0,maximum_wait_seconds=0):
            pytest.fail("zero wait cannot run under pressure")
    assert caught.value.reason=="single_output_admission_timeout"


def test_optional_gate_preserves_one_tile_contract():
    with resources.host_output_lease_scope(None,2,1,chromosome_index=0,job_index=0) as reservation:
        assert reservation.lease is None
        reservation.materialized()
        assert reservation.rows_materialized


@pytest.mark.parametrize("override",[{"rows":-1},{"traits":0},{"rows":True},
    {"retry_seconds":61},{"retry_seconds":True},{"maximum_wait_seconds":float("inf")},
    {"maximum_wait_seconds":-1},{"chromosome_index":-1},{"job_index":True}])
def test_invalid_output_policy_fails_before_factory(override):
    values=dict(rows=2,traits=1,chromosome_index=0,job_index=0)
    values.update(override)
    with pytest.raises(ValueError):
        with resources.host_output_lease_scope(lambda *args:pytest.fail("invalid arguments called factory"),**values):
            pass
