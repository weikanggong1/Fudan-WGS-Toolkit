"""Anonymous live-resource endpoints and real backing allocation arithmetic."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from fudan_wgs_toolkit import phewas_resources as r


G=r.GiB


def setup_host(monkeypatch,constraints):
    identity=dict(pid=os.getpid(),start_ticks=1,boot_id='test')
    monkeypatch.setattr(r,'host_scope',lambda:('e'*64,dict(node='test')))
    monkeypatch.setattr(r,'process_identity',lambda pid:identity if pid==identity['pid'] else None)
    monkeypatch.setattr(r,'read_host_memory',lambda:dict(constraints[0],constraints=constraints))


def constraint(limit,current=0,scope='cgroup_v2'):
    return dict(limit_bytes=limit,current_bytes=current,effective_current_bytes=current,
                inactive_file_bytes=0,scope=scope)


def test_automatic_host_policy_uses_smallest_ancestor_or_physical_capacity(monkeypatch,tmp_path):
    setup_host(monkeypatch,[constraint(80*G),constraint(20*G),constraint(128*G,scope='system_available_memory')])
    gate=r.HostMemoryGate(tmp_path)
    assert gate.limit_bytes==20*G and gate.reserve_bytes==2*G
    with gate.lease(1*G) as lease:
        assert lease.last_admission['automatic_limit']
        assert lease.last_admission['automatic_reserve']
        assert lease.last_admission['capacity_bytes']==18*G
    assert json.loads(gate.path.read_text())['policy']==dict(limit_bytes=20*G,reserve_bytes=2*G)


def test_explicit_host_limit_and_reserve_remain_exact(monkeypatch,tmp_path):
    setup_host(monkeypatch,[constraint(80*G),constraint(128*G,scope='system_available_memory')])
    gate=r.HostMemoryGate(tmp_path,host_limit_gib=12,reserve_gib=3)
    assert gate.limit_bytes==12*G and gate.reserve_bytes==3*G
    with gate.lease(8*G):pass
    with pytest.raises(r.HostMemoryRequestTooLarge):
        with gate.lease(10*G):pass


def test_automatic_host_capacity_zero_allows_release_but_no_new_allocation(monkeypatch,tmp_path):
    setup_host(monkeypatch,[constraint(0),constraint(128*G,scope='system_available_memory')])
    gate=r.HostMemoryGate(tmp_path)
    assert gate.limit_bytes==gate.reserve_bytes==0
    with gate.lease() as lease:lease.set_future(0,phase='release')
    with pytest.raises(r.HostMemoryRequestTooLarge):
        with gate.lease(1):pass


def test_zero_cgroup_maximum_is_finite_and_reports_anonymous_mapping_counters(tmp_path):
    (tmp_path/'memory.max').write_text('0')
    (tmp_path/'memory.current').write_text('0')
    (tmp_path/'memory.stat').write_text('anon 0\nfile 0\nfile_mapped 0\nshmem 0\ninactive_file 0\n')
    result=r._memory_constraint(2,tmp_path)
    assert result['limit_bytes']==0 and result['anonymous_bytes']==0
    assert result['mapped_file_bytes']==0 and result['current_bytes_include_shared_background']
    assert str(tmp_path) not in json.dumps(result)


def test_process_counter_reads_distinguish_missing_pss_from_zero(monkeypatch,tmp_path):
    folder=tmp_path/'123';folder.mkdir()
    (folder/'status').write_text('VmRSS: 50 kB\nRssAnon: 30 kB\nRssFile: 18 kB\nRssShmem: 2 kB\nVmSwap: 1 kB\n')
    real_path=Path
    monkeypatch.setattr(r,'Path',lambda value:real_path(str(value).replace('/proc',str(tmp_path),1)))
    result=r._process_memory_counters(123,include_pss=True)
    assert result['rss_bytes']==50*1024 and result['pss_bytes'] is None
    (folder/'smaps_rollup').write_text('Rss: 50 kB\nPss: 40 kB\nPss_Anon: 30 kB\nPss_File: 9 kB\nPss_Shmem: 1 kB\nSwapPss: 0 kB\n')
    result=r._process_memory_counters(123,include_pss=True)
    assert result['pss_bytes']==40*1024 and result['mapped_file_pss_bytes']==9*1024
    assert r._process_memory_counters(123,include_pss=False)['pss_bytes'] is None


def test_process_tree_coverage_identity_brackets_and_private_output(monkeypatch):
    identities={pid:dict(pid=pid,start_ticks=pid,boot_id='test') for pid in (1,2,3)}
    monkeypatch.setattr(r,'process_identity',lambda pid:identities.get(pid))
    monkeypatch.setattr(r,'_verified_children',lambda pid:[2,3] if pid==1 else [])
    def counters(pid,include_pss):
        if pid==3:identities.pop(3)
        return dict(rss_bytes=100,pss_bytes=None if pid==2 else 50)
    monkeypatch.setattr(r,'_process_memory_counters',counters)
    result=r.process_memory_snapshot(identities[1])
    assert result['observed_process_count']==2 and result['rss_bytes']==200
    assert result['pss_bytes'] is None and result['observed_subtotals']['pss_bytes']==50
    assert result['counter_process_coverage']['pss_bytes']==1
    encoded=json.dumps(result)
    assert 'start_ticks' not in encoded and 'boot_id' not in encoded and '"pid"' not in encoded
    assert not result['rss_sum_is_unique_memory']


def test_tensor_backing_ownership_deduplicates_views_without_copying():
    dose=torch.ones((4,10),dtype=torch.uint8)
    array=np.arange(100,dtype=np.int64)
    result=r.tensor_storage_footprint([dose[:,:1],dose[:,1:],array[:1],array[99:]])
    assert result['retained_storage_bytes']==40+800
    assert result['retained_storage_bytes_by_device']=={'cpu':840}
    assert result['ownership_coverage_is_explicit']


def test_live_gpu_guard_does_not_add_owned_or_reserved_bytes_again(monkeypatch):
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda _:100)
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda _:140)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda _:(200,500))
    result=r.sample_gpu_memory('cuda:0',limit_bytes=300,reserve_bytes=20,required_new_bytes=180)
    assert result['available_new_bytes']==180
    assert result['new_allocation_admitted']
    assert result['owned_storage_already_in_allocated']
    result=r.sample_gpu_memory('cuda:0',limit_bytes=300,reserve_bytes=20,required_new_bytes=181)
    assert not result['new_allocation_admitted']
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda _:(10,500))
    assert r.sample_gpu_memory('cuda:0',reserve_bytes=20)['available_new_bytes']==30


def test_optional_telemetry_is_rate_limited_and_does_not_retain_or_alias_owners(monkeypatch):
    calls=[]
    monkeypatch.setattr(r,'read_host_memory',lambda:(calls.append('host') or {'limit_bytes':100}))
    monkeypatch.setattr(r,'process_memory_snapshot',lambda **_:{'rss_bytes':5})
    clock=[1.]
    monkeypatch.setattr(r.time,'monotonic',lambda:clock[0])
    monitor=r.ResourceTelemetry(5)
    first=monitor.sample(queue_summary={'retained_bytes':3})
    first['host']['limit_bytes']=0
    clock[0]=2.
    second=monitor.sample()
    assert second['reused_sample'] and second['host']['limit_bytes']==100
    assert calls==['host'] and second['sample_age_seconds']==1
    clock[0]=7.
    assert not monitor.sample()['reused_sample'] and calls==['host','host']
    assert not monitor.sample(include_pss=False)['reused_sample']


def test_automatic_reserve_uses_actual_capacity_even_if_explicit_budget_is_wider(monkeypatch,tmp_path):
    setup_host(monkeypatch,[constraint(20*G),constraint(128*G,scope='system_available_memory')])
    gate=r.HostMemoryGate(tmp_path,host_limit_gib=1000)
    assert gate.limit_bytes==1000*G and gate.reserve_bytes==2*G
    with gate.lease(1*G) as lease:assert lease.last_admission['capacity_bytes']==18*G


def test_resume_may_change_resource_policy_only_after_all_live_owners_release(monkeypatch,tmp_path):
    setup_host(monkeypatch,[constraint(80*G)])
    first=r.HostMemoryGate(tmp_path,host_limit_gib=12)
    second=r.HostMemoryGate(tmp_path,host_limit_gib=8)
    with first.lease(1*G):
        with pytest.raises(ValueError,match='same limit/reserve'):
            with second.lease():pass
    with second.lease(1*G):
        assert json.loads(second.path.read_text())['policy']['limit_bytes']==8*G
