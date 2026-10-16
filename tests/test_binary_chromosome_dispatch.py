"""Actual ownership/checkpoint/descriptor regressions, not scientific timing."""
import copy
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

import pytest

from fudan_wgs_toolkit import phewas_partition as partition
from fudan_wgs_toolkit.chromosome_csv import ChromosomeCheckpointCSVWriter
from fudan_wgs_toolkit.worker_file_resources import configure_worker_file_capacity

BINDING = dict(source_implementation_sha256='source',input_manifest_sha256='inputs',
    configuration_sha256='fixed-scientific-schedule-and-trait-tiles',models={'0':'model0','1':'model1'})


def configuration():
    return dict(chromosomes=[dict(name=name,jobs=[dict(kind=('individual','coding','noncoding')[i%3],
        checkpoint_job_index=100+i) for i in range(count)])
        for name,count in [('1',9),('2',8),('21',3),('22',2)]])


def entries():
    # Whole-chromosome scheduling intentionally does not scan large model
    # banks. Their immutable metadata is checked by the model barrier.
    return [dict(trait_index=index,path='unused',sha256=str(index)*64,family='binomial')
            for index in range(3)]


def reports(dispatch):
    return [dict(worker_id=worker['worker_id'], assigned_trait_indices=worker['trait_indices'],
        assigned_chromosomes=worker['assigned_chromosomes'],
        association_dispatch_sha256=dispatch['dispatch_binding_sha256'],
        assigned_genetic_schedule_sha256=worker['genetic_schedule_sha256'],
        completed_genetic_schedule_sha256=worker['genetic_schedule_sha256'],
        genetic_schedule_coverage_verified=True,observed_finished_genetic_jobs=worker['genetic_job_visits'],
        job_groups=worker['genetic_job_visits'],trait_rows={i:{} for i in worker['trait_indices']})
        for worker in dispatch['workers']]


@pytest.mark.parametrize('workers',[1,2,4,8])
def test_lpt_whole_chromosomes_all_traits_exactly_once_and_legal_idle(workers):
    original = configuration()
    dispatch = partition.build_trait_dispatch(entries(),workers,original,
        source_binding={'source':'immutable'},strategy='whole_chromosomes',model_cache_capacity_bytes=1)
    assert dispatch['strategy']=='whole_chromosomes'
    assert dispatch['worker_genetic_job_visits']==22
    assert dispatch['global_unique_genetic_job_groups']==22
    assigned, schedules = [],[]
    for worker in dispatch['workers']:
        subset = partition.dispatch_configuration(original,dispatch,worker['worker_id'])
        assigned.extend(chromosome['name'] for chromosome in subset['chromosomes'])
        schedules.extend(partition.genetic_schedule(subset))
        assert worker['trait_indices']==([0,1,2] if subset['chromosomes'] else [])
        for chromosome in subset['chromosomes']:
            assert chromosome == next(item for item in original['chromosomes'] if item['name']==chromosome['name'])
    assert sorted(assigned)==['1','2','21','22']
    assert sorted(schedules)==sorted(partition.genetic_schedule(original))
    completion = partition.verify_dispatch_reports(dispatch,reports(dispatch))
    assert completion['completed_worker_genetic_job_visits']==22
    assert completion['genetic_schedule_coverage_verified'] is True


def test_lpt_balance_is_deterministic_and_keeps_chromosome_order():
    original = configuration()
    first=partition.build_trait_dispatch(entries(),2,original,source_binding={},strategy='whole_chromosomes')
    second=partition.build_trait_dispatch(entries(),2,original,source_binding={},strategy='whole_chromosomes')
    assert first==second
    assert [worker['genetic_job_visits'] for worker in first['workers']]==[11,11]
    assert [worker['assigned_chromosomes'] for worker in first['workers']]==[['1','22'],['2','21']]


def test_resumed_gpu_count_changes_only_ownership_and_never_job_identity():
    original=configuration()
    schedules=[]
    for count in (1,2,8):
        dispatch=partition.build_trait_dispatch(entries(),count,original,source_binding={},strategy='whole_chromosomes')
        schedules.append(sorted(job for worker in dispatch['workers'] for job in
            partition.genetic_schedule(partition.dispatch_configuration(original,dispatch,worker['worker_id']))))
    assert schedules[0]==schedules[1]==schedules[2]


def test_observed_schedule_matches_global_hash_and_rejects_missing_or_duplicate_job():
    config=configuration()
    audit=partition.GeneticScheduleAudit(config)
    for ci,reference in enumerate(config['chromosomes']):
        for ji,job in enumerate(reference['jobs']):
            audit.observe(dict(event='finished',chromosome_index=ci,job_index=ji,kind=job['kind']))
    report=audit.finish()
    assert report['completed_genetic_schedule_sha256']==partition._hash(partition.genetic_schedule(config))
    assert report['observed_finished_genetic_jobs']==22
    incomplete=partition.GeneticScheduleAudit(config)
    with pytest.raises(RuntimeError,match='omitted'):
        incomplete.finish()
    incomplete.observe(dict(event='finished',chromosome_index=0,job_index=0,kind='individual'))
    with pytest.raises(RuntimeError,match='ordered schedule'):
        incomplete.observe(dict(event='finished',chromosome_index=0,job_index=0,kind='individual'))


@pytest.mark.parametrize('damage',['owner','schedule','configuration','report'])
def test_chr_ownership_and_missing_completion_fail_closed(damage):
    original=configuration()
    dispatch=partition.build_trait_dispatch(entries(),2,original,source_binding={},strategy='whole_chromosomes')
    if damage=='report':
        values=reports(dispatch); values[0]['assigned_chromosomes']=['2']
        with pytest.raises(RuntimeError,match='chromosome'):
            partition.verify_dispatch_reports(dispatch,values)
        return
    if damage=='configuration':
        original['chromosomes'][0]['jobs'][0]['checkpoint_job_index']=900
    elif damage=='owner':
        dispatch['chromosome_units'][0]['worker_id']=1
    else:
        dispatch['chromosome_units'][0]['genetic_schedule'][0][1]=900
    dispatch['dispatch_binding_sha256']=partition._hash({key:value for key,value in dispatch.items()
        if key!='dispatch_binding_sha256'})
    with pytest.raises(ValueError):
        partition.dispatch_configuration(original,dispatch,0)


def rows(directory,report):
    values=[]
    for receipt in report['receipts']:
        path=Path(directory)/receipt['csv_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256']
        with gzip.open(path,'rt',newline='') as handle:
            values.extend(csv.DictReader(handle))
    return values


def row(chromosome='21',p=.25):
    return dict(Chr=chromosome,P=p)


def test_concurrent_chromosomes_do_not_compete_for_same_trait_locks(tmp_path):
    first=ChromosomeCheckpointCSVWriter(tmp_path,binding=BINDING,chromosomes=['21'],worker_id='0')
    second=ChromosomeCheckpointCSVWriter(tmp_path,binding=BINDING,chromosomes=['22'],worker_id='1')
    first.write_tile('coding','21',[(0,[row()]),(1,[row()])],job_index=100)
    second.write_tile('coding','22',[(0,[row('22')]),(1,[row('22')])],job_index=100)
    a,b=first.finish(),second.finish()
    assert len(rows(tmp_path,a))==len(rows(tmp_path,b))==2
    assert {item['csv_file'].split('/')[1] for item in a['receipts']}=={'chr21'}
    assert {item['csv_file'].split('/')[1] for item in b['receipts']}=={'chr22'}


def test_transition_releases_previous_chromosome_locks_and_bounds_writer_count(tmp_path):
    writer=ChromosomeCheckpointCSVWriter(tmp_path,binding=BINDING,chromosomes=['21','22'])
    writer.write_tile('coding','21',[(0,[row()]),(1,[])],job_index=1)
    previous=writer.current
    writer.write_tile('noncoding','22',[(0,[row('22')]),(1,[])],job_index=2)
    assert previous._closed and not previous._owners and not previous.streams
    report=writer.finish()
    assert report['maximum_simultaneously_active_chromosome_writers']==1
    assert report['rows']['coding']==report['rows']['noncoding']==1


def test_resume_reassignment_preserves_csv_bytes_and_recovers_tail_without_duplicate(tmp_path):
    writer=ChromosomeCheckpointCSVWriter(tmp_path,binding=BINDING,chromosomes=['21'],worker_id='0')
    first=writer.write_tile('coding','21',[(0,[row()]),(1,[row(p=.5)])],job_index=100)
    directory=writer.current.directory
    paths=list(directory.rglob('*.partial'))
    writer.abort()
    for path in paths:
        with path.open('ab') as handle:
            handle.write(b'uncommitted-tail')
    resumed=ChromosomeCheckpointCSVWriter(tmp_path,binding=BINDING,chromosomes=['21'],resume=True,worker_id='7')
    assert resumed.checkpoint_tile('coding','21',100,[0,1])==first
    resumed.write_tile('coding','21',[(0,[row(p=.75)]),(1,[])],job_index=101)
    report=resumed.finish()
    assert report['reused_blocks']==1 and report['computed_blocks']==1
    assert [item['P'] for item in rows(tmp_path,report)]==['0.25','0.75','0.5']
    hashes={receipt['csv_file']:receipt['sha256'] for receipt in report['receipts']}
    replay=ChromosomeCheckpointCSVWriter(tmp_path,binding=BINDING,chromosomes=['21'],resume=True,worker_id='2')
    assert replay.checkpoint_tile('coding','21',100,[0,1]) is not None
    assert replay.checkpoint_tile('coding','21',101,[0,1]) is not None
    replayed=replay.finish()
    assert replayed['computed_blocks']==0 and replayed['reused_blocks']==2
    assert hashes=={receipt['csv_file']:receipt['sha256'] for receipt in replayed['receipts']}


def test_same_chromosome_concurrent_owner_is_rejected(tmp_path):
    first=ChromosomeCheckpointCSVWriter(tmp_path,binding=BINDING,chromosomes=['21'],worker_id='0')
    second=ChromosomeCheckpointCSVWriter(tmp_path,binding=BINDING,chromosomes=['21'],worker_id='1')
    try:
        assert first.checkpoint_tile('coding','21',100,[0]) is None
        with pytest.raises(RuntimeError,match='active writer'):
            second.checkpoint_tile('coding','21',100,[0])
    finally:
        first.abort(); second.abort()


def test_fd_admission_fails_before_outputs_when_hard_limit_cannot_fit(monkeypatch):
    from fudan_wgs_toolkit import worker_file_resources as files
    monkeypatch.setattr(files.resource,'getrlimit',lambda which:(1024,2048))
    with pytest.raises(RuntimeError,match='hard limit'):
        configure_worker_file_capacity(799)


def test_real_799_locks_exceed_old_soft_limit_and_commit_valid_csv(tmp_path):
    # A subprocess changes its own resource limit only. This exercises the
    # real writer and 799 simultaneous OS locks, with 256 unrelated live FDs
    # standing for loader/CUDA/model handles. It is not a speed benchmark.
    source=r'''
import json, os, resource, sys
from fudan_wgs_toolkit.chromosome_csv import ChromosomeCheckpointCSVWriter
from fudan_wgs_toolkit.worker_file_resources import configure_worker_file_capacity
soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE,(min(1024,hard),hard))
admission=configure_worker_file_capacity(799)
handles=[os.open('/dev/null',os.O_RDONLY) for _ in range(256)]
binding=dict(source_implementation_sha256='source',input_manifest_sha256='inputs',configuration_sha256='ordered',models={'0':'bound'})
writer=ChromosomeCheckpointCSVWriter(sys.argv[1],binding=binding,chromosomes=['21'],file_descriptor_admission=admission)
writer.write_tile('coding','21',[(i,[dict(Chr='21',P=.25)]) for i in range(799)],job_index=0)
result=writer.finish()
for handle in handles: os.close(handle)
print(json.dumps(dict(rows=result['rows'],files=result['files'],file_descriptors=result['file_descriptors'],verified=result['committed_range_checksums_verified'])))
'''
    result=subprocess.run([sys.executable,'-c',source,str(tmp_path/'descriptor_fixture')],
        text=True,capture_output=True,check=True,timeout=120)
    evidence=json.loads(result.stdout.strip().splitlines()[-1])
    assert evidence['rows']['coding']==evidence['files']==799
    assert evidence['verified'] is True
    assert evidence['file_descriptors']['soft_limit_before']==1024
    admission=evidence['file_descriptors']
    assert admission['minimum_required_soft_limit']==2*799+2048
    assert admission['preferred_soft_limit']==8192
    assert admission['soft_limit']>=admission['minimum_required_soft_limit']
    assert admission['hard_limit_constrained']==(admission['hard_limit'] is not None and admission['hard_limit']<8192)
    if not admission['hard_limit_constrained']:
        assert admission['soft_limit']>=8192
    else:
        assert admission['soft_limit']==admission['hard_limit']
    assert evidence['file_descriptors']['maximum_observed_open_files']>1024
    assert evidence['file_descriptors']['observed_minimum_headroom']>0


def test_small_trait_writer_accepts_4096_hard_limit_without_changing_parent(tmp_path):
    before=resource.getrlimit(resource.RLIMIT_NOFILE)
    if before[1]!=resource.RLIM_INFINITY and before[1]<4096:
        pytest.skip('isolated test needs permission to lower its own hard limit to 4096')
    source=r'''
import json, os, resource, sys
from fudan_wgs_toolkit.chromosome_csv import ChromosomeCheckpointCSVWriter
from fudan_wgs_toolkit.worker_file_resources import configure_worker_file_capacity
resource.setrlimit(resource.RLIMIT_NOFILE,(1024,4096))
admission=configure_worker_file_capacity(3)
handles=[os.open('/dev/null',os.O_RDONLY) for _ in range(256)]
binding=dict(source_implementation_sha256='source',input_manifest_sha256='inputs',configuration_sha256='ordered',models={'0':'bound'})
writer=ChromosomeCheckpointCSVWriter(sys.argv[1],binding=binding,chromosomes=['21'],file_descriptor_admission=admission)
writer.write_tile('coding','21',[(i,[dict(Chr='21',P=.25)]) for i in range(3)],job_index=0)
result=writer.finish()
for handle in handles: os.close(handle)
print(json.dumps(dict(rows=result['rows'],files=result['files'],file_descriptors=result['file_descriptors'],verified=result['committed_range_checksums_verified'],limits=list(resource.getrlimit(resource.RLIMIT_NOFILE)))))
'''
    result=subprocess.run([sys.executable,'-c',source,str(tmp_path/'small_descriptor_fixture')],
        text=True,capture_output=True,check=True,timeout=120)
    evidence=json.loads(result.stdout.strip().splitlines()[-1])
    assert resource.getrlimit(resource.RLIMIT_NOFILE)==before
    assert evidence['rows']['coding']==evidence['files']==3
    assert evidence['verified'] is True
    assert evidence['limits']==[4096,4096]
    admission=evidence['file_descriptors']
    assert admission['minimum_required_soft_limit']==2054
    assert admission['preferred_soft_limit']==8192
    assert admission['required_soft_limit']==4096
    assert admission['hard_limit_constrained'] is True
    assert admission['raised'] is True
    assert 256<admission['maximum_observed_open_files']<4096
    assert admission['observed_minimum_headroom']>0


@pytest.mark.parametrize('trait_count', [0,3,16])
def test_restrictive_hard_limit_still_rejects_actual_trait_minimum(monkeypatch,trait_count):
    from fudan_wgs_toolkit import worker_file_resources as files
    minimum=2*trait_count+2048
    monkeypatch.setattr(files.resource,'getrlimit',lambda which:(1024,minimum-1))
    changed=[]
    monkeypatch.setattr(files.resource,'setrlimit',lambda *args:changed.append(args))
    with pytest.raises(RuntimeError,match=f'at least {minimum}'):
        configure_worker_file_capacity(trait_count)
    assert changed==[]
