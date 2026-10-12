"""Metadata-only dispatch and global scientific coverage contracts."""
import copy
import hashlib
import json

import pytest

from fudan_wgs_toolkit import phewas_partition as partition


def model(tmp_path,index,*,n=20,q=2,axis=None,x_hash=None,family='gaussian'):
    root=tmp_path/f'trait_{index}'
    (root/'normal').mkdir(parents=True)
    def spec(name,shape,dtype='<f4'):
        return dict(file=name+'.npy',shape=shape,dtype=dtype,
            sha256=x_hash if name=='x' and x_hash else hashlib.sha256(f'{index}:{name}'.encode()).hexdigest())
    normal=dict(family=family,arrays=dict(x=spec('x',[n,q]),
        scaled_residuals=spec('scaled_residuals',[n]),inverse_variance=spec('inverse_variance',[n])))
    normal_path=root/'normal/state.private.json'
    normal_path.write_text(json.dumps(normal))
    meta=dict(family=family,n=n,sample_rows_sha256=axis or hashlib.sha256(f'axis:{index}'.encode()).hexdigest(),
        state_sha256=hashlib.sha256(normal_path.read_bytes()).hexdigest(),spa_state_sha256=None)
    if family=='binomial':
        (root/'spa').mkdir()
        spa=dict(family=family,arrays=dict(x=spec('x',[n,q],'<f8'),
            scaled_residuals=spec('scaled_residuals',[n],'<f8')))
        path=root/'spa/state.private.json'
        path.write_text(json.dumps(spa))
        meta['spa_state_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    path=root/'model.private.json'
    path.write_text(json.dumps(meta))
    return dict(trait_index=index,path=str(root),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),family=family)


@pytest.fixture
def schedule():
    return dict(chromosomes=[dict(name='21',jobs=[dict(kind='individual',checkpoint_job_index=0),
        dict(kind='coding',checkpoint_job_index=1),dict(kind='noncoding',checkpoint_job_index=2),
        dict(kind='ncrna',checkpoint_job_index=3)])])


def reports(dispatch):
    return [dict(worker_id=worker['worker_id'],assigned_trait_indices=worker['trait_indices'],
        association_dispatch_sha256=dispatch['dispatch_binding_sha256'],
        assigned_genetic_schedule_sha256=worker['genetic_schedule_sha256'],assigned_trait_tiles=worker['trait_tiles'],
        job_groups=worker['genetic_job_visits'],trait_rows={index:{} for index in worker['trait_indices']})
        for worker in dispatch['workers']]


def test_whole_exact_ordered_cohorts_stay_together_and_global_ids_survive(tmp_path,schedule):
    same='1'*64
    entries=[model(tmp_path,0,axis=same),model(tmp_path,1,axis=same),
             model(tmp_path,2,n=100),model(tmp_path,3,n=40)]
    dispatch=partition.build_trait_dispatch(entries,3,schedule,source_binding={'source':'bound'},
        strategy='ordered_cohorts',binary_normal_cache='stored_native')
    assigned=[worker['trait_indices'] for worker in dispatch['workers']]
    assert sorted(index for indices in assigned for index in indices)==[0,1,2,3]
    assert any(indices==[0,1] for indices in assigned)
    assert all(not ({0,1}.intersection(indices) and not {0,1}.issubset(indices)) for indices in assigned)
    assert dispatch['worker_genetic_job_visits']==12
    assert dispatch['global_unique_genetic_job_groups']==4
    assert dispatch==partition.build_trait_dispatch(entries,3,schedule,source_binding={'source':'bound'},
        strategy='ordered_cohorts',binary_normal_cache='stored_native')
    completion=partition.verify_dispatch_reports(dispatch,reports(dispatch))
    assert completion['completed_genetic_job_groups']==4
    assert completion['completed_worker_genetic_job_visits']==12


def test_equal_sample_counts_do_not_merge_different_ordered_axes(tmp_path,schedule):
    entries=[model(tmp_path,0,n=20,axis='1'*64),model(tmp_path,1,n=20,axis='2'*64)]
    dispatch=partition.build_trait_dispatch(entries,2,schedule,source_binding={},strategy='ordered_cohorts')
    assert len(dispatch['ordered_cohorts'])==2
    assert [worker['trait_indices'] for worker in dispatch['workers']]==[[0],[1]]


def test_auto_selects_model_fit_capacity_not_trait_count(tmp_path,schedule):
    entries=[model(tmp_path,0),model(tmp_path,1)]
    small=partition.build_trait_dispatch(entries,2,schedule,source_binding={},
        model_cache_capacity_bytes=10**9)
    large=partition.build_trait_dispatch(entries,2,schedule,source_binding={},
        model_cache_capacity_bytes=1)
    unknown=partition.build_trait_dispatch(entries,2,schedule,source_binding={},model_cache_capacity_bytes=None)
    assert small['strategy']==unknown['strategy']=='genetic_jobs'
    assert large['strategy']=='cohort_tiles'
    assert sum(worker['genetic_job_visits'] for worker in small['workers'])==4
    assert sum(worker['genetic_job_visits'] for worker in large['workers'])==8


def test_dispatch_changes_preserve_single_and_gene_global_checkpoint_ids(tmp_path,schedule):
    entries=[model(tmp_path,i) for i in range(3)]
    for count in (1,2,4):
        dispatch=partition.build_trait_dispatch(entries,count,schedule,source_binding={},strategy='ordered_cohorts')
        for i,worker in enumerate(dispatch['workers']):
            config=partition.dispatch_configuration(schedule,dispatch,i)
            if worker['trait_indices']:
                assert config==schedule
            else:
                assert config['chromosomes']==[]
        assert sum(worker['genetic_job_visits'] for worker in dispatch['workers'])==4*min(count,3)


def test_shared_design_deduplication_recomputed_inside_each_shard(tmp_path,schedule):
    entries=[model(tmp_path,0,axis='1'*64,x_hash='4'*64),
             model(tmp_path,1,axis='1'*64,x_hash='4'*64)]
    dispatch=partition.build_trait_dispatch(entries,2,schedule,source_binding={},strategy='ordered_cohorts')
    nonempty=[worker for worker in dispatch['workers'] if worker['trait_indices']]
    assert len(nonempty)==1
    assert nonempty[0]['estimated_persistent_model_bytes']==dispatch['estimated_total_persistent_model_bytes']
    assert partition.verify_dispatch_reports(dispatch,reports(dispatch))['completed_worker_genetic_job_visits']==4


@pytest.mark.parametrize('field',['assigned_trait_indices','association_dispatch_sha256',
    'assigned_genetic_schedule_sha256','job_groups','worker_id','trait_rows'])
def test_false_completion_or_cross_worker_trait_claim_is_rejected(tmp_path,schedule,field):
    entries=[model(tmp_path,i) for i in range(2)]
    dispatch=partition.build_trait_dispatch(entries,2,schedule,source_binding={},strategy='ordered_cohorts')
    values=reports(dispatch)
    if field=='job_groups':values[0][field]-=1
    elif field=='worker_id':values[0][field]=1
    elif field=='trait_rows':values[0][field]={999:{}}
    elif field=='assigned_trait_indices':values[0][field]=[999]
    else:values[0][field]='invalid'
    with pytest.raises(RuntimeError):partition.verify_dispatch_reports(dispatch,values)


def test_attempt_binding_is_immutable_but_resource_redispatch_has_new_receipt(tmp_path,schedule):
    entries=[model(tmp_path/'models',i) for i in range(2)]
    first=partition.build_trait_dispatch(entries,2,schedule,source_binding={'plan_sha256':'old'},strategy='ordered_cohorts')
    assert partition.publish_trait_dispatch(tmp_path,first,'first')==first['dispatch_binding_sha256']
    partition.publish_trait_dispatch(tmp_path,first,'first')
    changed=copy.deepcopy(first)
    changed['workers'][0]['trait_indices']=[999]
    with pytest.raises(ValueError,match='binding|disagree'):partition.publish_trait_dispatch(tmp_path,changed,'first')
    second=partition.build_trait_dispatch(entries,1,schedule,source_binding={'plan_sha256':'new'},strategy='ordered_cohorts')
    partition.publish_trait_dispatch(tmp_path,second,'second')
    assert first['dispatch_binding_sha256']!=second['dispatch_binding_sha256']
    assert (tmp_path/'association_dispatch.first.private.json').exists()


def test_metadata_change_or_duplicate_schedule_cannot_enter_dispatch(tmp_path,schedule):
    entries=[model(tmp_path,0)]
    (tmp_path/'trait_0/model.private.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum|changed'):
        partition.build_trait_dispatch(entries,2,schedule,source_binding={},strategy='ordered_cohorts')
    duplicate=copy.deepcopy(schedule)
    duplicate['chromosomes'][0]['jobs'].append(dict(kind='coding',checkpoint_job_index=1))
    with pytest.raises(ValueError,match='duplicate'):partition.genetic_schedule(duplicate)


def test_few_large_cohorts_use_all_workers_through_fixed_global_tiles(tmp_path,schedule):
    entries=[model(tmp_path,index,axis='1'*64) for index in range(20)]
    dispatch=partition.build_trait_dispatch(entries,4,schedule,source_binding={},
        strategy='auto',model_cache_capacity_bytes=1,trait_batch_size=2)
    assert dispatch['strategy']=='cohort_tiles' and len(dispatch['cohort_tiles'])==10
    assert all(worker['trait_indices'] for worker in dispatch['workers'])
    assert len(dispatch['ordered_cohorts'])==1
    assert [tile['trait_indices'] for tile in dispatch['cohort_tiles']]==[list(range(i,i+2)) for i in range(0,20,2)]
    assert sorted(index for worker in dispatch['workers'] for index in worker['trait_indices'])==list(range(20))
    for worker in dispatch['workers']:
        config=partition.dispatch_configuration(schedule,dispatch,worker['worker_id'])
        assert config['chromosomes']==schedule['chromosomes']
        assert config['_association_trait_tiles']==worker['trait_tiles']
    assert partition.verify_dispatch_reports(dispatch,reports(dispatch))['completed_worker_genetic_job_visits']==16


def test_tile_scientific_ids_and_boundaries_survive_gpu_count_and_ownership(tmp_path,schedule):
    # Traits in one exact cohort need not have consecutive global indices.
    entries=[model(tmp_path,index,axis='1'*64 if index%2 else '2'*64) for index in range(16)]
    layouts=[]
    for count in (1,3,8):
        dispatch=partition.build_trait_dispatch(entries,count,schedule,source_binding={},
            strategy='cohort_tiles',trait_batch_size=3)
        layouts.append([(tile['tile_binding_sha256'],tile['trait_indices']) for tile in dispatch['cohort_tiles']])
        assert partition.verify_dispatch_reports(dispatch,reports(dispatch))['completed_genetic_job_groups']==4
        for worker in dispatch['workers']:
            # Nonadjacent tiles assigned to one GPU remain separate globally.
            config=partition.dispatch_configuration(schedule,dispatch,worker['worker_id'])
            assert config['_association_trait_tiles']==worker['trait_tiles']
    assert layouts[0]==layouts[1]==layouts[2]


@pytest.mark.parametrize('damage',['boundary','axis','trait_assignment','report_tile'])
def test_corrupt_cohort_tile_cannot_claim_success_even_with_new_dispatch_hash(tmp_path,schedule,damage):
    entries=[model(tmp_path,index,axis='3'*64) for index in range(8)]
    dispatch=partition.build_trait_dispatch(entries,3,schedule,source_binding={},strategy='cohort_tiles',trait_batch_size=2)
    result=reports(dispatch)
    if damage=='report_tile':
        result[0]['assigned_trait_tiles']=[]
        with pytest.raises(RuntimeError,match='tile'):
            partition.verify_dispatch_reports(dispatch,result)
        return
    changed=copy.deepcopy(dispatch)
    tile=changed['cohort_tiles'][0]
    if damage=='boundary':tile['trait_indices']=tile['trait_indices'][::-1]
    elif damage=='axis':tile['ordered_sample_rows_sha256']='4'*64
    else:changed['workers'][tile['worker_id']]['trait_indices'].remove(tile['trait_indices'][0])
    changed['dispatch_binding_sha256']=partition._hash({key:value for key,value in changed.items() if key!='dispatch_binding_sha256'})
    with pytest.raises(ValueError,match='tile|cohort'):
        partition.dispatch_configuration(schedule,changed,0)


def test_auto_whole_cohorts_preserved_when_each_cohort_fits_cache(tmp_path,schedule):
    entries=[model(tmp_path,index,axis=str(index+1)*64,n=100) for index in range(8)]
    full=partition.build_trait_dispatch(entries,4,schedule,source_binding={},strategy='ordered_cohorts')
    max_cohort=max(group['estimated_model_bytes_with_lazy'] for group in full['ordered_cohorts'])
    dispatch=partition.build_trait_dispatch(entries,4,schedule,source_binding={},
        strategy='auto',model_cache_capacity_bytes=max_cohort+1)
    assert dispatch['strategy']=='ordered_cohorts' and dispatch['cohort_tiles']==[]


def chromosome_schedule():
    return dict(chromosomes=[dict(name=str(index+1), jobs=[dict(kind='coding' if job else 'individual')
        for job in range(count)]) for index,count in enumerate((8,2,5,1,4))])


@pytest.mark.parametrize('workers',[1,2,4,8])
def test_chromosome_workers_cover_all_traits_and_never_split_chromosomes(tmp_path,workers):
    entries=[model(tmp_path,index,axis='1'*64 if index%2 else '2'*64) for index in range(5)]
    configuration=chromosome_schedule()
    dispatch=partition.build_trait_dispatch(entries,workers,configuration,source_binding={},
        strategy='chromosomes',trait_batch_size=2,model_cache_capacity_bytes=1)
    assert dispatch['strategy']=='chromosomes'
    assert all(worker['trait_indices']==list(range(5)) for worker in dispatch['workers'])
    assert dispatch['worker_genetic_job_visits']==dispatch['global_unique_genetic_job_groups']==20
    found=[]; outputs=reports(dispatch)
    for worker,report in zip(dispatch['workers'],outputs):
        local=partition.dispatch_configuration(configuration,dispatch,worker['worker_id'])
        report['assigned_chromosomes']=[reference['name'] for reference in local['chromosomes']]
        assert report['assigned_chromosomes']==worker['chromosomes']
        found.extend(report['assigned_chromosomes'])
        for reference in local['chromosomes']:
            original=next(value for value in configuration['chromosomes'] if value['name']==reference['name'])
            assert len(reference['jobs'])==len(original['jobs'])
            assert [job['checkpoint_job_index'] for job in reference['jobs']]==list(range(len(original['jobs'])))
        assert '_association_trait_tiles' not in local
    assert sorted(found)==['1','2','3','4','5']
    assert partition.verify_dispatch_reports(dispatch,outputs)['completed_genetic_job_groups']==20
    assert dispatch==partition.build_trait_dispatch(entries,workers,configuration,source_binding={},
        strategy='chromosomes',trait_batch_size=2,model_cache_capacity_bytes=1)


def test_chromosome_assignment_balances_job_count_proxy_and_preserves_resume_ids(tmp_path):
    entries=[model(tmp_path,index) for index in range(2)]; configuration=chromosome_schedule()
    schedules=[]
    for count in (2,4):
        dispatch=partition.build_trait_dispatch(entries,count,configuration,source_binding={},strategy='chromosomes')
        schedules.append(sorted(tuple(job) for worker in dispatch['workers']
            for job in partition.genetic_schedule(partition.dispatch_configuration(configuration,dispatch,worker['worker_id']))))
        if count==2:
            assert [worker['genetic_job_visits'] for worker in dispatch['workers']]==[10,10]
    assert schedules[0]==schedules[1]==sorted(tuple(job) for job in partition.genetic_schedule(configuration))


@pytest.mark.parametrize('damage',['owner','chromosome_list','job_count','report','schedule'])
def test_chromosome_binding_rejects_wrong_owner_even_with_resigned_dispatch(tmp_path,damage):
    entries=[model(tmp_path,index) for index in range(2)]; configuration=chromosome_schedule()
    dispatch=partition.build_trait_dispatch(entries,2,configuration,source_binding={},strategy='chromosomes')
    if damage=='report':
        values=reports(dispatch)
        for worker,report in zip(dispatch['workers'],values):report['assigned_chromosomes']=worker['chromosomes']
        values[0]['assigned_chromosomes']=[]
        with pytest.raises(RuntimeError,match='chromosome'):partition.verify_dispatch_reports(dispatch,values)
        return
    if damage=='owner':dispatch['chromosome_assignments'][0]['worker_id']=1
    elif damage=='chromosome_list':dispatch['workers'][0]['chromosomes'].append('2')
    elif damage=='job_count':dispatch['chromosome_assignments'][0]['genetic_job_groups']-=1
    else:configuration['chromosomes'][0]['jobs'].pop()
    dispatch['dispatch_binding_sha256']=partition._hash({k:v for k,v in dispatch.items() if k!='dispatch_binding_sha256'})
    with pytest.raises(ValueError,match='chromosome'):
        partition.dispatch_configuration(configuration,dispatch,0)


def test_duplicate_chromosome_names_cannot_be_dispatched(tmp_path):
    configuration=dict(chromosomes=[dict(name='21',jobs=[dict(kind='coding')]),
        dict(name='21',jobs=[dict(kind='individual')])])
    entries=[model(tmp_path,0)]
    with pytest.raises(ValueError,match='distinct chromosome'):
        partition.build_trait_dispatch(entries,2,configuration,source_binding={},strategy='chromosomes')
