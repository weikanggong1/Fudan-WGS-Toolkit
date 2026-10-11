"""Resource dispatch for stable global traits and exact ordered cohorts.

Dispatch changes neither fitted models nor mathematical/checkpoint identity.
Large fitted banks can stay within each GPU's cache by assigning whole ordered
cohorts to one worker, while each such worker scans the complete genetic plan.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import uuid


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def gpu_model_cache_capacity(gpu_ids,*,memory_limit_gib=None,reserve_bytes=2**30,
                             cache_fraction=.5):
    """Resolve a common cache capacity before workers allocate fitted states.

    CUDA limits use physical capacity, caller cap and current driver/allocator
    headroom. A CPU-only contract test may supply an explicit GPU budget; no
    arbitrary capacity is invented when neither CUDA nor that budget exists.
    """
    if type(reserve_bytes) is not int or reserve_bytes<0:
        raise ValueError('GPU reserve must be a nonnegative byte count')
    if (isinstance(cache_fraction,bool) or not isinstance(cache_fraction,(int,float))
            or not math.isfinite(cache_fraction) or not 0<cache_fraction<=1):
        raise ValueError('model cache fraction must be in (0, 1]')
    if memory_limit_gib is not None and (isinstance(memory_limit_gib,bool)
            or not isinstance(memory_limit_gib,(int,float)) or not math.isfinite(memory_limit_gib)
            or memory_limit_gib<=0):
        raise ValueError('GPU memory limit must be finite and positive or None')
    explicit = None if memory_limit_gib is None else int(memory_limit_gib*2**30)
    import torch
    if not torch.cuda.is_available():
        return None if explicit is None else int(max(0,explicit-reserve_bytes)*cache_fraction)
    available = []
    for index in gpu_ids:
        device = torch.device(f'cuda:{index}')
        free,total = torch.cuda.mem_get_info(device)
        allocated,reserved = torch.cuda.memory_allocated(device),torch.cuda.memory_reserved(device)
        cap = total if explicit is None else min(total,explicit)
        available.append(max(0,min(cap-allocated,free+max(0,reserved-allocated))-reserve_bytes))
    return int(min(available)*cache_fraction) if available else None


def genetic_schedule(configuration):
    """Global, chromosome-local immutable job keys in their scientific order."""
    keys = []
    for reference in configuration['chromosomes']:
        for index,job in enumerate(reference['jobs']):
            ordinal = job.get('checkpoint_job_index',index)
            if type(ordinal) is not int or ordinal<0:
                raise ValueError('checkpoint job index must be a nonnegative global integer')
            keys.append([str(reference['name']),ordinal,job['kind']])
    if len({tuple(key) for key in keys}) != len(keys):
        raise ValueError('global genetic schedule contains duplicate job identities')
    return keys


def _ordered_cohorts(entries):
    groups = {}
    for entry in entries:
        path = Path(entry['path'])/'model.private.json'
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=entry['sha256']:
            raise ValueError('fitted model metadata changed before trait dispatch')
        model = json.loads(data)
        axis_hash = model.get('sample_rows_sha256')
        n = model.get('n')
        if (not isinstance(axis_hash,str) or len(axis_hash)!=64
                or any(c not in '0123456789abcdef' for c in axis_hash)
                or type(n) is not int or n<1 or model.get('family')!=entry['family']):
            raise ValueError('ordered-cohort dispatch requires bound sample-axis metadata')
        key = (axis_hash,n)
        group = groups.setdefault(key,dict(ordered_sample_rows_sha256=axis_hash,n=n,trait_indices=[]))
        group['trait_indices'].append(entry['trait_index'])
    return sorted(groups.values(),key=lambda group:group['trait_indices'][0])


def build_trait_dispatch(entries,worker_count,configuration,*,source_binding,
                         strategy='auto',model_cache_capacity_bytes=None,
                         binary_normal_cache='derived_fp64',trait_batch_size=32):
    """Weight whole exact ordered cohorts by authenticated fitted-state bytes.

    Only tiny model/state metadata is read; tensor arrays are not reloaded or
    fitted. Exact sample-row *file* hashes retain both ordering and NPY geometry.
    ModelRepository separately authenticates those axes before association.
    """
    if type(worker_count) is not int or worker_count<1:
        raise ValueError('dispatch worker count must be a positive integer')
    if strategy not in ('auto','genetic_jobs','ordered_cohorts','cohort_tiles'):
        raise ValueError('unknown association partition strategy')
    if type(trait_batch_size) is not int or trait_batch_size<1:
        raise ValueError('trait batch size must be a positive integer')
    if model_cache_capacity_bytes is not None and (type(model_cache_capacity_bytes) is not int
            or model_cache_capacity_bytes<0):
        raise ValueError('model cache capacity must be a nonnegative byte count or None')
    indices = [entry.get('trait_index') for entry in entries]
    if (not indices or any(type(index) is not int or index<0 for index in indices)
            or len(set(indices))!=len(indices) or indices!=sorted(indices)):
        raise ValueError('trait dispatch requires ordered distinct global trait indices')
    jobs = genetic_schedule(configuration)
    selected,footprint = strategy,None
    if strategy!='genetic_jobs' and worker_count>1:
        from .phewas_model_footprint import estimate_fitted_model_storage
        footprint = estimate_fitted_model_storage(entries,binary_normal_cache=binary_normal_cache)
    if strategy=='auto':
        total = (None if footprint is None else footprint['persistent_with_lazy_upper_bound_bytes']
            +footprint['active_spa_projection_bytes_upper_bound']
            +min(trait_batch_size,len(entries))*footprint['normal_derived_active_bytes_upper_bound'])
        selected = ('ordered_cohorts' if worker_count>1 and model_cache_capacity_bytes is not None
                    and total>model_cache_capacity_bytes else 'genetic_jobs')
    workers = [dict(worker_id=i,trait_indices=[],trait_tiles=[],estimated_persistent_model_bytes=0)
               for i in range(worker_count)]
    cohorts,tiles = [],[]
    if selected in ('ordered_cohorts','cohort_tiles'):
        from .phewas_model_footprint import estimate_fitted_subset_storage
        if footprint is None:
            from .phewas_model_footprint import estimate_fitted_model_storage
            footprint = estimate_fitted_model_storage(entries,binary_normal_cache=binary_normal_cache)
        cohorts = _ordered_cohorts(entries)
        for cohort_index,group in enumerate(cohorts):
            subset = estimate_fitted_subset_storage(footprint,group['trait_indices'])
            group.update(cohort_index=cohort_index,
                estimated_persistent_model_bytes=subset['persistent_loaded_bytes'],
                estimated_model_bytes_with_lazy=subset['persistent_with_lazy_upper_bound_bytes'])
        # Stable cohort tiles prevent a large/shared input axis from leaving
        # most GPUs idle. Boundaries are scientific-layout dependent, whereas
        # only tile ownership changes with GPU count or memory capacity.
        if strategy=='auto' and selected=='ordered_cohorts' and (
                len(cohorts)<worker_count or any(
                    group['estimated_model_bytes_with_lazy']>model_cache_capacity_bytes
                    for group in cohorts)):
            selected='cohort_tiles'
        if selected=='cohort_tiles':
            for group in cohorts:
                for tile_index,begin in enumerate(range(0,len(group['trait_indices']),trait_batch_size)):
                    ids=group['trait_indices'][begin:begin+trait_batch_size]
                    subset=estimate_fitted_subset_storage(footprint,ids)
                    tile=dict(cohort_index=group['cohort_index'],tile_index=tile_index,
                        ordered_sample_rows_sha256=group['ordered_sample_rows_sha256'],n=group['n'],
                        trait_indices=ids,trait_batch_size=trait_batch_size)
                    tile['tile_binding_sha256']=_hash(tile)
                    tile.update(estimated_persistent_model_bytes=subset['persistent_loaded_bytes'],
                        estimated_model_bytes_with_lazy=subset['persistent_with_lazy_upper_bound_bytes'])
                    tiles.append(tile)
            units=tiles
        else:
            units=cohorts
        for unit in sorted(units,key=lambda group:(-group['estimated_model_bytes_with_lazy'],
                                                   group['trait_indices'][0])):
            target = min(workers,key=lambda worker:(worker['estimated_persistent_model_bytes'],
                                                    len(worker['trait_indices']),worker['worker_id']))
            target['trait_indices'].extend(unit['trait_indices'])
            target['estimated_persistent_model_bytes'] += unit['estimated_model_bytes_with_lazy']
            unit['worker_id'] = target['worker_id']
            if selected=='cohort_tiles':
                target['trait_tiles'].append(unit)
        for worker in workers:
            worker['trait_indices'].sort()
            worker['trait_tiles'].sort(key=lambda tile:(tile['cohort_index'],tile['tile_index']))
            subset = estimate_fitted_subset_storage(footprint,worker['trait_indices'])
            worker['estimated_persistent_model_bytes'] = subset['persistent_loaded_bytes']
            worker['estimated_model_bytes_with_lazy'] = subset['persistent_with_lazy_upper_bound_bytes']
            worker['active_spa_projection_bytes_upper_bound'] = subset['active_spa_projection_bytes_upper_bound']
    else:
        for worker in workers:
            worker['trait_indices'] = list(indices)
    for worker in workers:
        assigned_jobs = ((jobs if worker['trait_indices'] else [])
            if selected in ('ordered_cohorts','cohort_tiles') else jobs[worker['worker_id']::worker_count])
        worker.update(genetic_job_visits=len(assigned_jobs),genetic_schedule_sha256=_hash(assigned_jobs))
    record = dict(schema_version=1,strategy=selected,requested_strategy=strategy,
        worker_count=worker_count,global_trait_indices=indices,source_binding=source_binding,
        global_model_metadata=[dict(trait_index=e['trait_index'],sha256=e['sha256']) for e in entries],
        model_cache_capacity_bytes=model_cache_capacity_bytes,
        estimated_total_persistent_model_bytes=None if footprint is None else footprint['persistent_loaded_bytes'],
        estimated_total_model_bytes_with_lazy=(None if footprint is None else footprint['persistent_with_lazy_upper_bound_bytes']),
        active_spa_projection_bytes_upper_bound=(None if footprint is None else footprint['active_spa_projection_bytes_upper_bound']),
        active_derived_normal_bytes_upper_bound=(None if footprint is None else
            min(trait_batch_size,len(entries))*footprint['normal_derived_active_bytes_upper_bound']),
        global_unique_genetic_job_groups=len(jobs),global_genetic_schedule_sha256=_hash(jobs),
        worker_genetic_job_visits=sum(worker['genetic_job_visits'] for worker in workers),
        workers=workers,ordered_cohorts=cohorts,cohort_tiles=tiles,trait_batch_size=trait_batch_size,
        model_estimate_scope='metadata geometry; resident tensor estimate, not measured GPU memory')
    record['dispatch_binding_sha256'] = _hash(record)
    return record


def _validate_dispatch(dispatch):
    if (not isinstance(dispatch,dict) or dispatch.get('schema_version')!=1
            or dispatch.get('dispatch_binding_sha256')!=_hash({key:value for key,value in dispatch.items()
                                                              if key!='dispatch_binding_sha256'})):
        raise ValueError('trait dispatch binding changed')
    count,indices,workers = dispatch['worker_count'],dispatch['global_trait_indices'],dispatch['workers']
    if (type(count) is not int or count<1 or len(workers)!=count
            or any(type(index) is not int or index<0 for index in indices)
            or len(set(indices))!=len(indices)
            or [worker['worker_id'] for worker in workers]!=list(range(count))):
        raise ValueError('trait dispatch worker/global coverage is invalid')
    visits = [worker['genetic_job_visits'] for worker in workers]
    if (any(type(value) is not int or value<0 for value in visits)
            or sum(visits)!=dispatch['worker_genetic_job_visits']):
        raise ValueError('trait dispatch genetic visits are invalid')
    if dispatch['strategy'] in ('ordered_cohorts','cohort_tiles'):
        assigned = [index for worker in workers for index in worker['trait_indices']]
        cohorts = [index for group in dispatch['ordered_cohorts'] for index in group['trait_indices']]
        if sorted(assigned)!=sorted(indices) or sorted(cohorts)!=sorted(indices):
            raise ValueError('ordered-cohort dispatch must assign every trait exactly once')
        if dispatch['strategy']=='ordered_cohorts':
            for group in dispatch['ordered_cohorts']:
                worker = workers[group['worker_id']]
                if not set(group['trait_indices']).issubset(worker['trait_indices']):
                    raise ValueError('ordered sample cohort was split between workers')
        else:
            groups={group['cohort_index']:group for group in dispatch['ordered_cohorts']}
            tiles=dispatch['cohort_tiles']
            if sorted(index for tile in tiles for index in tile['trait_indices'])!=sorted(indices):
                raise ValueError('cohort tiles must cover every global trait exactly once')
            expected_tiles=[]
            for group in dispatch['ordered_cohorts']:
                for tile_index,begin in enumerate(range(0,len(group['trait_indices']),dispatch['trait_batch_size'])):
                    expected_tiles.append((group['cohort_index'],tile_index,
                        group['trait_indices'][begin:begin+dispatch['trait_batch_size']]))
            if [(tile['cohort_index'],tile['tile_index'],tile['trait_indices']) for tile in tiles]!=expected_tiles:
                raise ValueError('cohort tile boundaries changed')
            for tile in tiles:
                group=groups[tile['cohort_index']]
                scientific={key:tile[key] for key in ('cohort_index','tile_index',
                    'ordered_sample_rows_sha256','n','trait_indices','trait_batch_size')}
                if (tile['tile_binding_sha256']!=_hash(scientific)
                        or tile['ordered_sample_rows_sha256']!=group['ordered_sample_rows_sha256']
                        or tile['n']!=group['n'] or tile['trait_batch_size']!=dispatch['trait_batch_size']
                        or not set(tile['trait_indices']).issubset(workers[tile['worker_id']]['trait_indices'])):
                    raise ValueError('cohort tile axis or worker binding changed')
            for worker in workers:
                if worker['trait_tiles']!=[tile for tile in tiles if tile['worker_id']==worker['worker_id']]:
                    raise ValueError('worker cohort tile binding differs')
        for worker in workers:
            expected = dispatch['global_unique_genetic_job_groups'] if worker['trait_indices'] else 0
            if worker['genetic_job_visits']!=expected:
                raise ValueError('ordered-cohort worker omitted genetic jobs')
    elif dispatch['strategy']=='genetic_jobs':
        if (any(worker['trait_indices']!=indices for worker in workers)
                or sum(visits)!=dispatch['global_unique_genetic_job_groups']):
            raise ValueError('genetic dispatch must cover the global schedule once')
    else:
        raise ValueError('unsupported trait dispatch strategy')


def dispatch_configuration(configuration,dispatch,worker_id):
    """Retain global checkpoint IDs across cards, cohorts and resume attempts."""
    _validate_dispatch(dispatch)
    if type(worker_id) is not int or not 0<=worker_id<dispatch['worker_count']:
        raise ValueError('worker ID differs from trait dispatch')
    result,chromosomes,ordinal = dict(configuration),[],0
    for original in configuration['chromosomes']:
        jobs = []
        for index,original_job in enumerate(original['jobs']):
            job = dict(original_job)
            job.setdefault('checkpoint_job_index',index)
            if ((dispatch['strategy'] in ('ordered_cohorts','cohort_tiles') and dispatch['workers'][worker_id]['trait_indices'])
                    or (dispatch['strategy']=='genetic_jobs' and ordinal%dispatch['worker_count']==worker_id)):
                jobs.append(job)
            ordinal += 1
        if jobs:
            chromosomes.append(dict(original,jobs=jobs))
    result['chromosomes'] = chromosomes
    if dispatch['strategy']=='cohort_tiles':
        result['_association_trait_tiles'] = dispatch['workers'][worker_id]['trait_tiles']
    return result


def publish_trait_dispatch(directory,dispatch,attempt_id):
    """Publish an attempt-specific binding once; all worker peers must agree."""
    _validate_dispatch(dispatch)
    directory = Path(directory)
    path = directory/f'association_dispatch.{attempt_id}.private.json'
    data = (_canonical(dispatch)+'\n').encode()
    candidate = path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    candidate.write_bytes(data)
    try:
        try:
            os.link(candidate,path)
        except FileExistsError:
            if path.read_bytes()!=data:
                raise ValueError('workers disagree on the bound trait/resource dispatch')
    finally:
        candidate.unlink(missing_ok=True)
    return dispatch['dispatch_binding_sha256']


def verify_dispatch_reports(dispatch,reports):
    """Separate unique scientific jobs from replicated worker scan visits."""
    _validate_dispatch(dispatch)
    if len(reports)!=dispatch['worker_count']:
        raise RuntimeError('association worker coverage differs from dispatch')
    for expected,report in zip(dispatch['workers'],reports):
        if (report.get('worker_id')!=expected['worker_id']
                or report.get('association_dispatch_sha256')!=dispatch['dispatch_binding_sha256']
                or report.get('assigned_trait_indices')!=expected['trait_indices']
                or report.get('assigned_genetic_schedule_sha256')!=expected['genetic_schedule_sha256']
                or report.get('job_groups')!=expected['genetic_job_visits']):
            raise RuntimeError('association trait/job completion differs from bound dispatch')
        if dispatch['strategy']=='cohort_tiles' and report.get('assigned_trait_tiles')!=expected['trait_tiles']:
            raise RuntimeError('association fixed cohort tile completion differs from dispatch')
        if 'trait_rows' in report and report['job_groups']:
            if {int(index) for index in report['trait_rows']}!=set(expected['trait_indices']):
                raise RuntimeError('association report omitted or added a dispatched trait')
    return dict(association_partition_strategy=dispatch['strategy'],
        association_dispatch_sha256=dispatch['dispatch_binding_sha256'],
        completed_genetic_job_groups=dispatch['global_unique_genetic_job_groups'],
        completed_worker_genetic_job_visits=sum(report['job_groups'] for report in reports),
        global_trait_assignment_verified=True,genetic_schedule_coverage_verified=True)
