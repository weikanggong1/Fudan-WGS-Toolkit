"""Worker allocation and failed-fit gates without launching CUDA or jobs."""
import pytest
from fudan_wgs_toolkit import phewas_run as run


@pytest.fixture
def isolated_worker(tmp_path, monkeypatch):
    from fudan_wgs_toolkit.phewas_cache import _stream_bindings
    container = tmp_path/'container'
    container.mkdir()
    for name in ('data.bin', 'headers.bin', 'counts.bin'):
        (container/name).write_bytes(b'01')
    affinity = {}
    records = []
    monkeypatch.setattr(run.os, 'sched_getaffinity',
                        lambda pid: affinity.get(pid, set(range(64))))
    monkeypatch.setattr(run.os, 'sched_setaffinity',
                        lambda pid, cpus: affinity.__setitem__(pid, set(cpus)))
    monkeypatch.setattr(run.torch, 'set_num_threads', lambda count: None)
    monkeypatch.setattr(run.torch, 'set_num_interop_threads', lambda count: None)
    monkeypatch.setattr(run.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(run.torch.cuda, 'set_device', lambda device: None)
    monkeypatch.setattr(run, '_implementation_sha256', lambda: 'source-proof')
    monkeypatch.setattr(run, 'atomic_json', lambda path, value: records.append((path, value)))
    plan = dict(output_directory=str(tmp_path), cpu_offsets={'0': 8},
                source_implementation_sha256='source-proof',
                genotype_streams=_stream_bindings([dict(container_directory=str(container))]))
    return plan, affinity, records


def test_worker_binds_preexisting_helper_threads_to_the_eight_cpu_slice(isolated_worker, monkeypatch):
    plan, affinity, records = isolated_worker
    monkeypatch.setattr(run, 'fit_models', lambda *args, **kwargs: {'errors': [], 'models': []})
    result = run._execute_worker(plan, fit_only=True)
    assert result['errors'] == []
    identity = next(value for path, value in records if path.name.endswith('identity.private.json'))
    assert identity['cpus'] == list(range(8, 16))
    assert identity['all_thread_affinities_verified'] is True
    assert len(identity['thread_affinities']) > 0
    assert all(cpus == list(range(8, 16)) for cpus in identity['thread_affinities'].values())
    assert affinity[0] == set(range(8, 16))
    assert all(cpus == set(range(8, 16)) for cpus in affinity.values())


def test_partial_null_fit_cannot_publish_a_successful_fit_process_exit(isolated_worker, monkeypatch):
    plan, _, records = isolated_worker
    monkeypatch.setattr(run, 'fit_models', lambda *args, **kwargs:
                        {'errors': [{'error_type': 'ArithmeticError'}], 'models': [{'trait_index': 0}]})
    with pytest.raises(RuntimeError, match='assigned null fits failed'):
        run.execute_worker(plan, fit_only=True)
    receipt = next(value for path, value in records if path.name.endswith('exit.anonymous.json'))
    assert receipt['exit_code'] == 1
    assert any(path.name.endswith('failed.private.json') for path, _ in records)


@pytest.mark.parametrize('failure',['none','association','writer_constructor'])
def test_worker_finally_closes_repository_and_preserves_original_error(isolated_worker,monkeypatch,failure):
    from fudan_wgs_toolkit.phewas_runtime import runtime
    from fudan_wgs_toolkit import phewas_resources
    plan,_,records=isolated_worker
    plan.update(configuration={'chromosomes':[] if failure=='none' else [{'name':'21'}]},
                host_memory_limit_gib=12)
    monkeypatch.setattr(run,'fit_models',lambda *args,**kwargs:{'errors':[]})
    monkeypatch.setattr(run,'_all_models',lambda *args:[])
    actions=[]
    class Repository:
        def __init__(self,*args,**kwargs):actions.append('repository')
        def summary(self):return {'endpoint':'before close'}
        def close(self):
            actions.append('close')
            if failure!='none':raise RuntimeError('cleanup failure must not mask original')
    class Writer:
        def __init__(self,*args,**kwargs):
            if failure=='writer_constructor':raise ArithmeticError('original writer failure')
        def finish(self):return {'complete':True}
        def abort(self):
            actions.append('abort')
            raise MemoryError('abort cleanup failure')
    policy=[]
    class Gate:
        def __init__(self,output,**kwargs):policy.append(kwargs)
        def lease_factory(self,*args):pass
    monkeypatch.setattr(run,'ModelRepository',Repository)
    monkeypatch.setattr(run,'StreamingCSVWriter',Writer)
    monkeypatch.setattr(phewas_resources,'HostMemoryGate',Gate)
    def association(*args,**kwargs):raise ArithmeticError('original association failure')
    monkeypatch.setattr(runtime,'run_batched_configuration',association)
    if failure=='none':
        report=run.execute_worker(plan)
        assert report['model_repository']=={'endpoint':'before close'}
        assert actions==['repository','close']
        assert policy==[dict(host_limit_gib=12,reserve_gib=None)]
        assert next(value for path,value in records if path.name.endswith('exit.anonymous.json'))['exit_code']==0
    else:
        with pytest.raises(ArithmeticError,match='original'):
            run.execute_worker(plan)
        assert actions[-1]=='close'
        assert ('abort' in actions)==(failure=='association')
        assert next(value for path,value in records if path.name.endswith('exit.anonymous.json'))['exit_code']==1


def test_diagnostic_or_exit_write_failure_does_not_replace_worker_error(isolated_worker,monkeypatch):
    plan,_,_=isolated_worker
    def original(*args,**kwargs):raise ArithmeticError('primary calculation failure')
    def disk_failure(*args,**kwargs):raise OSError('secondary report disk failure')
    monkeypatch.setattr(run,'_execute_worker',original)
    monkeypatch.setattr(run,'atomic_json',disk_failure)
    with pytest.raises(ArithmeticError,match='primary calculation'):
        run.execute_worker(plan)
