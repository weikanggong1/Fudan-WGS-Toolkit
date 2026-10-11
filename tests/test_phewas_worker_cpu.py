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
