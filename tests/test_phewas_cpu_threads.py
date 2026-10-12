"""PyTorch intra-op CPU control contracts; no cache workers or GPU are started."""
from copy import deepcopy
import json
import os
import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.phewas_runtime import runtime
from fudan_wgs_toolkit import cli


@pytest.mark.parametrize('requested', [1, 2, 4])
def test_cpu_threads_apply_during_actual_cpu_operation_and_restore(monkeypatch, requested):
    original = torch.get_num_threads()
    interop = torch.get_num_interop_threads()
    environment = dict(os.environ)
    observed = []
    def compute(*args, **kwargs):
        observed.append(torch.get_num_threads())
        # This executes a CPU tensor operation while the requested setting is active.
        value = torch.ones((17, 13)) @ torch.ones((13, 11))
        assert value.device.type == 'cpu' and torch.all(value == 13)
        return {'cpu_operation_completed': True}
    monkeypatch.setattr(runtime, '_run', compute)
    report = runtime.run_configuration([], cache_specs={}, cpu_threads=requested)
    assert observed == [requested] and torch.get_num_threads() == original
    assert report['cpu_thread_control'] == dict(requested=requested, effective=requested,
                                              previous=original, restored=True, scope='pytorch_intraop')
    assert torch.get_num_interop_threads() == interop and dict(os.environ) == environment
    assert not runtime._LOCK.locked()


def test_python_default_is_two_cpu_intraop_threads(monkeypatch):
    observed = []
    monkeypatch.setattr(runtime, '_run', lambda *args, **kwargs: observed.append(torch.get_num_threads()) or {})
    report = runtime.run_configuration([], cache_specs={})
    assert observed == [2] and report['cpu_thread_control']['requested'] == 2


@pytest.mark.parametrize('failure', [ValueError, RuntimeError, KeyboardInterrupt])
def test_failed_run_restores_cpu_threads_and_releases_lock(monkeypatch, failure):
    original = torch.get_num_threads()
    def fail(*args, **kwargs):
        assert torch.get_num_threads() == 3
        raise failure('controlled failure')
    monkeypatch.setattr(runtime, '_run', fail)
    with pytest.raises(failure, match='controlled failure'):
        runtime.run_configuration([], cache_specs={}, cpu_threads=3)
    assert torch.get_num_threads() == original and not runtime._LOCK.locked()


@pytest.mark.parametrize('invalid', [True, False, 0, -1, 2.0, '2', None, np.int64(2)])
def test_invalid_cpu_threads_fail_before_thread_mutation_or_run(monkeypatch, invalid):
    called = []
    monkeypatch.setattr(torch, 'set_num_threads', lambda value: called.append(value))
    monkeypatch.setattr(runtime, '_run', lambda *args, **kwargs: called.append('run'))
    with pytest.raises(ValueError, match='positive integer'):
        runtime.run_configuration([], cache_specs={}, cpu_threads=invalid)
    assert called == [] and not runtime._LOCK.locked()


def test_busy_context_does_not_change_cpu_threads(monkeypatch):
    called = []
    monkeypatch.setattr(torch, 'set_num_threads', lambda value: called.append(value))
    assert runtime._LOCK.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match='already running'):
            runtime.run_configuration([], cache_specs={}, cpu_threads=4)
    finally:
        runtime._LOCK.release()
    assert called == []


def test_initial_thread_setup_failure_still_attempts_restore_and_unlocks(monkeypatch):
    state = {'threads': 7}
    calls = []
    monkeypatch.setattr(torch, 'get_num_threads', lambda: state['threads'])
    def set_threads(value):
        calls.append(value)
        state['threads'] = value
        if value == 2:
            raise RuntimeError('thread setup failed')
    monkeypatch.setattr(torch, 'set_num_threads', set_threads)
    with pytest.raises(RuntimeError, match='thread setup failed'):
        runtime.run_configuration([], cache_specs={})
    assert calls == [2, 7] and state['threads'] == 7 and not runtime._LOCK.locked()


def test_effective_setting_must_match_request(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, 'get_num_threads', lambda: 7)
    monkeypatch.setattr(torch, 'set_num_threads', lambda value: calls.append(value))
    with pytest.raises(RuntimeError, match='was not applied'):
        runtime.run_configuration([], cache_specs={}, cpu_threads=3)
    assert calls == [3, 7] and not runtime._LOCK.locked()


@pytest.mark.parametrize('requested', [None, 1, 8])
def test_public_cli_cpu_threads_are_forwarded_without_reading_inputs(tmp_path, monkeypatch, requested):
    from fudan_wgs_toolkit import run as public
    calls=[]
    monkeypatch.setattr(public,'run_WGS_all_binanry',lambda **arguments:calls.append(arguments) or {'completed':True})
    arguments=['phenotypes.csv','covariates.csv','prepared','--output-directory',str(tmp_path/'results')]
    if requested is not None:arguments+=['--cpu-threads',str(requested)]
    cli.main(arguments)
    assert calls[0]['cpu_threads']==(8 if requested is None else requested)
    assert calls[0]['prepared_directory']=='prepared'


@pytest.mark.parametrize('value', ['0','-2','1.5','True'])
def test_cli_rejects_invalid_threads_before_input_reads(tmp_path,value):
    with pytest.raises(SystemExit) as error:
        cli.main(['missing.csv','missing_covariates.csv','missing_prepared',
                  '--output-directory',str(tmp_path/'results'),'--cpu-threads',value])
    assert error.value.code==2 and not (tmp_path/'results').exists()


@pytest.mark.parametrize('value',[True,False,0,-1,1.5,'2',None])
def test_public_python_cpu_threads_fail_before_input_read(tmp_path,monkeypatch,value):
    from fudan_wgs_toolkit.run import run_WGS_all_binanry
    monkeypatch.setattr(runtime,'_run',lambda *args,**kwargs:pytest.fail('invalid threads ran'))
    with pytest.raises(ValueError,match='positive integer'):
        run_WGS_all_binanry('missing.csv','missing_covariates.csv','missing_prepared',
                    output_directory=tmp_path/'results',cpu_threads=value)
    assert not (tmp_path/'results').exists()


def test_retired_cpu_worker_flag_is_not_a_public_option(tmp_path):
    with pytest.raises(SystemExit) as error:
        cli.main(['a.csv','b.csv','prepared','--output-directory',str(tmp_path/'results'),'--cpu-workers','4'])
    assert error.value.code==2
