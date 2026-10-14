"""CPU environment contracts executing the mature dispatcher ASTs.

Process/worker execution and scientific files are fixtures. No GPU is launched;
these tests establish startup environment restoration, not a benchmark.
"""
import ast
import builtins
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'fudan_wgs_toolkit'
THREAD_ENV=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')
SPEC=importlib.util.spec_from_file_location('dispatcher_allocator_scope_fixture',PACKAGE/'_continuous_spawn_allocator.py')
ALLOCATOR=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(ALLOCATOR)


def dispatcher_ast(kind):
    """Compile original statements; do not copy the spawn/restore algorithm."""
    path=PACKAGE/('prepared_association.py' if kind=='prepared' else 'phewas_run.py')
    tree=ast.parse(path.read_text())
    if kind=='prepared':
        node=next(item for item in tree.body if isinstance(item,ast.FunctionDef) and item.name=='_dispatch')
    else:
        original=next(item for item in tree.body if isinstance(item,ast.FunctionDef) and item.name=='run_phewas')
        context_index=next(i for i,item in enumerate(original.body) if isinstance(item,ast.Assign)
            and any(isinstance(target,ast.Name) and target.id=='context' for target in item.targets)
            and isinstance(item.value,ast.Call) and isinstance(item.value.func,ast.Attribute)
            and item.value.func.attr=='get_context')
        # Keep the mature inline branch as well as every following spawn/join/
        # report statement. Input/model preparation before this branch is out
        # of this CPU environment fixture's scope.
        inline=original.body[context_index-1]
        assert isinstance(inline,ast.If)
        node=ast.FunctionDef(name='_dispatch',args=ast.arguments(posonlyargs=[],
            args=[ast.arg(arg=name) for name in ('plan','output','gpu_ids','cpu_threads')],
            vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),
            body=original.body[context_index-1:],decorator_list=[])
        if 'type_params' in ast.FunctionDef._fields:node.type_params=[]
    tree=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(tree)
    return compile(tree,str(path),'exec')


class Harness:
    def __init__(self,root,*,coordinate=False,fail=None):
        self.root=root;self.coordinate=coordinate;self.fail=fail
        self.a_started=threading.Event();self.b_at_scope=threading.Event()
        self.observed=[];self.closed_handles=[];self.join_calls=0;self.errors=[]

    def importer(self,name,globals=None,locals=None,fromlist=(),level=0):
        if name=='_continuous_spawn_allocator':
            if self.coordinate and threading.current_thread().name=='dispatch_B':self.b_at_scope.set()
            return SimpleNamespace(continuous_spawn_allocator_scope=ALLOCATOR.continuous_spawn_allocator_scope)
        if name=='phewas_partition':return SimpleNamespace(verify_dispatch_reports=lambda *args:{'complete':True})
        if name=='phewas_trait_csv':
            return SimpleNamespace(invocation_lease_for_spawn=object,
                close_pending_invocation_lease=lambda handle:self.closed_handles.append(handle),_note=lambda *args:None)
        return builtins.__import__(name,globals,locals,fromlist,level)

    def process(self,**arguments):
        owner=self;name=threading.current_thread().name;index=arguments['kwargs']['worker_id']
        class FakeProcess:
            pid=None;exitcode=0
            def start(self):
                owner.observed.append(dict(caller=name,worker=index,
                    environment={key:os.environ.get(key) for key in (*THREAD_ENV,'PYTHONMALLOC')}))
                if owner.coordinate and name=='dispatch_A' and index==0:
                    owner.a_started.set();assert owner.b_at_scope.wait(5),'peer never reached actual scope import'
                if owner.fail==(name,index):raise RuntimeError('fixture_process_start_failed')
            def is_alive(self):return False
            def terminate(self):raise AssertionError('fixture never owns a live OS process')
            def kill(self):raise AssertionError('fixture never owns a live OS process')
            def join(self,*args,**kwargs):owner.join_calls+=1
        return FakeProcess()

    def call(self,kind,name,*,threads=4,contract='continuous',family='gaussian',devices=(4,5)):
        output=self.root/name;output.mkdir()
        attempt='fixture';(output/f'association_dispatch.{attempt}.private.json').write_text('{}')
        for index in range(len(devices)):
            (output/f'worker_{index}.attempt_{attempt}.report.private.json').write_text('{}')
            (output/f'worker_{index}.report.private.json').write_text('{}')
        plan=dict(output_directory=str(output),attempt_id=attempt,input_contract=contract,
            families={'anonymous_fixture':family},_continuous_spawn_families=[family])
        namespace=dict(__builtins__=dict(vars(builtins),__import__=self.importer),Path=Path,json=json,
            os=SimpleNamespace(environ=os.environ,sched_getaffinity=lambda *args:{0},sched_setaffinity=lambda *args:None),
            mp=SimpleNamespace(get_context=lambda *args:SimpleNamespace(Process=self.process)),
            _worker=lambda *args,**kwargs:{},execute_worker=lambda *args,**kwargs:{},
            _inline_cpu_policy=lambda *args:nullcontext(),
            torch=SimpleNamespace(get_num_threads=lambda:4,set_num_threads=lambda *args:None),
            verify_dispatch_reports=lambda *args:{'complete':True},atomic_json=lambda *args:None)
        exec(dispatcher_ast(kind),namespace)
        if kind=='prepared':return namespace['_dispatch'](plan,[],dict(attempt_id=attempt,cpu_threads=threads),list(devices))
        return namespace['_dispatch'](plan,output,list(devices),threads)


@pytest.fixture
def original_environment(monkeypatch):
    values={THREAD_ENV[0]:'9',THREAD_ENV[1]:None,THREAD_ENV[2]:'3',THREAD_ENV[3]:''}
    for key,value in values.items():
        if value is None:monkeypatch.delenv(key,raising=False)
        else:monkeypatch.setenv(key,value)
    monkeypatch.delenv('PYTHONMALLOC',raising=False)
    return dict(values,PYTHONMALLOC=None)


@pytest.mark.parametrize('first,second',[('prepared','prepared'),('phewas','phewas'),
    ('prepared','phewas'),('phewas','prepared')])
def test_concurrent_dispatchers_capture_thread_environment_inside_the_shared_lock(tmp_path,original_environment,first,second):
    harness=Harness(tmp_path,coordinate=True)
    def call(kind,name,threads):
        try:harness.call(kind,name,threads=threads)
        except BaseException as error:harness.errors.append(error)
    a=threading.Thread(target=call,args=(first,'A',4),name='dispatch_A')
    b=threading.Thread(target=call,args=(second,'B',2),name='dispatch_B')
    a.start();assert harness.a_started.wait(5)
    b.start();a.join(10);b.join(10)
    assert not a.is_alive() and not b.is_alive() and not harness.errors,harness.errors
    assert {key:os.environ.get(key) for key in original_environment}==original_environment
    for item in harness.observed:
        expected='4' if item['caller']=='dispatch_A' else '2'
        assert all(item['environment'][key]==expected for key in THREAD_ENV)
        supported=sys.implementation.name=='cpython' and sys.platform=='linux' and not sys.flags.ignore_environment
        assert item['environment']['PYTHONMALLOC']==('malloc' if supported else None)


@pytest.mark.parametrize('kind',['prepared','phewas'])
def test_actual_dispatcher_start_failure_restores_environment_and_retires_unstarted_handle(tmp_path,original_environment,kind):
    harness=Harness(tmp_path,fail=(threading.current_thread().name,1))
    with pytest.raises(RuntimeError,match='fixture_process_start_failed'):harness.call(kind,'failure')
    assert len(harness.closed_handles)==1 and harness.join_calls>=1
    assert {key:os.environ.get(key) for key in original_environment}==original_environment


@pytest.mark.parametrize('kind',['prepared','phewas'])
@pytest.mark.parametrize('contract,family',[('mixed','gaussian'),('continuous','binomial')])
def test_inactive_allocator_policy_preserves_parent_settings_in_the_mature_dispatchers(tmp_path,original_environment,kind,contract,family):
    harness=Harness(tmp_path);harness.call(kind,'inactive',contract=contract,family=family)
    assert all(item['environment']['PYTHONMALLOC'] is None for item in harness.observed)
    assert {key:os.environ.get(key) for key in original_environment}==original_environment


@pytest.mark.parametrize('kind',['prepared','phewas'])
def test_single_gpu_inline_dispatch_does_not_set_startup_environment(tmp_path,original_environment,kind):
    harness=Harness(tmp_path);result=harness.call(kind,'inline',devices=(4,))
    assert len(result)==1 and harness.observed==[]
    assert {key:os.environ.get(key) for key in original_environment}==original_environment


@pytest.mark.parametrize('kind',['prepared','phewas'])
def test_explicit_parent_allocator_setting_remains_exact(tmp_path,original_environment,monkeypatch,kind):
    monkeypatch.setenv('PYTHONMALLOC','pymalloc_debug')
    harness=Harness(tmp_path);harness.call(kind,'explicit')
    assert all(item['environment']['PYTHONMALLOC']=='pymalloc_debug' for item in harness.observed)
    assert os.environ['PYTHONMALLOC']=='pymalloc_debug'
    assert {key:os.environ.get(key) for key in THREAD_ENV}=={key:original_environment[key] for key in THREAD_ENV}
