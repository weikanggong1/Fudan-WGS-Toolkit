import ast
import gc
import json
from pathlib import Path
import pytest
import torch
from fudan_wgs_toolkit import statistics as stats
from fudan_wgs_toolkit import _cuda_memory as mem
from fudan_wgs_toolkit import phewas_resources as resources
from fudan_wgs_toolkit.precision_audit import DenseProductAudit, explicit_binary_spa_fp64

REFERENCE = Path(__file__).parent/'data/cct_reference_d64de49.py'
source = REFERENCE.read_text()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_cct_tensor')
namespace = dict(vars(stats))
exec(compile(ast.Module(body=[node], type_ignores=[]), str(REFERENCE), 'exec'), namespace)
reference = namespace['_cct_tensor']

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('n', [1, 2, 7, 32, 257, 4096, 4097])
@pytest.mark.parametrize('kind', ['normal', 'tiny', 'boundary', 'strided'])
@pytest.mark.parametrize('sync_light', [False, True])
def test_cct_exact_original(n, kind, sync_light):
    gen = torch.Generator().manual_seed(1823+n)
    p = torch.rand(n, dtype=torch.float64, generator=gen).cuda()
    if kind == 'tiny': p[::2] = 1e-30
    if kind == 'boundary':
        edge = torch.tensor([1e-16], dtype=torch.float64, device='cuda')
        p[::3] = edge
        p[1::3] = torch.nextafter(edge, edge.new_zeros(()))
        p[2::3] = torch.nextafter(edge, edge.new_ones(()))
    if kind == 'strided':
        storage = torch.empty(2*n, dtype=torch.float64, device='cuda')
        storage[::2] = p
        p = storage[::2]
    for weights in [None, torch.rand(n, generator=gen, dtype=torch.float64).cuda()]:
        for internal in [False, True]:
            a = reference(p, weights, internal=internal, sync_light=sync_light)
            b = stats._cct_tensor(p, weights, internal=internal, sync_light=sync_light)
            assert torch.equal(a.view(torch.int64), b.view(torch.int64))

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('values', [[], [float('nan')], [float('inf')], [-1.], [1.1], [0.,1.], [0.,.3], [1.,.3]])
@pytest.mark.parametrize('internal', [False, True])
def test_cct_boundaries_and_errors(values, internal):
    p = torch.tensor(values, device='cuda', dtype=torch.float64)
    def outcome(fn):
        try:
            with __import__('warnings').catch_warnings(record=True) as warnings:
                result = fn(p, internal=internal, sync_light=True)
            return ('ok', result.view(torch.int64).item(), [str(w.message) for w in warnings])
        except Exception as e: return (type(e), str(e))
    assert outcome(reference) == outcome(stats._cct_tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_native_allocator_snapshots_are_fresh_and_exact():
    for n in [0, 1024*1024, 17]:
        x = torch.empty(n, device='cuda', dtype=torch.float64)
        assert mem.allocator_bytes(0) == (torch.cuda.memory_allocated(0), torch.cuda.memory_reserved(0))
        del x
        gc.collect()
        assert mem.allocator_bytes(0) == (torch.cuda.memory_allocated(0), torch.cuda.memory_reserved(0))


def test_allocator_empty_and_failure_contract(monkeypatch):
    monkeypatch.setattr(mem, 'memory_stats_as_nested_dict', lambda device: {})
    assert mem.allocator_bytes(0) == (0,0)
    def fail(device): raise RuntimeError('native snapshot failure')
    monkeypatch.setattr(mem, 'memory_stats_as_nested_dict', fail)
    with pytest.raises(RuntimeError, match='native snapshot failure'): mem.allocator_bytes(0)


def test_durable_equal_budget_only_skips_phase_publication(tmp_path, monkeypatch):
    gate = resources.HostMemoryGate(tmp_path/'output', host_limit_gib=4, reserve_gib=.125)
    calls = []
    original = gate._fresh_task_memory
    def sampled(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(gate, '_fresh_task_memory', sampled)
    with gate.lease(1024*1024) as lease:
        writes = gate.metadata()['ledger_writes']
        syncs = gate.metadata()['ledger_fsyncs']
        samples = len(calls)
        for phase in ['first', 'second', 'third']:
            lease.set_future(1024*1024, phase=phase)
            assert lease.phase == phase
            assert lease.last_admission['ledger_phase_coalesced']
        assert gate.metadata()['ledger_writes'] == writes
        assert gate.metadata()['ledger_fsyncs'] == syncs
        assert len(calls) == samples
        lease.label = 'updated label'
        lease.set_future(1024*1024, phase='relabel')
        assert gate.metadata()['ledger_writes'] == writes+1
        assert json.loads(gate.path.read_text())['leases'][lease.token]['label'] == 'updated label'
        lease.set_future(2*1024*1024, phase='growth')
        assert len(calls) == samples+1
        assert gate.metadata()['ledger_fsyncs'] == syncs+2
        lease.set_future(0, phase='released budget')
        assert json.loads(gate.path.read_text())['leases'][lease.token]['future_bytes'] == 0
    assert not json.loads(gate.path.read_text())['leases']
    gate.close()


def test_audit_still_rejects_and_counts_hidden_fp64():
    a = torch.eye(2, dtype=torch.float64)
    audit = DenseProductAudit(forced=True)
    with pytest.raises(RuntimeError, match='hidden FP64'):
        with audit: a @ a
    assert audit.fp64_products == audit.observed_products == 1
    with explicit_binary_spa_fp64(), audit: a @ a
    assert audit.explicit_binary_spa_fp64_products == 1
    assert audit.observed_products == 2
