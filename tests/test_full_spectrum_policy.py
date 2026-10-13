"""Complete-spectrum policy/memory/lifetime contracts, not real-data gates."""
import gc
import inspect
import weakref
from unittest.mock import patch

import pytest
import torch

from fudan_wgs_toolkit import _weighted_spectra as weighted, statistics, run, phewas_run
from fudan_wgs_toolkit.pipeline import AnalysisOptions
from fudan_wgs_toolkit.phewas_cache import _analysis_defaults
from fudan_wgs_toolkit.phewas_runtime.runtime import _gene_tail_new_bytes


def inputs():
    c = torch.tensor([[3., .2, .1], [.2, 2., .3], [.1, .3, 1.]], dtype=torch.float32)
    u = torch.tensor([.5, 1., .3], dtype=torch.float32)
    return u, c, torch.tensor([.001, .002, .004]), torch.tensor([4., 16., 24.])


def call_native(**options):
    u, c, maf, mac = inputs()
    with patch('fudan_wgs_toolkit._burden.ieee_burden_product', new=lambda a, b: a@b):
        return statistics.association_test(u, c, maf, mac,
            annotations=[[10.], [20.], [30.]], names=['functional'], matmul_mode='tf32', **options)


def test_public_and_bound_defaults_choose_full_spectrum():
    for function in (statistics.association_test, phewas_run.build_configuration,
                     phewas_run.prepare_phewas_run, phewas_run.run_phewas):
        assert inspect.signature(function).parameters['long_mask_method'].default == 'full'
    assert AnalysisOptions().long_mask_method == 'full'
    defaults = _analysis_defaults(20, 4096, 5000, 512, 1729)
    assert defaults['long_mask_method'] == 'full'
    assert _analysis_defaults(20, 4096, 5000, 512, 1729,
                             long_mask_method='fastskat')['long_mask_method'] == 'fastskat'


def test_sequential_full_preserves_each_native_matrix_without_proportional_scaling(monkeypatch):
    _, c, _, _ = inputs()
    d = torch.tensor([[1., 2., .3], [2., 4., 1.7], [3., 6., 2.9]], dtype=torch.float32)
    original = torch.linalg.eigvalsh
    expected = torch.stack([original((c*w[:, None])*w[None, :], UPLO='U').double() for w in d.T])
    calls = []
    def leaf(matrix, **kwargs):
        calls.append((matrix.dtype, tuple(matrix.shape), kwargs))
        return original(matrix, **kwargs)
    monkeypatch.setattr(torch.linalg, 'eigvalsh', leaf)
    monkeypatch.setattr(statistics, '_native_weight_relations',
                        lambda *a, **k: pytest.fail('full mode reused/scaled a spectrum'))
    before_c, before_d = c.clone(), d.clone()
    actual = weighted.sequential_complete_spectra(c, d)
    assert torch.equal(actual, expected)
    assert calls == [(torch.float32, (3, 3), {'UPLO': 'U'})]*3
    assert torch.equal(c, before_c) and torch.equal(d, before_d)


def test_default_full_has_no_forced_5000_or_small_threshold_dispatch(monkeypatch):
    monkeypatch.setattr(statistics, '_fastskat_hybrid_pvalues',
                        lambda *a, **k: pytest.fail('full entered approximate spectrum'))
    monkeypatch.setattr(statistics, '_liu_hutchinson_pvalues',
                        lambda *a, **k: pytest.fail('full entered Liu'))
    monkeypatch.setattr(statistics, '_native_weighted_spectra',
                        lambda *a, **k: pytest.fail('full entered legacy proportional batching'))
    a = call_native(long_mask_threshold=1)
    b = call_native(long_mask_threshold=5000)
    assert a == b
    assert a['num_variant'] == 3 and a['cMAC'] == 44
    meta = statistics.statistics_execution_metadata()
    assert meta['spectrum_policy'] == 'full' and meta['exact_spectrum_threshold'] is None
    assert meta['tail_metadata'][-1]['approximate'] is False


@pytest.mark.parametrize('method,leaf', [('fastskat', '_fastskat_hybrid_pvalues'),
                                         ('liu', '_liu_hutchinson_pvalues')])
def test_explicit_legacy_approximation_remains_available(method, leaf, monkeypatch):
    called = []
    def original_approx(q, c, d, **kwargs):
        called.append((c.shape, kwargs))
        return torch.full_like(q, .25, dtype=torch.float64)
    monkeypatch.setattr(statistics, leaf, original_approx)
    monkeypatch.setattr(weighted, 'sequential_complete_spectra',
                        lambda *a, **k: pytest.fail('explicit legacy mode entered full helper'))
    result = call_native(long_mask_threshold=2, long_mask_method=method, long_mask_rank=1)
    assert result['num_variant'] == 3 and len(called) == 1
    assert called[0][1]['threshold'] == 2


def test_full_tail_is_the_established_same_configuration_complete_spectrum():
    u, c, maf, _ = inputs()
    _, d, _ = statistics.annotation_weights(maf.double(), torch.tensor([[10.], [20.], [30.]]),
                                            dtype=torch.float32)
    q = (u.square()[:, None]*d.square()).sum(dim=0)
    spectra = weighted.sequential_complete_spectra(c, d)
    p = statistics._saddlepoint_skat_pvalues(q, spectra)
    actual = call_native(long_mask_threshold=1)
    control = call_native(long_mask_threshold=1, _skat_pvalues=p)
    assert actual == control


def test_legacy_batch_preference_does_not_reject_one_legal_full_matrix(monkeypatch):
    monkeypatch.setattr(weighted, 'WEIGHTED_BATCH_LIMIT_BYTES', 1)
    result = weighted.sequential_complete_spectra(torch.eye(3), torch.ones(3, 2))
    assert torch.equal(result, torch.ones(2, 3, dtype=torch.float64))
    assert weighted.execution_metadata()['batch_limit_scope'].startswith('legacy proportional')


def test_explicit_working_set_accounting_includes_spectral_output_conversion():
    assert weighted.full_workspace_unit_bytes(7) == 16*7*7 + 12*7
    with pytest.raises(ValueError):
        weighted.full_workspace_unit_bytes(0)
    full = AnalysisOptions(long_mask_threshold=2, long_mask_method='full')
    approximate = AnalysisOptions(long_mask_threshold=2, long_mask_method='fastskat', long_mask_rank=1)
    assert _gene_tail_new_bytes(20, 3, full) == 32*3*3 + 4*20 + 256*2**20
    assert _gene_tail_new_bytes(20, 3, approximate) == 16*3*3 + 64*3 + 4*20


def test_retained_solver_traceback_releases_both_inputs_and_weighted_matrix(monkeypatch):
    owners = []
    def failure(matrix, **kwargs):
        owners.append(weakref.ref(matrix))
        # Deliberately retain the leaf frame to prove recursive cleanup.
        held = matrix.clone()
        owners.append(weakref.ref(held))
        raise RuntimeError('injected full eigensolver failure')
    monkeypatch.setattr(torch.linalg, 'eigvalsh', failure)
    def failed_call():
        c, d = torch.eye(3), torch.ones(3, 2)
        owners.extend([weakref.ref(c), weakref.ref(d)])
        try:
            weighted.sequential_complete_spectra(c, d)
        except RuntimeError as error:
            return error
        finally:
            c = d = None
    error = failed_call()
    gc.collect()
    assert error.__traceback__ is not None
    assert all(owner() is None for owner in owners)


def test_full_result_does_not_keep_input_autograd_owners():
    c, d = torch.eye(3, requires_grad=True), torch.ones(3, 2, requires_grad=True)
    result = weighted.sequential_complete_spectra(c, d)
    assert result.grad_fn is None and not result.requires_grad


@pytest.mark.parametrize('method', [None, 'lanczos', 'FULL'])
def test_invalid_method_rejected_without_scientific_fallback(method):
    with pytest.raises(ValueError, match='long_mask_method'):
        call_native(long_mask_method=method)
    with pytest.raises(ValueError, match='long_mask_method'):
        AnalysisOptions(long_mask_method=method)


@pytest.mark.parametrize("option", ["--long-mask-method", "--long-mask-threshold", "--long-mask-rank"])
def test_binary_public_cli_rejects_spectrum_options(monkeypatch, option):
    signature = inspect.signature(run.run_WGS_all_binanry)
    assert "long_mask_method" not in signature.parameters
    selected = []
    monkeypatch.setattr(run, "run_WGS_all_binanry", lambda **kwargs: selected.append(kwargs) or {"completed": True})
    with pytest.raises(SystemExit) as error:
        run.main(["phenotypes.csv", "covariates.csv", "prepared", "--output-directory", "results",
                  option, "1"])
    assert error.value.code == 2 and selected == []


def test_existing_plan_cli_cannot_change_bound_spectrum_policy(tmp_path, monkeypatch):
    import json
    path = tmp_path/'plan.json'
    path.write_text(json.dumps({'configuration': {'analysis_options': {'long_mask_method': 'fastskat'}}}))
    monkeypatch.setattr(phewas_run, 'execute_worker', lambda *a, **k: pytest.fail('modified plan executed'))
    with pytest.raises(SystemExit):
        phewas_run.main(['--plan', str(path), '--long-mask-method', 'full'])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA allocation/live guard contract')
def test_cuda_full_rechecks_new_workspace_for_every_weight(monkeypatch):
    from fudan_wgs_toolkit import _fit_structure_cache
    c = torch.eye(4, device='cuda')
    d = torch.arange(1., 4., device='cuda')[None, :].expand(4, -1).clone()
    calls = []
    def guard(new_bytes, device, **kwargs):
        calls.append((new_bytes, kwargs['context']))
    monkeypatch.setattr(_fit_structure_cache, 'guard_cuda_workspace', guard)
    before_tf32 = torch.backends.cuda.matmul.allow_tf32
    result = weighted.sequential_complete_spectra(c, d)
    unit = weighted.full_workspace_unit_bytes(4)
    assert calls == [(8*3*4+unit, 'complete STAAR output/workspace')] + \
                    [(unit, 'complete STAAR weight workspace')]*3
    assert result.device == c.device and result.dtype == torch.float64
    assert torch.equal(result, d.T.double().square())
    assert torch.backends.cuda.matmul.allow_tf32 == before_tf32


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA allocation/live guard contract')
def test_cuda_full_budget_rejection_precedes_first_matrix_or_solver(monkeypatch):
    from fudan_wgs_toolkit import _fit_structure_cache
    c, d = torch.eye(4, device='cuda'), torch.ones(4, 3, device='cuda')
    def guard(new_bytes, device, **kwargs):
        raise MemoryError('injected live workspace rejection')
    monkeypatch.setattr(_fit_structure_cache, 'guard_cuda_workspace', guard)
    monkeypatch.setattr(torch.linalg, 'eigvalsh', lambda *a, **k: pytest.fail('rejected full set solved'))
    allocated = torch.cuda.memory_allocated(c.device)
    with pytest.raises(MemoryError, match='live workspace'):
        weighted.sequential_complete_spectra(c, d)
    assert torch.cuda.memory_allocated(c.device) == allocated
