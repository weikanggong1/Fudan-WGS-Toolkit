"""Synthetic unit oracles for complete H32 batching and bounded exact reuse."""
from contextlib import nullcontext
from types import SimpleNamespace
import gc
import weakref

import pytest
import torch

from fudan_wgs_toolkit import _weighted_spectra as weighted
from fudan_wgs_toolkit import _complete_spectrum_scope as scopes


@pytest.fixture(autouse=True)
def reset_metadata():
    weighted.execution_metadata(reset=True)
    yield
    weighted.execution_metadata(reset=True)


def operands(m=7):
    generator = torch.Generator().manual_seed(716)
    matrix = torch.randn((m, m), generator=generator)
    covariance = matrix @ matrix.T + torch.eye(m)
    first = torch.linspace(.1, 1.7, m)
    near = first.clone()
    near[-1] = torch.nextafter(near[-1], torch.tensor(float('inf')))
    weights = torch.stack((first, first, torch.zeros(m), torch.zeros(m),
                           2 * first, near), dim=1)
    return covariance, weights


def leaf_spy(monkeypatch, behavior=None):
    original = torch.linalg.eigvalsh
    calls = []
    def solve(matrix, **kwargs):
        calls.append(matrix.detach().clone())
        if behavior is not None:
            behavior(matrix)
        return original(matrix, **kwargs)
    monkeypatch.setattr(torch.linalg, 'eigvalsh', solve)
    return calls


@pytest.mark.parametrize('batch_size', [2, 3, 32])
def test_complete_batches_keep_each_H32_bit_exact_and_all_rows(monkeypatch, batch_size):
    covariance, weights = operands()
    expected = weighted.sequential_complete_spectra(covariance, weights)
    weighted.execution_metadata(reset=True)
    calls = leaf_spy(monkeypatch)
    with scopes.complete_spectrum_batch_scope(weight_batch_size=batch_size):
        actual = weighted.sequential_complete_spectra(covariance, weights)
    assert torch.equal(actual, expected)
    assert actual.shape == (6, len(covariance))
    representatives = [0, 2, 4, 5]
    actual_matrices = [row for batch in calls for row in (batch if batch.ndim == 3 else batch[None])]
    assert len(actual_matrices) == len(representatives)
    for matrix, index in zip(actual_matrices, representatives):
        weight = weights[:, index]
        assert matrix.dtype == torch.float32
        assert torch.equal(matrix, (covariance*weight[:, None])*weight[None, :])
    report = weighted.execution_metadata()
    assert report['full_requested_weights'] == report['full_sequential_spectra'] == 6
    assert report['full_exact_duplicate_reused_spectra'] == 2
    assert report['full_fp32_solver_matrices'] == 4
    assert report['full_fp32_solver_calls'] == len(calls)
    assert report['complete_spectrum_optimization']['max_relation_workspace_bytes'] <= 256*2**10


def test_signed_zero_near_and_proportional_columns_are_not_duplicate():
    covariance, weights = operands(4)
    weights[0, 0] = 0.
    weights[0, 1] = -0.
    representatives, mapping = scopes.exact_weight_mapping(weights, 4096)
    assert representatives == [0, 1, 2, 4, 5]
    assert mapping == [0, 1, 2, 2, 4, 5]


@pytest.mark.parametrize('budget', [0, 1, 64])
def test_relation_budget_decline_keeps_original_complete_route(monkeypatch, budget):
    covariance, weights = operands()
    expected = weighted.sequential_complete_spectra(covariance, weights)
    weighted.execution_metadata(reset=True)
    calls = leaf_spy(monkeypatch)
    with scopes.complete_spectrum_batch_scope(relation_workspace_bytes=budget):
        actual = weighted.sequential_complete_spectra(covariance, weights)
    assert torch.equal(actual, expected)
    assert all(matrix.ndim == 2 for matrix in calls)
    assert weighted.execution_metadata()['complete_spectrum_optimization']['relation_budget_declines'] == 1


def test_large_M_above_batch_preference_still_solves_every_full_row(monkeypatch):
    covariance, weights = operands()
    calls = leaf_spy(monkeypatch)
    with scopes.complete_spectrum_batch_scope(maximum_variants=3):
        result = weighted.sequential_complete_spectra(covariance, weights)
    assert result.shape == (6, 7)
    assert len(calls) == 4 and all(matrix.ndim == 2 for matrix in calls)
    assert weighted.execution_metadata()['full_batch_api_calls'] == 0


def test_batch_convergence_failure_unwinds_before_mature_ordered_full_fallback(monkeypatch):
    covariance, weights = operands()
    expected = weighted.sequential_complete_spectra(covariance, weights)
    weighted.execution_metadata(reset=True)
    def fail_batch(matrix):
        if matrix.ndim == 3:
            raise torch.linalg.LinAlgError('algorithm failed to converge')
    calls = leaf_spy(monkeypatch, fail_batch)
    with scopes.complete_spectrum_batch_scope():
        actual = weighted.sequential_complete_spectra(covariance, weights)
    assert torch.equal(actual, expected)
    assert calls[0].ndim == 3 and len(calls) == 5
    assert all(matrix.dtype == torch.float32 for matrix in calls)
    report = weighted.execution_metadata()
    assert report['full_batch_convergence_failures'] == 1
    assert report['full_batch_sequential_fallbacks'] == 4
    assert report['full_retry_attempts'] == 0
    assert report['full_requested_weights'] == report['full_sequential_spectra'] == 6


def test_nonconvergence_batch_failure_is_not_silently_retried(monkeypatch):
    covariance, weights = operands()
    calls = leaf_spy(monkeypatch, lambda matrix: (_ for _ in ()).throw(RuntimeError('illegal API invocation')))
    with scopes.complete_spectrum_batch_scope(), pytest.raises(RuntimeError, match='illegal API'):
        weighted.sequential_complete_spectra(covariance, weights)
    assert len(calls) == 1
    assert weighted.execution_metadata()['full_batch_sequential_fallbacks'] == 0


@pytest.mark.parametrize('m,selected', [(32, False), (33, True), (128, True), (129, False), (131, False), (512, False), (513, False)])
def test_explicit_full_batch_uses_owned_selector_only_in_its_complete_range(monkeypatch, m, selected):
    # CPU transport spy only. It checks complete H32 construction, route and
    # metrics; actual CUDA Jacobi precision/performance is separately gated.
    covariance = torch.diag(torch.arange(1., m+1))
    weights = torch.stack((torch.ones(m), torch.full((m,), 2.), torch.full((m,), 3.)), dim=1)
    calls = []
    class Solver:
        last_execution_backend = None
        def eigvalsh(self, matrices, *, UPLO):
            assert UPLO == 'U' and matrices.dtype == torch.float32
            calls.append(matrices.clone())
            self.last_execution_backend = 'cusolverDnSsyevjBatched'
            return matrices.diagonal(dim1=-2, dim2=-1).sort().values
    monkeypatch.setattr(weighted, '_SOLVER', Solver())
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda value: True))
    monkeypatch.setattr(weighted, '_full_batch_capacity', lambda c, remaining, maximum: min(remaining, maximum))
    from fudan_wgs_toolkit import _fit_structure_cache
    monkeypatch.setattr(_fit_structure_cache, 'guard_cuda_workspace', lambda *args, **kwargs: None)
    with scopes.complete_spectrum_batch_scope(maximum_variants=1536):
        result = weighted.sequential_complete_spectra(covariance, weights)
    expected = torch.stack([(covariance*weight[:, None])*weight[None, :]
                            for weight in weights.T])
    assert result.shape == (3, m) and torch.equal(result, expected.diagonal(dim1=-2, dim2=-1).double())
    assert len(calls) == int(selected)
    if selected:
        assert torch.equal(calls[0], expected)  # Same complete represented H32.
    report = weighted.execution_metadata()
    route = 'cusolverDnSsyevjBatched' if selected else 'torch.linalg.eigvalsh:complete_weight_batch'
    assert report['actual_backend_calls'] == {route: 1}
    assert report['actual_backend_matrices'] == {route: 3}
    assert report['eigen_api_calls'] == 1
    assert report['full_selected_solver_calls'] == int(selected)
    assert report['full_selected_solver_matrices'] == 3*int(selected)
    assert report['full_requested_weights'] == report['full_sequential_spectra'] == 3


@pytest.mark.parametrize('error', [MemoryError('fresh selector workspace refused'),
    RuntimeError('cuSOLVER eigen batch failed/nonconverged; info must be zero for every matrix')])
def test_selected_API_or_budget_failure_is_not_guessed_to_be_pure_convergence(monkeypatch, error):
    covariance = torch.eye(33)
    weights = torch.stack((torch.ones(33), torch.full((33,), 2.)), dim=1)
    refs = []
    class Solver:
        last_execution_backend = None
        def eigvalsh(self, matrix, *, UPLO):
            refs.append(weakref.ref(matrix))
            raise error
    monkeypatch.setattr(weighted, '_SOLVER', Solver())
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda value: True))
    monkeypatch.setattr(weighted, '_full_batch_capacity', lambda *args: 2)
    from fudan_wgs_toolkit import _fit_structure_cache
    monkeypatch.setattr(_fit_structure_cache, 'guard_cuda_workspace', lambda *args, **kwargs: None)
    with scopes.complete_spectrum_batch_scope(), pytest.raises(type(error)) as captured:
        weighted.sequential_complete_spectra(covariance, weights)
    assert captured.value is error
    assert error.weighted_spectrum_failure == 'selected_complete_fp32_solve_failed'
    gc.collect()
    assert all(ref() is None for ref in refs)
    report = weighted.execution_metadata()
    assert report['full_selected_solver_failures'] == 1
    assert report['full_retry_attempts'] == report['full_batch_sequential_fallbacks'] == 0


def test_selected_typed_convergence_failure_preserves_original_ordered_full_fallback(monkeypatch):
    covariance, weights = operands(33)
    expected = weighted.sequential_complete_spectra(covariance, weights)
    weighted.execution_metadata(reset=True)
    class Solver:
        last_execution_backend = None
        def eigvalsh(self, matrix, *, UPLO):
            raise torch.linalg.LinAlgError('algorithm failed to converge')
    calls = leaf_spy(monkeypatch)
    monkeypatch.setattr(weighted, '_SOLVER', Solver())
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda value: True))
    monkeypatch.setattr(weighted, '_full_batch_capacity', lambda c, remaining, maximum: remaining)
    from fudan_wgs_toolkit import _fit_structure_cache
    monkeypatch.setattr(_fit_structure_cache, 'guard_cuda_workspace', lambda *args, **kwargs: None)
    with scopes.complete_spectrum_batch_scope():
        result = weighted.sequential_complete_spectra(covariance, weights)
    assert torch.equal(result, expected)
    for matrix, index in zip(calls, (0, 2, 4, 5)):
        assert torch.equal(matrix, (covariance*weights[:, index, None])*weights[None, :, index])
    report = weighted.execution_metadata()
    assert report['full_selected_solver_failures'] == report['full_batch_convergence_failures'] == 1
    assert report['full_batch_sequential_fallbacks'] == 4 and report['full_retry_attempts'] == 0
    assert report['full_requested_weights'] == report['full_sequential_spectra'] == 6


def test_same_scope_same_C_and_new_bit_identical_weights_reuse_full_spectrum(monkeypatch):
    covariance, weights = operands()
    calls = leaf_spy(monkeypatch)
    with scopes.same_covariance_spectrum_scope():
        first = weighted.sequential_complete_spectra(covariance, weights)
        second = weighted.sequential_complete_spectra(covariance, weights.clone())
        assert torch.equal(first, second)
        assert scopes.execution_metadata()['live_retained_bytes'] > 0
    report = weighted.execution_metadata()
    assert len(calls) == 4
    assert report['full_requested_weights'] == report['full_sequential_spectra'] == 12
    assert report['full_scope_reused_spectra'] == 6
    assert report['complete_spectrum_optimization']['reuse_hits'] == 1
    assert report['complete_spectrum_optimization']['live_retained_bytes'] == 0


@pytest.mark.parametrize('change', ['different_C_object', 'mutate_C', 'mutate_D', 'mutate_output'])
def test_reuse_rejects_changed_inputs_and_mutable_cached_result(monkeypatch, change):
    covariance, weights = operands()
    calls = leaf_spy(monkeypatch)
    with scopes.same_covariance_spectrum_scope():
        first = weighted.sequential_complete_spectra(covariance, weights)
        if change == 'different_C_object':
            covariance = covariance.clone()
        elif change == 'mutate_C':
            covariance.add_(torch.eye(len(covariance)))
        elif change == 'mutate_D':
            weights[0, 0] = torch.nextafter(weights[0, 0], torch.tensor(float('inf')))
        else:
            first[0, 0].add_(3.)
        actual = weighted.sequential_complete_spectra(covariance, weights)
    expected = weighted.sequential_complete_spectra(covariance, weights)
    assert torch.equal(actual, expected)
    assert len(calls) >= 12
    assert scopes.execution_metadata()['reuse_hits'] == 0


def test_reuse_has_explicit_byte_budget_and_no_cache_between_scopes(monkeypatch):
    covariance, weights = operands()
    calls = leaf_spy(monkeypatch)
    with scopes.same_covariance_spectrum_scope(cache_limit_bytes=0):
        weighted.sequential_complete_spectra(covariance, weights)
        weighted.sequential_complete_spectra(covariance, weights)
    with scopes.same_covariance_spectrum_scope():
        weighted.sequential_complete_spectra(covariance, weights)
    with scopes.same_covariance_spectrum_scope():
        weighted.sequential_complete_spectra(covariance, weights)
    assert len(calls) == 16
    assert scopes.execution_metadata()['reuse_budget_declines'] == 2
    assert scopes.execution_metadata()['reuse_hits'] == 0


def test_inference_tensors_without_mutation_counters_decline_reuse_safely(monkeypatch):
    calls = leaf_spy(monkeypatch)
    with torch.inference_mode(), scopes.same_covariance_spectrum_scope():
        covariance, weights = operands()
        first = weighted.sequential_complete_spectra(covariance, weights)
        second = weighted.sequential_complete_spectra(covariance, weights)
        assert torch.equal(first, second)
    assert len(calls) == 8
    assert scopes.execution_metadata()['reuse_unversioned_declines'] == 2
    assert scopes.execution_metadata()['reuse_hits'] == 0


def test_relation_fp64_input_is_not_reinterpreted_as_doubled_fp32_columns():
    with pytest.raises(ValueError, match='FP32'):
        scopes.exact_weight_mapping(torch.ones(5, 3, dtype=torch.float64), 4096)


def test_scope_releases_original_owners_even_when_caller_retains_exception():
    refs = []
    def call():
        covariance, weights = operands()
        refs.extend((weakref.ref(covariance), weakref.ref(weights)))
        try:
            with scopes.same_covariance_spectrum_scope():
                weighted.sequential_complete_spectra(covariance, weights)
                raise RuntimeError('caller failed')
        finally:
            covariance = weights = None
    with pytest.raises(RuntimeError) as captured:
        call()
    gc.collect()
    assert captured.value.__traceback__ is not None
    assert all(ref() is None for ref in refs)
    assert scopes.execution_metadata()['live_retained_bytes'] == 0


def test_extreme_score_reevaluation_reuses_only_spectrum_and_preserves_new_Q(monkeypatch):
    from fudan_wgs_toolkit import statistics
    from fudan_wgs_toolkit import _burden
    from fudan_wgs_toolkit.phewas_runtime import runtime
    # Isolate the covariance/spectrum lifetime on CPU. This is a unit oracle,
    # not a claim about the real native CUDA burden kernel or its performance.
    monkeypatch.setattr(_burden, 'ieee_burden_product', lambda left, right: left @ right)
    covariance, unused = operands(5)
    maf = torch.linspace(.002, .008, 5)
    first_score, original_score = torch.ones(5), torch.arange(1., 6.)
    payload = dict(score=first_score, covariance=covariance, maf=maf, mac=torch.ones(5),
                   cmac=25., rv_num_cutoff=1)
    expected = statistics.association_test(**dict(payload, score=original_score), matmul_mode='tf32')
    weighted.execution_metadata(reset=True)
    calls = leaf_spy(monkeypatch)
    observed = []
    def evaluate(value, model):
        result = statistics.association_test(**value, matmul_mode='tf32')
        observed.append(result)
        return result
    monkeypatch.setattr(runtime, '_gene_normal_score_extreme', lambda *args: False)
    monkeypatch.setattr(runtime, '_gene_probability_extreme', lambda value: len(observed) == 1)
    monkeypatch.setattr(runtime, '_shared_gene_scores', lambda *args: original_score[:, None])
    pipeline = SimpleNamespace(_evaluate_prepared=evaluate,
        profiler=SimpleNamespace(measure=lambda *args, **kwargs: nullcontext()))
    report = {}
    actual = runtime._evaluate_gaussian_gene_score(pipeline, payload, SimpleNamespace(),
        torch.empty(0), SimpleNamespace(), shared_simt=True, report=report,
        reuse_complete_spectra=True)
    assert actual == expected and len(observed) == 2
    assert observed[0]['WGS-O'] != observed[1]['WGS-O']
    assert len(calls) == 2  # The two beta families, once; Q changes on reevaluation.
    assert report['gene_score_additional_tail_evaluations'] == 1
    assert scopes.execution_metadata()['reuse_hits'] == 1
    assert scopes.execution_metadata()['live_retained_bytes'] == 0


def test_changed_actual_rare_selection_cannot_reuse_equal_shape_or_weights(monkeypatch):
    from fudan_wgs_toolkit import statistics, _burden
    monkeypatch.setattr(_burden, 'ieee_burden_product', lambda left, right: left @ right)
    covariance, unused = operands(5)
    first_maf = torch.tensor([0., .005, .005, .005, .005])
    second_maf = torch.tensor([.005, 0., .005, .005, .005])
    payload = dict(score=torch.arange(1., 6.), covariance=covariance,
                   mac=torch.ones(5), cmac=25., rv_num_cutoff=1)
    with scopes.same_covariance_spectrum_scope():
        statistics.association_test(**payload, maf=first_maf, matmul_mode='tf32')
        actual = statistics.association_test(**payload, maf=second_maf, matmul_mode='tf32')
    expected = statistics.association_test(**payload, maf=second_maf, matmul_mode='tf32')
    assert actual == expected
    assert scopes.execution_metadata()['reuse_hits'] == 0
    assert scopes.execution_metadata()['reuse_input_changes'] == 1


def test_batch_capacity_uses_fresh_total_budget_and_accounts_gathered_D(monkeypatch):
    from fudan_wgs_toolkit import tf32, _fit_structure_cache
    covariance = SimpleNamespace(shape=(7, 7), is_cuda=True, device=torch.device('cuda'))
    unit = weighted.full_workspace_unit_bytes(7)+4*7
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 900)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 1000)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (5000, 10000))
    snapshots = []
    def availability(**kwargs):
        snapshots.append(kwargs)
        return dict(available_bytes=2*unit+1)
    monkeypatch.setattr(tf32, '_product_workspace_availability', availability)
    admissions = []
    monkeypatch.setattr(_fit_structure_cache, 'guard_cuda_workspace',
        lambda required, device, **kwargs: admissions.append(required))
    assert weighted._full_batch_capacity(covariance, 20, 8) == 2
    assert admissions == [2*unit]
    assert snapshots[0]['allocated'] == 900 and snapshots[0]['free'] == 5000
    def rejected(*args, **kwargs):
        raise MemoryError('explicit whole-process admission rejected')
    monkeypatch.setattr(_fit_structure_cache, 'guard_cuda_workspace', rejected)
    with pytest.raises(MemoryError, match='whole-process'):
        weighted._full_batch_capacity(covariance, 20, 8)
    assert weighted.execution_metadata()['full_batch_sequential_fallbacks'] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA complete batching')
def test_cuda_batch_complete_probability_oracle_and_live_admission(monkeypatch):
    from fudan_wgs_toolkit import _fit_structure_cache
    from fudan_wgs_toolkit.statistics import _saddlepoint_skat_pvalues
    covariance, weights = operands(23)
    covariance, weights = covariance.cuda(), weights.cuda()
    expected = weighted.sequential_complete_spectra(covariance, weights)
    guards = []
    original = _fit_structure_cache.guard_cuda_workspace
    def guard(required, device, *, context):
        guards.append((required, context))
        return original(required, device, context=context)
    monkeypatch.setattr(_fit_structure_cache, 'guard_cuda_workspace', guard)
    with scopes.complete_spectrum_batch_scope(weight_batch_size=3):
        actual = weighted.sequential_complete_spectra(covariance, weights)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    nonzero = [0, 1, 4, 5]
    score = torch.ones(23, device=covariance.device)
    q = (score.square()[:, None]*weights.square()).sum(dim=0)[nonzero]
    actual_p = _saddlepoint_skat_pvalues(q, actual[nonzero])
    expected_p = _saddlepoint_skat_pvalues(q, expected[nonzero])
    assert torch.isfinite(actual_p).all() and (actual_p > 0).all()
    assert torch.max(torch.abs(torch.log10(actual_p)-torch.log10(expected_p))) < .001
    assert any(name == 'complete STAAR weight batch workspace' for _, name in guards)
    assert scopes.execution_metadata()['exact_relation_host_transfers'] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA owned FULL small-spectrum selector')
@pytest.mark.parametrize('m', [33, 42, 131, 512])
def test_cuda_explicit_full_selector_complete_spectra_and_probability_oracle(m):
    from fudan_wgs_toolkit.statistics import _saddlepoint_skat_pvalues
    from fudan_wgs_toolkit._precision_eigen import precision_eigen_execution_metadata
    covariance, weights = (value.cuda() for value in operands(m))
    before_covariance, before_weights = covariance.clone(), weights.clone()
    expected = weighted.sequential_complete_spectra(covariance, weights)
    precision_eigen_execution_metadata(reset=True)
    with weighted.eigensolver_context('cusolver_batched', memory_limit=20*2**30) as state:
        with scopes.complete_spectrum_batch_scope(maximum_variants=512, weight_batch_size=8):
            actual = weighted.sequential_complete_spectra(covariance, weights)
        report = weighted.execution_metadata()
    assert actual.shape == expected.shape == (6, m)
    assert torch.equal(covariance, before_covariance) and torch.equal(weights, before_weights)
    assert torch.count_nonzero(actual[2]) == torch.count_nonzero(actual[3]) == 0
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    rows = (0, 1, 4, 5)
    complete = expected[list(rows)]
    q = complete.sum(dim=1)+8*torch.sqrt(2*complete.square().sum(dim=1))
    reference = _saddlepoint_skat_pvalues(q, complete)
    candidate = _saddlepoint_skat_pvalues(q, actual[list(rows)])
    assert torch.isfinite(candidate).all() and (candidate > 0).all()
    assert torch.max(torch.abs(torch.log10(candidate)-torch.log10(reference))) < .001
    selected = m <= 128
    assert state['selector']['selected_calls'] == int(selected)
    if selected:
        assert state['selector']['backend']['n_above32_successful_calls'] == 1
        assert state['selector']['backend']['matrices'] == 4
    else:
        assert state['selector']['backend'] is None  # Complete Torch, no native Jacobi API.
    route = 'cusolverDnSsyevjBatched' if selected else 'torch.linalg.eigvalsh:complete_weight_batch'
    assert report['actual_backend_calls'] == {route: 1}
    assert report['actual_backend_matrices'] == {route: 4}
    assert report['full_selected_solver_calls'] == int(selected) and report['full_sequential_spectra'] == 6
    assert precision_eigen_execution_metadata()['gpu_eigen_calls'] == int(not selected)
