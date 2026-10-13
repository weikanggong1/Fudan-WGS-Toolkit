"""Failure-only complete-spectrum contracts; synthetic inputs are unit tests."""
import gc
import weakref

import pytest
import torch

from fudan_wgs_toolkit import _weighted_spectra as weighted, statistics


CONVERGENCE = 'linalg.eigh: The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues'


@pytest.fixture(autouse=True)
def reset_metadata():
    weighted.execution_metadata(reset=True)
    yield
    weighted.execution_metadata(reset=True)


def fail_fp32(monkeypatch, double_result=None):
    original = torch.linalg.eigvalsh
    calls = []
    def leaf(matrix, *, UPLO):
        calls.append((matrix.dtype, matrix.clone(), UPLO))
        if matrix.dtype == torch.float32:
            raise torch.linalg.LinAlgError(CONVERGENCE)
        return original(matrix, UPLO=UPLO) if double_result is None else double_result(matrix)
    monkeypatch.setattr(torch.linalg, 'eigvalsh', leaf)
    return original, calls


def test_retry_solves_the_same_h32_upper_triangle_and_keeps_complete_zero_modes(monkeypatch):
    # Repeated values and two exact zero modes exercise the reported failure
    # class without replacing a real-data accuracy or performance benchmark.
    covariance = torch.tensor([[1., 1., .25, 0.], [1., 1., .25, 0.],
                               [.25, .25, 2., 0.], [0., 0., 0., 0.]])
    weights = torch.tensor([[.7], [.7], [1.3], [0.]])
    expected_h32 = (covariance*weights[:, 0, None])*weights[None, :, 0]
    original, calls = fail_fp32(monkeypatch)
    expected = original(expected_h32.double(), UPLO='U')
    before_c, before_d = covariance.clone(), weights.clone()
    recorder = []
    monkeypatch.setattr(statistics, 'record_gpu_eigen_route', lambda matrix: recorder.append(matrix.dtype))
    actual = weighted.sequential_complete_spectra(covariance, weights)
    assert torch.equal(actual[0], expected)
    assert len(calls) == 2 and calls[0][0] == torch.float32 and calls[1][0] == torch.float64
    assert torch.equal(calls[1][1], calls[0][1].double())
    assert all(call[2] == 'U' for call in calls)
    assert recorder == [torch.float64]
    assert actual.shape == (1, 4) and actual.dtype == torch.float64
    assert torch.equal(covariance, before_c) and torch.equal(weights, before_d)
    metadata = weighted.execution_metadata()
    assert metadata['full_retry_attempts'] == metadata['full_retry_successes'] == 1
    assert metadata['full_retry_failures'] == 0
    assert metadata['actual_backend_calls'] == {'torch.linalg.eigvalsh:fp64_convergence_retry': 1}
    report = metadata['full_retry_reports'][0]
    assert report['status'] == 'retry_complete' and report['output_length'] == 4
    assert report['structural_zero_modes'] == 1 and report['numerical_zero_eigenvalues'] >= 2
    assert report['numerical_rank'] <= 2 and report['same_H32_upper_triangle']
    assert not report['regularization'] and not report['eigenvalues_truncated']


def test_normal_fp32_path_remains_bit_exact_and_never_uses_retry_budget(monkeypatch):
    covariance = torch.tensor([[3., .2, .1], [.2, 2., .3], [.1, .3, 1.]])
    weights = torch.tensor([[1., 2.], [2., 4.], [3., 6.]])
    original = torch.linalg.eigvalsh
    expected = torch.stack([original((covariance*w[:, None])*w[None, :], UPLO='U').double()
                            for w in weights.T])
    calls = []
    def leaf(matrix, **kwargs):
        calls.append(matrix.dtype)
        return original(matrix, **kwargs)
    monkeypatch.setattr(torch.linalg, 'eigvalsh', leaf)
    monkeypatch.setattr(weighted, '_retry_complete_spectrum', lambda *a, **k: pytest.fail('normal path retried'))
    actual = weighted.sequential_complete_spectra(covariance, weights, fallback_workspace_limit_bytes=0)
    assert torch.equal(actual, expected) and calls == [torch.float32]*2
    assert weighted.execution_metadata()['full_retry_attempts'] == 0


@pytest.mark.parametrize('value', [True, -1, 1.5, '4GiB'])
def test_retry_budget_must_be_explicit_nonnegative_integer_bytes(value, monkeypatch):
    monkeypatch.setattr(torch.linalg, 'eigvalsh', lambda *a, **k: pytest.fail('invalid budget solved'))
    with pytest.raises(ValueError, match='workspace limit'):
        weighted.sequential_complete_spectra(torch.eye(2), torch.ones(2, 1),
                                            fallback_workspace_limit_bytes=value)


def test_retry_budget_rejection_precedes_dense_fp64_copy_and_marks_incomplete(monkeypatch):
    _, calls = fail_fp32(monkeypatch)
    required = weighted.full_retry_workspace_unit_bytes(3)
    assert required == 32*3*3+64*3
    with pytest.raises(MemoryError, match='explicit retry budget') as failed:
        weighted.sequential_complete_spectra(torch.eye(3), torch.ones(3, 1),
                                            fallback_workspace_limit_bytes=required-1)
    assert [call[0] for call in calls] == [torch.float32]
    assert failed.value.weighted_spectrum_failure == 'retry_workspace_rejected'
    metadata = weighted.execution_metadata()
    assert metadata['full_retry_workspace_rejections'] == metadata['full_retry_failures'] == 1
    assert metadata['full_sequential_spectra'] == 0
    assert metadata['full_retry_reports'][0]['status'] == 'retry_workspace_rejected'
    assert metadata['actual_backend_calls'] == {}


def test_default_retry_budget_inherits_existing_process_policy(monkeypatch):
    from fudan_wgs_toolkit import tf32
    required = weighted.full_retry_workspace_unit_bytes(3)
    monkeypatch.setattr(tf32, '_memory_limit_bytes', required-1)
    _, calls = fail_fp32(monkeypatch)
    with pytest.raises(MemoryError, match='explicit retry budget'):
        weighted.sequential_complete_spectra(torch.eye(3), torch.ones(3, 1))
    assert [call[0] for call in calls] == [torch.float32]
    assert weighted.execution_metadata()['full_retry_reports'][0]['workspace_limit_bytes'] == required-1


def test_actual_retry_allocation_failure_stays_distinct_from_convergence(monkeypatch):
    dtypes = []
    def leaf(matrix, **kwargs):
        dtypes.append(matrix.dtype)
        if matrix.dtype == torch.float32:
            raise torch.linalg.LinAlgError(CONVERGENCE)
        raise torch.OutOfMemoryError('injected solver scratch allocation failure')
    monkeypatch.setattr(torch.linalg, 'eigvalsh', leaf)
    with pytest.raises(torch.OutOfMemoryError, match='scratch allocation failure') as failed:
        weighted.sequential_complete_spectra(torch.eye(3), torch.ones(3, 1))
    assert failed.value.weighted_spectrum_failure == 'retry_allocation_failed'
    assert dtypes == [torch.float32, torch.float64]
    metadata = weighted.execution_metadata()
    assert metadata['full_retry_allocation_failures'] == metadata['full_retry_failures'] == 1
    assert metadata['full_retry_workspace_rejections'] == metadata['full_retry_successes'] == 0
    assert metadata['full_retry_reports'][0]['status'] == 'retry_allocation_failed'


@pytest.mark.parametrize('target,value', [('covariance', float('nan')), ('covariance', float('inf')),
                                         ('weights', float('nan')), ('weights', float('-inf'))])
def test_nonfinite_inputs_are_rejected_before_any_solve_or_retry(target, value, monkeypatch):
    covariance, weights = torch.eye(3), torch.ones(3, 1)
    (covariance if target == 'covariance' else weights)[0, 0] = value
    monkeypatch.setattr(torch.linalg, 'eigvalsh', lambda *a, **k: pytest.fail('non-finite input solved'))
    with pytest.raises(ValueError, match='finite covariance') as failed:
        weighted.sequential_complete_spectra(covariance, weights)
    assert failed.value.weighted_spectrum_failure == 'nonfinite_input_rejected'
    metadata = weighted.execution_metadata()
    assert metadata['full_nonfinite_rejections'] == 1 and metadata['full_retry_attempts'] == 0


def test_finite_input_weighting_overflow_cannot_reach_fp64_retry(monkeypatch):
    _, calls = fail_fp32(monkeypatch)
    with pytest.raises(ValueError, match='H32 is non-finite') as failed:
        weighted.sequential_complete_spectra(torch.eye(2)*1e30, torch.ones(2, 1)*1e20)
    assert failed.value.weighted_spectrum_failure == 'nonfinite_weighted_matrix_rejected'
    assert len(calls) == 1 and weighted.execution_metadata()['full_retry_attempts'] == 0


@pytest.mark.parametrize('error', [RuntimeError('unrelated solver failure'),
                                  MemoryError('allocation failed'),
                                  torch.linalg.LinAlgError('illegal eigensolver argument')])
def test_nonconvergence_unrelated_errors_propagate_without_retry(error, monkeypatch):
    calls = []
    def leaf(matrix, **kwargs):
        calls.append(matrix.dtype)
        raise error
    monkeypatch.setattr(torch.linalg, 'eigvalsh', leaf)
    with pytest.raises(type(error), match=str(error)):
        weighted.sequential_complete_spectra(torch.eye(2), torch.ones(2, 1))
    assert calls == [torch.float32] and weighted.execution_metadata()['full_retry_attempts'] == 0


@pytest.mark.parametrize('values,message', [([1., 1.], 'incomplete'),
    ([1., float('nan'), 1.], 'non-finite'), ([3., 2., 1.], 'unordered'),
    ([-1., 1., 3.], 'non-PSD'), ([1., 1., 2.], 'trace check')])
def test_invalid_retry_spectra_remain_explicit_failures(values, message, monkeypatch):
    fail_fp32(monkeypatch, lambda matrix: torch.tensor(values, dtype=torch.float64, device=matrix.device))
    with pytest.raises(ArithmeticError, match=message) as failed:
        weighted.sequential_complete_spectra(torch.eye(3), torch.ones(3, 1))
    assert failed.value.weighted_spectrum_failure == 'retry_validation_failed'
    metadata = weighted.execution_metadata()
    assert metadata['full_retry_failures'] == 1 and metadata['full_retry_successes'] == 0
    assert metadata['full_retry_reports'][0]['status'] == 'retry_validation_failed'


def test_retry_cannot_replace_structural_zeros_with_a_full_rank_spectrum(monkeypatch):
    fail_fp32(monkeypatch, lambda matrix: matrix.new_tensor([.5, .5, 1.]))
    with pytest.raises(ArithmeticError, match='structural zero modes'):
        weighted.sequential_complete_spectra(torch.eye(3), torch.tensor([[0.], [1.], [1.]]))


def test_small_psd_rounding_negatives_are_retained_without_projection(monkeypatch):
    covariance = torch.diag(torch.tensor([-1e-8, 1., 1.]))
    original, _ = fail_fp32(monkeypatch)
    expected = original(covariance.double(), UPLO='U')
    actual = weighted.sequential_complete_spectra(covariance, torch.ones(3, 1))
    assert torch.equal(actual[0], expected) and actual[0, 0] < 0
    report = weighted.execution_metadata()['full_retry_reports'][0]
    assert report['negative_eigenvalues'] == 1 and report['status'] == 'retry_complete'


def test_second_solver_failure_terminates_without_a_third_route_and_releases_owners(monkeypatch):
    owners, dtypes = [], []
    def leaf(matrix, **kwargs):
        dtypes.append(matrix.dtype)
        held = matrix.clone()
        owners.extend([weakref.ref(matrix), weakref.ref(held)])
        raise torch.linalg.LinAlgError(CONVERGENCE)
    monkeypatch.setattr(torch.linalg, 'eigvalsh', leaf)
    def call():
        covariance, weights = torch.eye(3), torch.ones(3, 1)
        owners.extend([weakref.ref(covariance), weakref.ref(weights)])
        try:
            weighted.sequential_complete_spectra(covariance, weights)
        except torch.linalg.LinAlgError as error:
            return error
        finally:
            covariance = weights = None
    failure = call()
    gc.collect()
    assert failure.weighted_spectrum_failure == 'retry_solve_failed'
    assert dtypes == [torch.float32, torch.float64]
    assert failure.__traceback__ is not None and all(owner() is None for owner in owners)
    assert weighted.execution_metadata()['full_retry_failures'] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA live/process retry guard')
def test_cuda_retry_rechecks_full_h64_solver_allowance_before_allocation(monkeypatch):
    from fudan_wgs_toolkit import _fit_structure_cache
    original = torch.linalg.eigvalsh
    calls, guards = [], []
    def leaf(matrix, **kwargs):
        calls.append(matrix.dtype)
        if matrix.dtype == torch.float32:
            raise torch.linalg.LinAlgError(CONVERGENCE)
        return original(matrix, **kwargs)
    def guard(new_bytes, device, *, context):
        guards.append((new_bytes, context))
        if context == 'complete STAAR convergence retry':
            raise MemoryError('injected live retry workspace rejection')
    monkeypatch.setattr(torch.linalg, 'eigvalsh', leaf)
    monkeypatch.setattr(_fit_structure_cache, 'guard_cuda_workspace', guard)
    covariance, weights = torch.eye(3, device='cuda'), torch.ones(3, 1, device='cuda')
    allocated = torch.cuda.memory_allocated(covariance.device)
    with pytest.raises(MemoryError, match='live retry workspace') as failed:
        weighted.sequential_complete_spectra(covariance, weights)
    assert failed.value.weighted_spectrum_failure == 'retry_workspace_rejected'
    assert calls == [torch.float32]
    assert guards[-1] == (weighted.full_retry_workspace_unit_bytes(3), 'complete STAAR convergence retry')
    assert torch.cuda.memory_allocated(covariance.device) == allocated
