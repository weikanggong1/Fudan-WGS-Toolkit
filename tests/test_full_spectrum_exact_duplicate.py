"""Call-local bit-exact duplicate reuse; synthetic inputs are unit tests."""
import gc
import weakref

import pytest
import torch

from fudan_wgs_toolkit import _weighted_spectra as weighted


@pytest.fixture(autouse=True)
def reset_metadata():
    weighted.execution_metadata(reset=True)
    yield
    weighted.execution_metadata(reset=True)


def test_duplicate_and_zero_weights_match_independent_full_h32_bit_exactly(monkeypatch):
    covariance = torch.tensor([[3., .2, .1, .5], [.2, 2., .3, .1],
                               [.1, .3, 1., .25], [.5, .1, .25, 2.]])
    vector = torch.tensor([.1, 1.7, 3.25, .9])
    near = vector.clone()
    near[-1] = torch.nextafter(near[-1], torch.tensor(float('inf')))
    near_scaled = 2*vector
    near_scaled[-1] = torch.nextafter(near_scaled[-1], torch.tensor(float('inf')))
    weights = torch.stack([vector, vector, torch.zeros(4), torch.zeros(4),
                           2*vector, 2*vector, near, near_scaled], dim=1)
    original = torch.linalg.eigvalsh
    matrices = [(covariance*w[:, None])*w[None, :] for w in weights.T]
    expected = torch.stack([original(matrix, UPLO='U').double() for matrix in matrices])
    calls = []
    def leaf(matrix, **kwargs):
        calls.append((matrix.clone(), kwargs))
        return original(matrix, **kwargs)
    monkeypatch.setattr(torch.linalg, 'eigvalsh', leaf)
    actual = weighted.sequential_complete_spectra(covariance, weights)
    assert torch.equal(actual, expected) and actual.shape == (8, 4)
    expected_indices = [0, 2, 4, 6, 7]
    assert len(calls) == len(expected_indices)
    for (matrix, options), index in zip(calls, expected_indices):
        assert matrix.dtype == torch.float32 and torch.equal(matrix, matrices[index])
        assert options == {'UPLO': 'U'}
    assert torch.equal(actual[2], torch.zeros(4, dtype=torch.float64))
    metadata = weighted.execution_metadata()
    assert metadata['full_requested_weights'] == metadata['full_sequential_spectra'] == 8
    assert metadata['full_fp32_solver_calls'] == metadata['eigen_api_calls'] == 5
    assert metadata['full_exact_duplicate_reused_spectra'] == 3
    assert metadata['complete_representative_spectra'] == 5
    assert metadata['actual_backend_calls'] == {'torch.linalg.eigvalsh': 5}
    assert metadata['full_retry_attempts'] == 0


def test_signed_zero_bit_differences_do_not_reuse(monkeypatch):
    weights = torch.tensor([[0., -0., 0.], [1., 1., 1.], [2., 2., 2.]])
    original = torch.linalg.eigvalsh
    calls = []
    def leaf(matrix, **kwargs):
        calls.append(matrix.clone())
        return original(matrix, **kwargs)
    monkeypatch.setattr(torch.linalg, 'eigvalsh', leaf)
    actual = weighted.sequential_complete_spectra(torch.eye(3), weights)
    assert len(calls) == 2
    assert weighted.execution_metadata()['full_exact_duplicate_reused_spectra'] == 1
    assert torch.equal(actual[0], actual[1]) and torch.equal(actual[0], actual[2])


def test_noncontiguous_weight_storage_supports_bit_view_without_copy(monkeypatch):
    storage = torch.tensor([[1., 7., 1., 7.], [2., 9., 2., 9.], [3., 5., 3., 5.]])
    weights = storage[:, ::2]
    assert not weights.is_contiguous()
    monkeypatch.setattr(torch.Tensor, 'contiguous', lambda *a, **k: pytest.fail('duplicate detection copied weights'))
    actual = weighted.sequential_complete_spectra(torch.eye(3), weights)
    assert torch.equal(actual, torch.tensor([[1., 4., 9.], [1., 4., 9.]], dtype=torch.float64))
    assert weighted.execution_metadata()['full_fp32_solver_calls'] == 1


def test_different_covariance_calls_never_share_a_spectrum(monkeypatch):
    original = torch.linalg.eigvalsh
    calls = []
    def leaf(matrix, **kwargs):
        calls.append(matrix.clone())
        return original(matrix, **kwargs)
    monkeypatch.setattr(torch.linalg, 'eigvalsh', leaf)
    weights = torch.ones(3, 3)
    first = weighted.sequential_complete_spectra(torch.eye(3), weights)
    second = weighted.sequential_complete_spectra(2*torch.eye(3), weights)
    assert len(calls) == 2
    assert torch.equal(first, torch.ones(3, 3, dtype=torch.float64))
    assert torch.equal(second, 2*first)
    metadata = weighted.execution_metadata()
    assert metadata['full_requested_weights'] == metadata['full_sequential_spectra'] == 6
    assert metadata['full_fp32_solver_calls'] == 2
    assert metadata['full_exact_duplicate_reused_spectra'] == 4


def test_result_and_metadata_do_not_retain_duplicate_input_or_autograd_owners():
    def call():
        covariance = torch.eye(3, requires_grad=True)
        weights = torch.ones(3, 3, requires_grad=True)
        owners = [weakref.ref(covariance), weakref.ref(weights)]
        return weighted.sequential_complete_spectra(covariance, weights), owners
    result, owners = call()
    gc.collect()
    assert result.grad_fn is None and not result.requires_grad
    assert all(owner() is None for owner in owners)
    assert weighted.execution_metadata()['full_exact_duplicate_reused_spectra'] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA distinct-weight admission budget')
def test_cuda_duplicate_reuse_requires_no_new_matrix_workspace(monkeypatch):
    from fudan_wgs_toolkit import _fit_structure_cache
    covariance, weights = torch.eye(3, device='cuda'), torch.ones(3, 3, device='cuda')
    calls = []
    def guard(required, device, *, context):
        calls.append((required, context))
    monkeypatch.setattr(_fit_structure_cache, 'guard_cuda_workspace', guard)
    actual = weighted.sequential_complete_spectra(covariance, weights)
    unit = weighted.full_workspace_unit_bytes(3)
    assert calls == [(8*3*3+unit, 'complete STAAR output/workspace'),
                     (unit, 'complete STAAR weight workspace')]
    assert torch.equal(actual, torch.ones(3, 3, dtype=torch.float64, device=covariance.device))
    assert weighted.execution_metadata()['full_sequential_spectra'] == 3
