"""Synthetic complete-tail row oracles; real-data timing remains external."""
import gc
import weakref

import pytest
import torch

from fudan_wgs_toolkit import _native_tail_groups as grouped
from fudan_wgs_toolkit.statistics import _saddlepoint_skat_pvalues


@pytest.fixture(autouse=True)
def metadata():
    grouped.execution_metadata(reset=True)
    yield
    grouped.execution_metadata(reset=True)


def rows(m):
    eigenvalues = torch.linspace(.2, 1.7, m, dtype=torch.float64)
    spectra = torch.stack((eigenvalues, eigenvalues*.7, eigenvalues*1.3))
    # Exercise zero Q, a near-mean moment row and a positive Saddle row.
    q = torch.tensor([0., float(spectra[1].sum()),
        float(spectra[2].sum()+8*torch.sqrt(2*spectra[2].square().sum()))])
    return q, spectra


def test_independent_complete_rows_merge_without_changing_P_or_order():
    groups = [rows(7), rows(19), rows(7), rows(19), rows(7)]
    q, spectrum = groups[2]
    q = q.clone()
    q[0] = .003
    q[1] *= .07  # A different bracket width extends inactive rows' iterations.
    groups[2] = q, spectrum
    q, spectrum = groups[4]
    groups[4] = q*1.17, spectrum*1.31
    expected = [_saddlepoint_skat_pvalues(*item) for item in groups]
    actual = grouped.complete_tail_groups(groups)
    for result, control in zip(actual, expected):
        torch.testing.assert_close(result, control, atol=0., rtol=0.)
    report = grouped.execution_metadata()
    assert report['requested_groups'] == 5 and report['requested_rows'] == 15
    assert report['actual_tail_calls'] == report['merged_tail_calls'] == 2
    assert report['merged_groups'] == 5
    assert not torch.equal(actual[0], actual[2])


@pytest.mark.parametrize('kwargs', [{'maximum_workspace_bytes': 1}, {'maximum_rows': 3}])
def test_merge_limits_keep_all_original_complete_tail_groups(kwargs):
    groups = [rows(7), rows(7), rows(7)]
    expected = [_saddlepoint_skat_pvalues(*item) for item in groups]
    actual = grouped.complete_tail_groups(groups, **kwargs)
    assert all(torch.equal(result, control) for result, control in zip(actual, expected))
    report = grouped.execution_metadata()
    assert report['requested_groups'] == report['actual_tail_calls'] == 3
    assert report['merged_tail_calls'] == 0


@pytest.mark.parametrize('bad', ['negative_Q', 'nonfinite_Q', 'nonfinite_spectrum', 'zero_spectrum'])
def test_original_invalid_or_degenerate_tail_is_never_converted_to_success(bad):
    q, spectra = rows(7)
    q = q.clone()
    spectra = spectra.clone()
    if bad == 'negative_Q':
        q[-1] = -1.
    elif bad == 'nonfinite_Q':
        q[-1] = float('nan')
    elif bad == 'nonfinite_spectrum':
        spectra[-1, -1] = float('inf')
    else:
        spectra[-1].zero_()
    with pytest.raises((ValueError, ArithmeticError)):
        grouped.complete_tail_groups([rows(7), (q, spectra)])


def test_returned_probability_storage_does_not_hold_spectra_or_input_graph():
    refs = []
    def call():
        groups = [rows(7), rows(7)]
        for item in groups:
            for value in item:
                value.requires_grad_(True)
        refs.extend(weakref.ref(value) for item in groups for value in item)
        return grouped.complete_tail_groups(groups)
    outputs = call()
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert [value.untyped_storage().nbytes() for value in outputs] == [24, 24]
    assert all(not value.requires_grad and value.grad_fn is None for value in outputs)


def test_retained_failure_traceback_does_not_keep_complete_spectrum_owners():
    refs = []
    def call():
        q, spectrum = rows(7)
        q[-1] = -1.
        refs.extend((weakref.ref(q), weakref.ref(spectrum)))
        groups = [(q, spectrum), rows(7)]
        try:
            return grouped.complete_tail_groups(groups)
        finally:
            groups = q = spectrum = None
    with pytest.raises(ValueError) as captured:
        call()
    assert captured.value.__traceback__ is not None
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_invalid_shapes_or_low_precision_are_rejected_before_merge():
    q, spectra = rows(7)
    with pytest.raises(ValueError, match='dtypes'):
        grouped.complete_tail_groups([(q.half(), spectra)])
    with pytest.raises(ValueError, match='shapes'):
        grouped.complete_tail_groups([(q[:-1], spectra)])


def test_underflow_preserves_original_zero_P_without_probability_floor():
    q, spectrum = rows(19)
    q[-1] = 1e30
    expected = _saddlepoint_skat_pvalues(q, spectrum)
    actual = grouped.complete_tail_groups([(q, spectrum), rows(19)])[0]
    assert expected[-1] == actual[-1] == 0
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA cross-model complete tails')
@pytest.mark.parametrize('m', [23, 131, 1473, 9950])
def test_cuda_grouped_tail_precision_and_live_budget(m, monkeypatch):
    from fudan_wgs_toolkit import _fit_structure_cache
    groups = [(q.cuda(), spectra.cuda()) for q, spectra in (rows(m), rows(m))]
    q, spectrum = groups[1]
    q[-1] = spectrum[-1].sum()+30*torch.sqrt(2*spectrum[-1].square().sum())
    expected = [_saddlepoint_skat_pvalues(*item) for item in groups]
    actual = grouped.complete_tail_groups(groups)
    for result, control in zip(actual, expected):
        assert torch.isfinite(result).all() and (result > 0).all()
        assert torch.max(torch.abs(torch.log10(result)-torch.log10(control))) < .001
    def rejected(*args, **kwargs):
        raise MemoryError('fresh live merge workspace rejected')
    monkeypatch.setattr(_fit_structure_cache, 'guard_cuda_workspace', rejected)
    guarded = grouped.complete_tail_groups(groups)
    assert all(torch.equal(result, control) for result, control in zip(guarded, expected))
    assert grouped.execution_metadata()['live_budget_declines'] == 1
