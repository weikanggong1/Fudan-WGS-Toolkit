"""Synthetic exact FULL preparation/finalization contracts, not benchmarks."""
from contextlib import nullcontext
from types import SimpleNamespace
import gc
import weakref

import pytest
import torch

from fudan_wgs_toolkit import _native_tail_groups as grouped, statistics


@pytest.fixture(autouse=True)
def cpu_native_oracle(monkeypatch):
    from fudan_wgs_toolkit import _burden
    # CPU isolates the spectrum/tail/output contract; real native CUDA burden
    # precision and end-to-end gains must be assessed independently.
    monkeypatch.setattr(_burden, 'ieee_burden_product', lambda left, right: left @ right)
    grouped.execution_metadata(reset=True)
    yield
    grouped.execution_metadata(reset=True)


def inputs(m=7, scale=1., rare='mixed'):
    generator = torch.Generator().manual_seed(92+m)
    matrix = torch.randn(m, m, generator=generator)
    covariance = matrix @ matrix.T+torch.eye(m)
    mac = torch.arange(1., m+1)*4
    if rare == 'all':
        mac.fill_(1.)
    elif rare == 'none':
        mac.fill_(21.)
    return dict(score=torch.linspace(-.7, 1.3, m)*scale,
        covariance=covariance, maf=torch.linspace(.002, .008, m), mac=mac,
        annotations=torch.stack((torch.linspace(0., 37., m),
                                 torch.linspace(23., 1., m)), dim=1),
        names=['functional_a', 'functional_b'], cmac=37.5,
        rv_num_cutoff=1, matmul_mode='tf32')


@pytest.mark.parametrize('rare', ['mixed', 'all', 'none'])
@pytest.mark.parametrize('batch_copy', [True, False])
def test_pending_groups_preserve_full_output_fields_order_and_values(rare, batch_copy, monkeypatch):
    options = [inputs(7, .2, rare), inputs(7, 11., rare), inputs(11, 1.3, rare)]
    for option in options:
        option.update(output_batch_optimization=batch_copy, tail_optimization=True)
    expected = [statistics.association_test(**option) for option in options]
    calls = []
    original = statistics._saddlepoint_skat_pvalues
    def tail(q, spectrum):
        calls.append(tuple(spectrum.shape))
        return original(q, spectrum)
    monkeypatch.setattr(statistics, '_saddlepoint_skat_pvalues', tail)
    pendings = [statistics.association_test(**option, _defer_full_tail=True)
                for option in options]
    assert calls == [] and all(isinstance(p, grouped.PendingCompleteTail) for p in pendings)
    assert all(torch.isnan(p.pvalues[0]).all() for p in pendings)
    assert [p.complete_spectra.shape for p in pendings] == [(6, 7), (6, 7), (6, 11)]
    actual = grouped.finish_complete_tails(pendings)
    for control, result in zip(expected, actual):
        assert list(result) == list(control)
        assert result == control and result['cMAC'] == 37.5
    assert calls == [(12, 7), (6, 11)]
    assert all(p._closed and p.retained_bytes == 0 for p in pendings)
    report = grouped.execution_metadata()
    assert report['pending_preparations'] == report['pending_finalizations'] == report['pending_closes'] == 3


@pytest.mark.parametrize('kwargs', [{'maximum_workspace_bytes': 1}, {'maximum_rows': 6}])
def test_explicit_merge_limits_retain_individual_complete_outputs(kwargs):
    options = [inputs(7, .2), inputs(7, 11.)]
    expected = [statistics.association_test(**option) for option in options]
    pendings = [statistics.association_test(**option, _defer_full_tail=True) for option in options]
    assert grouped.finish_complete_tails(pendings, **kwargs) == expected
    assert grouped.execution_metadata()['actual_tail_calls'] == 2  # Only finalizer calls belong to this counter.
    assert grouped.execution_metadata()['merged_tail_calls'] == 0


@pytest.mark.parametrize('kwargs', [dict(matmul_mode='fp64'), dict(long_mask_method='fastskat'),
    dict(long_mask_method='liu'), dict(_skat_pvalues=torch.ones(2)), dict(_defer_full_tail=1)])
def test_defer_requires_native_FULL_and_cannot_accept_approximation_or_precomputed_P(kwargs):
    kwargs = dict(dict(matmul_mode='tf32', _defer_full_tail=True), **kwargs)
    with pytest.raises(ValueError, match='deferred tails|_defer_full_tail'):
        statistics.association_test(None, None, None, None, **kwargs)


def test_pending_does_not_retain_original_matrices_scores_or_autograd_and_closes_returned_owners():
    original_refs, pending_refs = [], []
    def prepare():
        options = inputs()
        for name in ('score', 'covariance', 'maf', 'mac', 'annotations'):
            original_refs.append(weakref.ref(options[name]))
            options[name].requires_grad_(True)
        pending = statistics.association_test(**options, _defer_full_tail=True)
        pending_refs.extend(weakref.ref(value) for value in
            (pending.q_values, pending.complete_spectra, pending.pvalues))
        assert all(value.grad_fn is None and not value.requires_grad for value in
                   (pending.q_values, pending.complete_spectra, pending.pvalues))
        assert not hasattr(pending, '__dict__')
        return pending
    pending = prepare()
    gc.collect()
    assert all(ref() is None for ref in original_refs)
    expected_bytes = 6*(4+8*7+3*8)
    assert pending.retained_bytes == expected_bytes
    result = grouped.finish_complete_tails([pending])[0]
    assert all(isinstance(value, (float, int)) for value in result.values())
    gc.collect()
    assert pending.retained_bytes == 0 and all(ref() is None for ref in pending_refs)


def test_finalizer_failure_closes_all_pending_owners_with_retained_traceback():
    first = statistics.association_test(**inputs(), _defer_full_tail=True)
    second = statistics.association_test(**inputs(), _defer_full_tail=True)
    first.q_values[-1] = -1.
    refs = [weakref.ref(value) for item in (first, second)
            for value in (item.q_values, item.complete_spectra, item.pvalues)]
    with pytest.raises(ValueError) as captured:
        grouped.finish_complete_tails([first, second])
    assert captured.value.__traceback__ is not None
    gc.collect()
    assert first.retained_bytes == second.retained_bytes == 0
    assert all(ref() is None for ref in refs)


def test_duplicate_or_closed_pending_cannot_double_finalize():
    pending = statistics.association_test(**inputs(), _defer_full_tail=True)
    with pytest.raises(ValueError, match='distinct open'):
        grouped.finish_complete_tails([pending, pending])
    assert pending.retained_bytes == 0
    with pytest.raises(ValueError, match='distinct open'):
        grouped.finish_complete_tails([pending])


def test_accidentally_shortened_pending_spectrum_is_rejected_before_tail_dispatch(monkeypatch):
    pending = statistics.association_test(**inputs(), _defer_full_tail=True)
    pending.complete_spectra = pending.complete_spectra[:, :-1]
    def forbidden(*args, **kwargs):
        raise AssertionError('truncated pending spectrum reached a tail')
    monkeypatch.setattr(statistics, '_saddlepoint_skat_pvalues', forbidden)
    with pytest.raises(ValueError, match='complete Q/spectrum'):
        grouped.finish_complete_tails([pending])
    assert pending.retained_bytes == 0


@pytest.mark.parametrize('normal_extreme,gene_extreme', [(False, False), (True, False), (False, True)])
def test_completed_first_evaluation_preserves_both_mature_score_fallbacks(monkeypatch, normal_extreme, gene_extreme):
    from fudan_wgs_toolkit.phewas_runtime import runtime
    score = torch.tensor([1., 2.])
    original = torch.tensor([1.1, 2.2])
    payload = dict(score=score, covariance=torch.eye(2))
    first = {'WGS-O': 0. if gene_extreme else .3, 'num_variant': 2, 'cMAC': 4.}
    calls, checks, report = [], [], {}
    def normal(current, covariance):
        checks.append(current is score)
        return normal_extreme
    def evaluate(current, model):
        assert torch.equal(current['score'], original)
        calls.append(True)
        return dict(first, **{'WGS-O': .7})
    monkeypatch.setattr(runtime, '_gene_normal_score_extreme', normal)
    monkeypatch.setattr(runtime, '_shared_gene_scores', lambda *args: original[:, None])
    pipeline = SimpleNamespace(_evaluate_prepared=evaluate,
        profiler=SimpleNamespace(measure=lambda *args, **kwargs: nullcontext()))
    result = runtime._evaluate_gaussian_gene_score(pipeline, payload, SimpleNamespace(),
        torch.empty(0), SimpleNamespace(), shared_simt=True, report=report,
        first_evaluation=first)
    assert checks == [True]
    assert len(calls) == int(normal_extreme or gene_extreme)
    assert result['WGS-O'] == (.7 if calls else .3)
    assert report['gene_score_tail_evaluations'] == 1+int(gene_extreme)
    assert report.get('gene_score_completed_first_evaluations', 0) == int(not normal_extreme)
    assert report.get('gene_score_completed_first_evaluations_discarded', 0) == int(normal_extreme)
    assert report.get('gene_score_additional_tail_evaluations', 0) == int(gene_extreme)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA native FULL pending finalization')
def test_cuda_pending_complete_fields_keep_strict_logP_and_no_missing_rows(monkeypatch):
    monkeypatch.undo()  # Use the real CUDA native burden route.
    options = [inputs(23, .7), inputs(23, 3.1)]
    for option in options:
        for name in ('score', 'covariance', 'maf', 'mac', 'annotations'):
            option[name] = option[name].cuda()
    expected = [statistics.association_test(**option) for option in options]
    actual = grouped.finish_complete_tails([
        statistics.association_test(**option, _defer_full_tail=True) for option in options])
    for control, result in zip(expected, actual):
        assert list(control) == list(result)
        for name in control:
            if name not in ('num_variant', 'cMAC'):
                assert 0 < result[name] <= 1 and 0 < control[name] <= 1
                assert abs(torch.log10(torch.tensor(result[name], dtype=torch.float64))
                    -torch.log10(torch.tensor(control[name], dtype=torch.float64))) < .001
