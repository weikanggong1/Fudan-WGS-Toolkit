"""Shared score contracts; small fixtures are not performance benchmarks."""
from types import SimpleNamespace
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.phewas_runtime.scores import fp32_score_columns
from fudan_wgs_toolkit.phewas_runtime.runtime import _shared_gene_scores


def operands(device='cpu'):
    # FP32 residuals have cancellation and small values that are vulnerable to
    # a wide product selecting TF32 input rounding. Keep the native formula.
    n, m = 4099, 13
    rows = torch.arange(n, device=device, dtype=torch.float32)
    genotype = ((rows[:, None]*torch.arange(1, m+1, device=device)) % 7)/3
    residuals = [torch.sin(rows*.01371+i*.11317)*(1+i*.31)
                 + torch.where(rows.long()%2 == 0, .0001537, -.0001713)
                 for i in range(3)]
    return genotype, residuals


@pytest.mark.parametrize('transpose_layout', [False, True])
def test_original_fitted_vector_formula_and_inputs_preserved(transpose_layout):
    genotype, residuals = operands()
    if transpose_layout:
        genotype = genotype.T.contiguous().T
    before_g = genotype.clone()
    before_u = [x.clone() for x in residuals]
    expected = torch.stack([torch.mv(genotype.T, u) for u in residuals], dim=1)
    result = fp32_score_columns(genotype, residuals)
    assert torch.equal(result, expected)
    assert result.shape == (genotype.shape[1], len(residuals))
    assert torch.equal(genotype, before_g)
    assert all(torch.equal(x, y) for x, y in zip(residuals, before_u))


def test_output_panel_view_preserves_sentinels_and_reuses_storage():
    genotype, residuals = operands()
    owner = torch.full((19, 3), -999., dtype=torch.float32)
    out = owner[2:15]
    result = fp32_score_columns(genotype, residuals, out=out)
    assert result is out
    assert torch.equal(owner[2:15], torch.stack([torch.mv(genotype.T, u) for u in residuals], dim=1))
    assert torch.all(owner[:2] == -999.) and torch.all(owner[15:] == -999.)


@pytest.mark.parametrize('fault', ['g_dtype', 'g_rank', 'u_dtype', 'u_rank', 'u_rows',
                                   'out_dtype', 'out_shape', 'g_alias', 'u_alias'])
def test_invalid_inputs_fail_before_any_original_vector_is_overwritten(fault):
    genotype, residuals = operands()
    if fault == 'g_dtype': genotype = genotype.double()
    elif fault == 'g_rank': genotype = genotype[:, 0]
    elif fault == 'u_dtype': residuals[1] = residuals[1].double()
    elif fault == 'u_rank': residuals[1] = residuals[1][:, None]
    elif fault == 'u_rows': residuals[1] = residuals[1][:-1]
    out = None
    if fault == 'out_dtype': out = torch.empty((13, 3), dtype=torch.float64)
    elif fault == 'out_shape': out = torch.empty((13, 4), dtype=torch.float32)
    elif fault == 'g_alias': out = genotype.flatten()[:39].reshape(13, 3)
    elif fault == 'u_alias': out = residuals[0][:39].reshape(13, 3)
    before_g, before_u = genotype.clone(), [u.clone() for u in residuals]
    with pytest.raises(ValueError):
        fp32_score_columns(genotype, residuals, out=out)
    assert torch.equal(genotype, before_g)
    assert all(torch.equal(u, v) for u, v in zip(residuals, before_u))


def test_empty_trait_axis_and_no_variant_output_are_well_defined():
    genotype, residuals = operands()
    assert fp32_score_columns(genotype, []).shape == (13, 0)
    assert fp32_score_columns(genotype[:, :0], residuals).shape == (0, 3)


@pytest.mark.parametrize('panel_width', [1, 5, 13])
def test_host_mask_panels_preserve_full_variant_order_and_original_vectors(panel_width):
    genotype, residuals = operands()
    # Use a non-contiguous host matrix to exercise the actual panel extraction.
    host = np.asfortranarray(genotype.numpy())
    models = [SimpleNamespace(device='cpu', scaled_residuals=u) for u in residuals]
    guards = []
    broker = SimpleNamespace(_guard=guards.append)
    result = _shared_gene_scores(host, models, broker, variant_tile_size=panel_width)
    expected = torch.empty_like(result)
    for begin in range(0, 13, panel_width):
        panel = torch.as_tensor(host[:, begin:begin+panel_width], dtype=torch.float32)
        expected[begin:begin+panel_width] = torch.stack([torch.mv(panel.T, u) for u in residuals], dim=1)
    assert torch.equal(result, expected)
    assert len(guards) == 1
    # Admission does not include the N x traits stack that is no longer allocated.
    assert guards[0] == 4*(4099*min(13, panel_width)+13*3+min(13, panel_width)*3)+64*2**20


@pytest.mark.skipif(not torch.cuda.is_available(), reason='original CUDA FP32 vector route requires CUDA')
def test_cuda_wide_shared_scores_use_ieee_fp32_while_native_covariance_policy_is_restored():
    from fudan_wgs_toolkit import tf32
    genotype, residuals = operands('cuda:0')
    from fudan_wgs_toolkit._burden import _LOCK, _ieee_flags
    from fudan_wgs_toolkit.pipeline import _individual_log_probabilities
    original = torch.stack([tf32.matmul(genotype.T, u, mode='tf32') for u in residuals], dim=1)
    # The CUDA matrix route has its own FP32 reduction order. Verify that
    # exact route, then independently check its effect on ordinary tails.
    with _LOCK, _ieee_flags():
        expected = torch.mm(genotype.T, torch.stack(residuals, dim=1))
    tf32.execution_metadata(reset=True)
    actual = fp32_score_columns(genotype, residuals)
    assert torch.equal(actual, expected)
    variance = genotype.square().sum(dim=0)[:, None].expand_as(actual)
    difference = (_individual_log_probabilities(actual, variance)
                  - _individual_log_probabilities(original, variance)).abs()/np.log(10)
    assert float(difference.max()) < .001
    from fudan_wgs_toolkit.phewas_runtime.scores import execution_metadata
    assert execution_metadata()['fp32_simt_products'] >= 1
    stats = tf32.execution_metadata()
    assert stats['fp32_vector_product_count'] == 0
    assert stats['tf32_gemm_call_count'] == 0
    assert stats['logical_product_count'] == 0


def test_singleton_retains_the_exact_mature_vector_formula():
    genotype, residuals = operands()
    actual = fp32_score_columns(genotype, residuals[:1])
    assert torch.equal(actual[:, 0], torch.mv(genotype.T, residuals[0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA optional stack admission contract')
def test_optional_matrix_admission_failure_uses_no_stacked_workspace(monkeypatch):
    from fudan_wgs_toolkit.phewas_runtime import scores
    genotype, residuals = operands('cuda:0')
    from fudan_wgs_toolkit.tf32 import matmul
    expected = torch.stack([matmul(genotype.T, u, mode='tf32') for u in residuals], dim=1)
    monkeypatch.setattr(scores, '_simt_admission', lambda *args: False)
    def forbidden(*args, **kwargs):raise AssertionError('denied optional stack was allocated')
    monkeypatch.setattr(torch, 'stack', forbidden)
    scores.execution_metadata(reset=True)
    actual = fp32_score_columns(genotype, residuals)
    assert torch.equal(actual, expected)
    stats = scores.execution_metadata()
    assert stats['simt_admission_fallbacks'] == 1
    assert stats['fp32_vector_products'] == 3
    assert stats['fp32_simt_products'] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA IEEE matrix exception contract')
def test_ieee_score_failure_restores_original_flags_and_preserves_inputs_out(monkeypatch):
    genotype, residuals = operands('cuda:0')
    out = torch.full((13, 3), -917., dtype=torch.float32, device='cuda:0')
    before_g, before_u = genotype.clone(), [u.clone() for u in residuals]
    original_precision = torch.get_float32_matmul_precision()
    original_flag = torch.backends.cuda.matmul.allow_tf32
    def fail(*args, **kwargs):
        assert torch.get_float32_matmul_precision() == 'highest'
        assert torch.backends.cuda.matmul.allow_tf32 is False
        raise RuntimeError('injected score product failure')
    monkeypatch.setattr(torch, 'mm', fail)
    with pytest.raises(RuntimeError, match='injected score'):
        fp32_score_columns(genotype, residuals, out=out)
    assert torch.get_float32_matmul_precision() == original_precision
    assert torch.backends.cuda.matmul.allow_tf32 == original_flag
    assert torch.all(out == -917.)
    assert torch.equal(genotype, before_g)
    assert all(torch.equal(u, v) for u, v in zip(residuals, before_u))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native IEEE policy override check requires CUDA')
def test_cuda_forced_cublas_tf32_override_is_rejected_before_score_publication(monkeypatch):
    genotype, residuals = operands('cuda:0')
    out = torch.full((13, 3), -917., device='cuda:0')
    monkeypatch.setenv('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', '1')
    with pytest.raises(ValueError, match='unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'):
        fp32_score_columns(genotype, residuals, out=out)
    assert torch.all(out == -917.)


@pytest.mark.parametrize('final_tail', [False, True])
def test_sensitive_gene_score_restores_complete_original_trait_column_before_export(final_tail):
    from fudan_wgs_toolkit.phewas_runtime.runtime import _evaluate_gaussian_gene_score
    genotype, residuals = operands()
    model = SimpleNamespace(device='cpu', scaled_residuals=residuals[0])
    original = torch.mv(genotype.T, residuals[0])
    candidate = original+.017
    covariance = torch.eye(len(original))*1000.
    if not final_tail:
        # Deliberately sensitive fitted diagonal, with all normal tails extreme.
        covariance = torch.diag(candidate.square()/(400*np.log(10)))
    immutable = covariance.clone()
    calls, guards, report = [], [], {}
    def evaluate(payload, received):
        assert received is model and payload['covariance'] is covariance
        calls.append(payload['score'].detach().clone())
        if final_tail and len(calls) == 1:
            return {'WGS-O': 0., 'num_variant': len(original), 'cMAC': 0.}
        assert torch.equal(payload['score'], original)
        return {'WGS-O': .25, 'num_variant': len(original), 'cMAC': 0.}
    payload = dict(score=candidate, covariance=covariance)
    result = _evaluate_gaussian_gene_score(SimpleNamespace(_evaluate_prepared=evaluate, profiler=SimpleNamespace(measure=lambda *a, **k: nullcontext())),
        payload, model, genotype, SimpleNamespace(_guard=guards.append), shared_simt=True, report=report)
    assert result['WGS-O'] == .25
    assert torch.equal(covariance, immutable)
    assert torch.equal(calls[-1], original) and len(guards) == 1
    assert report['gene_score_original_fallback_columns'] == 1
    assert report['gene_score_tail_evaluations'] == (2 if final_tail else 1)
    reason = 'gene_tail' if final_tail else 'normal_tail'
    assert report['gene_score_original_fallback_'+reason] == 1
    assert report.get('gene_score_additional_tail_evaluations', 0) == int(final_tail)


def test_mature_gene_vector_column_is_not_recomputed_or_filtered_by_sensitive_gate():
    from fudan_wgs_toolkit.phewas_runtime.runtime import _evaluate_gaussian_gene_score
    genotype, residuals = operands()
    original = torch.mv(genotype.T, residuals[0])
    payload = dict(score=original, covariance=torch.eye(len(original))*1e-10)
    calls, report = [], {}
    def evaluate(received, model):
        assert received is payload
        calls.append(True)
        return {'WGS-O': 0.}
    def denied(*args):raise AssertionError('existing mature vector was recomputed')
    result = _evaluate_gaussian_gene_score(SimpleNamespace(_evaluate_prepared=evaluate, profiler=SimpleNamespace(measure=lambda *a, **k: nullcontext())),
        payload, SimpleNamespace(device='cpu', scaled_residuals=residuals[0]), genotype,
        SimpleNamespace(_guard=denied), shared_simt=False, report=report)
    assert result == {'WGS-O': 0.} and calls == [True]
    assert not report.get('gene_score_original_fallback_columns')


@pytest.mark.parametrize('value,expected', [(0., True), (1e-101, True), (1e-100, False),
    (.25, False), (float('nan'), False), (-.01, False), (None, False)])
def test_native_gene_tail_zero_is_a_fallback_signal_without_probability_floor(value, expected):
    from fudan_wgs_toolkit.phewas_runtime.runtime import _gene_probability_extreme
    assert _gene_probability_extreme({'WGS-O': value, 'cMAC': 0., 'num_variant': 0}) is expected


@pytest.mark.parametrize('name,value,expected', [('WGS-O_log', 101., False),
    ('WGS-O_log', 101*np.log(10), True), ('WGS-O_log10', 101., True)])
def test_sensitive_gene_native_log_suffix_uses_declared_natural_or_base10_units(name, value, expected):
    from fudan_wgs_toolkit.phewas_runtime.runtime import _gene_probability_extreme
    assert _gene_probability_extreme({name: value}) is expected
