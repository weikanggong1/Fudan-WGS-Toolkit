"""Pointwise-tail and transport contracts; small fixtures are not benchmarks."""
from types import SimpleNamespace
from numbers import Real

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.phewas_runtime import single
from fudan_wgs_toolkit.pipeline import _individual_log_probabilities
from test_phewas_batched_runtime import cpu_product_oracle, model_for
from test_phewas_single_batches import make_pipeline, minor_block, states_fixture, assert_tables_equal


def prepared_tile(families):
    rows = np.arange(256)[::-1].copy()
    block = minor_block(states_fixture(), rows, np.arange(10))
    models = [model_for(rows, .8+i*.3) for i in range(len(families))]
    for model, family in zip(models, families):
        model.family = family
        if family == 'binomial':
            model.precision = model.inverse_variance
    pipelines = [make_pipeline(rows, [], model=model) for model in models]
    state, ordinal = pipelines[0]._prepare_individual_block(block, 0, mac_cutoff=1)
    prepared = pipelines[0]._prepare_individual_trait(state, 0, mac_cutoff=1)
    prepared.pop('model')
    return models, pipelines, prepared, ordinal


def test_shared_tail_keeps_extreme_finite_logs_nan_and_nonpositive_variance(
        monkeypatch, cpu_product_oracle):
    models, pipelines, prepared, ordinal = prepared_tile(['gaussian']*3)
    genotype = prepared['genotype']
    # Isolate the pointwise tail contract from FP32 score reduction rounding.
    # Score arithmetic has its separate independent-core test below: here an
    # artificially tiny variance creates very large finite native log(P).
    from fudan_wgs_toolkit.phewas_runtime import scores as score_module
    monkeypatch.setattr(score_module, '_simt_admission', lambda *args: False)
    original_score_columns = single.fp32_score_columns
    observed = {}
    def captured_scores(g, residuals):
        assert g is genotype
        assert all(left is right.scaled_residuals for left, right in zip(residuals, models))
        result = original_score_columns(g, residuals)
        observed['scores'] = result.detach().clone()
        return result
    monkeypatch.setattr(single, 'fp32_score_columns', captured_scores)
    patterns = torch.tensor([1e-8, 1., 0., -1., float('nan'), float('inf')], dtype=torch.float32)
    variances = [patterns[torch.arange(genotype.shape[1]) % len(patterns)].clone() for _ in models]
    model_positions = {id(model): index for index, model in enumerate(models)}
    monkeypatch.setattr(single, '_variance_from_shared_genotype',
                        lambda model, *unused: variances[model_positions[id(model)]])
    calls = []
    def tail(score, variance):
        calls.append(tuple(score.shape))
        return _individual_log_probabilities(score, variance)
    monkeypatch.setattr(single, '_individual_log_probabilities', tail)
    single.execution_metadata(reset=True)
    actual, actual_ordinal = single.process_single_shared_genotype(pipelines, None, ordinal, '21',
        mac_cutoff=1, subset_variants_num=3, prepared_state=(prepared, ordinal))
    scores = observed['scores']
    expected = []
    for index, pipeline in enumerate(pipelines):
        variance = variances[index]
        values = torch.stack((variance, scores[:, index], variance.double().sqrt(),
                              _individual_log_probabilities(scores[:, index], variance)), dim=1)
        expected.append(pipeline._individual_records_from_values(
            dict(prepared, model=models[index]), values.numpy(), subset_variants_num=3))
    assert actual_ordinal == ordinal and calls == [(len(models), scores.shape[0])]
    for left, right in zip(actual, expected):
        # Compare every native field and ordering marker, including finite
        # score-derived log(P) when the ordinary probability underflows.
        assert_tables_equal(left, right)
    assert any(row['pvalue'] == 0 and np.isfinite(row['pvalue_log10'])
               and row['pvalue_log10'] > 300 for rows in actual for row in rows)
    metrics = single.execution_metadata()
    assert metrics['pointwise_tail_batches'] == metrics['result_transfer_batches'] == 1
    assert metrics['result_transfer_values'] == 4*scores.numel()
    assert metrics['phenotype_score_gemm_calls'] == 0
    assert metrics['phenotype_score_batches'] == 1
    assert (metrics['phenotype_score_fp32_vector_calls']
            + metrics['phenotype_score_fp32_simt_gemm_calls']) in (1, len(models))


def binary_correction(index, model, genotype, score, variance):
    n = len(score)
    selected = torch.arange(n) % 2 == 0
    failed = torch.arange(n) % 3 == 0
    values = torch.full((n,), .125, dtype=torch.float64)
    values[0] = 0
    log10 = -torch.log10(values)
    return dict(normal_only=False, pvalues=values, pvalue_log10=log10,
        pvalue_log=log10*np.log(10), normal_pvalues=torch.full_like(values, .25),
        normal_pvalue_log10=torch.full_like(values, -np.log10(.25)),
        spa_selected=selected, spa_zero_log_unverified=values == 0,
        spa=SimpleNamespace(used_bisection=~selected, failed=failed,
                            iterations=torch.arange(n, dtype=torch.int64), iteration_limit=~failed))


def test_mixed_family_packing_keeps_all_spa_fields_without_continuous_padding(cpu_product_oracle):
    models, pipelines, prepared, ordinal = prepared_tile(
        ['gaussian', 'binomial', 'gaussian', 'binomial'])
    called = []
    def correction(*args):
        called.append(args[0])
        return binary_correction(*args)
    single.execution_metadata(reset=True)
    actual, actual_ordinal = single.process_single_shared_genotype(pipelines, None, ordinal, '21',
        mac_cutoff=1, prepared_state=(prepared, ordinal), binary_correction=correction)
    n = len(actual[0])
    assert actual_ordinal == ordinal and called == [1, 3] and n > 0
    for index, rows in enumerate(actual):
        assert len(rows) == n
        if index in called:
            expected = binary_correction(index, models[index], prepared['genotype'],
                                         torch.empty(n), torch.empty(n))
            for j, row in enumerate(rows):
                assert row['pvalue'] == (0. if j == 0 else .125)
                assert row['normal_pvalue'] == .25
                assert row['normal_pvalue_log10'] == -np.log10(.25)
                assert row['spa_selected'] == (j % 2 == 0)
                assert row['spa_failed'] == (j % 3 == 0)
                assert row['spa_used_bisection'] == (j % 2 != 0)
                assert row['spa_iterations'] == j
                assert row['spa_iteration_limit'] == (j % 3 != 0)
                assert row['spa_zero_log_unverified'] == (j == 0)
                assert row['pvalue_log10'] == float(expected['pvalue_log10'][j])
                assert row['pvalue_log'] == float(expected['pvalue_log'][j])
        else:
            assert all('normal_pvalue' not in row and 'spa_selected' not in row for row in rows)
    metrics = single.execution_metadata()
    assert metrics['result_transfer_values'] == n*(4*len(models)+11*len(called))
    assert metrics['result_transfer_batches'] == 1


@pytest.mark.parametrize('failure', ['normal_only', 'shape'])
def test_invalid_binary_callback_still_fails_before_result_transfer(cpu_product_oracle, failure):
    _, pipelines, prepared, ordinal = prepared_tile(['gaussian', 'binomial'])
    def correction(*args):
        value = binary_correction(*args)
        if failure == 'normal_only':
            value['normal_only'] = True
        else:
            value['pvalue_log10'] = value['pvalue_log10'][:-1]
        return value
    single.execution_metadata(reset=True)
    with pytest.raises(RuntimeError, match='formal binary Single|original columns/device'):
        single.process_single_shared_genotype(pipelines, None, ordinal, '21',
            mac_cutoff=1, prepared_state=(prepared, ordinal), binary_correction=correction)
    assert single.execution_metadata()['result_transfer_batches'] == 0


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
                    reason='native Single projection contract requires an Ampere CUDA GPU')
@pytest.mark.parametrize('width', [1, 2, 5])
def test_cuda_shared_single_retains_independent_fp32_score_for_every_tile_width(width, record_property):
    """Small native score/variance contracts; this is not a real-data benchmark."""
    rows = np.arange(256)[::-1].copy()
    block = minor_block(states_fixture(), rows, np.arange(10), device='cuda:0')
    models = [model_for(rows, .81371+i*.28743, device='cuda:0') for i in range(width)]
    pipelines = [make_pipeline(rows, [], model=model, device='cuda:0') for model in models]
    state, ordinal = pipelines[0]._prepare_individual_block(block, 0, mac_cutoff=1)
    prepared = pipelines[0]._prepare_individual_trait(state, 0, mac_cutoff=1)
    prepared.pop('model')
    genotype = prepared['genotype']
    expected = []
    for model, pipeline in zip(models, pipelines):
        score, variance = model.individual_score_variance(genotype, matmul_mode='tf32')
        values = torch.stack((variance, score, variance.double().sqrt(),
                              _individual_log_probabilities(score, variance)), dim=1)
        expected.append(pipeline._individual_records_from_values(
            dict(prepared, model=model), values.cpu().numpy(), subset_variants_num=3))
    single.execution_metadata(reset=True)
    actual, new_ordinal = single.process_single_shared_genotype(
        pipelines, None, ordinal, '21', mac_cutoff=1, subset_variants_num=3,
        prepared_state=(prepared, ordinal))
    assert new_ordinal == ordinal and prepared['genotype'] is genotype
    deltas = {'Score': 0., 'Est': 0., 'logP': 0.}
    for actual_rows, expected_rows in zip(actual, expected):
        if width == 1:
            # The singleton keeps the original FP32 vector reduction exactly.
            assert_tables_equal(actual_rows, expected_rows)
            continue
        assert len(actual_rows) == len(expected_rows)
        for left, right in zip(actual_rows, expected_rows):
            assert list(left) == list(right)
            for key in left:
                av, bv = left[key], right[key]
                if key in ('Score', 'Est'):
                    np.testing.assert_allclose(av, bv, rtol=5e-6, atol=5e-6, equal_nan=True)
                    if np.isfinite(av) and np.isfinite(bv):
                        deltas[key] = max(deltas[key], abs(av-bv))
                elif key in ('pvalue', 'pvalue_log10', 'pvalue_log'):
                    if not (np.isfinite(av) and np.isfinite(bv)):
                        np.testing.assert_allclose(av, bv, rtol=0, atol=0, equal_nan=True)
                    else:
                        if key == 'pvalue':
                            assert 0 <= av <= 1 and 0 <= bv <= 1
                            # A zero retains its native finite log(P); no floor.
                            la, lb = left['pvalue_log10'], right['pvalue_log10']
                        else:
                            divisor = np.log(10.) if key == 'pvalue_log' else 1.
                            la, lb = av/divisor, bv/divisor
                        if np.isfinite(la) and np.isfinite(lb):
                            assert abs(la-lb) < .001
                            deltas['logP'] = max(deltas['logP'], abs(la-lb))
                        else:
                            np.testing.assert_allclose(la, lb, rtol=0, atol=0, equal_nan=True)
                elif isinstance(av, Real) and isinstance(bv, Real):
                    # AF/MAC/N, original variance/SE and all ordering markers
                    # stay exact: only the full-FP32 score reduction changes.
                    np.testing.assert_allclose(av, bv, rtol=0, atol=0, equal_nan=True)
                else:
                    assert av == bv
    for name, delta in deltas.items():
        record_property('maximum_'+name+'_absolute_delta', delta)
    metrics = single.execution_metadata()
    assert metrics['phenotype_score_gemm_calls'] == 0
    refined = metrics['phenotype_score_extreme_tail_fallback_columns']
    assert metrics['phenotype_score_batches'] == 1 + refined
    record_property('extreme_tail_native_score_columns', refined)
    if width == 1:
        assert metrics['phenotype_score_fp32_vector_calls'] == 1
        assert metrics['phenotype_score_fp32_simt_gemm_calls'] == 0
        assert refined == metrics['phenotype_score_control_d2h_batches'] == 0
    else:
        simt = metrics['phenotype_score_fp32_simt_gemm_calls']
        if simt:
            assert simt == 1 and refined > 0
            assert metrics['phenotype_score_fp32_vector_calls'] == refined
            assert metrics['phenotype_score_control_d2h_batches'] == 1
            assert metrics['phenotype_score_control_d2h_bytes'] == width
        else:
            assert metrics['phenotype_score_fp32_vector_calls'] == width
            assert refined == metrics['phenotype_score_control_d2h_batches'] == 0
    assert metrics['original_variance_calls'] == width
    assert metrics['result_transfer_batches'] == 1
    assert metrics['dense_genotype_h2d_bytes'] == metrics['dense_genotype_d2h_bytes'] == 0


def test_sensitive_column_is_recomputed_whole_before_binary_correction(
        monkeypatch, cpu_product_oracle):
    """Fault-injected arithmetic routes verify the fallback/SPA protocol."""
    models, pipelines, prepared, ordinal = prepared_tile(['gaussian', 'binomial'])
    genotype = prepared['genotype']
    native = [torch.mv(genotype.T, model.scaled_residuals) for model in models]
    calls = []
    statistics = {'fp32_simt_products': 0, 'fp32_vector_products': 0,
                  'simt_admission_fallbacks': 0}
    def controlled_scores(g, vectors):
        assert g is genotype and tuple(g.shape) == tuple(genotype.shape)
        calls.append(tuple(id(value) for value in vectors))
        if len(vectors) == 2:
            statistics['fp32_simt_products'] += 1
            # Only the binary column is deliberately sensitive. Do not hide
            # its error with a selected-variant GEMV or a probability floor.
            return torch.stack((native[0], native[1]+.01), dim=1)
        statistics['fp32_vector_products'] += 1
        assert vectors[0] is models[1].scaled_residuals
        return native[1][:, None].clone()
    monkeypatch.setattr(single, 'fp32_score_columns', controlled_scores)
    monkeypatch.setattr(single, 'score_execution_metadata', lambda: dict(statistics))
    positions = {id(model): i for i, model in enumerate(models)}
    variances = [torch.full_like(native[0], 1e8), torch.full_like(native[1], 1e-8)]
    monkeypatch.setattr(single, '_variance_from_shared_genotype',
        lambda model, *unused: variances[positions[id(model)]])
    corrected = []
    def correction(index, model, g, score, variance):
        assert index == 1 and model is models[1] and g is genotype
        assert torch.equal(score, native[1]) and torch.equal(variance, variances[1])
        corrected.append(index)
        return binary_correction(index, model, g, score, variance)
    single.execution_metadata(reset=True)
    actual, _ = single.process_single_shared_genotype(pipelines, None, ordinal, '21',
        mac_cutoff=1, prepared_state=(prepared, ordinal), binary_correction=correction)
    assert corrected == [1] and len(calls) == 2 and len(calls[-1]) == 1
    assert np.array_equal([row['Score'] for row in actual[1]], native[1].numpy())
    metrics = single.execution_metadata()
    assert metrics['phenotype_score_extreme_tail_fallback_columns'] == 1
    assert metrics['phenotype_score_fp32_vector_calls'] == 1
    assert metrics['phenotype_score_fp32_simt_gemm_calls'] == 1
    assert metrics['phenotype_score_batches'] == 2
    assert metrics['phenotype_score_control_d2h_batches'] == 0  # CPU protocol fixture.
    assert metrics['result_transfer_batches'] == 1
