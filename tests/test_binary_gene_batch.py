"""Independent-model and bounded allocation contracts, not real-data benchmarks."""
from contextlib import contextmanager
import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.binary import (ExactBinaryBurdenCache, prepare_binary_burdens,
                                    staar_binary_phewas, _spa_combination)
from fudan_wgs_toolkit.binary_gene_batch import evaluate_binary_burden_tile, _spa_combination_rows
from fudan_wgs_toolkit.binary_null import binary_burden_score_covariance_batch, binary_burden_batch_workspace
from fudan_wgs_toolkit.phewas_models import prepare_phewas_model
from fudan_wgs_toolkit.statistics import annotation_weights


def make_models(n=64, count=3, *, native=False):
    x = np.column_stack((np.ones(n), np.linspace(-1, 1, n)))
    rng = np.random.default_rng(417)
    result = []
    for trait in range(count):
        y = rng.binomial(1, .2+.05*(trait % 7), n).astype(float)
        model = prepare_phewas_model(y, np.arange(n).astype(str), x,
                                    association_mode='fp64')
        if native:
            model.model.set_matmul_mode('tf32')
        result.append(model)
    return result


def genetic(m, n=64, dtype=torch.float64):
    generator = torch.Generator().manual_seed(19+m)
    g = (torch.rand((n, m), generator=generator) < .09).to(dtype)
    f = torch.linspace(.001, .009, m, dtype=torch.float64)
    a = torch.stack((torch.arange(m).remainder(9), torch.arange(m).remainder(5)), 1).double()/3
    return g, f, a


def factories(prepared):
    result = []
    for item in prepared:
        @contextmanager
        def acquire(value=item.spa_model):
            yield value
        result.append(acquire)
    return result


@pytest.mark.parametrize('m', [2, 31, 511, 512, 513, 4999, 5000, 5001, 15001, 50000, 100000])
@pytest.mark.parametrize('layout', ['c', 'f'])
def test_all_mask_lengths_match_independent_burden_spa_and_schema(m, layout):
    prepared = make_models()
    g, f, a = genetic(m)
    if layout == 'f':
        g = g.T.contiguous().T
    baseline = [staar_binary_phewas(item.model, g, f, a, ['a', 'b'], spa_model=item.spa_model,
                                   p_filter_cutoff=.8, return_diagnostics=True) for item in prepared]
    shared = prepare_binary_burdens(prepared[0].model, g, f, a, ['a', 'b'])
    requests = []
    cache = ExactBinaryBurdenCache(shared, allocation_guard=requests.append)
    actual, execution = evaluate_binary_burden_tile([item.model for item in prepared], g, f, a,
        ['a', 'b'], prepared_burdens=shared, spa_acquires=factories(prepared),
        exact_burden_cache=cache, p_filter_cutoff=.8)
    assert execution['native_projection_batches'] == 0  # This is an actual FP64 CPU reference.
    for (result, selected, failed), (expected, diagnostic) in zip(actual, baseline):
        assert set(result) == set(expected)
        for name in expected:
            assert result[name] == pytest.approx(expected[name], abs=2e-10, rel=2e-10)
        assert selected == int(diagnostic['spa_selected'].sum())
        assert failed == int(diagnostic['spa'].failed.sum())
    assert cache.columns_computed == len(cache.filled) <= shared.burdens.shape[1]
    assert cache.build_calls <= 1
    assert not requests or max(requests) >= 8*g.shape[0]*shared.burdens.shape[1]


def test_exact_cache_reuses_only_selected_columns_and_rejects_mutation():
    item = make_models(count=1)[0]
    g, f, a = genetic(519)
    shared = prepare_binary_burdens(item.model, g, f, a, ['a', 'b'])
    cache = ExactBinaryBurdenCache(shared, variant_tile_size=37)
    first = cache.get(shared, torch.tensor([1, 3]))
    second = cache.get(shared, torch.tensor([3, 4]))
    assert cache.columns_computed == 3 and cache.build_calls == 2
    expected = torch.zeros_like(cache.values)
    weights = annotation_weights(f, a)[0]
    for start in range(0, len(f), 37):
        expected.add_(g[:, start:start+37]@weights[start:start+37])
    torch.testing.assert_close(first, expected[:, [1, 3]], rtol=0, atol=2e-12)
    torch.testing.assert_close(second, expected[:, [3, 4]], rtol=0, atol=2e-12)
    assert cache.get(shared, torch.tensor([], dtype=torch.int64)).shape == (64, 0)
    g[0, 0] += 1
    with pytest.raises(ValueError, match='mutated'):
        cache.get(shared, torch.tensor([1]))


def test_exact_cache_admits_before_any_storage_or_compute():
    item = make_models(count=1)[0]
    g, f, a = genetic(13)
    shared = prepare_binary_burdens(item.model, g, f, a, ['a', 'b'])
    def reject(required):
        assert required >= 8*64*6
        raise MemoryError('no optional exact slab')
    cache = ExactBinaryBurdenCache(shared, allocation_guard=reject)
    with pytest.raises(MemoryError, match='optional exact'):
        cache.get(shared, torch.tensor([1]))
    assert cache.values is None and cache.weights is None and not cache.filled


@pytest.mark.parametrize('row', [
    [1., 1., float('nan')], [0., 0., 1.], [0., .1, 1.],
    [.2, .4, .8], [1e-20, .02, .9], [1e-12, .9, 1.], [.99, .95, .98]])
def test_packed_combination_retains_original_failure_zero_and_tail_rules(row):
    values = torch.tensor(row, dtype=torch.float64)
    actual = _spa_combination_rows(values[None])[0]
    expected = _spa_combination(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-16)


@pytest.mark.parametrize('sparse', [False, True])
def test_native_batch_matches_each_independent_fitted_formula_with_cpu_oracle(monkeypatch, sparse):
    # The oracle executes the independent FP32 products on CPU. CUDA gates
    # separately verify that the production backend actually emits TF32 MMA.
    from fudan_wgs_toolkit import binary_null, tf32
    monkeypatch.setattr(binary_null, 'matmul', lambda left, right, *, mode: left@right)
    original = tf32.batched_matmul
    monkeypatch.setattr(tf32, 'batched_matmul', lambda left, right, **kwargs:
                        original(left, right, mode='fp64'))
    prepared = make_models(native=True)
    g, f, a = genetic(47, dtype=torch.float32)
    for i, item in enumerate(prepared):
        if sparse:
            diagonal = item.model.precision
            item.model.precision = torch.diag(diagonal).to_sparse()
    z = g@annotation_weights(f, a, dtype=torch.float32)[0]
    expected = [item.model.score_covariance(z) for item in prepared]
    requests = []
    scores, covariances, execution = binary_burden_score_covariance_batch(
        [item.model for item in prepared], z, allocation_guard=requests.append)
    assert execution == dict(native_projection_batches=1, admission_fallbacks=0)
    assert len(requests) == 1
    for i, (score, covariance) in enumerate(expected):
        torch.testing.assert_close(scores[i], score, rtol=0, atol=0)
        torch.testing.assert_close(covariances[i], covariance, rtol=0, atol=0)
    # Rejecting optional stack needs no alternate model, no changed formula.
    scores, covariances, execution = binary_burden_score_covariance_batch(
        [item.model for item in prepared], z, allocation_guard=lambda required: False)
    assert execution == dict(native_projection_batches=0, admission_fallbacks=1)
    for i, (score, covariance) in enumerate(expected):
        torch.testing.assert_close(scores[i], score, rtol=0, atol=0)
        torch.testing.assert_close(covariances[i], covariance, rtol=0, atol=0)


def test_normal_batch_workspace_is_bounded_and_contains_real_optional_storage():
    n, b, q, t = 61, 26, 16, 32
    estimate = binary_burden_batch_workspace(n, b, q, t)
    buffers = [torch.empty((t, n, b)), torch.empty((n, b)),
               torch.empty((n, t)), torch.empty((t, b, b)), torch.empty((t, q, b))]
    assert estimate >= sum(value.untyped_storage().nbytes() for value in buffers)+256*2**20
    assert binary_burden_batch_workspace(343034, 26, 16, 799) == binary_burden_batch_workspace(343034, 26, 16, 32)


def test_independent_batch_guard_runs_before_stack_without_external_callback(monkeypatch):
    from fudan_wgs_toolkit import binary_null
    monkeypatch.setattr(binary_null, 'matmul', lambda left, right, *, mode: left@right)
    prepared = make_models(native=True)
    g, f, a = genetic(13, dtype=torch.float32)
    z = g@annotation_weights(f, a, dtype=torch.float32)[0]
    observed = []
    def reject(value, required):
        assert value is z and required >= 4*len(prepared)*z.numel()
        observed.append(required)
        raise MemoryError('built-in preflight rejected')
    monkeypatch.setattr(binary_null, '_guard_binary_burden_batch_workspace', reject)
    empty = torch.empty
    def checked_empty(shape, *args, **kwargs):
        assert tuple(shape) != (len(prepared), *z.shape)
        return empty(shape, *args, **kwargs)
    monkeypatch.setattr(torch, 'empty', checked_empty)
    scores, covariances, execution = binary_burden_score_covariance_batch(
        [item.model for item in prepared], z)
    assert len(observed) == 1
    assert execution == dict(native_projection_batches=0, admission_fallbacks=1)
    for i, item in enumerate(prepared):
        score, covariance = item.model.score_covariance(z)
        torch.testing.assert_close(scores[i], score, rtol=0, atol=0)
        torch.testing.assert_close(covariances[i], covariance, rtol=0, atol=0)


def test_tile_rejects_different_ordered_samples_before_any_spa():
    prepared = make_models(count=2)
    g, f, a = genetic(17)
    shared = prepare_binary_burdens(prepared[0].model, g, f, a, ['a', 'b'])
    prepared[1].model.sample_ids = prepared[1].model.sample_ids[::-1]
    with pytest.raises(ValueError, match='sample axis'):
        evaluate_binary_burden_tile([item.model for item in prepared], g, f, a, ['a', 'b'],
            prepared_burdens=shared, spa_acquires=factories(prepared),
            exact_burden_cache=ExactBinaryBurdenCache(shared))


def test_exact_columns_survive_multiple_trait_tiles_but_not_another_mask():
    prepared = make_models(count=5)
    g, f, a = genetic(517)
    shared = prepare_binary_burdens(prepared[0].model, g, f, a, ['a', 'b'])
    cache = ExactBinaryBurdenCache(shared)
    for begin in (0, 2, 4):
        tile = prepared[begin:begin+2]
        actual, _ = evaluate_binary_burden_tile([item.model for item in tile], g, f, a,
            ['a', 'b'], prepared_burdens=shared, spa_acquires=factories(tile),
            exact_burden_cache=cache, p_filter_cutoff=1)
        for item, (result, _, _) in zip(tile, actual):
            expected = staar_binary_phewas(item.model, g, f, a, ['a', 'b'],
                                          spa_model=item.spa_model, p_filter_cutoff=1)
            for field in expected:
                assert result[field] == pytest.approx(expected[field], abs=2e-10, rel=2e-10)
    assert cache.build_calls == 1 and cache.columns_computed == 6
    other = prepare_binary_burdens(prepared[0].model, g[:, :12], f[:12], a[:12], ['a', 'b'])
    with pytest.raises(ValueError, match='different'):
        cache.get(other, torch.tensor([1]))


def test_native_projection_chunks_above_32_before_optional_allocation(monkeypatch):
    from fudan_wgs_toolkit import binary_null, tf32
    monkeypatch.setattr(binary_null, 'matmul', lambda left, right, *, mode: left@right)
    original = tf32.batched_matmul
    monkeypatch.setattr(tf32, 'batched_matmul', lambda left, right, **kwargs:
                        original(left, right, mode='fp64'))
    prepared = make_models(count=35, native=True)
    g, f, a = genetic(19, dtype=torch.float32)
    z = g@annotation_weights(f, a, dtype=torch.float32)[0]
    requests = []
    scores, covariances, execution = binary_burden_score_covariance_batch(
        [item.model for item in prepared], z, allocation_guard=requests.append)
    assert execution == dict(native_projection_batches=2, admission_fallbacks=0)
    assert len(requests) == 2 and requests[0] > requests[1]
    for i, item in enumerate(prepared):
        score, covariance = item.model.score_covariance(z)
        torch.testing.assert_close(scores[i], score, rtol=0, atol=0)
        torch.testing.assert_close(covariances[i], covariance, rtol=0, atol=0)


def test_near_filter_scores_retain_original_strict_spa_boundary(monkeypatch):
    from fudan_wgs_toolkit import binary_null, tf32
    from fudan_wgs_toolkit.phewas_runtime import scores as score_module
    from fudan_wgs_toolkit.statistics import _chi1_from_score
    monkeypatch.setattr(binary_null, 'matmul', lambda left, right, *, mode: left@right)
    original = tf32.batched_matmul
    monkeypatch.setattr(tf32, 'matmul', lambda left, right, *, mode: left@right)
    monkeypatch.setattr(tf32, 'batched_matmul', lambda left, right, **kwargs:
                        original(left, right, mode='fp64'))
    original_score = score_module.fp32_score_columns
    monkeypatch.setattr(score_module, 'fp32_score_columns', lambda *args, **kwargs:
                        original_score(*args, **kwargs)*(1+1e-6))
    prepared = make_models(native=True)
    g, f, a = genetic(57, dtype=torch.float32)
    shared = prepare_binary_burdens(prepared[0].model, g, f, a, ['a', 'b'])
    score, cov = prepared[0].model.score_covariance(shared.burdens)
    cutoff = float(_chi1_from_score(score, cov.diagonal())[0])
    actual, execution = evaluate_binary_burden_tile([item.model for item in prepared], g, f, a,
        ['a', 'b'], prepared_burdens=shared, spa_acquires=factories(prepared),
        exact_burden_cache=ExactBinaryBurdenCache(shared), p_filter_cutoff=cutoff)
    expected, diagnostic = staar_binary_phewas(prepared[0].model, g, f, a, ['a', 'b'],
        spa_model=prepared[0].spa_model, prepared_burdens=shared,
        p_filter_cutoff=cutoff, return_diagnostics=True)
    assert execution['threshold_score_rechecks'] >= 1
    assert actual[0][1] == int(diagnostic['spa_selected'].sum())
    assert actual[0][0]['Burden(1,25)'] == expected['Burden(1,25)']
