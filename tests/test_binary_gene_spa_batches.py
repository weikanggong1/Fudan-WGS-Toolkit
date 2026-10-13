"""Selected heterogeneous SPA batches; fixtures are allocation contracts."""
from contextlib import contextmanager
import pytest
import torch

from fudan_wgs_toolkit import binary_gene_batch
from fudan_wgs_toolkit.binary import prepare_binary_burdens, ExactBinaryBurdenCache, staar_binary_phewas
from test_binary_gene_batch import make_models, genetic


def setup(count=9):
    prepared = make_models(count=count)
    g, f, a = genetic(57)
    burdens = prepare_binary_burdens(prepared[0].model, g, f, a, ['a', 'b'])
    return prepared, g, f, a, burdens


def test_selected_trait_batches_hold_original_leases_and_match_independent_outputs(monkeypatch):
    prepared, g, f, a, burdens = setup()
    active, events = set(), []
    factories = []
    for trait, item in enumerate(prepared):
        @contextmanager
        def acquire(trait=trait, model=item.spa_model):
            active.add(trait); events.append(('enter', trait))
            try:
                yield model
            finally:
                active.remove(trait); events.append(('exit', trait))
        factories.append(acquire)
    original = binary_gene_batch.binary_spa
    widths = []
    def selected(score, projected, probabilities, **kwargs):
        assert 1 <= len(active) <= 4
        assert projected.shape == probabilities.shape
        assert projected.shape[1] <= 16 and kwargs['combine_tails'] is True
        assert projected.dtype == probabilities.dtype == torch.float64
        widths.append(projected.shape[1])
        return original(score, projected, probabilities, **kwargs)
    monkeypatch.setattr(binary_gene_batch, 'binary_spa', selected)
    actual, execution = binary_gene_batch.evaluate_binary_burden_tile(
        [item.model for item in prepared], g, f, a, ['a', 'b'],
        prepared_burdens=burdens, spa_acquires=factories,
        exact_burden_cache=ExactBinaryBurdenCache(burdens), p_filter_cutoff=1)
    assert not active and len(events) == 2*len(prepared)
    assert execution['spa_batched_traits'] == len(prepared)
    assert execution['spa_trait_batches'] == execution['spa_combined_tail_batches'] == len(widths)
    assert max(widths) <= 16 and execution['spa_batch_admission_fallbacks'] == 0
    for item, (result, chosen, failed) in zip(prepared, actual):
        expected, diagnostic = staar_binary_phewas(item.model, g, f, a, ['a', 'b'],
            spa_model=item.spa_model, prepared_burdens=burdens, p_filter_cutoff=1,
            return_diagnostics=True)
        assert result == pytest.approx(expected, abs=2e-12, rel=2e-12)
        assert chosen == int(diagnostic['spa_selected'].sum())
        assert failed == int(diagnostic['spa'].failed.sum())


@pytest.mark.parametrize('reject', ['callback', 'built_in', 'sidecar'])
def test_spa_batch_admission_failure_releases_leases_and_uses_independent_path(monkeypatch, reject):
    prepared, g, f, a, burdens = setup(3)
    active, enters, releases = set(), [], []
    factories = []
    first = True
    for trait, item in enumerate(prepared):
        @contextmanager
        def acquire(trait=trait, model=item.spa_model):
            nonlocal first
            if reject == 'sidecar' and first and trait == 1:
                first = False
                raise MemoryError('optional second fitted state rejected')
            active.add(trait); enters.append(trait)
            try:
                yield model
            finally:
                active.remove(trait)
        factories.append(acquire)
    requests = []
    def guard(required):
        requests.append(required)
        return reject != 'callback'
    if reject == 'built_in':
        monkeypatch.setattr(binary_gene_batch, '_guard_exact_burden_workspace',
                            lambda *args: (_ for _ in ()).throw(MemoryError('optional stack rejected')))
    actual, execution = binary_gene_batch.evaluate_binary_burden_tile(
        [item.model for item in prepared], g, f, a, ['a', 'b'],
        prepared_burdens=burdens, spa_acquires=factories,
        exact_burden_cache=ExactBinaryBurdenCache(burdens), p_filter_cutoff=1,
        normal_allocation_guard=guard, normal_workspace_release=lambda: releases.append(True))
    assert not active and requests and releases
    assert execution['spa_batch_admission_fallbacks'] >= 1
    for item, (result, _, _) in zip(prepared, actual):
        expected = staar_binary_phewas(item.model, g, f, a, ['a', 'b'],
            spa_model=item.spa_model, prepared_burdens=burdens, p_filter_cutoff=1)
        assert result == pytest.approx(expected, abs=2e-12, rel=2e-12)


def test_selected_packing_never_splits_a_large_trait_or_exceeds_bounds():
    selected = torch.zeros((11, 26), dtype=torch.bool)
    counts = [8, 8, 1, 2, 3, 4, 8, 26, 0, 1, 2]
    for trait, count in enumerate(counts):
        selected[trait, :count] = True
    batches = list(binary_gene_batch._selected_spa_batches(selected))
    assert all(len(batch) <= 4 and sum(len(columns) for _, columns in batch) <= 16 for batch in batches)
    assert {trait for batch in batches for trait, _ in batch} == {i for i, n in enumerate(counts) if 0 < n <= 8}


def test_diagonal_and_full_covariance_normal_products_have_identical_output():
    prepared, g, f, a, burdens = setup(1)
    model = prepared[0].model
    score, covariance = model.score_covariance(burdens.burdens)
    full, full_diag = staar_binary_phewas(model, g, f, a, ['a', 'b'], prepared_burdens=burdens,
        spa_model=prepared[0].spa_model, _normal_products=(score, covariance), return_diagnostics=True)
    diagonal, diag = staar_binary_phewas(model, g, f, a, ['a', 'b'], prepared_burdens=burdens,
        spa_model=prepared[0].spa_model, _normal_products=(score, covariance.diagonal()), return_diagnostics=True)
    assert full == diagonal
    assert full_diag['covariance_shape'] == [6, 6] and diag['covariance_shape'] == [6]
    torch.testing.assert_close(full_diag['spa_selected'], diag['spa_selected'], rtol=0, atol=0)


@pytest.mark.parametrize('variance_factor', [.9995, 1.0005])
def test_near_filter_rechecks_vector_score_and_retains_repaired_variance(monkeypatch, variance_factor):
    from fudan_wgs_toolkit import binary_null, tf32
    from fudan_wgs_toolkit.statistics import _chi1_from_score
    monkeypatch.setattr(binary_null, 'matmul', lambda left, right, *, mode: left@right)
    monkeypatch.setattr(tf32, 'matmul', lambda left, right, *, mode: left@right)
    prepared = make_models(count=1, native=True)
    g, f, a = genetic(57, dtype=torch.float32)
    model = prepared[0].model
    burdens = prepare_binary_burdens(model, g, f, a, ['a', 'b'])
    score, covariance = model.score_covariance(burdens.burdens)
    cutoff = float(_chi1_from_score(score, covariance.diagonal())[0])
    repaired_variance = covariance.diagonal()*variance_factor
    def altered(*args, **kwargs):
        return ((score*.99999)[None].clone(), repaired_variance[None].clone(),
                dict(native_projection_batches=1, admission_fallbacks=0))
    monkeypatch.setattr(binary_gene_batch, 'binary_burden_score_variance_batch', altered)
    def forbidden_covariance(*args, **kwargs):
        raise AssertionError('near-cutoff vector recheck must not call the old covariance')
    monkeypatch.setattr(model,'score_covariance',forbidden_covariance)
    @contextmanager
    def acquire():
        yield prepared[0].spa_model
    actual, execution = binary_gene_batch.evaluate_binary_burden_tile([model], g, f, a, ['a', 'b'],
        prepared_burdens=burdens, spa_acquires=[acquire],
        exact_burden_cache=ExactBinaryBurdenCache(burdens), p_filter_cutoff=cutoff)
    expected, diagnostic = staar_binary_phewas(model, g, f, a, ['a', 'b'],
        spa_model=prepared[0].spa_model, prepared_burdens=burdens,
        _normal_products=(score,repaired_variance),p_filter_cutoff=cutoff, return_diagnostics=True)
    assert execution['threshold_score_rechecks'] == execution['threshold_variances_preserved'] == 1
    assert execution['threshold_product_rechecks'] == 0
    assert actual[0][1] == int(diagnostic['spa_selected'].sum())
    assert actual[0][0] == pytest.approx(expected, abs=2e-12, rel=2e-12)


def test_spa_batch_solver_failure_releases_all_sidecars_and_workspace(monkeypatch):
    prepared, g, f, a, burdens = setup(2)
    active, released = set(), []
    factories = []
    for trait, item in enumerate(prepared):
        @contextmanager
        def acquire(trait=trait, model=item.spa_model):
            active.add(trait)
            try:
                yield model
            finally:
                active.remove(trait)
        factories.append(acquire)
    def fail(*args, **kwargs):
        assert active == {0, 1}
        raise ValueError('injected solver failure')
    monkeypatch.setattr(binary_gene_batch, 'binary_spa', fail)
    with pytest.raises(ValueError, match='solver failure'):
        binary_gene_batch.evaluate_binary_burden_tile([item.model for item in prepared],
            g, f, a, ['a', 'b'], prepared_burdens=burdens, spa_acquires=factories,
            exact_burden_cache=ExactBinaryBurdenCache(burdens), p_filter_cutoff=1,
            normal_allocation_guard=lambda required: True,
            normal_workspace_release=lambda: released.append(True))
    assert not active and len(released) == 2  # normal and optional SPA workspace


@pytest.mark.parametrize('changes, message', [
    ({'p_filter_cutoff': 0}, 'cutoffs'),
    ({'p_filter_cutoff': 1.01}, 'cutoffs'),
    ({'rv_num_cutoff': 58}, 'rare-variant count'),
    ({'rv_num_cutoff_max': 57}, 'rare-variant count'),
])
def test_selected_batch_preserves_original_entry_validation(changes, message):
    prepared, g, f, a, burdens = setup(2)
    entered = []
    @contextmanager
    def acquire():
        entered.append(True)
        yield prepared[0].spa_model
    with pytest.raises(ValueError, match=message):
        binary_gene_batch.evaluate_binary_burden_tile([item.model for item in prepared],
            g, f, a, ['a', 'b'], prepared_burdens=burdens, spa_acquires=[acquire, acquire],
            exact_burden_cache=ExactBinaryBurdenCache(burdens), **changes)
    assert entered == []


def test_selected_batch_requires_verified_lazy_factory_binding_before_acquisition():
    prepared, g, f, a, burdens = setup(2)
    prepared[1].model.null_fit_source_sha256 = None
    entered = []
    @contextmanager
    def acquire():
        entered.append(True)
        yield prepared[0].spa_model
    with pytest.raises(ValueError, match='verified normal fitted-source'):
        binary_gene_batch.evaluate_binary_burden_tile([item.model for item in prepared],
            g, f, a, ['a', 'b'], prepared_burdens=burdens, spa_acquires=[acquire, acquire],
            exact_burden_cache=ExactBinaryBurdenCache(burdens), p_filter_cutoff=1)
    assert entered == []
