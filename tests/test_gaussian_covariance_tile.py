"""Pair-major contracts; CPU fixtures do not establish real-data speed."""
from contextlib import contextmanager
import weakref

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit import tf32, null_model
from fudan_wgs_toolkit._blocked_covariance import _PanelPreparation
from fudan_wgs_toolkit._gaussian_covariance_tile import (
    blocked_gaussian_covariance_tile, gaussian_covariance_tile_workspace_bytes)
from test_gaussian_covariance_shared import Broker, model, genotype, covariance, cpu_products


@pytest.mark.parametrize('count', [1, 2, 4])
@pytest.mark.parametrize('fortran', [False, True])
@pytest.mark.parametrize('capacity', [0, 500, 2**20])
def test_independent_nonconstant_precisions_match_every_original_product(
        cpu_products, count, fortran, capacity):
    models = [model(shift=.21*i) for i in range(count)]
    host, broker, report = genotype(m=10, fortran=fortran), Broker(), {}
    expected = [covariance(fitted, host, broker, width=3, capacity=capacity)
                for fitted in models]
    broker.guards.clear()
    actual = blocked_gaussian_covariance_tile(models, host, broker,
        variant_tile_size=3, capacity_bytes=capacity, report=report)
    assert isinstance(actual, tuple) and len(actual) == count
    assert len({out.untyped_storage().data_ptr() for out in actual}) == count
    for out, reference in zip(actual, expected):
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
        torch.testing.assert_close(out, out.T, rtol=0, atol=0)
    metrics = report['gene_pair_major_covariance']['recent_calls'][-1]
    assert metrics['panel_pairs'] == 10
    assert metrics['directed_products'] == count*16
    assert metrics['peak_storage_bytes'] <= capacity
    assert metrics['storage_bytes_after_close'] == metrics['pinned_panels_after_close'] == 0
    assert not metrics['precision_shared'] and not metrics['covariance_shared']
    assert not metrics['model_projection_shared'] and not metrics['collection_forced']
    assert not broker.callbacks
    assert broker.guards[0] == gaussian_covariance_tile_workspace_bytes(models, host,
                                                                       variant_tile_size=3)
    assert broker.guards.count(broker.guards[0]) == 1  # All independent NEW outputs once.


def test_zero_capacity_reads_and_rotates_pairs_once_for_four_models(cpu_products):
    models, host, broker = [model(shift=.21*i) for i in range(4)], genotype(m=19), Broker()
    report = {}
    baseline_preparations = 0
    for fitted in models:
        baseline = {}
        covariance(fitted, host, broker, capacity=0, report=baseline)
        baseline_preparations += baseline['gene_panel_preparation_cache']['panel_preparations']
    actual = blocked_gaussian_covariance_tile(models, host, broker,
        variant_tile_size=3, capacity_bytes=0, report=report)
    metrics = report['gene_pair_major_covariance']['recent_calls'][-1]
    assert actual is not None
    assert baseline_preparations == 4*28
    assert metrics['panel_preparations'] == 28
    assert metrics['capacity_bytes'] == metrics['peak_storage_bytes'] == 0


def test_small_projections_are_cached_inside_same_total_capacity(cpu_products):
    models, host, broker, report = [model(shift=.21*i) for i in range(4)], genotype(m=19), Broker(), {}
    blocked_gaussian_covariance_tile(models, host, broker, variant_tile_size=3,
                                    capacity_bytes=2**20, report=report)
    metrics = report['gene_pair_major_covariance']['recent_calls'][-1]
    assert metrics['panel_preparations'] == 7
    assert metrics['projection_preparations'] == 4*7
    assert metrics['projection_hits'] > 0
    assert metrics['independent_projection_capacity_bytes'] == 8*19*3*3
    assert (metrics['raw_preparation_capacity_bytes']+
            metrics['independent_projection_capacity_bytes']) == 2**20


def test_each_model_retains_its_own_covariate_width(cpu_products):
    models = [model(shift=.21*i) for i in range(4)]
    for fitted, q in zip(models, (3, 2, 1, 3)):
        fitted.x = fitted.x[:, :q].contiguous()
        fitted.precision_x = fitted.precision_x[:, :q].contiguous()
        fitted.fixed_effect_covariance = torch.linalg.inv(fitted.x.T@fitted.precision_x)
    host, broker = genotype(m=7), Broker()
    expected = [covariance(fitted, host, broker, width=3) for fitted in models]
    actual = blocked_gaussian_covariance_tile(models, host, broker,
        variant_tile_size=3, capacity_bytes=2**20)
    for out, reference in zip(actual, expected):
        torch.testing.assert_close(out, reference, rtol=0, atol=0)


@pytest.mark.parametrize('changed', ['axis', 'rotation', 'rows', 'eigenvalues', 'mode', 'dtype', 'spa'])
def test_incompatible_model_declines_before_materialization(cpu_products, monkeypatch, changed):
    models, host, broker, report = [model(), model(shift=.2)], genotype(), Broker(), {}
    if changed == 'axis': models[1].sample_ids = models[1].sample_ids[::-1].copy()
    elif changed == 'rotation': models[1].spectrum.blocks[0][1].add_(.01)
    elif changed == 'rows': models[1].spectrum.blocks[0][0].add_(1)
    elif changed == 'eigenvalues': models[1].spectrum.eigenvalues[0] += .1
    elif changed == 'mode': models[1].matmul_mode = 'fp64'
    elif changed == 'dtype': models[1].precision_x = models[1].precision_x.double()
    else: models[1].use_spa = True
    monkeypatch.setattr(torch, 'empty', lambda *a, **k: pytest.fail('declined path allocated output'))
    actual = blocked_gaussian_covariance_tile(models, host, broker, report=report)
    assert actual is None and not broker.guards and not broker.callbacks
    assert report['gene_pair_major_covariance']['recent_calls'][-1]['fallback_reason'] == 'incompatible_exact_axis_rotation_or_state'


def test_joint_output_admission_declines_without_registering_future_or_hiding_later_errors(
        cpu_products, monkeypatch):
    models, host, broker, report = [model(), model(shift=.2)], genotype(), Broker(), {}
    required = gaussian_covariance_tile_workspace_bytes(models, host, variant_tile_size=3)
    def refuse(size):
        assert size == required
        raise MemoryError('joint NEW outputs do not fit')
    broker._guard = refuse
    monkeypatch.setattr(torch, 'empty', lambda *a, **k: pytest.fail('declined path allocated output'))
    assert blocked_gaussian_covariance_tile(models, host, broker, variant_tile_size=3,
                                            report=report) is None
    assert report['gene_pair_major_covariance']['recent_calls'][-1]['admission_new_bytes'] == required
    assert required == 4*2*10*10+20*23*3+16*3*3+256*2**20
    assert not broker.callbacks


def test_live_capacity_never_increases_original_fifteen_percent_budget(cpu_products):
    models, host, broker, report = [model(), model(shift=.2)], genotype(), Broker(), {}
    scratch = gaussian_covariance_tile_workspace_bytes(models, host, variant_tile_size=3)-4*2*10*10
    broker._live_available_bytes = lambda: scratch+10000
    blocked_gaussian_covariance_tile(models, host, broker, variant_tile_size=3,
                                    capacity_bytes=16*2**30, report=report)
    metrics = report['gene_pair_major_covariance']['recent_calls'][-1]
    assert metrics['capacity_bytes'] == 1500
    assert metrics['peak_storage_bytes'] <= 1500


def test_partial_output_allocator_failure_propagates_and_releases_earlier_output(
        cpu_products, monkeypatch):
    models, host, broker = [model(), model(shift=.2)], genotype(), Broker()
    model_refs = [weakref.ref(fitted) for fitted in models]
    refs = []
    original = torch.empty
    def allocate(*args, **kwargs):
        if refs: raise MemoryError('actual allocator refused the second output')
        result = original(*args, **kwargs)
        refs.append(weakref.ref(result))
        return result
    monkeypatch.setattr(torch, 'empty', allocate)
    retained = None
    try:
        blocked_gaussian_covariance_tile(models, host, broker, variant_tile_size=3)
    except MemoryError as error:
        retained = error
    assert retained is not None and refs[0]() is None and not broker.callbacks
    del models, host, broker
    assert all(ref() is None for ref in model_refs)


def test_whole_run_report_has_bounded_recent_calls_and_exact_cumulative_counts(cpu_products):
    models, host, broker, report = [model()], genotype(m=1), Broker(), {}
    for _ in range(20):
        blocked_gaussian_covariance_tile(models, host, broker, variant_tile_size=3,
                                        capacity_bytes=0, report=report)
    incompatible = model(shift=.2)
    incompatible.sample_ids = incompatible.sample_ids[::-1].copy()
    for _ in range(3):
        assert blocked_gaussian_covariance_tile([models[0], incompatible], host, broker,
                                                report=report) is None
    metrics = report['gene_pair_major_covariance']
    assert metrics['calls'] == 23 and metrics['model_count'] == 26
    assert metrics['completed_calls'] == metrics['admitted_calls'] == 20
    assert metrics['declined_calls'] == 3 and metrics['failed_calls'] == 0
    assert metrics['panel_preparations'] == metrics['directed_products'] == 20
    assert metrics['fallback_reasons'] == {'incompatible_exact_axis_rotation_or_state': 3}
    assert len(metrics['recent_calls']) == 8
    assert not broker.callbacks


def test_pressure_does_not_evict_active_shared_pair(cpu_products, monkeypatch):
    models, host, broker = [model(), model(shift=.2)], genotype(), Broker()
    active = []
    original_guard = _PanelPreparation._guard
    def pressure(cache, size):
        if cache.pins:
            before = {key: tuple(value.untyped_storage().data_ptr() for value in cache.panels[key][0])
                      for key in cache.panels if cache.pins.get(key)}
            for callback in tuple(broker.callbacks): callback()
            after = {key: tuple(value.untyped_storage().data_ptr() for value in cache.panels[key][0])
                     for key in before}
            assert before == after
            active.append(len(before))
        return original_guard(cache, size)
    monkeypatch.setattr(_PanelPreparation, '_guard', pressure)
    actual = blocked_gaussian_covariance_tile(models, host, broker,
        variant_tile_size=3, capacity_bytes=2**20)
    for out, fitted in zip(actual, models):
        torch.testing.assert_close(out, covariance(fitted, host, broker, width=3), rtol=0, atol=0)
    assert 2 in active and not broker.callbacks


@pytest.mark.parametrize('failure', ['guard', 'mutation', 'active_borrowed_pair'])
def test_compute_error_propagates_and_closes_owners_even_with_retained_exception(
        cpu_products, monkeypatch, failure):
    models, host, broker, report = [model(), model(shift=.2)], genotype(), Broker(), {}
    refs = [weakref.ref(value) for value in (*models, host, broker, models[0].x, models[1].precision_x)]
    original_borrow = _PanelPreparation.borrow
    seen = []
    @contextmanager
    def track(cache, begin, end):
        values = None
        try:
            with original_borrow(cache, begin, end) as values:
                seen.extend(weakref.ref(value) for value in values)
                if failure == 'mutation': models[1].inverse_variance.add_(.01)
                yield values
        finally:
            values = None
    monkeypatch.setattr(_PanelPreparation, 'borrow', track)
    guard = None
    if failure == 'guard':
        def guard(size):
            if broker.callbacks: raise MemoryError('actual computation admission failure')
        broker._guard = guard
    elif failure == 'active_borrowed_pair':
        original_guard = _PanelPreparation._guard
        def guard(cache, size):
            if cache.__class__.__name__ == '_BorrowedPreparation' and cache.pins:
                raise MemoryError('actual independent projection admission failure')
            return original_guard(cache, size)
        monkeypatch.setattr(_PanelPreparation, '_guard', guard)
    retained = None
    try:
        blocked_gaussian_covariance_tile(models, host, broker, variant_tile_size=3,
                                        capacity_bytes=2**20, report=report)
    except (MemoryError, ValueError) as error:
        retained = error
    assert retained is not None and not broker.callbacks
    metrics = report['gene_pair_major_covariance']['recent_calls'][-1]
    assert metrics['admitted'] and metrics['fallback_reason'] is None
    assert metrics['storage_bytes_after_close'] == metrics['pinned_panels_after_close'] == 0
    # The fixture wrapper deliberately retains its own lexical models. Native
    # preparation objects and tensor returns must still release immediately.
    assert all(ref() is None for ref in seen)
    monkeypatch.undo()
    del models, host, broker, track, guard
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize('count', [0, 5])
def test_invalid_model_tile_has_explicit_bound(count):
    with pytest.raises(ValueError, match='one to four'):
        blocked_gaussian_covariance_tile([model()]*count, genotype(), Broker())


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
                   reason='native TF32 contract requires an Ampere CUDA device')
@pytest.mark.parametrize('fortran', [False, True])
@pytest.mark.parametrize('m', [513, 734])
def test_actual_cuda_preserves_original_directed_products_and_partial_panels(fortran, m):
    models = [model(1536, shift=.21*i, device='cuda:0') for i in range(4)]
    host, broker, report = genotype(1536, m, fortran=fortran), Broker(), {}
    expected = [covariance(fitted, host, broker, width=512, capacity=0) for fitted in models]
    actual = blocked_gaussian_covariance_tile(models, host, broker,
        variant_tile_size=512, capacity_bytes=0, report=report)
    for out, reference in zip(actual, expected):
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
    metrics = report['gene_pair_major_covariance']['recent_calls'][-1]
    assert metrics['panel_preparations'] == 3 and metrics['directed_products'] == 16
    assert metrics['storage_bytes_after_close'] == 0 and not broker.callbacks
