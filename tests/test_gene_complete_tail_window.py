"""Synthetic real orchestration oracles; actual-data throughput is separate."""
from contextlib import contextmanager
from types import SimpleNamespace
import gc
import weakref

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit import _native_tail_groups as grouped
from fudan_wgs_toolkit.pipeline import AnalysisOptions
from fudan_wgs_toolkit.phewas_runtime import runtime
from fudan_wgs_toolkit.phewas_runtime.metadata import SharedMetadataReader
from fudan_wgs_toolkit.phewas_runtime.shared_state import SharedStateBroker
from test_phewas_batched_runtime import Repository, descriptor, model_for, cpu_product_oracle
from test_phewas_shared_state import MetadataReader, make_cache


@pytest.fixture(autouse=True)
def native_cpu_burden(monkeypatch):
    from fudan_wgs_toolkit import _burden
    original = _burden.ieee_burden_product
    # Keep real native CUDA products for the separately skipped component.
    monkeypatch.setattr(_burden, 'ieee_burden_product',
        lambda left, right: original(left, right) if left.is_cuda else left @ right)
    grouped.execution_metadata(reset=True)
    yield
    grouped.execution_metadata(reset=True)


def run_gene(tmp_path, monkeypatch, *, enabled, models_count=8, tile_size=8,
             device='cpu', configure=None, report=None):
    n, m = 256, 6
    raw = np.zeros((m, n), dtype=np.uint8)
    for index in range(m):
        raw[index, index] = 1
    class Reader(MetadataReader):
        def read_field(self, name, indices=None):
            value = np.full(m, 'PASS') if name == 'annotation/filter' else np.arange(m)
            if name == 'chromosome':
                value = np.full(m, '1')
            return value if indices is None else value[indices]
    descriptors = [descriptor(index, np.arange(n)) for index in range(models_count)]
    repository = Repository(descriptors,
        [model_for(np.arange(n), .7+index, device=device) for index in range(models_count)])
    _, cohorts = runtime._descriptor_groups(repository)
    writes, preparation_calls, covariance_refs, pending_refs = [], [], [], []
    report = report if report is not None else {}
    report.update(association_rows={'ncrna': 0},
        trait_rows={index: {'ncrna': 0} for index in range(models_count)},
        binary_spa_selected=0, binary_spa_failed=0)
    original = runtime.LimitedMaskPipeline._evaluate_prepared
    def evaluate(pipeline, payload, model):
        assert repository.active > 0  # Every finalizer/fallback stays in tile lease.
        index = next(i for i, item in enumerate(repository.models) if item is model)
        preparation_calls.append((index, bool(payload.get('_defer_full_tail', False))))
        if configure is not None:
            configure(index, payload, repository, report)
        value = original(pipeline, payload, model)
        if isinstance(value, grouped.PendingCompleteTail):
            assert payload['_defer_full_tail'] is True
            covariance_refs.append(weakref.ref(payload['covariance']))
            pending_refs.extend(weakref.ref(owner) for owner in
                (value.q_values, value.complete_spectra, value.pvalues))
        return value
    monkeypatch.setattr(runtime.LimitedMaskPipeline, '_evaluate_prepared', evaluate)
    monkeypatch.setattr(runtime.LimitedMaskPipeline, '_select_mask_chunks',
        lambda self, *args: {'ncRNA': np.arange(m)})
    if device == 'cpu':
        # The cached long-mask transport is CUDA-only. Keep the fitted-model
        # FP32 covariance formula as this CPU orchestration oracle.
        monkeypatch.setattr(runtime.LimitedMaskPipeline, '_long_mask_products',
            lambda self, model, host, **kwargs: model.score_covariance(torch.from_numpy(host)))
    with SharedStateBroker(SharedMetadataReader(Reader(n, m)),
            make_cache(tmp_path/'cache', raw), device=device) as broker:
        axis = broker.reader_view(np.arange(n))
        try:
            options = AnalysisOptions(wrapper_semantics='base', memory_limit_gib=20)
            configuration = dict(_continuous_tail_groups=enabled)
            structural = runtime._thin_pipeline(axis, repository.models[0], axis._axis.samples,
                options, configuration)
            structural.models = [SimpleNamespace(device=broker.device, use_spa=False)]
            runtime._batched_gene(cohorts, repository, broker, broker._reader, options,
                configuration, dict(gene_name='synthetic_group', start=0, end=5),
                SimpleNamespace(write=lambda *args, **kwargs: writes.append(args)),
                report, set(), '1', 0, tile_size, structural, 'ncrna')
            assert repository.active == 0
        finally:
            axis.close()
    gc.collect()
    assert all(ref() is None for ref in covariance_refs+pending_refs)
    return writes, preparation_calls, report


@pytest.mark.parametrize('tile_size,windows', [(8, 2), (3, 3)])
def test_full_window_keeps_all_original_fields_rows_order_and_tile_leases(tmp_path, monkeypatch,
        cpu_product_oracle, tile_size, windows):
    # Isolate each instrumentation wrapper rather than nesting it in the second run.
    with monkeypatch.context() as patch:
        expected, scalar_calls, control_report = run_gene(tmp_path/'control', patch,
            enabled=False, tile_size=tile_size)
    actual, calls, report = run_gene(tmp_path/'candidate', monkeypatch,
        enabled=True, tile_size=tile_size)
    assert actual == expected and len(actual) == 8
    assert [row[1] for row in actual] == list(range(8))
    assert all(flag for _, flag in calls) and all(not flag for _, flag in scalar_calls)
    assert report['gene_complete_tail_windows'] == windows
    assert report['gene_complete_tail_deferred_models'] == 8
    assert report['gene_complete_tail_requested_groups'] == 8
    assert report['gene_complete_tail_actual_tail_calls'] == windows
    assert report['gene_complete_tail_max_models'] == min(tile_size, 4)
    assert report['gene_score_completed_first_evaluations'] == 8
    assert not any(key.startswith('gene_complete_tail_') for key in control_report)


def test_first_extreme_normal_tail_finishes_prior_window_then_uses_immediate_original_route(
        tmp_path, monkeypatch, cpu_product_oracle):
    original = runtime._gene_normal_score_extreme
    target = -.2*3.7
    monkeypatch.setattr(runtime, '_gene_normal_score_extreme',
        lambda score, covariance: bool(abs(float(score[0])-target) < 1e-6) or original(score, covariance))
    actual, calls, report = run_gene(tmp_path, monkeypatch, enabled=True, models_count=6)
    assert len(actual) == 6
    assert calls == [(0, True), (1, True), (2, True), (3, False), (4, True), (5, True)]
    assert report['gene_complete_tail_normal_immediate_evaluations'] == 1
    assert report['gene_complete_tail_windows'] == 2
    assert report['gene_complete_tail_deferred_models'] == 5


def test_gene_extreme_completed_tail_replays_original_score_with_flag_free_payload(
        tmp_path, monkeypatch, cpu_product_oracle):
    from fudan_wgs_toolkit.phewas_runtime import scores
    sequence = iter(range(100))
    monkeypatch.setattr(scores, 'execution_metadata',
        lambda: dict(fp32_simt_products=next(sequence)))
    original = grouped.finish_complete_tails
    first = True
    def finalizer(pendings, **kwargs):
        nonlocal first
        result = original(pendings, **kwargs)
        if first:
            result[1]['WGS-O'] = 0.  # Explicit synthetic sensitive-tail trigger.
            first = False
        return result
    monkeypatch.setattr(grouped, 'finish_complete_tails', finalizer)
    actual, calls, report = run_gene(tmp_path, monkeypatch, enabled=True, models_count=4)
    assert len(actual) == 4
    assert calls == [(0, True), (1, True), (2, True), (3, True), (1, False)]
    assert report['gene_score_original_fallback_gene_tail'] == 1
    assert report['gene_score_additional_tail_evaluations'] == 1
    assert report['gene_score_completed_first_evaluations'] == 4


def test_fresh_admission_failure_flushes_live_owners_then_retries_covariance(
        tmp_path, monkeypatch, cpu_product_oracle):
    report, phases, failed, covariance_refs = {}, [], False, []
    class Workspace:
        def set_required(self, required, *, phase='workspace'):
            nonlocal failed
            phases.append((phase, required))
            if (phase == 'gene_covariance' and report.get('gene_complete_tail_deferred_models') == 2
                    and not failed):
                failed = True
                raise MemoryError('synthetic live admission refusal with two retained C owners')
            if phase == 'gene_covariance' and failed and len(covariance_refs) == 2:
                assert all(ref() is None for ref in covariance_refs)
        def ensure_free(self, *args, **kwargs):
            pass
    @contextmanager
    def scope(*args, **kwargs):
        yield Workspace()
    monkeypatch.setattr(runtime, '_gene_workspace_scope', scope)
    def configure(index, payload, repository, report):
        covariance_refs.append(weakref.ref(payload['covariance']))
    actual, calls, result = run_gene(tmp_path, monkeypatch, enabled=True,
        models_count=4, report=report, configure=configure)
    assert failed and len(actual) == 4 and len(calls) == 4
    assert result['gene_complete_tail_admission_flushes'] == 1
    assert result['gene_complete_tail_windows'] == 2
    assert result['gene_complete_tail_max_models'] == 2
    assert any(phase == 'gene_complete_tail_batch' for phase, _ in phases)
    assert len({required for phase, required in phases if phase == 'gene_covariance'}) == 1


def test_prepare_admission_failure_flushes_and_retries_same_model_without_missing_rows(
        tmp_path, monkeypatch, cpu_product_oracle):
    failed = False
    def configure(index, payload, repository, report):
        nonlocal failed
        if index == 2 and payload.get('_defer_full_tail') and not failed:
            failed = True
            raise MemoryError('synthetic NEW spectrum admission refused')
    actual, calls, report = run_gene(tmp_path, monkeypatch, enabled=True,
        models_count=4, configure=configure)
    assert len(actual) == 4 and calls == [(0, True), (1, True), (2, True), (2, True), (3, True)]
    assert report['gene_complete_tail_prepare_flushes'] == 1
    assert report['gene_complete_tail_requested_groups'] == 4


def test_numerical_error_preserved_and_queued_owners_released_without_finalizing_later_models(
        tmp_path, monkeypatch, cpu_product_oracle):
    covariance_refs, spectrum_refs, repositories = [], [], []
    original = grouped.PendingCompleteTail.__init__
    def constructor(pending, *args, **kwargs):
        original(pending, *args, **kwargs)
        spectrum_refs.append(weakref.ref(pending.complete_spectra))
    monkeypatch.setattr(grouped.PendingCompleteTail, '__init__', constructor)
    def configure(index, payload, repository, report):
        covariance_refs.append(weakref.ref(payload['covariance']))
        repositories.append(repository)
        if index == 2:
            raise ArithmeticError('synthetic scientific preparation failure')
    with pytest.raises(ArithmeticError, match='scientific preparation') as captured:
        run_gene(tmp_path, monkeypatch, enabled=True, configure=configure)
    assert captured.value.__traceback__ is not None
    gc.collect()
    assert all(repository.active == 0 for repository in repositories)
    assert len(spectrum_refs) == 2 and len(covariance_refs) == 3
    assert all(ref() is None for ref in spectrum_refs+covariance_refs)


def test_finalization_failure_is_real_and_releases_window_without_output_or_model_lease(
        tmp_path, monkeypatch, cpu_product_oracle):
    covariance_refs, spectrum_refs, repositories, report = [], [], [], {}
    original = grouped.PendingCompleteTail.__init__
    def constructor(pending, *args, **kwargs):
        original(pending, *args, **kwargs)
        spectrum_refs.append(weakref.ref(pending.complete_spectra))
    def configure(index, payload, repository, report):
        covariance_refs.append(weakref.ref(payload['covariance']))
        repositories.append(repository)
    def failed(*args, **kwargs):
        raise ArithmeticError('synthetic final-tail failure')
    monkeypatch.setattr(grouped.PendingCompleteTail, '__init__', constructor)
    monkeypatch.setattr(grouped, 'finish_complete_tails', failed)
    with pytest.raises(ArithmeticError, match='final-tail') as captured:
        run_gene(tmp_path, monkeypatch, enabled=True, configure=configure, report=report)
    assert captured.value.__traceback__ is not None
    gc.collect()
    assert len(spectrum_refs) == len(covariance_refs) == 4
    assert all(repository.active == 0 for repository in repositories)
    assert all(ref() is None for ref in spectrum_refs+covariance_refs)
    assert report['association_rows']['ncrna'] == 0


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
                   reason='actual CUDA full-window field precision and original products')
def test_cuda_native_window_complete_probability_coverage_and_strict_log_error(tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        expected, _, _ = run_gene(tmp_path/'control', patch, enabled=False, device='cuda:0', models_count=4)
    actual, _, report = run_gene(tmp_path/'candidate', monkeypatch, enabled=True,
        device='cuda:0', models_count=4)
    for control, value in zip(expected, actual):
        left, right = control[3][0], value[3][0]
        assert list(left) == list(right)
        for name in left:
            if name.startswith(('SKAT(', 'Burden(', 'ACAT-', 'STAAR-')):
                assert 0 < left[name] <= 1 and 0 < right[name] <= 1
                assert abs(np.log10(left[name])-np.log10(right[name])) < .001
            else:
                assert left[name] == right[name]
    assert report['gene_complete_tail_max_models'] == 4
    assert report['gene_complete_tail_merged_tail_calls'] == 1
