"""Live resident admission contracts; CPU route markers are not GPU evidence."""
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.pipeline import AnalysisOptions
from fudan_wgs_toolkit.phewas_runtime import runtime
from test_phewas_device_gene_prepare import fixture_pipeline, assert_same_preparation
from test_phewas_batched_runtime import Repository, descriptor, cpu_product_oracle


def routed_broker(broker, events, guard):
    """Only mark admission device; mature six-state math stays on the CPU."""
    def read(indices):
        events.append(('read', len(indices)))
        return broker.read_states(indices)
    return SimpleNamespace(device='cuda:0', _guard=guard,
                           read_states=read, trait_block=broker.trait_block)


def allow_cpu_output(monkeypatch):
    original = torch.empty
    def empty(*args, **kwargs):
        if 'device' in kwargs and torch.device(kwargs['device']).type == 'cuda':
            kwargs['device'] = 'cpu'
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, 'empty', empty)


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('imputation', ['mean', 'minor'])
@pytest.mark.parametrize('cohort', ['reversed', 'subset'])
def test_explicit_flag_preserves_all_original_qc_weights_order_and_fill_fields(
        tmp_path, monkeypatch, enabled, imputation, cohort):
    rows = np.arange(512)[::-1].copy() if cohort == 'reversed' else np.arange(128)[::-1].copy()
    indices = np.asarray([5, 2, 9, 4, 1, 6, 0, 8, 3, 7, 10, 11], dtype=np.int64)
    with fixture_pipeline(tmp_path/'host', device='cpu', rows=rows, imputation=imputation) as (p, broker):
        expected = runtime._prepare_shared_gene(p, indices, broker)
    with fixture_pipeline(tmp_path/'candidate', device='cpu', rows=rows, imputation=imputation) as (p, broker):
        p.options = replace(p.options, long_mask_threshold=2, long_mask_method='full')
        p._continuous_live_resident = enabled
        events = []
        def guard(required):
            assert not events  # Admission happens before read_states or output allocation.
            events.append(('guard', required))
        routed = routed_broker(broker, events, guard)
        allow_cpu_output(monkeypatch)
        actual = runtime._prepare_shared_gene(p, indices, routed)
        if enabled:
            assert_same_preparation(actual, expected)
            assert events[0] == ('guard', 12*len(rows)*len(indices)+32*len(indices)**2+256*2**20)
            assert '_genotype_host' not in actual
        else:
            assert events[0][0] == 'read' and all(kind != 'guard' for kind, _ in events)
            assert '_genotype_device' not in actual
            for key in expected:
                if isinstance(expected[key], np.ndarray):
                    np.testing.assert_array_equal(actual[key], expected[key])
                else: assert actual[key] == expected[key]
        assert p.options.long_mask_threshold == 2 and p.options.long_mask_method == 'full'
        assert len(actual['maf']) > p.options.long_mask_threshold
        assert not p.skipped_sets  # A route threshold is never an eligible-mask filter.


@pytest.mark.parametrize('error_type', [MemoryError, torch.cuda.OutOfMemoryError])
def test_long_mask_preallocation_admission_refusal_runs_full_host_preparation(
        tmp_path, monkeypatch, error_type):
    rows, indices = np.arange(512), np.arange(12)
    with fixture_pipeline(tmp_path/'cache', device='cpu', rows=rows) as (p, broker):
        expected = runtime._prepare_shared_gene(p, indices, broker)
        p.options = replace(p.options, long_mask_threshold=2, long_mask_method='full')
        p._continuous_live_resident = True
        events, host_calls = [], []
        def refuse(required):
            assert not events
            events.append(('guard', required))
            raise error_type('optional live resident admission refused')
        routed = routed_broker(broker, events, refuse)
        original_empty = torch.empty
        def no_device_output(*args, **kwargs):
            if 'device' in kwargs and torch.device(kwargs['device']).type == 'cuda':
                pytest.fail('refused path allocated device output')
            return original_empty(*args, **kwargs)
        monkeypatch.setattr(torch, 'empty', no_device_output)
        p.host_memory_guard = lambda **request: host_calls.append(request)
        p.host_memory_set_future = lambda *args, **kwargs: None
        actual = runtime._prepare_shared_gene(p, indices, routed)
        assert '_genotype_device' not in actual and events[0][0] == 'guard'
        np.testing.assert_array_equal(actual['_genotype_host'], expected['_genotype_host'])
        for key in ('maf', 'mac', '_variant_indices', '_extraction_groups', 'annotations'):
            np.testing.assert_array_equal(actual[key], expected[key])
        assert actual['names'] == expected['names'] and actual['cmac'] == expected['cmac']
        assert host_calls[-1]['materialized_bytes'] >= actual['_genotype_host'].nbytes
        assert not p.skipped_sets


def test_actual_device_helper_midallocation_exception_releases_parts_before_host_read(
        tmp_path, monkeypatch):
    rows, indices = np.arange(512), np.arange(12)
    with fixture_pipeline(tmp_path/'cache', device='cpu', rows=rows) as (p, broker):
        expected = runtime._prepare_shared_gene(p, indices, broker)
        p.options = replace(p.options, long_mask_threshold=2)
        p._continuous_live_resident = True
        temporaries, events, reading_host, failed = [], [], [], []
        original_block, original_empty = broker.trait_block, torch.empty
        def block(*args, **kwargs):
            value = original_block(*args, **kwargs)
            if not failed: temporaries.append(weakref.ref(value))
            return value
        def empty(*args, **kwargs):
            if 'device' in kwargs and torch.device(kwargs['device']).type == 'cuda':
                temporary = original_empty((16,), dtype=torch.float32)
                temporaries.append(weakref.ref(temporary))
                failed.append(True)
                raise torch.cuda.OutOfMemoryError('actual optional output allocation failed')
            return original_empty(*args, **kwargs)
        monkeypatch.setattr(torch, 'empty', empty)
        routed = routed_broker(broker, events, lambda required: events.append(('guard', required)))
        routed.trait_block = block
        def host_guard(**request):
            assert temporaries and all(reference() is None for reference in temporaries)
            reading_host.append(True)
        p.host_memory_guard = host_guard
        p.host_memory_set_future = lambda *args, **kwargs: None
        actual = runtime._prepare_shared_gene(p, indices, routed)
        np.testing.assert_array_equal(actual['_genotype_host'], expected['_genotype_host'])
        assert all(reference() is None for reference in temporaries)
        assert reading_host and not p.skipped_sets


@pytest.mark.parametrize('enabled', [False, True])
def test_thin_pipeline_propagates_only_boolean_execution_flag(tmp_path, enabled):
    with fixture_pipeline(tmp_path/'cache', device='cpu', rows=np.arange(512)) as (p, broker):
        candidate = runtime._thin_pipeline(p.genotype, p.models[0], p.union_rows, p.options,
            {'_continuous_live_resident': enabled, 'annotation_names': p.annotation_names}, p)
        assert candidate._continuous_live_resident is enabled
        assert candidate.options is p.options
        assert candidate.options.long_mask_method == 'full'
        assert candidate.options.variant_type == p.options.variant_type


@pytest.mark.parametrize('enabled', [None, 1, 'true'])
def test_thin_pipeline_rejects_ambiguous_flag_values(tmp_path, enabled):
    with fixture_pipeline(tmp_path/'cache', device='cpu', rows=np.arange(512)) as (p, broker):
        with pytest.raises(ValueError, match='must be a boolean'):
            runtime._thin_pipeline(p.genotype, p.models[0], p.union_rows, p.options,
                                   {'_continuous_live_resident': enabled}, p)


def test_binary_only_long_mask_keeps_original_host_route_with_flag(tmp_path, monkeypatch):
    with fixture_pipeline(tmp_path/'cache', device='cpu', rows=np.arange(512)) as (p, broker):
        p.options = replace(p.options, long_mask_threshold=2)
        p._continuous_live_resident = True
        p._binary_burden_only = True
        monkeypatch.setattr(runtime, '_prepare_shared_gene_device', lambda *a, **k:
                            pytest.fail('Gaussian candidate broadened the binary device route'))
        routed = routed_broker(broker, [], lambda required: None)
        actual = runtime._prepare_shared_gene(p, np.arange(12), routed)
        assert '_genotype_device' not in actual and not p.skipped_sets


class _CudaBranchMarker(torch.Tensor):
    @property
    def is_cuda(self):
        return True


@pytest.mark.parametrize('route', ['flag_false', 'live_admitted', 'live_refused',
                                   'upload_oom', 'post_upload_refused',
                                   'covariance_oom', 'covariance_refused'])
def test_real_gene_dispatch_upload_branch_reserves_new_bytes_and_keeps_complete_output(
        tmp_path, monkeypatch, cpu_product_oracle, route):
    phases, copies, future, writes, tensor_refs, covariances = [], [], [], [], [], []
    with fixture_pipeline(tmp_path/'cache', device='cpu', rows=np.arange(512)) as (p, broker):
        options = replace(p.options, long_mask_threshold=2, long_mask_method='full', variant_tile_size=3)
        repository = Repository([descriptor(0, np.arange(512))], p.models)
        _, groups = runtime._descriptor_groups(repository)
        class CudaRoute:
            device = torch.device('cuda:0')
            def __getattr__(self, name): return getattr(broker, name)
        routed = CudaRoute()
        real_prepare, real_tensor = runtime._prepare_shared_gene, torch.as_tensor
        real_required = runtime._DirectWorkspace.set_required
        real_covariance, real_blocked = runtime._gaussian_covariance, runtime._blocked_gaussian_covariance
        def prepare(pipeline, indices, current):
            # This test isolates the second upload choice: first preparation
            # uses the authentic CPU broker and every scientific filter.
            prepared = real_prepare(pipeline, indices, broker)
            tensor_refs.append(weakref.ref(prepared['_genotype_host']))
            return prepared
        def as_tensor(value, *args, **kwargs):
            if 'device' in kwargs and torch.device(kwargs['device']).type == 'cuda':
                assert phases[-1][1] == 'gene_genotype_upload'
                copies.append(tuple(value.shape))
                kwargs['device'] = 'cpu'
                if route == 'upload_oom':
                    temporary = torch.ones(16)
                    tensor_refs.append(weakref.ref(temporary))
                    raise torch.cuda.OutOfMemoryError('actual optional upload failed')
                result = real_tensor(value, *args, **kwargs).as_subclass(_CudaBranchMarker)
                tensor_refs.append(weakref.ref(result))
                return result
            return real_tensor(value, *args, **kwargs)
        def set_required(workspace, required, *, phase='workspace'):
            phases.append((required, phase))
            if route == 'live_refused' and phase == 'gene_genotype_upload':
                raise MemoryError('optional live NEW covariance/G upload rejected')
            if route == 'post_upload_refused' and phase == 'gene_genotype_resident':
                raise MemoryError('actual post-upload covariance headroom refused')
            return real_required(workspace, required, phase=phase)
        def scores(genotype, models, current):
            return torch.stack(tuple(real_tensor(genotype).T@model.scaled_residuals for model in models), 1)
        def covariance(model, genotype, **kwargs):
            assert route in ('live_admitted','covariance_oom','covariance_refused') and genotype.is_cuda
            if route in ('covariance_oom','covariance_refused'):
                temporary=torch.ones(16)
                tensor_refs.append(weakref.ref(temporary))
                error_type=(torch.cuda.OutOfMemoryError if route=='covariance_oom' else MemoryError)
                raise error_type('actual resident covariance fixture failure')
            result = real_covariance(model, genotype, **kwargs)
            covariances.append(result.detach().clone())
            return result
        def blocked(model, host, current, **kwargs):
            assert route != 'live_admitted' and isinstance(host, np.ndarray)
            result = real_blocked(model, host, broker, **kwargs)
            covariances.append(result.detach().clone())
            return result
        monkeypatch.setattr(runtime, '_prepare_shared_gene', prepare)
        monkeypatch.setattr(torch, 'as_tensor', as_tensor)
        monkeypatch.setattr(runtime._DirectWorkspace, 'set_required', set_required)
        monkeypatch.setattr(runtime, '_shared_gene_scores', scores)
        monkeypatch.setattr(runtime, '_gaussian_covariance', covariance)
        monkeypatch.setattr(runtime, '_blocked_gaussian_covariance', blocked)
        monkeypatch.setattr(runtime.LimitedMaskPipeline, '_select_mask_chunks',
                            lambda *args: {'ncRNA': np.arange(12)})
        monkeypatch.setattr(runtime.LimitedMaskPipeline, '_supports_long_tf32_products', lambda *args: False)
        monkeypatch.setattr(runtime.LimitedMaskPipeline, '_evaluate_prepared',
            lambda self, payload, model: dict(num_variant=len(payload['maf']), cMAC=payload['cmac'], **{'WGS-O': .25}))
        config = dict(_continuous_live_resident=route != 'flag_false', annotation_names=p.annotation_names,
            _host_memory_set_future=lambda amount, **kwargs: future.append((amount, kwargs['phase'])))
        structural = runtime._thin_pipeline(p.genotype, p.models[0], p.union_rows, options, config)
        structural.annotations = p.annotations
        report = dict(association_rows={'ncrna': 0}, trait_rows={0: {'ncrna': 0}},
                      binary_spa_selected=0, binary_spa_failed=0)
        runtime._batched_gene(groups, repository, routed, broker._reader, options, config,
            dict(gene_name='anonymous_fixture', start=0, end=11),
            SimpleNamespace(write=lambda *args, **kwargs: writes.append(args)), report,
            set(), '21', 0, 1, structural, 'ncrna')
        assert len(writes) == 1 and report['association_rows']['ncrna'] == 1
        assert len(covariances) == 1 and options.long_mask_method == 'full'
        assert all(reference() is None for reference in tensor_refs)
        assert future[-1] == (0, 'gene_mask_released') and repository.active == 0
        if route == 'flag_false':
            assert not copies and all(phase != 'gene_genotype_upload' for _, phase in phases)
        elif route in ('covariance_oom','covariance_refused'):
            assert len(copies)==1 and report['gene_resident_covariance_host_fallbacks']==1
            assert report['gene_resident_covariance_fallback_mask_retained'] is True
            assert any(phase=='gene_resident_covariance_host_fallback' for _,phase in phases)
            assert report['gene_resident_covariance_last_fallback_geometry']['m']==copies[0][1]
        elif route != 'live_admitted':
            assert len(copies) == (0 if route == 'live_refused' else 1)
            assert any(phase == 'gene_host_fallback' for _, phase in phases)
            assert report['gene_resident_upload_fallbacks'] == 1
        else:
            assert len(copies) == 1 and copies[0][1] > options.long_mask_threshold
            n, m = copies[0]
            covariance_new = runtime._gene_resident_covariance_new_bytes(n,m,p.models[0].x.shape[1],
                has_blocks=bool(p.models[0].spectrum.blocks),
                maximum_family_rows=max((len(rows) for rows,_ in p.models[0].spectrum.blocks),default=0))
            assert next(required for required, phase in phases if phase == 'gene_genotype_upload') >= covariance_new+4*n*m
            assert next(required for required, phase in phases if phase == 'gene_genotype_resident') >= covariance_new


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual long resident preparation requires CUDA')
@pytest.mark.parametrize('cohort', ['reversed', 'subset'])
def test_actual_cuda_explicit_long_route_matches_same_full_host_mask_fields(tmp_path, cohort):
    rows = np.arange(512)[::-1].copy() if cohort == 'reversed' else np.arange(128)[::-1].copy()
    indices = np.asarray([5, 2, 9, 4, 1, 6, 0, 8, 3, 7, 10, 11], dtype=np.int64)
    with fixture_pipeline(tmp_path/'host', device='cpu', rows=rows) as (p, broker):
        expected = runtime._prepare_shared_gene(p, indices, broker)
    with fixture_pipeline(tmp_path/'device', device='cuda:0', rows=rows) as (p, broker):
        p.options = replace(p.options, long_mask_threshold=2, long_mask_method='full')
        p._continuous_live_resident = True
        actual = runtime._prepare_shared_gene(p, indices, broker)
        assert_same_preparation(actual, expected)
        assert len(actual['maf']) > p.options.long_mask_threshold and not p.skipped_sets
