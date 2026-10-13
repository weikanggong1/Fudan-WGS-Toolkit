"""Streaming contracts and CPU oracles; no claim of real-data performance."""
from contextlib import contextmanager
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.binary import (ExactBinaryBurdenCache, _BinaryPanelSource,
    prepare_binary_burdens, staar_binary_phewas)
from fudan_wgs_toolkit.binary_gene_batch import BinaryHostPanelSource
from fudan_wgs_toolkit.genotype_device import DeviceMinorBlock
from fudan_wgs_toolkit.phewas_models import prepare_phewas_model
from fudan_wgs_toolkit.phewas_runtime import runtime
from fudan_wgs_toolkit.phewas_resources import tensor_storage_footprint
from fudan_wgs_toolkit.phewas_runtime.shared_state import SharedStateBroker
from test_phewas_device_gene_prepare import fixture_pipeline
from test_phewas_batched_runtime import Repository, descriptor, model_for, cpu_product_oracle
from test_phewas_shared_state import MetadataReader, make_cache


@pytest.mark.parametrize('imputation', ['mean', 'minor'])
@pytest.mark.parametrize('cohort', ['reversed', 'subset'])
def test_streaming_recipe_preserves_filter_sort_af_half_calls_and_fp32(tmp_path, imputation, cohort):
    rows = np.arange(512)[::-1].copy() if cohort == 'reversed' else np.arange(128)[::-1].copy()
    indices = np.asarray([5, 2, 9, 4, 1, 6, 0, 8, 3, 7, 10, 11], dtype=np.int64)
    with fixture_pipeline(tmp_path/'cache', device='cpu', rows=rows, imputation=imputation) as (p, broker):
        expected = runtime._prepare_shared_gene(p, indices, broker)
        p._binary_burden_only = True
        calls = []
        p.host_memory_guard = lambda **request: calls.append(request)
        actual = runtime._prepare_shared_gene(p, indices, broker)
        source = actual['_genotype_host']
        assert isinstance(source, BinaryHostPanelSource)
        for name in ('maf', 'mac', 'annotations', '_variant_indices', '_extraction_groups'):
            np.testing.assert_array_equal(actual[name], expected[name])
        dense = source.read_columns(np.arange(source.shape[1]))
        assert dense.stride() == (1, len(rows))
        np.testing.assert_array_equal(dense.numpy(), expected['_genotype_host'])
        subset = np.asarray([source.shape[1]-1, 0, 2])
        np.testing.assert_array_equal(source.read_columns(subset).numpy(),
                                      expected['_genotype_host'][:, subset])
        assert max(item['m'] for item in calls) <= 512
        assert actual['cmac'] is None  # Deferred to the original burden panel fold.
        with pytest.raises(ValueError):
            source.variants.flags.writeable = True
        with pytest.raises(ValueError):
            source.expected_maf.flags.writeable = True
        p.genotype.close()
        with pytest.raises(ValueError, match='binding'):
            source.read_columns(np.arange(source.shape[1]))


@contextmanager
def many_variant_pipeline(path, m=1025):
    n = 512
    raw = np.zeros((m, n), dtype=np.uint8)
    raw[np.arange(m), np.arange(m) % n] = 1
    raw[1, 2:7] = 3
    raw[2, 11:13] = [4, 5]
    with SharedStateBroker(MetadataReader(n, m), make_cache(path, raw), device='cpu') as broker:
        with broker.reader_view(np.arange(n)) as reader:
            model = model_for(reader._axis.samples, device='cpu')
            options = runtime.AnalysisOptions(wrapper_semantics='base', genotype_block_size=700)
            p = runtime._thin_pipeline(reader, model, reader._axis.samples, options,
                                      dict(annotation_names=[]))
            p.annotations = lambda selected, **kwargs: SimpleNamespace(annotations={})
            yield p, broker


def test_recipe_storage_diagnostics_count_direct_arrays_without_reading_panels(tmp_path, monkeypatch):
    with many_variant_pipeline(tmp_path/'cache') as (p, broker):
        p._binary_burden_only = True
        source = runtime._prepare_shared_gene(p, np.arange(1025), broker)['_genotype_host']
        monkeypatch.setattr(broker, 'read_states', lambda *args: pytest.fail('telemetry read genotype'))
        arrays = source.explicit_storage_owners()
        assert list(arrays) == ['samples', 'identity', 'variants', 'expected_maf', 'sample_ids']
        for name, owner in arrays.items():
            assert owner is getattr(source, name) and owner.ndim == 1
        expected = tensor_storage_footprint(arrays)
        # Duplicate recipe/array aliases count backing storage once. The
        # scalar observation describes logical geometry, not matrix bytes.
        actual = runtime._workspace_storage_diagnostics(
            dict(genotype=source, aliases=[source, arrays, source.samples]))
        for name in ('retained_storage_bytes', 'retained_storage_bytes_by_device',
                     'unique_cuda_storage_bytes'):
            assert actual[name] == expected[name]
        assert actual['retained_storage_bytes'] > 0
        assert actual['retained_storage_bytes'] < 4*np.prod(source.shape)
        assert actual['host_panel_sources'] == [dict(source_type='BinaryHostPanelSource',
            logical_shape=[512, 1025], maximum_panel_width=512,
            whole_matrix_materialized=False, storage_owner_scope='direct recipe arrays only',
            retained_array_names=list(arrays))]
        # The general storage counter remains strict, and its adapter must
        # never swallow a new unknown owner's TypeError or report zero bytes.
        with pytest.raises(TypeError, match='explicit tensors'):
            tensor_storage_footprint(source)
        with pytest.raises(TypeError, match='explicit tensors'):
            runtime._workspace_storage_diagnostics(dict(genotype=object()))
        x = torch.ones(3)
        assert runtime._workspace_storage_diagnostics(dict(x=x)) == tensor_storage_footprint(dict(x=x))


def test_binary_gene_runtime_memory_sample_accepts_streamed_prepared_source(
        tmp_path, monkeypatch, cpu_product_oracle):
    """Actual broker/runtime CPU integration; fixture data is synthetic."""
    with many_variant_pipeline(tmp_path/'cache') as (structural, broker):
        n = len(structural.union_rows)
        ids = broker._reader.sample_ids()
        fitted = prepare_phewas_model((np.arange(n) % 7 == 0).astype(float), ids,
            np.column_stack((np.ones(n), np.linspace(-1, 1, n))), association_mode='fp64')
        fitted.model.set_matmul_mode('tf32')
        ds = [descriptor(0, np.arange(n), 'binomial')]
        class BinaryRepository(Repository):
            @contextmanager
            def acquire_spa(self, index, device):
                assert index == 0 and device == 'cpu'
                yield fitted.spa_model
        repository = BinaryRepository(ds, [fitted.model])
        _, groups = runtime._descriptor_groups(repository)
        structural.models = [SimpleNamespace(device=broker.device, use_spa=False)]
        monkeypatch.setattr(runtime.LimitedMaskPipeline, '_select_mask_chunks',
                            lambda self, *args: {'ncRNA': np.arange(1025)})
        samples, writes = [], []
        def memory_sample(phase, **owners):
            assert phase == 'gene_prepared'
            assert isinstance(owners['genotype'], BinaryHostPanelSource)
            # Same explicit adapter used by the production telemetry closure.
            samples.append(runtime._workspace_storage_diagnostics(owners))
        configuration = dict(annotation_names=[], _memory_sample=memory_sample)
        report = dict(association_rows={'ncrna': 0}, trait_rows={0: {'ncrna': 0}},
                      binary_spa_selected=0, binary_spa_failed=0)
        runtime._batched_gene(groups, repository, broker, broker._reader, structural.options,
            configuration, dict(gene_name='fixture_gene', start=0, end=1024),
            SimpleNamespace(write=lambda *args, **kwargs: writes.append(args)),
            report, set(), '21', 0, 1, structural, 'ncrna')
        assert len(samples) == len(writes) == 1 and repository.active == 0
        assert samples[0]['retained_storage_bytes'] > 0
        assert samples[0]['unique_cuda_storage_bytes'] == 0
        assert samples[0]['host_panel_sources'][0]['logical_shape'] == [512, 1025]
        assert report['association_rows']['ncrna'] == 1
        assert writes[0][3][0]['#SNV'] == 1025
        assert np.isfinite(writes[0][3][0]['STAAR-B'])


def test_real_recipe_reads_only_bounded_panels_and_preserves_original_burden_fold(tmp_path, monkeypatch):
    with many_variant_pipeline(tmp_path/'cache') as (p, broker):
        indices = np.arange(1025)
        original = runtime._prepare_shared_gene(p, indices, broker)
        p._binary_burden_only = True
        streamed = runtime._prepare_shared_gene(p, indices, broker)
        source = streamed['_genotype_host']
        n = source.shape[0]
        fitted = prepare_phewas_model((np.arange(n) % 7 == 0).astype(float),
            source.sample_ids, np.column_stack((np.ones(n), np.linspace(-1, 1, n))),
            association_mode='fp64')
        expected = prepare_binary_burdens(fitted.model, original['_genotype_host'], original['maf'])
        reads, refs = [], []
        mature = BinaryHostPanelSource.read_columns
        def read(owner, columns):
            assert not refs or refs[-1]() is None
            result = mature(owner, columns)
            reads.append(len(columns))
            refs.append(weakref.ref(result))
            return result
        monkeypatch.setattr(BinaryHostPanelSource, 'read_columns', read)
        actual = prepare_binary_burdens(fitted.model, source, streamed['maf'])
        assert reads == [512, 512, 1]
        torch.testing.assert_close(actual.burdens, expected.burdens, rtol=0, atol=0)
        assert actual.cmac == expected.cmac
        selected = torch.tensor([0, 1])
        cache = ExactBinaryBurdenCache(actual)
        reference_cache = ExactBinaryBurdenCache(expected)
        torch.testing.assert_close(cache.get(actual, selected), reference_cache.get(expected, selected),
                                   rtol=0, atol=0)
        first = staar_binary_phewas(fitted.model, source, streamed['maf'],
            prepared_burdens=actual, spa_model=fitted.spa_model,
            _exact_burden_cache=cache, p_filter_cutoff=.8)
        second = staar_binary_phewas(fitted.model, original['_genotype_host'], original['maf'],
            prepared_burdens=expected, spa_model=fitted.spa_model, p_filter_cutoff=.8)
        assert first == second


def test_panel_preflight_rejects_before_reader_or_output_without_resetting_retired_lease(tmp_path, monkeypatch):
    with fixture_pipeline(tmp_path/'cache', device='cpu', rows=np.arange(512)) as (p, broker):
        p._binary_burden_only = True
        source = runtime._prepare_shared_gene(p, np.arange(12), broker)['_genotype_host']
        calls = []
        def reject(width, materialized):
            raise MemoryError('no host panel allowance')
        object.__setattr__(source, 'host_guard', reject)
        object.__setattr__(source, 'host_ready', lambda: calls.append('ready'))
        monkeypatch.setattr(broker, 'read_states', lambda *args: pytest.fail('read before admission'))
        with pytest.raises(MemoryError, match='allowance'):
            source.read_columns(np.arange(2))
        assert calls == []


def test_second_panel_admission_preserves_retired_host_gate_error(tmp_path):
    with fixture_pipeline(tmp_path/'cache', device='cpu', rows=np.arange(512)) as (p, broker):
        p._binary_burden_only = True
        source = runtime._prepare_shared_gene(p, np.arange(12), broker)['_genotype_host']
        calls = []
        def guard(width, materialized):
            calls.append(materialized)
            if materialized:
                raise MemoryError('retired after compact allocation')
        def forbidden_reset():
            pytest.fail('cannot reset a retired host gate')
        object.__setattr__(source, 'host_guard', guard)
        object.__setattr__(source, 'host_ready', forbidden_reset)
        with pytest.raises(MemoryError, match='retired'):
            source.read_columns(np.arange(2))
        assert calls == [0, 1024]


def test_recipe_rejects_changed_axis_maf_and_oversized_reads(tmp_path):
    with fixture_pipeline(tmp_path/'cache', device='cpu', rows=np.arange(512)) as (p, broker):
        p._binary_burden_only = True
        source = runtime._prepare_shared_gene(p, np.arange(12), broker)['_genotype_host']
        with pytest.raises(ValueError, match='bounded'):
            source.read_columns(np.zeros(513, dtype=np.int64))
        source_maf = source.expected_maf
        object.__setattr__(source, 'expected_maf', source_maf.copy())
        with pytest.raises(ValueError, match='immutable'):
            source.validate()
        object.__setattr__(source, 'expected_maf', source_maf)
        owner = source._source_owner
        object.__setattr__(source, '_source_owner', (None, None, None))
        with pytest.raises(ValueError, match='source'):
            source.validate()
        object.__setattr__(source, '_source_owner', owner)
        with pytest.raises(ValueError, match='bounded'):
            source.read_columns(np.asarray([-1]))


def test_panel_conversion_failure_releases_traceback_owned_output(tmp_path, monkeypatch):
    with fixture_pipeline(tmp_path/'cache', device='cpu', rows=np.arange(512)) as (p, broker):
        p._binary_burden_only = True
        source = runtime._prepare_shared_gene(p, np.arange(12), broker)['_genotype_host']
        owners = []
        def fail(block, *args, **kwargs):
            output = kwargs['out']
            owners.append(weakref.ref(output))
            raise MemoryError('conversion failure')
        monkeypatch.setattr(DeviceMinorBlock, 'trait_dense_columns', fail)
        with pytest.raises(MemoryError, match='conversion') as caught:
            source.read_columns(np.arange(2))
        assert caught.value.__traceback__ is not None
        assert owners[0]() is None


class ContractPanelSource(_BinaryPanelSource):
    """Synthetic allocation geometry/oracle only, no IO benchmark."""
    def __init__(self, g, sample_ids):
        self.g, self.sample_ids, self.shape = g, sample_ids, g.shape
        self.reads = []
    def validate(self):
        return (id(self), self.g._version)
    def read_columns(self, columns):
        assert len(columns) <= 512
        self.reads.append(len(columns))
        return self.g[:, torch.tensor(columns)].contiguous().T.contiguous().T


@pytest.mark.parametrize('m', [2, 31, 511, 512, 513, 4999, 5000, 5001, 15001, 50000, 100000])
def test_all_lengths_streamed_burdens_and_exact_cache_keep_panel_order(m):
    n = 64
    fitted = prepare_phewas_model((np.arange(n) % 7 == 0).astype(float), np.arange(n).astype(str),
        np.column_stack((np.ones(n), np.linspace(-1, 1, n))), association_mode='fp64')
    g = torch.zeros((m, n), dtype=torch.float32).T
    g[torch.arange(m) % n, torch.arange(m)] = 1
    f = np.linspace(.001, .009, m)
    source = ContractPanelSource(g, fitted.model.sample_ids)
    expected = prepare_binary_burdens(fitted.model, g, f)
    actual = prepare_binary_burdens(fitted.model, source, f)
    torch.testing.assert_close(actual.burdens, expected.burdens, rtol=0, atol=0)
    index = torch.tensor([0, 1])
    torch.testing.assert_close(ExactBinaryBurdenCache(actual).get(actual, index),
        ExactBinaryBurdenCache(expected).get(expected, index), rtol=0, atol=0)
    assert sum(source.reads) == 2*m and max(source.reads) <= 512
    assert actual.cmac == expected.cmac
