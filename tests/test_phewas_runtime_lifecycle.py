"""Small ownership/admission contracts, not performance benchmarks."""
from contextlib import contextmanager
from types import SimpleNamespace
import weakref
from copy import deepcopy

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit import _cached_covariance
from fudan_wgs_toolkit.pipeline import AnalysisOptions
from fudan_wgs_toolkit.phewas_runtime import runtime
from fudan_wgs_toolkit.phewas_runtime.metadata import SharedMetadataReader
from fudan_wgs_toolkit.phewas_runtime.shared_state import SharedStateBroker
from test_phewas_batched_runtime import cpu_product_oracle, descriptor, model_for
from test_phewas_shared_state import MetadataReader, fixture_states, make_cache


def test_healthy_allocation_retains_reusable_allocator_blocks(monkeypatch):
    monkeypatch.setattr(_cached_covariance, '_release_unused_cuda_blocks',
                        lambda device: pytest.fail('healthy allocations must not purge the allocator'))
    expected = object()
    assert _cached_covariance._allocate_with_pressure_reclaim(lambda: expected, 'cuda:0') is expected


@pytest.mark.parametrize('succeed_after_reclaim', [True, False])
def test_allocation_reclaims_once_only_after_actual_oom(monkeypatch, succeed_after_reclaim):
    calls, cleanup_calls, recorded = [], [], []
    cleanup = {'released_bytes': 123}
    monkeypatch.setattr(_cached_covariance, '_release_unused_cuda_blocks',
                        lambda device: cleanup_calls.append(device) or cleanup)
    def allocate():
        calls.append(True)
        if len(calls) == 1 or not succeed_after_reclaim:
            raise torch.cuda.OutOfMemoryError('injected allocation pressure')
        return 'allocated'
    if succeed_after_reclaim:
        assert _cached_covariance._allocate_with_pressure_reclaim(allocate, 'cuda:3', recorded.append) == 'allocated'
    else:
        with pytest.raises(torch.cuda.OutOfMemoryError):
            _cached_covariance._allocate_with_pressure_reclaim(allocate, 'cuda:3', recorded.append)
    assert len(calls) == 2 and cleanup_calls == ['cuda:3'] and recorded == [cleanup]


def test_nonallocation_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(_cached_covariance, '_release_unused_cuda_blocks',
                        lambda device: pytest.fail('a scientific error must not be retried'))
    def invalid():
        raise ValueError('invalid input')
    with pytest.raises(ValueError, match='invalid input'):
        _cached_covariance._allocate_with_pressure_reclaim(invalid, 'cuda:0')


class _SingleReader(MetadataReader):
    def read_field(self, name, indices=None):
        values = (np.full(self.n_variants, 'PASS') if name == 'annotation/filter'
                  else np.full(self.n_variants, '21') if name == 'chromosome'
                  else np.arange(self.n_variants))
        return values if indices is None else values[indices]
    def read_ref_alt(self, indices=None):
        n = self.n_variants if indices is None else len(indices)
        return np.full(n, 'A'), np.full(n, 'C')


@pytest.mark.parametrize('failure', [MemoryError, torch.cuda.OutOfMemoryError])
def test_single_failed_block_preparation_unpins_reader_axis(tmp_path, monkeypatch, failure):
    n, m = 24, 10
    rows = np.arange(n)
    with SharedStateBroker(SharedMetadataReader(_SingleReader(n, m)),
                          make_cache(tmp_path/'cache', fixture_states(m=m, n=n)),
                          device='cpu') as broker:
        structural_axis = broker.reader_view(rows)
        options = AnalysisOptions(wrapper_semantics='base', memory_limit_gib=80)
        structural = runtime._thin_pipeline(structural_axis, model_for(rows), rows, options, {})
        structural._base_mask = lambda *args: np.ones(m, dtype=bool)
        initial_pins = structural_axis._axis.pins

        def reject(*args, **kwargs):
            raise failure('injected block admission failure')

        monkeypatch.setattr(broker, 'trait_block', reject)
        groups = [dict(rows=rows, descriptors=[descriptor(0, rows)])]
        with pytest.raises(failure, match='injected block admission failure'):
            runtime._batched_single(groups, None, broker, broker._reader, options,
                {'individual_genotype_block_size': 4}, {'mac_cutoff': 1},
                SimpleNamespace(), dict(association_rows={'individual': 0}), set(),
                '21', 0, 1, structural)
        assert structural_axis._axis.pins == initial_pins
        structural_axis.close()


@pytest.mark.parametrize('mode', ['normal', 'empty', 'output_error'])
def test_single_releases_unowned_model_and_prepared_state_between_tiles(
        tmp_path, monkeypatch, cpu_product_oracle, mode):
    n, m = 24, 10
    rows = np.arange(n)
    raw = np.zeros((m, n), dtype=np.uint8) if mode == 'empty' else fixture_states(m=m, n=n)
    model_refs, state_refs, acquisitions = [], [], []
    class EphemeralRepository:
        def descriptors(self):
            return [descriptor(i, rows) for i in range(3)]
        @contextmanager
        def acquire(self, indices, device):
            assert all(reference() is None for reference in model_refs)
            values = [model_for(rows, .7+i) for i in indices]
            model_refs.extend(weakref.ref(value) for value in values)
            acquisitions.append(list(indices))
            try:
                yield values
            finally:
                values.clear()
    class State(dict):
        pass
    original_prepare = runtime.LimitedMaskPipeline._prepare_individual_block
    def prepare(self, *args, **kwargs):
        value, ordinal = original_prepare(self, *args, **kwargs)
        if value is not None:
            value = State(value)
            state_refs.append(weakref.ref(value))
        return value, ordinal
    monkeypatch.setattr(runtime.LimitedMaskPipeline, '_prepare_individual_block', prepare)
    repository = EphemeralRepository()
    _, groups = runtime._descriptor_groups(repository)
    writes = []
    def write(*args, **kwargs):
        assert all(reference() is None for reference in state_refs)
        writes.append(args)
        if mode == 'output_error':
            raise OSError('injected output failure')
    with SharedStateBroker(SharedMetadataReader(_SingleReader(n, m)),
                          make_cache(tmp_path/'cache', raw), device='cpu') as broker:
        axis = broker.reader_view(rows)
        options = AnalysisOptions(wrapper_semantics='base', memory_limit_gib=80)
        structural = runtime._thin_pipeline(axis, model_for(rows), rows, options, {})
        structural.models = [SimpleNamespace(device=broker.device, use_spa=False)]
        structural._base_mask = lambda *args: np.ones(m, dtype=bool)
        configuration = {'individual_genotype_block_size': 4}
        report = dict(association_rows={'individual': 0},
                      trait_rows={i: {'individual': 0} for i in range(3)},
                      binary_spa_selected=0, binary_spa_failed=0)
        def run():
            runtime._batched_single(groups, repository, broker, broker._reader, options,
                configuration, {'mac_cutoff': 1}, SimpleNamespace(write=write), report,
                set(), '21', 0, 2, structural)
        if mode == 'output_error':
            with pytest.raises(OSError, match='injected output failure'):
                run()
        else:
            run()
        assert all(reference() is None for reference in model_refs+state_refs)
        assert acquisitions and writes
        if mode == 'empty':
            assert report['association_rows']['individual'] == 0


def test_host_conversion_stays_within_estimated_tile_and_credits_only_filled_columns(tmp_path, monkeypatch):
    n, m = 512, 259
    raw = np.zeros((m, n), dtype=np.uint8)
    raw[np.arange(m), np.arange(m)] = 1
    dense_widths, output_owners, admissions = [], [], []
    # The object is imported by the mature pipeline and shared broker.
    from fudan_wgs_toolkit.pipeline import DeviceMinorBlock
    original_dense = DeviceMinorBlock.trait_dense_columns
    def dense(self, trait_rows, columns, *args, **kwargs):
        dense_widths.append(len(columns))
        output = kwargs['out']
        assert output.shape == (n, m) and output.stride() == (1, n)
        output_owners.append(output.untyped_storage().data_ptr())
        return original_dense(self, trait_rows, columns, *args, **kwargs)
    monkeypatch.setattr(DeviceMinorBlock, 'trait_dense_columns', dense)
    monkeypatch.setattr(DeviceMinorBlock, 'trait_dense', lambda *args, **kwargs:
        pytest.fail('host conversion must write into its admitted output'))
    with SharedStateBroker(MetadataReader(n, m), make_cache(tmp_path/'cache', raw), device='cpu') as broker:
        axis = broker.reader_view(np.arange(n)[::-1].copy())
        pipeline = runtime._thin_pipeline(axis, model_for(axis._axis.samples), axis._axis.samples,
            AnalysisOptions(wrapper_semantics='base', genotype_block_size=1000), {})
        pipeline.annotations = lambda selected, **kwargs: SimpleNamespace(annotations={})
        pipeline.host_memory_guard = lambda **kwargs: admissions.append(kwargs)
        pipeline.host_memory_set_future = lambda *args, **kwargs: None
        actual = runtime._prepare_shared_gene(pipeline, np.arange(m), broker)
        assert actual is not None and dense_widths == [128, 128, 3]
        assert len(set(output_owners)) == 1
        expected = (raw[:, ::-1] == 1).T.astype(np.float32)
        np.testing.assert_array_equal(actual['_genotype_host'], expected)
        credits = [item['materialized_bytes'] for item in admissions]
        assert credits == [0, n*m, n*m+4*n*128, n*m+4*n*256, 5*n*m]
        assert all(item['n'] == n and item['m'] == m for item in admissions)


def test_gene_lease_remaining_future_is_zero_on_success_and_exception():
    for fail in (False, True):
        futures, closed = [], []
        @contextmanager
        def factory(**kwargs):
            try:
                yield SimpleNamespace(guard=lambda **kwargs: None,
                                      set_future=lambda value, **kwargs: futures.append((value, kwargs['phase'])))
            finally:
                closed.append(True)
        def execute():
            with runtime._job_host_memory_scope({}, factory, kind='coding', chromosome_index=0,
                                                job_index=2) as configuration:
                configuration['_host_memory_set_future'](2048, phase='prepare')
                if fail:
                    raise ValueError('injected preparation failure')
        if fail:
            with pytest.raises(ValueError):
                execute()
        else:
            execute()
        assert futures == [(2048, 'prepare'), (0, 'job_released')] and closed == [True]


def test_late_cohort_deferral_retains_published_cohort_and_only_restores_selected_counters(monkeypatch):
    from fudan_wgs_toolkit.phewas_resources import HostMemoryDeferred
    report = dict(association_rows={'coding': 0}, trait_rows={i: {'coding': 0} for i in range(1342)},
                  binary_spa_selected=0, binary_spa_failed=0, job_groups=0)
    writer_rows, attempted, waits = [], [], []
    writer = SimpleNamespace(write=lambda kind, index, chromosome, result, **kwargs: writer_rows.append(index))
    def run_cohort(index):
        count = 0
        def attempt(configuration, current_writer):
            nonlocal count
            count += 1
            attempted.append(index)
            report['association_rows']['coding'] += 1
            report['trait_rows'][index]['coding'] += 1
            current_writer.write('coding', index, '21', [{'p': .5}])
            if index == 1 and count == 1:
                raise HostMemoryDeferred(1024, available_bytes=512)
        runtime._execute_batched_job(attempt, configuration={}, host_memory_lease_factory=None,
            kind='coding', chromosome_index=0, job_index=2, writer=writer, report=report,
            trait_indices=[index])
    def wait(seconds):
        assert writer_rows == [0] and report['trait_rows'][0]['coding'] == 1
        assert report['trait_rows'][1]['coding'] == 0
        waits.append(seconds)
    monkeypatch.setattr(runtime.time, 'sleep', wait)
    unchanged_counter = report['trait_rows'][1341]
    run_cohort(0)
    run_cohort(1)
    assert attempted == [0, 1, 1] and writer_rows == [0, 1] and waits == [5.]
    assert report['association_rows']['coding'] == 2 and report['trait_rows'][1341] is unchanged_counter


def test_deferred_preparation_traceback_releases_actual_arrays_before_wait(monkeypatch):
    from fudan_wgs_toolkit.phewas_resources import HostMemoryDeferred
    references, attempts = [], []
    def prepare_partial():
        host = np.ones((257, 259), dtype=np.float32, order='F')
        dosage = torch.ones((257, 259), dtype=torch.uint8)
        parts = [host, dosage]
        references.extend(weakref.ref(value) for value in parts)
        raise HostMemoryDeferred(host.nbytes+dosage.numel(), available_bytes=128)
    def attempt(configuration, writer):
        attempts.append(True)
        if len(attempts) == 1:
            prepare_partial()
    def wait(seconds):
        assert all(reference() is None for reference in references)
    monkeypatch.setattr(runtime.time, 'sleep', wait)
    runtime._execute_batched_job(attempt, configuration={}, host_memory_lease_factory=None,
        kind='coding', chromosome_index=0, job_index=0, writer=SimpleNamespace(write=lambda *args: None),
        report={})
    assert len(attempts) == 2 and all(reference() is None for reference in references)


@pytest.mark.parametrize('name', ['sample_axis_cache_bytes', 'sample_axis_host_cache_bytes', 'job_state_cache_bytes'])
@pytest.mark.parametrize('value', [-1, True, 3.5])
def test_cache_capacity_validation_precedes_any_cuda_or_writer_mutation(name, value):
    writer = SimpleNamespace(write=lambda *args: None, abort=lambda: pytest.fail('invalid options precede writer ownership'))
    with pytest.raises(ValueError, match=name):
        runtime.run_batched_configuration({}, model_repository=None, reader_factory=lambda *args: None,
                                          writer=writer, **{name: value})


@pytest.mark.parametrize('name', ['job_state_cache_live_fraction', 'device_cache_live_fraction'])
@pytest.mark.parametrize('value', [0, 1.1, True, float('nan')])
def test_cache_live_fraction_validation_precedes_any_cuda_or_writer_mutation(name, value):
    writer = SimpleNamespace(write=lambda *args: None, abort=lambda: pytest.fail('invalid options precede writer ownership'))
    with pytest.raises(ValueError, match=name):
        runtime.run_batched_configuration({}, model_repository=None, reader_factory=lambda *args: None,
                                          writer=writer, **{name: value})


def test_weighted_solver_diagnostic_history_is_bounded_with_exact_lifetime_counts():
    from fudan_wgs_toolkit import _weighted_spectra
    old = deepcopy(_weighted_spectra._METRICS)
    old_context = _weighted_spectra._LAST_SOLVER_CONTEXT
    try:
        _weighted_spectra.execution_metadata(reset=True)
        sizes = [4 if index % 3 == 0 else 8 for index in range(10_000)]
        for size in sizes:
            _weighted_spectra._record_batch_size(size)
        report = _weighted_spectra.execution_metadata()
        assert len(report['batch_sizes']) == _weighted_spectra._BATCH_SAMPLE_LIMIT == 64
        assert report['batch_sizes'] == sizes[-64:]
        assert report['batch_size_counts'] == {4: 3334, 8: 6666}
        assert sum(report['batch_size_counts'].values()) == len(sizes)
        report['batch_size_counts'][4] = -1
        assert _weighted_spectra._METRICS['batch_size_counts'][4] == 3334
        _weighted_spectra.execution_metadata(reset=True)
        assert not _weighted_spectra._METRICS['batch_sizes']
        assert not _weighted_spectra._METRICS['batch_size_counts']
    finally:
        _weighted_spectra._METRICS.clear()
        _weighted_spectra._METRICS.update(old)
        _weighted_spectra._LAST_SOLVER_CONTEXT = old_context


def _category_pipeline(metadata, *, structural=None, options=None, catalog=None):
    names = ('GENCODE.Category', 'GENCODE.Info', 'GeneHancer', 'CAGE', 'DHS')
    configuration = dict(qc_path='qc', _resolved_annotation_catalog=(
        {name: name for name in names} if catalog is None else catalog))
    return runtime._thin_pipeline(metadata, SimpleNamespace(device='cpu', use_spa=False),
        np.asarray([0]), options or AnalysisOptions(annotation_block_size=3),
        configuration, structural)


def test_distinct_noncoding_ncrna_selectors_share_one_complete_category_scan():
    from test_annotation_index import MetadataFile, make_pipeline
    baseline, baseline_reader = make_pipeline()
    expected_codes = baseline._annotation_categories().copy()
    expected = baseline.prepare_annotation_index('21', categories=['UTR', 'ncRNA', 'upstream'],
                                                include_ncrna=False)
    metadata = MetadataFile()
    structural = _category_pipeline(metadata)
    noncoding = _category_pipeline(metadata, structural=structural)
    ncrna = _category_pipeline(metadata, structural=structural)
    actual = noncoding.prepare_annotation_index('21', categories=['UTR'], include_ncrna=False)
    assert ncrna.prepare_annotation_index('21', categories=['ncRNA'], include_ncrna=False) is actual
    # A new selector created after both categories were prepared uses the same
    # lazy holder, even though the structural owner's original pointer was None.
    third = _category_pipeline(metadata, structural=structural)
    assert third.prepare_annotation_index('21', categories=['upstream'], include_ncrna=False) is actual
    codes = [pipeline._annotation_categories() for pipeline in (structural, noncoding, ncrna, third)]
    assert all(value is codes[0] for value in codes)
    np.testing.assert_array_equal(codes[0], expected_codes)
    assert codes[0].dtype == np.uint8 and not codes[0].flags.writeable
    for gene in expected.genes:
        for category in ('UTR', 'ncRNA', 'upstream'):
            np.testing.assert_array_equal(actual.indices(gene, category), expected.indices(gene, category))
    expected_chunks = (metadata.n_variants+2)//3
    assert metadata.calls.count('GENCODE.Category') == expected_chunks
    assert baseline_reader.calls.count('GENCODE.Category') == expected_chunks


def test_failed_category_scan_cannot_publish_partial_array_to_another_selector():
    from test_annotation_index import MetadataFile
    metadata = MetadataFile()
    original = metadata.read_field
    category_calls = 0
    def fail_second_block(path, indices=None):
        nonlocal category_calls
        if path == 'GENCODE.Category':
            category_calls += 1
            if category_calls == 2:
                raise OSError('injected category block failure')
        return original(path, indices)
    metadata.read_field = fail_second_block
    structural = _category_pipeline(metadata)
    first = _category_pipeline(metadata, structural=structural)
    second = _category_pipeline(metadata, structural=structural)
    with pytest.raises(OSError, match='category block'):
        first._annotation_categories()
    assert first._category_codes is None and first._shared_annotation_state['codes'] is None
    codes = second._annotation_categories()
    assert category_calls == 2+(metadata.n_variants+2)//3
    np.testing.assert_array_equal(codes, np.asarray([1, 2, 3, 6, 0, 4, 8, 4, 3], dtype=np.uint8))
    assert first._annotation_categories() is codes and structural._annotation_categories() is codes
    assert category_calls == 5


def test_shared_category_method_does_not_retain_short_lived_pipeline_models():
    from test_annotation_index import MetadataFile
    metadata = MetadataFile()
    structural = _category_pipeline(metadata)
    selector = _category_pipeline(metadata, structural=structural)
    class Model:
        pass
    model = Model()
    model_reference, selector_reference = weakref.ref(model), weakref.ref(selector)
    selector.models = [model]
    del model
    selector._annotation_categories()
    del selector
    # Check reference-count release without gc.collect(): a bound-method cycle
    # would keep the model and metadata view alive until a later collection.
    assert selector_reference() is None and model_reference() is None
    assert structural._annotation_categories().dtype == np.uint8


@pytest.mark.parametrize('different', ['source', 'options', 'catalog'])
def test_annotation_sharing_rejects_different_metadata_or_options(different):
    from test_annotation_index import MetadataFile
    metadata = MetadataFile()
    structural = _category_pipeline(metadata)
    arguments = dict(metadata=metadata)
    if different == 'source':
        arguments['metadata'] = MetadataFile()
    elif different == 'options':
        arguments['options'] = AnalysisOptions(annotation_block_size=4)
    else:
        arguments['catalog'] = {'GENCODE.Category': 'another_category_source'}
    other = _category_pipeline(**arguments)
    with pytest.raises(ValueError, match='identical metadata source and options'):
        runtime._share_annotations([structural, other])
    assert structural._base_masks is not other._base_masks


def test_reader_diagnostics_keep_read_and_hash_timers_without_private_metadata():
    reader = SimpleNamespace(reader_metadata=dict(metadata_reads=17, metadata_read_seconds=4.25,
        metadata_checked_files=8, metadata_checksum_verification_seconds=1.75,
        metadata_checksums_enabled=True, sample_keys=['private_sample'], private_path='private_path',
        metadata_manifest_sha256='private_binding', unknown_counter=99))
    diagnostics = runtime._reader_metadata_diagnostics(SharedMetadataReader(reader))
    assert diagnostics == dict(metadata_reads=17, metadata_read_seconds=4.25,
        metadata_checked_files=8, metadata_checksum_verification_seconds=1.75,
        metadata_checksums_enabled=True)


@pytest.mark.parametrize('outcome', ['success', 'attempt_error', 'publish_error'])
def test_real_host_gate_recovery_drops_optional_lru_without_refilling_or_masking_deferral(
        tmp_path, monkeypatch, outcome):
    import json
    import os
    from fudan_wgs_toolkit import phewas_resources as resources
    owner = dict(pid=os.getpid(), start_ticks=1, boot_id='anonymous-fixture')
    monkeypatch.setattr(resources, 'host_scope', lambda: ('f'*64, {'fixture': True}))
    monkeypatch.setattr(resources, 'process_identity', lambda pid: owner if pid == owner['pid'] else None)
    metadata = SharedMetadataReader(_SingleReader(1, 32), capacity_bytes=4096)
    position = metadata.pin_field('position')
    qc = metadata.pin_field('annotation/filter')
    samples = metadata.sample_ids()
    structural = runtime._thin_pipeline(metadata, SimpleNamespace(device='cpu', use_spa=False),
        np.asarray([0]), AnalysisOptions(), {})
    index = object()
    structural._annotation_indexes[('21', 'SNV')] = index
    codes = np.arange(32, dtype=np.uint8)
    structural._category_codes = codes
    runtime._share_annotations([structural])
    retained = metadata.read_field('unused_field')
    retained_reference = weakref.ref(retained)
    del retained
    active_reference = [None]
    def live_memory():
        active = None if active_reference[0] is None else active_reference[0]()
        shared_with_lru = any(active is value[0] for value in metadata._cache.values())
        active_bytes = 0 if active is None or shared_with_lru else active.nbytes
        current = 200+metadata._bytes+active_bytes
        return dict(current_bytes=current, effective_current_bytes=current, limit_bytes=1024,
                    inactive_file_bytes=0, scope='cgroup_v2')
    monkeypatch.setattr(resources, 'read_host_memory', live_memory)
    monkeypatch.setattr(resources, '_process_memory_counters', lambda pid,**kwargs:
        dict(pss_bytes=live_memory()['effective_current_bytes'],rss_bytes=live_memory()['effective_current_bytes']))
    gate = resources.HostMemoryGate(tmp_path, host_limit_gib=1024/resources.GiB, reserve_gib=0)
    report = dict(association_rows={'coding': 0}, trait_rows={0: {'coding': 0}})
    attempts, waits, writes = [], [], []
    def attempt(configuration, writer):
        attempts.append(True)
        values = metadata.read_field('weights', np.asarray([0]))
        active_reference[0] = weakref.ref(values)
        report['association_rows']['coding'] += 1
        report['trait_rows'][0]['coding'] += 1
        writer.write('coding', 0, '21', [{'p': .25}])
        # With optional cache: 200+264+800 exceeds 1024. During recovery:
        # 200+8+800 fits; the live measurement still counts active input bytes.
        configuration['_host_memory_set_future'](800, phase='fixture_preparation')
        assert not metadata._cache_admission and not metadata._cache and metadata._bytes == 0
        assert structural.position is position and structural.qc is qc
        assert metadata.sample_ids() is samples
        assert structural._annotation_indexes[('21', 'SNV')] is index
        assert structural._annotation_categories() is codes
        if outcome == 'attempt_error':
            raise OSError('injected preparation error')
    def wait(seconds):
        assert retained_reference() is None and active_reference[0]() is None
        assert not metadata._cache_admission and not metadata._cache
        assert json.loads(gate.path.read_text())['leases'] == {}
        assert report['association_rows']['coding'] == 0 and not writes
        waits.append(seconds)
    def publish(*args):
        if outcome == 'publish_error':
            raise OSError('injected publish error')
        writes.append(args)
    monkeypatch.setattr(runtime.time, 'sleep', wait)
    kwargs = dict(configuration={}, host_memory_lease_factory=gate.lease_factory,
        kind='coding', chromosome_index=0, job_index=0, writer=SimpleNamespace(write=publish),
        report=report, trait_indices=[0], host_cache_recovery=metadata.pressure_recovery)
    if outcome == 'success':
        runtime._execute_batched_job(attempt, **kwargs)
        assert len(writes) == 1 and report['association_rows']['coding'] == 1
    else:
        with pytest.raises(OSError, match='injected'):
            runtime._execute_batched_job(attempt, **kwargs)
        assert not writes
    assert len(attempts) == 2 and waits == [5.]
    assert metadata._cache_admission and metadata.capacity == 4096
    assert json.loads(gate.path.read_text())['leases'] == {}
    assert metadata.metrics['pressure_evictions'] == 2
    metadata.read_field('weights', np.asarray([0]))
    assert metadata._cache and metadata._bytes == 8


def test_closed_host_lease_cleanup_is_idempotent_but_new_requests_remain_strict(tmp_path, monkeypatch):
    import os
    import json
    from fudan_wgs_toolkit import phewas_resources as resources
    owner = dict(pid=os.getpid(), start_ticks=1, boot_id='anonymous-fixture')
    monkeypatch.setattr(resources, 'host_scope', lambda: ('f'*64, {'fixture': True}))
    monkeypatch.setattr(resources, 'process_identity', lambda pid: owner if pid == owner['pid'] else None)
    monkeypatch.setattr(resources, '_process_memory_counters', lambda pid,**kwargs:dict(pss_bytes=90,rss_bytes=90))
    monkeypatch.setattr(resources, 'read_host_memory', lambda: dict(current_bytes=90, limit_bytes=100,
        effective_current_bytes=90, inactive_file_bytes=0, scope='cgroup_v2'))
    gate = resources.HostMemoryGate(tmp_path, host_limit_gib=100/resources.GiB, reserve_gib=0)
    with runtime._job_host_memory_scope({}, gate.lease_factory,
            kind='coding', chromosome_index=0, job_index=0) as configuration:
        setter = configuration['_host_memory_set_future']
        with pytest.raises(resources.HostMemoryDeferred):
            setter(20, phase='fixture_preparation')
        # No cleanup ledger rewrite or false admission credit after retirement.
        ledger = gate.path.read_bytes()
        assert setter(0, phase='gene_mask_released') == 0.
        assert gate.path.read_bytes() == ledger
        with pytest.raises(RuntimeError, match='not active'):
            setter(1, phase='new_allocation')
    assert json.loads(gate.path.read_text())['leases'] == {}


def test_healthy_cohort_keeps_existing_metadata_lru():
    metadata = SharedMetadataReader(_SingleReader(1, 4), capacity_bytes=64)
    cached = metadata.read_field('weights')
    writes = []
    runtime._execute_batched_job(lambda configuration, writer: writer.write('coding', 0, '21', [{'p': .5}]),
        configuration={}, host_memory_lease_factory=None, kind='coding', chromosome_index=0,
        job_index=0, writer=SimpleNamespace(write=lambda *args: writes.append(args)), report={},
        host_cache_recovery=metadata.pressure_recovery)
    assert len(writes) == 1 and metadata.read_field('weights') is cached
    assert metadata._cache_admission and metadata._bytes == 32
    assert metadata.metrics.get('pressure_evictions', 0) == 0
