"""Small transport/scientific contracts; these fixtures are not benchmarks."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.null_model import GaussianNullModel, KinshipSpectrum
from fudan_wgs_toolkit.pipeline import AnalysisOptions
from fudan_wgs_toolkit.phewas_runtime import runtime, single
from fudan_wgs_toolkit.phewas_runtime.metadata import SharedMetadataReader
from fudan_wgs_toolkit.phewas_runtime.shared_state import SharedStateBroker
from test_phewas_shared_state import MetadataReader, make_cache, fixture_states, oracle, assert_block
from test_phewas_single_batches import minor_block, make_pipeline, states_fixture


@pytest.fixture
def cpu_product_oracle(monkeypatch):
    """Exercise orchestration/FP32 formulas without claiming CUDA TF32 execution."""
    from fudan_wgs_toolkit import tf32, null_model, binary_null
    def product(left, right, *, mode='tf32'):
        assert mode == 'tf32' and left.device.type == right.device.type == 'cpu'
        return left@right
    monkeypatch.setattr(tf32, 'matmul', product)
    monkeypatch.setattr(null_model, 'matmul', product)
    monkeypatch.setattr(binary_null, 'matmul', product)


def model_for(rows, multiplier=1., device='cpu'):
    n = len(rows)
    x = torch.ones((n, 1), dtype=torch.float32, device=device)
    residual = torch.linspace(-.2, .3, n, device=device)*multiplier
    precision = torch.linspace(.8, 1.1, n, device=device)
    covariance = torch.tensor([[1/float(precision.sum())]], dtype=torch.float32, device=device)
    return GaussianNullModel(np.asarray([f'sample{i}' for i in rows]), x, residual,
        torch.zeros(1, device=device), torch.ones(2, device=device), torch.ones(2, device=device),
        covariance, KinshipSpectrum(torch.ones(n, device=device), []), precision,
        x*precision[:, None], 1, True, has_kinship=False, matmul_mode='tf32')


class Repository:
    def __init__(self, descriptors, models=()):
        self.items, self.models = descriptors, list(models)
        self.active, self.acquired = 0, []
    def descriptors(self):
        return self.items
    @contextmanager
    def acquire(self, indices, device):
        self.active += 1
        self.acquired.append(list(indices))
        try:
            yield [self.models[i] for i in indices]
        finally:
            self.active -= 1


def descriptor(index, rows, family='gaussian'):
    return dict(trait_index=index, n=len(rows), family=family, ordered_rows=np.asarray(rows), projection_key=None)


def test_cohort_grouping_retains_request_order_and_never_groups_by_n_or_projection_name():
    rows = np.asarray([3, 1, 8, 0])
    ds = [descriptor(0, rows), descriptor(1, rows.copy(), 'binomial'), descriptor(2, rows[::-1])]
    ds[0]['projection_key'] = ds[2]['projection_key'] = 'same-unverified-name'
    actual, groups = runtime._descriptor_groups(Repository(ds))
    assert actual == ds
    assert [[d['trait_index'] for d in group['descriptors']] for group in groups] == [[0, 1], [2]]
    np.testing.assert_array_equal(groups[0]['rows'], rows)
    np.testing.assert_array_equal(groups[1]['rows'], rows[::-1])


@pytest.mark.parametrize('change', ['empty', 'boolean', 'float_rows', 'bad_n', 'family', 'duplicate'])
def test_invalid_descriptor_is_rejected(change):
    ds = [descriptor(0, np.arange(5))]
    if change == 'empty': ds = []
    elif change == 'boolean': ds[0]['trait_index'] = True
    elif change == 'float_rows': ds[0]['ordered_rows'] = np.arange(5.)
    elif change == 'bad_n': ds[0]['n'] = 4
    elif change == 'family': ds[0]['family'] = 'joint'
    else: ds.append(dict(ds[0]))
    with pytest.raises(ValueError):
        runtime._descriptor_groups(Repository(ds))


def test_no_artificial_budget_and_none_uses_actual_device_capacity(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device:
                        calls.append(device) or SimpleNamespace(total_memory=96*2**30))
    assert runtime._resolved_budget(None, 'cuda:3') == 96
    assert runtime._resolved_budget(128., 'cuda:3') == 128
    assert calls == ['cuda:3']


@pytest.mark.parametrize('value', [False, 0, -1, float('nan'), float('inf'), '80'])
def test_invalid_runtime_budget_is_rejected(value):
    with pytest.raises(ValueError):
        runtime._resolved_budget(value, 'cuda:0')


def test_portable_logical_axis_is_bound_to_exact_physical_cache_rows(tmp_path):
    raw = fixture_states(m=9, n=11)
    original = np.arange(100, 111)[::-1].copy()
    container = make_cache(tmp_path/'cache', raw, original)
    reader = MetadataReader(11, len(raw))
    reader.manifest = {'source_sample_rows': 'rows.npy'}
    reader._array = lambda path: original
    with SharedStateBroker(reader, container, device='cpu', portable_axis=True) as broker:
        samples, columns = np.asarray([10, 2, 4]), np.asarray([2, 5, 3])
        block = broker.trait_block(broker.read_states(columns), samples)
        assert_block(block, oracle(raw, np.arange(11), samples, columns), samples)
    container = make_cache(tmp_path/'bad_cache', raw, original)
    reader._array = lambda path: original[::-1]
    with pytest.raises(ValueError, match='portable source sample rows'):
        SharedStateBroker(reader, container, device='cpu', portable_axis=True)
    container.close()


def test_bound_sample_axes_have_limited_device_vectors_and_reupload_exactly(tmp_path):
    raw = fixture_states(m=10, n=23)
    with SharedStateBroker(MetadataReader(23, 10), make_cache(tmp_path/'cache', raw),
                          device='cpu', sample_axis_cache_bytes=6*8) as broker:
        columns = np.asarray([4, 2, 9])
        slab = broker.read_states(columns)
        for samples in (np.arange(6), np.arange(6)[::-1], np.arange(10, 16), np.arange(6)):
            assert_block(broker.trait_block(slab, samples), oracle(raw, np.arange(23), samples, columns), samples)
        assert broker.metrics['current_sample_axis_device_bytes'] == 6*8
        assert broker.metrics['sample_axis_device_uploads'] == 4
        assert broker.metrics['sample_axis_device_evictions'] == 3
        assert broker.metrics['sample_axis_device_bytes_highwater'] == 6*8
        assert len(broker._axes) == 3


def test_thin_pipeline_shares_structural_metadata_and_one_current_identity_axis(tmp_path):
    raw = fixture_states(m=9, n=11)
    with SharedStateBroker(MetadataReader(11, 9), make_cache(tmp_path/'cache', raw), device='cpu') as broker:
        axis = broker.reader_view(np.asarray([8, 1, 4, 2]))
        model = model_for(axis._axis.samples)
        first = runtime._thin_pipeline(axis, model, axis._axis.samples, AnalysisOptions(wrapper_semantics='base'), {})
        second = runtime._thin_pipeline(axis, model, axis._axis.samples, first.options, {}, first)
        assert first.union_ids is second.union_ids is None
        assert first.trait_rows[0] is second.trait_rows[0]
        assert first.position is second.position and first.qc is second.qc
        assert first._base_masks is second._base_masks
        np.testing.assert_array_equal(second.union_rows, [8, 1, 4, 2])


def test_model_sample_binding_checked_once_and_rejects_permuted_ids():
    reader = SharedMetadataReader(MetadataReader(10, 7))
    d = descriptor(0, np.asarray([8, 2, 4]))
    verified = set()
    runtime._check_model(model_for(d['ordered_rows']), d, reader, verified)
    assert verified == {0}
    broken = model_for(d['ordered_rows'][::-1])
    with pytest.raises(ValueError, match='portable ordered sample binding'):
        runtime._check_model(broken, dict(d, trait_index=1), reader, verified)


@pytest.mark.parametrize('imputation', ['mean', 'minor'])
def test_streamed_gene_preparation_matches_mature_rare_orientation_and_group_order(tmp_path, imputation):
    n, m = 512, 9
    raw = np.zeros((m, n), dtype=np.uint8)
    raw[0, 3] = 1
    raw[1, 7:10] = [1, 4, 5]
    raw[2] = 2; raw[2, 9] = 1
    raw[3] = 1
    raw[4, 11] = 1; raw[4, 50:65] = 3
    raw[5, :16] = 1
    raw[6, 18:20] = [4, 5]
    raw[7, 20:23] = [1, 1, 1]
    metadata = MetadataReader(n, m)
    with SharedStateBroker(metadata, make_cache(tmp_path/'cache', raw), device='cpu') as broker:
        axis = broker.reader_view(np.arange(n)[::-1].copy())
        p = runtime._thin_pipeline(axis, model_for(axis._axis.samples), axis._axis.samples,
            AnalysisOptions(wrapper_semantics='base', genotype_block_size=3, imputation=imputation), {})
        leases = []
        p.host_memory_guard = lambda **value: leases.append(value)
        p.annotations = lambda selected, **kwargs: SimpleNamespace(annotations={})
        indices = np.arange(m)
        expected = p._prepare_test_set(indices, _defer_score=True)[0]
        leases.clear()
        actual = runtime._prepare_shared_gene(p, indices, broker)
        assert actual is not None
        for key in ('maf', 'mac', 'annotations', '_variant_indices', '_extraction_groups'):
            np.testing.assert_array_equal(actual[key], expected[key])
        np.testing.assert_array_equal(actual['_genotype_host'], expected['_genotype_host'].numpy())
        assert actual['_genotype_host'].dtype == np.float32
        assert actual['_genotype_host'].flags.f_contiguous
        assert leases and all(value['phase'] == 'prepare' for value in leases)
        assert leases[-1]['m'] == len(actual['maf'])


def test_shared_single_score_and_mature_variance_use_one_cohort_genotype(cpu_product_oracle):
    rows = np.arange(256)[::-1].copy()
    block = minor_block(states_fixture(), rows, np.arange(10))
    models = [model_for(rows, .8), model_for(rows, 1.2)]
    pipelines = [make_pipeline(rows, [], model=model) for model in models]
    first = pipelines[0]
    state, ordinal = first._prepare_individual_block(block, 0, mac_cutoff=1)
    prepared = first._prepare_individual_trait(state, 0, mac_cutoff=1)
    prepared.pop('model')
    genotype = prepared['genotype']
    expected = [model.individual_score_variance(genotype) for model in models]
    single.execution_metadata(reset=True)
    actual, final_ordinal = single.process_single_shared_genotype(pipelines, None, ordinal, '21',
        mac_cutoff=1, prepared_state=(prepared, ordinal))
    assert final_ordinal == ordinal
    for records, (score, variance) in zip(actual, expected):
        np.testing.assert_allclose([row['Score'] for row in records], score.numpy(), rtol=5e-6, atol=5e-6)
        np.testing.assert_array_equal([row['Score_se'] for row in records], variance.double().sqrt().numpy())
        from fudan_wgs_toolkit.pipeline import _individual_log_probabilities
        expected_logp = _individual_log_probabilities(score, variance).numpy()/np.log(10.)
        actual_logp = np.asarray([row['pvalue_log10'] for row in records])
        np.testing.assert_array_equal(np.isfinite(actual_logp), np.isfinite(expected_logp))
        finite = np.isfinite(expected_logp)
        assert np.all(np.abs(actual_logp[finite]-expected_logp[finite]) < .001)
    metrics = single.execution_metadata()
    assert metrics['phenotype_score_gemm_calls'] == 0
    assert metrics['phenotype_score_batches'] == 1
    assert (metrics['phenotype_score_fp32_vector_calls'],
            metrics['phenotype_score_fp32_simt_gemm_calls']) in ((2, 0), (0, 1))
    assert metrics['original_variance_calls'] == 2
    assert metrics['result_transfer_batches'] == 1
    assert metrics['dense_genotype_h2d_bytes'] == metrics['dense_genotype_d2h_bytes'] == 0
    assert prepared['genotype'] is genotype


def test_single_binary_callback_formal_p_replaces_nominal_and_flags_share_one_transfer(cpu_product_oracle):
    rows = np.arange(256)
    block = minor_block(states_fixture(), rows, np.arange(10))
    model = model_for(rows)
    model.family = 'binomial'
    model.precision = model.inverse_variance
    p = make_pipeline(rows, [], model=model)
    called = []
    def correct(index, model, genotype, score, variance):
        called.append((index, genotype.device.type, genotype.dtype))
        n = len(score)
        zero = torch.zeros(n, dtype=torch.bool)
        values = torch.full((n,), .125, dtype=torch.float64)
        log10 = -torch.log10(values)
        return dict(normal_only=False, pvalues=values, pvalue_log10=log10,
            pvalue_log=log10*np.log(10), normal_pvalues=torch.full_like(values, .25),
            normal_pvalue_log10=torch.full_like(values, -np.log10(.25)),
            spa_selected=~zero, spa_zero_log_unverified=zero,
            spa=SimpleNamespace(used_bisection=zero, failed=zero,
                                iterations=torch.ones(n, dtype=torch.int64), iteration_limit=zero))
    single.execution_metadata(reset=True)
    outputs, ordinal = single.process_single_shared_genotype([p], block, 0, '21',
        mac_cutoff=1, binary_correction=correct)
    assert ordinal and called == [(0, 'cpu', torch.float32)]
    assert outputs[0] and all(row['pvalue'] == .125 and row['normal_pvalue'] == .25 for row in outputs[0])
    assert all(row['spa_selected'] and not row['spa_failed'] for row in outputs[0])
    assert single.execution_metadata()['result_transfer_batches'] == 1


def test_blocked_gaussian_panels_retain_non_diagonal_relatedness(cpu_product_oracle):
    rows = np.arange(23)
    model = model_for(rows)
    rotation = torch.tensor([[.8, -.6], [.6, .8]], dtype=torch.float32)
    model.spectrum.blocks = [(torch.tensor([3, 17]), rotation)]
    model.has_kinship = True
    genotype = np.asarray((np.arange(23*9).reshape(23, 9)%5)/3, dtype=np.float32, order='F')
    guards = []
    broker = SimpleNamespace(_guard=lambda required: guards.append(required))
    expected = model.score_covariance(torch.from_numpy(genotype))[1]
    actual = runtime._blocked_gaussian_covariance(model, genotype, broker, variant_tile_size=3)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(actual.numpy(), actual.T.numpy())
    diagonal = model_for(rows).score_covariance(torch.from_numpy(genotype))[1]
    assert not torch.allclose(actual, diagonal, atol=1e-4, rtol=1e-4)
    assert len(guards) == 1 and guards[0] >= actual.numel()*4


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
                    reason='actual native TF32/transport proof requires Ampere CUDA')
def test_gpu_single_one_raw_upload_multiple_trait_tiles_and_no_dense_roundtrip(tmp_path):
    raw = states_fixture().T.copy()
    rows = np.arange(256)
    with SharedStateBroker(MetadataReader(256, 10), make_cache(tmp_path/'cache', raw),
                          device='cuda:0') as broker:
        slab = broker.read_states(np.arange(10), minimum_mac_bound=1)
        block = broker.trait_block(slab, rows, minimum_mac=1)
        models = [model_for(rows, .7+i, device='cuda:0') for i in range(4)]
        pipelines = [make_pipeline(rows, [], model=m, device='cuda:0') for m in models]
        state, ordinal = pipelines[0]._prepare_individual_block(block, 0, mac_cutoff=1)
        prepared = pipelines[0]._prepare_individual_trait(state, 0, mac_cutoff=1)
        prepared.pop('model')
        before = broker.metrics['h2d_bytes']
        single.execution_metadata(reset=True)
        for begin in (0, 2):
            result, new = single.process_single_shared_genotype(pipelines[begin:begin+2], None,
                ordinal, '21', mac_cutoff=1, prepared_state=(prepared, ordinal))
            assert new == ordinal and len(result) == 2
        assert broker.metrics['gpu_csr_uploads'] == 1
        assert broker.metrics['h2d_bytes'] == before
        assert single.execution_metadata()['result_transfer_batches'] == 2
        assert single.execution_metadata()['dense_genotype_h2d_bytes'] == 0
        assert single.execution_metadata()['dense_genotype_d2h_bytes'] == 0


def test_batched_single_reads_each_physical_request_once_for_all_tiles_and_cleans_on_output_error(tmp_path, cpu_product_oracle):
    n, m = 24, 10
    raw = fixture_states(m=m, n=n)
    class Reader(MetadataReader):
        def read_field(self, name, indices=None):
            value = np.full(m, 'PASS') if name == 'annotation/filter' else np.arange(m)
            if name == 'chromosome': value = np.full(m, '21')
            return value if indices is None else value[indices]
        def read_ref_alt(self, indices=None):
            selected = np.arange(m) if indices is None else indices
            return np.full(len(selected), 'A'), np.full(len(selected), 'C')
    ds = [descriptor(i, np.arange(n)) for i in range(3)]
    repository = Repository(ds, [model_for(np.arange(n), .7+i) for i in range(3)])
    _, groups = runtime._descriptor_groups(repository)
    with SharedStateBroker(SharedMetadataReader(Reader(n, m)), make_cache(tmp_path/'cache', raw), device='cpu') as broker:
        metadata = broker._reader
        axis = broker.reader_view(np.arange(n))
        progress = []
        config = dict(individual_genotype_block_size=4, single_progress_frame_interval=2,
                      single_progress_seconds=10**9, _batched_emit=progress.append)
        options = AnalysisOptions(wrapper_semantics='base', memory_limit_gib=80)
        structural = runtime._thin_pipeline(axis, repository.models[0], axis._axis.samples, options, config)
        structural._base_mask = lambda *args: np.ones(m, dtype=bool)
        report = dict(association_rows={'individual': 0}, trait_rows={i: {'individual': 0} for i in range(3)},
                      binary_spa_selected=0, binary_spa_failed=0)
        writes = []
        writer = SimpleNamespace(write=lambda *args, **kwargs: writes.append(args))
        runtime._batched_single(groups, repository, broker, metadata, options, config,
            {'mac_cutoff': 1, 'variant_type': 'variant'}, writer, report, set(), '21', 0, 2, structural)
        assert broker.metrics['state_read_calls'] == 3
        assert broker.metrics['frame_reads'] == broker.metrics['device_csr_uploads'] == 1
        assert broker.metrics['trait_count_calls'] == 3
        assert len(writes) == 9 and repository.active == 0
        assert len({report['trait_rows'][i]['individual'] for i in range(3)}) == 1
        assert len(progress) == 1 and progress[0]['physical_requests'] == 2
        assert progress[0]['completed_rows'] > 0
        assert progress[0]['reader_metrics_scope'] == 'reader lifetime; not this job delta'
        def fail(*args, **kwargs):
            raise OSError('output reserve')
        with pytest.raises(OSError, match='output reserve'):
            runtime._batched_single(groups, repository, broker, metadata, options, config,
                {'mac_cutoff': 1, 'variant_type': 'variant'}, SimpleNamespace(write=fail), report,
                set(), '21', 0, 2, structural)
        assert repository.active == 0


def test_batched_entry_restores_threads_and_aborts_writer_on_missing_cuda(monkeypatch):
    before = torch.get_num_threads()
    aborted = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='requires CUDA'):
        runtime.run_batched_configuration({}, model_repository=Repository([]),
            reader_factory=lambda *args: None, writer=SimpleNamespace(write=lambda *args: None,
                                                                    abort=lambda: aborted.append(True)))
    assert aborted == [True] and torch.get_num_threads() == before
    assert runtime._LOCK.acquire(blocking=False)
    runtime._LOCK.release()


def test_gene_job_reuses_cohort_preparation_across_tiles_and_writes_every_trait(tmp_path, cpu_product_oracle, monkeypatch):
    n, m = 256, 6
    raw = np.zeros((m, n), dtype=np.uint8)
    for i in range(m): raw[i, i] = 1
    class Reader(MetadataReader):
        def read_field(self, name, indices=None):
            value = np.full(m, 'PASS') if name == 'annotation/filter' else np.arange(m)
            if name == 'chromosome': value = np.full(m, '21')
            return value if indices is None else value[indices]
    ds = [descriptor(i, np.arange(n)) for i in range(3)]
    repository = Repository(ds, [model_for(np.arange(n), .7+i) for i in range(3)])
    _, groups = runtime._descriptor_groups(repository)
    writes = []
    prepared_calls = []
    native_prepare = runtime._prepare_shared_gene
    def prepare(*args):
        prepared_calls.append(len(args[1]))
        return native_prepare(*args)
    monkeypatch.setattr(runtime, '_prepare_shared_gene', prepare)
    monkeypatch.setattr(runtime.LimitedMaskPipeline, '_select_mask_chunks', lambda self, *args: {'ncRNA': np.arange(m)})
    monkeypatch.setattr(runtime.LimitedMaskPipeline, '_long_mask_products',
                        lambda self, model, host, **kwargs: model.score_covariance(torch.from_numpy(host)))
    monkeypatch.setattr(runtime.LimitedMaskPipeline, '_evaluate_prepared',
                        lambda self, payload, model: dict(num_variant=len(payload['maf']), cMAC=payload['cmac'],
                                                         **{'WGS-O': .25}))
    with SharedStateBroker(SharedMetadataReader(Reader(n, m)), make_cache(tmp_path/'cache', raw), device='cpu') as broker:
        axis = broker.reader_view(np.arange(n))
        options = AnalysisOptions(wrapper_semantics='base', memory_limit_gib=80)
        structural = runtime._thin_pipeline(axis, repository.models[0], axis._axis.samples, options, {})
        # Match production: the full null is released after structural setup.
        structural.models = [SimpleNamespace(device=broker.device, use_spa=False)]
        report = dict(association_rows={'ncrna': 0}, trait_rows={i: {'ncrna': 0} for i in range(3)},
                      binary_spa_selected=0, binary_spa_failed=0)
        runtime._batched_gene(groups, repository, broker, broker._reader, options, {},
            dict(gene_name='fixture_gene', start=0, end=5), SimpleNamespace(write=lambda *args, **kwargs: writes.append(args)),
            report, set(), '21', 0, 2, structural, 'ncrna')
        assert prepared_calls == [m]
        assert broker.metrics['state_read_calls'] == broker.metrics['frame_reads'] == 1
        assert len(writes) == 3 and {item[1] for item in writes} == {0, 1, 2}
        assert all(len(item[3]) == 1 and item[3][0]['STAAR-O'] == .25 for item in writes)
        assert report['association_rows']['ncrna'] == 3 and repository.active == 0


def test_gene_host_lease_deferral_releases_models_and_lease_before_any_output(tmp_path, monkeypatch):
    n, m = 256, 6
    raw = np.zeros((m, n), dtype=np.uint8)
    for i in range(m):
        raw[i, i] = 1
    ds = [descriptor(i, np.arange(n)) for i in range(2)]
    repository = Repository(ds, [model_for(np.arange(n)) for _ in ds])
    _, groups = runtime._descriptor_groups(repository)
    writes, called, scopes = [], [], []
    class Deferred(MemoryError):
        pass
    def guard(**kwargs):
        called.append(kwargs)
        raise Deferred('shared admission temporarily unavailable')
    @contextmanager
    def factory(**kwargs):
        scopes.append(('enter', kwargs))
        try:
            yield SimpleNamespace(guard=guard)
        finally:
            scopes.append(('released', kwargs))
    monkeypatch.setattr(runtime.LimitedMaskPipeline, '_select_mask_chunks',
                        lambda self, *args: {'ncRNA': np.arange(m)})
    with SharedStateBroker(SharedMetadataReader(MetadataReader(n, m)),
                          make_cache(tmp_path/'cache', raw), device='cpu') as broker:
        axis = broker.reader_view(np.arange(n))
        options = AnalysisOptions(wrapper_semantics='base', memory_limit_gib=80)
        structural = runtime._thin_pipeline(axis, repository.models[0], axis._axis.samples, options, {})
        report = dict(association_rows={'ncrna': 0}, trait_rows={i: {'ncrna': 0} for i in range(2)},
                      binary_spa_selected=0, binary_spa_failed=0)
        with pytest.raises(Deferred, match='temporarily unavailable'):
            with runtime._job_host_memory_scope({}, factory, kind='ncrna', chromosome_index=2,
                                               job_index=7) as configuration:
                runtime._batched_gene(groups, repository, broker, broker._reader, options, configuration,
                    dict(gene_name='fixture_gene', start=0, end=5),
                    SimpleNamespace(write=lambda *a, **kw: writes.append(a)), report,
                    set(), '21', 7, 2, structural, 'ncrna')
        assert not writes and repository.active == 0
        assert called == [dict(n=n, m=m, phase='prepare')]
        assert [item[0] for item in scopes] == ['enter', 'released']
        assert scopes[0][1] == dict(kind='ncrna', chromosome_index=2, job_index=7)
        assert report['association_rows']['ncrna'] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA allocation admission contract')
def test_native_weight_relation_admission_uses_caller_budget_above_40_gib(monkeypatch):
    from fudan_wgs_toolkit import statistics, tf32
    before = tf32._memory_limit_bytes
    tf32.configure_tf32(memory_limit_gib=80)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda *args: 45*2**30)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda *args: 45*2**30)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda *args: (30*2**30, 80*2**30))
    rows = torch.tensor([[1., 2., 3.], [2., 4., 6.], [1., 4., 9.]], device='cuda:0')
    try:
        _, _, relations = statistics._native_weight_relations(rows)
        assert relations[0][1] and not relations[0][2]
    finally:
        tf32._memory_limit_bytes = before


def test_host_deferral_unwinds_then_retries_without_duplicate_outputs_or_scientific_counters(monkeypatch):
    import weakref
    from fudan_wgs_toolkit.phewas_resources import HostMemoryDeferred
    repository = Repository([descriptor(0, np.arange(4))], [model_for(np.arange(4))])
    leases, guards, attempts, writes, events, waits, prepared_references = [], [], [], [], [], [], []
    class Prepared:
        pass
    def guard(**kwargs):
        guards.append(kwargs)
        if len(guards) == 1:
            raise HostMemoryDeferred(1024, available_bytes=512)
    @contextmanager
    def factory(**kwargs):
        token = object()
        leases.append(token)
        try:
            yield SimpleNamespace(guard=guard)
        finally:
            leases.remove(token)
    report = dict(association_rows={'coding': 5}, trait_rows={0: {'coding': 5}},
                  binary_spa_selected=7, binary_spa_failed=3, job_groups=2,
                  reader_cumulative_calls=0)
    def attempt(configuration, current_writer):
        attempts.append(True)
        prepared = Prepared()
        prepared_references.append(weakref.ref(prepared))
        with repository.acquire([0], 'cpu'):
            report['reader_cumulative_calls'] += 1
            report['association_rows']['coding'] += 1
            report['trait_rows'][0]['coding'] += 1
            report['binary_spa_selected'] += 2
            report['binary_spa_failed'] += 1
            report['job_groups'] += 1
            # Even an early write is private until the complete attempt returns.
            current_writer.write('coding', 0, '1', [{'p': .25}], job_index=4)
            configuration['host_memory_guard'](n=4, m=6, phase='prepare')
    def wait(delay):
        assert not leases and repository.active == 0 and not writes
        assert prepared_references[0]() is None
        waits.append(delay)
    monkeypatch.setattr(runtime.time, 'sleep', wait)
    runtime._execute_batched_job(attempt, configuration={}, host_memory_lease_factory=factory,
        kind='coding', chromosome_index=0, job_index=4,
        writer=SimpleNamespace(write=lambda *a, **kw: writes.append((a, kw))), report=report,
        emit=events.append)
    assert len(attempts) == len(guards) == 2 and waits == [5.]
    assert len(writes) == 1 and writes[0][1]['job_index'] == 4
    assert report['association_rows'] == {'coding': 6} and report['trait_rows'] == {0: {'coding': 6}}
    assert report['binary_spa_selected'] == 9 and report['binary_spa_failed'] == 4
    assert report['job_groups'] == 3 and report['reader_cumulative_calls'] == 2
    assert report['host_memory_deferrals'] == 1 and report['host_memory_retry_wait_seconds'] >= 0
    assert events == [dict(event='host_memory_deferred', attempt=1, retry_after_seconds=5.,
                           required_bytes=1024, available_bytes=512)]
    assert not leases and repository.active == 0 and prepared_references[1]() is None


def test_permanent_host_capacity_failure_is_not_retried_and_does_not_publish(monkeypatch):
    from fudan_wgs_toolkit.phewas_resources import HostMemoryRequestTooLarge
    released, waits, writes = [], [], []
    @contextmanager
    def factory(**kwargs):
        try:
            yield SimpleNamespace(guard=lambda **kw: None)
        finally:
            released.append(True)
    def attempt(configuration, current_writer):
        current_writer.write('coding', 0, '1', [{'p': .25}])
        raise HostMemoryRequestTooLarge(100, 50)
    monkeypatch.setattr(runtime.time, 'sleep', waits.append)
    with pytest.raises(HostMemoryRequestTooLarge):
        runtime._execute_batched_job(attempt, configuration={}, host_memory_lease_factory=factory,
            kind='coding', chromosome_index=0, job_index=0,
            writer=SimpleNamespace(write=lambda *a, **kw: writes.append(a)), report={})
    assert released == [True] and not waits and not writes


def test_publish_failure_is_outside_retry_handler(monkeypatch):
    from fudan_wgs_toolkit.phewas_resources import HostMemoryDeferred
    attempts, writes, waits = [], [], []
    def attempt(configuration, current_writer):
        attempts.append(True)
        current_writer.write('coding', 0, '1', [{'p': .25}])
    def write(*a, **kw):
        writes.append(True)
        raise HostMemoryDeferred(10, available_bytes=5)
    monkeypatch.setattr(runtime.time, 'sleep', waits.append)
    with pytest.raises(HostMemoryDeferred):
        runtime._execute_batched_job(attempt, configuration={}, host_memory_lease_factory=None,
            kind='coding', chromosome_index=0, job_index=0,
            writer=SimpleNamespace(write=write), report={})
    assert attempts == writes == [True] and not waits


def test_private_gene_schema_maps_main_labels_without_recomputing_values():
    probability = torch.tensor(.125, dtype=torch.float64)
    source = {'WGS-O': probability, 'WGS-B(1,25)': .25, 'WGS-S(1,1)': .5,
              'WGS-A(1,25)': .75, 'SKAT(1,25)': .2, 'num_variant': 4}
    mapped = runtime._phewas_result_fields(source)
    assert mapped['STAAR-O'] is probability
    assert mapped == {'STAAR-O': probability, 'STAAR-B(1,25)': .25,
                      'STAAR-S(1,1)': .5, 'STAAR-A(1,25)': .75,
                      'SKAT(1,25)': .2, 'num_variant': 4}
    assert 'WGS-O' in source and 'STAAR-O' not in source
    assert runtime._phewas_result_fields(None) is None
    with pytest.raises(ValueError, match='conflicting public and private'):
        runtime._phewas_result_fields({'WGS-O': .25, 'STAAR-O': .5})


def test_modern_prepared_population_automatically_uses_logical_sample_rows(tmp_path):
    raw = fixture_states(m=9, n=11)
    physical_rows = np.arange(100, 111)[::-1].copy()
    reader = MetadataReader(11, len(raw))
    reader.sample_axis_kind = 'prepared_population'
    with SharedStateBroker(reader, make_cache(tmp_path/'cache', raw, physical_rows), device='cpu') as broker:
        samples, columns = np.asarray([10, 2, 4]), np.asarray([2, 5, 3])
        assert np.array_equal(broker.samples, np.arange(11))
        block = broker.trait_block(broker.read_states(columns), samples)
        assert_block(block, oracle(raw, np.arange(11), samples, columns), samples)
