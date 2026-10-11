"""Exact bounded-host/device preparation contracts, not real-data benchmarks."""
from contextlib import contextmanager
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.genotype_device import DeviceMinorBlock
from fudan_wgs_toolkit.masks import annotation_phred_matrix
from fudan_wgs_toolkit.pipeline import AnalysisOptions
from fudan_wgs_toolkit.phewas_runtime import runtime
from fudan_wgs_toolkit.phewas_runtime.shared_state import SharedStateBroker
from test_phewas_batched_runtime import Repository, descriptor, model_for
from test_phewas_shared_state import MetadataReader, make_cache


def gene_states():
    n, m = 512, 12
    raw = np.zeros((m, n), dtype=np.uint8)
    raw[0, 3] = 1
    raw[1, 7:10] = [1, 4, 5]
    raw[2] = 2
    raw[2, 9] = 1  # Reference is the minor allele: extraction group 2.
    raw[3] = 1  # Common on every cohort, therefore excluded.
    raw[4, 11] = 1
    raw[4, 50:65] = 3  # Rare with >1% missing alleles: group 1.
    raw[5, :16] = 1  # Rare on full cohort, common on the 128-row cohort.
    raw[6, 18:20] = [4, 5]  # Half calls keep source AF but missing dosage.
    raw[7, 20:23] = 1
    raw[8] = 0  # Monomorphic, excluded.
    raw[9] = 3  # Entirely missing, excluded.
    raw[10, 300:303] = 1  # No minor allele in the smaller cohort.
    raw[11, 30] = 1
    return raw


@contextmanager
def fixture_pipeline(path, *, device, rows, imputation='mean', maximum=None,
                     prefilter_max=1_000_000, retained_max=1_000_000):
    raw = gene_states()
    options = AnalysisOptions(wrapper_semantics='base', genotype_block_size=3,
        rare_maf_cutoff=.05, imputation=imputation,
        rv_num_cutoff_max_prefilter=prefilter_max, rv_num_cutoff_max=retained_max)
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)),
            make_cache(path, raw), device=device) as broker:
        axis = broker.reader_view(rows)
        try:
            model = model_for(axis._axis.samples, device=device)
            configuration = dict(annotation_names=['CADD', 'aPC.LocalDiversity'],
                maximum_mask_variants=maximum)
            p = runtime._thin_pipeline(axis, model, axis._axis.samples, options, configuration)
            cadd = np.linspace(1., 17., len(raw))
            cadd[7] = np.nan
            diversity = np.linspace(3., 9., len(raw))
            p.annotations = lambda selected, **kwargs: SimpleNamespace(annotations={
                'CADD': cadd[selected], 'aPC.LocalDiversity': diversity[selected]})
            yield p, broker
        finally:
            axis.close()


def assert_same_preparation(actual, expected):
    assert actual is not None and expected is not None
    for key in ('maf', 'mac', 'annotations', '_variant_indices', '_extraction_groups'):
        np.testing.assert_array_equal(actual[key], expected[key])
    for key in ('names', 'cmac', 'acat_calibration', 'rare_maf_cutoff',
                'rv_num_cutoff', 'rv_num_cutoff_max'):
        assert actual[key] == expected[key]
    dosage = actual['_genotype_device']
    assert dosage.dtype == torch.float32
    assert dosage.stride() == (1, dosage.shape[0])
    assert expected['_genotype_host'].dtype == np.float32
    assert expected['_genotype_host'].flags.f_contiguous
    np.testing.assert_array_equal(dosage.cpu().numpy(), expected['_genotype_host'])


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable for exact preparation contracts'))])
@pytest.mark.parametrize('imputation', ['mean', 'minor'])
@pytest.mark.parametrize('cohort', ['reversed', 'subset'])
def test_device_gene_preserves_cohort_af_half_calls_layout_groups_and_annotations(
        tmp_path, monkeypatch, device, imputation, cohort):
    rows = np.arange(512)[::-1].copy() if cohort == 'reversed' else np.arange(128)[::-1].copy()
    indices = np.asarray([5, 2, 9, 4, 1, 6, 0, 8, 3, 7, 10, 11], dtype=np.int64)
    with fixture_pipeline(tmp_path/'host', device='cpu', rows=rows, imputation=imputation) as (p, broker):
        expected = runtime._prepare_shared_gene(p, indices, broker)
        mature = p._prepare_test_set(indices, _defer_score=True)[0]
        np.testing.assert_array_equal(expected['_genotype_host'], mature['_genotype_host'].numpy())
        np.testing.assert_array_equal(expected['_variant_indices'], mature['_variant_indices'])
    with fixture_pipeline(tmp_path/'device', device=device, rows=rows, imputation=imputation) as (p, broker):
        annotations = p.annotations(indices).annotations
        phred, names = annotation_phred_matrix(annotations, p.annotation_names,
            variant_type=p.options.variant_type, number_variants=len(indices))
        # The direct route must gather into one final F-strided G. Calling
        # trait_dense would rebuild the removed full floating block.
        monkeypatch.setattr(DeviceMinorBlock, 'trait_dense', lambda *args, **kwargs:
            pytest.fail('device preparation rebuilt a full floating block'))
        host_guard_calls = []
        p.host_memory_guard = lambda **request: host_guard_calls.append(request)
        actual = runtime._prepare_shared_gene_device(p, indices, broker, phred, names)
        assert_same_preparation(actual, expected)
        assert not host_guard_calls  # This route owns no floating host G.
        selected = actual['_variant_indices'].tolist()
        assert 3 not in selected and 8 not in selected and 9 not in selected
        if cohort == 'subset':
            assert 5 not in selected and 10 not in selected
        else:
            assert 5 in selected and 10 in selected
        assert actual['_extraction_groups'].tolist() == sorted(actual['_extraction_groups'])
        assert {0, 1, 2}.issubset(set(actual['_extraction_groups']))


@pytest.mark.parametrize('bound', ['prefilter', 'retained', 'explicit_limit'])
def test_device_gene_keeps_mature_variant_limit_error_and_skip_semantics(tmp_path, bound):
    config = dict(maximum=2) if bound == 'explicit_limit' else (
        dict(prefilter_max=2) if bound == 'prefilter' else dict(retained_max=3))
    indices = np.arange(12)
    with fixture_pipeline(tmp_path/'host', device='cpu', rows=np.arange(512), **config) as (p, broker):
        if bound == 'explicit_limit':
            expected = runtime._prepare_shared_gene(p, indices, broker)
            skipped = p.skipped_sets.copy()
        else:
            with pytest.raises(ValueError) as caught:
                runtime._prepare_shared_gene(p, indices, broker)
            expected_error = str(caught.value)
    with fixture_pipeline(tmp_path/'device', device='cpu', rows=np.arange(512), **config) as (p, broker):
        phred, names = annotation_phred_matrix(p.annotations(indices).annotations,
            p.annotation_names, number_variants=len(indices))
        if bound == 'explicit_limit':
            assert expected is None
            assert runtime._prepare_shared_gene_device(p, indices, broker, phred, names) is None
            assert p.skipped_sets == skipped
        else:
            with pytest.raises(ValueError, match=expected_error):
                runtime._prepare_shared_gene_device(p, indices, broker, phred, names)


@pytest.mark.parametrize('error_type', [MemoryError, torch.cuda.OutOfMemoryError])
def test_optional_device_route_releases_exception_owners_before_bounded_host_fallback(
        tmp_path, monkeypatch, error_type):
    indices = np.arange(12)
    with fixture_pipeline(tmp_path/'cache', device='cpu', rows=np.arange(512)) as (p, broker):
        expected = runtime._prepare_shared_gene(p, indices, broker)
        owner = []
        def fail_resident(*args, **kwargs):
            temporary = torch.ones(16)
            owner.append(weakref.ref(temporary))
            raise error_type('simulated resident allocation rejection')
        monkeypatch.setattr(runtime, '_prepare_shared_gene_device', fail_resident)
        # Only the admission branch is forced here. All subsequent state and
        # bounded-host work use real CPU DeviceMinorBlock implementations.
        routed = SimpleNamespace(device='cuda:0', read_states=broker.read_states,
            trait_block=broker.trait_block)
        ledger = []
        def host_guard(**request):
            assert owner and owner[0]() is None
            ledger.append(request)
        p.host_memory_guard = host_guard
        p.host_memory_set_future = lambda *args, **kwargs: None
        actual = runtime._prepare_shared_gene(p, indices, routed)
        np.testing.assert_array_equal(actual['_genotype_host'], expected['_genotype_host'])
        for key in ('maf', 'mac', '_variant_indices', '_extraction_groups', 'annotations'):
            np.testing.assert_array_equal(actual[key], expected[key])
        assert actual['cmac'] == expected['cmac']
        assert ledger and all(request['phase'] == 'prepare' for request in ledger)
        assert ledger[-1]['m'] == len(actual['maf'])
        assert ledger[-1]['materialized_bytes'] >= actual['_genotype_host'].nbytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual resident-device fallback owner contract')
@pytest.mark.parametrize('family', ['gaussian', 'binomial'])
def test_gene_tile_rejected_resident_workspace_materializes_host_before_covariance(
        tmp_path, monkeypatch, family):
    n, m = 512, 12
    rows, phases, ledger, writes, owner = np.arange(n), [], [], [], []
    reservations = []
    class WorkspaceRepository(Repository):
        @contextmanager
        def reserve_workspace(self, device, required_bytes, **kwargs):
            reservations.append(required_bytes)
            if required_bytes:
                raise MemoryError('optional resident peak was admitted outside the recovery handler')
            def replace(required, *, phase):
                phases.append(phase)
                if phase == 'gene_genotype_upload':
                    raise MemoryError('simulated resident covariance workspace rejection')
            yield SimpleNamespace(set_required=replace, ensure_free=lambda *a, **kw: None)
    ds = [descriptor(0, rows, family=family)]
    model = model_for(rows, device='cuda:0');model.family = family
    repository = WorkspaceRepository(ds, [model])
    _, groups = runtime._descriptor_groups(repository)
    native_prepare = runtime._prepare_shared_gene
    def observed_prepare(*args):
        prepared = native_prepare(*args)
        assert '_genotype_device' in prepared
        owner.append(weakref.ref(prepared['_genotype_device']))
        return prepared
    def bounded_covariance(self, model, host, **kwargs):
        assert isinstance(host, np.ndarray) and host.flags.f_contiguous
        assert owner[0]() is None  # Original device G lost every live owner.
        assert 'gene_host_fallback' in phases
        assert ('future', 0, 'gene_host_fallback_materialized') in ledger
        return None, torch.eye(host.shape[1], device='cuda:0')
    monkeypatch.setattr(runtime, '_prepare_shared_gene', observed_prepare)
    monkeypatch.setattr(runtime.LimitedMaskPipeline, '_select_mask_chunks',
        lambda self, *args: {'ncRNA': np.arange(m)})
    monkeypatch.setattr(runtime.LimitedMaskPipeline, '_long_mask_products', bounded_covariance)
    monkeypatch.setattr(runtime, '_gaussian_covariance', lambda *a, **kw:
        pytest.fail('rejected resident G remained selected for covariance'))
    monkeypatch.setattr(runtime, '_shared_gene_scores', lambda host, models, broker:
        torch.zeros((host.shape[1], len(models)), device='cuda:0'))
    monkeypatch.setattr(runtime.LimitedMaskPipeline, '_evaluate_prepared',
        lambda self, payload, model: dict(num_variant=len(payload['maf']),
            cMAC=payload['cmac'], **{'WGS-O': .25}))
    if family == 'binomial':
        from fudan_wgs_toolkit import binary
        monkeypatch.setattr(binary, 'prepare_binary_burdens', lambda *a, **k: object())
        def ordinary_binary(fitted, host, *args, **kwargs):
            assert isinstance(host, np.ndarray) and host.flags.f_contiguous
            assert owner[0]() is None and 'gene_host_fallback' in phases
            assert ('future', 0, 'gene_host_fallback_materialized') in ledger
            selected = torch.zeros(1, dtype=torch.bool, device='cuda:0')
            # Native binary statistics include the fields consumed by the
            # real ncRNA record formatter, just as the Gaussian fixture does.
            return dict(num_variant=m, cMAC=0, **{'STAAR-B': .25}), dict(spa_selected=selected,
                spa=SimpleNamespace(failed=selected), variant_covariance_constructed=False)
        monkeypatch.setattr(binary, 'staar_binary_phewas', ordinary_binary)

    configuration = dict(host_memory_guard=lambda **request: ledger.append(('guard', request)),
        _host_memory_set_future=lambda value, **kw: ledger.append(('future', value, kw['phase'])))
    with fixture_pipeline(tmp_path/'cache', device='cuda:0', rows=rows) as (p, broker):
        structural = runtime._thin_pipeline(p.genotype, repository.models[0], rows,
            p.options, configuration)
        structural.models = [SimpleNamespace(device=broker.device, use_spa=False)]
        # Annotation and genomic selection are fixed at the small fixture;
        # actual DeviceMinorBlock and D2H fallback are exercised above.
        structural.annotations = p.annotations
        report = dict(association_rows={'ncrna': 0}, trait_rows={0: {'ncrna': 0}},
            binary_spa_selected=0, binary_spa_failed=0)
        runtime._batched_gene(groups, repository, broker, broker._reader, p.options,
            configuration, dict(gene_name='fixture', start=0, end=m-1),
            SimpleNamespace(write=lambda *a, **kw: writes.append(a)), report,
            set(), '21', 0, 2, structural, 'ncrna')
        assert len(writes) == 1 and report['association_rows']['ncrna'] == 1
        assert owner[0]() is None and repository.active == 0
        assert reservations == [0, 0]
        assert any(item[1]['materialized_bytes']==0 for item in ledger if item[0]=='guard')
        assert ledger[-1] == ('future', 0, 'gene_mask_released')
