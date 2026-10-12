"""CPU semantic contracts; synthetic values are not GPU benchmarks."""
from collections import OrderedDict
import numpy as np
import pytest
import torch
from fudan_wgs_toolkit.local_mask_reuse import ordered_mask_columns, valid_index_sets
from fudan_wgs_toolkit.pipeline import PheWASPipeline, AnalysisOptions
from fudan_wgs_toolkit.null_model import fit_gaussian_null
from fudan_wgs_toolkit.masks import VariantAnnotations
from fudan_wgs_toolkit.profiling import StageProfiler


def pipeline_fixture(*, mode="tf32", max_count=100, max_prefilter=100):
    pipeline = PheWASPipeline.__new__(PheWASPipeline)
    model = fit_gaussian_null(np.arange(8.) ** .8, sample_ids=np.arange(8).astype(str))
    model.set_matmul_mode(mode)  # Mock products; no CUDA/TF32 execution here.
    pipeline.models = [model]
    pipeline.options = AnalysisOptions(rare_maf_cutoff=.2, genotype_block_size=2,
        wrapper_semantics="base", rv_num_cutoff_max=max_count,
        rv_num_cutoff_max_prefilter=max_prefilter)
    pipeline.union_rows = np.arange(8)
    pipeline.trait_rows = [np.arange(8)]
    pipeline.annotation_names = ["Generic.Score", "CADD"]
    pipeline.profiler = StageProfiler("cpu", enabled=False)
    pipeline.statistics_execution = "serial"
    pipeline.local_mask_reuse = False
    pipeline.local_mask_reuse_counters = dict.fromkeys(["families", "union_score_calls", "reused_masks",
        "input_variant_columns", "union_variant_columns", "prepared_union_variant_columns", "fallback_memory", "fallback_unsupported", "fallback_mapping", "cache_hits", "duplicate_mask_hits", "single_mask_paths", "fallback_cuda_oom", "fallback_geometry", "geometry_checks", "geometry_union_covariance_cells", "geometry_mask_covariance_cells"], 0)
    pipeline._test_set_cache = OrderedDict()
    # Columns include flipped orientation, missing-driven extractor grouping,
    # insufficient masks and nonrare variants. Imputed values are fractional.
    genotype = np.arange(48, dtype=np.float64).reshape(8, 6) / 73
    genotype[2, 1] = .03125
    frequencies = np.array([.004, .06, .08, .1, .35, .15])
    ref_af = np.array([.996, .06, .92, .9, .65, .85])
    missing = np.array([0, 0, .03, 0, 0, 0])
    pipeline.reads = []; pipeline.score_modes = []
    class Block:
        def __init__(self, columns):
            self.columns = columns
            self.union_ref_af = ref_af[columns]
        def allele_missing_rate(self):
            return missing[self.columns]
        def trait_dense(self, rows, imputation, **kwargs):
            assert imputation == "mean" and kwargs == {"frequency_mode": "reference"}
            return genotype[np.ix_(rows, self.columns)], frequencies[self.columns], None, None, None
    def blocks(indices, rows, block_size):
        pipeline.reads.append(np.asarray(indices).copy())
        for offset in range(0, len(indices), block_size):
            yield Block(np.asarray(indices)[offset:offset + block_size])
    pipeline._minor_blocks = blocks
    annotation_values = np.array([4., 1., 6., 3., 8., 2.])
    pipeline.annotations = lambda indices, **kwargs: VariantAnnotations(np.asarray(indices), np.ones(len(indices)),
        {"Generic.Score": annotation_values[np.asarray(indices)], "CADD": np.array([np.nan, 2., 3., 4., 5., 6.])[np.asarray(indices)]})
    def score(values, **kwargs):
        pipeline.score_modes.append(model.matmul_mode)
        values = torch.as_tensor(values)
        # Exact per-column mock isolates indexing/preparation contracts from
        # CPU BLAS output-shape-dependent rounding (not a TF32 substitute).
        columns = [values[:, index].contiguous() for index in range(values.shape[1])]
        return torch.stack([column.sum() for column in columns]), torch.stack([
            torch.stack([(left * right).sum() for right in columns]) for left in columns])
    model.score_covariance = score
    def evaluate(payload, unused_model):
        if payload is None:
            return None
        return {key: (value.clone() if isinstance(value, torch.Tensor) else value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in payload.items()}
    pipeline._evaluate_prepared = evaluate
    return pipeline


def assert_payload_equal(expected, actual):
    assert len(expected) == len(actual)
    for reference, candidate in zip(expected, actual):
        assert (reference[0] is None) == (candidate[0] is None)
        if reference[0] is None:
            continue
        assert reference[0].keys() == candidate[0].keys()
        for key, value in reference[0].items():
            other = candidate[0][key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other), key
            elif isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, other, err_msg=key)
            else:
                assert value == other, key


def test_union_reads_once_scores_once_preserves_original_mask_order_and_cmac():
    masks = [np.array([3, 1, 0]), np.array([2, 0, 1]), np.array([4, 5])]
    original = pipeline_fixture(); candidate = pipeline_fixture()
    expected = original._run_mask_sets(masks)
    candidate.local_mask_reuse = True
    actual = candidate._run_mask_sets(masks)
    assert_payload_equal(expected, actual)
    assert len(original.reads) == 3 and len(candidate.reads) == 1
    assert len(original.score_modes) == 2 and candidate.score_modes == ["tf32"]
    assert candidate.local_mask_reuse_counters["reused_masks"] == 2
    assert actual[-1] == [None]
    assert candidate.local_mask_reuse_counters["prepared_union_variant_columns"] == 4


def test_each_mask_limits_apply_independently_not_to_union():
    masks = [np.array([0, 1, 2]), np.array([1, 2, 3]), np.array([2, 3, 5])]
    candidate = pipeline_fixture(max_count=4, max_prefilter=4)
    candidate.local_mask_reuse = True
    results = candidate._run_mask_sets(masks)
    assert all(row[0] is not None for row in results)
    assert candidate.local_mask_reuse_counters["union_variant_columns"] == 5
    with pytest.raises(ValueError, match="prefilter"):
        candidate._run_mask_sets([np.array([0, 1, 2, 3]), np.array([3, 5])])


def test_union_memory_rejection_returns_same_mode_original_path(monkeypatch):
    masks = [np.array([0, 1, 2]), np.array([1, 2, 3])]
    candidate = pipeline_fixture(mode="tf32")
    candidate.local_mask_reuse = True
    original_limit = candidate._limit
    def limited(model, number_variants, **kwargs):
        if number_variants > 3:
            raise MemoryError("union only exceeds budget")
        original_limit(model, number_variants, **kwargs)
    monkeypatch.setattr(candidate, "_limit", limited)
    candidate._run_mask_sets(masks)
    assert candidate.local_mask_reuse_counters["fallback_memory"] == 1
    assert candidate.score_modes == ["tf32", "tf32"]
    assert len(candidate.reads) == 1  # Reuse filtered/imputed host columns.
    assert candidate.local_mask_reuse_counters["fallback_host_reused_masks"] == 2
    assert candidate._local_union_host_prepared is None


def test_fp64_control_and_duplicate_mappings_use_original_path():
    candidate = pipeline_fixture(mode="fp64")
    candidate.local_mask_reuse = True
    candidate._run_mask_sets([np.array([0, 1]), np.array([2, 3])])
    assert candidate.local_mask_reuse_counters["fallback_unsupported"] == 1
    assert candidate.score_modes == ["fp64", "fp64"]
    candidate = pipeline_fixture(); candidate.local_mask_reuse = True
    candidate._run_mask_sets([np.array([0, 0, 1]), np.array([2, 3])])
    assert candidate.local_mask_reuse_counters["fallback_mapping"] == 1
    assert candidate.score_modes == ["tf32", "tf32"]


def test_index_mapping_keeps_input_order_then_stable_group_order():
    physical = np.array([0, 2, 5, 1])
    groups = np.array([0, 1, 1, 2])
    np.testing.assert_array_equal(ordered_mask_columns(physical, groups, [5, 1, 2, 0], grouped=True), [0, 2, 1, 3])
    np.testing.assert_array_equal(ordered_mask_columns(physical, groups, [5, 1, 2, 0], grouped=False), [2, 3, 1, 0])
    assert valid_index_sets([np.array([3, 1]), np.array([], dtype=np.int64)])
    assert not valid_index_sets([np.array([1, 1])])
    assert not valid_index_sets([np.array([1.])])


def test_pipeline_window_entry_removed():
    assert not hasattr(PheWASPipeline, "sliding")


def test_original_semantic_rare_count_error_is_not_swallowed():
    candidate = pipeline_fixture(max_count=3)
    candidate.local_mask_reuse = True
    with pytest.raises(ValueError, match="rare variant count"):
        candidate._run_mask_sets([np.array([0, 1, 2]), np.array([3, 5])])
    assert candidate.local_mask_reuse_counters["fallback_memory"] == 0
    assert candidate.score_modes == []


@pytest.mark.parametrize("enabled", [False, True])
def test_weight_batch_flag_forwarded_to_independent_mask_statistics(monkeypatch, enabled):
    import fudan_wgs_toolkit.pipeline as module
    candidate = pipeline_fixture()
    candidate.weight_batch_optimization = enabled
    candidate.statistics_tail_optimization = False
    captured = {}
    monkeypatch.setattr(module, "association_test", lambda **kwargs: captured.update(kwargs) or {})
    # Use the original evaluator instead of the index-contract mock.
    PheWASPipeline._evaluate_prepared(candidate, {}, candidate.models[0])
    assert captured["weight_batch_optimization"] is enabled
    assert captured["matmul_mode"] == "tf32"


def test_prior_cache_hit_is_excluded_from_union_and_result_is_copied():
    candidate = pipeline_fixture(); candidate.local_mask_reuse = True
    cached = np.array([0, 1]); first = candidate.test_set(cached)
    candidate.reads.clear(); candidate.score_modes.clear()
    results = candidate._run_mask_sets([np.array([2, 3]), cached, np.array([2, 3, 5])])
    assert_payload_equal([first], [results[1]])
    assert results[1][0] is not first[0]
    np.testing.assert_array_equal(candidate.reads[0], [2, 3, 5])
    assert candidate.local_mask_reuse_counters["cache_hits"] == 1
    assert len(candidate.score_modes) == 1


def test_duplicate_family_masks_tail_once_and_restore_original_positions():
    candidate = pipeline_fixture(); candidate.local_mask_reuse = True
    masks = [np.array([0, 1]), np.array([2, 3]), np.array([0, 1]), np.array([4, 5]), np.array([2, 3])]
    evaluated = []; evaluate = candidate._evaluate_prepared
    candidate._evaluate_prepared = lambda payload, model: evaluated.append(payload) or evaluate(payload, model)
    results = candidate._run_mask_sets(masks)
    assert sum(payload is not None for payload in evaluated) == 2  # NULL has no tail work.
    assert_payload_equal([results[0], results[1]], [results[2], results[4]])
    assert results[0][0] is not results[2][0]
    assert candidate.local_mask_reuse_counters["duplicate_mask_hits"] == 2
    assert len(candidate._test_set_cache) == 3
    assert candidate._test_set_cache[candidate._set_key(masks[3])] == [None]


def test_all_cached_needs_no_reads_products_or_tail():
    candidate = pipeline_fixture(); candidate.local_mask_reuse = True
    masks = [np.array([0, 1]), np.array([4, 5])]
    expected = [candidate.test_set(indices) for indices in masks]
    candidate.reads.clear(); candidate.score_modes.clear()
    candidate._evaluate_prepared = lambda *args: pytest.fail("cached tail rerun")
    assert_payload_equal(expected, candidate._run_mask_sets(masks))
    assert candidate.reads == [] and candidate.score_modes == []
    assert candidate.local_mask_reuse_counters["cache_hits"] == 2
    assert candidate.local_mask_reuse_counters["union_score_calls"] == 0


def test_one_uncached_unique_uses_original_test_set_without_union():
    candidate = pipeline_fixture(); candidate.local_mask_reuse = True
    cached = np.array([0, 1]); candidate.test_set(cached)
    candidate.reads.clear(); candidate.score_modes.clear()
    missing = np.array([2, 3])
    results = candidate._run_mask_sets([cached, missing, missing])
    assert_payload_equal([results[1]], [results[2]])
    np.testing.assert_array_equal(candidate.reads[0], missing)
    assert len(candidate.score_modes) == 1
    assert candidate.local_mask_reuse_counters["single_mask_paths"] == 1
    assert candidate.local_mask_reuse_counters["union_score_calls"] == 0


def test_different_mask_input_orders_are_not_cache_aliases():
    candidate = pipeline_fixture(); candidate.local_mask_reuse = True
    masks = [np.array([2, 3]), np.array([3, 2])]
    expected_pipeline = pipeline_fixture()
    expected = expected_pipeline._run_mask_sets(masks)
    results = candidate._run_mask_sets(masks)
    assert_payload_equal(expected, results)
    assert candidate.local_mask_reuse_counters["duplicate_mask_hits"] == 0
    assert candidate.local_mask_reuse_counters["reused_masks"] == 2
    assert len(candidate._test_set_cache) == 2


def test_union_gpu_guard_uses_actual_eligible_rare_not_raw_candidates(monkeypatch):
    candidate = pipeline_fixture(); candidate.local_mask_reuse = True
    observed_counts = []; original_limit = candidate._limit
    def limit(model, number_variants, **kwargs):
        observed_counts.append(number_variants)
        assert number_variants <= 4, "nonrare candidate incorrectly budgeted as dense V"
        return original_limit(model, number_variants, **kwargs)
    monkeypatch.setattr(candidate, "_limit", limit)
    candidate._run_mask_sets([np.array([0, 1, 2, 4]), np.array([1, 2, 3, 4])])
    assert observed_counts == [4]
    assert candidate.local_mask_reuse_counters["union_variant_columns"] == 5
    assert candidate.local_mask_reuse_counters["prepared_union_variant_columns"] == 4
    assert candidate.local_mask_reuse_counters["fallback_memory"] == 0


def test_covariance_geometry_rejects_disjoint_but_accepts_nested_rare_masks():
    from fudan_wgs_toolkit.local_mask_reuse import union_covariance_geometry
    disjoint = union_covariance_geometry([0, 1, 2, 3], [[0, 1], [2, 3]], minimum_variants=2)
    assert disjoint == {"union_covariance_cells": 16, "mask_covariance_cells": 8, "beneficial": False}
    nested = union_covariance_geometry([0, 1, 2], [[0, 1, 2], [1, 2]], minimum_variants=2)
    assert nested["beneficial"] and nested["mask_covariance_cells"] == 13


def test_unprofitable_rare_union_falls_back_before_union_gpu_guard(monkeypatch):
    candidate = pipeline_fixture(); candidate.local_mask_reuse = True
    candidate.models[0].spectrum.blocks=[(torch.tensor([0,1]),torch.eye(2))]  # excluded tile scope
    guarded = []; original_limit = candidate._limit
    monkeypatch.setattr(candidate, "_limit", lambda model, count, **kwargs:
        guarded.append(count) or original_limit(model, count, **kwargs))
    results = candidate._run_mask_sets([np.array([0, 1, 4]), np.array([2, 3, 4])])
    assert all(row[0] is not None for row in results)
    assert guarded == [2, 2]  # Never raw5 or rare4 covariance allocation.
    assert candidate.local_mask_reuse_counters["fallback_geometry"] == 1
    assert candidate.local_mask_reuse_counters["geometry_checks"] == 1
    assert candidate.local_mask_reuse_counters["geometry_union_covariance_cells"] == 16
    assert candidate.local_mask_reuse_counters["geometry_mask_covariance_cells"] == 8
    assert candidate.local_mask_reuse_counters["union_score_calls"] == 0
    assert candidate.score_modes == ["tf32", "tf32"]


@pytest.mark.parametrize("failure, counter", [(torch.OutOfMemoryError, "fallback_cuda_oom"),
                                                (MemoryError, "fallback_memory")])
def test_failed_score_frames_and_temporaries_released_before_same_mode_fallback(failure, counter):
    import weakref, sys
    candidate = pipeline_fixture(); candidate.local_mask_reuse = True
    original_score = candidate.models[0].score_covariance
    references = []; calls = 0
    def failing_score(genotype, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            temporary = torch.ones(16)
            references.extend([weakref.ref(genotype), weakref.ref(temporary)])
            # A cycle exercises explicit gc beyond normal frame refcounting.
            cycle = [temporary]; cycle.append(cycle)
            raise failure("mock allocation failure")
        assert sys.exc_info()[1] is None, "fallback invoked inside active except"
        assert all(reference() is None for reference in references), "failed tensor frame still alive"
        return original_score(genotype, **kwargs)
    candidate.models[0].score_covariance = failing_score
    results = candidate._run_mask_sets([np.array([0, 1, 2]), np.array([1, 2, 3])])
    assert all(row[0] is not None for row in results)
    assert candidate.local_mask_reuse_counters[counter] == 1
    assert candidate.score_modes == ["tf32", "tf32"]
    assert calls == 3


def test_oom_during_mask_tail_releases_union_partial_results_before_retry():
    import weakref, sys
    candidate = pipeline_fixture(); candidate.local_mask_reuse = True
    evaluate = candidate._evaluate_prepared; references = []; calls = 0
    def tail(payload, model):
        nonlocal calls
        calls += 1
        if calls == 1:
            result = evaluate(payload, model)
            references.append(weakref.ref(result["covariance"]))
            return result
        if calls == 2:
            references.append(weakref.ref(payload["covariance"]))
            raise torch.OutOfMemoryError("mock tail allocation failure")
        assert sys.exc_info()[1] is None
        assert all(reference() is None for reference in references)
        return evaluate(payload, model)
    candidate._evaluate_prepared = tail
    candidate._run_mask_sets([np.array([0, 1, 2]), np.array([1, 2, 3])])
    assert candidate.local_mask_reuse_counters["fallback_cuda_oom"] == 1
    assert candidate.local_mask_reuse_counters["union_score_calls"] == 1  # Attempt really computed U/V.
    assert candidate.local_mask_reuse_counters["reused_masks"] == 0  # Its partial output was discarded.
    assert calls == 4


def test_union_recovery_does_not_retry_a_failing_individual_mask():
    candidate = pipeline_fixture(); candidate.local_mask_reuse = True
    calls = 0
    def failing_score(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise torch.OutOfMemoryError("always fails")
    candidate.models[0].score_covariance = failing_score
    with pytest.raises(torch.OutOfMemoryError, match="always fails"):
        candidate._run_mask_sets([np.array([0, 1, 2]), np.array([1, 2, 3])])
    assert calls == 2  # One union attempt, one original-mask failure; no loop.


def test_cuda_recovery_cleanup_order_without_initializing_cuda(monkeypatch):
    import gc
    from contextlib import nullcontext
    from fudan_wgs_toolkit.local_mask_reuse import release_failed_union
    events = []
    monkeypatch.setattr(gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append(("sync", device)))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    release_failed_union("cuda:0")
    assert events == ["gc", ("sync", "cuda:0"), "empty_cache"]


def test_nonallocation_union_exception_propagates_without_cleanup(monkeypatch):
    from fudan_wgs_toolkit import local_mask_reuse
    monkeypatch.setattr(local_mask_reuse, "release_failed_union", lambda device: pytest.fail("numerical error cleanup"))
    def invalid():
        raise ValueError("invalid score data")
    with pytest.raises(ValueError, match="invalid score data"):
        local_mask_reuse.attempt_union(invalid, device="cpu")


def test_geometry_fallback_reuses_host_preparation_and_matches_per_mask():
    masks=[np.array([0,1]),np.array([2,3])]
    original=pipeline_fixture();candidate=pipeline_fixture()
    candidate.models[0].spectrum.blocks=[(torch.tensor([0,1]),torch.eye(2))]  # test legacy gate
    expected=original._run_mask_sets(masks)
    candidate.local_mask_reuse=True
    actual=candidate._run_mask_sets(masks)
    assert_payload_equal(expected,actual)
    assert len(candidate.reads)==1
    assert candidate.local_mask_reuse_counters['fallback_geometry']==1
    assert candidate.local_mask_reuse_counters['fallback_host_reused_masks']==2
    assert candidate._local_union_host_prepared is None


@pytest.mark.parametrize('fallback', ['none', 'memory', 'geometry'])
def test_direct_host_fp32_gene_and_union_fallback_match_old_preparation(monkeypatch, fallback):
    from fudan_wgs_toolkit.genotype import SparseMinorBlock
    genotype = (np.arange(48).reshape(8, 6) % 3).astype(float)
    genotype[[1, 4, 7], [0, 2, 3]] = np.nan
    def make_pipeline():
        pipeline = pipeline_fixture()
        def blocks(indices, rows, block_size):
            pipeline.reads.append(np.asarray(indices).copy())
            for offset in range(0, len(indices), block_size):
                columns = np.asarray(indices)[offset:offset+block_size]
                values = genotype[:, columns]
                row, col = np.nonzero((values != 0) | np.isnan(values))
                yield SparseMinorBlock(row, col, values[row, col], np.arange(8),
                                       columns, np.full(len(columns), .95))
        pipeline._minor_blocks = blocks
        return pipeline
    masks = ([np.array([0, 1]), np.array([2, 3])] if fallback == 'geometry'
             else [np.array([0, 1, 2]), np.array([1, 2, 3])])
    original_dense = SparseMinorBlock.trait_dense
    def old_route(block, *args, **kwargs):
        kwargs.pop('dtype', None)
        return original_dense(block, *args, **kwargs)
    monkeypatch.setattr(SparseMinorBlock, 'trait_dense', old_route)
    reference = make_pipeline()._run_mask_sets(masks)
    requested = []
    def direct_route(block, *args, **kwargs):
        requested.append(kwargs.get('dtype'))
        return original_dense(block, *args, **kwargs)
    monkeypatch.setattr(SparseMinorBlock, 'trait_dense', direct_route)
    candidate = make_pipeline(); candidate.local_mask_reuse = True
    if fallback == 'geometry':
        candidate.models[0].spectrum.blocks=[(torch.tensor([0,1]),torch.eye(2))]
    if fallback == 'memory':
        original_limit = candidate._limit
        def limit(model, count, **kwargs):
            if count > 3:
                raise MemoryError('union exceeds fixture budget')
            return original_limit(model, count, **kwargs)
        candidate._limit = limit
    actual = candidate._run_mask_sets(masks)
    assert_payload_equal(reference, actual)
    assert requested and all(dtype == np.float32 for dtype in requested)
    assert len(candidate.reads) == 1
    if fallback != 'none':
        assert candidate.local_mask_reuse_counters['fallback_'+fallback] == 1
        assert candidate.local_mask_reuse_counters['fallback_host_reused_masks'] == 2
    assert all(item[0]['score'].dtype == torch.float32 for item in actual)




def resident_family_fixture(*, resident):
    """Equivalent sparse and device dosage views, independent of source SDKs."""
    from fudan_wgs_toolkit.genotype import SparseMinorBlock, _allele_frequency_summary
    from fudan_wgs_toolkit.genotype_device import DeviceMinorBlock
    calls = np.zeros((6, 8, 2), dtype=np.int64)
    calls[:, 0, 1] = 1
    calls[1:4, 1, 1] = 1
    calls[0, 6] = [0, 3]
    calls[2, 7] = [3, 3]
    calls[4, :, 1] = 1
    calls[5] = 3
    pipeline = pipeline_fixture()
    pipeline.trait_rows = [np.array([5, 2, 0, 6, 1, 4, 3, 7])]
    pipeline._resident_gene_reader_options = lambda indices: dict(device='cpu', resident=True) if resident else {}

    def blocks(indices, rows, block_size, **kwargs):
        pipeline.reads.append(np.asarray(indices).copy())
        for offset in range(0, len(indices), block_size):
            columns = np.asarray(indices)[offset:offset+block_size]
            values = calls[columns][:, rows, :]
            reference = (values == 0).sum(axis=(1, 2), dtype=np.int64)
            called = (values != 3).sum(axis=(1, 2), dtype=np.int64)
            af, missing, mac, ref_ac, called_ac = _allele_frequency_summary(reference, called, len(rows))
            alt = (values == 1).sum(axis=2).T
            dosage = np.where(af >= .5, alt, 2-alt).astype(np.uint8)
            dosage[(values == 3).any(axis=2).T] = 3
            if resident:
                yield DeviceMinorBlock(torch.as_tensor(dosage), np.asarray(rows), columns,
                                       af, mac, missing, ref_ac, called_ac)
            else:
                dense = dosage.astype(np.float64)
                dense[dosage == 3] = np.nan
                row, col = np.nonzero((dense != 0) | np.isnan(dense))
                yield SparseMinorBlock(row, col, dense[row, col], np.asarray(rows), columns,
                                       af, mac, missing, ref_ac, called_ac)
    pipeline._minor_blocks = blocks
    return pipeline


@pytest.mark.parametrize('semantics', ['base','phewas'])
@pytest.mark.parametrize('imputation', ['mean','minor'])
@pytest.mark.parametrize('fallback', ['none','geometry','memory','oom'])
def test_resident_gene_matches_host_masks_and_never_rereads(monkeypatch, semantics, imputation, fallback):
    from dataclasses import replace
    from fudan_wgs_toolkit.genotype_device import DeviceMinorBlock
    masks = ([np.array([1,0]),np.array([3,2])] if fallback=='geometry'
             else [np.array([2,0,1,4]),np.array([1,3,2,5])])
    host=resident_family_fixture(resident=False);candidate=resident_family_fixture(resident=True)
    for pipeline in (host,candidate):pipeline.options=replace(pipeline.options,wrapper_semantics=semantics,imputation=imputation)
    expected=host._run_mask_sets(masks)
    candidate.local_mask_reuse=True
    if fallback=='geometry':
        candidate.models[0].spectrum.blocks=[(torch.tensor([0,1]),torch.eye(2))]
    floating_columns=[]; original_dense=DeviceMinorBlock.trait_dense
    def dense(block,*args,**kwargs):
        floating_columns.extend(block.variant_indices.tolist())
        return original_dense(block,*args,**kwargs)
    monkeypatch.setattr(DeviceMinorBlock,'trait_dense',dense)
    if fallback=='memory':
        original_limit=candidate._limit
        def limit(model,count,**kwargs):
            if count>3:raise MemoryError('union budget')
            return original_limit(model,count,**kwargs)
        candidate._limit=limit
    if fallback=='oom':
        score=candidate.models[0].score_covariance
        def fail_union(g,**kwargs):
            if g.shape[1]>3:raise torch.OutOfMemoryError('product fixture')
            return score(g,**kwargs)
        candidate.models[0].score_covariance=fail_union
    actual=candidate._run_mask_sets(masks)
    # cMAC keeps the same sum definition; GPU/NumPy reduction bits are not promised.
    for lhs,rhs in zip(expected,actual):
        if lhs[0] is not None:
            assert abs(lhs[0].pop('cmac')-rhs[0].pop('cmac'))<1e-5
    assert_payload_equal(expected,actual)
    assert len(candidate.reads)==1 and all(column not in (4,5) for column in floating_columns)
    assert candidate._local_union_device_prepared is None
    if fallback!='none':
        counter='fallback_cuda_oom' if fallback=='oom' else 'fallback_'+fallback
        assert candidate.local_mask_reuse_counters[counter]==1
        assert candidate.local_mask_reuse_counters['fallback_device_reused_masks']==2


def test_resident_decoder_failure_propagates_without_cpu_reread():
    candidate=resident_family_fixture(resident=True);candidate.local_mask_reuse=True
    candidate._minor_blocks=lambda *a,**k: (_ for _ in ()).throw(torch.OutOfMemoryError('decoder'))
    with pytest.raises(torch.OutOfMemoryError,match='decoder'):
        candidate._run_mask_sets([np.array([0,1]),np.array([1,2])])
    assert candidate.local_mask_reuse_counters['fallback_cuda_oom']==0
    assert candidate._local_union_device_prepared is None


@pytest.mark.parametrize('binding',['cap','live','fits'])
def test_resident_gene_route_guard_uses_storage_and_decoder_before_read(monkeypatch,binding):
    import fudan_wgs_toolkit.pipeline as module
    from types import SimpleNamespace
    pipeline=PheWASPipeline.__new__(PheWASPipeline)
    pipeline.resident_genotypes=True;pipeline.union_rows=np.arange(100)
    pipeline.options=AnalysisOptions(memory_limit_gib=.5)
    pipeline.local_mask_reuse_counters={}
    pipeline.genotype=SimpleNamespace(_flat_reader=object(),genotype_raw_memory_bytes=1024,n_samples=1000)
    model=SimpleNamespace(n_pheno=1,use_spa=False,matmul_mode='tf32',device='cuda:0')
    pipeline.models=[model];monkeypatch.setattr(module,'GaussianNullModel',SimpleNamespace)
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda *a: 400*2**20 if binding=='cap' else 0)
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda *a: 400*2**20 if binding=='cap' else 0)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda *a:(1024 if binding=='live' else 2**30,2**30))
    result=pipeline._resident_gene_reader_options(np.arange(100))
    assert bool(result)==(binding=='fits')
    assert pipeline.local_mask_reuse_counters['resident_gene_storage_reserve_bytes_max']>256*2**20


def test_resident_phewas_subset_rows_match_original_frequency_and_no_dense_early(monkeypatch):
    from dataclasses import replace
    from fudan_wgs_toolkit.genotype_device import DeviceMinorBlock
    host=resident_family_fixture(resident=False);candidate=resident_family_fixture(resident=True)
    for pipeline in (host,candidate):
        pipeline.options=replace(pipeline.options,wrapper_semantics='phewas',rare_maf_cutoff=.4)
        pipeline.trait_rows=[np.array([7,0,3,1,6])]
        pipeline.models[0].sample_ids=pipeline.models[0].sample_ids[:5]
    masks=[np.array([0,1,2]),np.array([3,2,1])]
    expected=host._run_mask_sets(masks)
    candidate.local_mask_reuse=True
    original_limit=candidate._limit;guarded=[]
    def limit(model,count,**kwargs):
        guarded.append(count);return original_limit(model,count,**kwargs)
    candidate._limit=limit
    dense=DeviceMinorBlock.trait_dense
    def after_guard(block,*args,**kwargs):
        assert guarded,'floating slab allocated before actual rare guard'
        return dense(block,*args,**kwargs)
    monkeypatch.setattr(DeviceMinorBlock,'trait_dense',after_guard)
    actual=candidate._run_mask_sets(masks)
    for lhs,rhs in zip(expected,actual):
        if lhs[0] is not None:assert abs(lhs[0].pop('cmac')-rhs[0].pop('cmac'))<1e-5
    assert_payload_equal(expected,actual)
    assert len(candidate.reads)==1


def test_resident_failed_union_releases_floating_frame_before_fallback():
    import weakref
    candidate=resident_family_fixture(resident=True);candidate.local_mask_reuse=True
    original=candidate.models[0].score_covariance;refs=[]
    def products(g,**kwargs):
        if g.shape[1]>3:
            temporary=g.clone();refs.extend([weakref.ref(g),weakref.ref(temporary)])
            raise torch.OutOfMemoryError('union fixture')
        assert all(ref() is None for ref in refs)
        assert all(block.dosage.dtype==torch.uint8 for block in candidate._local_union_device_prepared['_resident_blocks'])
        return original(g,**kwargs)
    candidate.models[0].score_covariance=products
    candidate._run_mask_sets([np.array([2,0,1]),np.array([1,3,2])])
    assert refs and len(candidate.reads)==1


@pytest.mark.parametrize('semantics',['base','phewas'])
def test_resident_materialization_preserves_old_column_major_score_layout(semantics):
    from dataclasses import replace
    host=resident_family_fixture(resident=False);candidate=resident_family_fixture(resident=True)
    for pipeline in (host,candidate):pipeline.options=replace(pipeline.options,wrapper_semantics=semantics)
    indices=np.array([2,0,3,1])
    old=host._prepare_test_set(indices,_defer_score=True)[0]['_genotype_host']
    prepared=candidate._prepare_test_set(indices,_defer_score=True)[0]
    actual=candidate._materialize_resident_gene(prepared)
    assert old.flags.f_contiguous
    assert actual.stride()==torch.as_tensor(old).stride()==(1,8)
    assert actual.T.is_contiguous()  # G.T enters FP32 GEMV with original layout.
    assert actual.untyped_storage().nbytes()==actual.numel()*4
    assert np.array_equal(actual.numpy().view(np.uint32),old.view(np.uint32))
    columns=np.array([2,0,1])
    sliced=candidate._materialize_resident_gene(prepared,columns)
    assert sliced.stride()==torch.as_tensor(old[:,columns]).stride()==(1,8)
    assert np.array_equal(sliced.numpy().view(np.uint32),old[:,columns].view(np.uint32))


def test_small_tile_real_47_37_10_shape_work_uses_actual_backend(monkeypatch):
    from fudan_wgs_toolkit.local_mask_reuse import small_native_union_work
    import fudan_wgs_toolkit.tf32 as backend
    masks=[np.arange(37),np.arange(37,47)]
    cost=small_native_union_work(np.arange(47),masks,minimum_variants=2,samples=42652,covariates=20)
    assert cost['beneficial']
    assert cost['union']['covariance']['work']==4096*42656
    assert cost['separate']['covariance']==6144*42656
    assert cost['union']['score']['route']=='gemv'
    assert cost['union']['score']['work']==cost['separate']['score']==42652*47
    assert cost['union_calls']==5 and cost['separate_calls']==10
    assert all(value['work']<=cost['separate'][name] for name,value in cost['union'].items())
    monkeypatch.setattr(backend,'_native_geometry',lambda *a:(16,16,16,4,3))
    changed=small_native_union_work(np.arange(47),masks,minimum_variants=2,samples=42652,covariates=20)
    assert changed['union']['covariance']['work']==48*48*42656
    assert changed['union']['covariance']['work']!=cost['union']['covariance']['work']


@pytest.mark.parametrize('excluded',['large','zero_covariates','fp64','multiple','rotations','spa'])
def test_small_tile_gate_excludes_other_model_scopes(monkeypatch,excluded):
    import fudan_wgs_toolkit.pipeline as module
    from types import SimpleNamespace
    pipeline=PheWASPipeline.__new__(PheWASPipeline)
    pipeline.options=AnalysisOptions()
    pipeline.local_mask_reuse_counters=dict.fromkeys(['geometry_checks','geometry_union_covariance_cells','geometry_mask_covariance_cells'],0)
    m=65 if excluded=='large' else 47
    p=0 if excluded=='zero_covariates' else 20
    model=SimpleNamespace(n=42652,n_pheno=1,use_spa=excluded=='spa',
        matmul_mode='fp64' if excluded=='fp64' else 'tf32',x=torch.empty((1,p)),
        spectrum=SimpleNamespace(blocks=[object()] if excluded=='rotations' else []))
    monkeypatch.setattr(module,'GaussianNullModel',SimpleNamespace)
    pipeline.models=[model]*(2 if excluded=='multiple' else 1)
    result=pipeline._mask_union_geometry(np.arange(m),[np.arange(37),np.arange(37,m)],model)
    assert not result['beneficial']
    assert 'geometry_small_tile_checks' not in pipeline.local_mask_reuse_counters


def test_small_tile_gate_keeps_old_gate_and_rejects_mixed_vector_routes(monkeypatch):
    from fudan_wgs_toolkit.local_mask_reuse import small_native_union_work
    # One variant mask changes cross/covariance routes, so unlike compute
    # classes cannot be silently combined into a TF32 lane-cost comparison.
    cost=small_native_union_work(np.arange(4),[[0],[1,2,3]],minimum_variants=1,samples=42652,covariates=20)
    assert not cost['beneficial']
    candidate=resident_family_fixture(resident=True)
    candidate.models[0].x=torch.ones((8,2),dtype=torch.float32)
    old=candidate._mask_union_geometry(np.arange(3),[[0,1,2],[1,2]],candidate.models[0])
    assert old['beneficial'] and 'small_tile_work' not in old


def test_small_tile_accepts_actual_rare_decoded_union_without_second_read():
    host=resident_family_fixture(resident=False);candidate=resident_family_fixture(resident=True)
    candidate.models[0].x=torch.ones((8,2),dtype=torch.float32)
    masks=[np.array([1,0,4]),np.array([3,2,5])]
    expected=host._run_mask_sets(masks)
    candidate.local_mask_reuse=True
    actual=candidate._run_mask_sets(masks)
    for lhs,rhs in zip(expected,actual):
        if lhs[0] is not None:assert abs(lhs[0].pop('cmac')-rhs[0].pop('cmac'))<1e-5
    assert_payload_equal(expected,actual)
    counters=candidate.local_mask_reuse_counters
    assert counters['geometry_union_covariance_cells']==16
    assert counters['geometry_mask_covariance_cells']==8
    assert counters['geometry_small_tile_accepted']==1
    assert counters['geometry_small_tile_union_logical_calls']==5
    assert counters['geometry_small_tile_mask_logical_calls']==10
    assert counters['fallback_geometry']==0 and counters['union_score_calls']==1
    assert len(candidate.reads)==1 and candidate.score_modes==['tf32']
    assert candidate._local_union_device_prepared is None


@pytest.mark.parametrize('product_shape',[(20,47,42652),(47,47,42652),(47,20,20),(47,47,20)])
def test_small_tile_rejects_if_any_individual_product_cost_increases(monkeypatch,product_shape):
    from fudan_wgs_toolkit.local_mask_reuse import small_native_union_work
    import fudan_wgs_toolkit.tf32 as backend
    original=backend._native_geometry
    def geometry(m,n,k):
        bm,bn,bk,warps,stages=original(m,n,k)
        if (m,n,k)==product_shape:bm*=8;bn*=8
        return bm,bn,bk,warps,stages
    monkeypatch.setattr(backend,'_native_geometry',geometry)
    cost=small_native_union_work(np.arange(47),[np.arange(37),np.arange(37,47)],
                                minimum_variants=2,samples=42652,covariates=20)
    assert not cost['beneficial']


def test_intercept_real_47_case_independent_routes_and_bounded_outer():
    from fudan_wgs_toolkit.local_mask_reuse import small_native_union_work
    cost=small_native_union_work(np.arange(47),[np.arange(37),np.arange(37,47)],
                                minimum_variants=2,samples=42652,covariates=1)
    assert cost['beneficial'] and cost['policy']=='intercept_independent_routes'
    assert cost['union']['covariance']=={'route':'tf32_mma','work':174718976}
    assert cost['separate']['covariance']==262078464
    for name in ('cross','score'):
        assert cost['union'][name]=={'route':'gemv','work':2004644}
        assert cost['separate'][name]==2004644
    assert cost['union']['projection_left']=={'route':'gemv','work':47}
    assert cost['separate']['projection_left']==47
    assert cost['union']['projection']=={'route':'outer','work':2209}
    assert cost['separate']['projection']==1469
    assert cost['outer_extra_output_bytes']==740*4
    assert 'not complete allocation/workspace' in cost['outer_extra_output_bytes_scope']
    assert (cost['union_calls'],cost['separate_calls'])==(5,10)


@pytest.mark.parametrize('size,first,accepted',[(64,37,True),(64,32,False),(65,37,False)])
def test_intercept_outer_shape_limit_and_strict_mma_saving(size,first,accepted):
    from fudan_wgs_toolkit.local_mask_reuse import small_native_union_work
    cost=small_native_union_work(np.arange(size),[np.arange(first),np.arange(first,size)],
                                minimum_variants=2,samples=42652,covariates=1)
    assert cost['beneficial'] is accepted
    if size==64:
        assert cost['union']['projection']['work']==4096
        if first==32:
            assert cost['union']['covariance']['work']==cost['separate']['covariance']
        else:
            assert cost['outer_extra_output_bytes']<=4096*4


def test_intercept_rejects_vector_work_increase_despite_covariance_saving():
    from fudan_wgs_toolkit.local_mask_reuse import small_native_union_work
    cost=small_native_union_work(np.arange(47),[np.arange(37),np.arange(37,42)],
                                minimum_variants=2,samples=42652,covariates=1)
    assert cost['union']['covariance']['work']<cost['separate']['covariance']
    assert cost['union']['score']['work']>cost['separate']['score']
    assert not cost['beneficial']


def test_intercept_rejects_changed_backend_route(monkeypatch):
    import fudan_wgs_toolkit.tf32 as backend
    from fudan_wgs_toolkit.local_mask_reuse import small_native_union_work
    original=backend._native_route
    monkeypatch.setattr(backend,'_native_route',lambda a,b:'gemv' if a[1]==1 and a[0]==47 and b[1]==47 else original(a,b))
    cost=small_native_union_work(np.arange(47),[np.arange(37),np.arange(37,47)],
                                minimum_variants=2,samples=42652,covariates=1)
    assert cost['union']['projection']['route']=='gemv'
    assert not cost['beneficial']


def test_intercept_gate_accepts_decoded_disjoint_masks_with_same_statistics():
    host=resident_family_fixture(resident=False);candidate=resident_family_fixture(resident=True)
    assert candidate.models[0].x.shape[1]==1
    masks=[np.array([1,0,4]),np.array([3,2,5])]
    expected=host._run_mask_sets(masks);candidate.local_mask_reuse=True
    actual=candidate._run_mask_sets(masks)
    for lhs,rhs in zip(expected,actual):
        if lhs[0] is not None:assert abs(lhs[0].pop('cmac')-rhs[0].pop('cmac'))<1e-5
    assert_payload_equal(expected,actual)
    counters=candidate.local_mask_reuse_counters
    assert counters['geometry_small_tile_p1_accepted']==1
    assert counters['geometry_small_tile_outer_extra_output_bytes']==(16-8)*4
    assert counters['union_score_calls']==1 and counters['fallback_geometry']==0
    assert candidate.score_modes==['tf32'] and len(candidate.reads)==1
