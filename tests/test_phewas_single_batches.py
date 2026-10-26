"""Bounded synthetic contracts for independent samples and shared Single tails.

These fixtures establish equivalence to standalone kernels and metadata;
scientific timing/accuracy claims require the separate real-data run.
"""
from types import SimpleNamespace
from numbers import Real

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.genotype import _allele_frequency_summary
from fudan_wgs_toolkit.genotype_device import DeviceMinorBlock
from fudan_wgs_toolkit.binary_null import BinaryNullModel
from fudan_wgs_toolkit.null_model import fit_gaussian_null
from fudan_wgs_toolkit.pipeline import AnalysisOptions, PheWASPipeline
from fudan_wgs_toolkit import pipeline as pipeline_module
from fudan_wgs_toolkit.phewas_runtime import single


def states_fixture():
    states = np.zeros((256, 10), dtype=np.uint8)
    states[::2, 0] = 1
    states[0, 1] = 1
    states[:, 2] = 2
    states[0, 2] = 1
    states[:, 3] = 3
    states[:4, 4] = [4, 5, 1, 4]
    states[0, 5] = 1
    states[1:8, 5] = 4
    states[200:, 6] = 1
    states[:, 7] = 1
    states[100, 8] = 2
    states[150, 9] = 1
    return states


def minor_block(states, samples, variants, device="cpu"):
    selected = states[np.asarray(samples)[:, None], np.asarray(variants)[None, :]]
    alleles = np.asarray([(2, 2), (1, 2), (0, 2), (0, 0), (1, 1), (0, 1)])
    reference = alleles[selected, 0].sum(0, dtype=np.int64)
    called = alleles[selected, 1].sum(0, dtype=np.int64)
    af, missing, mac, ref_ac, called = _allele_frequency_summary(reference, called, len(samples))
    dosage = np.where(selected < 3, 2 - selected.astype(np.int16), 3)
    dosage = np.where((dosage != 3) & (af[None, :] >= .5), 2 - dosage, dosage).astype(np.uint8)
    return DeviceMinorBlock(torch.as_tensor(dosage, device=device), np.asarray(samples), np.asarray(variants),
                            af, mac, missing, ref_ac, called)


def make_pipeline(samples, blocks, *, model=None, family="gaussian", device="cpu"):
    n = len(samples)
    if model is None:
        residual = torch.linspace(-.03, .07, n, dtype=torch.float32, device=device)
        weights = torch.linspace(.9, 1.2, n, dtype=torch.float32, device=device)
        def core(genotype):
            # Small test core with nonuniform row weights exposes reordered G.
            return (genotype * residual[:, None]).sum(0), (genotype.square() * weights[:, None]).sum(0) + .25
        model = SimpleNamespace(n=n, n_pheno=1, family=family, use_spa=False, device=device,
                                matmul_mode="tf32", x=torch.ones((n, 2), device=device, dtype=torch.float32),
                                individual_score_variance=core)
    pipeline = PheWASPipeline.__new__(PheWASPipeline)
    pipeline.models = [model]
    pipeline.options = AnalysisOptions(wrapper_semantics="base")
    pipeline.resident_genotypes = True
    pipeline.trait_rows = [np.arange(n)]
    pipeline.union_rows = np.asarray(samples)
    pipeline.position = np.arange(10) * 10 + 100
    pipeline._base_mask = lambda *args: np.ones(10, dtype=bool)
    pipeline._minor_blocks = lambda *args, **kwargs: iter(blocks)
    pipeline._limit = lambda *args, **kwargs: None
    ref = np.asarray(["A", "C", "G", "T"])
    pipeline.genotype = SimpleNamespace(n_variants=10,
        read_field=lambda name, selected: np.full(len(selected), "1"),
        read_ref_alt=lambda selected: (ref[selected % 4], ref[(selected + 1) % 4]))
    return pipeline


def standalone_rows(pipeline):
    return [row for _, rows in pipeline.iter_individual_records(
        "1", mac_cutoff=1, subset_variants_num=3) for row in rows]


def assert_tables_equal(left, right):
    # Finalization also checks factor levels and row names across chunk cuts.
    left_table = PheWASPipeline.individual_tables([left])[0]
    right_table = PheWASPipeline.individual_tables([right])[0]
    assert left_table.row_names == right_table.row_names
    assert left_table.factor_levels == right_table.factor_levels
    assert len(left_table) == len(right_table)
    for left_row, right_row in zip(left_table, right_table):
        assert list(left_row) == list(right_row)
        for key in left_row:
            left_value, right_value = left_row[key], right_row[key]
            if isinstance(left_value, Real) and isinstance(right_value, Real):
                np.testing.assert_allclose(left_value, right_value, rtol=0, atol=0, equal_nan=True,
                                           err_msg=f"native Single field {key}")
            else:
                assert left_value == right_value


def test_different_sample_axes_ragged_ready_blocks_match_standalone_and_copy_once():
    states = states_fixture()
    samples = [np.arange(255, -1, -1), np.asarray([0] + list(range(159, 0, -1)))]
    blocks = [[minor_block(states, samples[0], np.arange(4)), minor_block(states, samples[0], np.arange(4, 10))],
              [minor_block(states, samples[1], np.arange(10))]]
    pipelines = [make_pipeline(rows, pieces, family=family) for rows, pieces, family in
                 zip(samples, blocks, ["gaussian", "binomial"])]
    expected = [standalone_rows(pipeline) for pipeline in pipelines]
    single.execution_metadata(reset=True)
    first, ordinals = single.process_single_batches(
        pipelines, [blocks[0][0], None], [0, 0], "1", mac_cutoff=1, subset_variants_num=3)
    assert first[1] == [] and ordinals[1] == 0
    second, ordinals = single.process_single_batches(
        pipelines, [blocks[0][1], blocks[1][0]], ordinals, "1", mac_cutoff=1, subset_variants_num=3)
    for trait in range(2):
        assert_tables_equal(expected[trait], first[trait] + second[trait])
        assert ordinals[trait] == sum(int((block.initial_mac() >= 1).sum()) for block in blocks[trait])
    report = single.execution_metadata()
    assert report["active_trait_blocks"] == report["computed_trait_blocks"] == 3
    assert report["pointwise_tail_batches"] == report["result_transfer_batches"] == 2
    assert report["result_transfer_values"] == 4 * sum(map(len, expected))
    assert not report["covariance_shared"] and not report["union_sample_padding"]


def test_none_and_fully_filtered_blocks_keep_empty_results_and_independent_ordinals():
    states = states_fixture()
    samples = np.arange(256)
    pipeline = make_pipeline(samples, [])
    block = minor_block(states, samples, np.asarray([3]))
    single.execution_metadata(reset=True)
    result, ordinals = single.process_single_batches([pipeline], [block], [7], "1", mac_cutoff=1)
    assert result == [[]] and ordinals == [7]
    result, ordinals = single.process_single_batches([pipeline], [None], ordinals, "1", mac_cutoff=1)
    assert result == [[]] and ordinals == [7]
    report = single.execution_metadata()
    assert report["calls"] == 2 and report["active_trait_blocks"] == 1
    assert report["computed_trait_blocks"] == report["result_transfer_batches"] == 0


@pytest.mark.parametrize("offset", [-1, True, 1.2])
def test_invalid_ordinal_fails_before_preparing_data(offset):
    pipeline = make_pipeline(np.arange(256), [])
    with pytest.raises(ValueError, match="ordinal_offset"):
        single.process_single_batches([pipeline], [None], [offset], "1")


@pytest.mark.parametrize("change", ["samples", "spa", "precision", "wrapper", "models"])
def test_invalid_shared_single_semantics_are_rejected(change):
    states, samples = states_fixture(), np.arange(256)
    block = minor_block(states, samples, np.arange(10))
    pipeline = make_pipeline(samples, [block])
    if change == "samples":
        block.sample_indices = block.sample_indices[::-1]
    elif change == "spa":
        pipeline.models[0].use_spa = True
    elif change == "precision":
        pipeline.models[0].matmul_mode = "fp64"
    elif change == "wrapper":
        pipeline.options = AnalysisOptions(wrapper_semantics="phewas")
    else:
        pipeline.models.append(pipeline.models[0])
    with pytest.raises(ValueError):
        single.process_single_batches([pipeline], [block], [0], "1", mac_cutoff=1)


@pytest.mark.parametrize("state", ["complete", "spa", "dense", "no_projection", "cov_shape", "fp64"])
def test_resident_gene_binary_guard_admits_only_complete_native_diagonal_state(monkeypatch, state):
    # Geometry-only admission test: mock CUDA counters, never allocate tensors
    # or imply a GPU association result from these descriptors.
    device = torch.device("cuda:0")
    def descriptor(shape, *, dtype=torch.float32):
        return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype,
                               layout=torch.strided, device=device)
    model = BinaryNullModel.__new__(BinaryNullModel)
    model.x = descriptor((256, 2))
    model.precision = descriptor((256,))
    model.precision_x = descriptor((256, 2))
    model.fixed_effect_covariance = descriptor((2, 2))
    model.n_pheno, model.use_spa, model.matmul_mode = 1, False, "tf32"
    if state == "spa":
        model.use_spa = True
    elif state == "dense":
        model.precision = descriptor((256, 256))
    elif state == "no_projection":
        model.precision_x = None
    elif state == "cov_shape":
        model.fixed_effect_covariance = descriptor((3, 3))
    elif state == "fp64":
        model.fixed_effect_covariance = descriptor((2, 2), dtype=torch.float64)
    pipeline = PheWASPipeline.__new__(PheWASPipeline)
    pipeline.models = [model]
    pipeline.resident_genotypes = True
    pipeline.union_rows = np.arange(256)
    pipeline.options = AnalysisOptions()
    pipeline.genotype = SimpleNamespace(_flat_reader=object(), n_samples=256, genotype_raw_memory_bytes=1024)
    pipeline.local_mask_reuse_counters = {}
    queried = []
    monkeypatch.setattr(pipeline_module, "allocator_bytes", lambda device: queried.append("allocator") or (0, 0))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: queried.append("free") or (80 * 2**30, 80 * 2**30))
    result = pipeline._resident_gene_reader_options(np.arange(10))
    if state == "complete":
        assert result == dict(device=device, resident=True)
        assert queried == ["allocator", "free"]
    else:
        assert result == {} and queried == []


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
                    reason="native TF32 contract requires an Ampere CUDA GPU")
def test_real_native_gaussian_cores_keep_k_order_and_shared_tail_logp():
    states = states_fixture()
    samples = [np.arange(255, -1, -1), np.asarray([0] + list(range(159, 0, -1)))]
    device = "cuda:0"
    pipelines, blocks = [], []
    for sample in samples:
        scaled = sample.astype(np.float64) / 256
        x = np.column_stack((np.ones(len(sample)), scaled, np.square(scaled)))
        y = np.sin(scaled * 8) + np.cos(scaled * 3)
        model = fit_gaussian_null(y, sample_ids=sample.astype(str), covariates=x,
                                  device=device, matmul_mode="tf32")
        block = minor_block(states, sample, np.arange(10), device=device)
        blocks.append(block)
        pipelines.append(make_pipeline(sample, [block], model=model, device=device))
    expected = [standalone_rows(pipeline) for pipeline in pipelines]
    result, ordinals = single.process_single_batches(
        pipelines, blocks, [0, 0], "1", mac_cutoff=1, subset_variants_num=3)
    for trait in range(2):
        assert ordinals[trait] == int((blocks[trait].initial_mac() >= 1).sum())
        # Strict equality is expected: only pointwise tails and copy grouping
        # are shared; the original FP32 GEMV/TF32 covariance calls are intact.
        assert_tables_equal(expected[trait], result[trait])
