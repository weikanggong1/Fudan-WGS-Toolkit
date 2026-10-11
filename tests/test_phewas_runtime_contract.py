"""CPU admission, scheduling and cleanup contracts; no timing claims."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.genotype_device import DeviceMinorBlock
from fudan_wgs_toolkit.phewas_runtime import runtime
from fudan_wgs_toolkit.phewas_runtime.buffers import EffectiveBuffer
from fudan_wgs_toolkit.phewas_runtime.metadata import SharedMetadataReader


def configurations(tmp_path, *, kinds=("individual",)):
    configs = []
    for trait in range(2):
        arguments = {"individual": dict(variant_type="variant"),
                     "coding": dict(gene_name="gene_0001", start=1, end=100),
                     "noncoding": dict(gene_name="gene_0001"),
                     "ncrna": dict(gene_name="gene_0001")}
        jobs = [dict(name=f"job_{number}", kind=kind, arguments=deepcopy(arguments[kind]),
                     output=str(tmp_path / f"trait_{trait}" / f"result_{number}.csv"),
                     object_name=f"object_{kind}", layout="base")
                for number, kind in enumerate(kinds)]
        configs.append(dict(phenotypes=[dict(name=f"trait_{trait}", model=str(tmp_path / f"state_{trait}.npz"),
                                             sample_indices_file=str(tmp_path / f"rows_{trait}.npy"))],
                            chromosomes=[dict(name="21", genotype=str(tmp_path / "source.genotype"), jobs=jobs)],
                            analysis_options=dict(memory_limit_gib=20)))
    return configs


def test_aligned_analyses_keep_independent_models_outputs_and_input_unchanged(tmp_path):
    original = configurations(tmp_path, kinds=("individual", "coding"))
    before = deepcopy(original)
    result = runtime.validate_analyses(original)
    assert original == before
    assert len(result) == 2
    assert result[0]["phenotypes"] != result[1]["phenotypes"]
    for config in result:
        assert config["analysis_options"]["wrapper_semantics"] == "base"
        assert config["individual_effective_block_size"] == 1024
    assert result[0]["chromosomes"][0]["jobs"][0]["output"] != result[1]["chromosomes"][0]["jobs"][0]["output"]


def forbid_gpu_preflight(monkeypatch):
    def unexpected_gpu_use(*args, **kwargs):
        raise AssertionError("configuration preflight must not initialize or inspect CUDA")

    for name in ("init", "is_available", "current_device", "get_device_properties", "memory_allocated"):
        monkeypatch.setattr(torch.cuda, name, unexpected_gpu_use)


@pytest.mark.parametrize("budget", [40, True, float("nan"), float("inf"), "20", 0],
                         ids=["over_limit", "boolean", "nan", "infinite", "string", "zero"])
def test_memory_limit_preflight_rejects_invalid_budget_without_cuda(tmp_path, monkeypatch, budget):
    configs = configurations(tmp_path)
    # All traits receive the same value: this must fail the budget gate, not alignment.
    for config in configs:
        config["analysis_options"]["memory_limit_gib"] = budget
    forbid_gpu_preflight(monkeypatch)
    with pytest.raises(ValueError, match="memory_limit_gib"):
        runtime.validate_analyses(configs)


def test_memory_limit_preflight_accepts_twenty_gib_without_cuda(tmp_path, monkeypatch):
    configs = configurations(tmp_path)
    forbid_gpu_preflight(monkeypatch)
    validated = runtime.validate_analyses(configs)
    assert [config["analysis_options"]["memory_limit_gib"] for config in validated] == [20, 20]


def test_phewas_default_budget_remains_twenty_gib_with_larger_standalone_default(tmp_path, monkeypatch):
    configs = configurations(tmp_path)
    for config in configs:
        config["analysis_options"].pop("memory_limit_gib")
    forbid_gpu_preflight(monkeypatch)
    validated = runtime.validate_analyses(configs)
    for config in validated:
        assert "memory_limit_gib" not in configs[0]["analysis_options"]
        assert runtime.AnalysisOptions(**config["analysis_options"]).memory_limit_gib == 20


@pytest.mark.parametrize("change", ["no_phenotype", "two_phenotypes", "fp64", "precision_control", "batched",
    "host_genotype", "split_k", "wrapper", "layout", "window", "non_csv_output", "mask_limit", "block_size",
    "no_jobs", "different_source", "different_arguments", "different_setting", "trait_output_collision",
    "saved_model_format", "model_input_collision", "individual_repeated", "gene_kind_collision", "gene_object_collision",
    "boolean_split_k", "unknown_arguments", "nonboolean_flag", "fit_precision_conflict",
    "fit_split_conflict", "fit_family_conflict", "joint_fit", "obsolete_reconstruction"])
def test_invalid_production_config_fails_before_any_gpu_use(tmp_path, change):
    configs = configurations(tmp_path, kinds=("individual", "coding"))
    config = configs[0]
    jobs = config["chromosomes"][0]["jobs"]
    if change == "no_phenotype":
        config["phenotypes"] = []
    elif change == "two_phenotypes":
        config["phenotypes"].append(deepcopy(config["phenotypes"][0]))
    elif change in ("fp64", "precision_control", "batched", "host_genotype", "split_k"):
        key, value = {"fp64": ("matmul_mode", "fp64"), "precision_control": ("precision_control", True),
            "batched": ("statistics_execution", "batched"), "host_genotype": ("resident_genotypes", False),
            "split_k": ("tf32_split_k", 32)}[change]
        config[key] = value
    elif change == "wrapper":
        config["analysis_options"]["wrapper_semantics"] = "phewas"
    elif change == "layout":
        jobs[0]["layout"] = "phewas"
    elif change == "window":
        jobs[0]["kind"] = "sliding"
    elif change == "non_csv_output":
        jobs[0]["output"] = str(tmp_path / "result.txt")
    elif change == "mask_limit":
        config["maximum_mask_variants"] = 0
    elif change == "block_size":
        config["individual_effective_block_size"] = True
    elif change == "no_jobs":
        config["chromosomes"][0]["jobs"] = []
    elif change == "different_source":
        config["chromosomes"][0]["genotype"] = str(tmp_path / "other_source.genotype")
    elif change == "different_arguments":
        jobs[0]["arguments"]["mac_cutoff"] = 21
    elif change == "different_setting":
        config["weight_batch_optimization"] = False
    elif change == "trait_output_collision":
        configs[1]["chromosomes"][0]["jobs"][0]["output"] = jobs[0]["output"]
    elif change == "saved_model_format":
        config["phenotypes"][0]["save_model"] = jobs[0]["output"]
    elif change == "model_input_collision":
        config["phenotypes"][0]["save_model"] = configs[1]["phenotypes"][0]["model"]
    elif change == "individual_repeated":
        jobs.append(deepcopy(jobs[0]))
    elif change == "boolean_split_k":
        config["tf32_split_k"] = False
    elif change == "unknown_arguments":
        jobs[0]["arguments"]["unsupported_argument"] = 1
    elif change == "nonboolean_flag":
        config["statistics_tail_optimization"] = "false"
    elif change == "fit_precision_conflict":
        config["phenotypes"][0]["fit_options"] = {"matmul_mode": "fp64"}
    elif change == "fit_split_conflict":
        config["phenotypes"][0]["fit_options"] = {"tf32_split_k": 32}
    elif change == "fit_family_conflict":
        config["phenotypes"][0]["family"] = "gaussian"
        config["phenotypes"][0]["fit_options"] = {"family": "binomial"}
    elif change == "joint_fit":
        config["phenotypes"][0]["fit_options"] = {"joint_mode": "ordinary"}
    elif change == "obsolete_reconstruction":
        config["tf32_binned_tile_shape"] = [32, 64]
    else:
        copied = deepcopy(jobs[1])
        if change == "gene_kind_collision":
            copied["kind"] = "noncoding"
        else:
            copied["object_name"] = "different_object"
        jobs.append(copied)
    with pytest.raises(ValueError):
        runtime.validate_analyses(configs)


def test_compatible_gene_batches_share_file_but_csv_writer_runs_only_at_last_job(tmp_path, monkeypatch):
    configs = configurations(tmp_path, kinds=("coding", "coding"))
    for config in configs:
        config["chromosomes"][0]["jobs"][1]["output"] = config["chromosomes"][0]["jobs"][0]["output"]
    configs = runtime.validate_analyses(configs)
    output = runtime._CSVOutputs(configs[0])
    writes = []
    monkeypatch.setattr(runtime, "write_association_batch", lambda path, results, **options:
                        writes.append((path, list(results), options)) or {"rows":2})
    first, last = configs[0]["chromosomes"][0]["jobs"]
    result_a, result_b = [[{"first": 1}]], [[{"second": 2}]]
    output.append(first, result_a)
    assert writes == []
    with pytest.raises(RuntimeError, match="incomplete"):
        output.check()
    output.append(last, result_b)
    output.check()
    assert output.files == 1
    assert writes[0][1] == [result_a, result_b]
    assert writes[0][2] == dict(kind="coding", layout="base", exclude_columns=(), empty_columns=None)


def block(samples, variants):
    samples, variants = np.asarray(samples), np.asarray(variants)
    dosage = torch.as_tensor((samples[:, None] + variants[None, :]) % 3, dtype=torch.uint8)
    width = len(variants)
    result = DeviceMinorBlock(dosage, samples, variants,
        np.linspace(.6, .8, width), np.arange(width) + 1., np.linspace(.001, .02, width),
        np.arange(width) + 5., np.arange(width) + 10)
    result._store_counts(np.arange(len(samples)), np.arange(width) + 10., np.arange(width))
    return result


def test_effective_buffer_splits_preserving_variant_order_summaries_counts_and_tail():
    samples = np.asarray([5, 1, 4, 2])
    first, second = block(samples, [10]), block(samples, [11, 12, 13, 14])
    buffer = EffectiveBuffer(samples, 3)
    buffer.append(first)
    assert buffer.take() is None
    buffer.append(second)
    packed = buffer.take()
    assert packed.variant_indices.tolist() == [10, 11, 12]
    torch.testing.assert_close(packed.dosage, torch.cat((first.dosage, second.dosage[:, :2]), 1))
    for name in ("union_ref_af", "union_initial_mac", "union_missing_rate", "union_ref_ac", "union_called_alleles"):
        np.testing.assert_array_equal(getattr(packed, name), np.concatenate((getattr(first, name), getattr(second, name)[:2])))
    np.testing.assert_array_equal(packed._cached_counts(np.arange(4))[0], [10, 10, 11])
    assert buffer.count == 2 and buffer.take() is None
    tail = buffer.take(tail=True)
    assert tail.variant_indices.tolist() == [13, 14]
    assert buffer.count == 0 and buffer.take(tail=True) is None
    np.testing.assert_array_equal(tail._cached_counts(np.arange(4))[0], [12, 13])


def test_effective_buffer_guard_rejects_before_splitting_or_consuming_parts():
    samples = np.arange(4)
    first, second = block(samples, [1]), block(samples, [2, 3, 4, 5])
    required = []
    def reject(size):
        required.append(size)
        raise MemoryError("test live budget")
    buffer = EffectiveBuffer(samples, 3, allocation_guard=reject)
    buffer.append(first)
    buffer.append(second)
    with pytest.raises(MemoryError, match="live budget"):
        buffer.take()
    assert required == [4 * 3 + 64 * 2**20]
    assert buffer.count == 5 and list(buffer.parts) == [first, second]


def test_effective_buffer_rejects_changed_sample_axis_and_large_index_geometry():
    buffer = EffectiveBuffer(np.arange(4), 3)
    with pytest.raises(ValueError, match="sample order"):
        buffer.append(block(np.arange(3, -1, -1), [1]))
    with pytest.raises(MemoryError, match="int32"):
        EffectiveBuffer(np.arange(4), np.iinfo(np.int32).max)


def test_metadata_reuses_equal_axes_only_and_bounds_lru_bytes():
    calls = []
    class Reader:
        reader_metadata = {"input_format": "synthetic"}
        def read_field(self, field, rows=None):
            calls.append((field, None if rows is None else tuple(rows)))
            return np.asarray(np.arange(4) if rows is None else rows, dtype=np.int64) + len(field)
        def read_ref_alt(self, rows=None):
            calls.append(("alleles", None if rows is None else tuple(rows)))
            count = 4 if rows is None else len(rows)
            return np.full(count, "A"), np.full(count, "G")
    reader = SharedMetadataReader(Reader(), capacity_bytes=64)
    value = reader.read_field("first", np.arange(4, dtype=np.int32))
    assert reader.read_field("first", np.arange(4, dtype=np.int64)) is value
    reverse = reader.read_field("first", np.arange(3, -1, -1))
    assert not np.array_equal(value, reverse)
    reader.read_field("second", np.arange(4))
    assert reader.metrics["evictions"] == 1 and reader.metrics["highwater_bytes"] <= 64
    reader.read_ref_alt([0, 1])
    assert reader.read_ref_alt(np.asarray([0, 1]))[0].tolist() == ["A", "A"]
    assert reader.reader_metadata["shared_metadata"]["capacity_bytes"] == 64
    with pytest.raises(ValueError, match="integer axis"):
        reader.read_field("field", [0., 1.])
    reader.clear()
    assert reader._bytes == 0 and not reader._cache


def test_changed_source_proof_closes_broker_container_reader_and_contexts(tmp_path, monkeypatch):
    configs = configurations(tmp_path, kinds=("coding",))[:1]
    states = dict(reader_closed=False, container_closed=False, broker_closed=False,
                  solver_closed=False, proof_calls=0)
    class Reader:
        n_samples, n_variants = 4, 10
        reader_metadata = {"input_format": "synthetic"}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            states["reader_closed"] = True
        def read_field(self, field, rows=None):
            values = np.arange(self.n_variants) if field == "position" else np.full(self.n_variants, "PASS")
            return values if rows is None else values[np.asarray(rows)]
        def sample_ids(self):
            return np.asarray([f"s{i}" for i in range(self.n_samples)])
    class Container:
        def __init__(self, *args):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
        def close(self):
            states["container_closed"] = True
    class Broker:
        metrics = {}
        def __init__(self, reader, container, **kwargs):
            self.reader, self.container = reader, container
            self.own_container = kwargs.get("own_container", True)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            states["broker_closed"] = True
            if self.own_container:
                self.container.close()
        def reader_view(self, rows):
            return self.reader
    class Pipeline:
        def __init__(self, *args, **kwargs):
            self._base_masks, self._annotation_indexes, self._coding_masks_cache = {}, {}, {}
            self._category_codes = None
            self.skipped_sets, self.local_mask_reuse_counters = [], {}
        def coding(self, **kwargs):
            return [[{"dummy_count": 1}]]
    @contextmanager
    def solver(*args, **kwargs):
        try:
            yield {}
        finally:
            states["solver_closed"] = True
    def proof():
        states["proof_calls"] += 1
        if states["proof_calls"] == 1:
            return {"verified": 1}
        raise ValueError("source changed")
    model = SimpleNamespace(n=4, x=torch.ones((4, 1), dtype=torch.float32), family="gaussian", use_spa=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(runtime, "_load_models", lambda *args: ([model], [None]))
    monkeypatch.setattr(runtime.cli, "_weighted_eigensolver_settings", lambda *args: {"effective": "torch"})
    monkeypatch.setattr(runtime.cli, "_scheduled_index_categories", lambda jobs: [])
    monkeypatch.setattr(runtime.cli, "_bind_genotype_samples", lambda *args: np.arange(4))
    monkeypatch.setattr(runtime.cli, "_association_result_row_count", lambda *args, **kwargs: 1)
    monkeypatch.setattr(runtime, "PortableMetadataReader", lambda *args, **kwargs: Reader())
    monkeypatch.setattr(runtime, "Container", Container)
    monkeypatch.setattr(runtime, "SharedStateBroker", Broker)
    monkeypatch.setattr(runtime, "LimitedMaskPipeline", Pipeline)
    monkeypatch.setattr(runtime, "write_association_batch", lambda *args, **kwargs: None)
    from fudan_wgs_toolkit import _weighted_spectra
    monkeypatch.setattr(_weighted_spectra, "eigensolver_context", solver)
    spec = SimpleNamespace(expected_binding={"verified": 1}, source_proof=proof,
                           directory=tmp_path / "verified_cache", expected_samples=np.arange(4))
    with pytest.raises(ValueError, match="source changed"):
        runtime.run_configuration(configs, cache_specs={tmp_path / "source.genotype": spec})
    assert states == dict(reader_closed=True, container_closed=True, broker_closed=True,
                          solver_closed=True, proof_calls=2)
    assert not runtime._LOCK.locked()
