"""Binary admission geometry and policy; these are not throughput benchmarks."""
import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.masks import annotation_phred_matrix
from fudan_wgs_toolkit.phewas_runtime import runtime
from test_phewas_device_gene_prepare import fixture_pipeline


@pytest.mark.parametrize("families, expected", [
    ([], .5), (["gaussian"], .5), (["binomial"], .75),
    (["binomial", "binomial"], .75), (["binomial", "gaussian"], .5)])
def test_default_retention_only_changes_complete_binary_sweep(families, expected):
    assert runtime._model_cache_live_fraction({}, [{"family": f} for f in families]) == expected


@pytest.mark.parametrize("value", [None, .25, .5, .9])
def test_explicit_optional_cache_policy_wins(value):
    for family in ("gaussian", "binomial"):
        assert runtime._model_cache_live_fraction({"model_cache_live_fraction": value},
            [{"family": family}]) == value


def test_binary_device_preparation_preserves_data_and_linear_peak(tmp_path):
    indices = np.asarray([5, 2, 9, 4, 1, 6, 0, 8, 3, 7, 10, 11], dtype=np.int64)
    with fixture_pipeline(tmp_path / "source", device="cpu", rows=np.arange(512)[::-1].copy()) as (p, broker):
        annotations = p.annotations(indices).annotations
        phred, names = annotation_phred_matrix(annotations, p.annotation_names,
            variant_type=p.options.variant_type, number_variants=len(indices))
        guards = []
        broker._guard = lambda required: guards.append(required)
        gaussian = runtime._prepare_shared_gene_device(p, indices, broker, phred, names)
        original = guards[0]
        guards.clear()
        # Preparation accesses only n and the verified sample/genetic axis.
        p.models[0].family = "binomial"
        binary = runtime._prepare_shared_gene_device(p, indices, broker, phred, names)
        assert guards[0] == original - 32 * len(indices)**2
        assert guards[0] >= 12 * p.models[0].n * len(indices)
        for key in ("maf", "mac", "annotations", "_variant_indices", "_extraction_groups"):
            np.testing.assert_array_equal(binary[key], gaussian[key])
        assert torch.equal(binary["_genotype_device"], gaussian["_genotype_device"])
        assert binary["_genotype_device"].stride() == gaussian["_genotype_device"].stride()
        assert binary["names"] == gaussian["names"] and binary["cmac"] == gaussian["cmac"]
        # A mixed cohort can use a binary anchor, while a later Gaussian
        # still retains the original preparation reservation.
        guards.clear()
        p._binary_burden_only = False
        runtime._prepare_shared_gene_device(p, indices, broker, phred, names)
        assert guards[0] == original


@pytest.mark.parametrize("resident", [False, True])
def test_budget_covers_actual_burden_panel_projection_and_tail_buffers(resident):
    n, m, q, annotations = 61, 717, 5, 3
    b, panel = 2 * (annotations + 1), min(m, 512)
    # Materialize representative actual tensors for the independent peak
    # phases, rather than inferring bytes from an M-by-M variant covariance.
    buffers = [torch.empty((n, m if resident else panel), dtype=torch.float32),
        torch.empty((n, panel), dtype=torch.float32),
        torch.empty((n, panel), dtype=torch.float64),
        torch.empty((m, b), dtype=torch.float32),
        torch.empty((m, b), dtype=torch.float64),
        torch.empty((q, b), dtype=torch.float64),
        torch.empty((b, b), dtype=torch.float64)]
    buffers.extend(torch.empty((n, b), dtype=torch.float32) for _ in range(4))
    buffers.extend(torch.empty((n, b), dtype=torch.float64) for _ in range(20))
    actual = sum(value.untyped_storage().nbytes() for value in buffers)
    budget = runtime._binary_gene_workspace_new_bytes(n, m, q, annotations,
        resident_genotype=resident)
    assert budget >= actual + 256 * 2**20


def test_small_burden_inference_no_long_variant_covariance_admission():
    n, m, q, annotations = 343034, 5000, 16, 12
    binary = runtime._binary_gene_workspace_new_bytes(n, m, q, annotations)
    gaussian = 8*n*m + 32*m*m + 256*2**20
    assert binary < gaussian
    # Large host masks remain bounded by the panel, K, and linear weights;
    # increasing M beyond 512 never allocates a full CUDA samples-by-M slab.
    small = runtime._binary_gene_workspace_new_bytes(n, 10000, q, annotations,
        resident_genotype=False)
    large = runtime._binary_gene_workspace_new_bytes(n, 100000, q, annotations,
        resident_genotype=False)
    assert large < 6 * 2**30 and large - small < 128 * 2**20


@pytest.mark.parametrize('binary, fraction', [(False, .5), (True, .75)])
def test_preallocation_capacity_matches_live_runtime_default(monkeypatch, binary, fraction):
    from fudan_wgs_toolkit.phewas_partition import gpu_model_cache_capacity
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert gpu_model_cache_capacity((0,), memory_limit_gib=40, binary_only=binary) == int(39*2**30*fraction)
    descriptors = [{'family': 'binomial' if binary else 'gaussian'}]
    assert runtime._model_cache_live_fraction({}, descriptors) == fraction


def test_explicit_preallocation_fraction_still_wins_for_binary(monkeypatch):
    from fudan_wgs_toolkit.phewas_partition import gpu_model_cache_capacity
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert gpu_model_cache_capacity((0,), memory_limit_gib=40,
        binary_only=True, cache_fraction=.5) == int(39*2**30*.5)
    with pytest.raises(ValueError, match='binary_only'):
        gpu_model_cache_capacity((0,), memory_limit_gib=40, binary_only=1)


def test_binary_preallocation_uses_real_driver_cap_and_own_live_allocations(monkeypatch):
    from fudan_wgs_toolkit.phewas_partition import gpu_model_cache_capacity
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (79*2**30, 80*2**30))
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 2*2**30)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 3*2**30)
    assert gpu_model_cache_capacity((0, 1), memory_limit_gib=40,
        binary_only=True) == int(37*2**30*.75)


@pytest.mark.parametrize('binary', [False, True])
def test_legacy_entry_transmits_its_binary_contract_into_cache_budget(tmp_path, monkeypatch, binary):
    from fudan_wgs_toolkit import cpu_budget, phewas_partition, phewas_run
    monkeypatch.setattr(phewas_run, 'prepare_phewas_run', lambda *args, **kwargs: dict(
        output_directory=str(tmp_path), attempt_id='cache-contract', configuration=dict(chromosomes=[])))
    monkeypatch.setattr(cpu_budget, 'detect_cpu_capacity', lambda: dict(effective_cores=8))
    monkeypatch.setattr(phewas_run.os, 'sched_getaffinity', lambda pid: set(range(8)))
    monkeypatch.setattr(phewas_run.os, 'sched_setaffinity', lambda pid, cpus: None)
    observed = []
    def capacity(gpu_ids, **policy):
        observed.append(policy)
        return 1
    class ReachedAssociation(Exception):
        pass
    def stop_at_execution(plan, **kwargs):
        assert plan['association_model_cache_capacity_bytes'] == 1
        raise ReachedAssociation
    monkeypatch.setattr(phewas_partition, 'gpu_model_cache_capacity', capacity)
    monkeypatch.setattr(phewas_run, 'execute_worker', stop_at_execution)
    with pytest.raises(ReachedAssociation):
        phewas_run.run_phewas('phenotype.csv', 'covariates.csv', 'prepared',
            output_directory=tmp_path, gpu_ids=(0,), binary_only=binary,
            continuous_transform='none', cpu_threads=1)
    assert observed == [dict(memory_limit_gib=None, reserve_bytes=2**30, binary_only=binary)]


def test_modern_public_binary_entry_transmits_its_contract_before_any_fit(tmp_path, monkeypatch):
    from test_prepared_association import fixture, call
    from fudan_wgs_toolkit import run, phewas_partition
    root, phenotype, covariate, _ = fixture(tmp_path)
    monkeypatch.setattr(run.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(run.torch.cuda, 'device_count', lambda: 2)
    observed = []
    class ReachedAdmission(Exception):
        pass
    def stop_at_capacity(gpu_ids, **policy):
        observed.append(policy)
        raise ReachedAdmission
    monkeypatch.setattr(phewas_partition, 'gpu_model_cache_capacity', stop_at_capacity)
    with pytest.raises(ReachedAdmission):
        call(root, phenotype, covariate, tmp_path/'analysis', memory_limit_gib=40)
    assert observed == [dict(memory_limit_gib=40, reserve_bytes=2**30, binary_only=True)]
