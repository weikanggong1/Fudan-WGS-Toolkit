"""Cache admission contracts; CUDA comparisons are component tests only."""
from types import SimpleNamespace
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit._cached_covariance import (
    _release_unused_cuda_blocks, cached_workspace_estimate,
    plan_cached_workspace, score_covariance_cached,
)
from fudan_wgs_toolkit.null_model import GaussianNullModel, KinshipSpectrum


GIB = 2**30


def test_admission_selects_full_then_two_panels_for_same_population():
    full = plan_cached_workspace(339013, 9854, 23, free_bytes=80 * GIB)
    assert full["full_resident"]
    assert full["panel_variant_size"] == 9854
    assert full["requested_variant_tile_size"] == full["effective_variant_tile_size"] == 4096
    long = plan_cached_workspace(339013, 21231, 23, free_bytes=80 * GIB)
    assert not long["full_resident"]
    assert long["panel_variant_size"] == 4096
    assert long["conservative_new_workspace_bytes"] < long["available_new_bytes"]
    # A full resident long matrix cannot be admitted even on an 80-GiB card.
    full_long = cached_workspace_estimate(
        339013, 21231, 23, variant_tile_size=4096,
        panel_variant_size=21231, full_resident=True)
    assert full_long["new_storage_bytes"] > 40 * GIB


def test_live_gpu_guard_does_not_credit_unreleased_allocator_fragments():
    args = (339013, 21231, 23)
    constrained = plan_cached_workspace(*args, free_bytes=24 * GIB)
    assert constrained["effective_variant_tile_size"] == 3584
    assert constrained["panel_variant_size"] == 3584
    assert constrained["available_new_bytes"] == 24 * GIB - 256 * 2**20
    fragmented = plan_cached_workspace(
        *args, free_bytes=24 * GIB, allocated_bytes=GIB,
        reserved_bytes=4 * GIB)
    assert fragmented["panel_variant_size"] == constrained["panel_variant_size"]
    assert fragmented["available_new_bytes"] == 24 * GIB - 256 * 2**20
    assert fragmented["allocator_budget_basis_bytes"] == 4 * GIB
    assert fragmented["allocator_reservation_credit_bytes"] == 0
    # A reservation consumes the process cap even when part is unallocated.
    capped = plan_cached_workspace(
        *args, free_bytes=80 * GIB, allocated_bytes=15 * GIB,
        reserved_bytes=30 * GIB)
    assert capped["effective_variant_tile_size"] == 512
    assert capped["available_new_bytes"] == 10 * GIB - 256 * 2**20
    assert capped["conservative_new_workspace_bytes"] <= capped["available_new_bytes"]


def test_population_panel_width_shrinks_within_twenty_gib_including_live_model():
    # Plan only: never allocate the population-sized dosage or covariance.
    n, m, q = 340795, 8021, 23
    allocated, reserved = 2 * GIB, 3 * GIB
    plan = plan_cached_workspace(
        n, m, q, memory_limit_gib=20, allocated_bytes=allocated,
        reserved_bytes=reserved, free_bytes=80 * GIB)
    assert not plan["full_resident"]
    assert plan["requested_variant_tile_size"] == 4096
    assert plan["effective_variant_tile_size"] == plan["panel_variant_size"] == 2560
    # Four original/weighted panels, one covariance and unchanged allowances.
    elements = (4 * n * 2560 + m * m + 6 * 2560**2 + 2 * q * m
                + m + 2 * n * 512 + 3 * n * 512)
    assert plan["conservative_new_workspace_bytes"] == 4 * elements
    assert plan["available_new_bytes"] == 20 * GIB - reserved - 256 * 2**20
    assert reserved + 4 * elements + 256 * 2**20 <= 20 * GIB
    # Even the minimum panel at the next larger aligned product cannot fit.
    next_width = cached_workspace_estimate(
        n, m, q, variant_tile_size=3072, panel_variant_size=3072)
    assert next_width["new_storage_bytes"] > plan["available_new_bytes"]
    full = cached_workspace_estimate(
        n, m, q, variant_tile_size=4096, panel_variant_size=m,
        full_resident=True)
    assert full["new_storage_bytes"] > plan["available_new_bytes"]
    with pytest.raises(MemoryError, match="requires"):
        plan_cached_workspace(
            n, m, q, panel_variant_size=4096, memory_limit_gib=20,
            allocated_bytes=allocated, reserved_bytes=reserved,
            free_bytes=80 * GIB)


def test_live_free_memory_can_require_smaller_tile_than_process_cap():
    plan = plan_cached_workspace(
        340795, 8021, 23, memory_limit_gib=20, allocated_bytes=2 * GIB,
        reserved_bytes=3 * GIB, free_bytes=8 * GIB)
    assert plan["effective_variant_tile_size"] == plan["panel_variant_size"] == 512
    assert plan["available_new_bytes"] == 8 * GIB - 256 * 2**20
    assert plan["conservative_new_workspace_bytes"] <= plan["available_new_bytes"]


def test_large_weighted_allocation_plan_accounts_for_unreleasable_reservations():
    # Plan only: the allocated shape needs two approximately 7-GiB original
    # and weighted panels. Aggregate free fragments cannot admit that pair.
    options = dict(memory_limit_gib=20, allocated_bytes=GIB, free_bytes=62 * GIB)
    clean = plan_cached_workspace(338000, 5500, 23, reserved_bytes=GIB, **options)
    assert clean["full_resident"]
    fragmented = plan_cached_workspace(338000, 5500, 23, reserved_bytes=8 * GIB, **options)
    assert not fragmented["full_resident"]
    assert fragmented["effective_variant_tile_size"] == fragmented["panel_variant_size"] == 1536
    assert fragmented["conservative_new_workspace_bytes"] + 8 * GIB + 256 * 2**20 <= 20 * GIB
    with pytest.raises(MemoryError, match="minimum 512-column"):
        plan_cached_workspace(338000, 5500, 23, reserved_bytes=20 * GIB, **options)
    # Inconsistent snapshots must not drop already live allocations.
    guarded = plan_cached_workspace(33, 100, 3, memory_limit_gib=20,
        allocated_bytes=8 * GIB, reserved_bytes=GIB, free_bytes=62 * GIB)
    assert guarded["allocator_budget_basis_bytes"] == 8 * GIB


def test_cache_cleanup_remeasures_selected_device_and_never_assumes_all_fragments_released(monkeypatch):
    state = dict(allocated=2 * GIB, reserved=18 * GIB, free=62 * GIB)
    contexts, active, empty_calls = [], [], []
    target = "cuda:3"

    @contextmanager
    def selected_device(device):
        assert device == target
        contexts.append(device); active.append(device)
        try:
            yield
        finally:
            active.pop()

    def on_target():
        assert active == [target]

    def empty_cache():
        on_target(); empty_calls.append(True)
        # Remaining reservations still contain unreleasable fragments.
        state["reserved"] = 4 * GIB
        state["free"] = 76 * GIB

    def memory_allocated(device):
        on_target(); assert device == target
        return state["allocated"]

    def memory_reserved(device):
        on_target(); assert device == target
        return state["reserved"]

    def mem_get_info():
        on_target()
        return state["free"], 80 * GIB

    monkeypatch.setattr(torch.cuda, "device", selected_device)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(torch.cuda, "memory_allocated", memory_allocated)
    monkeypatch.setattr(torch.cuda, "memory_reserved", memory_reserved)
    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
    cleanup = _release_unused_cuda_blocks(target)
    assert empty_calls == [True] and contexts == [target] * 3 and active == []
    assert cleanup["released_bytes"] == 14 * GIB
    assert cleanup["before"]["allocated_bytes"] == cleanup["after"]["allocated_bytes"] == 2 * GIB
    assert cleanup["after"]["reserved_bytes"] == 4 * GIB
    assert cleanup["after"]["free_bytes"] == 76 * GIB
    assert cleanup["host_wall_seconds"] >= 0
    plan = plan_cached_workspace(340795, 8021, 23, memory_limit_gib=20, **cleanup["after"])
    assert plan["allocator_budget_basis_bytes"] == 4 * GIB
    assert plan["available_new_bytes"] == 16 * GIB - 256 * 2**20
    assert plan["allocator_reservation_credit_bytes"] == 0


def test_long_cached_pipeline_guard_cleans_before_planning_and_uses_fresh_budget(monkeypatch):
    import fudan_wgs_toolkit._cached_covariance as cached
    from fudan_wgs_toolkit.pipeline import PheWASPipeline
    pipeline = PheWASPipeline.__new__(PheWASPipeline)
    pipeline.options = SimpleNamespace(sample_block_size=None, long_mask_threshold=5000,
        covariance_backend="cached", cached_variant_tile_size=4096,
        memory_limit_gib=20, long_mask_rank=512)
    model = SimpleNamespace(n=340795, n_pheno=1, matmul_mode="tf32", use_spa=False,
        spectrum=SimpleNamespace(blocks=[]), x=SimpleNamespace(shape=(340795, 23)),
        device="cuda:3")
    events = []
    after = dict(allocated_bytes=2 * GIB, reserved_bytes=3 * GIB, free_bytes=80 * GIB)
    original_plan = cached.plan_cached_workspace
    observed = {}

    def clean(device):
        assert device == model.device
        events.append("cleanup")
        return dict(after=after)

    def plan(n, m, q, **options):
        assert events == ["cleanup"]
        assert (n, m, q) == (340795, 8021, 23)
        for key, value in after.items():
            assert options[key] == value
        events.append("plan")
        observed.update(original_plan(n, m, q, **options))
        return observed

    monkeypatch.setattr(cached, "_release_unused_cuda_blocks", clean)
    monkeypatch.setattr(cached, "plan_cached_workspace", plan)
    estimate = pipeline._workspace_estimate(model, 8021)
    spectrum = 4 * (4 * 8021**2 + 16 * 8021 * 512 + model.n)
    assert estimate == max(observed["conservative_new_workspace_bytes"], spectrum)
    assert events == ["cleanup", "plan"]
    assert observed["allocator_budget_basis_bytes"] == after["reserved_bytes"]
    # Small masks and Single keep their established formula and do not invoke
    # CUDA cleanup or the long-mask planner at this earlier guard.
    events.clear()
    assert pipeline._workspace_estimate(model, 4999) == 4 * (4 * model.n * 4999 + 6 * 4999**2 + model.n)
    assert pipeline._workspace_estimate(model, 8021, individual=True) == 4 * (
        4 * model.n * 8021 + model.n + (23 + 8) * 8021)
    assert events == []


def test_minimum_tile_retains_all_allowances_and_rejects_one_byte_shortfall():
    n, m, q = 340795, 8021, 23
    minimum = 4 * (4 * n * 512 + m * m + 6 * 512**2 + 2 * q * m
                   + m + 5 * n * 512)
    options = dict(memory_limit_gib=20, allocated_bytes=2 * GIB,
                   reserved_bytes=2 * GIB)
    plan = plan_cached_workspace(
        n, m, q, free_bytes=minimum + 256 * 2**20, **options)
    assert plan["effective_variant_tile_size"] == plan["panel_variant_size"] == 512
    assert plan["conservative_new_workspace_bytes"] == minimum
    with pytest.raises(MemoryError, match="minimum 512-column"):
        plan_cached_workspace(
            n, m, q, free_bytes=minimum + 256 * 2**20 - 1, **options)
    with pytest.raises(MemoryError, match="minimum 512-column"):
        plan_cached_workspace(
            n, m, q, memory_limit_gib=8, allocated_bytes=2 * GIB,
            free_bytes=80 * GIB)


def test_explicit_panel_preserves_requested_tile_when_a_smaller_one_would_fit():
    with pytest.raises(MemoryError, match="requires"):
        plan_cached_workspace(340795, 8021, 23, panel_variant_size=4096,
                              memory_limit_gib=20, free_bytes=80 * GIB)
    explicit = plan_cached_workspace(
        340795, 8021, 23, variant_tile_size=512, panel_variant_size=1024,
        memory_limit_gib=20, allocated_bytes=2 * GIB, free_bytes=80 * GIB)
    assert explicit["requested_variant_tile_size"] == explicit["effective_variant_tile_size"] == 512
    assert explicit["panel_variant_size"] == 1024


def test_explicit_panel_and_too_large_dense_matrix_rejected_before_allocation():
    with pytest.raises(MemoryError, match="requires"):
        plan_cached_workspace(339013, 21231, 23, panel_variant_size=8192,
                              free_bytes=80 * GIB)
    with pytest.raises(MemoryError, match="cannot admit"):
        plan_cached_workspace(339013, 100000, 23, free_bytes=80 * GIB)
    empty = plan_cached_workspace(33, 0, 3)
    assert empty["cache_mode"] == "empty"
    assert empty["conservative_new_workspace_bytes"] == 0


@pytest.mark.parametrize("kwargs", [
    {"variant_tile_size": True}, {"variant_tile_size": 4095},
    {"panel_variant_size": 2048}, {"memory_limit_gib": 0},
    {"memory_limit_gib": float("nan")}, {"allocated_bytes": -1},
    {"free_bytes": -1},
])
def test_invalid_cache_options_have_no_implicit_fallback(kwargs):
    with pytest.raises(ValueError):
        plan_cached_workspace(33, 100, 3, **kwargs)


def test_explicit_large_gpu_workspace_budget_has_no_fixed_cap():
    result = plan_cached_workspace(33, 100, 3, memory_limit_gib=41)
    smaller = plan_cached_workspace(33, 100, 3, memory_limit_gib=20)
    assert result['available_new_bytes']-smaller['available_new_bytes'] == 21*2**30
    assert result['full_resident']


def test_cached_backend_rejects_cpu_fp64_and_unaccepted_symmetry():
    model = SimpleNamespace(
        family="gaussian", n_pheno=1, use_spa=False,
        spectrum=SimpleNamespace(blocks=[]), device=torch.device("cpu"),
        matmul_mode="tf32")
    with pytest.raises(ValueError, match="CUDA model"):
        score_covariance_cached(model, np.ones((4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="native TF32"):
        score_covariance_cached(model, np.ones((4, 3), dtype=np.float32),
                                matmul_mode="fp64")
    with pytest.raises(ValueError, match="symmetry=average"):
        score_covariance_cached(model, np.ones((4, 3), dtype=np.float32),
                                symmetry="mirror")


def _cuda_model(n):
    device = torch.device("cuda:0")
    x = torch.stack((torch.ones(n), torch.arange(n).float() / n,
                     (torch.arange(n) % 7).float() / 7), dim=1).to(device)
    inverse = (1.0 + (torch.arange(n) % 5).float() / 5).to(device)
    residual = ((torch.arange(n) % 11).float() / 11 - 0.5).to(device)
    zeros = torch.zeros(3, dtype=torch.float32, device=device)
    spectrum = KinshipSpectrum(torch.ones(n, device=device), [])
    return GaussianNullModel(
        sample_ids=np.arange(n).astype(str), x=x, scaled_residuals=residual,
        coefficients=zeros, theta=zeros, precision_theta=zeros,
        fixed_effect_covariance=torch.eye(3, device=device) / n,
        spectrum=spectrum, inverse_variance=inverse,
        precision_x=inverse[:, None] * x, iterations=1, converged=True,
        matmul_mode="tf32")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA component comparison")
@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("m", [513, 4097, 4609])
def test_cuda_full_and_two4096_panels_match_legacy512_at_tail_boundary(layout, m):
    from fudan_wgs_toolkit import tf32
    if torch.cuda.get_device_capability(0)[0] < 8:
        pytest.skip("native TF32 requires Ampere or newer")
    pytest.importorskip("triton")
    tf32.configure_tf32(memory_limit_gib=40, split_k=0)
    n = 1025
    generator = np.random.default_rng(173)
    genotype = np.array(generator.integers(0, 3, size=(n, m)),
                        dtype=np.float32, order=layout)
    model = _cuda_model(n)
    baseline_u, baseline_v = model.score_covariance_tiled(
        genotype, variant_tile_size=512)
    for panel_size in (None, 4096):
        u, v, report = model.score_covariance_cached(
            genotype, panel_variant_size=panel_size, profile=True)
        assert torch.equal(u, baseline_u)
        assert torch.equal(v, baseline_v)
        assert report["preparation_tile_size"] == 512
        assert report["variant_tile_size"] == 4096
        assert report["requested_variant_tile_size"] == report["effective_variant_tile_size"] == 4096
        assert report["cache_mode"] == (
            "full_original_and_weighted" if panel_size is None
            else "two_original_and_weighted_panels")
        assert report["prepared_variant_tiles"] == (m + 511) // 512
        assert report["covariance_d2h_bytes"] == 0
        assert report["max_observed_allocated_bytes"] <= 40 * GIB
        assert report["symmetry"] == "average"
        assert report["singleton_product_blocks"] > 0
        assert report["singleton_covariance_tile_size"] == 512
        del u, v
    # Validate the final column of the second panel, before returning outputs.
    bad = genotype.copy(order=layout)
    bad[-1, -1] = np.nan
    with pytest.raises(ValueError, match="genotype must be finite"):
        model.score_covariance_cached(bad, panel_variant_size=4096)


def _cuda_binary_model(n):
    from fudan_wgs_toolkit.binary_null import binary_prefitted_state
    x = np.column_stack((np.ones(n), np.arange(n) / n, np.arange(n) % 7 / 7)).astype(np.float32)
    precision = (.1 + np.arange(n) % 5 / 25).astype(np.float32)
    sx = precision[:, None] * x
    cov = (np.eye(3) / n).astype(np.float32)
    cov[0, 1] = .01 / n  # Both directed products must retain supplied asymmetry.
    return binary_prefitted_state(
        sample_ids=np.arange(n).astype(str), covariates=x,
        residual=(np.arange(n) % 11 / 11 - .5).astype(np.float32),
        fitted_probability=np.full(n, .4, dtype=np.float32), xw=sx.T,
        projection_left=x @ cov, fixed_effect_covariance=cov,
        precision=precision, precision_covariates=sx, has_kinship=True,
        use_spa=False, device="cuda:0", matmul_mode="tf32")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA component comparison")
@pytest.mark.parametrize("family", ["gaussian", "binomial"])
@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("n,m,effective,full", [(257, 1537, 1024, True), (4097, 2049, 512, False)])
def test_cuda_budget_reduces_actual_product_width_and_matches_legacy512(
        monkeypatch, family, layout, n, m, effective, full):
    import importlib
    from fudan_wgs_toolkit import tf32
    if torch.cuda.get_device_capability(0)[0] < 8:
        pytest.skip("native TF32 requires Ampere or newer")
    pytest.importorskip("triton")
    tf32.configure_tf32(memory_limit_gib=20, split_k=0)
    model = _cuda_model(n) if family == "gaussian" else _cuda_binary_model(n)
    genotype = np.array(np.random.default_rng(811).integers(0, 3, size=(n, m)),
                        dtype=np.float32, order=layout)
    # Include a noninteger imputed dosage; this test never changes its values.
    genotype[2, -1] = .173
    baseline_u, baseline_v = model.score_covariance_tiled(genotype, variant_tile_size=512)
    q = int(model.precision_x.shape[1])
    target = cached_workspace_estimate(
        n, m, q, variant_tile_size=effective,
        panel_variant_size=m if full else effective, full_resident=full)
    # A real per-call ceiling admits the target plus the unchanged reserve.
    # Existing baseline/model CUDA tensors are explicitly included in the cap.
    with torch.cuda.device(model.device):
        torch.cuda.empty_cache()
    reserved = torch.cuda.memory_reserved(model.device)
    cap_bytes = reserved + target["new_storage_bytes"] + 256 * 2**20 + 2**20
    cap_gib = cap_bytes / GIB
    assert cap_gib < 20
    product_shapes = []
    module = importlib.import_module(type(model).__module__)
    original_product = module.matmul

    def observed_product(a, b, *, mode):
        assert mode == "tf32" and a.dtype == b.dtype == torch.float32
        if a.ndim == b.ndim == 2 and a.shape[1] == n and a.shape[0] > q:
            product_shapes.append((int(a.shape[0]), int(b.shape[1])))
        return original_product(a, b, mode=mode)

    monkeypatch.setattr(module, "matmul", observed_product)
    u, v, report = model.score_covariance_cached(
        genotype, memory_limit_gib=cap_gib, profile=True)
    assert torch.equal(u, baseline_u) and torch.equal(v, baseline_v)
    assert report["requested_variant_tile_size"] == 4096
    assert report["effective_variant_tile_size"] == report["variant_tile_size"] == effective
    assert report["cache_mode"] == (
        "full_original_and_weighted" if full else "two_original_and_weighted_panels")
    assert product_shapes and max(max(shape) for shape in product_shapes) == effective
    assert report["preparation_tile_size"] == report["singleton_covariance_tile_size"] == 512
    assert report["prepared_variant_tiles"] == (m + 511) // 512
    assert report["symmetry"] == "average"
    assert report["singleton_product_blocks"] > 0
    assert report["covariance_d2h_bytes"] == 0
    assert report["max_observed_allocated_bytes"] <= cap_bytes
    assert report["conservative_new_workspace_bytes"] <= report["available_new_bytes"]
    assert report["admission_budget_basis"] == "remaining_cuda_allocator_reservations"
    assert report["allocator_budget_basis_bytes"] == max(
        report["allocated_before_bytes"], report["reserved_before_bytes"])
    assert report["allocator_reservation_credit_bytes"] == 0
    assert report["allocator_cleanup_calls"] == 1 + 2 * report["weighted_panel_calls"]
    assert report["allocator_cleanup_released_bytes"] >= 0
    assert report["allocator_cleanup_host_wall_seconds"] >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA allocator cleanup contract")
def test_cuda_cleanup_releases_deleted_scratch_and_preserves_live_tensor():
    from fudan_wgs_toolkit import tf32
    tf32.configure_tf32(memory_limit_gib=20, split_k=0)
    device = torch.device("cuda:0")
    live = torch.arange(1025, dtype=torch.float32, device=device)
    expected = live.clone()
    scratch = torch.empty((16 * 2**20,), dtype=torch.float32, device=device)
    del scratch
    cleanup = _release_unused_cuda_blocks(device)
    assert cleanup["before"]["allocated_bytes"] == cleanup["after"]["allocated_bytes"]
    assert cleanup["released_bytes"] == cleanup["before"]["reserved_bytes"] - cleanup["after"]["reserved_bytes"] >= 0
    assert cleanup["after"]["reserved_bytes"] >= cleanup["after"]["allocated_bytes"]
    assert torch.equal(live, expected)
