"""Constant precision component contracts; real-data accuracy is checked separately."""
import numpy as np
import pytest
import torch
from contextlib import nullcontext

from fudan_wgs_toolkit.null_model import fit_gaussian_null


def _fitted_model(n=41, *, device="cpu"):
    rows = np.arange(n, dtype=np.float64)
    x = np.column_stack((np.ones(n), rows / n, (rows % 7) / 7))
    y = np.sin(rows * .37) + .2 * np.cos(rows * .11) + .4 * x[:, 1]
    model = fit_gaussian_null(y, covariates=x, device=device, matmul_mode="fp64")
    model.set_matmul_mode("tf32")
    return model


def _fitted_state(model):
    return {name: getattr(model, name).clone() for name in (
        "x", "scaled_residuals", "coefficients", "theta", "precision_theta",
        "inverse_variance", "precision_x", "fixed_effect_covariance")}


def _independent_projector_control(model, genotype):
    """Build the small full projector from the original saved fitted state."""
    g = torch.as_tensor(genotype, dtype=torch.float64, device=model.device)
    a = model.precision_x.double()
    projector = torch.diag(model.inverse_variance.double()) - a @ model.fixed_effect_covariance.double() @ a.T
    return g.T @ model.scaled_residuals.double(), g.T @ projector @ g


def _component_roundoff_bound(model, genotype):
    """Precision-derived component bound, including projection cancellation."""
    g = torch.as_tensor(genotype, dtype=torch.float64, device=model.device).abs()
    a = model.precision_x.double().abs()
    c = model.fixed_effect_covariance.double().abs()
    gram_scale = float((g.T @ (model.inverse_variance.double().abs()[:, None]*g)).max())
    cross_scale = a.T @ g
    projection_scale = float((cross_scale.T @ c @ cross_scale).max())
    # Six TF32 input-rounding sites bound the projection: A on each side,
    # each saved cross-product, C and the intermediate projected-left input.
    # Four FP32 dot-product sums have K <= N. Use absolute component scales,
    # because Gram - projection can make individual V entries nearly zero.
    tf32_half_ulp = .5 * 2.0**-10
    fp32_half_ulp = torch.finfo(torch.float32).eps / 2
    gamma_n = model.n*fp32_half_ulp / (1-model.n*fp32_half_ulp)
    return (6*tf32_half_ulp + 4*gamma_n + 6*fp32_half_ulp) * (gram_scale+projection_scale)


def test_normalized_state_uses_saved_precision_without_changing_fit():
    model = _fitted_model()
    # The final theta need not be the state used to form the returned precision.
    model.theta[0].mul_(1.25)
    before = _fitted_state(model)
    inv, _, _, scale = model._covariance_product_state()
    assert torch.equal(inv, torch.ones_like(inv))
    assert torch.equal(scale, before["inverse_variance"][0])
    assert not torch.equal(scale, model.theta[0].reciprocal())
    for name, original in before.items():
        assert torch.equal(getattr(model, name), original), name


def test_covariance_is_returned_in_original_units_against_full_projector(monkeypatch):
    import fudan_wgs_toolkit.null_model as module
    model = _fitted_model()
    g = (np.arange(model.n * 7).reshape(model.n, 7) % 3).astype(np.float32)
    expected_u, expected_v = _independent_projector_control(model, g)
    # CPU component control exercises algebra and fitted-state rounding only;
    # it is not evidence of CUDA execution or an end-to-end benchmark.
    monkeypatch.setattr(module, "matmul", lambda a, b, *, mode: a.double() @ b.double())
    u, v = model.score_covariance(g)
    torch.testing.assert_close(u, expected_u, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(v, expected_v, rtol=2e-6, atol=2e-6)
    _, sample_v = model.score_covariance_sample_block(g, sample_block_size=13)
    _, tiled_v = model.score_covariance_tiled(g, variant_tile_size=3)
    torch.testing.assert_close(sample_v.double(), expected_v, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(tiled_v.double(), expected_v, rtol=2e-6, atol=2e-6)


def test_cache_reuses_storage_and_invalidates_mutation_replacement_and_mode():
    model = _fitted_model()
    first = model._covariance_product_state()
    second = model._covariance_product_state()
    assert all(a is b for a, b in zip(first, second))
    model.precision_x.add_(.0001)
    changed = model._covariance_product_state()
    assert changed[1] is not first[1]
    model.inverse_variance = model.inverse_variance.clone().mul_(1.5)
    replacement = model._covariance_product_state()
    assert replacement[0] is not changed[0]
    assert torch.equal(replacement[3], model.inverse_variance[0])
    model.set_matmul_mode("tf32")
    assert model._constant_precision_covariance_cache is None
    refreshed = model._covariance_product_state()
    assert refreshed[1] is not replacement[1]
    # Cache references keep replaced source identities alive until invalidation.
    assert model._constant_precision_covariance_cache[1][0] is model.inverse_variance


def _mock_cache_admission(monkeypatch, module, *, allocated, cap, reserved=None):
    """Exercise actual CUDA admission arithmetic with CPU operand storage."""
    actual = module._guard_constant_precision_cache_allocation
    calls = []
    def simulated(sources, device):
        calls.append(tuple(value.shape for value in sources))
        actual(sources, torch.device("cuda:0"))
    monkeypatch.setattr(module, "_guard_constant_precision_cache_allocation", simulated)
    monkeypatch.setattr(module._tf32, "execution_metadata", lambda: {
        "tf32_memory_guard": {"process_allocated_limit_bytes": cap, "live_cuda_reserve_bytes": 0}})
    monkeypatch.setattr(module._tf32, "_allocator_backend", lambda: "native")
    monkeypatch.setattr(module, "allocator_bytes", lambda device:
        (allocated, allocated if reserved is None else reserved))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (2**30, 2**30))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    return calls


def test_first_cache_low_budget_rejects_before_scalar_or_state_allocation(monkeypatch):
    import fudan_wgs_toolkit.null_model as module
    model = _fitted_model()
    before = _fitted_state(model)
    calls = _mock_cache_admission(monkeypatch, module, allocated=1024, cap=1025)
    with monkeypatch.context() as allocation:
        allocation.setattr(torch.Tensor, "clone", lambda *args, **kwargs:
            pytest.fail("owned scalar/cache allocation preceded budget admission"))
        with pytest.raises(MemoryError, match="Constant precision covariance cache requires estimated"):
            model._covariance_product_state()
    assert len(calls) == 1
    assert model._constant_precision_covariance_cache is None
    for name, value in before.items():
        assert torch.equal(getattr(model, name), value), name


def test_cached_state_reuse_does_not_readmit_or_allocate(monkeypatch):
    import fudan_wgs_toolkit.null_model as module
    model = _fitted_model()
    calls = _mock_cache_admission(monkeypatch, module, allocated=1024, cap=2**20, reserved=2**20)
    first = model._covariance_product_state()
    with monkeypatch.context() as allocation:
        allocation.setattr(torch.Tensor, "clone", lambda *args, **kwargs:
            pytest.fail("a cache hit cloned normalized state"))
        second = model._covariance_product_state()
    assert len(calls) == 1
    assert all(a is b for a, b in zip(first, second))


def test_invalidated_cache_references_are_removed_before_live_snapshot(monkeypatch):
    import fudan_wgs_toolkit.null_model as module
    model = _fitted_model()
    model._covariance_product_state()
    model.precision_x.add_(.0001)
    calls = _mock_cache_admission(monkeypatch, module, allocated=1024, cap=2**20)
    def snapshot(device):
        assert model._constant_precision_covariance_cache is None
        return 1024, 1024
    monkeypatch.setattr(module, "allocator_bytes", snapshot)
    model._covariance_product_state()
    assert len(calls) == 1


def test_nonconstant_state_does_not_request_normalization_allocation(monkeypatch):
    import fudan_wgs_toolkit.null_model as module
    model = _fitted_model()
    model.inverse_variance[0].mul_(1.01)
    monkeypatch.setattr(module, "_guard_constant_precision_cache_allocation", lambda *args:
        pytest.fail("an ineligible state requested normalization storage"))
    inv, a, c, scale = model._covariance_product_state()
    assert inv is model.inverse_variance and a is model.precision_x and c is model.fixed_effect_covariance
    assert scale is None


@pytest.mark.parametrize("gate", ["fp64", "nonconstant", "blocks", "multiple", "spa", "family"])
def test_ineligible_states_retain_exact_original_operands(gate):
    model = _fitted_model()
    mode = "tf32"
    if gate == "fp64":
        model.set_matmul_mode("fp64")
        mode = "fp64"
    elif gate == "nonconstant":
        model.inverse_variance[0].mul_(1.01)
    elif gate == "blocks":
        model.spectrum.blocks = [(torch.tensor([0, 1]), torch.eye(2))]
    elif gate == "multiple":
        model.n_pheno = 2
    elif gate == "spa":
        model.use_spa = True
    elif gate == "family":
        model.family = "binomial"
    inv, a, c, scale = model._covariance_product_state(mode)
    assert inv is model.inverse_variance
    assert a is model.precision_x
    assert c is model.fixed_effect_covariance
    assert scale is None


def test_fp64_covariance_retains_original_math():
    model = _fitted_model()
    model.set_matmul_mode("fp64")
    g = (np.arange(model.n * 5).reshape(model.n, 5) % 3).astype(np.float64)
    expected_u, expected_v = _independent_projector_control(model, g)
    u, v = model.score_covariance(g)
    torch.testing.assert_close(u, expected_u, rtol=0, atol=0)
    torch.testing.assert_close(v, expected_v, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("invalid_scale", [0., float("nan"), float("inf")])
def test_invalid_constant_scale_preserves_existing_error_path(invalid_scale):
    model = _fitted_model()
    model.inverse_variance.fill_(invalid_scale)
    inv, a, c, scale = model._covariance_product_state()
    assert inv is model.inverse_variance and a is model.precision_x and c is model.fixed_effect_covariance
    assert scale is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native TF32 CUDA component contract")
@pytest.mark.parametrize("layout", ["C", "F"])
def test_native_backend_scale_contract_and_no_extra_mma(layout, record_property):
    from fudan_wgs_toolkit import tf32
    if torch.cuda.get_device_capability(0)[0] < 8:
        pytest.skip("native TF32 requires Ampere or newer")
    pytest.importorskip("triton")
    tf32.configure_tf32(memory_limit_gib=20, split_k=0)
    model = _fitted_model(513, device="cuda:0")
    generator = np.random.default_rng(198)
    g = np.array(generator.integers(0, 3, size=(model.n, 513)), dtype=np.float32, order=layout)
    tf32.execution_metadata(reset=True)
    u, v = model.score_covariance(g)
    report = tf32.execution_metadata()
    assert report["logical_product_count"] == 5
    assert report["tf32_gemm_call_count"] == 4
    assert report["fp32_vector_product_count"] == 1
    assert report["component_reconstruction_product_count"] == 0
    assert report["fp64_gemm_fallback_count"] == 0
    # Singleton tails keep their established FP32 routes. Different product
    # geometry has FP32 rounding differences; original units must still agree.
    sample_u, sample_v = model.score_covariance_sample_block(g, sample_block_size=model.n)
    tiled_u, tiled_v = model.score_covariance_tiled(g, variant_tile_size=512)
    cached_u, cached_v, cached_report = model.score_covariance_cached(g, variant_tile_size=512)
    expected_u, expected_v = _independent_projector_control(model, g)
    torch.testing.assert_close(u.double(), expected_u, rtol=2e-5, atol=2e-5)
    # TF32 has 10 stored fraction bits. Derive a fixed component bound from
    # input ulps and the original Gram/projection scales, not observed error.
    # These component bounds do not replace real-data -log10(P) <= .001.
    matrix_bound = _component_roundoff_bound(model, g)
    record_property("tf32_component_roundoff_bound", matrix_bound)
    record_property("ordinary_tail_fp64_max_error", float((v[-1].double()-expected_v[-1]).abs().max()))
    for backend, other_v in (("sample", sample_v), ("tiled", tiled_v), ("cached", cached_v)):
        record_property(backend+"_tail_fp64_max_error", float((other_v[-1].double()-expected_v[-1]).abs().max()))
    assert float((v.double() - expected_v).abs().max()) <= matrix_bound
    for backend, other_u, other_v in (("sample", sample_u, sample_v),
                                      ("tiled", tiled_u, tiled_v),
                                      ("cached", cached_u, cached_v)):
        torch.testing.assert_close(other_u, u, rtol=2e-5, atol=2e-5)
        # The non-singleton block keeps the same actual MMA contract.
        assert torch.equal(other_v[:-1, :-1], v[:-1, :-1])
        # The singleton uses the historical FP32 GEMV/dot contract. Compare
        # it independently to the saved-state full FP64 projector, without
        # forcing it to imitate the less precise ordinary TF32 tail.
        tail_error = float((other_v[-1].double()-expected_v[-1]).abs().max())
        assert tail_error <= matrix_bound
    assert cached_report["allocated_before_bytes"] >= sum(value.numel()*value.element_size()
        for value in model._covariance_product_state() if value is not None)
