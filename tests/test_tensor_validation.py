"""CPU checks for bounded finite validation; these are not GPU benchmarks."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import fudan_wgs_toolkit.tensor_validation as validation
import fudan_wgs_toolkit.null_model as gaussian_module
import fudan_wgs_toolkit.binary_null as binary_module
import fudan_wgs_toolkit.statistics as statistics_module
from fudan_wgs_toolkit.null_model import GaussianNullModel, KinshipSpectrum
from fudan_wgs_toolkit.binary_null import BinaryNullModel


def matrix_view(layout, *, dtype=torch.float32):
    values = torch.arange(11 * 17, dtype=torch.float32).reshape(11, 17).to(dtype=dtype)
    if layout == "c":
        return values
    if layout == "f":
        return values.T.contiguous().T
    if layout == "strided":
        storage = torch.full((25, 53), -99, dtype=dtype)
        result = storage[2:24:2, 1:52:3]
        result.copy_(values)
        return result
    if layout == "offset_c":
        storage = torch.full((15, 17), -99, dtype=dtype)
        storage[2:13].copy_(values)
        return storage[2:13]
    if layout == "broadcast":
        return torch.arange(17, dtype=dtype)[None, :].expand(11, 17)
    raise AssertionError(layout)


class OperationTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.operations = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        output = function(*args, **(kwargs or {}))
        tensors = output if isinstance(output, (tuple, list)) else (output,)
        self.operations.append((str(function), [item for item in tensors if isinstance(item, torch.Tensor)]))
        return output


@pytest.mark.parametrize("layout", ["c", "f", "strided", "offset_c", "broadcast"])
def test_every_block_shares_storage_and_covers_all_values_without_copy(layout, monkeypatch):
    value = matrix_view(layout)
    visited = []
    actual_isfinite = torch.isfinite
    before = value.clone()
    original_stride, original_offset, original_version = value.stride(), value.storage_offset(), value._version

    def checked_finite(block):
        assert block.numel() <= 13
        assert block.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        assert block.device == value.device and block.dtype == value.dtype
        visited.append(block)
        return actual_isfinite(block)

    monkeypatch.setattr(torch, "isfinite", checked_finite)
    trace = OperationTrace()
    with trace:
        assert validation._all_finite_blocked(value, 13) is True
    assert sum(block.numel() for block in visited) == value.numel()
    inspected = torch.cat([block.reshape(-1) for block in visited]).sort().values
    torch.testing.assert_close(inspected, value.reshape(-1).sort().values, rtol=0, atol=0)
    torch.testing.assert_close(value, before, rtol=0, atol=0)
    assert (value.stride(), value.storage_offset(), value._version) == (original_stride, original_offset, original_version)
    names = [name for name, _ in trace.operations]
    assert sum("_local_scalar_dense" in name for name in names) == 1
    assert not any(any(forbidden in name for forbidden in ("clone", "contiguous", "copy_", "_to_copy"))
                   for name in names)
    # The actual isfinite decomposition must also stay within the block cap.
    for name, outputs in trace.operations:
        if name.startswith(("aten.abs.", "aten.ne.", "aten.eq.", "aten.mul.")):
            assert all(output.numel() <= 13 for output in outputs)


@pytest.mark.parametrize("layout", ["c", "f", "strided"])
@pytest.mark.parametrize("coordinate", [(0, 0), (0, 16), (5, 8), (10, 16)])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nan_and_both_infinities_are_found_at_chunk_edges(layout, coordinate, invalid, monkeypatch):
    value = matrix_view(layout)
    value[coordinate] = invalid
    visited_elements = []
    actual_isfinite = torch.isfinite

    def recording_finite(block):
        visited_elements.append(block.numel())
        return actual_isfinite(block)

    monkeypatch.setattr(torch, "isfinite", recording_finite)
    assert validation._all_finite_blocked(value, 13) is False
    # A bad early chunk must not short-circuit the remaining checks.
    assert sum(visited_elements) == value.numel()
    assert max(visited_elements) <= 13


def test_all_nan_still_inspects_complete_matrix(monkeypatch):
    value = torch.full((11, 17), float("nan")).T
    calls = []
    original = torch.isfinite

    def checked(block):
        calls.append(block.numel())
        return original(block)

    monkeypatch.setattr(torch, "isfinite", checked)
    assert validation._all_finite_blocked(value, 7) is False
    assert sum(calls) == value.numel() and len(calls) > 1


def test_noncontiguous_higher_dimensional_views_are_bounded_without_cloning():
    value = torch.arange(5 * 7 * 9, dtype=torch.float64).reshape(5, 7, 9)[::2, 1::2, ::2].permute(2, 0, 1)
    blocks = list(validation._iter_finite_blocks(value, 7))
    assert sum(block.numel() for block in blocks) == value.numel()
    assert all(block.numel() <= 7 and block.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
               for block in blocks)
    torch.testing.assert_close(torch.cat([block.reshape(-1) for block in blocks]).sort().values,
                               value.reshape(-1).sort().values, rtol=0, atol=0)
    assert validation._all_finite_blocked(value, 7) is True
    value[-1, -1, -1] = float("nan")
    assert validation._all_finite_blocked(value, 7) is False


@pytest.mark.parametrize("shape", [(0,), (11, 0), (0, 17), (0, 0)])
def test_empty_views_are_finite_without_elementwise_flags(shape, monkeypatch):
    value = torch.empty(shape)
    monkeypatch.setattr(torch, "isfinite", lambda value: pytest.fail("empty blocked tensor needs no flags"))
    assert list(validation._iter_finite_blocks(value, 3)) == []
    assert validation._all_finite_blocked(value, 3) is True
    assert validation.finite_check_scratch_bytes(shape) == 0


@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.int64, torch.float32, torch.float64,
                                  torch.complex64, torch.complex128])
def test_blocked_dtype_semantics_equal_original(dtype):
    value = matrix_view("f", dtype=dtype)
    assert validation._all_finite_blocked(value, 11) == bool(torch.isfinite(value).all())
    if dtype.is_floating_point or dtype.is_complex:
        value[10, 16] = complex(0, float("inf")) if dtype.is_complex else float("nan")
        assert validation._all_finite_blocked(value, 11) == bool(torch.isfinite(value).all()) is False


@pytest.mark.parametrize("value", [0., float("nan"), float("inf"), -float("inf")])
def test_scalar_semantics(value):
    tensor = torch.tensor(value)
    assert validation._all_finite_blocked(tensor, 1) == bool(torch.isfinite(tensor).all())


def test_validation_preserves_gradient_graph_and_input_version():
    leaf = torch.arange(11 * 17, dtype=torch.float32, requires_grad=True)
    value = leaf.reshape(11, 17).T
    version, graph = value._version, value.grad_fn
    assert validation._all_finite_blocked(value, 13) is True
    assert value.requires_grad and value.grad_fn is graph and value._version == version
    assert leaf.grad is None
    value.square().sum().backward()
    torch.testing.assert_close(leaf.grad, 2 * leaf.detach(), rtol=0, atol=0)


def test_cpu_public_entry_keeps_single_established_check(monkeypatch):
    value = matrix_view("strided")
    calls = []
    original = torch.isfinite
    monkeypatch.setattr(validation, "_all_finite_blocked", lambda *args: pytest.fail("CPU should retain its original expression"))

    def checked(tensor):
        calls.append(tensor)
        return original(tensor)

    monkeypatch.setattr(torch, "isfinite", checked)
    assert validation.tensor_all_finite(value) is True
    assert len(calls) == 1 and calls[0] is value
    value[-1, -1] = float("nan")
    assert validation.tensor_all_finite(value) is False
    assert len(calls) == 2 and calls[1] is value


@pytest.mark.parametrize('layout', ['c', 'f', 'strided'])
def test_statistics_validation_scans_every_block_and_rejects_last_nonfinite(layout, monkeypatch):
    value = matrix_view(layout)
    calls = []
    original = torch.isfinite
    def bounded(tensor):
        return validation._all_finite_blocked(tensor, 13)
    def observed(block):
        assert block.numel() <= 13
        assert block.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        calls.append(block.numel())
        return original(block)
    monkeypatch.setattr(statistics_module, 'tensor_all_finite', bounded)
    monkeypatch.setattr(torch, 'isfinite', observed)
    statistics_module._finite(value, 'genotype')
    assert sum(calls) == value.numel()
    calls.clear()
    value[-1, -1] = float('nan')
    with pytest.raises(ValueError, match='genotype must contain only finite values'):
        statistics_module._finite(value, 'genotype')
    assert sum(calls) == value.numel()


@pytest.mark.parametrize("shape,element_size", [((345_967, 4_999), 4), ((2,), 4), ((187,), 4),
                                              ((3_000_000,), 8), ((9_000_000,), 16)])
def test_cuda_routing_uses_shape_only_cap_without_gpu_allocation(monkeypatch, shape, element_size):
    number_elements = int(np.prod(shape))
    value = SimpleNamespace(device=torch.device("cuda"), layout=torch.strided,
        is_floating_point=lambda: True, is_complex=lambda: False,
        numel=lambda: number_elements, element_size=lambda: element_size)
    calls = []

    def blocked(tensor, limit):
        calls.append((tensor, limit))
        return False

    monkeypatch.setattr(validation, "_all_finite_blocked", blocked)
    assert validation.tensor_all_finite(value) is False
    assert len(calls) == 1 and calls[0][0] is value
    limit = calls[0][1]
    assert 1 <= limit <= 2**23
    assert limit <= max(1, number_elements // 2)
    assert (element_size + 3) * limit <= 64 * 1024**2
    assert validation.finite_check_scratch_bytes(shape, element_size=element_size) >= (element_size + 3) * limit
    if number_elements > 2 and element_size == 4:
        assert validation.finite_check_scratch_bytes(shape) <= 4 * number_elements


@pytest.mark.parametrize("floating,number_elements", [(False, 1_000_000), (True, 0)])
def test_cuda_integer_and_empty_routes_need_no_flags(monkeypatch, floating, number_elements):
    value = SimpleNamespace(device=torch.device("cuda"), layout=torch.strided,
        is_floating_point=lambda: floating, is_complex=lambda: False, numel=lambda: number_elements)
    monkeypatch.setattr(validation, "_all_finite_blocked", lambda *args: pytest.fail("this route needs no flags"))
    assert validation.tensor_all_finite(value) is True


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_invalid_internal_chunk_limit_is_rejected(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        list(validation._iter_finite_blocks(torch.ones((2, 3)), invalid))


def fitted_model(kind, mode):
    dtype = torch.float32 if mode == "tf32" else torch.float64
    n = 12
    x = torch.stack((torch.ones(n, dtype=dtype), torch.linspace(-1, 1, n, dtype=dtype)), dim=1)
    precision = torch.linspace(.2, .4, n, dtype=dtype)
    sx = precision[:, None] * x
    covariance = torch.linalg.inv(x.T @ sx) * .97
    residual = torch.linspace(-.3, .4, n, dtype=dtype)
    if kind == "gaussian":
        return GaussianNullModel(np.arange(n), x, residual, torch.zeros(2, dtype=dtype),
            torch.ones(2, dtype=dtype), torch.ones(2, dtype=dtype), covariance,
            KinshipSpectrum(torch.ones(n, dtype=dtype), []), precision, sx, 0, True,
            matmul_mode=mode)
    return BinaryNullModel(np.arange(n), x, residual, torch.full((n,), .4, dtype=dtype),
        sx.T, sx @ covariance, covariance, precision=precision, precision_x=sx,
        use_spa=False, matmul_mode=mode)


@pytest.mark.parametrize("kind", ["gaussian", "binary"])
@pytest.mark.parametrize("mode", ["tf32", "fp64"])
@pytest.mark.parametrize("method", ["score_covariance", "individual_score_variance"])
def test_model_products_and_outputs_remain_exactly_the_same(kind, mode, method, monkeypatch):
    module = gaussian_module if kind == "gaussian" else binary_module
    model = fitted_model(kind, mode)
    g = (torch.arange(model.n * 4, dtype=model.x.dtype).reshape(model.n, 4) % 3).T.contiguous().T
    before = g.clone()
    product_calls = []

    def cpu_product(left, right, *, mode):
        product_calls.append((mode, left.shape, right.shape))
        return left @ right

    monkeypatch.setattr(module, "matmul", cpu_product)
    monkeypatch.setattr(module, "tensor_all_finite", lambda value: bool(torch.isfinite(value).all()))
    expected = getattr(model, method)(g)
    expected_calls = list(product_calls)
    product_calls.clear()
    model._centered_single_projection_cache = None
    monkeypatch.setattr(module, "tensor_all_finite", lambda value: validation._all_finite_blocked(value, 7))
    actual = getattr(model, method)(g)
    assert product_calls == expected_calls
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
    torch.testing.assert_close(g, before, rtol=0, atol=0)


@pytest.mark.parametrize("kind,method", [("gaussian", "score_covariance"),
                                        ("gaussian", "individual_score_variance"),
                                        ("binary", "score_covariance"),
                                        ("binary", "individual_score_variance")])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_model_genotype_validation_still_rejects_all_nonfinite_values(kind, method, invalid, monkeypatch):
    module = gaussian_module if kind == "gaussian" else binary_module
    model = fitted_model(kind, "tf32")
    g = torch.zeros((model.n, 4)).T.contiguous().T
    g[-1, -1] = invalid
    monkeypatch.setattr(module, "tensor_all_finite", lambda value: validation._all_finite_blocked(value, 7))
    monkeypatch.setattr(module, "matmul", lambda *args, **kwargs: pytest.fail("invalid input must fail before association products"))
    with pytest.raises(ValueError, match="finite samples-by-variants"):
        getattr(model, method)(g)
