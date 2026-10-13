"""Bounded binary cMAC contracts; CUDA peak checks require a real GPU."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from fudan_wgs_toolkit.binary import prepare_binary_burdens
from fudan_wgs_toolkit.pipeline import _cmac_scalar_sum


class ReductionInputs(TorchDispatchMode):
    def __init__(self, source):
        super().__init__()
        self.source_storage = source.untyped_storage().data_ptr()
        self.blocks = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if str(function).startswith("aten.sum."):
            value = args[0]
            if value.untyped_storage().data_ptr() == self.source_storage:
                self.blocks.append((value.numel(), value.dtype, value.stride()))
        return function(*args, **(kwargs or {}))


@pytest.mark.parametrize("layout", ["c", "f"])
def test_complete_large_cmac_uses_bounded_views_and_exact_scalar(layout):
    """Inspect actual reduction inputs, including a fractional final block."""
    n, m = 8193, 1025
    source = torch.ones((n, m) if layout == "c" else (m, n), dtype=torch.float32)
    if layout == "f":
        source = source.T
    source[-1, -1] = .125
    before = (source.untyped_storage().data_ptr(), source.stride(), source._version)
    trace = ReductionInputs(source)
    with trace:
        actual = _cmac_scalar_sum(source)
    assert actual == n*m-.875
    assert len(trace.blocks) > 1
    assert sum(size for size, _, _ in trace.blocks) == source.numel()
    assert max(size for size, _, _ in trace.blocks) <= 8*2**20
    assert all(dtype == torch.float32 for _, dtype, _ in trace.blocks)
    assert (source.untyped_storage().data_ptr(), source.stride(), source._version) == before


def test_host_binary_panels_retain_original_burdens_and_cmac():
    """The resident-only change leaves the established CPU panel path intact."""
    source = np.asarray([[0., 1., .125, 0.], [1., 0., 2., .25], [0., 1., 0., 1.]],
                        dtype=np.float32, order="F")
    model = SimpleNamespace(family="binomial", n_pheno=1, matmul_mode="fp64",
        n=3, device=torch.device("cpu"), sample_ids=np.asarray(["a", "b", "c"]))
    result = prepare_binary_burdens(model, source, np.asarray([.001, .002, .003, .004]),
                                   variant_tile_size=2)
    torch.testing.assert_close(result.burdens,
        torch.as_tensor(source, dtype=torch.float64) @ result.weights, rtol=0, atol=1e-12)
    assert result.cmac == source.sum(dtype=np.float64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA memory check")
@pytest.mark.parametrize("layout", ["c", "f"])
def test_real_cuda_binary_cmac_reuses_bounded_helper_and_peak(layout, monkeypatch):
    """Exercise the actual resident burden path, not an injected CUDA allocator."""
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("native TF32 requires an Ampere or newer GPU")
    pytest.importorskip("triton")
    from fudan_wgs_toolkit import pipeline, tf32
    device = torch.device("cuda:0")
    model = SimpleNamespace(family="binomial", n_pheno=1, matmul_mode="tf32",
        n=32768, device=device, sample_ids=np.arange(32768).astype(str))
    free, total = torch.cuda.mem_get_info(device)
    if free < 512*2**20:
        pytest.skip("insufficient live headroom for the bounded CUDA component check")
    tf32.configure_tf32(memory_limit_gib=total/2**30, memory_reserve_bytes=32*2**20)
    # Warm the exact native product route before measuring its component peak.
    prepare_binary_burdens(SimpleNamespace(**{**vars(model), "n": 16,
        "sample_ids": np.arange(16).astype(str)}),
        torch.ones((16, 3), dtype=torch.float32, device=device), np.full(3, .001))
    n, m = model.n, 513
    source = torch.ones((n, m) if layout == "c" else (m, n),
                        dtype=torch.float32, device=device)
    if layout == "f":
        source = source.T
    source[-1, -1] = .125
    actual_helper = pipeline._cmac_scalar_sum
    calls, samples = [], []
    def observed(value):
        calls.append(value)
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        result = actual_helper(value)
        torch.cuda.synchronize(device)
        samples.append(torch.cuda.max_memory_allocated(device)-before)
        return result
    monkeypatch.setattr(pipeline, "_cmac_scalar_sum", observed)
    result = prepare_binary_burdens(model, source, np.full(m, .001))
    assert len(calls) == 1 and calls[0] is source
    assert result.cmac == n*m-.875
    # This measures only cMAC's actual CUDA peak above its already live input.
    # It does not claim an end-to-end memory or scientific benchmark.
    assert samples[0] <= 64*2**20 + 2*2**20
    del calls[:]
