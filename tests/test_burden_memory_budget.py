"""CPU control-flow checks with mocked CUDA counters; no GPU benchmark."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
import weakref

import pytest
import torch

from fudan_wgs_toolkit import _burden, statistics, tf32


GIB = 2**30


class _CudaOperand(torch.Tensor):
    """CPU storage reporting CUDA metadata solely to exercise admission."""

    @staticmethod
    def __new__(cls, shape):
        return torch.zeros(shape, dtype=torch.float32).as_subclass(cls)

    @property
    def is_cuda(self):
        return True

    @property
    def device(self):
        return torch.device('cuda:0')


@pytest.fixture
def mocked_cuda(monkeypatch):
    state = SimpleNamespace(allocated=21 * GIB, reserved=22 * GIB,
                            free=50 * GIB, queries=0, mm_calls=0,
                            seen_flags=[], mm_error=None)
    monkeypatch.setattr(tf32, '_memory_limit_bytes', 80 * GIB)
    monkeypatch.setattr(tf32, '_memory_reserve_bytes', 256 * 2**20)
    metrics = deepcopy(_burden._METRICS)
    metrics.update(selected_fp32_mm_calls=0, live_workspace_queries=0,
                   workspace_rejections=0, last_rejection=None)
    monkeypatch.setattr(_burden, '_METRICS', metrics)
    monkeypatch.delenv('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', raising=False)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: state.allocated)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: state.reserved)
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())

    def memory_info():
        state.queries += 1
        return state.free, 80 * GIB

    monkeypatch.setattr(torch.cuda, 'mem_get_info', memory_info)
    precision = torch.get_float32_matmul_precision()
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32

    def selected_mm(left, right):
        state.mm_calls += 1
        state.seen_flags.append((torch.get_float32_matmul_precision(),
                                 torch.backends.cuda.matmul.allow_tf32))
        if state.mm_error is not None:
            raise state.mm_error
        return torch.zeros((left.shape[0], right.shape[1]), dtype=torch.float32)

    monkeypatch.setattr(torch, 'mm', selected_mm)
    try:
        yield state
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def _operands():
    return _CudaOperand((8, 8)), _CudaOperand((8, 2))


def test_default_burden_budget_inherits_global_limit_above_20_gib(mocked_cuda):
    result = _burden.ieee_burden_product(*_operands())
    assert result.shape == (8, 2) and result.dtype == torch.float32
    assert mocked_cuda.allocated > 20 * GIB
    assert mocked_cuda.mm_calls == mocked_cuda.queries == 1
    assert _burden.execution_metadata()['workspace_rejections'] == 0


def test_guard_defaults_inherit_global_budget(mocked_cuda):
    left = SimpleNamespace(shape=(8, 8), device='cuda:0')
    right = SimpleNamespace(shape=(8, 2), device='cuda:0')
    _burden.guard_workspace(left, right)
    assert mocked_cuda.queries == 1 and mocked_cuda.mm_calls == 0


def test_explicit_smaller_cap_rejects_before_mm_with_anonymous_diagnostics(mocked_cuda):
    with pytest.raises(MemoryError, match=r'binding=process_cap.*no precision/chunk fallback'):
        _burden.ieee_burden_product(*_operands(), memory_limit_gib=8)
    report = _burden.execution_metadata()
    rejection = report['last_rejection']
    assert mocked_cuda.mm_calls == 0
    assert rejection['process_limit_bytes'] == 8 * GIB
    assert rejection['allocated_bytes'] == 21 * GIB
    assert rejection['binding'] == 'process_cap' and rejection['available_bytes'] == 0
    assert rejection['left_shape'] == [8, 8] and rejection['right_shape'] == [8, 2]
    assert rejection['required_bytes'] == 4 * (64 + 16 + 16)
    assert report['workspace_rejections'] == 1
    # A caller editing its report cannot mutate the stored numerical snapshot.
    rejection['left_shape'][0] = 999
    assert _burden.execution_metadata()['last_rejection']['left_shape'] == [8, 8]


def test_explicit_larger_cap_cannot_override_global_limit(mocked_cuda, monkeypatch):
    monkeypatch.setattr(tf32, '_memory_limit_bytes', 8 * GIB)
    with pytest.raises(MemoryError, match='binding=process_cap'):
        _burden.ieee_burden_product(*_operands(), memory_limit_gib=80)
    assert _burden.execution_metadata()['last_rejection']['process_limit_bytes'] == 8 * GIB
    assert mocked_cuda.mm_calls == 0


def test_guard_explicit_byte_cap_cannot_override_global_limit(mocked_cuda, monkeypatch):
    monkeypatch.setattr(tf32, '_memory_limit_bytes', 8 * GIB)
    with pytest.raises(MemoryError, match='binding=process_cap'):
        _burden.guard_workspace(*_operands(), limit_bytes=80 * GIB)
    assert _burden.execution_metadata()['last_rejection']['process_limit_bytes'] == 8 * GIB


def test_live_free_memory_still_rejects_with_large_budget(mocked_cuda):
    mocked_cuda.allocated = mocked_cuda.reserved = GIB
    mocked_cuda.free = 0
    with pytest.raises(MemoryError, match='binding=live_free'):
        _burden.ieee_burden_product(*_operands())
    rejection = _burden.execution_metadata()['last_rejection']
    assert rejection['process_limit_bytes'] == 80 * GIB
    assert rejection['live_available_bytes'] < 0 and rejection['available_bytes'] == 0
    assert mocked_cuda.mm_calls == 0


def test_configured_reserve_is_inherited(mocked_cuda, monkeypatch):
    monkeypatch.setattr(tf32, '_memory_reserve_bytes', GIB)
    mocked_cuda.allocated = mocked_cuda.reserved = GIB
    mocked_cuda.free = GIB // 2
    with pytest.raises(MemoryError, match='binding=live_free'):
        _burden.ieee_burden_product(*_operands())
    assert _burden.execution_metadata()['last_rejection']['reserve_bytes'] == GIB
    assert mocked_cuda.mm_calls == 0


@pytest.mark.parametrize('limit', [True, False, 0, -1, float('inf'), float('nan'), '80'])
def test_invalid_explicit_gib_budget_rejected_before_query(mocked_cuda, limit):
    with pytest.raises(ValueError, match='memory_limit_gib'):
        _burden.ieee_burden_product(*_operands(), memory_limit_gib=limit)
    assert mocked_cuda.queries == mocked_cuda.mm_calls == 0


@pytest.mark.parametrize('fails', [False, True])
def test_ieee_flags_are_applied_and_restored_on_success_or_error(mocked_cuda, fails):
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    previous = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)
    if fails:
        mocked_cuda.mm_error = RuntimeError('anonymous product failure')
        with pytest.raises(RuntimeError, match='anonymous product failure'):
            _burden.ieee_burden_product(*_operands())
    else:
        _burden.ieee_burden_product(*_operands())
    assert mocked_cuda.seen_flags == [('highest', False)]
    assert (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32) == previous
    assert _burden.execution_metadata()['selected_fp32_mm_calls'] == (0 if fails else 1)


def test_tf32_environment_override_is_still_rejected(mocked_cuda, monkeypatch):
    monkeypatch.setenv('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', '1')
    with pytest.raises(ValueError, match='unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'):
        _burden.ieee_burden_product(*_operands())
    assert mocked_cuda.queries == mocked_cuda.mm_calls == 0


def test_native_burden_workspace_is_released_before_probability_output(monkeypatch):
    score = torch.tensor([.2, -.1, .5], dtype=torch.float32)
    covariance = torch.tensor([[2., .1, .3], [.1, 1., .2], [.3, .2, 3.]], dtype=torch.float32)
    payload = dict(score=score, covariance=covariance, maf=[.001, .002, .004],
                   mac=[1., 2., 30.], annotations=[[10.], [20.], [30.]],
                   _skat_pvalues=[.1, .2, .3, .4])
    expected = statistics.association_test(**payload, matmul_mode='fp64')
    product_refs = []
    rare_covariance_refs = []
    original_cct = statistics._cct_rows

    def cpu_product(left, right, **kwargs):
        if product_refs:
            # The full product has already been reduced before rare allocation.
            assert product_refs[0]() is None
            rare_covariance_refs.append(weakref.ref(left))
        result = torch.mm(left, right)
        product_refs.append(weakref.ref(result))
        return result

    def check_released_then_cct(*args, **kwargs):
        assert len(product_refs) == 2 and len(rare_covariance_refs) == 1
        assert all(ref() is None for ref in product_refs + rare_covariance_refs)
        return original_cct(*args, **kwargs)

    monkeypatch.setattr(_burden, 'ieee_burden_product', cpu_product)
    monkeypatch.setattr(statistics, '_cct_rows', check_released_then_cct)
    actual = statistics.association_test(**payload, matmul_mode='tf32')
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], abs=2e-6, rel=2e-6)
