"""Exact Boolean CCT predicates in one kernel; no floating-point reduction."""
import torch
import triton
import triton.language as tl


@triton.jit
def _predicates(P, OUT, N, STRIDE, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    active = i < N
    p = tl.load(P + i * STRIDE, active, other=0.5)
    finite = (p == p) & (tl.abs(p) != float('inf'))
    invalid = active & ~finite
    outside = active & ((p < 0) | (p > 1))
    zero = active & (p == 0)
    one = active & (p == 1)
    small = active & (p < 1e-16)
    tl.store(OUT, tl.sum(invalid.to(tl.int32), 0) == 0)
    tl.store(OUT + 1, tl.sum(outside.to(tl.int32), 0) != 0)
    tl.store(OUT + 2, tl.sum(zero.to(tl.int32), 0) != 0)
    tl.store(OUT + 3, tl.sum(one.to(tl.int32), 0) != 0)
    tl.store(OUT + 4, tl.sum(small.to(tl.int32), 0) != 0)


def cct_flags(p):
    # The caller bounds scratch and launch geometry to small CUDA vectors.
    flags = torch.empty(5, dtype=torch.bool, device=p.device)
    with torch.cuda.device(p.device):
        _predicates[(1,)](p, flags, p.numel(), p.stride(0),
                         BLOCK=triton.next_power_of_2(p.numel()),
                         num_warps=4, enable_fp_fusion=False)
    return flags.cpu().tolist()
