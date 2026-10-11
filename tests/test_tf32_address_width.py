"""Actual >int32 storage offsets; small integer matrices are exact controls.

The sparse-layout allocation is about 8 GiB. It writes six values, keeps the
existing CUDA cap/live-free guard, and tests all four operand stride directions.
This synthetic component does not attribute any real-data failure or speedup.
"""
import unittest
import torch

from fudan_wgs_toolkit import tf32
from fudan_wgs_toolkit._fit_structure_cache import available_cuda_bytes, guard_cuda_workspace


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
class NativeAddressWidth(unittest.TestCase):
    def test_low_k_strided_product_retains_rna_ties_and_fp32_output(self):
        device = torch.device('cuda',torch.cuda.current_device())
        if torch.cuda.get_device_capability(device)[0] < 8:
            self.skipTest('native TF32 requires Ampere or newer')
        source = torch.tensor([[1+2**-11,-1-2**-11],[2**-10,3+2**-11],
                               [-3,1+3*2**-11]],dtype=torch.float32)
        # Explicit nearest/ties-away rounded values; the second product has
        # only two terms, so the FP32 oracle is independent of matrix dispatch.
        rounded = torch.tensor([[1+2**-10,-1-2**-10],[2**-10,3],
                                [-3,1+2**-9]],dtype=torch.float32)
        b = torch.tensor([[1.25,-2.75,3.5],[-.875,4,1.125]],dtype=torch.float32)
        expected = rounded[:,0,None]*b[0] + rounded[:,1,None]*b[1]
        storage = torch.empty((3,4),dtype=torch.float32,device=device)
        left = storage[:,::2]
        left.copy_(source)
        actual = tf32.matmul(left,b.to(device),mode='tf32')
        self.assertEqual(actual.dtype,torch.float32)
        self.assertTrue(torch.equal(actual.cpu().view(torch.int32),expected.view(torch.int32)))

    def test_stride_below_int32_with_offsets_above_int32_on_each_operand_axis(self):
        stride = 1 << 30
        elements = 2*stride+2
        required = 4*elements
        device = torch.device('cuda',torch.cuda.current_device())
        if torch.cuda.get_device_capability(device)[0] < 8:
            self.skipTest('native TF32 requires Ampere or newer')
        available = available_cuda_bytes(device)
        if available < required+2**20:
            self.skipTest('live CUDA policy cannot admit sparse 8GiB address component')
        storage = large = left = right = actual = oracle = dense = None
        try:
            guard_cuda_workspace(required+2**20,device,context='native 64-bit address component')
            storage = torch.empty((elements,),dtype=torch.float32,device=device)
            # AS0, AS1, BS0 and BS1 remain signed-int32 scalar parameters;
            # the offset multiplication, rather than the stride itself, needs
            # 64 bits. No GEMV/outer route can bypass the native GEMM checks.
            for side, shape, strides in (
                    ('left',(3,2),(stride,1)),('left',(2,3),(1,stride)),
                    ('right',(3,2),(stride,1)),('right',(2,3),(1,stride))):
                with self.subTest(operand=side, strides=strides):
                    large = storage.as_strided(shape,strides)
                    dense = torch.arange(1,7,dtype=torch.float32,device=device).reshape(shape)
                    large.copy_(dense)
                    if side == 'left':
                        left = large
                        right = torch.arange(1,shape[1]*2+1,dtype=torch.float32,device=device).reshape(shape[1],2)
                        oracle = dense @ right
                    else:
                        left = torch.arange(1,2*shape[0]+1,dtype=torch.float32,device=device).reshape(2,shape[0])
                        right = large
                        oracle = left @ dense
                    actual = tf32.matmul(left,right,mode='tf32')
                    torch.cuda.synchronize(device)
                    self.assertTrue(torch.equal(actual.view(torch.int32),oracle.view(torch.int32)))
                    left = right = actual = oracle = large = dense = None
        finally:
            storage = large = left = right = actual = oracle = dense = None


if __name__ == '__main__':
    unittest.main()
