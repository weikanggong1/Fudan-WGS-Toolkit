"""CPU contracts: configurable budgets retain fresh live-memory protection."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock
import unittest

from fudan_wgs_toolkit.cuda_eigen import _backend as backend
from fudan_wgs_toolkit.cuda_eigen.small_solver import FP32SmallSpectrumSolver
from fudan_wgs_toolkit import _weighted_spectra as weighted, tf32
from test_cuda_eigen_backend import API, Torch, Tensor

GIB=2**30

class CapacityCUDA:
    def __init__(self,allocated,free,reserved=None):
        self.allocated=allocated;self.free=free
        self.reserved=allocated if reserved is None else reserved
    def memory_allocated(self,device):return self.allocated
    def memory_reserved(self,device):return self.reserved
    def mem_get_info(self):return self.free,80*GIB
    def device(self,device):return nullcontext()

class Covariance:
    is_cuda=True;device='cuda:0';shape=(256,256)
    def numel(self):return 256*256
    def element_size(self):return 4

class MemoryBudgetContracts(unittest.TestCase):
    def test_selector_accepts_explicit_40gib_without_loading_backend(self):
        t=Torch()
        solver=FP32SmallSpectrumSolver(torch_module=t,memory_limit=40*GIB,
            _backend_factory=lambda **kwargs:self.fail('constructor loaded solver backend'))
        self.assertEqual(solver.memory_limit,40*GIB)
        self.assertIsNone(solver.backend)
        solver.close()
        self.assertEqual(FP32SmallSpectrumSolver(torch_module=t).memory_limit,40*GIB)

    def test_positive_integer_budget_validation_rejects_bool_float_zero_negative(self):
        for value in (False,True,0,-1,40.0,'40',None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    FP32SmallSpectrumSolver(torch_module=Torch(),memory_limit=value)
                with self.assertRaises(ValueError):
                    backend.available_bytes(0,0,0,limit=value)

    def test_40gib_selected_route_preserves_full_spectrum_and_original_scope(self):
        t=Torch();api=API()
        solver=FP32SmallSpectrumSolver(torch_module=t,memory_limit=40*GIB,
            _backend_factory=lambda **kwargs:backend.BatchedEigenBackend(api=api,**kwargs))
        output=solver.eigvalsh(Tensor((2,33,33)))
        self.assertEqual(output.shape,(2,33))
        self.assertEqual(solver.backend.limit,40*GIB)
        self.assertEqual(solver.last_execution_backend,'cusolverDnSsyevjBatched')
        self.assertEqual(solver.backend.report()['eigenvalues'],66)
        self.assertEqual(solver.backend.report()['info_nonzero'],0)
        self.assertEqual(solver.eigvalsh(Tensor((2,32,32))),'original')
        self.assertEqual(solver.last_execution_backend,'torch.linalg.eigvalsh')
        solver.close()

    def test_process_cap_still_limits_available_workspace_above_20gib(self):
        self.assertEqual(backend.available_bytes(22*GIB,22*GIB,60*GIB,
            limit=40*GIB,reserve=0),18*GIB)
        self.assertEqual(backend.available_bytes(41*GIB,41*GIB,60*GIB,
            limit=40*GIB,reserve=0),0)
        self.assertEqual(backend.available_bytes(100,200,300,limit=1000,reserve=50),350)

    def test_actual_free_and_reserve_still_limit_workspace(self):
        self.assertEqual(backend.available_bytes(1*GIB,1*GIB,100*2**20,
            limit=40*GIB,reserve=256*2**20),0)
        self.assertEqual(backend.available_bytes(1*GIB,2*GIB,300*2**20,
            limit=40*GIB,reserve=256*2**20),GIB+44*2**20)

    def test_backend_guard_still_blocks_process_cap_and_free_memory(self):
        for allocated,free in ((39*GIB,60*GIB),(GIB,100*2**20)):
            t=Torch();t.cuda.allocated=allocated;t.cuda.reserved=allocated;t.cuda.free=free
            api=API();engine=backend.BatchedEigenBackend(torch_module=t,api=api,memory_limit=40*GIB)
            with self.assertRaises(MemoryError):engine._guard(2*GIB)
            self.assertEqual(api.calls,0)
            self.assertEqual(engine.report()['workspace_rejections'],1)
            engine.close()

    def test_solver_geometry_and_cusolver_size_guard_unchanged(self):
        self.assertEqual(backend.geometry((3,4,4)),(4,3,(3,4),252))
        for shape in ((1,513,513),(0,33,33),(33,34),(2**31,33,33)):
            with self.assertRaises(ValueError):backend.geometry(shape)

    def capacity(self,limit,allocated,free):
        t=SimpleNamespace(cuda=CapacityCUDA(allocated,free))
        with mock.patch.object(weighted,'torch',t),mock.patch.object(tf32,'_memory_limit_bytes',limit), \
                mock.patch.object(weighted,'allocator_bytes',
                    return_value=(allocated,t.cuda.reserved)) as allocator:
            try:
                return weighted._batch_capacity(Covariance(),1000)
            finally:
                allocator.assert_called_once_with(Covariance.device)

    def test_weighted_batches_use_40gib_process_budget_and_keep_512mib_batch_cap(self):
        count=self.capacity(40*GIB,22*GIB,60*GIB)
        self.assertGreater(count,0)
        self.assertLessEqual(count*weighted.workspace_unit_bytes(Covariance()),512*2**20)
        with self.assertRaises(MemoryError):self.capacity(20*GIB,22*GIB,60*GIB)

    def test_weighted_batches_still_reject_process_cap_and_insufficient_live_free(self):
        with self.assertRaises(MemoryError):self.capacity(40*GIB,40*GIB+1,60*GIB)
        with self.assertRaises(MemoryError):self.capacity(40*GIB,GIB,100*2**20)

if __name__=='__main__':unittest.main()
