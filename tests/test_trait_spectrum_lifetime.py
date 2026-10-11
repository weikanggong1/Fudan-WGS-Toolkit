"""Held tracebacks must release existing single-trait spectrum owners."""
import gc
import unittest
from unittest.mock import patch
import weakref

import torch

from fudan_wgs_toolkit import _weighted_spectra as spectra
from fudan_wgs_toolkit import statistics


class NativeSpectrumLifetimeTests(unittest.TestCase):
    def _failed_native(self, error_type=RuntimeError):
        covariance = torch.eye(3)
        weights = torch.tensor([[1., 2.], [2., 1.], [3., 4.]])
        owners = [weakref.ref(value) for value in (covariance, weights)]
        try:
            spectra.native_weighted_spectra(covariance, weights)
        except error_type as error:
            return error, owners
        finally:
            covariance = weights = None
        self.fail('injected eigensolver failure did not propagate')

    def test_real_capacity_rejection_held_exception_drops_argument_owners(self):
        # Exercise the actual workspace leaf, not an injected leaf which
        # already discarded its own arguments. The unchanged formula must
        # reject one complete matrix at this budget without a precision retry.
        with patch.object(spectra, 'WEIGHTED_BATCH_LIMIT_BYTES', new=1):
            held, inputs = self._failed_native(error_type=MemoryError)
        gc.collect()
        self.assertIsInstance(held, MemoryError)
        self.assertIsNotNone(held.__traceback__)
        self.assertTrue(all(owner() is None for owner in inputs))

    def test_held_solver_exception_releases_native_inputs_and_intermediates(self):
        owners = []
        original_relations = statistics._native_weight_relations

        def relations(rows):
            result = original_relations(rows)
            owners.extend(weakref.ref(value) for value in (rows, result[0], result[1]))
            return result

        def fail_leaf(matrices, **kwargs):
            owners.append(weakref.ref(matrices))
            # Isolate the toolkit's caller frames from a deliberately retained
            # test leaf frame. This leaf contributes no tensor argument owner.
            matrices = None
            raise RuntimeError('injected eigensolver failure')

        with patch.object(statistics, '_native_weight_relations', new=relations), \
                patch.object(torch.linalg, 'eigvalsh', new=fail_leaf):
            held, inputs = self._failed_native()
        gc.collect()
        self.assertIsNotNone(held.__traceback__)
        self.assertTrue(all(owner() is None for owner in inputs+owners))

    def test_partial_second_solve_failure_releases_completed_spectra(self):
        owners = []
        calls = 0
        original = torch.linalg.eigvalsh

        def leaf(matrices, **kwargs):
            nonlocal calls
            calls += 1
            owners.append(weakref.ref(matrices))
            if calls == 2:
                matrices = None
                raise RuntimeError('injected eigensolver failure')
            values = original(matrices, **kwargs)
            owners.append(weakref.ref(values))
            return values

        with patch.object(spectra, 'WEIGHTED_BATCH_LIMIT_BYTES',
                          new=spectra.workspace_unit_bytes(torch.eye(3))), \
                patch.object(torch.linalg, 'eigvalsh', new=leaf):
            held, inputs = self._failed_native()
        gc.collect()
        self.assertEqual(calls, 2)
        self.assertIsNotNone(held.__traceback__)
        self.assertTrue(all(owner() is None for owner in inputs+owners))

    def test_invalid_shape_held_exception_drops_argument_owners(self):
        def invalid():
            covariance = torch.eye(4)
            weights = torch.ones(3, 2)
            owners = [weakref.ref(value) for value in (covariance, weights)]
            try:
                spectra.native_weighted_spectra(covariance, weights)
            except ValueError as error:
                return error, owners
            finally:
                covariance = weights = None
        held, owners = invalid()
        gc.collect()
        self.assertIsNotNone(held.__traceback__)
        self.assertTrue(all(owner() is None for owner in owners))

    def test_solver_lock_rejection_also_drops_argument_owner(self):
        class BusyLock:
            def acquire(self, **kwargs):
                return False
            def release(self):
                raise AssertionError('unacquired lock was released')

        def rejected():
            matrices = torch.eye(3)[None]
            owner = weakref.ref(matrices)
            try:
                spectra._complete_eigvalsh(matrices)
            except RuntimeError as error:
                return error, owner
            finally:
                matrices = None
        with patch.object(spectra, '_SOLVER_LOCK', new=BusyLock()):
            held, owner = rejected()
        gc.collect()
        self.assertIsNotNone(held.__traceback__)
        self.assertIsNone(owner())

    def test_success_preserves_original_two_multiply_spectrum_control(self):
        a = torch.tensor([[3., .2, .1], [.2, 2., .3], [.1, .3, 1.]])
        weights = torch.tensor([[1., 2., .5], [2., 4., 1.], [3., 6., 2.]])
        before_covariance, before_weights = a.clone(), weights.clone()
        control = torch.stack([torch.linalg.eigvalsh(a*w[:, None]*w[None, :], UPLO='U')
                               for w in weights.T]).double()
        actual = spectra.native_weighted_spectra(a, weights)
        torch.testing.assert_close(actual, control, rtol=2e-6, atol=1e-7)
        self.assertTrue(torch.equal(a, before_covariance))
        self.assertTrue(torch.equal(weights, before_weights))


if __name__ == '__main__':
    unittest.main()
