"""Geometry reuse and batched decomposition; not a population benchmark."""
import unittest
from unittest.mock import patch
from contextlib import nullcontext
import numpy as np
import torch
from fudan_wgs_toolkit._fit_structure_cache import FitStructureCache, available_cuda_bytes, guard_cuda_workspace
from fudan_wgs_toolkit.null_model import KinshipSpectrum, fit_gaussian_null
from fudan_wgs_toolkit.phewas_models import SparseKinshipData, prepare_phewas_model
from fudan_wgs_toolkit.phewas_run import _design

def mixed_grm():
    n = 13
    rows = np.array([0, 2, 4, 6, 7, 9, 10], dtype=np.int64)
    cols = np.array([1, 3, 5, 7, 8, 10, 11], dtype=np.int64)
    return SparseKinshipData(np.arange(n).astype(str), np.linspace(0.9, 1.1, n), rows, cols, np.full(len(rows), 0.12))

def spectrum_parameters(grm):
    parameters = grm.fit_parameters()
    parameters['diagonal'] = parameters.pop('kinship_diagonal')
    return parameters

class _Monkeypatch:

    def __init__(self, test):
        self.test = test

    def setattr(self, target, name, value):
        owner = patch.object(target, name, value)
        owner.start()
        self.test.addCleanup(owner.stop)

class TestFitStructureCache(unittest.TestCase):

    def test_eigh_batches_only_equal_size_families_and_preserves_axis(self):
        monkeypatch = _Monkeypatch(self)
        grm = mixed_grm()
        calls = []
        original = torch.linalg.eigh

        def record(matrix, *args, **kwargs):
            calls.append(tuple(matrix.shape))
            return original(matrix, *args, **kwargs)
        monkeypatch.setattr(torch.linalg, 'eigh', record)
        batched = KinshipSpectrum.from_sparse(**spectrum_parameters(grm))
        assert calls == [(3, 2, 2), (2, 3, 3)]
        scalar = KinshipSpectrum.from_sparse(**spectrum_parameters(grm), eigh_batch_bytes=1)
        assert calls[2:] == [(2, 2)] * 3 + [(3, 3)] * 2
        assert [rows.tolist() for (rows, _) in batched.blocks] == [rows.tolist() for (rows, _) in scalar.blocks]
        torch.testing.assert_close(batched.eigenvalues, scalar.eigenvalues, atol=1e-14, rtol=1e-14)
        z = torch.arange(39, dtype=torch.float64).reshape(13, 3) / 17
        torch.testing.assert_close(batched.rotate(z), scalar.rotate(z), atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(batched.rotate(batched.rotate(z), inverse=True), z, atol=1e-14, rtol=1e-14)

    def test_batched_psd_and_family_size_guards_remain_errors(self):
        grm = mixed_grm()
        with self.assertRaisesRegex(ValueError, 'block exceeds'):
            KinshipSpectrum.from_sparse(**spectrum_parameters(grm), max_block_size=2)
        values = grm.edge_values.copy()
        values[0] = 2.0
        with self.assertRaisesRegex(ValueError, 'positive semidefinite'):
            KinshipSpectrum.from_sparse(grm.diagonal, grm.edge_rows, grm.edge_cols, values)
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            KinshipSpectrum.from_sparse(**spectrum_parameters(grm), eigh_batch_bytes=0)

    def test_spectrum_key_binds_order_grm_values_and_dtype(self):
        monkeypatch = _Monkeypatch(self)
        grm = mixed_grm()
        cache = FitStructureCache()
        builds = []
        original = KinshipSpectrum.from_sparse.__func__

        def record(cls, *args, **kwargs):
            builds.append(kwargs['dtype'])
            return original(cls, *args, **kwargs)
        monkeypatch.setattr(KinshipSpectrum, 'from_sparse', classmethod(record))
        with cache.acquire_spectrum(grm.sample_ids, **spectrum_parameters(grm)) as first:
            first_values = first.eigenvalues
        with cache.acquire_spectrum(grm.sample_ids.copy(), **spectrum_parameters(grm)) as second:
            assert second is not first
            assert second.eigenvalues is first_values
        changed = grm.diagonal.copy()
        changed[0] += 0.01
        with cache.acquire_spectrum(grm.sample_ids, changed, grm.edge_rows, grm.edge_cols, grm.edge_values):
            pass
        with cache.acquire_spectrum(grm.sample_ids[::-1], **spectrum_parameters(grm)):
            pass
        with cache.acquire_spectrum(grm.sample_ids, **spectrum_parameters(grm), dtype=torch.float32):
            pass
        assert builds == [torch.float64, torch.float64, torch.float64, torch.float32]
        assert cache.stats()['hits'] == 1
        assert cache.stats()['active_leases'] == 0

    def test_mutated_cached_spectrum_is_rebuilt_and_conversion_is_isolated(self):
        grm = mixed_grm()
        cache = FitStructureCache()
        with cache.acquire_spectrum(grm.sample_ids, **spectrum_parameters(grm)) as borrowed:
            original = borrowed.eigenvalues.clone()
            borrowed.eigenvalues = borrowed.eigenvalues.float()
            borrowed.blocks = [(rows, rotation.float()) for (rows, rotation) in borrowed.blocks]
        with cache.acquire_spectrum(grm.sample_ids, **spectrum_parameters(grm)) as borrowed:
            assert borrowed.eigenvalues.dtype == torch.float64
            torch.testing.assert_close(borrowed.eigenvalues, original, atol=0, rtol=0)
            borrowed.eigenvalues.add_(0.5)
        with cache.acquire_spectrum(grm.sample_ids, **spectrum_parameters(grm)) as rebuilt:
            torch.testing.assert_close(rebuilt.eigenvalues, original, atol=0, rtol=0)
        assert cache.stats()['misses'] == 2

    def test_design_key_binds_sample_order_covariates_and_names(self):
        ids = np.arange(10).astype(str)
        raw = np.column_stack((np.linspace(-2, 2, 10), np.ones(10)))
        cache = FitStructureCache()
        calls = []

        def build(values, names):
            calls.append(1)
            return _design(values, names)
        with cache.acquire_design(ids, raw, ['first', 'constant'], build) as (x, metadata):
            first = x
            assert not x.flags.writeable
            metadata['center'][0] = 99
        with cache.acquire_design(ids, raw.copy(), ['first', 'constant'], build) as (x, metadata):
            assert x is first
            assert metadata['center'][0] != 99
        for (new_ids, new_raw, names) in ((ids[::-1], raw, ['first', 'constant']), (ids, raw + 0.1, ['first', 'constant']), (ids, raw, ['renamed', 'constant'])):
            with cache.acquire_design(new_ids, new_raw, names, build):
                pass
        assert len(calls) == 4

    def test_lru_does_not_evict_active_leases_and_oversized_entries_are_uncached(self):
        ids = np.arange(8).astype(str)
        raw = np.arange(8, dtype=float).reshape(-1, 1)
        cache = FitStructureCache(max_entries=1)
        with cache.acquire_design(ids, raw, ['x'], _design) as (first, _):
            with cache.acquire_design(ids, raw + 1, ['x'], _design):
                assert cache.stats()['entries'] == 1
                assert cache.stats()['active_leases'] == 2
            with cache.acquire_design(ids, raw, ['x'], _design) as (again, _):
                assert again is first
            with self.assertRaisesRegex(RuntimeError, 'active leases'):
                cache.close()
        with cache.acquire_design(ids, raw + 2, ['x'], _design):
            pass
        assert cache.stats()['evictions'] == 1
        cache.close()
        assert cache.stats()['retained_bytes'] == 0
        tiny = FitStructureCache(max_bytes=1)
        with tiny.acquire_design(ids, raw, ['x'], _design):
            assert tiny.stats()['retained_bytes'] == 0
            with self.assertRaisesRegex(RuntimeError, 'active leases'):
                tiny.close()
        assert tiny.stats()['active_leases'] == 0

    def test_subset_reuse_binds_exact_source_and_requested_order(self):
        grm = mixed_grm()
        cache = FitStructureCache()
        ids = grm.sample_ids[[5, 4, 0, 1]]
        with cache.acquire_kinship_subset(grm, ids) as first:
            pass
        with cache.acquire_kinship_subset(grm, ids.copy()) as again:
            assert again is first
        with cache.acquire_kinship_subset(grm, ids[::-1]) as reversed_subset:
            np.testing.assert_array_equal(reversed_subset.sample_ids, ids[::-1])
            assert reversed_subset is not first
        changed = SparseKinshipData(grm.sample_ids, grm.diagonal + 0.1, grm.edge_rows, grm.edge_cols, grm.edge_values)
        with cache.acquire_kinship_subset(changed, ids) as new_subset:
            assert new_subset is not first
            np.testing.assert_allclose(new_subset.diagonal, first.diagonal + 0.1)

    def test_gaussian_fit_cache_preserves_independent_fits_and_fp64_spectrum(self):
        n = 48
        ids = np.arange(n).astype(str)
        x = np.column_stack((np.ones(n), np.linspace(-1, 1, n)))
        grm = SparseKinshipData(ids, np.ones(n), np.arange(0, n, 2), np.arange(1, n, 2), np.full(n // 2, 0.15))
        cache = FitStructureCache()
        y = np.sin(np.arange(n)) + 0.4 * x[:, 1]
        original_spectrum = KinshipSpectrum.from_sparse.__func__
        def scalar_spectrum(cls, *args, **kwargs):
            kwargs['eigh_batch_bytes'] = 1
            return original_spectrum(cls, *args, **kwargs)
        with patch.object(KinshipSpectrum, 'from_sparse', classmethod(scalar_spectrum)):
            reference = prepare_phewas_model(y, ids, x, kinship=grm,
                continuous_transform='none', association_mode='fp64')
        first = prepare_phewas_model(y, ids, x, kinship=grm, continuous_transform='none', association_mode='tf32', fit_structure_cache=cache)
        second = prepare_phewas_model(y, ids, x, kinship=grm, continuous_transform='none', association_mode='fp64', fit_structure_cache=cache)
        assert first.model.spectrum.eigenvalues.dtype == torch.float32
        assert second.model.spectrum.eigenvalues.dtype == torch.float64
        for name in ('coefficients', 'theta', 'precision_theta', 'scaled_residuals', 'fixed_effect_covariance', 'inverse_variance', 'precision_x'):
            torch.testing.assert_close(getattr(reference.model, name), getattr(second.model, name), atol=5e-12, rtol=5e-12)
        assert second.model.iterations == reference.model.iterations
        assert second.model.converged == reference.model.converged
        assert cache.stats()['hits'] >= 2
        assert cache.stats()['active_leases'] == 0
        third = prepare_phewas_model(y + 0.3 * np.cos(np.arange(n)), ids, x, kinship=grm, continuous_transform='none', association_mode='fp64', fit_structure_cache=cache)
        assert third.model.theta.data_ptr() != second.model.theta.data_ptr()
        assert third.model.scaled_residuals.data_ptr() != second.model.scaled_residuals.data_ptr()
        cache.close()

    def test_fit_error_releases_spectrum_lease(self):
        monkeypatch = _Monkeypatch(self)
        import fudan_wgs_toolkit.null_model as module
        grm = mixed_grm()
        cache = FitStructureCache()

        def fail(*args, **kwargs):
            raise ArithmeticError('deliberate fit failure')
        monkeypatch.setattr(module, '_fit_gaussian_null', fail)
        with self.assertRaisesRegex(ArithmeticError, 'deliberate'):
            fit_gaussian_null(np.arange(13, dtype=float), sample_ids=grm.sample_ids, **grm.fit_parameters(), fit_structure_cache=cache)
        assert cache.stats()['active_leases'] == 0
        cache.close()

    def test_inference_mode_spectrum_is_not_retained_without_mutation_versions(self):
        grm = mixed_grm()
        cache = FitStructureCache()
        with torch.inference_mode():
            with cache.acquire_spectrum(grm.sample_ids, **spectrum_parameters(grm)):
                assert cache.stats()['active_leases'] == 1
        assert cache.stats()['entries'] == 0
        assert cache.stats()['active_leases'] == 0

    def test_cuda_admission_uses_owned_cache_live_free_and_process_cap(self):
        monkeypatch = _Monkeypatch(self)
        from fudan_wgs_toolkit import tf32
        monkeypatch.setattr(tf32, 'execution_metadata', lambda : dict(tf32_memory_guard=dict(process_allocated_limit_bytes=500, live_cuda_reserve_bytes=20)))
        monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 100)
        monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 150)
        monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda : (200, 1000))
        monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
        assert available_cuda_bytes('cpu') is None
        assert available_cuda_bytes('cuda:0') == 230
        guard_cuda_workspace(230, 'cuda:0')
        with self.assertRaisesRegex(MemoryError, 'available 230'):
            guard_cuda_workspace(231, 'cuda:0', context='test structure')
        monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 490)
        assert available_cuda_bytes('cuda:0') == 10

    def test_modified_design_is_not_reused_or_evicted_during_active_lease(self):
        ids = np.arange(8).astype(str)
        raw = np.arange(8, dtype=float).reshape(-1, 1)
        cache = FitStructureCache()
        (expected, _) = _design(raw, ['x'])
        with cache.acquire_design(ids, raw, ['x'], _design) as (x, _):
            x.flags.writeable = True
            x[0, 0] = 99
            with self.assertRaisesRegex(RuntimeError, 'modified in place'):
                with cache.acquire_design(ids, raw, ['x'], _design):
                    pass
        with cache.acquire_design(ids, raw, ['x'], _design) as (rebuilt, _):
            np.testing.assert_array_equal(rebuilt, expected)
        assert cache.stats()['active_leases'] == 0

    def test_uncacheable_subset_does_not_evict_reusable_spectrum(self):
        n = 16
        ids = np.arange(n).astype(str)
        grm = SparseKinshipData(ids, np.ones(n), np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64), np.empty(0))
        cache = FitStructureCache(max_bytes=1024)
        raw = np.zeros((n, 1))
        def builder(values, names):
            return np.zeros((n, 2)), {'columns': 2}
        for _ in range(3):
            with cache.acquire_design(ids, raw, ['x'], builder):
                with cache.acquire_kinship_subset(grm, ids) as subset:
                    with cache.acquire_spectrum(ids, **spectrum_parameters(subset)):
                        pass
        self.assertEqual(cache.stats()['misses'], 5)  # design, spectrum, three subsets
        self.assertEqual(cache.stats()['hits'], 4)
        self.assertEqual(cache.stats()['uncached_builds'], 3)
        self.assertEqual(cache.stats()['evictions'], 0)
        self.assertEqual(cache.stats()['entries'], 2)
        cache.close()

    def test_population_dimension_design_subset_spectrum_fit_default_budget(self):
        # Resource-capacity regression at the actual sample/design dimensions.
        # Anonymous zero arrays test byte admission, not scientific fit accuracy.
        n = 343034
        ids = np.arange(n).astype(str)
        raw = np.zeros((n, 15), dtype=np.float64)
        grm = SparseKinshipData(ids, np.ones(n), np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64), np.empty(0))
        cache = FitStructureCache()
        builds = []
        def builder(values, names):
            builds.append(1)
            return np.zeros((n, 16), dtype=np.float64), {'columns': 16}
        for _ in range(2):
            with cache.acquire_design(ids, raw, list(map(str, range(15))), builder):
                with cache.acquire_kinship_subset(grm, ids) as subset:
                    with cache.acquire_spectrum(ids, **spectrum_parameters(subset)):
                        pass
        self.assertEqual(n * 16 * 8, 43908352)
        self.assertEqual(len(builds), 1)
        self.assertEqual(cache.stats()['misses'], 3)
        self.assertEqual(cache.stats()['hits'], 3)
        self.assertEqual(cache.stats()['entries'], 3)
        self.assertEqual(cache.stats()['evictions'], 0)
        self.assertLessEqual(cache.stats()['retained_bytes'], cache.max_bytes)
        self.assertEqual(cache.stats()['active_leases'], 0)
        cache.close()

if __name__ == '__main__':
    unittest.main()
