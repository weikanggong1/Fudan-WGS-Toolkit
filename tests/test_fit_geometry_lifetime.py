"""Failed geometry constructors release their own and borrowed traceback refs."""
import gc
import weakref

import numpy as np
import pytest
import torch

import fudan_wgs_toolkit._fit_structure_cache as structure
import fudan_wgs_toolkit.binary_null as binary
from fudan_wgs_toolkit.null_model import KinshipSpectrum


def _held_spectrum_error(error_type):
    diagonal = np.ones(4)
    rows = np.array([0, 2])
    cols = rows + 1
    values = np.full(2, .12)
    inputs = [weakref.ref(value) for value in (diagonal, rows, cols, values)]
    try:
        KinshipSpectrum.from_sparse(diagonal, rows, cols, values,
                                    eigh_batch_bytes=1)
    except error_type as error:
        return error, inputs
    finally:
        diagonal = rows = cols = values = None
    pytest.fail("injected spectrum failure did not propagate")


def _track_spectrum_allocations(monkeypatch):
    owners = []
    original_tensor, original_eigh = torch.as_tensor, torch.linalg.eigh
    def tensor(*args, **kwargs):
        result = original_tensor(*args, **kwargs)
        owners.append(weakref.ref(result))
        if args and isinstance(args[0], np.ndarray):
            owners.append(weakref.ref(args[0]))
        return result
    def eigh(matrix, *args, **kwargs):
        result = original_eigh(matrix, *args, **kwargs)
        owners.extend(weakref.ref(value) for value in result)
        return result
    monkeypatch.setattr(torch, "as_tensor", tensor)
    monkeypatch.setattr(torch.linalg, "eigh", eigh)
    return owners


def test_held_psd_error_releases_complete_solver_chunk(monkeypatch):
    owners = _track_spectrum_allocations(monkeypatch)
    original_eigh = torch.linalg.eigh
    def negative(matrix, *args, **kwargs):
        values, rotation = original_eigh(matrix, *args, **kwargs)
        # Exercise the constructor's real PSD rejection, while retaining
        # only weak references in this successful injected solver frame.
        values[0] = -1.
        return values, rotation
    monkeypatch.setattr(torch.linalg, "eigh", negative)
    held, inputs = _held_spectrum_error(ValueError)
    assert "not positive semidefinite" in str(held)
    gc.collect()
    assert held.__traceback__ is not None
    assert all(owner() is None for owner in owners + inputs)


def test_held_second_solver_error_releases_partial_blocks(monkeypatch):
    owners = _track_spectrum_allocations(monkeypatch)
    original_eigh = torch.linalg.eigh
    calls = 0
    def fail(matrix, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            matrix = None
            raise RuntimeError("second family solver failed")
        return original_eigh(matrix, *args, **kwargs)
    monkeypatch.setattr(torch.linalg, "eigh", fail)
    held, inputs = _held_spectrum_error(RuntimeError)
    gc.collect()
    assert calls == 2 and held.__traceback__ is not None
    assert all(owner() is None for owner in owners + inputs)


def test_held_next_chunk_admission_error_releases_partial_blocks(monkeypatch):
    owners = _track_spectrum_allocations(monkeypatch)
    calls = 0
    def guard(required_bytes, device, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:  # diagonal, first family, second family
            raise MemoryError("next family workspace unavailable")
    monkeypatch.setattr(structure, "guard_cuda_workspace", guard)
    held, inputs = _held_spectrum_error(MemoryError)
    gc.collect()
    assert calls == 3 and held.__traceback__ is not None
    assert all(owner() is None for owner in owners + inputs)


def test_held_initial_admission_error_releases_input_arrays(monkeypatch):
    def guard(required_bytes, device, **kwargs):
        raise MemoryError("initial spectrum workspace unavailable")
    monkeypatch.setattr(structure, "guard_cuda_workspace", guard)
    held, inputs = _held_spectrum_error(MemoryError)
    gc.collect()
    assert held.__traceback__ is not None
    assert all(owner() is None for owner in inputs)


def test_successful_spectrum_keeps_returned_owners_and_original_geometry():
    diagonal = np.ones(4)
    rows, cols = np.array([0, 2]), np.array([1, 3])
    spectrum = KinshipSpectrum.from_sparse(diagonal, rows, cols, np.full(2, .12))
    assert len(spectrum.blocks) == 2
    torch.testing.assert_close(spectrum.eigenvalues,
        torch.tensor([.88, 1.12, .88, 1.12], dtype=torch.float64), atol=1e-14, rtol=1e-14)
    values = torch.arange(8, dtype=torch.float64).reshape(4, 2)
    torch.testing.assert_close(spectrum.rotate(spectrum.rotate(values), inverse=True),
                               values, atol=1e-14, rtol=1e-14)


def _held_precision_error(error_type):
    kinship = binary._BlockKinship(np.ones(4), np.array([0, 2]),
        np.array([1, 3]), np.full(2, .12), device="cpu", max_block_size=2)
    weights = torch.full((4,), .2, dtype=torch.float64)
    tau = torch.tensor(.4, dtype=torch.float64)
    owners = [weakref.ref(kinship), weakref.ref(kinship.diagonal),
              weakref.ref(weights), weakref.ref(tau)]
    owners.extend(weakref.ref(matrix) for _, _, matrix in kinship._batches)
    precision = object.__new__(binary._BlockPrecision)
    try:
        precision.__init__(kinship, weights, tau, compute_logdet=True)
    except error_type as error:
        return error, owners, precision
    finally:
        kinship = weights = tau = None
    pytest.fail("injected precision failure did not propagate")


@pytest.mark.parametrize("initial_guard", [True, False])
def test_failed_precision_drops_borrowed_geometry_without_mutating_it(monkeypatch, initial_guard):
    if initial_guard:
        original_guard = structure.guard_cuda_workspace
        def guard(required_bytes, device, *, context="null-fit structure"):
            if context == "binary precision diagonal":
                raise MemoryError("initial precision workspace unavailable")
            return original_guard(required_bytes, device, context=context)
        monkeypatch.setattr(structure, "guard_cuda_workspace", guard)
        error_type = MemoryError
    else:
        def fail(matrix, *args, **kwargs):
            matrix = None
            raise RuntimeError("precision factor failed")
        monkeypatch.setattr(torch.linalg, "cholesky", fail)
        error_type = RuntimeError
    held, owners, precision = _held_precision_error(error_type)
    gc.collect()
    assert held.__traceback__ is not None
    assert precision.blocks == precision._batches == []
    assert precision.diagonal is precision.logdet is precision.kinship is None
    assert all(owner() is None for owner in owners)
