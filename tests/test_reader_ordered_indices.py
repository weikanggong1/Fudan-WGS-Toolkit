"""Index-order and bounded-workspace contracts; fixtures are not benchmarks."""
import numpy as np
import pytest

from fudan_wgs_toolkit.genotype import _indices


@pytest.mark.parametrize("values", [np.asarray([], dtype=np.int64),
    np.asarray([3], dtype=np.int32), np.asarray([0, 2, 7], dtype=np.uint16),
    np.arange(8, dtype=np.int64)[::2]])
def test_ordered_reader_axis_preserves_values_without_sort_workspace(values, monkeypatch):
    expected = values.astype(np.int64)
    monkeypatch.setattr(np, "unique", lambda *_args, **_kwargs:
        pytest.fail("strictly ordered index validation allocated a unique sort"))
    observed = _indices(values, 8, "variant_indices")
    assert observed.dtype == np.int64
    np.testing.assert_array_equal(observed, expected)
    if values.dtype == np.int64:
        assert observed is values


@pytest.mark.parametrize("values", [[7, 3, 0], [2, 0, 7, 3]])
def test_unordered_reader_axis_preserves_requested_order(values):
    observed = _indices(np.asarray(values, dtype=np.int64), 8, "variant_indices")
    np.testing.assert_array_equal(observed, values)


@pytest.mark.parametrize("values", [[0, 0, 2], [2, 0, 2], [3, 3]])
def test_ordered_and_unordered_duplicate_axes_still_fail(values):
    with pytest.raises(ValueError, match="variant_indices contains duplicate indices"):
        _indices(np.asarray(values, dtype=np.int64), 8, "variant_indices")


@pytest.mark.parametrize("values, exception, message", [
    (np.asarray([False, True]), ValueError, "one-dimensional integer"),
    (np.asarray([0., 1.]), ValueError, "one-dimensional integer"),
    (np.asarray([[0, 1]]), ValueError, "one-dimensional integer"),
    (np.asarray([-1, 0]), IndexError, "outside the genotype dimension"),
    (np.asarray([0, 8]), IndexError, "outside the genotype dimension"),
    (np.asarray([2**64-1], dtype=np.uint64), IndexError, "outside the genotype dimension"),
])
def test_ordered_fast_path_keeps_dtype_geometry_and_range_checks(values, exception, message):
    with pytest.raises(exception, match=message):
        _indices(values, 8, "variant_indices")


def test_readonly_noncontiguous_axis_keeps_original_ownership():
    values = np.arange(8, dtype=np.int64)[::2]
    values.flags.writeable = False
    observed = _indices(values, 8, "variant_indices")
    assert observed is values and not observed.flags.writeable
    assert not observed.flags.c_contiguous
