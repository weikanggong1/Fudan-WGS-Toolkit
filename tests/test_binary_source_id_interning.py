"""Immutable binary source ID ownership; these are not WGS benchmarks."""
import gc
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from fudan_wgs_toolkit import binary_null
from test_binary_phewas_burden import fixture, immutable_source_arguments


@pytest.fixture
def admitted_inputs(monkeypatch):
    pool = weakref.WeakValueDictionary()
    monkeypatch.setattr(binary_null, "_IMMUTABLE_BINARY_ID_AXES", pool)
    prepared, *_ = fixture()
    return immutable_source_arguments(prepared.spa_model), pool


def make_source(arguments, sample_ids=None):
    options = dict(arguments)
    if sample_ids is not None:
        options["sample_ids"] = sample_ids
    return binary_null.ImmutableFittedBinarySource(**options)


def test_equal_axes_share_truly_immutable_bytes_across_sources(admitted_inputs):
    arguments, pool = admitted_inputs
    first = make_source(arguments)
    second = make_source(arguments, arguments["sample_ids"].copy())
    assert first.sample_ids is second.sample_ids and len(pool) == 1
    assert isinstance(first.sample_ids.base, bytes)
    assert not first.sample_ids.flags.writeable
    np.testing.assert_array_equal(first.sample_ids, arguments["sample_ids"])
    with pytest.raises(ValueError):
        first.sample_ids.setflags(write=True)
    with pytest.raises(ValueError):
        second.sample_ids[0] = "changed"
    # Derivations preserve the same axis while keeping their model state separate.
    first_normal, second_normal = first.normal(), second.normal()
    assert first_normal.sample_ids is second_normal.sample_ids is first.sample_ids
    assert first_normal is not second_normal


def test_original_mutable_input_cannot_change_shared_axis(admitted_inputs):
    arguments, pool = admitted_inputs
    original = arguments["sample_ids"].copy()
    expected = original.copy()
    first = make_source(arguments, original)
    second = make_source(arguments, expected)
    original[0] = "xx"
    np.testing.assert_array_equal(first.sample_ids, expected)
    assert second.sample_ids is first.sample_ids
    changed = make_source(arguments, original)
    assert changed.sample_ids is not first.sample_ids and len(pool) == 2
    np.testing.assert_array_equal(changed.sample_ids, original)


@pytest.mark.parametrize("change", ["reverse", "rotate", "replace", "wider_dtype"])
def test_different_order_values_or_dtype_do_not_share(admitted_inputs, change):
    arguments, pool = admitted_inputs
    original = arguments["sample_ids"].copy()
    altered = {"reverse": lambda: original[::-1],
               "rotate": lambda: np.roll(original, 1),
               "replace": lambda: np.asarray(["xx", *original[1:]]),
               "wider_dtype": lambda: original.astype("U8")}[change]()
    first, second = make_source(arguments, original), make_source(arguments, altered)
    assert first.sample_ids is not second.sample_ids and len(pool) == 2
    np.testing.assert_array_equal(first.sample_ids, original)
    np.testing.assert_array_equal(second.sample_ids, altered)


def test_each_source_still_validates_ids_before_pool_lookup(admitted_inputs, monkeypatch):
    arguments, pool = admitted_inputs
    first = make_source(arguments)
    native_unique, scanned = np.unique, []

    def checked_unique(values):
        scanned.append(np.asarray(values).copy())
        return native_unique(values)

    monkeypatch.setattr(np, "unique", checked_unique)
    second = make_source(arguments)
    assert len(scanned) == 1 and second.sample_ids is first.sample_ids
    original = arguments["sample_ids"].copy()
    duplicate, empty = original.copy(), original.copy()
    duplicate[1], empty[1] = duplicate[0], ""
    for invalid in (original[:-1], original.reshape(8, 8), duplicate, empty):
        with pytest.raises(ValueError, match="unique and aligned"):
            make_source(arguments, invalid)
    assert len(pool) == 1


def test_hash_collision_never_merges_different_ordered_axes(admitted_inputs, monkeypatch):
    arguments, pool = admitted_inputs
    monkeypatch.setattr(binary_null, "hashlib", SimpleNamespace(
        sha256=lambda data: SimpleNamespace(digest=lambda: b"same-hash")))
    first = make_source(arguments)
    reversed_ids = arguments["sample_ids"][::-1]
    second = make_source(arguments, reversed_ids)
    assert first.sample_ids is not second.sample_ids
    np.testing.assert_array_equal(first.sample_ids, arguments["sample_ids"])
    np.testing.assert_array_equal(second.sample_ids, reversed_ids)
    assert len(pool) == 1  # At most the last weak value for this forced collision.


def test_pool_does_not_retain_axis_after_last_source_or_model(admitted_inputs):
    arguments, pool = admitted_inputs
    first, second = make_source(arguments), make_source(arguments)
    derived = first.normal()
    reference = weakref.ref(first.sample_ids)
    del first, second
    gc.collect()
    assert reference() is derived.sample_ids and len(pool) == 1
    del derived
    gc.collect()
    assert reference() is None and len(pool) == 0


def test_concurrent_interning_reuses_one_axis_without_strong_pool_ownership(admitted_inputs):
    arguments, pool = admitted_inputs
    ids = arguments["sample_ids"].copy()
    with ThreadPoolExecutor(max_workers=4) as executor:
        axes = list(executor.map(binary_null._intern_immutable_binary_sample_ids,
                                 [ids] * 16))
    assert all(axis is axes[0] for axis in axes) and len(pool) == 1
    reference = weakref.ref(axes[0])
    del axes
    gc.collect()
    assert reference() is None and len(pool) == 0
