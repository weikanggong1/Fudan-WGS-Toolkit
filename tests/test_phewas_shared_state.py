"""Lossless transport contracts; tiny fixtures are never performance evidence."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.cache_runtime import fast_container, sparse_codec_fast, store
from fudan_wgs_toolkit.phewas_runtime.shared_state import RawStateBlock, SharedStateBroker
from fudan_wgs_toolkit.phewas_runtime import shared_state


class MetadataReader:
    def __init__(self, samples, variants):
        self.n_samples, self.n_variants, self.closed = samples, variants, False
        self.reader_metadata = {'fixture': True}

    def sample_ids(self):
        return np.asarray([f'sample{i}' for i in range(self.n_samples)])

    def read_field(self, name, indices=None):
        values = np.arange(self.n_variants)
        return values if indices is None else values[indices]

    def close(self):
        self.closed = True


def make_cache(path, states, physical_samples=None):
    m, n = states.shape
    samples = np.arange(n, dtype=np.int64) if physical_samples is None else np.asarray(physical_samples, dtype=np.int64)
    binding = {'fixture': 'independent-phenotype-six-state'}
    writer = store.Writer(path, binding, samples, m, source_bytes=100 * 2**20)
    for start in range(0, m, store.CHUNK):
        part = states[start:start+store.CHUNK]
        counts = sparse_codec_fast.integer_counts(*sparse_codec_fast.compact(part), *part.shape)
        writer.append(part, counts)
    writer.finish()
    return fast_container.Container(path, binding, samples)


def fixture_states(m=13, n=11):
    states = ((np.arange(m)[:, None] * 3 + np.arange(n)[None, :]) % 6).astype(np.uint8)
    states[0] = 0
    states[1] = 3
    states[2] = 2
    states[3] = 0
    states[3, :3] = [1, 4, 5]
    states[4] = 2
    states[4, :3] = [1, 4, 5]
    return states


def oracle(raw, physical_axis, selected_samples, columns, minimum_mac=None):
    row_lookup = {int(value): i for i, value in enumerate(physical_axis)}
    rows = np.asarray([row_lookup[int(value)] for value in selected_samples], dtype=np.int64)
    s = raw[np.ix_(columns, rows)]
    n = len(rows)
    reference = np.asarray([2, 1, 0, 0, 1, 0], dtype=np.int64)[s].sum(1)
    called = np.asarray([2, 2, 2, 0, 1, 1], dtype=np.int64)[s].sum(1)
    r = reference.astype(np.float64)
    r[called == 0] = np.nan
    af = np.divide(r, called, out=np.full_like(r, np.nan), where=called > 0)
    missing_rate = (2*n-called)/(2*n) if n else np.full_like(r, np.nan)
    alt_ac = 2*np.rint(n*(1-missing_rate))-r
    initial_mac = np.where(r >= alt_ac, alt_ac, r)
    keep = np.arange(len(columns)) if minimum_mac is None else np.flatnonzero(initial_mac >= minimum_mac)
    flip = af >= .5
    dosage = np.where(s < 3, np.where(flip[:, None], s, 2-s), 3).T.astype(np.uint8)
    whole_missing = (s >= 3).sum(1)
    observed_mac = np.where(dosage < 3, dosage, 0).sum(0).astype(np.float64)
    return (columns[keep], dosage[:, keep], tuple(a[keep] for a in (
        af, initial_mac, missing_rate, r, called)), observed_mac[keep], whole_missing[keep])


def assert_block(block, expected, samples):
    variants, dosage, summaries, observed, missing = expected
    np.testing.assert_array_equal(block.variant_indices, variants)
    np.testing.assert_array_equal(block.sample_indices, samples)
    np.testing.assert_array_equal(block.dosage.cpu().numpy(), dosage)
    for name, values in zip(('union_ref_af', 'union_initial_mac', 'union_missing_rate',
                             'union_ref_ac', 'union_called_alleles'), summaries):
        np.testing.assert_array_equal(getattr(block, name), values)
    np.testing.assert_array_equal(block.observed_mac(np.arange(len(samples))), observed)
    for imputation in ('mean', 'minor'):
        dense, maf, mac, absent, is_alt = block.trait_dense(np.arange(len(samples)),
            imputation, frequency_mode='reference', dtype=torch.float32)
        fill = 2*np.where(summaries[0] >= 1-summaries[0], 1-summaries[0], summaries[0])
        wanted = dosage.astype(np.float32)
        row, col = np.nonzero(wanted == 3)
        wanted[row, col] = fill[col].astype(np.float32) if imputation == 'mean' else 0
        np.testing.assert_array_equal(dense.cpu().numpy(), wanted)
        np.testing.assert_array_equal(mac, observed)
        np.testing.assert_array_equal(absent, missing)
        np.testing.assert_array_equal(is_alt, summaries[0] >= .5)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable for lossless kernel contracts'))])
def test_each_cohort_has_exact_half_call_summaries_orientation_and_order(tmp_path, device):
    raw = fixture_states()
    physical = np.asarray([10, 2, 9, 0, 5, 1, 8, 3, 7, 6, 4], dtype=np.int64)
    container = make_cache(tmp_path/'cache', raw, physical)
    reader = MetadataReader(14, len(raw))
    with SharedStateBroker(reader, container, device=device) as broker:
        columns = np.asarray([11, 4, 1, 7, 3, 8, 0, 12, 2])
        source = broker.read_states(columns)
        np.testing.assert_array_equal(source.states.cpu().numpy(), raw[columns].T)
        for samples in (physical, physical[::-1].copy(), np.asarray([10, 2, 9, 5, 1, 4]),
                        np.asarray([9, 0, 5, 8, 3])):
            for threshold in (None, 0, 1, 2, 4):
                block = broker.trait_block(source, samples, minimum_mac=threshold)
                assert_block(block, oracle(raw, physical, samples, columns, threshold), samples)
        assert broker.metrics['frame_reads'] == 1
        assert broker.metrics['device_csr_uploads'] == 1


@pytest.mark.parametrize('threshold', [0, 1, 2, 5, 10])
def test_safe_physical_mac_bound_never_removes_a_cohort_site(tmp_path, threshold):
    raw = fixture_states(m=19, n=23)
    physical = np.arange(raw.shape[1])
    with SharedStateBroker(MetadataReader(len(physical), len(raw)),
            make_cache(tmp_path/'cache', raw), device='cpu') as broker:
        columns = np.arange(len(raw))
        source = broker.read_states(columns, minimum_mac_bound=threshold)
        for samples in (physical, physical[::2], physical[::-3], np.asarray([0, 1, 2])):
            actual = broker.trait_block(source, samples, minimum_mac=threshold)
            assert_block(actual, oracle(raw, physical, samples, columns, threshold), samples)


def test_source_frame_reads_and_device_payload_are_shared_without_cpu_lru(tmp_path):
    raw = fixture_states()
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)), make_cache(tmp_path/'cache', raw),
            device='cpu', compact_cache_bytes=0) as broker:
        first, second = broker.reader_view(np.asarray([10, 1, 8, 2])), broker.reader_view(np.asarray([4, 0, 7]))
        columns = np.asarray([8, 3, 4, 1])
        assert_block(first.minor_block(columns, first._axis.samples),
            oracle(raw, np.arange(raw.shape[1]), first._axis.samples, columns), first._axis.samples)
        assert_block(second.minor_block(columns, second._axis.samples),
            oracle(raw, np.arange(raw.shape[1]), second._axis.samples, columns), second._axis.samples)
        assert broker.metrics['frame_reads'] == 1
        assert broker.metrics['device_csr_uploads'] == 1
        assert broker.metrics['device_csr_hits'] == 1
        assert broker.metrics['h2d_bytes'] == 0
        first.close()
        second.minor_block(columns, second._axis.samples)
        with pytest.raises(RuntimeError, match='closed'):
            first.minor_block(columns, first._axis.samples)


def test_authenticated_axis_aliases_identity_mapping_and_copies_mutable_sources(tmp_path):
    from fudan_wgs_toolkit.phewas_runtime.shared_state import _immutable
    raw = fixture_states()
    selected = _immutable([10, 2, 7, 0])
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)), make_cache(tmp_path/'cache', raw),
            device='cpu') as broker:
        view = broker.reader_view(selected)
        assert view._axis.samples is selected and view._axis.cache_rows is selected
        with pytest.raises(ValueError):
            view._axis.samples.flags.writeable = True
        ordinary = np.asarray([3, 2, 1], dtype=np.int64)
        ordinary.flags.writeable = False
        copied = broker.reader_view(ordinary)
        assert copied._axis.samples is not ordinary
        ordinary.flags.writeable = True
        ordinary[:] = [0, 5, 6]
        np.testing.assert_array_equal(copied._axis.samples, [3, 2, 1])
        assert broker.metrics['sample_axis_borrowed_immutable_axes'] == 1
        view.close()
        copied.close()


def test_host_axis_lru_preserves_active_binding_and_removes_all_aliases(tmp_path):
    from fudan_wgs_toolkit.phewas_runtime.shared_state import _immutable
    raw = fixture_states()
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)), make_cache(tmp_path/'cache', raw),
            device='cpu', sample_axis_host_cache_bytes=3*8) as broker:
        first = broker.reader_view(_immutable([10, 2, 7]))
        pinned_axis = first._axis
        binding = first._single_axis_binding()
        same = _immutable([10, 2, 7])
        matched = broker.reader_view(same)
        assert matched._axis is pinned_axis
        matched.close()
        for rows in ([1, 2, 3], [4, 5, 6], [8, 9, 0]):
            other = broker.reader_view(_immutable(rows))
            assert broker._axes_by_object[id(pinned_axis.samples)] is pinned_axis
            other.close()
        assert binding.validate(first, pinned_axis.samples, pinned_axis.identity_rows, 3) is pinned_axis
        assert broker.metrics['current_sample_axis_host_bytes'] == 3*8
        assert id(same) in broker._axis_aliases
        first.close()
        first.close()  # An idempotent close must not drop another view's pin.
        replacement = broker.reader_view(_immutable([1, 2, 3]))
        assert id(pinned_axis.samples) not in broker._axes_by_object
        assert pinned_axis.digest not in broker._axes_by_digest
        assert id(same) not in broker._axis_aliases
        assert not pinned_axis.aliases
        assert id(pinned_axis) not in broker._device_axes
        replacement.close()
        assert broker.metrics['current_sample_axis_host_bytes'] <= 3*8
        assert broker.metrics['sample_axis_host_evictions'] >= 4


def test_axis_weak_aliases_do_not_retain_equal_external_arrays(tmp_path):
    import gc
    import weakref
    from fudan_wgs_toolkit.phewas_runtime.shared_state import _immutable
    raw = fixture_states()
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)), make_cache(tmp_path/'cache', raw),
            device='cpu') as broker:
        view = broker.reader_view(_immutable([10, 2, 7]))
        references = []
        for _ in range(300):
            equal = _immutable([10, 2, 7])
            references.append(weakref.ref(equal))
            matched = broker.reader_view(equal)
            matched.close()
        del equal, matched
        gc.collect()
        assert all(reference() is None for reference in references)
        assert broker.metrics['current_sample_axis_weak_alias_count'] == 0
        view.close()


def test_retired_axis_uses_weak_immutable_proof_without_repeated_population_scan(tmp_path, monkeypatch):
    from fudan_wgs_toolkit.phewas_runtime.shared_state import _immutable
    raw = fixture_states()
    rows = [_immutable([10, 2, 7]), _immutable([1, 3, 5]), _immutable([6, 8, 0])]
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)), make_cache(tmp_path/'cache', raw),
            device='cpu', sample_axis_host_cache_bytes=3*8) as broker:
        for axis in rows:
            broker.reader_view(axis).close()
        assert broker.metrics['sample_axis_digest_calls'] == 3
        assert broker.metrics['sample_axis_validation_calls'] == 3
        def forbidden(*args, **kwargs):
            raise AssertionError('unchanged immutable source must not repeat its population-sized scan')
        monkeypatch.setattr(np, 'unique', forbidden)
        monkeypatch.setattr(np, 'searchsorted', forbidden)
        for axis in rows:
            view = broker.reader_view(axis)
            assert view._axis.samples is axis and view._axis.cache_rows is axis
            view.close()
        assert broker.metrics['sample_axis_digest_calls'] == 3
        assert broker.metrics['sample_axis_validation_calls'] == 3
        assert broker.metrics['sample_axis_verified_rebinds'] == 3
        assert broker.metrics['current_sample_axis_host_bytes'] <= 3*8


def test_host_axis_lru_pins_batched_cohorts_until_materialization_finishes(tmp_path):
    raw = fixture_states()
    physical = np.asarray([10, 2, 9, 0, 5, 1, 8, 3, 7, 6, 4], dtype=np.int64)
    columns = np.asarray([3, 4, 8, 2, 12])
    axes = [np.asarray([10, 2, 7]), np.asarray([9, 0, 5]), np.asarray([4, 1, 3])]
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)),
            make_cache(tmp_path/'cache', raw, physical), device='cpu',
            sample_axis_host_cache_bytes=3*8*2) as broker:
        blocks = broker.trait_blocks(broker.read_states(columns), axes)
        for block, rows in zip(blocks, axes):
            assert_block(block, oracle(raw, physical, rows, columns), rows)
        assert broker.metrics['current_sample_axis_host_bytes'] <= 3*8*2
        assert broker.metrics['current_sample_axis_pinned_count'] == 0


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA raw reuse contract requires CUDA'))])
def test_job_raw_slab_read_decode_upload_once_across_distinct_cohorts(tmp_path, device):
    raw = fixture_states()
    columns = np.asarray([3, 4, 8, 2, 12])
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)), make_cache(tmp_path/'cache', raw),
            device=device, device_cache_bytes=0, compact_cache_bytes=0) as broker:
        with broker.raw_state_cache(2**20):
            source = broker.read_states(columns)
            first = broker.trait_block(source, np.asarray([10, 2, 7]))
            same = broker.read_states(columns.copy())
            second = broker.trait_block(same, np.asarray([9, 0, 5]))
            assert same is source
            for block, rows in ((first, np.asarray([10, 2, 7])), (second, np.asarray([9, 0, 5]))):
                assert_block(block, oracle(raw, np.arange(raw.shape[1]), rows, columns), rows)
            assert broker.metrics['frame_reads'] == 1
            assert broker.metrics['device_csr_uploads'] == 1
            assert broker.metrics['raw_state_materializations'] == 1
            assert broker.metrics['raw_state_cache_hits'] == 1
            assert broker.metrics['current_raw_state_cache_slabs'] == 1
        assert broker.metrics['current_raw_state_cache_bytes'] == 0
        assert broker.metrics['raw_state_cache_limit_bytes'] == 0
        # Scope retirement only drops the cached reference, not active readers.
        assert_block(broker.trait_block(source, np.asarray([10, 2, 7])),
            oracle(raw, np.arange(raw.shape[1]), np.asarray([10, 2, 7]), columns), np.asarray([10, 2, 7]))
        broker.read_states(columns)
        assert broker.metrics['frame_reads'] == 2


def test_raw_slab_cache_budget_release_order_and_mac_binding_are_exact(tmp_path):
    raw = fixture_states()
    columns = np.asarray([3, 4, 8, 2, 12])
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)), make_cache(tmp_path/'cache', raw),
            device='cpu', device_cache_bytes=0, compact_cache_bytes=0) as broker:
        size = raw.shape[1]*len(columns)+16*len(columns)
        with broker.raw_state_cache(size):
            unfiltered = broker.read_states(columns)
            reverse = broker.read_states(columns[::-1])
            np.testing.assert_array_equal(reverse.variant_indices, columns[::-1])
            assert reverse is not unfiltered
            assert broker.metrics['current_raw_state_cache_bytes'] <= size
            filtered = broker.read_states(columns, minimum_mac_bound=10)
            assert filtered is not unfiltered
            assert broker.metrics['current_raw_state_cache_bytes'] <= size
            released = broker.release_device_cache()
            assert released > 0 and not broker._raw_sources
            np.testing.assert_array_equal(unfiltered.states.numpy(), raw[columns].T)
        with broker.raw_state_cache(1):
            broker.read_states(columns)
            assert broker.metrics['current_raw_state_cache_bytes'] == 0


def test_adaptive_cache_caps_use_live_available_and_leave_reserve(tmp_path, monkeypatch):
    """Mock only resource counters; no hardware/throughput claim is made."""
    raw = fixture_states()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (500, 2000))
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 200)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 300)
    monkeypatch.setattr(shared_state, 'allocator_bytes', lambda device: (200, 300))
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)), make_cache(tmp_path/'cache', raw),
            device='cuda:0', memory_limit_gib=1000/2**30, memory_reserve_bytes=100,
            device_cache_bytes=1000, device_cache_live_fraction=.25) as broker:
        # min(free + reusable, process cap - allocated) - reserve = 500.
        assert broker.metrics['gpu_csr_cache_limit_bytes'] == 125
        assert broker.metrics['gpu_csr_cache_configured_limit_bytes'] == 1000
        with broker.raw_state_cache(1000, live_fraction=.5):
            assert broker.metrics['raw_state_cache_limit_bytes'] == 250
        monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (0, 2000))
        monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 200)
        monkeypatch.setattr(shared_state, 'allocator_bytes', lambda device: (200, 200))
        with broker.raw_state_cache(1000):
            assert broker.metrics['raw_state_cache_limit_bytes'] == 0


def test_smaller_process_cap_retains_reserve_for_raw_cache_and_new_genotypes(tmp_path, monkeypatch):
    raw = fixture_states()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (1800, 2000))
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 200)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 200)
    monkeypatch.setattr(shared_state, 'allocator_bytes', lambda device: (200, 200))
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)), make_cache(tmp_path/'cache', raw),
            device='cuda:0', memory_limit_gib=1000/2**30, memory_reserve_bytes=100,
            device_cache_bytes=1000, device_cache_live_fraction=.25) as broker:
        assert broker._live_available_bytes() == 700
        assert broker.metrics['gpu_csr_cache_limit_bytes'] == 175
        with broker.raw_state_cache(1000, live_fraction=.5):
            assert broker.metrics['raw_state_cache_limit_bytes'] == 350
        broker._guard(700)
        with pytest.raises(MemoryError, match='live CUDA budget'):
            broker._guard(701)


def test_retiring_raw_cache_does_not_credit_active_external_storage(tmp_path, monkeypatch):
    """An externally held slab remains counted after cache-owned retirement."""
    raw = fixture_states()
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)), make_cache(tmp_path/'cache', raw),
            device='cpu', memory_reserve_bytes=0, device_cache_bytes=0, compact_cache_bytes=0) as broker:
        with broker.raw_state_cache(2**20):
            source = broker.read_states(np.asarray([3, 4, 8, 2, 12]))
            broker.device = torch.device('cuda:0')
            monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (100, 1000))
            monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 900)
            monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 900)
            monkeypatch.setattr(shared_state, 'allocator_bytes', lambda device: (900, 900))
            try:
                with pytest.raises(MemoryError, match='live CUDA budget'):
                    broker._guard(150)
                assert broker.metrics['current_raw_state_cache_bytes'] == 0
                np.testing.assert_array_equal(source.states.numpy(), raw[[3, 4, 8, 2, 12]].T)
            finally:
                broker.device = torch.device('cpu')


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable for cross-frame kernel contracts'))])
def test_cross_frame_unsorted_variant_axis_is_never_sorted(tmp_path, device):
    raw = fixture_states(m=2051, n=11)
    columns = np.asarray([2049, 1023, 15, 2048, 1024, 2050, 0, 1025])
    samples = np.asarray([10, 4, 1, 0])
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)), make_cache(tmp_path/'cache', raw),
            device=device) as broker:
        source = broker.read_states(columns)
        np.testing.assert_array_equal(source.variant_indices, columns)
        np.testing.assert_array_equal(source.states.cpu().numpy(), raw[columns].T)
        assert_block(broker.trait_block(source, samples),
            oracle(raw, np.arange(raw.shape[1]), samples, columns), samples)
        assert broker.metrics['frame_reads'] == 3


def test_effective_packing_preserves_standalone_axes_and_native_summaries(tmp_path):
    raw = fixture_states(m=2051, n=11)
    samples = np.asarray([7, 2, 9, 0, 5])
    columns = np.asarray([2049, 3, 1024, 7, 2048, 4, 2050, 0, 1025, 15, 1])
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)), make_cache(tmp_path/'cache', raw),
            device='cpu') as broker:
        view = broker.reader_view(samples)
        blocks = list(view.iter_effective_minor_blocks(columns, samples, block_size=3,
            effective_block_size=2, minimum_mac=1, device='cpu', resident=True))
        wanted = oracle(raw, np.arange(raw.shape[1]), samples, columns, 1)
        np.testing.assert_array_equal(np.concatenate([x.variant_indices for x in blocks]), wanted[0])
        np.testing.assert_array_equal(torch.cat([x.dosage for x in blocks], 1).numpy(), wanted[1])
        assert all(x.shape[1] <= 2 for x in blocks)
        assert view.reader_metadata['analysis_cache']['single_effective_blocks'] == len(blocks)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable for uint32 sample kernel contracts'))])
def test_uint32_physical_sample_indices_keep_high_samples(tmp_path, device):
    n = 65537
    raw = np.zeros((6, n), dtype=np.uint8)
    raw[2] = 2
    raw[:, [-1, 2, n//2]] = np.asarray([[0, 1, 4], [3, 5, 1], [2, 4, 0],
                                        [4, 5, 3], [5, 2, 1], [1, 2, 5]])
    columns, samples = np.asarray([5, 2, 0, 4]), np.asarray([65536, 2, n//2, 0])
    with SharedStateBroker(MetadataReader(n, len(raw)), make_cache(tmp_path/'cache', raw), device=device) as broker:
        source = broker.read_states(columns)
        assert_block(broker.trait_block(source, samples), oracle(raw, np.arange(n), samples, columns), samples)


def test_raw_state_mutation_foreign_broker_and_invalid_axis_are_rejected(tmp_path):
    raw = fixture_states()
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)), make_cache(tmp_path/'cache', raw), device='cpu') as broker:
        source = broker.read_states(np.asarray([3, 4, 5]))
        forged = RawStateBlock(source.states, source.sample_indices, source.variant_indices)
        with pytest.raises(ValueError, match='unaltered'):
            broker.trait_block(forged, np.asarray([1, 2]))
        source.states[0, 0] = 5
        with pytest.raises(ValueError, match='unaltered'):
            broker.trait_block(source, np.asarray([1, 2]))
        with pytest.raises((ValueError, IndexError), match='duplicate|dimension'):
            broker.reader_view(np.asarray([1, 1]))
        view = broker.reader_view(np.asarray([1, 2]))
        with pytest.raises(ValueError, match='sample order'):
            view.minor_block(np.asarray([3]), np.asarray([2, 1]))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA sample-axis reuse contract requires CUDA'))])
def test_equal_ordered_sample_copies_share_gpu_axis_but_reverse_order_does_not(tmp_path, device):
    raw = fixture_states(m=17, n=23)
    selected = np.asarray([22, 3, 12, 0, 7, 4], dtype=np.int64)
    columns = np.asarray([3, 4, 8, 2, 12])
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)), make_cache(tmp_path/'cache', raw),
                          device=device) as broker:
        view = broker.reader_view(selected)
        original_axis, row_uploads = view._axis, broker.metrics['index_h2d_bytes']
        for copied in (selected.copy(), selected.astype(np.int32), selected.astype(np.uint32)):
            matched = broker.reader_view(copied)
            assert matched._axis is original_axis
            assert matched._axis.device_rows is original_axis.device_rows
            assert broker.metrics['index_h2d_bytes'] == row_uploads
            assert broker.metrics['bound_sample_axes'] == 1
            assert broker.metrics['sample_axis_validation_calls'] == 1
        reversed_view = broker.reader_view(selected[::-1].copy())
        assert reversed_view._axis is not original_axis
        assert broker.metrics['bound_sample_axes'] == 2
        # Sample row vectors are now admitted lazily into a bounded LRU, so
        # 1000 different complete-case cohorts never stay resident on CUDA.
        np.testing.assert_array_equal(broker._device_rows(reversed_view._axis).cpu().numpy(), selected[::-1])
        source = broker.read_states(columns)
        for axis in (view._axis, reversed_view._axis):
            actual = broker.trait_block(source, axis.samples.copy())
            assert_block(actual, oracle(raw, np.arange(raw.shape[1]), axis.samples, columns), axis.samples)
        assert broker.metrics['bound_sample_axes'] == 2


@pytest.mark.parametrize('invalid', [np.asarray([3., 1., 2.]), np.asarray([3, 1, 2], dtype=object),
    np.asarray(['3', '1', '2']), np.asarray([True, False, True]), np.asarray([[3, 1, 2]]),
    np.asarray([2**64-1, 1, 2], dtype=np.uint64)])
def test_equal_shape_noninteger_or_unrepresentable_axis_cannot_reuse_a_binding(tmp_path, invalid):
    raw = fixture_states()
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)), make_cache(tmp_path/'cache', raw),
                          device='cpu') as broker:
        broker.reader_view(np.asarray([3, 1, 2], dtype=np.int64))
        before = broker.metrics
        with pytest.raises((ValueError, IndexError), match='integer'):
            broker.reader_view(invalid)
        assert broker.metrics['bound_sample_axes'] == before['bound_sample_axes']
        assert broker.metrics['index_h2d_bytes'] == before['index_h2d_bytes']


def test_unverified_container_and_changed_frame_are_rejected(tmp_path):
    reader = MetadataReader(11, 13)
    with pytest.raises(ValueError, match='verified'):
        SharedStateBroker(reader, SimpleNamespace(complete=True), device='cpu')
    raw = fixture_states()
    container = make_cache(tmp_path/'cache', raw)
    broker = SharedStateBroker(reader, container, device='cpu')
    path = Path(container.path)/'data.bin'
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match='SHA'):
        broker.read_states(np.asarray([3]))
    with pytest.raises(ValueError, match='changed'):
        broker.close()


def test_budget_and_lifecycle_fail_without_sdk_fallback(tmp_path):
    raw = fixture_states()
    reader = MetadataReader(raw.shape[1], len(raw))
    broker = SharedStateBroker(reader, make_cache(tmp_path/'cache', raw), device='cpu',
                              memory_limit_gib=.001, own_reader=True)
    with pytest.raises(MemoryError, match='budget'):
        broker.read_states(np.asarray([3]))
    broker.close()
    assert reader.closed
    with pytest.raises(RuntimeError, match='closed'):
        broker.read_states(np.asarray([3]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA transport contract requires CUDA')
def test_cuda_csr_h2d_is_once_per_frame_across_distinct_cohorts(tmp_path):
    raw = fixture_states(m=23, n=43)
    columns = np.asarray([3, 8, 4, 16, 1, 0, 2])
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)), make_cache(tmp_path/'cache', raw),
                          device='cuda', compact_cache_bytes=0) as broker:
        assert broker.device.index is not None
        for samples in (np.arange(raw.shape[1])[::2], np.arange(raw.shape[1])[::-3]):
            view = broker.reader_view(samples)
            assert_block(view.minor_block(columns, samples, device='cuda'),
                oracle(raw, np.arange(raw.shape[1]), samples, columns), samples)
        torch.cuda.synchronize()
        arrays = sparse_codec_fast.compact(raw)
        expected = sum(a.nbytes for a in arrays)
        assert broker.metrics['gpu_csr_uploads'] == 1
        assert broker.metrics['csr_h2d_bytes'] == expected
        assert broker.metrics['gpu_csr_hits'] == 1
        assert broker.metrics['frame_reads'] == 1


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA summary batch contract requires CUDA'))])
@pytest.mark.parametrize('minimum_mac', [None, 0, 2, 6])
def test_batched_integer_summaries_equal_independent_trait_blocks(tmp_path, device, minimum_mac):
    raw = fixture_states(m=21, n=17)
    columns = np.asarray([4, 3, 12, 1, 0, 2, 9, 18, 7])
    physical = np.asarray([16, 2, 9, 0, 5, 1, 8, 3, 7, 6, 4, 15, 13, 14, 11, 10, 12])
    axes = [physical.copy(), physical[::-2].copy(), physical[[3, 6, 10, 1, 8]],
            np.asarray([16, 2, 9]), np.empty(0, dtype=np.int64)]
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)),
                          make_cache(tmp_path/'cache', raw, physical), device=device) as broker:
        source = broker.read_states(columns)
        separate = [broker.trait_block(source, rows, minimum_mac) for rows in axes]
        before = broker.metrics
        together = broker.trait_blocks(source, [rows.copy() for rows in axes], minimum_mac)
        after = broker.metrics
        assert len(together) == len(axes)
        for rows, independent, batched in zip(axes, separate, together):
            assert_block(batched, oracle(raw, physical, rows, columns, minimum_mac), rows)
            np.testing.assert_array_equal(batched.dosage.cpu().numpy(), independent.dosage.cpu().numpy())
            for name in ('variant_indices', 'sample_indices', 'union_ref_af', 'union_initial_mac',
                         'union_missing_rate', 'union_ref_ac', 'union_called_alleles'):
                np.testing.assert_array_equal(getattr(batched, name), getattr(independent, name))
        expected_transfers = int(device.startswith('cuda'))
        assert after['trait_summary_d2h_calls']-before['trait_summary_d2h_calls'] == expected_transfers
        assert after['trait_summary_d2h_bytes']-before['trait_summary_d2h_bytes'] == expected_transfers*len(axes)*4*len(columns)*8
        assert after['trait_pending_summary_bytes_highwater'] == len(axes)*4*len(columns)*8
        assert after['bound_sample_axes'] == before['bound_sample_axes']


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA empty-summary batch contract requires CUDA'))])
def test_batched_summaries_empty_variants_and_no_traits_transfer_nothing(tmp_path, device):
    raw = fixture_states(m=13, n=11)
    axes = [np.arange(raw.shape[1]), np.arange(raw.shape[1])[::-2], np.empty(0, dtype=np.int64)]
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)), make_cache(tmp_path/'cache', raw),
                          device=device) as broker:
        source = broker.read_states(np.empty(0, dtype=np.int64))
        together = broker.trait_blocks(source, axes, minimum_mac=2)
        assert len(together) == len(axes)
        for rows, block in zip(axes, together):
            assert block.shape == (len(rows), 0)
            assert block.union_called_alleles.dtype == np.int64
        assert broker.trait_blocks(source, [], minimum_mac=2) == []
        assert broker.metrics['trait_summary_d2h_calls'] == 0
        assert broker.metrics['trait_summary_d2h_bytes'] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA summary transport contract requires CUDA')
def test_cuda_sixteen_traits_use_one_summary_d2h_with_the_same_integer_kernels(tmp_path):
    raw = fixture_states(m=19, n=43)
    columns = np.asarray([3, 4, 2, 12, 7, 14, 0, 1])
    all_rows = np.arange(raw.shape[1])
    axes = [np.roll(all_rows, trait)[::1+trait%4].copy() for trait in range(16)]
    with SharedStateBroker(MetadataReader(raw.shape[1], len(raw)), make_cache(tmp_path/'cache', raw),
                          device='cuda:0') as broker:
        source = broker.read_states(columns)
        separate = [broker.trait_block(source, rows, minimum_mac=2) for rows in axes]
        before = broker.metrics
        assert before['trait_count_calls'] == 16
        assert before['trait_summary_d2h_calls'] == 16
        batch = broker.trait_blocks(source, axes, minimum_mac=2)
        torch.cuda.synchronize()
        after = broker.metrics
        assert after['trait_count_calls']-before['trait_count_calls'] == 16
        assert after['trait_summary_d2h_calls']-before['trait_summary_d2h_calls'] == 1
        assert after['trait_summary_d2h_bytes']-before['trait_summary_d2h_bytes'] == 16*4*len(columns)*8
        assert after['gpu_csr_uploads'] == 1
        for rows, standalone, together in zip(axes, separate, batch):
            assert_block(together, oracle(raw, all_rows, rows, columns, 2), rows)
            np.testing.assert_array_equal(together.dosage.cpu().numpy(), standalone.dosage.cpu().numpy())
            np.testing.assert_array_equal(together.union_ref_af, standalone.union_ref_af)
