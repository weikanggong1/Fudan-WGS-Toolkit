"""Exact lossless multi-frame transport and bounded owner references."""
import gc
import weakref

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.cache_runtime import fast_container, sparse_codec_fast, store
from fudan_wgs_toolkit.phewas_runtime.shared_state import SharedStateBroker


class Reader:
    def __init__(self, n, m):
        self.n_samples, self.n_variants = n, m


def make_cache(path):
    m, n = store.CHUNK + 7, 11
    raw = ((np.arange(m)[:, None] * 3 + np.arange(n)[None, :]) % 6).astype(np.uint8)
    raw[2] = 0
    raw[store.CHUNK+1] = 0
    samples = np.arange(n, dtype=np.int64)
    binding = {'fixture': 'lossless-multiframe-transport'}
    writer = store.Writer(path, binding, samples, m, source_bytes=100*2**20)
    for start in range(0, m, store.CHUNK):
        part = raw[start:start+store.CHUNK]
        counts = sparse_codec_fast.integer_counts(*sparse_codec_fast.compact(part), *part.shape)
        writer.append(part, counts)
    writer.finish()
    return raw, fast_container.Container(path, binding, samples)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable for raw transport contract'))])
@pytest.mark.parametrize('minimum_mac', [None, 1, 4])
def test_multiframe_order_and_bound_use_one_merge_destination_without_cat(
        tmp_path, monkeypatch, device, minimum_mac):
    raw, cache = make_cache(tmp_path/'cache')
    n = raw.shape[1]
    columns = np.asarray([store.CHUNK+6, 3, store.CHUNK+1, 8, 2, store.CHUNK+3, 4])
    reference = raw[columns]
    ref = np.asarray([2, 1, 0, 0, 1, 0], dtype=np.int64)[reference].sum(1)
    called = np.asarray([2, 2, 2, 0, 1, 1], dtype=np.int64)[reference].sum(1)
    keep = np.ones(len(columns), dtype=bool) if minimum_mac is None else (
        np.minimum(ref, called-ref+1) >= minimum_mac)
    def forbidden(*args, **kwargs):
        raise AssertionError('raw slab merge must not allocate cat plus reordered slabs')
    with SharedStateBroker(Reader(n, len(raw)), cache, device=device,
            compact_cache_bytes=0, device_cache_bytes=0) as broker:
        guards = []
        original_guard = broker._guard
        def observed_guard(required):
            guards.append(required)
            return original_guard(required)
        monkeypatch.setattr(broker, '_guard', observed_guard)
        monkeypatch.setattr(torch, 'cat', forbidden)
        with broker.raw_state_cache(2**20):
            result = broker.read_states(columns, minimum_mac_bound=minimum_mac)
            np.testing.assert_array_equal(result.variant_indices, columns[keep])
            np.testing.assert_array_equal(result.states.cpu().numpy(), reference[keep].T)
            metrics = broker.metrics
            assert metrics['frame_reads'] == 2
            assert metrics['raw_slab_merge_calls'] == 1
            assert metrics['raw_slab_merge_copy_bytes'] == n*int(keep.sum())
            assert metrics['raw_slab_merge_new_bytes_highwater'] == n*int(keep.sum())
            assert guards[-1] == n*int(keep.sum())+64*2**20
            assert broker.read_states(columns.copy(), minimum_mac_bound=minimum_mac) is result
            assert broker.metrics['raw_slab_merge_calls'] == 1
            assert broker.metrics['frame_reads'] == 2
            assert broker.metrics['csr_host_copy_bytes'] > 0


def test_uncached_payload_and_per_frame_slabs_are_not_retained_by_loop_locals(tmp_path, monkeypatch):
    raw, cache = make_cache(tmp_path/'cache')
    columns = np.asarray([store.CHUNK+6, 3, 8, store.CHUNK+3, 4])
    with SharedStateBroker(Reader(raw.shape[1], len(raw)), cache, device='cpu',
            compact_cache_bytes=0, device_cache_bytes=0) as broker:
        payload_refs, slab_refs = [], []
        original_upload = broker._upload
        original_zeros = torch.zeros
        def upload(*args, **kwargs):
            result = original_upload(*args, **kwargs)
            payload_refs.extend(weakref.ref(value) for value in result)
            return result
        def zeros(*args, **kwargs):
            result = original_zeros(*args, **kwargs)
            slab_refs.append(weakref.ref(result))
            return result
        monkeypatch.setattr(broker, '_upload', upload)
        monkeypatch.setattr(torch, 'zeros', zeros)
        result = broker.read_states(columns)
        gc.collect()
        np.testing.assert_array_equal(result.states.numpy(), raw[columns].T)
        assert payload_refs and all(reference() is None for reference in payload_refs)
        assert len(slab_refs) == 2 and all(reference() is None for reference in slab_refs)
        assert broker.metrics['current_gpu_csr_cache_bytes'] == 0
