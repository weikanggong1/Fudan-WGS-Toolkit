"""Immutable read-ahead and lifecycle contracts, not performance benchmarks."""
import threading

import numpy as np
import pytest

from fudan_wgs_toolkit.cache_runtime import fast_container, sparse_codec_fast, store


def cache(tmp_path):
    states = ((np.arange(17)[:, None] + np.arange(9)[None, :]) % 6).astype(np.uint8)
    writer = store.Writer(tmp_path/'cache', {'fixture': 'prefetch'}, np.arange(9, dtype=np.int64),
                          len(states), source_bytes=100*2**20, chunk=4)
    for start in range(0, len(states), 4):
        part = states[start:start+4]
        counts = sparse_codec_fast.integer_counts(*sparse_codec_fast.compact(part), *part.shape)
        writer.append(part, counts)
    writer.finish()
    return tmp_path/'cache'


def test_prefetched_frames_match_validated_original_in_requested_order(tmp_path):
    path = cache(tmp_path)
    original = store.Container(path)
    with fast_container.Container(path) as reader:
        reader.enable_prefetch(2**20)
        for index in (2, 0, 4, 1, 3):
            assert reader.prefetch(index)
            actual, expected = reader.read_frame(index), original.read_frame(index)
            assert actual.keys() == expected.keys()
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
        assert reader.prefetch_metrics['hits'] == 5
        assert reader.prefetch_metrics['peak_compressed_bytes'] <= 2**20


def test_oversize_frame_is_not_scheduled_and_synchronous_read_still_works(tmp_path):
    with fast_container.Container(cache(tmp_path)) as reader:
        reader.enable_prefetch(1)
        assert not reader.prefetch(0)
        assert reader.prefetch_metrics['submitted'] == 0
        assert reader.prefetch_metrics['oversize_skipped'] == 1
        assert reader.read_frame(0)['m'] == 4


def test_only_one_pending_frame_and_close_joins_before_closing_handles(tmp_path, monkeypatch):
    reader = fast_container.Container(cache(tmp_path))
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    original = reader._compressed_ranges
    def blocked(index):
        entered.set()
        if not release.wait(3):
            raise RuntimeError('test did not release read')
        return original(index)
    monkeypatch.setattr(reader, '_compressed_ranges', blocked)
    reader.enable_prefetch(2**20)
    assert reader.prefetch(0) and entered.wait(3)
    assert not reader.prefetch(1)
    closer = threading.Thread(target=lambda: (reader.close(), closed.set()))
    closer.start()
    try:
        assert not closed.wait(.03)
        assert all(not item.handle.closed for item in reader._streams.values())
    finally:
        release.set(); closer.join(3)
    assert closed.is_set() and all(item.handle.closed for item in reader._streams.values())
    with pytest.raises(ValueError, match='closed'):
        reader.prefetch(0)


def test_prefetched_corruption_is_rejected_by_unchanged_frame_checksum(tmp_path):
    path = cache(tmp_path)
    reader = fast_container.Container(path)
    try:
        reader.enable_prefetch(2**20)
        data = path/'data.bin'
        with data.open('r+b') as stream:
            old = stream.read(1);stream.seek(0);stream.write(bytes([old[0] ^ 1]))
        assert reader.prefetch(0)
        with pytest.raises(ValueError, match='SHA'):
            reader.read_frame(0)
        # Source identity validation independently detects the mutation.
        with pytest.raises(ValueError, match='changed'):
            reader.check_stream_identity()
    finally:
        with pytest.raises(ValueError, match='changed'):
            reader.close()


def test_short_prefetch_read_failure_is_not_hidden(tmp_path, monkeypatch):
    with fast_container.Container(cache(tmp_path)) as reader:
        reader.enable_prefetch(2**20)
        def failed(index):
            raise ValueError('Short prefetched frame read')
        monkeypatch.setattr(reader, '_compressed_ranges', failed)
        assert reader.prefetch(0)
        with pytest.raises(ValueError, match='Short prefetched'):
            reader.read_frame(0)


@pytest.mark.parametrize('limit', [-1, True, 1.5, 256*2**20+1])
def test_invalid_prefetch_cap_rejected_without_pool(tmp_path, limit):
    with fast_container.Container(cache(tmp_path)) as reader:
        with pytest.raises(ValueError):
            reader.enable_prefetch(limit)
        assert reader._prefetch_pool is None
