"""Frame-aligned compact Single reads with bounded eligible-column batches.

The reader retains source/cohort MAC, allele summaries and caller order. Only
CPU compact arrays are accumulated; no genotype matrix is re-cached on disk.
"""
import time
import numpy as np


def _slice_prepared(prepared, start, stop):
    """Select a contiguous eligible-column interval before dense expansion."""
    columns = prepared['exception_col']
    keep = (columns >= start) & (columns < stop)
    return dict(prepared, columns=prepared['columns'][start:stop],
                exception_col=columns[keep] - start,
                exception_row=prepared['exception_row'][keep],
                exception_state=prepared['exception_state'][keep],
                summaries=tuple(a[start:stop] for a in prepared['summaries']))


def _merge_prepared(parts, sample_count):
    """Rebind compact columns to their exact retained physical variant axis."""
    # Each part is (prepared, physical_variant_indices), in original order.
    count = sum(len(indices) for _, indices in parts)
    if len(parts) == 1:
        item, variants = parts[0]
        return dict(item, cache_variant_count=count, cache_sample_count=sample_count,
                    columns=np.arange(count, dtype=np.int64),
                    samples=np.arange(sample_count, dtype=np.int64)), variants
    columns, rows, states, variants = [], [], [], []
    offset = 0
    for prepared, indices in parts:
        columns.append(prepared['exception_col'] + offset)
        rows.append(prepared['exception_row'])
        states.append(prepared['exception_state'])
        variants.append(indices)
        offset += len(indices)
    return dict(cache_variant_count=count, cache_sample_count=sample_count,
                columns=np.arange(count, dtype=np.int64), samples=np.arange(sample_count, dtype=np.int64),
                exception_col=np.concatenate(columns), exception_row=np.concatenate(rows),
                exception_state=np.concatenate(states),
                summaries=tuple(np.concatenate([p['summaries'][i] for p, _ in parts]) for i in range(5)),
                full_union_summaries=all(p['full_union_summaries'] for p, _ in parts)), np.concatenate(variants)


def _physical_requests(adapter, variants, block_size):
    """Avoid reading adjacent compressed frames twice for sorted Single axes."""
    if len(variants) < 2 or np.all(variants[1:] > variants[:-1]):
        offset = 0
        while offset < len(variants):
            frame = int(np.searchsorted(adapter._starts, variants[offset], side='right') - 1)
            stop = int(np.searchsorted(variants, adapter._starts[frame] + adapter._sizes[frame], side='left'))
            for begin in range(offset, stop, block_size):
                yield variants[begin:min(begin + block_size, stop)]
            offset = stop
    else:
        # Unsorted valid requests retain caller order rather than being sorted.
        for begin in range(0, len(variants), block_size):
            yield variants[begin:begin + block_size]



CROSS_FRAME_SINGLE_POLICY = 'source-upper-MAC-bounded-requests-v1'


def iter_cross_frame_requests(adapter, variant_indices, block_size=1024, *,
                              minimum_mac_bound=None, metrics=None):
    """Pack ordered source-safe columns across physical frames, at most B each.

    Only the exact integer upper bound already used by ``read_states`` is
    applied: ``min(full_REF, full_called - full_REF + 1) >= minimum_mac``.
    Every accepted request still goes through ``read_states`` and independent
    cohort MAC/orientation/imputation. No cohort summaries or model products
    are shared. Full-cache upper MAC may admit extra columns; it cannot remove
    a column that the existing reader would have returned for any cohort.

    Pending ownership is at most B int64 indices plus one physical request's
    small index/summary vectors. It retains no frame exception payload, raw
    genotype, dosage, model, output row or device tensor. A selected axis whose
    source bound rejects every column yields one empty request so a writer can
    persist legal zero-row coverage. A genuinely empty selected axis yields
    nothing. Request/ordinal/checkpoint semantics are deliberately versioned
    separately from the legacy per-frame policy by the caller.
    """
    from ..genotype import _indices
    if type(block_size) is not int or block_size < 1:
        raise ValueError('cross-frame Single block_size must be a positive integer')
    if (minimum_mac_bound is not None and
            (not np.isscalar(minimum_mac_bound) or not np.isfinite(minimum_mac_bound)
             or minimum_mac_bound < 0)):
        raise ValueError('minimum MAC must be finite and nonnegative')
    if not callable(getattr(adapter, '_source', None)):
        raise TypeError('cross-frame Single requires verified broker frame summaries')
    variants = _indices(variant_indices, adapter._reader.n_variants, 'variant_indices')
    counters = {} if metrics is None else metrics
    for name in ('physical_requests_examined', 'planning_source_lookups',
                 'source_bound_kept_variants', 'source_bound_rejected_variants',
                 'emitted_batches', 'emitted_empty_batches', 'maximum_request_columns'):
        counters.setdefault(name, 0)
    counters.setdefault('source_planning_host_seconds', 0.)
    pending, source = [], None

    def emit(values):
        # A bytes owner makes the published request genuinely immutable, even
        # if the consumer retains it while the generator fills its next batch.
        request = np.frombuffer(np.asarray(values, dtype=np.int64).tobytes(), dtype=np.int64)
        counters['emitted_batches'] += 1
        counters['emitted_empty_batches'] += int(not len(request))
        counters['maximum_request_columns'] = max(counters['maximum_request_columns'], len(request))
        return request

    try:
        for physical in _physical_requests(adapter, variants, block_size):
            counters['physical_requests_examined'] += 1
            began = time.perf_counter()
            if minimum_mac_bound is None:
                kept = physical
            else:
                frames = np.searchsorted(adapter._starts, physical, side='right') - 1
                keep = np.zeros(len(physical), dtype=bool)
                for frame in np.unique(frames):
                    source = adapter._source(int(frame))
                    counters['planning_source_lookups'] += 1
                    positions = np.flatnonzero(frames == frame)
                    columns = physical[positions] - adapter._starts[frame]
                    upper = np.minimum(source.ref_ac[columns],
                                       source.called[columns] - source.ref_ac[columns] + 1)
                    keep[positions] = upper >= minimum_mac_bound
                    source = None
                kept = physical[keep]
            counters['source_planning_host_seconds'] += time.perf_counter()-began
            counters['source_bound_kept_variants'] += len(kept)
            counters['source_bound_rejected_variants'] += len(physical)-len(kept)
            for variant in kept:
                pending.append(int(variant))
                if len(pending) == block_size:
                    request = emit(pending)
                    pending.clear()
                    yield request
                    del request
        if pending:
            request = emit(pending)
            pending.clear()
            yield request
        elif len(variants) and not counters['emitted_batches']:
            yield emit(())
    finally:
        pending.clear()
        source = None

def iter_effective_minor_blocks(adapter, variant_indices, union_sample_indices,
                                block_size=1024, *, effective_block_size=1024,
                                device=None, minimum_mac=None, resident=True):
    """Yield at most effective_block_size retained columns per device block.

    Physical input reads remain bounded by block_size and compressed frames.
    MAC filtering happens in the original cohort decoder before packing. The
    subsequent pipeline keeps its global MAC-filtered ordinal, independently
    of these compute batches, preserving native chunk/factor ordering.
    """
    from ..genotype import _indices
    if adapter._closed:
        raise RuntimeError('cache reader is closed')
    if not resident:
        raise ValueError('effective Single batches require resident CUDA dosage')
    if type(block_size) is not int or block_size < 1:
        raise ValueError('block_size must be a positive integer')
    if type(effective_block_size) is not int or effective_block_size < 1:
        raise ValueError('effective_block_size must be a positive integer')
    variants = _indices(variant_indices, adapter.n_variants, 'variant_indices')
    samples = _indices(union_sample_indices, adapter.n_samples, 'union_sample_indices')
    if len(samples) * effective_block_size > np.iinfo(np.int32).max:
        raise MemoryError('effective Single dosage geometry exceeds validated int32 CUDA indexing')
    metrics = adapter._metrics
    parts, pending_count = [], 0

    def materialize():
        began = time.perf_counter()
        prepared, selected_variants = _merge_prepared(parts, len(samples))
        pack_seconds = time.perf_counter() - began
        metrics['single_pack_seconds'] = metrics.get('single_pack_seconds', 0.) + pack_seconds
        add_prepare_seconds = getattr(adapter, '_add_compact_prepare_seconds', None)
        if add_prepare_seconds is not None:
            add_prepare_seconds(pack_seconds)
        else:
            # Preserve compatibility with synchronous third-party/mock adapters.
            metrics['compact_prepare_seconds'] += pack_seconds
        began = time.perf_counter()
        result = adapter._fast.to_minor_block(prepared, samples, selected_variants,
                                               device=device or adapter._device)
        seconds = time.perf_counter() - began
        metrics['materialize_seconds'] += seconds
        metrics['minor_block_wall_seconds'] += pack_seconds + seconds
        metrics['minor_block_calls'] += 1
        metrics['cuda_resident_calls'] += 1
        metrics['returned_variants'] += result.shape[1]
        metrics['single_effective_blocks'] = metrics.get('single_effective_blocks', 0) + 1
        metrics['single_effective_columns'] = metrics.get('single_effective_columns', 0) + result.shape[1]
        return result

    requests = _physical_requests(adapter, variants, block_size)
    prepare_requests = getattr(adapter, '_prepared_requests', None)
    iterator = (prepare_requests(requests, samples, minimum_mac) if prepare_requests is not None else
                (adapter._prepare(request, samples, minimum_mac) for request in requests))
    try:
        while True:
            began = time.perf_counter()
            item = next(iterator, None)
            metrics['minor_block_wall_seconds'] += time.perf_counter() - began
            if item is None:
                break
            prepared, bound_samples, bound_variants = item
            metrics['requested_variants'] += len(bound_variants)
            metrics['single_compact_requests'] = metrics.get('single_compact_requests', 0) + 1
            if not np.array_equal(bound_samples, samples):
                raise RuntimeError('Single compact request changed its bound sample axis')
            count = len(prepared['columns'])
            begin = 0
            while begin < count:
                stop = min(count, begin + effective_block_size - pending_count)
                part = prepared if begin == 0 and stop == count else _slice_prepared(prepared, begin, stop)
                parts.append((part, bound_variants[part['columns']]))
                pending_count += stop - begin
                begin = stop
                if pending_count == effective_block_size:
                    result = materialize()
                    parts.clear()
                    pending_count = 0
                    yield result
                    del result
    finally:
        iterator.close()
    if pending_count:
        result = materialize()
        parts.clear()
        yield result
