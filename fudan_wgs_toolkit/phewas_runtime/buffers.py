"""Bounded eligible-column queues with transactional, ordered storage views."""
from collections import deque
import math
import numpy as np
import torch
from ..genotype_device import DeviceMinorBlock
from ..phewas_resources import tensor_storage_footprint


_SUMMARIES = ('union_ref_af', 'union_initial_mac', 'union_missing_rate',
              'union_ref_ac', 'union_called_alleles')


def _column_view(part, start, stop, *, compact=False):
    """Slice dosage without a device transfer; copy only small host summaries."""
    if start == 0 and stop == part.shape[1] and not compact:
        return part
    dosage = part.dosage[:, start:stop]
    if compact:
        dosage = dosage.clone(memory_format=torch.contiguous_format)
    block = DeviceMinorBlock(dosage, part.sample_indices,
        part.variant_indices[start:stop].copy(),
        *[getattr(part, name)[start:stop].copy() for name in _SUMMARIES])
    cached = part._counts_cache
    if cached is not None and cached[0] == part._dosage_stamp():
        block._store_counts(cached[1], cached[2][start:stop], cached[3][start:stop])
    return block


def _dosage_bytes(parts):
    return tensor_storage_footprint([part.dosage for part in parts])['retained_storage_bytes']


class EffectiveBuffer:
    def __init__(self, samples, block_size, *, allocation_guard=None,
                 max_retained_bytes=None, tail_compaction_ratio=2.):
        self.samples = np.asarray(samples, dtype=np.int64).copy()
        if type(block_size) is not int or block_size < 1:
            raise ValueError('effective block size must be positive')
        if len(self.samples) * block_size > np.iinfo(np.int32).max:
            raise MemoryError('effective dosage exceeds validated int32 geometry')
        if max_retained_bytes is not None and (type(max_retained_bytes) is not int or max_retained_bytes < 0):
            raise ValueError('retained dosage budget must be a nonnegative integer byte count')
        if (isinstance(tail_compaction_ratio, bool) or not isinstance(tail_compaction_ratio, (int, float))
                or not math.isfinite(tail_compaction_ratio) or tail_compaction_ratio < 1):
            raise ValueError('tail compaction ratio must be finite and at least one')
        self.block_size, self.count, self.parts = block_size, 0, deque()
        self.allocation_guard = allocation_guard
        self.max_retained_bytes = max_retained_bytes
        self.tail_compaction_ratio = float(tail_compaction_ratio)

    def _check_retained(self, parts):
        if self.max_retained_bytes is not None and _dosage_bytes(parts) > self.max_retained_bytes:
            raise MemoryError('eligible-column queue exceeds its retained dosage storage budget')

    def append(self, block):
        if not np.array_equal(block.sample_indices, self.samples):
            raise ValueError('buffer block changed the singleton sample order')
        if block.shape[1]:
            # A small view can hold its original allocation alive. Compare real
            # backing bytes, deduplicating aliases, before mutating the queue.
            self._check_retained([*self.parts, block])
            self.parts.append(block)
            self.count += block.shape[1]

    def clear(self):
        """Release owned dosage after cancellation; never close its reader."""
        self.parts.clear()
        self.count = 0

    def resident_storage_summary(self):
        """Queue-owned endpoint; views keep full backing allocations alive."""
        parts = list(self.parts)
        dosage = tensor_storage_footprint([part.dosage for part in parts])
        host_arrays = [self.samples]
        for part in parts:
            host_arrays.extend([part.sample_indices, part.variant_indices,
                                *[getattr(part, name) for name in _SUMMARIES]])
            if part._counts_cache is not None:
                host_arrays.extend(part._counts_cache[1:])
        host = tensor_storage_footprint(host_arrays)
        return dict(queued_variant_count=self.count, queued_part_count=len(parts),
            logical_dosage_bytes=sum(part.dosage.numel()*part.dosage.element_size() for part in parts),
            retained_dosage_storage_bytes=dosage['retained_storage_bytes'],
            retained_dosage_storage_bytes_by_device=dosage['retained_storage_bytes_by_device'],
            retained_host_summary_storage_bytes=host['retained_storage_bytes'],
            maximum_retained_dosage_bytes=self.max_retained_bytes,
            returned_blocks_excluded=True, shared_backing_aliases_deduplicated=True)

    def take(self, *, tail=False):
        count = min(self.count, self.block_size)
        if count == 0 or (count < self.block_size and not tail):
            return None
        # Plan without consuming anything. A failed guard, clone, concatenation
        # or host-summary allocation leaves the complete original queue intact.
        selected_plan, remainder_plan, remaining = [], [], count
        tail_copy_bytes = 0
        for part in self.parts:
            width = part.shape[1]
            if remaining == 0:
                remainder_plan.append((part, 0, width, False))
            elif width <= remaining:
                selected_plan.append((part, 0, width))
                remaining -= width
            else:
                selected_plan.append((part, 0, remaining))
                logical_tail = part.shape[0]*(width-remaining)*part.dosage.element_size()
                compact = _dosage_bytes([part]) > self.tail_compaction_ratio*logical_tail
                remainder_plan.append((part, remaining, width, compact))
                if compact:
                    tail_copy_bytes += logical_tail
                remaining = 0
        if remaining:
            raise RuntimeError('eligible-column queue count does not match its blocks')
        concat_bytes = (sum(part.shape[0]*(stop-start)*part.dosage.element_size()
                        for part,start,stop in selected_plan) if len(selected_plan)>1 else 0)
        if self.allocation_guard is not None:
            # Existing backing storage is already allocated and never added to
            # the NEW-byte guard again. Keep the established native scratch.
            self.allocation_guard(concat_bytes+tail_copy_bytes+64*2**20)
        selected = [_column_view(part,start,stop) for part,start,stop in selected_plan]
        remainder = deque(_column_view(part,start,stop,compact=compact)
                          for part,start,stop,compact in remainder_plan)
        if len(selected) == 1:
            block = selected[0]
        else:
            block = DeviceMinorBlock(torch.cat([x.dosage for x in selected], dim=1), self.samples,
                np.concatenate([x.variant_indices for x in selected]),
                *[np.concatenate([getattr(x, name) for x in selected]) for name in _SUMMARIES])
            rows = np.arange(len(self.samples))
            counts = [x._cached_counts(rows) for x in selected]
            if all(c is not None for c in counts):
                block._store_counts(rows, np.concatenate([c[0] for c in counts]),
                                    np.concatenate([c[1] for c in counts]))
        self._check_retained(remainder)
        self.parts = remainder
        self.count -= count
        return block
