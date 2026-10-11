"""Storage ownership and interrupted allocation contracts, not benchmarks."""
import numpy as np
import pytest
import torch
from fudan_wgs_toolkit.genotype_device import DeviceMinorBlock
from fudan_wgs_toolkit.phewas_runtime.buffers import EffectiveBuffer


def block(dosage, start=0):
    width=dosage.shape[1]
    values=np.arange(start,start+width)
    result=DeviceMinorBlock(dosage,np.arange(dosage.shape[0]),values,
        *(values.astype(np.float64)+index for index in range(5)))
    result._store_counts(np.arange(dosage.shape[0]),values.astype(np.float64),values)
    return result


def test_continuous_split_keeps_backing_view_and_avoids_index_select(monkeypatch):
    dosage=torch.arange(40,dtype=torch.uint8).reshape(4,10)
    original=block(dosage)
    queue=EffectiveBuffer(np.arange(4),3)
    queue.append(original)
    monkeypatch.setattr(DeviceMinorBlock,'select_columns',lambda *_:pytest.fail('continuous slice copied'))
    packed=queue.take()
    assert packed.dosage.untyped_storage().data_ptr()==dosage.untyped_storage().data_ptr()
    torch.testing.assert_close(packed.dosage,dosage[:,:3])
    np.testing.assert_array_equal(packed._cached_counts(np.arange(4))[0],[0,1,2])
    summary=queue.resident_storage_summary()
    assert summary['logical_dosage_bytes']==28
    assert summary['retained_dosage_storage_bytes']==40
    assert summary['queued_variant_count']==7
    assert summary['returned_blocks_excluded']


def test_small_retained_tail_is_compacted_without_copying_selected_prefix():
    dosage=torch.arange(40,dtype=torch.uint8).reshape(4,10)
    queue=EffectiveBuffer(np.arange(4),9)
    queue.append(block(dosage))
    selected=queue.take()
    assert selected.dosage.untyped_storage().data_ptr()==dosage.untyped_storage().data_ptr()
    remainder=queue.parts[0]
    assert remainder.dosage.untyped_storage().data_ptr()!=dosage.untyped_storage().data_ptr()
    assert remainder.dosage.is_contiguous()
    torch.testing.assert_close(remainder.dosage,dosage[:,9:])
    assert queue.resident_storage_summary()['retained_dosage_storage_bytes']==4
    np.testing.assert_array_equal(remainder._cached_counts(np.arange(4))[0],[9])


@pytest.mark.parametrize('operation',['cat','clone'])
def test_allocation_failure_preserves_entire_queue_and_order(monkeypatch,operation):
    queue=EffectiveBuffer(np.arange(4),9 if operation=='clone' else 3)
    parts=[block(torch.ones((4,10),dtype=torch.uint8))] if operation=='clone' else [
        block(torch.ones((4,1),dtype=torch.uint8)),block(torch.ones((4,4),dtype=torch.uint8),1)]
    for part in parts:queue.append(part)
    original_count=queue.count
    def failure(*_args,**_kwargs):raise MemoryError('injected allocation failure')
    monkeypatch.setattr(torch.Tensor,'clone',failure) if operation=='clone' else monkeypatch.setattr(torch,'cat',failure)
    with pytest.raises(MemoryError,match='injected allocation'):
        queue.take()
    assert queue.count==original_count and list(queue.parts)==parts
    assert [part.variant_indices.tolist() for part in queue.parts]==[part.variant_indices.tolist() for part in parts]


def test_guard_reserves_only_new_concat_and_tail_bytes_before_allocating():
    requests=[]
    queue=EffectiveBuffer(np.arange(4),9,allocation_guard=requests.append)
    queue.append(block(torch.ones((4,10),dtype=torch.uint8)))
    queue.take()
    assert requests==[4+64*2**20]  # A view prefix plus one copied tail column.
    queue=EffectiveBuffer(np.arange(4),3,allocation_guard=requests.append)
    queue.append(block(torch.ones((4,1),dtype=torch.uint8)))
    queue.append(block(torch.ones((4,4),dtype=torch.uint8),1))
    queue.take()
    assert requests[-1]==12+64*2**20


def test_queue_budget_counts_aliased_backing_once_and_rejects_transactionally():
    dosage=torch.ones((4,10),dtype=torch.uint8)
    queue=EffectiveBuffer(np.arange(4),3,max_retained_bytes=40)
    first,second=block(dosage[:,:5]),block(dosage[:,5:],5)
    queue.append(first);queue.append(second)
    assert queue.resident_storage_summary()['retained_dosage_storage_bytes']==40
    with pytest.raises(MemoryError,match='retained dosage'):
        queue.append(block(torch.ones((4,1),dtype=torch.uint8),10))
    assert queue.count==10 and list(queue.parts)==[first,second]
    small=EffectiveBuffer(np.arange(4),1,max_retained_bytes=4)
    with pytest.raises(MemoryError,match='retained dosage'):
        small.append(block(dosage[:,:1]))
    assert small.count==0 and not small.parts


def test_view_count_cache_tracks_shared_in_place_changes():
    dosage=torch.ones((4,10),dtype=torch.uint8)
    original=block(dosage)
    queue=EffectiveBuffer(np.arange(4),3)
    queue.append(original)
    packed=queue.take()
    assert packed._cached_counts(np.arange(4)) is not None
    dosage[0,0]=2
    assert packed._cached_counts(np.arange(4)) is None
    assert original._cached_counts(np.arange(4)) is None


@pytest.mark.parametrize('options',[dict(max_retained_bytes=-1),dict(max_retained_bytes=True),
    dict(tail_compaction_ratio=float('nan')),dict(tail_compaction_ratio=.9)])
def test_invalid_storage_policy_rejected(options):
    with pytest.raises(ValueError):EffectiveBuffer(np.arange(4),3,**options)
