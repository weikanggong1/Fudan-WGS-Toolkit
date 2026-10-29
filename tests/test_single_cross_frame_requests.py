"""Bounded source-upper-MAC planning contracts; fixtures are not benchmarks."""
from types import SimpleNamespace

import numpy as np
import pytest

from fudan_wgs_toolkit.cache_runtime.single_batches import (
    CROSS_FRAME_SINGLE_POLICY, _physical_requests, iter_cross_frame_requests)


class SummaryAdapter:
    def __init__(self,reference,called,*,frame_size=4):
        self.reference=np.asarray(reference,dtype=np.int64)
        self.called=np.asarray(called,dtype=np.int64)
        self._starts=np.arange(0,len(reference),frame_size,dtype=np.int64)
        self._sizes=np.minimum(frame_size,len(reference)-self._starts)
        self._reader=SimpleNamespace(n_variants=len(reference))
        self.lookups=[]
    def _source(self,frame):
        self.lookups.append(frame);start=self._starts[frame];stop=start+self._sizes[frame]
        return SimpleNamespace(ref_ac=self.reference[start:stop],called=self.called[start:stop])


@pytest.mark.parametrize('axis',[
    np.arange(17,dtype=np.int64),np.asarray([0,2,4,7,8,12,16]),
    np.asarray([16,0,7,2,12,4,8])])
@pytest.mark.parametrize('threshold',[None,0,1,2,3,5,25])
def test_bounded_requests_equal_original_safe_upper_filter_and_preserve_order(axis,threshold):
    reference=np.asarray([0,1,2,3,5,7,9,4,8,0,1,3,6,7,9,2,4])
    called=np.asarray([10,9,9,8,10,9,10,5,9,0,2,5,7,8,10,7,9])
    adapter=SummaryAdapter(reference,called);metrics={}
    outputs=list(iter_cross_frame_requests(adapter,axis,5,minimum_mac_bound=threshold,metrics=metrics))
    upper=np.minimum(reference[axis],called[axis]-reference[axis]+1)
    expected=axis if threshold is None else axis[upper>=threshold]
    np.testing.assert_array_equal(np.concatenate(outputs) if outputs else [],expected)
    assert all(len(request)<=5 for request in outputs)
    assert all(not request.flags.writeable for request in outputs)
    assert metrics['maximum_request_columns']<=5
    assert metrics['source_bound_kept_variants']==len(expected)
    assert metrics['source_bound_rejected_variants']==len(axis)-len(expected)
    if len(expected):assert len(outputs)==(len(expected)+4)//5
    else:assert len(outputs)==1 and metrics['emitted_empty_batches']==1
    if threshold is None:assert adapter.lookups==[]


def test_underfilled_frames_pack_and_final_tail_is_bounded():
    reference=np.full(25,20);called=np.full(25,20)
    reference[[1,5,9,13,17,21,24]]=16
    adapter=SummaryAdapter(reference,called,frame_size=4);metrics={}
    outputs=list(iter_cross_frame_requests(adapter,np.arange(25),4,minimum_mac_bound=3,metrics=metrics))
    assert [x.tolist() for x in outputs]==[[1,5,9,13],[17,21,24]]
    assert metrics['physical_requests_examined']==7
    assert metrics['emitted_batches']==2
    assert CROSS_FRAME_SINGLE_POLICY.endswith('-v1')


def test_empty_axis_does_not_read_frames_or_emit_a_fake_checkpoint():
    adapter=SummaryAdapter([4]*8,[4]*8);metrics={}
    assert list(iter_cross_frame_requests(adapter,np.asarray([],dtype=np.int64),4,
        minimum_mac_bound=3,metrics=metrics))==[]
    assert adapter.lookups==[] and metrics['emitted_batches']==0


@pytest.mark.parametrize('size',[True,0,-1,1.5])
def test_bad_bound_rejected_before_frame_source(size):
    adapter=SummaryAdapter([1]*4,[4]*4)
    with pytest.raises(ValueError,match='positive integer'):
        list(iter_cross_frame_requests(adapter,np.arange(4),size,minimum_mac_bound=1))
    assert adapter.lookups==[]


@pytest.mark.parametrize('threshold',[-1,float('nan'),float('inf')])
def test_bad_mac_rejected_before_frame_source(threshold):
    adapter=SummaryAdapter([1]*4,[4]*4)
    with pytest.raises(ValueError,match='MAC'):
        list(iter_cross_frame_requests(adapter,np.arange(4),4,minimum_mac_bound=threshold))
    assert adapter.lookups==[]


@pytest.mark.parametrize('axis,error',[(np.asarray([1,1]),ValueError),(np.asarray([8]),IndexError),
    (np.asarray([-1]),IndexError),(np.asarray([1.]),ValueError)])
def test_bad_axis_rejected_before_source(axis,error):
    adapter=SummaryAdapter([1]*8,[4]*8)
    with pytest.raises(error):list(iter_cross_frame_requests(adapter,axis,4,minimum_mac_bound=1))
    assert adapter.lookups==[]


def test_source_error_propagates_after_only_bounded_prior_requests():
    adapter=SummaryAdapter([2]*20,[8]*20);metrics={};original=adapter._source
    def fail(frame):
        if frame==2:raise OSError('injected source failure')
        return original(frame)
    adapter._source=fail
    iterator=iter_cross_frame_requests(adapter,np.arange(20),4,minimum_mac_bound=1,metrics=metrics)
    assert next(iterator).tolist()==[0,1,2,3]
    assert next(iterator).tolist()==[4,5,6,7]
    with pytest.raises(OSError,match='injected source failure'):next(iterator)
    assert metrics['maximum_request_columns']==4
    iterator.close()


def test_explicit_generator_close_never_reads_the_rest_of_the_axis():
    adapter=SummaryAdapter([2]*100,[8]*100)
    iterator=iter_cross_frame_requests(adapter,np.arange(100),4,minimum_mac_bound=1)
    first=next(iterator);iterator.close()
    assert first.tolist()==[0,1,2,3] and adapter.lookups==[0]
