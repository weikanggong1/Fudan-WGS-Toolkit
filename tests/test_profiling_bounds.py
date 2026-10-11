"""Diagnostic event pressure must not retain unbounded CUDA owners."""
from contextlib import nullcontext

from fudan_wgs_toolkit import profiling


def test_unfinished_intervals_are_bounded_and_coverage_is_explicit(monkeypatch):
    class Event:
        ready = False

        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def query(self):
            return self.ready

        def elapsed_time(self, other):
            return 20.

    monkeypatch.setattr(profiling.torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(profiling.torch.cuda, 'Event', Event)
    profiler = profiling.StageProfiler('cuda:0')
    profiler.maximum_pending = 2
    for _ in range(10):
        with profiler.measure('work', gpu=True):
            pass
    result = profiler.report()
    assert result['pending_cuda_intervals'] == 2
    assert result['omitted_cuda_intervals'] == 8
    assert result['stages']['work']['calls'] == 10
    Event.ready = True
    result = profiler.report()
    assert result['pending_cuda_intervals'] == 0
    assert result['stages']['work']['cuda_stream_seconds'] == .04
