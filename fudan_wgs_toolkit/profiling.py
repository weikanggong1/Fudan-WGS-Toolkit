"""Bounded stage timing without synchronizing each GPU operation.

CUDA event intervals measure stream elapsed time, including idle/dispatch gaps;
they are not pure kernel timings. The caller resolves events at existing job
boundaries after synchronization. Nested/different stage times must not be added
to end-to-end wall time. All records are aggregate and contain no identifiers.
"""
from collections import defaultdict
from contextlib import contextmanager
import time
import torch


class StageProfiler:
    def __init__(self, device="cpu", enabled=True):
        self.device = str(device)
        self.enabled = enabled
        self.records = defaultdict(lambda: {"calls": 0, "host_wall_seconds": 0.0,
                                            "cuda_stream_seconds": 0.0})
        self.pending = []
        self.maximum_pending = 256
        self.omitted_cuda_intervals = 0

    @contextmanager
    def measure(self, name, *, gpu=False):
        if not self.enabled:
            yield
            return
        use_events = gpu and self.device.startswith("cuda")
        if use_events:
            with torch.cuda.device(self.device):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
        before = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - before
            if use_events:
                with torch.cuda.device(self.device):
                    end.record()
                if len(self.pending) >= self.maximum_pending:
                    self.flush()
                if len(self.pending) < self.maximum_pending:
                    self.pending.append((name, start, end))
                else:
                    # Diagnostic pressure never synchronizes a scientific mask.
                    # Report incomplete timing coverage instead of holding an
                    # unbounded list of CUDA event owners.
                    self.omitted_cuda_intervals += 1
            self.records[name]["calls"] += 1
            self.records[name]["host_wall_seconds"] += elapsed

    def flush(self):
        # CLI has already synchronized the job. Direct API can keep pending
        # events until the user chooses to synchronize and request a report.
        unresolved = []
        for name, start, end in self.pending:
            if end.query():
                self.records[name]["cuda_stream_seconds"] += start.elapsed_time(end) / 1000
            else:
                unresolved.append((name, start, end))
        self.pending = unresolved

    def report(self):
        self.flush()
        return {"enabled": self.enabled, "stages": dict(self.records),
                "pending_cuda_intervals": len(self.pending),
                "omitted_cuda_intervals": self.omitted_cuda_intervals,
                "timing_contract": "host wall and CUDA stream elapsed; overlapping metrics, not additive; pure kernel requires an external profiler"}
