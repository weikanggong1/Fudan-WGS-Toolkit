"""Per-process file descriptor admission for large binary phenotype banks."""
from __future__ import annotations

import os
from pathlib import Path
import resource
import time


def configure_worker_file_capacity(trait_count=0):
    """Raise only this process' soft limit before opening model/output owners.

    One active chromosome can own one lock and one sample-axis mapping per
    trait. An additional 2,048 descriptors cover bounded CSV append streams,
    metadata, CUDA and transient IO owners. Prefer at least 8,192 when the
    existing hard limit permits it. A lower hard limit is accepted only when
    it still satisfies that trait-dependent minimum; the hard limit is never
    raised. This changes resource admission, not an association calculation.
    """
    if type(trait_count) is not int or trait_count < 0:
        raise ValueError('file descriptor admission requires a nonnegative trait count')
    before, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    minimum = 2*trait_count+2048
    preferred = max(8192, minimum)
    if hard != resource.RLIM_INFINITY and hard < minimum:
        raise RuntimeError(f'worker needs at least {minimum} file descriptors; hard limit is {hard}')
    required = preferred if hard == resource.RLIM_INFINITY else min(preferred, hard)
    if before != resource.RLIM_INFINITY and before < required:
        resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard))
    effective, effective_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if effective != resource.RLIM_INFINITY and effective < required:
        raise RuntimeError('worker file descriptor soft limit did not satisfy admission')
    return dict(pid=os.getpid(), required_soft_limit=required,
        minimum_required_soft_limit=minimum, preferred_soft_limit=preferred,
        hard_limit_constrained=hard!=resource.RLIM_INFINITY and hard<preferred,
        soft_limit_before=None if before==resource.RLIM_INFINITY else before,
        soft_limit=None if effective==resource.RLIM_INFINITY else effective,
        hard_limit=None if effective_hard==resource.RLIM_INFINITY else effective_hard,
        raised=effective!=before, policy='process soft limit; one active chromosome writer')


class FileDescriptorTelemetry:
    """Throttled real process FD samples, including a verified startup limit."""
    def __init__(self, admission):
        self.admission = dict(admission)
        self.maximum = 0
        self.samples = 0
        self.last = 0.
        self.sample(force=True)

    def sample(self, *, force=False):
        now = time.monotonic()
        if not force and now-self.last < 1.:
            return
        directory = Path('/proc/self/fd')
        if not directory.is_dir():
            return
        # scandir's own handle is closed before counting; the observation
        # includes that transient handle and is explicitly a sampled maximum.
        with os.scandir(directory) as values:
            count = sum(1 for _ in values)
        self.maximum = max(self.maximum, count)
        self.samples += 1
        self.last = now

    def report(self):
        self.sample(force=True)
        soft = self.admission['soft_limit']
        return dict(self.admission, maximum_observed_open_files=self.maximum,
            observed_minimum_headroom=None if soft is None else max(0, soft-self.maximum),
            samples=self.samples, observation='throttled /proc/self/fd samples; not an exact peak')
