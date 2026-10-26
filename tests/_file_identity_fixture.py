"""Make a real rewrite observable when a filesystem coalesces timestamp ticks."""
import os
import time

import pytest


def rewrite_with_changed_ctime(path, payload, *, preserve_mtime=False):
    original = path.stat()
    deadline = time.monotonic()+2.
    while True:
        path.write_bytes(payload)
        if preserve_mtime:
            os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        if path.stat().st_ctime_ns != original.st_ctime_ns:
            return
        if time.monotonic() >= deadline:
            pytest.skip("filesystem does not expose a distinct ctime for the rewrite fixture")
        time.sleep(.01)
