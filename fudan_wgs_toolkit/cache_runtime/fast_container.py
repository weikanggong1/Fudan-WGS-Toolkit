"""Private immutable-container reader with three instance-owned stream handles.

All initialization and per-frame scientific validations run in unchanged store.
Unbuffered handles deliberately observe mutations even following repeated reads.
Stream identity checks happen at open/close, not per frame; callers must preserve
source-wide pre/post checks. No writes, global patching, GPU or shared FD state.
"""
import os
import threading
from . import store

_NAMES = ('data.bin', 'headers.bin', 'counts.bin')

def _identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

class _BorrowedHandle:
    def __init__(self, handle):
        self.handle = handle
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def seek(self, offset):
        self.offset = offset
        return offset
    def read(self, size):
        return os.pread(self.handle.fileno(), size, self.offset)

class _StreamPath:
    def __init__(self, path, handle, initial_stat):
        self.path, self.handle, self.initial_stat = path, handle, initial_stat
    def stat(self):
        return self.initial_stat
    def open(self, mode):
        if mode != 'rb':
            raise ValueError('Private stream proxy is read-only')
        return _BorrowedHandle(self.handle)

class _FramePath:
    def __init__(self, streams):
        self.streams = streams
    def __truediv__(self, name):
        return self.streams[name]


class _PrefetchedRange:
    def __init__(self, stream, offset, data):
        self.stream, self.offset, self.data = stream, offset, data
    def stat(self):
        return self.stream.initial_stat
    def open(self, mode):
        if mode != 'rb':
            raise ValueError('Prefetched stream is read-only')
        return self
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def seek(self, offset):
        if offset != self.offset:
            raise ValueError('Prefetched range offset differs')
        return offset
    def read(self, size):
        if size != len(self.data):
            raise ValueError('Prefetched range length differs')
        return self.data


class Container(store.Container):
    def __init__(self, path, expected_source_binding=None, expected_samples=None):
        super().__init__(path, expected_source_binding, expected_samples)
        self._lock = threading.RLock()
        self._closed = False
        self._streams = {}
        self._prefetch_pool = None
        self._prefetch_configured = False
        self._prefetch_future = None
        self._prefetch_frame = None
        self._prefetch_limit = 0
        self._prefetch_metrics = dict(submitted=0, hits=0, discarded=0,
                                      oversize_skipped=0, peak_compressed_bytes=0)
        try:
            for name in _NAMES:
                stream_path = self.path / name
                initial = stream_path.stat()
                offset_key, size_key = {
                    'data.bin': ('offset', 'size'),
                    'headers.bin': ('header_offset', 'header_size'),
                    'counts.bin': ('counts_offset', 'counts_size'),
                }[name]
                expected_size = (int(self.index[-1][offset_key]) + int(self.index[-1][size_key])
                                 if len(self.index) else 0)
                if initial.st_size != expected_size:
                    raise ValueError('Stream length changed while opening')
                handle = stream_path.open('rb', buffering=0)
                self._streams[name] = _StreamPath(stream_path, handle, initial)
                if _identity(os.fstat(handle.fileno())) != _identity(initial):
                    raise ValueError('Stream identity changed while opening')
            self._frame_path = _FramePath(self._streams)
        except BaseException:
            for stream in self._streams.values():
                stream.handle.close()
            self._closed = True
            raise

    def enable_prefetch(self, max_bytes=64*2**20):
        """Read one bounded compressed frame ahead; decoding remains unchanged."""
        if type(max_bytes) is not int or not 0 <= max_bytes <= 256*2**20:
            raise ValueError('prefetch bytes must be an integer in 0..256 MiB')
        with self._lock:
            if self._closed or self._prefetch_configured:
                raise ValueError('Prefetch can only be configured once on an open reader')
            self._prefetch_configured = True
            self._prefetch_limit = max_bytes
            if max_bytes:
                from concurrent.futures import ThreadPoolExecutor
                self._prefetch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='genetic-frame-read')

    def _compressed_ranges(self, frame_id):
        row = self.index[frame_id]
        result = {}
        for name, offset_key, size_key in (
                ('data.bin','offset','size'), ('headers.bin','header_offset','header_size'),
                ('counts.bin','counts_offset','counts_size')):
            stream = self._streams[name]
            offset, size = int(row[offset_key]), int(row[size_key])
            data = os.pread(stream.handle.fileno(), size, offset)
            if len(data) != size:
                raise ValueError('Short prefetched frame read')
            result[name] = _PrefetchedRange(stream, offset, data)
        return _FramePath(result)

    def prefetch(self, frame_id):
        """Schedule CPU-only pread without changing scientific validation."""
        if type(frame_id) is not int or not 0 <= frame_id < len(self.index):
            raise IndexError('Frame ID out of bounds')
        with self._lock:
            if self._closed:
                raise ValueError('Container is closed')
            if self._prefetch_pool is None:
                return False
            if self._prefetch_future is not None:
                if self._prefetch_frame == frame_id:
                    return True
                if not self._prefetch_future.done():
                    return False
                # Surface failed reads rather than hiding a corrupt source.
                future = self._prefetch_future
                self._prefetch_future = self._prefetch_frame = None
                future.result()
                del future
                self._prefetch_metrics['discarded'] += 1
            row = self.index[frame_id]
            size = int(row['size'])+int(row['header_size'])+int(row['counts_size'])
            if size > self._prefetch_limit:
                self._prefetch_metrics['oversize_skipped'] += 1
                return False
            self._prefetch_frame = frame_id
            self._prefetch_future = self._prefetch_pool.submit(self._compressed_ranges, frame_id)
            self._prefetch_metrics['submitted'] += 1
            self._prefetch_metrics['peak_compressed_bytes'] = max(
                self._prefetch_metrics['peak_compressed_bytes'], size)
            return True

    @property
    def prefetch_metrics(self):
        return dict(self._prefetch_metrics, limit_bytes=self._prefetch_limit,
                    max_pending_frames=1, cpu_only=True)

    def read_frame(self, frame_id):
        if type(frame_id) is not int or not 0 <= frame_id < len(self.index):
            raise IndexError('Frame ID out of bounds')
        with self._lock:
            if self._closed:
                raise ValueError('Container is closed')
            path = self._frame_path
            if self._prefetch_future is not None and self._prefetch_frame == frame_id:
                future = self._prefetch_future
                self._prefetch_future = self._prefetch_frame = None
                path = future.result()
                self._prefetch_metrics['hits'] += 1
            return store._read_frame(path,
                {key: self.index[frame_id][key] for key in store.INDEX.names}, self.manifest)

    def check_stream_identity(self):
        with self._lock:
            if self._closed:
                raise ValueError('Container is closed')
            for stream in self._streams.values():
                expected = _identity(stream.initial_stat)
                if (_identity(os.fstat(stream.handle.fileno())) != expected or
                        _identity(stream.path.stat()) != expected):
                    raise ValueError('Immutable stream changed after opening')
        return True

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                try:
                    if self._prefetch_pool is not None:
                        self._prefetch_pool.shutdown(wait=True, cancel_futures=True)
                        if self._prefetch_future is not None and not self._prefetch_future.cancelled():
                            self._prefetch_future.result()
                finally:
                    self._prefetch_future = self._prefetch_frame = None
                    self._prefetch_pool = None
                    self.check_stream_identity()
            finally:
                for stream in self._streams.values():
                    stream.handle.close()
                self._closed = True

    def __enter__(self):
        if self._closed:
            raise ValueError('Container is closed')
        return self

    def __exit__(self, exc_type, exc, traceback):
        # Preserve an existing validation failure while still closing every FD.
        try:
            self.close()
        except (OSError, ValueError):
            if exc_type is None:
                raise
        return False

    def __del__(self):
        # Best effort only; explicit close/context exit supplies identity proof.
        for stream in getattr(self, '_streams', {}).values():
            stream.handle.close()

open_container = Container
