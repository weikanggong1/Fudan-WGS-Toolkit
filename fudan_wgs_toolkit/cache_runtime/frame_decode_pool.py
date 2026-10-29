"""Bounded CPU-only frame decoding in independent, persistent interpreters.

Only immutable integer frames cross the process boundary. Children call the
original store decoder; the GPU consumer retains all population/threshold and
statistical operations. Private read-only mmap handoffs avoid large pickles.
No multiprocessing semaphore or resource-tracker process is created.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import numpy as np

from . import store, sparse_codec_fast

_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
_ARRAYS = ('offsets', 'sample_index', 'state', 'reference_alleles', 'called_alleles', 'half_missing_samples')


def _identity():
    fields = Path('/proc/self/stat').read_text().split(') ', 1)[1].split()
    return dict(pid=os.getpid(), uid=os.getuid(), starttime=int(fields[19]),
                session=os.getsid(0), pgrp=os.getpgrp(), parent=os.getppid(),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                effective_affinity=sorted(os.sched_getaffinity(0)))


def _child_identity(pid):
    fields = Path('/proc/%d/stat' % pid).read_text().split(') ', 1)[1].split()
    return dict(pid=pid, uid=Path('/proc/%d' % pid).stat().st_uid, starttime=int(fields[19]),
        session=os.getsid(pid), pgrp=os.getpgid(pid), parent=int(fields[1]),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        effective_affinity=sorted(os.sched_getaffinity(pid)))


def _cuda_initialized():
    module = sys.modules.get('torch')
    return bool(module is not None and module.cuda.is_initialized())


def _binding(container):
    return store.sha(store.canonical(dict(manifest=container.manifest,
        manifest_sha256=store.file_sha(container.path/'manifest.json'),
        decoder_sha256=store.file_sha(Path(store.__file__)),
        codec_sha256=store.file_sha(Path(sparse_codec_fast.__file__)))))


def _atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def _parent_alive(owner):
    if os.getppid() != owner['pid']:
        return False
    try:
        fields = Path('/proc/%d/stat' % owner['pid']).read_text().split(') ', 1)[1].split()
        return (fields[0] != 'Z' and int(fields[19]) == owner['starttime']
            and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == owner['boot_id'])
    except (OSError, IndexError, ValueError):
        return False


def _watch_parent(owner):
    while _parent_alive(owner):
        time.sleep(1)
    os._exit(70)


def _respond(value):
    sys.stdout.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
    sys.stdout.flush()


def _worker():
    """JSON protocol on stdio; stdout never contains source or array data."""
    from .fast_container import Container
    container = None
    try:
        setup = json.loads(sys.stdin.readline())
        if not _parent_alive(setup['owner']) or _cuda_initialized():
            raise RuntimeError('CPU frame child parent/CUDA guard failed')
        if any(os.environ.get(name) != '1' for name in _ENV):
            raise RuntimeError('CPU frame child numerical thread limit differs')
        threading.Thread(target=_watch_parent, args=(setup['owner'],), daemon=True,
                         name='frame-decode-parent-watch').start()
        container = Container(setup['source_directory'])
        if _binding(container) != setup['source_binding_sha256']:
            raise ValueError('CPU frame child immutable source binding differs')
        root = Path(setup['handoff_directory'])
        _respond(dict(ready=True, identity=_identity(), cuda_initialized=_cuda_initialized(),
                      source_binding_sha256=setup['source_binding_sha256'],
                      single_thread_environment={name:os.environ[name] for name in _ENV}))
        for line in sys.stdin:
            request = json.loads(line)
            if request['operation'] == 'close':
                container.close()
                container = None
                _respond(dict(closed=True, identity=_identity(), cuda_initialized=_cuda_initialized()))
                return 0
            if request['operation'] != 'decode' or _cuda_initialized():
                raise RuntimeError('Invalid CPU frame child request/CUDA state')
            frame_id, ordinal = request['frame_id'], request['ordinal']
            wall, cpu = time.perf_counter(), time.process_time()
            started_monotonic = time.monotonic()
            # This is the original, unchanged checksum/decoder implementation.
            frame = container.read_frame(frame_id)
            decoded = time.perf_counter()
            destination = root/str(ordinal)
            destination.mkdir(mode=0o700)
            arrays = {}
            for name in _ARRAYS:
                value = frame[name]
                path = destination/(name+'.npy')
                np.save(path, value, allow_pickle=False)
                os.chmod(path, 0o600)
                arrays[name] = dict(filename=path.name, dtype=value.dtype.str,
                    shape=list(value.shape), nbytes=value.nbytes, sha256=store.file_sha(path))
            _respond(dict(ordinal=ordinal, frame_id=frame_id, arrays=arrays,
                m=frame['m'], n=frame['n'], start=frame['start'], identity=_identity(),
                cuda_initialized=_cuda_initialized(),
                source_binding_sha256=setup['source_binding_sha256'],
                decode_wall_seconds=decoded-wall, wall_seconds=time.perf_counter()-wall,
                cpu_seconds=time.process_time()-cpu, started_monotonic=started_monotonic,
                ended_monotonic=time.monotonic()))
            del frame
        return 0
    except BaseException as error:
        _respond(dict(error=type(error).__name__+': '+str(error)))
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if container is not None:
            container.close()


class FrameDecodeProcessPool:
    """Ordered bounded read/decode ahead, with immutable raw integer handoff.

    The byte budget accounts conservative decode scratch plus pending handoff
    payloads. Child interpreter baselines and the consumer's existing compact
    LRU are separate and require process-tree host-memory admission. A frame
    larger than the budget is rejected, never silently dispatched over budget.
    """
    def __init__(self, container, processes, memory_bytes, temporary_directory=None,
                 worker_timeout_seconds=300.):
        if type(processes) is not int or not 1 <= processes <= 32:
            raise ValueError('frame prefetch processes must be in 1..32')
        if type(memory_bytes) is not int or memory_bytes <= 0:
            raise ValueError('frame prefetch memory bytes must be positive')
        if (isinstance(worker_timeout_seconds,bool) or not isinstance(worker_timeout_seconds,(int,float))
                or not math.isfinite(worker_timeout_seconds) or worker_timeout_seconds <= 0):
            raise ValueError('CPU frame worker timeout must be positive and finite')
        self.container, self.processes, self.memory_bytes = container, processes, memory_bytes
        self.worker_timeout_seconds = float(worker_timeout_seconds)
        self.pending_limit = 2*processes
        self._binding = _binding(container)
        self._owner = _identity()
        self._root = Path(tempfile.mkdtemp(prefix='frame-decode-', dir=temporary_directory))
        os.chmod(self._root, 0o700)
        self._handoff = self._root/'handoffs'
        self._handoff.mkdir(mode=0o700)
        self._registry_path = self._root/'registry.json'
        self._registry = dict(schema='frame-decode-pool-v1', owner=self._owner,
            requested_processes=processes, source_binding_sha256=self._binding,
            workers=[], pool_shutdown=False)
        self._executor = None
        self._workers = []
        self._available = queue.Queue()
        self._pending = OrderedDict()
        self._plan = deque()
        self._planned = set()
        self._estimates = {}
        self._charged = self._ordinal = 0
        self._closed = False
        self._metadata_lock = threading.RLock()
        self._recorded = set()
        self._metrics = dict(submitted=0, completed=0, hits=0, duplicate_requests=0,
            actual_completed_tasks=0, consumed=0, discarded=0, discarded_completed_tasks=0,
            cancelled_prefetch_requests=0,
            peak_pending_frames=0, peak_charged_bytes=0, errors=0,
            startup_seconds=0.,
            child_cpu_seconds=0., child_decode_wall_seconds=0., child_wall_seconds=0.,
            parent_wait_seconds=0., parent_load_seconds=0.)
        self._write_registry()

    def _write_registry(self):
        with self._metadata_lock:
            _atomic_json(self._registry_path, self._registry)

    def _exchange(self, worker, value):
        process = worker['process']
        if process.poll() is not None:
            raise RuntimeError('CPU frame child exited unexpectedly: %s' % process.returncode)
        try:
            if worker.get('failed'):
                raise RuntimeError('CPU frame child has a prior protocol failure')
            process.stdin.write(json.dumps(value, sort_keys=True)+'\n')
            process.stdin.flush()
            # TextIO.readline can block indefinitely after a partial line.
            # Read bounded raw pipe bytes under one deadline instead.
            deadline = time.monotonic()+self.worker_timeout_seconds
            payload = bytearray()
            while b'\n' not in payload:
                remaining = deadline-time.monotonic()
                if remaining <= 0 or not select.select([process.stdout.fileno()],[],[],remaining)[0]:
                    raise TimeoutError('CPU frame child response exceeded bounded timeout')
                part = os.read(process.stdout.fileno(),65536)
                if not part:
                    raise RuntimeError('CPU frame child closed its response pipe')
                payload.extend(part)
                if len(payload) > 256*2**10:
                    raise ValueError('CPU frame child response exceeds metadata byte limit')
            if payload.count(b'\n') != 1 or not payload.endswith(b'\n'):
                raise ValueError('CPU frame child sent unsolicited response bytes')
            line = payload.decode('utf-8')
        except BaseException:
            worker['failed'] = True
            raise
        result = json.loads(line)
        if 'error' in result:
            raise ValueError('CPU frame child failed: '+result['error'])
        if result.get('cuda_initialized') is not False:
            raise RuntimeError('CPU frame child initialized CUDA')
        identity = result['identity']
        expected = worker.get('identity')
        if expected and any(identity[key] != expected[key] for key in ('pid','uid','starttime','session','pgrp','parent','boot_id','effective_affinity')):
            raise RuntimeError('CPU frame child identity changed')
        return result

    def _start(self):
        if self._executor is not None:
            return
        environment = dict(os.environ, **{name:'1' for name in _ENV})
        environment['CUDA_VISIBLE_DEVICES'] = ''
        environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])+os.pathsep+environment.get('PYTHONPATH','')
        setup = dict(owner=self._owner, source_directory=str(self.container.path),
            handoff_directory=str(self._handoff), source_binding_sha256=self._binding)
        started = time.perf_counter()
        try:
            for index in range(self.processes):
                stderr = (self._root/('child-%d.stderr.log' % index)).open('w')
                os.chmod(stderr.name, 0o600)
                process = subprocess.Popen([sys.executable, '-u', '-m', __name__, '--worker'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                    text=True, bufsize=1, env=environment, start_new_session=False)
                worker = dict(process=process, stderr=stderr)
                self._workers.append(worker)
                # Register process birth before source validation can take time.
                identity = _child_identity(process.pid)
                worker['identity'] = identity
                record = dict(role='cpu_decode', identity=identity,
                    cuda_initialized=False, startup_verified=False,
                    single_thread_environment={name:'1' for name in _ENV}, actual_exit_code=None)
                self._registry['workers'].append(record)
                self._write_registry()
                result = self._exchange(worker, setup)
                if (result.get('ready') is not True or result['source_binding_sha256'] != self._binding
                        or result['identity']['parent'] != self._owner['pid']
                        or result['identity']['effective_affinity'] != self._owner['effective_affinity']):
                    raise RuntimeError('CPU frame child startup binding differs')
                worker['identity'] = result['identity']
                record.update(identity=result['identity'], startup_verified=True,
                    single_thread_environment=result['single_thread_environment'])
                self._available.put(worker)
                self._write_registry()
            self._executor = ThreadPoolExecutor(max_workers=self.processes, thread_name_prefix='cpu-frame-control')
            self._metrics['startup_seconds'] += time.perf_counter()-started
        except BaseException:
            self.close()
            raise

    def _estimate(self, frame_id):
        if frame_id not in self._estimates:
            row = self.container.index[frame_id]
            header = json.loads(store._range(self.container._frame_path/'headers.bin',
                int(row['header_offset']), int(row['header_size'])))
            m, n, _, raw = sparse_codec_fast._header(header['csr_meta'], sparse_codec_fast.DEFAULT_MAX_BYTES)
            if m != int(row['m']) or n != self.container.manifest['n'] or header['counts_raw_bytes'] != m*24:
                raise ValueError('CPU frame estimate geometry differs')
            # Covers decoder validation scratch, compressed input, handoff and
            # parent verification. This is admission accounting, not a hard RSS quota.
            self._estimates[frame_id] = 12*raw + 4*m*24 + int(row['size']) + int(row['counts_size']) + 2**20
        return self._estimates[frame_id]

    def _task(self, ordinal, frame_id):
        worker = self._available.get()
        try:
            result = self._exchange(worker, dict(operation='decode', ordinal=ordinal, frame_id=frame_id))
            self._record_result(result)
            return result
        except BaseException:
            worker['failed'] = True
            raise
        finally:
            self._available.put(worker)

    def prefetch_frames(self, frame_ids):
        if self._closed:
            raise RuntimeError('CPU frame pool is closed')
        for value in frame_ids:
            frame_id = int(value)
            if not 0 <= frame_id < len(self.container.index):
                raise IndexError('Frame ID out of bounds')
            if frame_id in self._pending or frame_id in self._planned:
                self._metrics['duplicate_requests'] += 1
                continue
            self._plan.append(frame_id)
            self._planned.add(frame_id)
        self._fill()

    def _fill(self):
        while self._plan and len(self._pending) < self.pending_limit:
            frame_id = self._plan[0]
            estimate = self._estimate(frame_id)
            if estimate > self.memory_bytes:
                raise MemoryError('CPU frame decode estimate exceeds pool byte budget')
            if self._charged+estimate > self.memory_bytes:
                break
            self._start()
            self._plan.popleft()
            self._planned.remove(frame_id)
            ordinal = self._ordinal
            self._ordinal += 1
            future = self._executor.submit(self._task, ordinal, frame_id)
            self._pending[frame_id] = (future, estimate, ordinal)
            self._charged += estimate
            self._metrics['submitted'] += 1
            self._metrics['peak_pending_frames'] = max(self._metrics['peak_pending_frames'], len(self._pending))
            self._metrics['peak_charged_bytes'] = max(self._metrics['peak_charged_bytes'], self._charged)

    def _load(self, result, ordinal, frame_id):
        if (result.get('ordinal') != ordinal or result.get('frame_id') != frame_id
                or result.get('source_binding_sha256') != self._binding or set(result['arrays']) != set(_ARRAYS)):
            raise ValueError('CPU frame result request/source binding differs')
        row = self.container.index[frame_id]
        if (result['m'],result['n'],result['start']) != (int(row['m']),self.container.manifest['n'],int(row['start'])):
            raise ValueError('CPU frame result geometry differs')
        path = self._handoff/str(ordinal)
        frame = {key:result[key] for key in ('m','n','start')}
        for name in _ARRAYS:
            descriptor = result['arrays'][name]
            if descriptor['filename'] != name+'.npy':
                raise ValueError('CPU frame handoff filename differs')
            source = path/descriptor['filename']
            if source.is_symlink() or store.file_sha(source) != descriptor['sha256']:
                raise ValueError('CPU frame handoff checksum differs')
            value = np.load(source, mmap_mode='r', allow_pickle=False)
            if (value.dtype.str != descriptor['dtype'] or list(value.shape) != descriptor['shape']
                    or value.nbytes != descriptor['nbytes'] or value.flags.writeable):
                raise ValueError('CPU frame handoff array schema differs')
            frame[name] = value
        # Linux mappings retain their inode while the existing bounded consumer
        # compact LRU owns the returned arrays; no persistent cache is published.
        shutil.rmtree(path)
        return frame

    def _record_result(self, result):
        with self._metadata_lock:
            if result['ordinal'] in self._recorded:
                return
            self._recorded.add(result['ordinal'])
            self._metrics['completed'] += 1
            self._metrics['actual_completed_tasks'] += 1
            for name, source in (('child_cpu_seconds','cpu_seconds'), ('child_wall_seconds','wall_seconds'),
                                 ('child_decode_wall_seconds','decode_wall_seconds')):
                self._metrics[name] += result[source]
            for record in self._registry['workers']:
                if record['identity']['pid'] == result['identity']['pid']:
                    record['completed_tasks'] = record.get('completed_tasks',0)+1
                    record['cpu_seconds'] = record.get('cpu_seconds',0.)+result['cpu_seconds']
                    record['last_task_window'] = dict(started_monotonic=result['started_monotonic'],
                        ended_monotonic=result['ended_monotonic'])
            self._write_registry()

    def _discard(self, frame_id):
        """Wait/check a no-longer-needed prefetch before releasing its charge."""
        future, estimate, ordinal = self._pending[frame_id]
        try:
            if not future.cancel():
                result = future.result()  # Failed work must never become silent eviction.
                if (result.get('ordinal') != ordinal or result.get('frame_id') != frame_id
                        or result.get('source_binding_sha256') != self._binding):
                    raise ValueError('Discarded CPU frame result identity differs')
                self._record_result(result)
                self._metrics['discarded_completed_tasks'] += 1
            else:
                self._metrics['cancelled_prefetch_requests'] += 1
            path = self._handoff/str(ordinal)
            if path.exists():
                shutil.rmtree(path)
            self._metrics['discarded'] += 1
        finally:
            self._pending.pop(frame_id)
            self._charged -= estimate

    def read_frame(self, frame_id):
        if self._closed:
            raise RuntimeError('CPU frame pool is closed')
        if frame_id not in self._pending:
            # Direct callers retain demand semantics even if the prior request
            # plan filled the queue, or consumer LRU hits skipped that prefix.
            estimate = self._estimate(frame_id)
            if estimate > self.memory_bytes:
                raise MemoryError('CPU frame decode estimate exceeds pool byte budget')
            while self._pending and (len(self._pending) >= self.pending_limit
                    or self._charged+estimate > self.memory_bytes):
                self._discard(next(iter(self._pending)))
            if frame_id not in self._planned:
                self._plan.appendleft(frame_id)
                self._planned.add(frame_id)
            else:
                self._plan.remove(frame_id)
                self._plan.appendleft(frame_id)
            self._fill()
        if frame_id not in self._pending:
            raise RuntimeError('CPU frame demand dispatch failed after bounded eviction')
        future, estimate, ordinal = self._pending[frame_id]
        started = time.perf_counter()
        try:
            result = future.result()
            self._metrics['parent_wait_seconds'] += time.perf_counter()-started
            started = time.perf_counter()
            value = self._load(result, ordinal, frame_id)
            self._metrics['parent_load_seconds'] += time.perf_counter()-started
            self._record_result(result)
            self._metrics['hits'] += 1
            self._metrics['consumed'] += 1
        except BaseException:
            self._metrics['errors'] += 1
            raise
        finally:
            self._pending.pop(frame_id)
            self._charged -= estimate
        self._fill()
        return value

    def metadata(self):
        with self._metadata_lock:
            metrics = dict(self._metrics)
            workers = json.loads(json.dumps(self._registry['workers']))
        return dict(metrics, requested_processes=self.processes,
            active_processes=sum(worker['process'].poll() is None for worker in self._workers),
            process_start_method='independent-python-exec-no-resource-tracker',
            worker_timeout_seconds=self.worker_timeout_seconds,
            pending_limit=self.pending_limit, pending_frames=len(self._pending),
            planned_frames=len(self._plan), charged_bytes=self._charged, memory_budget_bytes=self.memory_bytes,
            cpu_only=True, ordered_consumption=True, closed=self._closed,
            registry_path=str(self._registry_path), workers=workers,
            memory_contract='conservative scratch and ready handoff admission; child baselines and consumer LRU separate')

    def close(self):
        if self._closed:
            return
        self._closed = True
        failure = None
        try:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
            for future, _, _ in self._pending.values():
                if not future.cancelled():
                    try:
                        self._record_result(future.result())
                        self._metrics['discarded_completed_tasks'] += 1
                    except BaseException as error:
                        failure = failure or error
                else:
                    self._metrics['cancelled_prefetch_requests'] += 1
                self._metrics['discarded'] += 1
            for worker in self._workers:
                process = worker['process']
                try:
                    if process.poll() is None and not worker.get('failed'):
                        self._exchange(worker, dict(operation='close'))
                except BaseException as error:
                    failure = failure or error
                finally:
                    try:
                        if worker.get('failed') and process.poll() is None:
                            process.terminate()
                        try:
                            process.stdin.close()
                        except OSError as error:
                            failure = failure or error
                        try:
                            process.wait(timeout=min(30.,self.worker_timeout_seconds))
                        except subprocess.TimeoutExpired:
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired as error:
                                failure = failure or error
                            failure = failure or RuntimeError('CPU frame child did not exit after close')
                    finally:
                        process.stdout.close()
                        worker['stderr'].close()
                    for record in self._registry['workers']:
                        if record['identity']['pid'] == process.pid:
                            record['actual_exit_code'] = process.returncode
                    if process.returncode != 0:
                        failure = failure or RuntimeError('CPU frame child exited nonzero')
        finally:
            self._pending.clear()
            self._plan.clear()
            self._planned.clear()
            self._charged = 0
            self._registry['pool_shutdown'] = all(worker['process'].poll() is not None for worker in self._workers)
            self._write_registry()
            shutil.rmtree(self._handoff)
        if failure is not None:
            raise failure


if __name__ == '__main__':
    if sys.argv[1:] != ['--worker']:
        raise SystemExit('This module is a private CPU frame worker')
    raise SystemExit(_worker())
