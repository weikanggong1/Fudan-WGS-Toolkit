"""Bounded pipeline-owned streams for independent original Torch LOBPCG.

SPDX-License-Identifier: GPL-3.0-only
The installed 2.5.1 worker supplies every numerical operation. This adapter
only gives independent operators two persistent control threads and streams.
No operator stack, projected-eigh batching or new convergence rule is used.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import importlib
import threading
import traceback
import weakref

import torch

from ._fit_structure_cache import available_cuda_bytes, guard_cuda_workspace


def clear_exception_frames(error):
    pending, seen = [error], set()
    current = linked = None
    try:
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            traceback.clear_frames(current.__traceback__)
            for linked in (current.__cause__, current.__context__):
                if linked is not None:
                    pending.append(linked)
    finally:
        pending.clear()
        current = linked = error = None


def close_preserving_error(close, primary=None):
    """Release control owners; retain the original computation error priority."""
    try:
        close()
    except BaseException as cleanup:
        clear_exception_frames(cleanup)
        if primary is None:
            raise
        if hasattr(primary, 'add_note'):
            primary.add_note('LOBPCG owner cleanup also failed: '+type(cleanup).__name__)
    finally:
        close = primary = None


def workspace_estimate(m, k, workers):
    """NEW solver panels/scratch; exclude already resident independent A/X."""
    if any(type(v) is not int or v <= 0 for v in (m, k, workers)) or m < 3*k:
        raise ValueError('LOBPCG requires positive geometry and m >= 3*k')
    return dict(required_bytes=workers*(48*m*k + 400*k*k + 16*k),
                maximum_projected_size=3*k, whole_operator_stack_bytes=0,
                resident_operators_and_start_vectors_excluded=True,
                scratch_is_estimated_not_reserved=True)


class LOBPCGWorkerPool:
    """Exactly one persistent control thread for each persistent CUDA stream.

    Explicitly own this pool around a pipeline, never one mask. Thread-local
    native handles can retain allocator scratch after Python owners close;
    measured retained bytes are reported, not claimed released by shutdown.
    No matrix, Future, model or input event is cached between evaluations.
    """
    def __init__(self, max_workers=2):
        if type(max_workers) is not int or not 1 <= max_workers <= 2:
            raise ValueError('LOBPCG stream pool requires 1 to 2 threads')
        self.max_workers = max_workers
        self._executors = [ThreadPoolExecutor(max_workers=1,
            thread_name_prefix='wgs-lobpcg-slot-'+str(slot))
            for slot in range(max_workers)]
        self._lock = threading.RLock()
        self._closed = False
        self._device = None
        self._streams = []
        self._warmed = False
        self._allocator_warmup = None
        self._close_allocator = None
        self._threads = set()
        self._thread_refs = weakref.WeakSet()
        self._active_workers = 0
        self._active_high_water = 0
        self._workspace_leases = 0
        self._new_high_water = 0
        self._streams_created = 0
        self._close_errors = []

    def initial_native_workspace_estimate(self, device):
        with self._lock:
            if self._device is not None and self._device != str(device):
                raise ValueError('LOBPCG stream pool is bound to one CUDA device')
            # Two fixed (thread, stream) handle pairs, conservatively reserve
            # 32 MiB each before first use. This is NEW, not resident scratch.
            return 0 if self._warmed else self.max_workers*(32 << 20)

    def streams(self, device):
        with self._lock:
            if self._closed:
                raise RuntimeError('LOBPCG stream pool is closed')
            if self._device is not None and self._device != str(device):
                raise ValueError('LOBPCG stream pool is bound to one CUDA device')
            if not self._streams:
                with torch.cuda.device(device):
                    self._streams = [torch.cuda.Stream(device=device)
                                     for _ in range(self.max_workers)]
                self._device = str(device)
                self._streams_created += len(self._streams)
            return tuple(self._streams)

    def submit_slot(self, slot, function, *args):
        with self._lock:
            if self._closed:
                raise RuntimeError('LOBPCG stream pool is closed')
            return self._executors[slot].submit(function, *args)

    def note_thread(self):
        with self._lock:
            self._threads.add(threading.get_ident())
            self._thread_refs.add(threading.current_thread())

    def begin_worker(self):
        with self._lock:
            self._active_workers += 1
            self._active_high_water = max(self._active_high_water, self._active_workers)

    def end_worker(self):
        with self._lock:
            self._active_workers -= 1

    def begin_workspace(self, new_bytes):
        with self._lock:
            self._workspace_leases += 1
            self._new_high_water = max(self._new_high_water, int(new_bytes))

    def end_workspace(self):
        with self._lock:
            self._workspace_leases -= 1

    def note_warmup(self, device, before):
        with self._lock:
            if not self._warmed:
                self._warmed = True
                self._allocator_warmup = dict(
                    allocated_delta_bytes=int(torch.cuda.memory_allocated(device))-before[0],
                    reserved_delta_bytes=int(torch.cuda.memory_reserved(device))-before[1],
                    native_and_tensor_peak_are_not_separable=True)

    def metadata(self):
        with self._lock:
            return dict(max_workers=self.max_workers, persistent_threads=True,
                owns_cuda_streams=True, stream_count=len(self._streams),
                streams_created_at_most_once=True, streams_created_total=self._streams_created,
                observed_thread_count=len(self._threads),
                live_control_thread_count=sum(thread.is_alive() for thread in self._thread_refs),
                active_worker_count=self._active_workers,
                active_worker_high_water=self._active_high_water,
                active_solver_workspace_leases=self._workspace_leases,
                solver_new_peak_estimate_high_water_bytes=self._new_high_water,
                closed=self._closed, first_tile_allocator_warmup=self._allocator_warmup,
                after_owner_close_allocator=self._close_allocator,
                cleanup_error_types=list(self._close_errors),
                native_handle_cache_may_survive_owner_close=True)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            executors, self._executors = self._executors, []
        streams = first_error = executor = stream = None
        try:
            for executor in executors:
                try:
                    executor.shutdown(wait=True, cancel_futures=True)
                except BaseException as error:
                    clear_exception_frames(error)
                    self._close_errors.append(type(error).__name__)
                    if first_error is None:
                        first_error = error
            with self._lock:
                streams, self._streams = self._streams, []
            for stream in streams:
                try:
                    stream.synchronize()
                except BaseException as error:
                    clear_exception_frames(error)
                    self._close_errors.append(type(error).__name__)
                    if first_error is None:
                        first_error = error
            if streams:
                try:
                    self._close_allocator = dict(
                        allocated_bytes=int(torch.cuda.memory_allocated(streams[0].device)),
                        reserved_bytes=int(torch.cuda.memory_reserved(streams[0].device)),
                        cuda_driver_free_bytes=int(torch.cuda.mem_get_info(streams[0].device)[0]))
                except BaseException as error:
                    clear_exception_frames(error)
                    self._close_errors.append(type(error).__name__)
                    if first_error is None:
                        first_error = error
            if first_error is not None:
                raise first_error
        finally:
            executors.clear()
            if streams is not None:
                streams.clear()
            executors = streams = executor = stream = first_error = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        close_preserving_error(self.close, exc)


def _installed_worker():
    """Subclass the installed algorithm; isolate tracker without global patch."""
    module = importlib.import_module('torch._lobpcg')
    class _LocalWorker(module.LOBPCG):
        def call_tracker(self):
            if self.tracker is not None:
                self.tracker(self)
    return _LocalWorker


class _IndependentTile:
    def __init__(self, operators, starts, trackers, k, niter, tol, largest, pool):
        self.operators, self.starts, self.trackers = operators, starts, trackers
        self.k, self.niter, self.tol, self.largest = k, niter, tol, largest
        self.pool = pool
        self.device = operators[0].device
        self.main_stream = torch.cuda.current_stream(self.device)
        self.streams = pool.streams(self.device)
        self.input_ready = torch.cuda.Event()
        self.input_ready.record(self.main_stream)
        self.grad_enabled = torch.is_grad_enabled()
        self.inference_enabled = torch.is_inference_mode_enabled()
        self.autocast_enabled = torch.is_autocast_enabled('cuda')
        self.autocast_dtype = torch.get_autocast_dtype('cuda')

    def run_worker(self, slot):
        matrix = initial = worker = cls = result = completion = stream = None
        iparams = fparams = bparams = error = None
        try:
            self.pool.note_thread()
            self.pool.begin_worker()
            matrix, initial = self.operators[slot], self.starts[slot]
            stream = self.streams[slot]
            cls = _installed_worker()
            iparams = dict(m=initial.shape[0], n=initial.shape[1], k=self.k,
                           niter=self.niter, ortho_i_max=3, ortho_j_max=3)
            fparams = dict(tol=self.tol, ortho_tol=self.tol,
                           ortho_tol_drop=self.tol, ortho_tol_replace=self.tol)
            bparams = dict(largest=self.largest, ortho_use_drop=False)
            with (torch.cuda.device(self.device), torch.cuda.stream(stream),
                  torch.set_grad_enabled(self.grad_enabled),
                  torch.inference_mode(self.inference_enabled),
                  torch.autocast('cuda', enabled=self.autocast_enabled,
                                 dtype=self.autocast_dtype)):
                stream.wait_event(self.input_ready)
                matrix.record_stream(stream)
                initial.record_stream(stream)
                worker = cls(matrix, None, initial, None, iparams, fparams,
                             bparams, 'ortho', self.trackers[slot])
                worker.run()
                result = (worker.E[:self.k], worker.X[:, :self.k])
                completion = torch.cuda.Event()
                completion.record(stream)
            return result, completion
        except BaseException as error:
            clear_exception_frames(error)
            raise
        finally:
            self.pool.end_worker()
            if worker is not None:
                worker.A = worker.B = worker.X = worker.E = worker.R = worker.S = worker.iK = None
                worker.tvars.clear()
                worker.tracker = None
            matrix = initial = worker = cls = result = completion = stream = None
            iparams = fparams = bparams = error = None

    def run(self):
        futures, output, errors = [], [], []
        value = pair = event = error = context = future = tensor = None
        try:
            for slot in range(len(self.operators)):
                context = copy_context()
                futures.append(self.pool.submit_slot(slot, context.run, self.run_worker, slot))
            # Always join both slots, then raise in original operator order.
            # A completed exception Future may be held by an auditor, but its
            # numerical frames and tile owners must already have been cleared.
            for future in futures:
                try:
                    value = future.result()
                    output.append(value)
                    value = None
                except BaseException as error:
                    errors.append(error)
                    output.append(None)
            if errors:
                raise errors[0]
            pairs = []
            for pair, event in output:
                self.main_stream.wait_event(event)
                for tensor in pair:
                    tensor.record_stream(self.main_stream)
                pairs.append(pair)
            return pairs
        except BaseException as error:
            clear_exception_frames(error)
            raise
        finally:
            for error in errors:
                clear_exception_frames(error)
            futures.clear()
            output.clear()
            errors.clear()
            self.operators = self.starts = self.trackers = self.pool = None
            self.streams = self.input_ready = self.main_stream = None
            futures = output = errors = value = pair = event = error = None
            context = future = tensor = pairs = None


def independent_lobpcg(matrices, initial_vectors, *, k, niter=200, tol=1e-5,
                        largest=True, trackers=None, max_active_workers=2,
                        max_workspace_bytes=None, worker_pool=None):
    """Original independent solves; fixed-stream execution needs explicit pool.

    A/X are already resident. The caller separately budgets constructing all
    lookahead operators and buffered eigenpairs before invoking this helper.
    Unsupported input, no pool or insufficient workspace uses original Torch.
    """
    operators = starts = records = output = pair = tile = pool = None
    matrix = initial = tracker = before = None
    try:
        operators, starts = tuple(matrices), tuple(initial_vectors)
        if not operators or len(operators) != len(starts):
            raise ValueError('LOBPCG requires matching nonempty operator/start sequences')
        if type(k) is not int or k <= 0 or type(max_active_workers) is not int or max_active_workers <= 0:
            raise ValueError('LOBPCG k and worker tile must be positive integers')
        if trackers is None:
            trackers = [None]*len(operators)
        if len(trackers) != len(operators):
            raise ValueError('LOBPCG tracker sequence must match operators')
        m = operators[0].shape[0]
        for matrix, initial in zip(operators, starts):
            if (matrix.shape != (m, m) or initial.shape != (m, k)
                    or matrix.device != operators[0].device or initial.device != matrix.device
                    or matrix.dtype != torch.float32 or initial.dtype != matrix.dtype
                    or matrix.layout != torch.strided or initial.layout != torch.strided
                    or matrix.requires_grad or initial.requires_grad):
                raise ValueError('LOBPCG operators/starts require independent dense FP32 state')
        if max_workspace_bytes is not None and (type(max_workspace_bytes) is not int or max_workspace_bytes < 0):
            raise ValueError('LOBPCG workspace must be nonnegative bytes')
        if worker_pool is not None and not isinstance(worker_pool, LOBPCGWorkerPool):
            raise ValueError('LOBPCG pool must be an explicitly owned LOBPCGWorkerPool')
        pool = worker_pool
        supported = (str(torch.__version__).split('+')[0] == '2.5.1'
                     and operators[0].is_cuda and niter > 0 and tol > 0 and pool is not None)
        output, records = [], []
        offset = 0
        while offset < len(operators):
            width = 1
            if supported:
                available = available_cuda_bytes(operators[0].device)
                if max_workspace_bytes is not None:
                    available = min(available, max_workspace_bytes)
                native = pool.initial_native_workspace_estimate(operators[0].device)
                unit = workspace_estimate(m, k, 1)['required_bytes']
                width = min(len(operators)-offset, max_active_workers, pool.max_workers,
                            max(0, available-native)//unit)
            if width < 2:
                matrix, initial = operators[offset], starts[offset]
                pair = torch.lobpcg(matrix, k=k, B=None, X=initial, largest=largest,
                                   niter=niter, tol=tol, tracker=trackers[offset])
                output.append(pair)
                records.append(dict(route='original_torch_sequential', workers=1))
                offset += 1
                continue
            estimate = workspace_estimate(m, k, width)
            native = pool.initial_native_workspace_estimate(operators[0].device)
            guard_cuda_workspace(estimate['required_bytes']+native, operators[0].device,
                                 context='fixed-stream original LOBPCG tile NEW workspace')
            before = (int(torch.cuda.memory_allocated(operators[0].device)),
                      int(torch.cuda.memory_reserved(operators[0].device)))
            tile = _IndependentTile(operators[offset:offset+width], starts[offset:offset+width],
                                   trackers[offset:offset+width], k, niter, tol, largest, pool)
            pool.begin_workspace(estimate['required_bytes']+native)
            try:
                pair = tile.run()
            finally:
                pool.end_workspace()
            # Failure before the first native operation must not consume the
            # first-use NEW allowance. It is safe to keep it until success.
            pool.note_warmup(operators[0].device, before)
            output.extend(pair)
            records.append(dict(route='installed_torch_fixed_streams', workers=width,
                                estimated_new_bytes=estimate['required_bytes'],
                                initial_native_workspace_new_bytes=native))
            tile = pair = None
            offset += width
        return output, dict(tiles=records, whole_operator_stack_created=False,
            original_start_vectors_supplied=True, no_torch_global_patch=True,
            original_convergence_and_orthogonalization_rules=True,
            original_projected_eigh_per_worker=True,
            borrowed_installed_worker_version='2.5.1',
            worker_pool=pool.metadata() if pool is not None else None,
            actual_routes=[record['route'] for record in records])
    finally:
        matrices = initial_vectors = trackers = operators = starts = records = output = None
        pair = tile = pool = matrix = initial = tracker = before = worker_pool = None
