"""Bounded CPU-only output queue with per-tile durable acknowledgments.

A single consumer owns SQLite and all append handles. Scientific producers keep
no CUDA objects or lazy iterators in the queue. Completed job events are returned
only to the caller thread, after all preceding tiles have crossed file fsync and
the FULL SQLite batch commit; the consumer never calls runtime emit callbacks.
"""
from __future__ import annotations
from collections import deque, OrderedDict
from collections.abc import Mapping
from concurrent.futures import Future
from contextvars import copy_context
import json
from pathlib import Path
import sys
import threading
import time
import traceback
import numpy as np

from .phewas_storage import _rows, atomic_json
from .phewas_trait_csv import PerTraitCheckpointCSVWriter, _note


class DurableTileTicket(Future, Mapping):
    """A result is a committed receipt, never an enqueue acknowledgment."""
    def __init__(self, request_flush):
        super().__init__()
        self._request_flush = request_flush
    def __getitem__(self,key):
        return self.result()[key]
    def __iter__(self):
        return iter(self.result())
    def __len__(self):
        return len(self.result())
    def cancel(self):
        # A scientific write cannot be cancelled by discarding its durable
        # acknowledgment. Abort the owning writer explicitly instead.
        return False
    def result(self, timeout=None):
        if not self.done():
            self._request_flush()
        return super().result(timeout=timeout)


class _OversizeTile(Exception):
    pass


def _owned_size(value):
    """Conservative actual Python storage; shared immutable cells overcount."""
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_owned_size(k)+_owned_size(v) for k,v in value.items())
    elif isinstance(value, (list,tuple)):
        size += sum(_owned_size(v) for v in value)
    return size


def _cell_text(value):
    # Preserve csv.writer's exact scalar/None text. Converting NumPy float32
    # through Python float would change its textual precision, so use str on
    # the original CPU scalar. Arrays/tensors/lazy owners never enter the queue.
    if value is None:
        return ''
    if isinstance(value,np.generic):
        return str(value)
    if hasattr(value,'device') or getattr(value,'ndim',0) != 0:
        raise ValueError('asynchronous association cells must be CPU scalars')
    return str(value)


class AsyncPerTraitCheckpointCSVWriter:
    """Batch durable tile IO behind at most ``pending_bytes_limit`` owned bytes.

    ``write_tile``/``write`` return :class:`DurableTileTicket`. ``flush`` is a
    FIFO durability barrier. ``defer_completion`` queues a small job event and
    ``poll_durable_events`` returns only events covered by completed commits.
    Runtime emits these on its own thread. Fresh checkpoint misses avoid an IO
    RPC; a pending duplicate waits for its real durable receipt. Resume queries
    run on the consumer without forcing unrelated pending tiles to commit.

    Oversized tiles take a deliberate synchronous path after the queue drains:
    the original caller owns their existing rows while waiting, and no copied
    oversized payload or GPU/array owner is retained asynchronously.
    """
    checkpoint_tiles = True
    output_layout = 'per_trait'

    def __init__(self,directory,*,binding,pending_bytes_limit=64*2**20,batch_tiles=32,
                 flush_interval_seconds=2.,max_pending_tiles=64,max_deferred_events=256,
                 claim_cache_size=256,consumer_observer=None,_writer_factory=None,**writer_options):
        if (type(pending_bytes_limit) is not int or pending_bytes_limit<4096
                or type(batch_tiles) is not int or not 1<=batch_tiles<=256
                or type(max_pending_tiles) is not int or max_pending_tiles<batch_tiles
                or type(max_deferred_events) is not int or max_deferred_events<1
                or type(claim_cache_size) is not int or claim_cache_size<1
                or (consumer_observer is not None and not callable(consumer_observer))
                or not isinstance(flush_interval_seconds,(int,float))
                or not 0<flush_interval_seconds<=60):
            raise ValueError('invalid asynchronous output byte, count or time bounds')
        self.directory=Path(directory).resolve()
        self.resume=writer_options.get('resume',False)
        self.limit=pending_bytes_limit;self.batch_tiles=batch_tiles
        self.interval=float(flush_interval_seconds);self.max_pending_tiles=max_pending_tiles
        self.max_events=max_deferred_events
        self._claim_cache=OrderedDict();self._claim_cache_size=claim_cache_size
        self._consumer_observer=consumer_observer
        self._condition=threading.Condition();self._producer_lock=threading.Lock()
        self._queue=deque();self._events=deque();self._by_token={};self._by_identity={}
        # Fixed Bloom membership has no false negatives; positives only trigger
        # a consumer ledger lookup. It avoids an unbounded completed-token set
        # without changing fresh repeated-checkpoint semantics.
        self._durable_membership=bytearray(2**20)
        self._pending_bytes=self._pending_tiles=self._event_count=0
        self._flush_requested=False;self._cancel=False;self._closing=False
        self._closed=False;self._fatal=None;self._finished=None;self._backend=None
        self._ready=Future()
        self._stats=dict(pending_owned_bytes_peak=0,pending_tiles_peak=0,
            submitted_tiles=0,durable_tiles=0,durable_batches=0,oversize_synchronous_tiles=0,
            snapshot_seconds=0.,backpressure_seconds=0.,consumer_batch_seconds=0.,
            deferred_events=0,durable_events=0,pending_bytes_limit=pending_bytes_limit,
            batch_tiles_limit=batch_tiles,flush_interval_seconds=self.interval,
            sqlite_connection_owned_by_consumer=True,source_rows_retained_asynchronously=False,
            durable_membership_bytes=len(self._durable_membership),
            claim_cache_bound=claim_cache_size,claim_rpcs=0)
        factory=_writer_factory or PerTraitCheckpointCSVWriter
        context=copy_context()
        self._thread=threading.Thread(target=lambda:context.run(self._consume,factory,binding,writer_options),
                                      name='wgs-durable-output',daemon=False)
        self._thread.start()
        self._ready.result()
        self.worker_id=self._backend.worker_id
        self.binding_sha256=self._backend.binding_sha256

    def _check(self,*,allow_closed=False):
        if self._fatal is not None:
            raise self._fatal
        if (self._closed or self._closing) and not allow_closed:
            raise RuntimeError('asynchronous canonical CSV writer is closed')

    def _request_flush(self):
        with self._condition:
            self._flush_requested=True;self._condition.notify_all()

    def _reserve(self,amount,*,tile_bytes):
        if tile_bytes+amount>self.limit:
            raise _OversizeTile()
        began=time.perf_counter()
        with self._condition:
            self._check()
            while self._pending_bytes+amount>self.limit:
                self._flush_requested=True;self._condition.notify_all()
                self._condition.wait(.1);self._check()
            self._pending_bytes+=amount
            self._stats['pending_owned_bytes_peak']=max(self._stats['pending_owned_bytes_peak'],self._pending_bytes)
            self._stats['backpressure_seconds']+=time.perf_counter()-began

    def _release(self,amount,*,tile=False):
        with self._condition:
            self._pending_bytes-=amount
            if tile:self._pending_tiles-=1
            if self._pending_bytes<0 or self._pending_tiles<0:
                raise RuntimeError('asynchronous output accounting underflow')
            self._condition.notify_all()

    def _new_ticket(self):
        return DurableTileTicket(self._request_flush)

    def _rpc(self,method,*args,**kwargs):
        ticket=self._new_ticket()
        with self._condition:
            self._check();self._queue.append(('rpc',method,args,kwargs,ticket));self._condition.notify_all()
        # RPCs must not request a flush of unrelated pending writes. Only the
        # explicit flush/finish commands are FIFO durability barriers.
        return Future.result(ticket)

    def _observe_consumer(self):
        if self._consumer_observer is not None:
            self._consumer_observer()

    def _ensure_claims(self,traits,chromosome):
        chromosome=str(chromosome)
        with self._condition:
            self._check()
            needed=[int(i) for i in traits if (int(i),chromosome) not in self._claim_cache]
        if needed:
            self._rpc('__claim__',needed,chromosome)
            with self._condition:
                self._stats['claim_rpcs']+=1
                for trait in needed:
                    self._claim_cache[(trait,chromosome)]=True
                    self._claim_cache.move_to_end((trait,chromosome))
                while len(self._claim_cache)>self._claim_cache_size:self._claim_cache.popitem(last=False)

    @staticmethod
    def _identity_key(block_identity,ordinal_before):
        return json.dumps([block_identity,ordinal_before],sort_keys=True,allow_nan=False)

    def _membership_positions(self,token):
        mask=8*len(self._durable_membership)-1
        return [int(token[offset:offset+8],16)&mask for offset in (0,8,16,24)]

    def _known_durable(self,token):
        return all(self._durable_membership[index//8] & (1<<(index%8)) for index in self._membership_positions(token))

    def _mark_durable(self,token):
        for index in self._membership_positions(token):self._durable_membership[index//8] |= 1<<(index%8)

    def checkpoint_tile(self,kind,chromosome,job_index,trait_indices,*,block_index=0,
                        block_identity=None,ordinal_before=None):
        key,token=self._backend._key(kind,chromosome,job_index,trait_indices,block_index)
        self._ensure_claims(key['trait_indices'],key['chromosome'])
        with self._condition:
            self._check();pending=self._by_token.get(token)
            if pending is not None and self._by_identity[token]!=self._identity_key(block_identity,ordinal_before):
                raise ValueError('pending tile scientific identity differs')
            known=self._known_durable(token)
        if pending is not None:
            return pending.result()
        if not self.resume and not known:
            # Same as a fresh backend checkpoint miss; ownership and a repeated
            # committed key are still authenticated by write_tile on consumer.
            return None
        return self._rpc('checkpoint_tile',kind,chromosome,job_index,key['trait_indices'],
            block_index=block_index,block_identity=block_identity,ordinal_before=ordinal_before)

    def _snapshot(self,kind,chromosome,values,options):
        # Metadata is eagerly copied through the existing JSON-compatible
        # scientific contract; neither arbitrary objects nor source owners pass.
        metadata=json.loads(json.dumps(dict(kind=kind,chromosome=str(chromosome),**options),allow_nan=False))
        payload=dict(metadata,trait_results=[])
        owned=_owned_size(payload)+4096
        self._reserve(owned,tile_bytes=0)
        selected=failed=0
        frozen=None
        try:
            for trait,result in values:
                frozen=[];pair=(int(trait),frozen)
                amount=_owned_size(pair)
                self._reserve(amount,tile_bytes=owned);owned+=amount
                old_size=sys.getsizeof(payload['trait_results'])
                payload['trait_results'].append(pair)
                amount=sys.getsizeof(payload['trait_results'])-old_size
                self._reserve(amount,tile_bytes=owned);owned+=amount
                for source in _rows(result):
                    if 'trait_index' in source:
                        raise ValueError('association row cannot override its bound trait index')
                    if any(not isinstance(k,str) or not k for k in source):
                        raise ValueError('association row requires nonempty string column names')
                    source=source if type(source) is dict else dict(**source)
                    row={key:_cell_text(value) for key,value in source.items()}
                    amount=_owned_size(row)
                    self._reserve(amount,tile_bytes=owned);owned+=amount
                    old_size=sys.getsizeof(frozen);frozen.append(row)
                    amount=sys.getsizeof(frozen)-old_size
                    self._reserve(amount,tile_bytes=owned);owned+=amount
                    selected+=int(bool(source.get('spa_selected',False)))
                    failed+=int(bool(source.get('spa_failed',False)))
            if options.get('binary_counts') is None:
                counts=dict(selected=selected,failed=failed)
                amount=_owned_size(counts)+len('binary_counts')+sys.getsizeof(payload)
                self._reserve(amount,tile_bytes=owned);owned+=amount
                payload['binary_counts']=counts
            return payload,owned
        except BaseException:
            # A retained exception must not keep a copied oversized/invalid
            # source snapshot alive while the synchronous fallback waits.
            payload.clear()
            if frozen is not None: frozen.clear()
            row = source = result = values = None
            self._release(owned)
            raise

    def write_tile(self,kind,chromosome,trait_results,*,job_index,block_index=0,
                   block_identity=None,ordinal_before=None,ordinal_after=None,binary_counts=None):
        with self._producer_lock:
            self._check();values=list(trait_results)
            key,token=self._backend._key(kind,chromosome,job_index,[i for i,_ in values],block_index)
            self._ensure_claims(key['trait_indices'],key['chromosome'])
            with self._condition:
                existing=self._by_token.get(token)
                if existing is not None:
                    if self._by_identity[token]!=self._identity_key(block_identity,ordinal_before):
                        raise ValueError('pending tile scientific identity differs')
                    return existing
                while self._pending_tiles>=self.max_pending_tiles:
                    self._flush_requested=True;self._condition.notify_all()
                    self._condition.wait(.1);self._check()
            options=dict(job_index=job_index,block_index=block_index,block_identity=block_identity,
                         ordinal_before=ordinal_before,ordinal_after=ordinal_after,binary_counts=binary_counts)
            began=time.perf_counter()
            try:
                payload,owned=self._snapshot(kind,chromosome,values,options)
            except _OversizeTile:
                # The source remains owned by this waiting caller; draining
                # first prevents a second large pending copy and preserves the
                # mature scalar formatting path for oversized Single tiles.
                self.flush()
                result=self._rpc('write_tile',kind,chromosome,values,**options)
                ticket=self._new_ticket();ticket.set_result(result)
                with self._condition:
                    self._mark_durable(token)
                    self._stats['oversize_synchronous_tiles']+=1
                    self._stats['submitted_tiles']+=1;self._stats['durable_tiles']+=1
                return ticket
            except BaseException as error:
                # Invalid caller cells must not remain reachable through a
                # retained producer traceback (notably a rejected GPU tensor).
                values.clear();trait_results=None
                traceback.clear_frames(error.__traceback__)
                raise
            ticket=self._new_ticket()
            with self._condition:
                try:self._check()
                except BaseException:
                    self._release(owned);raise
                self._pending_tiles+=1;self._by_token[token]=ticket
                self._by_identity[token]=self._identity_key(block_identity,ordinal_before)
                self._stats['pending_tiles_peak']=max(self._stats['pending_tiles_peak'],self._pending_tiles)
                self._stats['submitted_tiles']+=1;self._stats['snapshot_seconds']+=time.perf_counter()-began
                self._queue.append(('tile',payload,owned,token,ticket));self._condition.notify_all()
            return ticket

    def write(self,kind,trait_index,chromosome,result,*,job_index=None,block_index=0,
              block_identity=None,ordinal_before=None,ordinal_after=None):
        return self.write_tile(kind,chromosome,[(trait_index,result)],job_index=job_index,
            block_index=block_index,block_identity=block_identity,ordinal_before=ordinal_before,ordinal_after=ordinal_after)

    def defer_completion(self,event):
        if not isinstance(event,dict):
            raise ValueError('durable completion event must be a dictionary')
        copied=json.loads(json.dumps(event,allow_nan=False));owned=_owned_size(copied)+4096
        with self._condition:
            self._check()
            if self._event_count>=self.max_events:
                raise BufferError('poll durable output events before deferring more jobs')
            self._event_count+=1
        try:self._reserve(owned,tile_bytes=0)
        except BaseException:
            with self._condition:self._event_count-=1
            raise
        ticket=self._new_ticket()
        with self._condition:
            self._queue.append(('event',copied,owned,ticket));self._stats['deferred_events']+=1
            self._condition.notify_all()
        return ticket

    def poll_durable_events(self):
        with self._condition:
            self._check(allow_closed=True);result=[]
            while self._events:
                event,owned=self._events.popleft();result.append(event)
                self._event_count-=1;self._pending_bytes-=owned
            self._condition.notify_all()
            return result

    def flush(self):
        return self._rpc('__flush__')
    drain=flush

    @property
    def row_counts(self):
        if self._closed:
            self._check(allow_closed=True)
            return {} if self._finished is None else dict(self._finished['rows'])
        return self._rpc('__row_counts__')

    def profile(self):
        with self._condition:
            return dict(self._stats,pending_owned_bytes=self._pending_bytes,
                        pending_tiles=self._pending_tiles,pending_durable_events=len(self._events))

    def finish(self):
        if self._finished is not None:return self._finished
        result=self._rpc('__finish__')
        self._thread.join();self._finished=result
        return result

    def abort(self):
        with self._condition:
            if self._closed:
                self._check(allow_closed=True);return
            self._cancel=True;self._closing=True;self._condition.notify_all()
        self._thread.join()
        self._check(allow_closed=True)
    close=abort

    def _consume(self,factory,binding,options):
        pending=[];events=[];deadline=None
        def publish_events():
            while events:
                event,owned,ticket=events.pop(0)
                with self._condition:
                    self._events.append((event,owned));self._stats['durable_events']+=1
                ticket.set_result(event)
        def flush_pending():
            nonlocal deadline
            if pending:
                began=time.perf_counter()
                receipts=self._backend.write_tiles([item[1] for item in pending])
                self._observe_consumer()
                if len(receipts)!=len(pending):
                    raise RuntimeError('durable output batch acknowledgment count differs')
                for command,receipt in zip(pending,receipts):
                    _,payload,owned,token,ticket=command
                    with self._condition:
                        self._by_token.pop(token,None);self._by_identity.pop(token,None)
                        self._mark_durable(token);self._stats['durable_tiles']+=1
                    ticket.set_result(receipt);self._release(owned,tile=True)
                pending.clear()
                with self._condition:
                    self._stats['durable_batches']+=1;self._stats['consumer_batch_seconds']+=time.perf_counter()-began
            publish_events();deadline=None
        command=None
        try:
            self._backend=factory(self.directory,binding=binding,**options)
            original_claim=self._backend._claim
            def observed_claim(trait,chromosome):
                recovered=trait if self._backend.ownership_scope=='trait' else (trait,str(chromosome))
                first=recovered not in self._backend._recovered_traits
                result=original_claim(trait,chromosome)
                if first:self._observe_consumer()
                return result
            self._backend._claim=observed_claim
            self._observe_consumer()
            self._ready.set_result(True)
            while True:
                with self._condition:
                    while True:
                        if self._cancel:break
                        due=pending and (self._flush_requested or time.monotonic()>=deadline)
                        if due:
                            command=('timed_flush',);self._flush_requested=False;break
                        if self._queue:
                            command=self._queue.popleft();break
                        timeout=max(0,deadline-time.monotonic()) if pending else None
                        self._condition.wait(timeout)
                    if self._cancel:break
                if command[0]=='timed_flush':flush_pending();continue
                if command[0]=='tile':
                    pending.append(command)
                    if deadline is None:deadline=time.monotonic()+self.interval
                    if len(pending)>=self.batch_tiles:flush_pending()
                elif command[0]=='event':
                    _,event,owned,ticket=command;events.append((event,owned,ticket))
                    if not pending:publish_events()
                else:
                    _,method,args,kwargs,ticket=command
                    if method=='__flush__':
                        flush_pending();self._observe_consumer();ticket.set_result(None)
                    elif method=='__row_counts__':
                        result=self._backend.row_counts;self._observe_consumer();ticket.set_result(result)
                    elif method=='__claim__':
                        for trait in args[0]:self._backend._claim(trait,args[1])
                        self._observe_consumer();ticket.set_result(None)
                    elif method=='__finish__':
                        flush_pending();result=self._backend.finish()
                        result['asynchronous_output']=self.profile()
                        atomic_json(self.directory/f'worker_{self._backend.worker_id}.complete.private.json',result)
                        ticket.set_result(result);break
                    else:
                        result=getattr(self._backend,method)(*args,**kwargs)
                        self._observe_consumer();ticket.set_result(result)
                command=None
            if self._cancel:self._backend.abort()
        except BaseException as error:
            with self._condition:self._fatal=error
            if not self._ready.done():self._ready.set_exception(error)
        finally:
            # One failure reaches every waiting producer/ticket. Never mark an
            # uncommitted queue item successful, including a deferred job event.
            with self._condition:
                remaining=list(pending)+list(self._queue)
                self._queue.clear()
                if command is not None and not any(item is command for item in remaining):remaining.append(command)
                failure=self._fatal or RuntimeError('asynchronous output was aborted before durability')
                for item in remaining:
                    if item[0]=='tile':
                        _,payload,owned,token,ticket=item
                        if token in self._by_token:
                            self._pending_bytes-=owned;self._pending_tiles-=1;self._by_token.pop(token,None);self._by_identity.pop(token,None)
                    elif item[0]=='event':
                        _,event,owned,ticket=item
                        self._pending_bytes-=owned;self._event_count-=1
                    elif item[0]=='rpc':ticket=item[-1]
                    else:continue
                    if not ticket.done():ticket.set_exception(failure)
                for event,owned,ticket in events:
                    self._pending_bytes-=owned;self._event_count-=1
                    if not ticket.done():ticket.set_exception(failure)
                pending.clear();events.clear();remaining.clear()
                command = item = payload = event = ticket = None
                self._closed=True;self._condition.notify_all()
            if self._backend is not None and not self._backend._closed:
                try:self._backend.abort()
                except BaseException as cleanup:
                    if self._fatal is not None:_note(self._fatal,f'output consumer cleanup also failed: {type(cleanup).__name__}: {cleanup}')
                    else:
                        with self._condition:self._fatal=cleanup
            if self._fatal is not None:
                traceback.clear_frames(self._fatal.__traceback__)
