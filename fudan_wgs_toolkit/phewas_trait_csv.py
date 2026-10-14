"""Canonical per-trait CSVs with durable, scientific tile checkpoints.

Each CSV is a concatenation of complete gzip members. A short SQLite transaction
commits the member ranges of one scientific tile only after every range has been
flushed and fsynced. The database contains metadata, never association arrays.
Uncommitted suffixes are preserved privately and truncated before resumption.
Corrupt committed ranges fail closed; they are never mistaken for missing work.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import csv
import fcntl
import gzip
import hashlib
import io
import json
import operator
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
import uuid

from .phewas_storage import _rows, atomic_json
from .csv_output import _write_bound_trait_rows

_KINDS = ('individual', 'coding', 'noncoding', 'ncrna')
_INVOCATION_LEASE = ContextVar('wgs_association_invocation_lease', default=None)


def _restore_invocation_lease(handle):
    return handle


class _InvocationLeaseTransfer:
    """Defer FD transfer until Process.start installs its spawn context.

    DupFd outside that context registers a duplicated descriptor with the
    resource sharer. An early child death before detach could then leave the
    invocation locked by that parent-side registration. The actual spawn
    pass_fds route holds the same open-file description in the child directly.
    This carrier owns no extra descriptor in its parent.
    """
    def __init__(self, descriptor):
        self._descriptor=descriptor

    def __reduce__(self):
        from multiprocessing.context import get_spawning_popen
        from multiprocessing.reduction import DupFd
        if self._descriptor is None:
            raise ValueError('spawn invocation lease transfer has been retired')
        if get_spawning_popen() is None:
            raise RuntimeError('invocation lease can only be serialized by Process.start')
        os.fstat(self._descriptor)
        return (_restore_invocation_lease,(DupFd(self._descriptor),))

    def detach(self):
        # Direct in-process/fork execution needs its own closeable descriptor;
        # normal spawn receives a restored Popen DupFd instead of this carrier.
        if self._descriptor is None:
            raise ValueError('spawn invocation lease transfer has been retired')
        descriptor=os.dup(self._descriptor)
        self._descriptor=None
        return descriptor

    def retire(self):
        self._descriptor=None


def invocation_lease_for_spawn():
    """Transfer the active lock's open-file description to one spawn child.

    This is runtime state, passed in Process kwargs rather than a deployment
    plan. A child detaches it before scientific work and keeps the same flock
    alive if its owning API parent exits unexpectedly.
    """
    descriptor=_INVOCATION_LEASE.get()
    if descriptor is None:
        return None
    return _InvocationLeaseTransfer(descriptor)


def close_pending_invocation_lease(handle):
    """Retire an unstarted carrier without closing the parent invocation FD."""
    if handle is not None:
        if not isinstance(handle,_InvocationLeaseTransfer):
            raise ValueError('pending invocation lease must be a parent transfer carrier')
        handle.retire()


@contextmanager
def held_invocation_lease(handle):
    """Hold a spawn-transferred invocation descriptor through worker cleanup."""
    if handle is None:
        yield
        return
    if not callable(getattr(handle,'detach',None)):
        raise ValueError('spawn invocation lease must be a transferable descriptor')
    descriptor=handle.detach()
    token=None
    try:
        if type(descriptor) is not int:
            raise ValueError('spawn invocation lease did not detach a file descriptor')
        os.fstat(descriptor)
        token=_INVOCATION_LEASE.set(descriptor)
        yield
    finally:
        if token is not None:
            _INVOCATION_LEASE.reset(token)
        # Closing this duplicate does not unlock the parent's/peers' shared
        # open-file description. Never use LOCK_UN on an inherited flock.
        if type(descriptor) is int:
            os.close(descriptor)


@contextmanager
def association_output_lease(output_directory):
    """Claim one public invocation before preparation or resume changes files.

    The sibling lock leaves a fresh output directory empty/nonexistent. All
    worker children also receive this lease before scientific work, then use
    their chromosome/trait locks. A parent's abrupt exit cannot release the
    output for resume until the last active owned worker closes its duplicate.
    """
    output=Path(output_directory).resolve()
    if not output.name:
        raise ValueError('association output must be a named directory')
    output.parent.mkdir(parents=True,exist_ok=True)
    # Keep long valid output names below the filesystem filename limit while
    # retaining a unique resolved-path binding. No identities are written.
    stem=output.name if len(os.fsencode(output.name))<=120 else hashlib.sha256(
        str(output).encode()).hexdigest()
    target=output.parent/f'.{stem}.wgs-association.lock'
    if target.is_symlink():
        raise ValueError('association invocation lock must not be a symlink')
    descriptor=os.open(target,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    token=None
    try:
        try:
            fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('association output already has an active public invocation') from error
        token=_INVOCATION_LEASE.set(descriptor)
        yield descriptor
    finally:
        if token is not None:
            _INVOCATION_LEASE.reset(token)
        os.close(descriptor)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _identity(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _nonnegative(value):
    return type(value) is int and value >= 0


def _note(error, message):
    if hasattr(error, 'add_note'):
        error.add_note(message)
    else:  # Python 3.10 also retains the original exception and cleanup detail.
        error.__notes__ = [*getattr(error, '__notes__', ()), message]


class CommittedCSVCorruption(RuntimeError):
    """A committed scientific range changed; retain evidence and stop recovery."""


class _AppendHash:
    def __init__(self, raw):
        self.raw, self.digest, self.size = raw, hashlib.sha256(), 0
    def write(self, data):
        count = self.raw.write(data)
        if count != len(data):
            raise OSError('short canonical CSV append')
        self.digest.update(data); self.size += count
        return count
    def flush(self):
        self.raw.flush()
    def tell(self):
        return self.raw.tell()


class _CheckedSlice(io.RawIOBase):
    """Read a bounded member/file while authenticating each committed range."""
    def __init__(self, path, start, end, ranges):
        self.raw = path.open('rb')
        self.raw.seek(start)
        self.position, self.end = start, end
        self.ranges, self.current = iter(ranges), None
        self.range_digest, self.digest = hashlib.sha256(), hashlib.sha256()
        try:
            self._next_range()
        except BaseException:
            self.raw.close()
            raise
    def _next_range(self):
        item = next(self.ranges, None)
        if item is None:
            if self.position != self.end:
                raise CommittedCSVCorruption('committed CSV range coverage is incomplete')
            self.current = None
            return
        if item['start'] != self.position or item['end'] <= item['start'] or item['end'] > self.end:
            raise CommittedCSVCorruption('committed CSV ranges overlap or contain a gap')
        self.current, self.range_digest = item, hashlib.sha256()
    def readable(self):
        return True
    def readinto(self, buffer):
        if self.position == self.end:
            return 0
        count = min(len(buffer), self.current['end']-self.position)
        actual = self.raw.readinto(memoryview(buffer)[:count])
        if not actual:
            raise CommittedCSVCorruption('committed CSV range was truncated')
        data = memoryview(buffer)[:actual]
        self.digest.update(data); self.range_digest.update(data)
        self.position += actual
        if self.position == self.current['end']:
            if self.range_digest.hexdigest() != self.current['sha256']:
                raise CommittedCSVCorruption('committed CSV range checksum differs')
            self._next_range()
        return actual
    def close(self):
        try:
            self.raw.close()
        finally:
            super().close()


class PerTraitCheckpointCSVWriter:
    """Bounded append writer; each trait owns real CSVs and durable tile ranges.

    Paths are ``traits/trait_0000/kind/chr21/schema/results.csv.gz``. Index
    directories avoid exposing phenotype names and are resolved by the private
    trait dictionary. ``binding`` has the same source/input/configuration/model
    contract as the immutable tile writer. Worker/GPU placement is excluded.
    At most ``max_open_streams`` append handles and ``verification_cache_size``
    verified tile records are retained. Active ownership uses OS locks, which
    are released automatically after a crash and explicitly on close. The
    continuous chromosome dispatch uses one lock per chromosome, so different
    workers may append the same traits on disjoint chromosomes safely without
    retaining a lock for every trait/chromosome pair.
    """
    checkpoint_tiles = True
    output_layout = 'per_trait'

    def __init__(self, directory, *, binding, resume=False, worker_id='0',
                 minimum_free_bytes=0, max_open_streams=48, verification_cache_size=256,
                 ownership_scope='trait'):
        required = {'source_implementation_sha256', 'input_manifest_sha256',
                    'configuration_sha256', 'models'}
        if (type(resume) is not bool or not isinstance(binding, dict) or not required.issubset(binding)
                or any(not isinstance(binding[name], str) or not binding[name] for name in required-{'models'})
                or not isinstance(binding['models'], (dict, list)) or not binding['models']):
            raise ValueError('checkpoint binding requires source, input, schedule and model hashes; resume must be boolean')
        if (not _nonnegative(minimum_free_bytes) or type(max_open_streams) is not int or max_open_streams < 1
                or type(verification_cache_size) is not int or verification_cache_size < 1):
            raise ValueError('invalid CSV reserve, stream or verification-cache bound')
        if ownership_scope not in ('trait', 'chromosome'):
            raise ValueError('CSV ownership_scope must be trait or chromosome')
        worker_id = str(worker_id)
        if not worker_id or Path(worker_id).name != worker_id or worker_id in ('.', '..') or '\\' in worker_id:
            raise ValueError('checkpoint worker ID must be a single safe filename component')
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.root = self.directory/'.trait_checkpoints'
        self.root.mkdir(exist_ok=True)
        if self.root.is_symlink():
            raise ValueError('checkpoint directory must not be a symlink')
        self.worker_id, self.resume = worker_id, resume
        self.ownership_scope = ownership_scope
        self.binding = json.loads(_canonical(binding))
        self.binding_sha256 = _sha(self.binding)
        self.minimum_free_bytes, self.max_open = minimum_free_bytes, max_open_streams
        self.verification_cache_size = verification_cache_size
        self.invocation = uuid.uuid4().hex
        self.streams, self._verified, self._owners = OrderedDict(), OrderedDict(), {}
        # Only handles with unsynchronized bytes need a durability barrier at
        # eviction/close. Successful member appends have already crossed it.
        self._dirty_streams = set()
        self._batch = None
        self.durability_profile = dict(data_fsync_calls=0, tile_commit_transactions=0,
            registration_transactions=0, batch_calls=0, batch_tiles=0, maximum_batch_tiles=0)
        self._recovered_traits = set()
        self._owned_traits = set()
        self.reused_blocks = self.computed_blocks = 0
        self.write_seconds = 0.
        self._closed, self._finished = False, None
        self.database_path = self.root/'ranges.sqlite'
        if self.database_path.is_symlink():
            raise ValueError('checkpoint database must not be a symlink')
        self.db = sqlite3.connect(self.database_path, timeout=60, isolation_level=None)
        try:
            self.db.execute('PRAGMA busy_timeout=60000')
            # Mode/schema establishment is a short separate owner operation.
            # Concurrent PRAGMA journal_mode upgrades can fail immediately on
            # shared filesystems even with SQLite's busy timeout configured.
            lock_path = self.root/'initialization.lock'
            if lock_path.is_symlink():
                raise ValueError('checkpoint initialization lock must not be a symlink')
            descriptor = os.open(lock_path, os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                mode = self.db.execute('PRAGMA journal_mode').fetchone()[0]
                if mode != 'wal':
                    mode = self.db.execute('PRAGMA journal_mode=WAL').fetchone()[0]
                if mode != 'wal':
                    raise OSError('checkpoint filesystem cannot establish SQLite WAL')
                self.db.execute('PRAGMA synchronous=FULL')
                self._initialize()
            finally:
                os.close(descriptor)
        except BaseException:
            self.db.close(); self._closed = True
            raise

    def _initialize(self):
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS files(
              id TEXT PRIMARY KEY,trait INTEGER NOT NULL,kind TEXT NOT NULL,chromosome TEXT NOT NULL,
              relative_path TEXT UNIQUE NOT NULL,columns_json TEXT NOT NULL,
              committed_end INTEGER NOT NULL DEFAULT 0,rows INTEGER NOT NULL DEFAULT 0,
              first_job INTEGER,last_job INTEGER);
            CREATE INDEX IF NOT EXISTS files_trait ON files(trait);
            CREATE TABLE IF NOT EXISTS blocks(
              token TEXT PRIMARY KEY,writer_id TEXT NOT NULL,invocation TEXT NOT NULL,
              kind TEXT NOT NULL,rows INTEGER NOT NULL,record_json TEXT NOT NULL,receipt_sha256 TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS blocks_writer ON blocks(writer_id);
            CREATE TABLE IF NOT EXISTS ranges(
              block_token TEXT NOT NULL,ordinal INTEGER NOT NULL,file_id TEXT NOT NULL,
              start INTEGER NOT NULL,end INTEGER NOT NULL,rows INTEGER NOT NULL,header INTEGER NOT NULL,
              sha256 TEXT NOT NULL,PRIMARY KEY(block_token,ordinal),UNIQUE(file_id,start));
            CREATE INDEX IF NOT EXISTS ranges_file ON ranges(file_id,start);
            CREATE TABLE IF NOT EXISTS visits(
              invocation TEXT NOT NULL,block_token TEXT NOT NULL,reused INTEGER NOT NULL,
              PRIMARY KEY(invocation,block_token));
        ''')
        contract = dict(schema_version=1, output_layout=self.output_layout,
            binding=self.binding, binding_sha256=self.binding_sha256)
        # Preserve the original trait-owner ledger bytes. Explicit chromosome
        # mode cannot share a ledger with a concurrently active legacy writer.
        if self.ownership_scope == 'chromosome':
            contract['ownership_scope'] = 'chromosome'
        expected = _canonical(contract)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            prior = self.db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
            if prior is None:
                self.db.execute("INSERT INTO metadata VALUES('binding',?)", (expected,))
            elif prior[0] != expected:
                raise ValueError('resume checkpoint source/input/schedule/model binding differs')
            if not self.resume and self.db.execute('SELECT 1 FROM blocks WHERE writer_id=? LIMIT 1', (self.worker_id,)).fetchone():
                raise FileExistsError('previous association checkpoints exist; choose resume=True or a new output directory')
            self.db.execute('COMMIT')
        except BaseException:
            if self.db.in_transaction:
                self.db.execute('ROLLBACK')
            raise

    def _require_open(self):
        if self._closed:
            raise RuntimeError('canonical CSV writer is closed')

    def _key(self, kind, chromosome, job_index, trait_indices, block_index):
        if kind not in _KINDS:
            raise ValueError('unsupported checkpoint association kind')
        original = list(trait_indices)
        try:
            if any(isinstance(i, bool) for i in original):
                raise TypeError('boolean trait index')
            traits = [operator.index(i) for i in original]
        except (ValueError, TypeError) as error:
            raise ValueError('checkpoint trait indices must be nonnegative integers') from error
        chromosome = str(chromosome)
        if (not traits or len(set(traits)) != len(traits) or any(i < 0 for i in traits)
                or not _nonnegative(job_index) or not _nonnegative(block_index)
                or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', chromosome)):
            raise ValueError('checkpoint requires ordered distinct traits and safe chromosome/job/block indices')
        key = dict(kind=kind, chromosome=chromosome, job_index=job_index,
                   block_index=block_index, trait_indices=traits)
        return key, _sha(key)

    def _path(self, relative):
        path = self.directory/relative
        if path.resolve() != path.absolute() or not path.is_relative_to(self.directory):
            raise CommittedCSVCorruption('canonical CSV path contains a symlink or escapes output')
        return path

    def _file(self, row):
        names = ('id','trait','kind','chromosome','relative_path','columns_json',
                 'committed_end','rows','first_job','last_job')
        value = dict(zip(names, row))
        value['columns'] = json.loads(value.pop('columns_json'))
        if (not isinstance(value['columns'], list) or not value['columns']
                or value['columns'][0] != 'trait_index' or len(set(value['columns'])) != len(value['columns'])
                or any(not isinstance(c, str) or not c for c in value['columns'])
                or not _nonnegative(value['committed_end']) or not _nonnegative(value['rows'])):
            raise CommittedCSVCorruption('canonical CSV file metadata differs')
        expected = self._file_identity(value['trait'],value['kind'],value['chromosome'],value['columns'])
        if value['id'] != expected[0] or value['relative_path'] != expected[1]:
            raise CommittedCSVCorruption('canonical CSV file identity differs')
        return value

    def _file_identity(self, trait, kind, chromosome, columns):
        signature = hashlib.sha256(json.dumps(columns).encode()).hexdigest()[:12]
        relative = f'traits/trait_{trait:04d}/{kind}/chr{chromosome}/{signature}/results.csv.gz'
        return _sha(dict(trait_index=trait,kind=kind,chromosome=chromosome,columns=columns)), relative

    def _file_info(self, file_id):
        row = self.db.execute('SELECT * FROM files WHERE id=?', (file_id,)).fetchone()
        if row is None:
            raise CommittedCSVCorruption('committed CSV file is absent from ledger')
        return self._file(row)

    def _actual_path(self, info):
        final = self._path(info['relative_path'])
        partial = final.with_name(final.name+'.partial')
        if partial.is_symlink():
            raise CommittedCSVCorruption('canonical CSV partial must not be a symlink')
        if final.exists() and partial.exists():
            raise CommittedCSVCorruption('canonical CSV has ambiguous final and partial owners')
        return partial if partial.exists() else final

    def _claim(self, trait, chromosome):
        chromosome = str(chromosome)
        key = trait if self.ownership_scope == 'trait' else chromosome
        recovered = trait if self.ownership_scope == 'trait' else (trait, chromosome)
        if recovered in self._recovered_traits:
            return
        owners = self.root/'owners'; owners.mkdir(exist_ok=True)
        new_owner = key not in self._owners
        descriptor = None
        try:
            if new_owner:
                target = owners/(f'trait_{trait:04d}.lock' if self.ownership_scope == 'trait'
                                 else f'chromosome_{chromosome}.lock')
                if target.is_symlink() or owners.is_symlink():
                    raise ValueError('CSV owner lock must not be a symlink')
                descriptor = os.open(target, os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW, 0o600)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError('trait CSV already has an active writer owner') from error
                self._owners[key] = descriptor
            scope = '' if self.ownership_scope == 'trait' else ' AND f.chromosome=?'
            parameters = (trait,self.invocation) if not scope else (trait,self.invocation,chromosome)
            if not self.resume and self.db.execute('''SELECT 1 FROM ranges r JOIN files f ON f.id=r.file_id
                JOIN blocks b ON b.token=r.block_token WHERE f.trait=? AND b.invocation<>?'''+scope+' LIMIT 1',
                parameters).fetchone():
                raise FileExistsError('previous trait CSV exists; choose resume=True or a new output directory')
            files = ('SELECT * FROM files WHERE trait=?' if not scope
                     else 'SELECT * FROM files WHERE trait=? AND chromosome=?')
            for row in self.db.execute(files, (trait,) if not scope else (trait,chromosome)):
                self._recover_tail(self._file(row))
            self._recovered_traits.add(recovered)
            self._owned_traits.add(trait)
        except BaseException:
            if new_owner and descriptor is not None:
                self._owners.pop(key,None)
                os.close(descriptor)
            raise

    def _close_stream(self, file_id):
        raw = self.streams.pop(file_id, None)
        if raw is not None:
            try:
                if file_id in self._dirty_streams:
                    raw.flush()
                    if self._batch is None:
                        os.fsync(raw.fileno())
                        self.durability_profile['data_fsync_calls'] += 1
            finally:
                try:
                    raw.close()
                finally:
                    self._dirty_streams.discard(file_id)

    def _recover_tail(self, info):
        self._close_stream(info['id'])
        path = self._actual_path(info)
        end = info['committed_end']
        if not path.exists():
            if end:
                raise CommittedCSVCorruption('committed canonical CSV file is missing')
            return
        size = path.stat().st_size
        if size < end:
            raise CommittedCSVCorruption('committed canonical CSV prefix was truncated')
        if size == end:
            return
        # Save only the uncommitted tail, never a second copy of the full data.
        evidence = self.root/'uncommitted'/uuid.uuid4().hex
        evidence.mkdir(parents=True)
        saved = evidence/'tail.gz'
        digest = hashlib.sha256()
        with path.open('rb') as source, saved.open('xb') as destination:
            source.seek(end)
            for block in iter(lambda: source.read(1 << 20), b''):
                destination.write(block); digest.update(block)
            destination.flush(); os.fsync(destination.fileno())
        atomic_json(evidence/'receipt.private.json',dict(csv_file=info['relative_path'],
            committed_end=end,uncommitted_bytes=size-end,sha256=digest.hexdigest(),binding_sha256=self.binding_sha256))
        with path.open('r+b') as destination:
            destination.truncate(end); destination.flush(); os.fsync(destination.fileno())

    def _ensure_file(self, trait, kind, chromosome, columns):
        file_id, relative = self._file_identity(trait,kind,chromosome,columns)
        if self._batch is not None:
            prior = self._batch['files'].get(file_id)
            if prior is not None:
                if prior['columns'] != columns:
                    raise CommittedCSVCorruption('canonical CSV schema signature collision')
                return prior
            row = self.db.execute('SELECT * FROM files WHERE id=?', (file_id,)).fetchone()
            if row is None:
                info = self._file((file_id,trait,kind,chromosome,relative,_canonical(columns),0,0,None,None))
                # A prior crash may have created this deterministic path before
                # its registration transaction became durable. Preserve/truncate
                # that orphan suffix before treating the new file as empty.
                if self._actual_path(info).exists():
                    self._recover_tail(info)
                self._batch['new_files'][file_id] = info.copy()
            else:
                info = self._file(row)
            if info['columns'] != columns:
                raise CommittedCSVCorruption('canonical CSV schema signature collision')
            self._batch['files'][file_id] = info
            self._batch['original'][file_id] = info.copy()
            return info
        row = self.db.execute('SELECT * FROM files WHERE id=?', (file_id,)).fetchone()
        if row is None:
            self.db.execute('INSERT OR IGNORE INTO files(id,trait,kind,chromosome,relative_path,columns_json) VALUES(?,?,?,?,?,?)',
                (file_id,trait,kind,chromosome,relative,_canonical(columns)))
            info = self._file_info(file_id)
            if self._actual_path(info).exists():
                self._recover_tail(info)
        else:
            info = self._file(row)
        if info['columns'] != columns:
            raise CommittedCSVCorruption('canonical CSV schema signature collision')
        return info

    def _prepare_first_files(self, kind, chromosome, trait_indices, values):
        """Register initial schemas together, before any compression or append.

        Peek one existing row per trait without copying association arrays. A
        malformed later result still fails at its original position, after
        earlier uncommitted appends have been preserved by the recovery path.
        File endpoints are reread for every tile; no mutable ledger/path cache
        survives a call, and no SQLite writer transaction spans scientific IO.
        """
        prepared, files, missing = [], {}, []
        for trait, (_, result) in zip(trait_indices, values):
            iterator = iter(_rows(result))
            try:
                pending = next(iterator, None)
            except TypeError as error:
                prepared.append((trait, iterator, None, error))
                continue
            prepared.append((trait, iterator, pending, None))
            if pending is None or 'trait_index' in pending:
                continue
            columns = ['trait_index', *pending]
            if any(not isinstance(column, str) or not column for column in columns):
                continue
            file_id, relative = self._file_identity(trait, kind, chromosome, columns)
            if self._batch is not None:
                files[file_id] = self._ensure_file(trait,kind,chromosome,columns)
                continue
            row = self.db.execute('SELECT * FROM files WHERE id=?', (file_id,)).fetchone()
            if row is None:
                missing.append((file_id, trait, kind, chromosome, relative, _canonical(columns)))
            else:
                info = self._file(row)
                if info['columns'] != columns:
                    raise CommittedCSVCorruption('canonical CSV schema signature collision')
                files[file_id] = info
        if missing:
            self.durability_profile['registration_transactions'] += 1
            self.db.execute('BEGIN IMMEDIATE')
            try:
                self.db.executemany('''INSERT OR IGNORE INTO files
                    (id,trait,kind,chromosome,relative_path,columns_json) VALUES(?,?,?,?,?,?)''', missing)
                for item in missing:
                    info = self._file_info(item[0])
                    if info['columns'] != json.loads(item[5]):
                        raise CommittedCSVCorruption('canonical CSV schema signature collision')
                    files[item[0]] = info
                self.db.execute('COMMIT')
                for item in missing:
                    if self._actual_path(files[item[0]]).exists():
                        self._recover_tail(files[item[0]])
            except BaseException:
                if self.db.in_transaction:
                    self.db.execute('ROLLBACK')
                raise
        return prepared, files

    def _stream(self, info, expected_end):
        file_id = info['id']
        if file_id not in self.streams:
            if shutil.disk_usage(self.directory).free < self.minimum_free_bytes:
                raise OSError('CSV output filesystem is below its free-space reserve')
            while len(self.streams) >= self.max_open:
                self._close_stream(next(iter(self.streams)))
            final = self._path(info['relative_path'])
            final.parent.mkdir(parents=True, exist_ok=True)
            self._path(info['relative_path'])
            partial = final.with_name(final.name+'.partial')
            if final.exists():
                final.replace(partial)
            if partial.is_symlink():
                raise CommittedCSVCorruption('canonical CSV partial must not be a symlink')
            self.streams[file_id] = partial.open('a+b')
        raw = self.streams[file_id]
        raw.seek(0,os.SEEK_END)
        if raw.tell() != expected_end:
            raise CommittedCSVCorruption('canonical CSV append endpoint differs from committed ledger')
        self.streams.move_to_end(file_id)
        return raw

    def _append_member(self, info, start, rows, *, trait_index=None):
        raw = self._stream(info,start)
        # Mark before gzip writes its header: even failed compression/fsync
        # leaves a dirty suffix that close/recovery must not call synchronized.
        self._dirty_streams.add(info['id'])
        hashing = _AppendHash(raw)
        count = selected = failed = 0
        with gzip.GzipFile(filename='',fileobj=hashing,mode='wb',compresslevel=1,mtime=0) as compressed:
            with io.TextIOWrapper(compressed,encoding='utf-8',newline='') as text:
                if trait_index is None:
                    output = csv.DictWriter(text,fieldnames=info['columns'],extrasaction='raise')
                    if start == 0:
                        output.writeheader()
                    for row in rows:
                        output.writerow(row); count += 1
                        selected += int(bool(row.get('spa_selected',False)))
                        failed += int(bool(row.get('spa_failed',False)))
                else:
                    output = csv.writer(text,lineterminator='\r\n')
                    if start == 0:
                        output.writerow(info['columns'])
                    count,selected,failed = _write_bound_trait_rows(
                        output,rows,trait_index,info['columns'][1:])
        raw.flush()
        if self._batch is None:
            os.fsync(raw.fileno())
            self.durability_profile['data_fsync_calls'] += 1
            self._dirty_streams.discard(info['id'])
        else:
            self._batch['dirty'].add(info['id'])
        return dict(file_id=info['id'],start=start,end=start+hashing.size,rows=count,
                    header=int(start==0),sha256=hashing.digest.hexdigest()), selected, failed

    def _ranges(self, token):
        for row in self.db.execute('SELECT file_id,start,end,rows,header,sha256 FROM ranges WHERE block_token=? ORDER BY ordinal', (token,)):
            yield dict(zip(('file_id','start','end','rows','header','sha256'),row))

    def _record(self, key, token, block_identity, ordinal_before):
        row = self.db.execute('SELECT record_json,receipt_sha256 FROM blocks WHERE token=?', (token,)).fetchone()
        if row is None:
            return None
        try:
            record = json.loads(row[0])
            body = {k:v for k,v in record.items() if k!='receipt_sha256'}
            counts = record['trait_rows']
            if (record.get('complete') is not True or record.get('binding_sha256') != self.binding_sha256
                    or record.get('schema_version') != 1
                    or record.get('key') != key or record.get('block_identity') != block_identity
                    or record.get('ordinal_before') != ordinal_before
                    or record.get('receipt_sha256') != row[1] or _sha(body) != row[1]
                    or record.get('kind') != key['kind'] or record.get('chromosome') != key['chromosome']
                    or not _nonnegative(record.get('rows')) or set(counts)!=set(map(str,key['trait_indices']))
                    or any(not _nonnegative(value) for value in counts.values()) or sum(counts.values())!=record['rows']):
                raise CommittedCSVCorruption('committed scientific tile binding or counts differ')
            if (not _nonnegative(record.get('binary_spa_selected'))
                    or not _nonnegative(record.get('binary_spa_failed'))
                    or record['binary_spa_failed'] > record['binary_spa_selected']):
                raise CommittedCSVCorruption('committed scientific tile SPA counts differ')
            if key['kind'] == 'individual':
                if (not _nonnegative(record.get('ordinal_before'))
                        or not _nonnegative(record.get('ordinal_after'))
                        or record['ordinal_after'] < record['ordinal_before']):
                    raise CommittedCSVCorruption('committed Single ordinal endpoints differ')
            elif record.get('ordinal_before') is not None or record.get('ordinal_after') is not None:
                raise CommittedCSVCorruption('committed gene tile contains Single ordinal endpoints')
            ranges = list(self._ranges(token))
            if record.get('ranges_sha256') != _sha(ranges) or record.get('range_count') != len(ranges):
                raise CommittedCSVCorruption('committed tile member-range binding differs')
            identities = tuple(_identity(self._actual_path(self._file_info(r['file_id']))) for r in ranges)
            cached = self._verified.get(token)
            if cached == (row[1],identities):
                self._verified.move_to_end(token)
                return record
            actual = dict.fromkeys(map(str,key['trait_indices']),0)
            for item in ranges:
                info = self._file_info(item['file_id'])
                if info['trait'] not in key['trait_indices'] or info['kind'] != key['kind'] or info['chromosome'] != key['chromosome']:
                    raise CommittedCSVCorruption('committed member trait or analysis differs')
                actual[str(info['trait'])] += self._read_range(info,item)
            if actual != counts:
                raise CommittedCSVCorruption('committed member row counts differ')
            if identities != tuple(_identity(self._actual_path(self._file_info(r['file_id']))) for r in ranges):
                raise CommittedCSVCorruption('committed CSV changed during tile readback')
            self._verified[token] = (row[1],identities)
            self._verified.move_to_end(token)
            while len(self._verified)>self.verification_cache_size:
                self._verified.popitem(last=False)
            return record
        except CommittedCSVCorruption:
            raise
        except (OSError,ValueError,KeyError,TypeError,EOFError,csv.Error) as error:
            raise CommittedCSVCorruption('committed CSV tile cannot be authenticated') from error

    def _read_range(self, info, item):
        path = self._actual_path(info)
        count = 0
        with _CheckedSlice(path,item['start'],item['end'],[item]) as raw:
            with io.BufferedReader(raw) as source, gzip.GzipFile(fileobj=source,mode='rb') as compressed:
                with io.TextIOWrapper(compressed,encoding='utf-8',newline='') as text:
                    rows = csv.reader(text)
                    if item['header'] and next(rows,None) != info['columns']:
                        raise CommittedCSVCorruption('committed CSV header differs')
                    for row in rows:
                        if len(row)!=len(info['columns']) or row[0]!=str(info['trait']):
                            raise CommittedCSVCorruption('committed CSV row schema or trait differs')
                        count += 1
        if count != item['rows']:
            raise CommittedCSVCorruption('committed CSV member row count differs')
        return count

    def _visit(self, token, reused):
        inserted = self.db.execute('INSERT OR IGNORE INTO visits VALUES(?,?,?)',
            (self.invocation,token,int(reused))).rowcount
        if inserted and reused:
            self.reused_blocks += 1

    def checkpoint_tile(self, kind, chromosome, job_index, trait_indices, *,
                        block_index=0, block_identity=None, ordinal_before=None):
        self._require_open()
        key,token = self._key(kind,chromosome,job_index,trait_indices,block_index)
        for trait in sorted(key['trait_indices']):
            self._claim(trait,key['chromosome'])
        if not self.resume and not self.db.execute('SELECT 1 FROM visits WHERE invocation=? AND block_token=?',
                (self.invocation,token)).fetchone():
            return None
        record = self._record(key,token,block_identity,ordinal_before)
        if record is not None:
            self._visit(token,True)
        return record

    def _commit_tile(self, token, record, ranges, before):
        self._commit_tiles([(token,record,ranges,before)])

    def _commit_tiles(self, tiles, new_files=()):
        """Publish already-synchronized member ranges in one FULL transaction."""
        self.db.execute('BEGIN IMMEDIATE')
        try:
            for info in new_files:
                self.db.execute('''INSERT INTO files
                    (id,trait,kind,chromosome,relative_path,columns_json) VALUES(?,?,?,?,?,?)''',
                    (info['id'],info['trait'],info['kind'],info['chromosome'],info['relative_path'],_canonical(info['columns'])))
            for token,record,ranges,before in tiles:
                if self.db.execute('SELECT 1 FROM blocks WHERE token=?',(token,)).fetchone():
                    raise RuntimeError('scientific tile was committed by another owner')
                ends,rows = {},{}
                for ordinal,item in enumerate(ranges):
                    file_id = item['file_id']
                    expected = ends.get(file_id,before[file_id])
                    if item['start'] != expected:
                        raise RuntimeError('scientific tile member ranges are not contiguous')
                    ends[file_id] = item['end']; rows[file_id] = rows.get(file_id,0)+item['rows']
                    self.db.execute('INSERT INTO ranges VALUES(?,?,?,?,?,?,?,?)',
                        (token,ordinal,file_id,item['start'],item['end'],item['rows'],item['header'],item['sha256']))
                for file_id,end in ends.items():
                    changed = self.db.execute('''UPDATE files SET committed_end=?,rows=rows+?,
                        first_job=COALESCE(first_job,?),last_job=? WHERE id=? AND committed_end=?''',
                        (end,rows[file_id],record['first_job'],record['last_job'],file_id,before[file_id])).rowcount
                    if changed != 1:
                        raise RuntimeError('canonical CSV committed endpoint changed during tile commit')
                self.db.execute('INSERT INTO blocks VALUES(?,?,?,?,?,?,?)',
                    (token,self.worker_id,self.invocation,record['kind'],record['rows'],_canonical(record),record['receipt_sha256']))
                self.db.execute('INSERT INTO visits VALUES(?,?,0)',(self.invocation,token))
            self.db.execute('COMMIT')
            self.durability_profile['tile_commit_transactions'] += 1
        except BaseException:
            if self.db.in_transaction:
                self.db.execute('ROLLBACK')
            raise

    def _sync_batch_files(self):
        """One data fsync per affected file, including evicted append handles."""
        for file_id in sorted(self._batch['dirty']):
            raw = self.streams.get(file_id)
            if raw is not None:
                raw.flush(); os.fsync(raw.fileno())
            else:
                # Dirty eviction closes/flushed the append handle without
                # claiming durability; retain this bounded file identity until
                # the whole batch crosses the durability barrier.
                info = self._batch['files'][file_id]
                with self._actual_path(info).open('rb') as source:
                    os.fsync(source.fileno())
            self.durability_profile['data_fsync_calls'] += 1
            self._dirty_streams.discard(file_id)

    def write_tiles(self, tiles):
        """Atomically commit up to 256 FIFO tiles; receipts are durable on return.

        Gzip members and scientific range checksums stay per tile. The batch
        creates initial file registrations in the same short SQL transaction
        as all ranges/blocks/visits, after every file has been synchronized.
        There is no SQLite write transaction spanning CSV compression or IO.
        Any failed append/fsync/commit preserves then truncates the entire
        uncommitted suffix, leaving all earlier durable tiles unchanged.
        """
        self._require_open()
        if self._batch is not None:
            raise RuntimeError('canonical CSV batches cannot be nested')
        values = list(tiles)
        if len(values)>256:
            raise ValueError('canonical CSV batch exceeds 256 tiles')
        if not values:
            return []
        started, prior_seconds = time.perf_counter(), self.write_seconds
        self._batch = dict(files={},new_files={},original={},dirty=set(),tiles=[],tokens={})
        try:
            receipts = [self.write_tile(**tile) for tile in values]
            self._sync_batch_files()
            pending = self._batch['tiles']
            if pending:
                self._commit_tiles(pending,self._batch['new_files'].values())
                self.computed_blocks += len(pending)
            self.durability_profile['batch_calls'] += 1
            self.durability_profile['batch_tiles'] += len(pending)
            self.durability_profile['maximum_batch_tiles'] = max(
                self.durability_profile['maximum_batch_tiles'],len(pending))
            self.write_seconds = prior_seconds + time.perf_counter()-started
            return receipts
        except BaseException as error:
            # Cached endpoints are virtual within this batch. Only the copied
            # original ledger endpoints are legal recovery/truncation limits.
            for original in self._batch['original'].values():
                try:
                    # A COMMIT may have succeeded before an exception reached
                    # its caller. The authoritative DB prefix then wins over
                    # the pre-batch fallback, just as for write_tile failures.
                    row = self.db.execute('SELECT * FROM files WHERE id=?',(original['id'],)).fetchone()
                    self._recover_tail(self._file(row) if row is not None else original)
                except BaseException as cleanup:
                    _note(error,f'uncommitted CSV batch cleanup also failed: {type(cleanup).__name__}: {cleanup}')
            raise
        finally:
            self._batch = None

    def write_tile(self, kind, chromosome, trait_results, *, job_index, block_index=0,
                   block_identity=None, ordinal_before=None, ordinal_after=None, binary_counts=None):
        self._require_open(); started = time.perf_counter()
        values = list(trait_results)
        key,token = self._key(kind,chromosome,job_index,[i for i,_ in values],block_index)
        if self._batch is not None and token in self._batch['tokens']:
            prior = self._batch['tokens'][token]
            if prior['block_identity'] != block_identity or prior['ordinal_before'] != ordinal_before:
                raise CommittedCSVCorruption('pending tile identity differs')
            return prior
        if kind=='individual':
            if not _nonnegative(ordinal_before) or not _nonnegative(ordinal_after) or ordinal_after<ordinal_before:
                raise ValueError('Single checkpoint requires nonnegative ordered ordinal endpoints')
        elif ordinal_before is not None or ordinal_after is not None:
            raise ValueError('gene checkpoint must not contain Single ordinal endpoints')
        if binary_counts is not None and (not isinstance(binary_counts,dict)
                or not _nonnegative(binary_counts.get('selected')) or not _nonnegative(binary_counts.get('failed'))
                or binary_counts['failed']>binary_counts['selected']):
            raise ValueError('checkpoint binary counts must be nonnegative with failed at most selected')
        previous = self.checkpoint_tile(kind,chromosome,job_index,key['trait_indices'],block_index=block_index,
            block_identity=block_identity,ordinal_before=ordinal_before)
        if previous is not None:
            return previous
        if self.db.execute('SELECT 1 FROM blocks WHERE token=?',(token,)).fetchone():
            raise FileExistsError('previous scientific tile exists; choose resume=True')
        # Recheck live space on every new scientific tile. An already-open
        # Single stream can remain live for hours; its initial check is stale.
        if shutil.disk_usage(self.directory).free < self.minimum_free_bytes:
            raise OSError('CSV output filesystem is below its free-space reserve')
        ranges,before,ends = [],{},{}
        counts = dict.fromkeys(map(str,key['trait_indices']),0)
        columns = ['trait_index']; selected = failed = 0
        try:
            prepared, initial_files = self._prepare_first_files(
                kind, key['chromosome'], key['trait_indices'], values)
            for trait, iterator, pending, initial_error in prepared:
                if initial_error is not None:
                    raise initial_error
                # Only one existing result is referenced; no copied tile is held.
                while pending is not None:
                    if 'trait_index' in pending:
                        raise ValueError('association row cannot override its bound trait index')
                    schema = ['trait_index',*pending]
                    source_schema = tuple(pending)
                    if any(not isinstance(c,str) or not c for c in schema):
                        raise ValueError('association row requires nonempty string column names')
                    file_id, _ = self._file_identity(trait, kind, key['chromosome'], schema)
                    info = initial_files.get(file_id)
                    if info is None:
                        info = self._ensure_file(trait,kind,key['chromosome'],schema)
                    before.setdefault(info['id'],info['committed_end'])
                    start = ends.get(info['id'],info['committed_end'])
                    def consecutive_rows():
                        nonlocal pending
                        while pending is not None:
                            if 'trait_index' in pending:
                                raise ValueError('association row cannot override its bound trait index')
                            if tuple(pending) != source_schema:
                                return
                            source = pending
                            pending = next(iterator,None)
                            yield source if kind=='individual' else dict(trait_index=trait,**source)
                    item,chosen,errors = self._append_member(info,start,consecutive_rows(),
                        trait_index=trait if kind=='individual' else None)
                    ranges.append(item); ends[info['id']] = item['end']
                    counts[str(trait)] += item['rows']; selected += chosen; failed += errors
                    for column in schema:
                        if column not in columns:
                            columns.append(column)
            record = dict(schema_version=1,complete=True,writer_id=self.worker_id,
                binding_sha256=self.binding_sha256,key=key,block_identity=block_identity,
                ordinal_before=ordinal_before,ordinal_after=ordinal_after,kind=kind,chromosome=key['chromosome'],
                first_job=job_index,last_job=job_index,rows=sum(counts.values()),trait_rows=counts,columns=columns,
                range_count=len(ranges),ranges_sha256=_sha(ranges),
                binary_spa_selected=selected if binary_counts is None else binary_counts['selected'],
                binary_spa_failed=failed if binary_counts is None else binary_counts['failed'])
            record['receipt_sha256'] = _sha(record)
            if self._batch is None:
                self._commit_tile(token,record,ranges,before)
                self.computed_blocks += 1
            else:
                self._batch['tiles'].append((token,record,ranges,before))
                self._batch['tokens'][token] = record
                for item in ranges:
                    info = self._batch['files'][item['file_id']]
                    info['committed_end'] = item['end']
                    info['rows'] += item['rows']
                    info['first_job'] = job_index if info['first_job'] is None else info['first_job']
                    info['last_job'] = job_index
            self.write_seconds += time.perf_counter()-started
            # Durable append/commit is not a readback. Resume authenticates the
            # existing ranges before skipping; finish reads each whole CSV once.
            return record
        except BaseException as error:
            if self._batch is not None:
                raise
            # This also handles a failure half way through a multi-trait tile.
            # A committed record stays intact; only uncommitted suffixes retire.
            for file_id in before:
                try:
                    self._recover_tail(self._file_info(file_id))
                except BaseException as cleanup:
                    _note(error,f'uncommitted CSV cleanup also failed: {type(cleanup).__name__}: {cleanup}')
            raise

    def write(self, kind, trait_index, chromosome, result, *, job_index=None,
              block_index=0, block_identity=None, ordinal_before=None, ordinal_after=None):
        return self.write_tile(kind,chromosome,[(trait_index,result)],job_index=job_index,block_index=block_index,
            block_identity=block_identity,ordinal_before=ordinal_before,ordinal_after=ordinal_after)

    @property
    def row_counts(self):
        if self._closed:
            return {} if self._finished is None else dict(self._finished['rows'])
        return dict(self.db.execute('''SELECT b.kind,SUM(b.rows) FROM blocks b JOIN visits v
            ON v.block_token=b.token WHERE v.invocation=? GROUP BY b.kind''',(self.invocation,)))

    def _finish_file(self, info):
        self._close_stream(info['id'])
        path = self._actual_path(info)
        before = _identity(path)
        if before[2] != info['committed_end']:
            raise CommittedCSVCorruption('canonical CSV final endpoint differs')
        ranges = (dict(start=row[0],end=row[1],sha256=row[2]) for row in self.db.execute(
            'SELECT start,end,sha256 FROM ranges WHERE file_id=? ORDER BY start',(info['id'],)))
        count = 0
        with _CheckedSlice(path,0,info['committed_end'],ranges) as raw:
            with io.BufferedReader(raw) as source, gzip.GzipFile(fileobj=source,mode='rb') as compressed:
                with io.TextIOWrapper(compressed,encoding='utf-8',newline='') as text:
                    reader = csv.reader(text)
                    if next(reader,None) != info['columns']:
                        raise CommittedCSVCorruption('canonical CSV final header differs')
                    for row in reader:
                        if len(row)!=len(info['columns']) or row[0]!=str(info['trait']):
                            raise CommittedCSVCorruption('canonical CSV final schema or trait differs')
                        count += 1
            digest = raw.digest.hexdigest()
        if count != info['rows'] or before != _identity(path):
            raise CommittedCSVCorruption('canonical CSV final rows or identity changed')
        final = self._path(info['relative_path'])
        if path != final:
            path.replace(final)
        receipt = dict(kind=info['kind'],chromosome=info['chromosome'],trait_index=info['trait'],
            rows=count,csv_file=info['relative_path'],sha256=digest,compressed_bytes=before[2],columns=info['columns'],
            first_job=info['first_job'],last_job=info['last_job'],binding_sha256=self.binding_sha256,
            output_layout=self.output_layout,range_sha256_verified=True,complete_gzip_CRC_and_schema_readback=True)
        atomic_json(final.with_name(final.name+'.receipt.json'),receipt)
        return receipt

    def finish(self):
        if self._finished is not None:
            return self._finished
        self._require_open()
        try:
            return self._finish()
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                _note(error,f'CSV owner cleanup also failed: {type(cleanup).__name__}: {cleanup}')
            raise

    def _finish(self):
        receipts = []; digest = hashlib.sha256(); blocks = 0
        # SQL cursors stream block metadata; none is accumulated in Python.
        for token,body,expected in self.db.execute('''SELECT b.token,b.record_json,b.receipt_sha256 FROM blocks b
            JOIN visits v ON v.block_token=b.token WHERE v.invocation=? ORDER BY b.token''',(self.invocation,)):
            record = json.loads(body)
            if _sha({k:v for k,v in record.items() if k!='receipt_sha256'})!=expected:
                raise CommittedCSVCorruption('scientific block metadata changed before finalization')
            if record['ranges_sha256'] != _sha(list(self._ranges(token))):
                raise CommittedCSVCorruption('scientific member-range metadata changed before finalization')
            digest.update(bytes.fromhex(token)); digest.update(bytes.fromhex(expected)); blocks += 1
        query = '''SELECT DISTINCT f.* FROM files f JOIN ranges r ON r.file_id=f.id
            JOIN visits v ON v.block_token=r.block_token WHERE v.invocation=?
            ORDER BY f.trait,f.kind,f.chromosome,f.relative_path'''
        for row in self.db.execute(query,(self.invocation,)):
            info = self._file(row)
            if self.db.execute('''SELECT 1 FROM ranges r WHERE r.file_id=? AND NOT EXISTS(
                SELECT 1 FROM visits v WHERE v.block_token=r.block_token AND v.invocation=?) LIMIT 1''',
                (info['id'],self.invocation)).fetchone():
                raise RuntimeError('canonical CSV contains committed blocks omitted by this invocation schedule')
            receipts.append(self._finish_file(info))
        counts = dict.fromkeys(_KINDS,0); counts.update(self.row_counts)
        trait_rows = {str(trait):dict.fromkeys(_KINDS,0) for trait in self._owned_traits}
        for receipt in receipts:
            trait_rows[str(receipt['trait_index'])][receipt['kind']] += receipt['rows']
        if any(sum(r['rows'] for r in receipts if r['kind']==kind)!=counts[kind] for kind in _KINDS):
            raise CommittedCSVCorruption('final canonical CSV rows differ from committed blocks')
        result = dict(schema_version=3,complete=True,worker_id=self.worker_id,binding_sha256=self.binding_sha256,
            output_layout=self.output_layout,rows=counts,trait_rows=trait_rows,files=len(receipts),receipts=receipts,
            committed_blocks=blocks,committed_blocks_binding_sha256=digest.hexdigest(),
            reused_blocks=self.reused_blocks,computed_blocks=self.computed_blocks,
            invalid_blocks_recomputed=0,write_seconds=self.write_seconds,
            checkpoint_database=str(self.database_path.relative_to(self.directory)),
            committed_range_checksums_verified=True,uncommitted_suffixes_never_reused=True,
            maximum_open_append_streams=self.max_open,verification_cache_bound=self.verification_cache_size,
            durability_profile=dict(self.durability_profile))
        atomic_json(self.directory/f'worker_{self.worker_id}.complete.private.json',result)
        self._finished = result
        self.close()
        return result

    def close(self):
        if self._closed:
            return
        error = None
        for file_id in list(self.streams):
            try:
                self._close_stream(file_id)
            except BaseException as cleanup:
                if error is None:
                    error = cleanup
                else:
                    _note(error,f'additional CSV stream cleanup failed: {cleanup}')
        self._verified.clear()
        for descriptor in self._owners.values():
            try:
                os.close(descriptor)
            except BaseException as cleanup:
                if error is None:
                    error = cleanup
                else:
                    _note(error,f'additional trait owner cleanup failed: {cleanup}')
        self._owners.clear()
        try:
            self.db.close()
        except BaseException as cleanup:
            if error is None:
                error = cleanup
            else:
                _note(error,f'checkpoint database cleanup failed: {cleanup}')
        self._closed = True
        if error is not None:
            raise error

    def abort(self):
        if self._closed:
            return
        try:
            atomic_json(self.directory/f'worker_{self.worker_id}.interrupted.private.json',
                dict(complete=False,binding_sha256=self.binding_sha256,output_layout=self.output_layout,
                     committed_blocks=self.db.execute('SELECT COUNT(*) FROM visits WHERE invocation=?',(self.invocation,)).fetchone()[0]))
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                _note(error,f'CSV owner cleanup also failed: {type(cleanup).__name__}: {cleanup}')
            raise
        self.close()
