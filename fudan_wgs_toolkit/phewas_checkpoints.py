"""Bound, readback-verified immutable association blocks for interrupted runs.

A block is one genetic job (or one physical Single request) and one fixed trait
tile. Only a complete marker plus matching CSV checksum/row counts may skip
association. A partial CSV or an unmarked finished file is never a checkpoint.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import operator
from pathlib import Path
import shutil
import time
import uuid
import zlib

from .csv_output import cell_text
from .phewas_storage import _rows, atomic_json, digest_file


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _identity(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


_CORRUPT_BLOCK_ERRORS = (OSError, ValueError, KeyError, TypeError, EOFError,
                         csv.Error, zlib.error)


def _nonnegative_integer(value):
    return type(value) is int and value >= 0


class CheckpointCSVWriter:
    """Stream complete scientific tiles with resumable, immutable CSV receipts.

    ``binding`` must identify the implementation, prepared scientific inputs,
    ordered scientific schedule and fitted models. GPU/worker count is excluded
    so dispatch may change without changing the scientific checkpoint identity.
    ``resume=False`` refuses completed blocks from the same worker; the caller
    requires a fresh run directory while new workers share one binding. It never
    silently adopts results from an earlier invocation. Published tiles may be
    reused within this invocation when a later cohort is temporarily deferred.
    Invalid blocks are retained under
    ``invalid`` and recomputed. Only this invocation's scheduled blocks enter its
    completion receipt, preventing omitted analysis kinds from being counted.
    """
    checkpoint_tiles = True

    def __init__(self, directory, *, binding, resume=False, worker_id='0',
                 minimum_free_bytes=0):
        required = {'source_implementation_sha256', 'input_manifest_sha256',
                    'configuration_sha256', 'models'}
        if (type(resume) is not bool or not isinstance(binding, dict)
                or not required.issubset(binding)
                or any(not isinstance(binding[name], str) or not binding[name]
                       for name in required-{'models'})
                or not isinstance(binding['models'], (dict, list)) or not binding['models']):
            raise ValueError('checkpoint binding requires source, input, schedule and model hashes; resume must be boolean')
        if type(minimum_free_bytes) is not int or minimum_free_bytes < 0:
            raise ValueError('CSV free-space reserve must be a nonnegative integer')
        worker_id = str(worker_id)
        if not worker_id or Path(worker_id).name != worker_id or worker_id in ('.', '..') or '\\' in worker_id:
            raise ValueError('checkpoint worker ID must be a single safe filename component')
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.worker_id, self.resume = str(worker_id), resume
        self.minimum_free_bytes = minimum_free_bytes
        self.binding = json.loads(_canonical(binding))
        self.binding_sha256 = hashlib.sha256(_canonical(self.binding).encode()).hexdigest()
        self.root = self.directory/'.checkpoints'
        self.root.mkdir(exist_ok=True)
        self.used, self.verified, self.write_seconds = {}, {}, 0.
        self.reused_blocks = self.computed_blocks = self.invalid_blocks = 0
        binding_path = self.root/'binding.private.json'
        record = dict(schema_version=1, binding=self.binding, binding_sha256=self.binding_sha256)
        if binding_path.exists():
            existing = json.loads(binding_path.read_text())
            if existing != record:
                raise ValueError('resume checkpoint source/input/schedule/model binding differs')
            if not resume and any(json.loads(p.read_text()).get('writer_id') == self.worker_id
                                  for p in (self.root/'blocks').rglob('*.complete.private.json')):
                raise FileExistsError('previous association checkpoints exist; choose resume=True or a new output directory')
        else:
            # Workers share a scientific binding; an exclusive create resolves
            # first-use races without replacing another worker's binding.
            candidate = binding_path.with_name(binding_path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
            candidate.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
            try:
                try:
                    os.link(candidate, binding_path)
                except FileExistsError:
                    if json.loads(binding_path.read_text()) != record:
                        raise ValueError('concurrent checkpoint binding differs')
            finally:
                candidate.unlink(missing_ok=True)

    def _key(self, kind, chromosome, job_index, trait_indices, block_index):
        if kind not in ('individual', 'coding', 'noncoding', 'ncrna'):
            raise ValueError('unsupported checkpoint association kind')
        try:
            original_traits = list(trait_indices)
            if any(isinstance(i, bool) for i in original_traits):
                raise TypeError('boolean trait index')
            traits = [operator.index(i) for i in original_traits]
        except (TypeError, ValueError) as error:
            raise ValueError('checkpoint trait indices must be nonnegative integers') from error
        if (not traits or len(set(traits)) != len(traits) or any(i < 0 for i in traits)
                or type(job_index) is not int or job_index < 0
                or type(block_index) is not int or block_index < 0):
            raise ValueError('checkpoint requires ordered distinct traits and nonnegative job/block indices')
        key = dict(kind=kind, chromosome=str(chromosome), job_index=job_index,
                   block_index=block_index, trait_indices=traits)
        token = hashlib.sha256(_canonical(key).encode()).hexdigest()
        folder = self.root/'blocks'/token[:2]
        return key, folder/(token+'.csv.gz'), folder/(token+'.complete.private.json')

    def _verify(self, key, path, marker, *, block_identity, ordinal_before):
        record = json.loads(marker.read_text())
        if not isinstance(record, dict):
            return None
        _, expected_path, expected_marker = self._key(key['kind'], key['chromosome'],
            key['job_index'], key['trait_indices'], key['block_index'])
        if (path != expected_path or marker != expected_marker
                or path.resolve() != expected_path.absolute()
                or marker.resolve() != expected_marker.absolute()
                or record.get('csv_file') != str(expected_path.relative_to(self.directory))
                or type(record.get('schema_version')) is not int or record['schema_version'] != 1
                or record.get('complete') is not True
                or record.get('binding_sha256') != self.binding_sha256
                or _canonical(record.get('key')) != _canonical(key)
                or record.get('block_identity') != block_identity
                or record.get('receipt_sha256') != hashlib.sha256(
                    _canonical({k: v for k, v in record.items() if k != 'receipt_sha256'}).encode()).hexdigest()
                or record.get('ordinal_before') != ordinal_before
                or record.get('kind') != key['kind']
                or record.get('chromosome') != key['chromosome']
                or type(record.get('first_job')) is not int or record['first_job'] != key['job_index']
                or type(record.get('last_job')) is not int or record['last_job'] != key['job_index']):
            return None
        counts = record.get('trait_rows')
        columns = record.get('columns')
        if (not _nonnegative_integer(record.get('rows')) or not isinstance(counts, dict)
                or set(counts) != set(map(str, key['trait_indices']))
                or any(not _nonnegative_integer(count) for count in counts.values())
                or sum(counts.values()) != record['rows']
                or not isinstance(columns, list) or not columns or columns[0] != 'trait_index'
                or any(not isinstance(name, str) or not name for name in columns)
                or len(set(columns)) != len(columns)
                or not _nonnegative_integer(record.get('compressed_bytes'))
                or not _nonnegative_integer(record.get('binary_spa_selected'))
                or not _nonnegative_integer(record.get('binary_spa_failed'))
                or record['binary_spa_failed'] > record['binary_spa_selected']):
            return None
        if key['kind'] == 'individual':
            if (not _nonnegative_integer(ordinal_before)
                    or not _nonnegative_integer(record.get('ordinal_before'))
                    or not _nonnegative_integer(record.get('ordinal_after'))
                    or record['ordinal_after'] < ordinal_before):
                return None
        elif record.get('ordinal_before') is not None or record.get('ordinal_after') is not None:
            return None
        identity = (_identity(path), _identity(marker))
        if record['compressed_bytes'] != identity[0][2]:
            return None
        token = str(marker)
        cached = self.verified.get(token)
        if cached is not None and cached[0] == identity:
            return cached[1]
        if digest_file(path) != record.get('sha256'):
            return None
        rows, counts = 0, dict.fromkeys(map(str, key['trait_indices']), 0)
        with gzip.open(path, 'rt', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != record.get('columns') or not reader.fieldnames or reader.fieldnames[0] != 'trait_index':
                return None
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    return None
                trait = row['trait_index']
                if trait not in counts:
                    return None
                counts[trait] += 1
                rows += 1
        if rows != record.get('rows') or counts != record.get('trait_rows'):
            return None
        if identity != (_identity(path), _identity(marker)):
            return None
        self.verified[token] = (identity, record)
        return record

    def checkpoint_tile(self, kind, chromosome, job_index, trait_indices, *,
                        block_index=0, block_identity=None, ordinal_before=None):
        key, path, marker = self._key(kind, chromosome, job_index, trait_indices, block_index)
        if (not self.resume and str(marker) not in self.used) or not marker.exists() or not path.exists():
            return None
        try:
            record = self._verify(key, path, marker, block_identity=block_identity,
                                  ordinal_before=ordinal_before)
        except _CORRUPT_BLOCK_ERRORS:
            record = None
        if record is not None:
            token = str(marker)
            if token not in self.used:
                self.reused_blocks += 1
            self.used[token] = record
        return record

    def write_tile(self, kind, chromosome, trait_results, *, job_index,
                   block_index=0, block_identity=None, ordinal_before=None, ordinal_after=None, binary_counts=None):
        started = time.perf_counter()
        trait_results = list(trait_results)
        key, path, marker = self._key(kind, chromosome, job_index,
                                    [trait for trait, _ in trait_results], block_index)
        traits = key['trait_indices']
        trait_results = [(trait, result) for trait, (_, result) in zip(traits, trait_results)]
        if kind == 'individual':
            if (not _nonnegative_integer(ordinal_before) or not _nonnegative_integer(ordinal_after)
                    or ordinal_after < ordinal_before):
                raise ValueError('Single checkpoint requires nonnegative ordered ordinal endpoints')
        elif ordinal_before is not None or ordinal_after is not None:
            raise ValueError('gene checkpoint must not contain Single ordinal endpoints')
        if binary_counts is not None and (not isinstance(binary_counts, dict)
                or not _nonnegative_integer(binary_counts.get('selected'))
                or not _nonnegative_integer(binary_counts.get('failed'))
                or binary_counts['failed'] > binary_counts['selected']):
            raise ValueError('checkpoint binary counts must be nonnegative with failed at most selected')
        previous = self.checkpoint_tile(kind, chromosome, job_index, traits,
            block_index=block_index, block_identity=block_identity, ordinal_before=ordinal_before)
        if previous is not None:
            return previous
        if str(marker) in self.used:
            raise RuntimeError('association tile was published more than once in one invocation')
        path.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.directory).free < self.minimum_free_bytes:
            raise OSError('CSV output filesystem is below its free-space reserve')
        # Preserve interrupted/untrusted artifacts. Their existence cannot skip
        # computation or make the next invocation append duplicate rows.
        stale = [p for p in (path, marker) if p.exists()]
        if stale:
            invalid = self.root/'invalid'/uuid.uuid4().hex
            invalid.mkdir(parents=True)
            for item in stale:
                item.replace(invalid/item.name)
            self.invalid_blocks += 1
        rows, columns = [], ['trait_index']
        trait_counts = dict.fromkeys(map(str, traits), 0)
        selected = failed = 0
        for trait, result in trait_results:
            for source in _rows(result):
                if 'trait_index' in source:
                    raise ValueError('association row cannot override its bound trait index')
                row = dict(trait_index=int(trait), **source)
                rows.append(row)
                trait_counts[str(trait)] += 1
                for name in row:
                    if name not in columns:
                        columns.append(name)
                selected += int(bool(source.get('spa_selected', False)))
                failed += int(bool(source.get('spa_failed', False)))
        partial = path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.partial')
        try:
            with gzip.open(partial, 'wt', encoding='utf-8', newline='', compresslevel=1) as handle:
                output = csv.writer(handle, lineterminator='\n')
                output.writerow(columns)
                for row in rows:
                    output.writerow([cell_text(row.get(name)) for name in columns])
            partial.replace(path)
            record = dict(schema_version=1, complete=True, writer_id=self.worker_id, binding_sha256=self.binding_sha256,
                key=key, block_identity=block_identity, ordinal_before=ordinal_before,
                ordinal_after=ordinal_after, kind=kind, chromosome=str(chromosome),
                rows=len(rows), trait_rows=trait_counts, columns=columns,
                csv_file=str(path.relative_to(self.directory)), sha256=digest_file(path),
                compressed_bytes=path.stat().st_size, first_job=job_index, last_job=job_index,
                binary_spa_selected=(selected if binary_counts is None else binary_counts['selected']),
                binary_spa_failed=(failed if binary_counts is None else binary_counts['failed']))
            record['receipt_sha256'] = hashlib.sha256(_canonical(record).encode()).hexdigest()
            atomic_json(marker, record)
            checked = self._verify(key, path, marker, block_identity=block_identity,
                                   ordinal_before=ordinal_before)
            if checked is None:
                raise RuntimeError('checkpoint CSV readback verification failed')
            self.used[str(marker)] = checked
            self.computed_blocks += 1
            self.write_seconds += time.perf_counter()-started
            return checked
        except BaseException:
            # A marker is the commit record; a failed readback cannot commit.
            if marker.exists():
                invalid = self.root/'invalid'/uuid.uuid4().hex
                invalid.mkdir(parents=True)
                marker.replace(invalid/marker.name)
            raise

    def write(self, kind, trait_index, chromosome, result, *, job_index=None,
              block_index=0, block_identity=None, ordinal_before=None, ordinal_after=None):
        return self.write_tile(kind, chromosome, [(trait_index, result)], job_index=job_index,
            block_index=block_index, block_identity=block_identity,
            ordinal_before=ordinal_before, ordinal_after=ordinal_after)

    def finish(self):
        for marker_name, record in self.used.items():
            path = self.directory/record['csv_file']
            try:
                checked = self._verify(record['key'], path, Path(marker_name),
                    block_identity=record['block_identity'],
                    ordinal_before=record['ordinal_before'])
            except _CORRUPT_BLOCK_ERRORS:
                checked = None
            if checked is None:
                raise RuntimeError('completed association checkpoint changed before final CSV audit')
        receipts = sorted(self.used.values(), key=lambda r: (
            int(r['chromosome']), r['first_job'], r['key']['block_index'], r['key']['trait_indices']))
        counts = {}
        for receipt in receipts:
            counts[receipt['kind']] = counts.get(receipt['kind'], 0)+receipt['rows']
        result = dict(schema_version=2, complete=True, worker_id=self.worker_id,
            binding_sha256=self.binding_sha256, rows=counts, files=len(receipts),
            receipts=receipts, reused_blocks=self.reused_blocks,
            computed_blocks=self.computed_blocks, invalid_blocks_recomputed=self.invalid_blocks,
            write_seconds=self.write_seconds)
        atomic_json(self.directory/f'worker_{self.worker_id}.complete.private.json', result)
        return result

    def abort(self):
        atomic_json(self.directory/f'worker_{self.worker_id}.interrupted.private.json',
            dict(complete=False, committed_blocks=len(self.used),
                 binding_sha256=self.binding_sha256))
