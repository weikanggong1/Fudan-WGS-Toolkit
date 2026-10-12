"""Stable chromosome-owned CSV checkpoints across GPU reassignment."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

from .phewas_storage import atomic_json
from .phewas_trait_csv import PerTraitCheckpointCSVWriter
from .worker_file_resources import FileDescriptorTelemetry

_KINDS = ('individual', 'coding', 'noncoding', 'ncrna')


class ChromosomeCheckpointCSVWriter:
    """Own every trait on one chromosome, then close before the next one.

    Checkpoints live below ``chromosomes/chr<name>`` rather than worker IDs.
    Consequently changing GPU count or chromosome ownership on resume reuses
    the same source-bound scientific tiles. Distinct chromosome workers never
    compete for trait locks or a SQLite database. The child writer retains
    its full compressed-range/CRC/schema verification and tail recovery.
    """
    checkpoint_tiles = True
    output_layout = 'per_trait'

    def __init__(self, directory, *, binding, chromosomes, resume=False,
                 worker_id='0', minimum_free_bytes=0, file_descriptor_admission=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.chromosomes = [str(name) for name in chromosomes]
        if (len(set(self.chromosomes)) != len(self.chromosomes)
                or any(not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name) for name in self.chromosomes)):
            raise ValueError('chromosome CSV ownership needs unique safe chromosome names')
        self.binding, self.resume = binding, resume
        self.worker_id, self.minimum_free_bytes = str(worker_id), minimum_free_bytes
        self.current, self.current_name = None, None
        self.outputs, self.completed = [], []
        self._finished, self._closed = None, False
        self.telemetry = (None if file_descriptor_admission is None
                          else FileDescriptorTelemetry(file_descriptor_admission))

    def _finish_current(self):
        if self.current is None:
            return
        result = self.current.finish()
        prefix = Path('chromosomes')/f'chr{self.current_name}'
        # Receipts remain relative to the common csv root, allowing the same
        # final export logic to consume single-card and multi-card reports.
        result = dict(result, chromosome=self.current_name,
            checkpoint_database=str(prefix/result['checkpoint_database']),
            receipts=[dict(receipt, csv_file=str(prefix/receipt['csv_file']))
                      for receipt in result['receipts']])
        self.outputs.append(result)
        self.completed.append(self.current_name)
        self.current, self.current_name = None, None
        if self.telemetry is not None:
            self.telemetry.sample(force=True)

    def _for(self, chromosome):
        if self._closed or self._finished is not None:
            raise RuntimeError('chromosome CSV writer is closed')
        name = str(chromosome)
        if self.current_name == name:
            return self.current
        position = len(self.completed)+(self.current is not None)
        if position >= len(self.chromosomes) or name != self.chromosomes[position]:
            raise ValueError('chromosome CSV callbacks differ from the assigned ordered schedule')
        self._finish_current()
        # Stable scientific owner excludes GPU/worker placement entirely.
        self.current = PerTraitCheckpointCSVWriter(self.directory/'chromosomes'/f'chr{name}',
            binding=self.binding, resume=self.resume, worker_id=f'chr{name}',
            minimum_free_bytes=self.minimum_free_bytes)
        self.current_name = name
        return self.current

    def checkpoint_tile(self, kind, chromosome, job_index, trait_indices, **options):
        result = self._for(chromosome).checkpoint_tile(kind, chromosome, job_index, trait_indices, **options)
        if self.telemetry is not None:
            self.telemetry.sample()
        return result

    def write_tile(self, kind, chromosome, trait_results, **options):
        result = self._for(chromosome).write_tile(kind, chromosome, trait_results, **options)
        if self.telemetry is not None:
            self.telemetry.sample()
        return result

    def write(self, kind, trait_index, chromosome, result, **options):
        return self._for(chromosome).write(kind, trait_index, chromosome, result, **options)

    @property
    def row_counts(self):
        values = dict.fromkeys(_KINDS, 0)
        for result in self.outputs:
            for kind, count in result['rows'].items():
                values[kind] += count
        if self.current is not None:
            for kind, count in self.current.row_counts.items():
                values[kind] += count
        return values

    def finish(self):
        if self._finished is not None:
            return self._finished
        self._finish_current()
        if self.completed != self.chromosomes:
            raise RuntimeError('chromosome CSV finalization omitted an assigned chromosome')
        trait_rows = {}
        digest = hashlib.sha256()
        for result in self.outputs:
            digest.update(result['chromosome'].encode())
            digest.update(bytes.fromhex(result['committed_blocks_binding_sha256']))
            for index, counts in result['trait_rows'].items():
                combined = trait_rows.setdefault(index, dict.fromkeys(_KINDS, 0))
                for kind, count in counts.items():
                    combined[kind] += count
        receipts = [receipt for result in self.outputs for receipt in result['receipts']]
        self._finished = dict(schema_version=3, complete=True, worker_id=self.worker_id,
            output_layout=self.output_layout, checkpoint_layout='chromosome_owned',
            assigned_chromosomes=list(self.chromosomes), rows=self.row_counts, trait_rows=trait_rows,
            files=len(receipts), receipts=receipts,
            committed_blocks=sum(result['committed_blocks'] for result in self.outputs),
            committed_blocks_binding_sha256=digest.hexdigest(),
            reused_blocks=sum(result['reused_blocks'] for result in self.outputs),
            computed_blocks=sum(result['computed_blocks'] for result in self.outputs),
            invalid_blocks_recomputed=0, write_seconds=sum(result['write_seconds'] for result in self.outputs),
            checkpoint_databases=[result['checkpoint_database'] for result in self.outputs],
            committed_range_checksums_verified=True, uncommitted_suffixes_never_reused=True,
            maximum_simultaneously_active_chromosome_writers=1,
            file_descriptors=None if self.telemetry is None else self.telemetry.report())
        atomic_json(self.directory/f'worker_{self.worker_id}.complete.private.json', self._finished)
        self._closed = True
        return self._finished

    def close(self):
        if self.current is not None:
            self.current.close()
            self.current, self.current_name = None, None
        self._closed = True

    def abort(self):
        if self.current is not None:
            self.current.abort()
            self.current, self.current_name = None, None
        self._closed = True
