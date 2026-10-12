"""Genetic-major associations on the completed FID/IID prepared cache.

Scientific identity is independent of GPU dispatch. Model and CSV checkpoint
proofs retain both FID and IID, per-trait complete cases and immutable arrays.
This module reads existing genetic caches only; it never prepares genetic data.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import gc
import hashlib
import json
import multiprocessing as mp
import os
import stat
import sys
from pathlib import Path
import time
import traceback
import uuid

import numpy as np
import torch

from . import run
from .cpu_budget import detect_cpu_capacity
from .identity import sample_keys
from .phewas_models import SparseKinshipData, prepare_phewas_model
from ._fit_structure_cache import FitStructureCache
from .phewas_cache import _stream_bindings, _verify_stream_bindings
from .phewas_storage import (ModelRepository, atomic_json, digest_file,
                             save_model_store, trait_index_sha256, write_trait_index)

_KINDS = ('individual', 'coding', 'noncoding', 'ncrna')


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _source_files(root, manifest_path, manifest, entries, phenotype_csv, covariate_csv, kinship_file):
    """Bind actual scientific input bytes, not counts or mutable file names."""
    files = [Path(phenotype_csv).resolve(), Path(covariate_csv).resolve(), manifest_path,
             run._cache_path(root, manifest.get('sample_pairs', 'sample_pairs.npy'))]
    if manifest.get('sample_ids'):
        files.append(run._cache_path(root, manifest['sample_ids']))
    for entry in entries:
        files.extend([Path(entry['container_directory'])/'manifest.json',
                      Path(entry['metadata_directory'])/'manifest.json'])
        files.extend(Path(entry[key]) for key in ('gene_catalog', 'promoter_intervals') if key in entry)
    if kinship_file is not None:
        files.append(kinship_file)
    sidecars = manifest.get('phewas') or {}
    if sidecars.get('cohort_rows'):
        files.append(run._cache_path(root, sidecars['cohort_rows']))
    return {str(path): digest_file(path) for path in dict.fromkeys(files)}


def _verify_inputs(binding):
    for path, expected in binding['files'].items():
        if digest_file(path) != expected:
            raise ValueError('prepared association scientific input checksum changed')
    root = Path(binding['prepared_directory'])
    if (root/'COMPLETE').read_text().strip() != binding['dataset_sha256']:
        raise ValueError('prepared dataset completion binding changed')
    _verify_stream_bindings(binding['genotype_streams'])


def _input_file_identity(path):
    """Immutable regular-file identity; include inode and change time."""
    try:
        value = Path(path).stat()
    except OSError as error:
        raise ValueError('prepared association scientific input became inaccessible') from error
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('prepared association scientific input must remain a regular file')
    return dict(dev=value.st_dev,ino=value.st_ino,size=value.st_size,
        mtime_ns=value.st_mtime_ns,ctime_ns=value.st_ctime_ns)


class _BinaryInputVerification:
    """Hash binary fit inputs at both boundaries, stat-check between fits.

    The starting hashes authenticate the exact immutable identity used during
    the loop; same-size edits and same-byte inode replacements fail closed.
    Full boundary hashing is bracketed by identical identities to reject
    changes during reads. Genotype streams retain their existing stat binding
    and per-frame checksum verification. No fitted array/source binding changes.
    """
    def __init__(self,binding,*,loaded_input_stats=None):
        self.binding = binding
        self.paths = sorted(set(binding['files']) |
            {str(Path(binding['prepared_directory'])/'COMPLETE')})
        self.full_hash_passes = self.stat_check_passes = 0
        self.final_full_hash_verified = False
        self.identities = None
        before = self._identities()
        if loaded_input_stats is not None and any(before.get(path)!=identity
                for path,identity in loaded_input_stats.items()):
            raise ValueError('binary input source changed after the phenotype matrices were read')
        _verify_inputs(binding)
        after = self._identities()
        if before!=after:
            raise ValueError('binary input identity changed during the starting full checksum pass')
        self.identities = after
        self.full_hash_passes += 1

    def _identities(self):
        return {path:_input_file_identity(path) for path in self.paths}

    def check(self):
        if self._identities()!=self.identities:
            raise ValueError('binary input source inode or content identity changed during null fitting')
        _verify_stream_bindings(self.binding['genotype_streams'])
        self.stat_check_passes += 1

    def finish(self):
        before = self._identities()
        if before!=self.identities:
            raise ValueError('binary input identity changed before the final full checksum pass')
        _verify_inputs(self.binding)
        after = self._identities()
        if after!=before:
            raise ValueError('binary input identity changed during the final full checksum pass')
        self.full_hash_passes += 1
        self.final_full_hash_verified = True

    def summary(self):
        return dict(full_hash_passes=self.full_hash_passes,stat_check_passes=self.stat_check_passes,
            verified_files=len(self.paths),final_full_hash_verified=self.final_full_hash_verified,
            file_identity_fields=['dev','ino','size','mtime_ns','ctime_ns'],
            scope='binary null-fit loop; full starting/ending SHA-256, immutable identities between traits')


@contextmanager
def prepared_reader(reference, spec=None):
    """Expose logical prepared rows after immutable container/axis checks."""
    from .cache_runtime.fast_container import Container
    from .cache_runtime.portable import PortableMetadataReader
    directory = Path(reference['container_directory'])
    expected = reference['genotype_manifest_sha256']
    proof = run._proof(directory, expected)
    metadata = PortableMetadataReader(reference['metadata_directory'], directory, verify_checksums=True)
    container = None
    try:
        if digest_file(Path(reference['metadata_directory'])/'manifest.json') != reference['metadata_manifest_sha256']:
            raise ValueError('prepared metadata manifest changed')
        if _array_hash(metadata.sample_ids()) != reference['sample_axis_sha256']:
            raise ValueError('prepared metadata FID/IID axis changed')
        physical = metadata._array(metadata.manifest['source_sample_rows'])
        container = Container(directory, proof, physical)
        if not np.array_equal(container.samples, physical):
            raise ValueError('prepared container source rows differ from metadata')
        yield dict(metadata=metadata, container=container, portable_axis=True)
    finally:
        try:
            if container is not None:
                container.close()
        finally:
            metadata.close()
            run._proof(directory, expected)
            if digest_file(Path(reference['metadata_directory'])/'manifest.json') != reference['metadata_manifest_sha256']:
                raise ValueError('prepared metadata changed during association')


def _configuration(manifest, entries, population_ids, output, options, source):
    """One bound genetic schedule shared by all phenotype models."""
    settings = run._analysis_settings(manifest)
    schedules, ordinal = [], 0
    for entry in entries:
        run._annotation_coverage(entry, gene_variant_type=options['gene_variant_type'],
            analyses=options['analyses'], expected_sample_keys=population_ids,
            expected_analysis=settings)
        cache = Path(entry['container_directory'])
        genotype_sha = digest_file(cache/'manifest.json')
        run._proof(cache, genotype_sha)
        jobs = run._jobs(entry, options['analyses'], output,
            single_mac_cutoff=options['single_mac_cutoff'],
            single_group_variants=options['single_group_variants'],
            single_region_size=options['single_region_size'])
        for job in jobs:
            # A scientific ordinal never changes with worker count or GPU IDs.
            job.pop('output', None)
            job['checkpoint_job_index'] = ordinal
            ordinal += 1
        schedules.append(dict(name=entry['name'], gds=str(cache), container_directory=str(cache),
            metadata_directory=entry['metadata_directory'],
            genotype_manifest_sha256=genotype_sha,
            metadata_manifest_sha256=digest_file(Path(entry['metadata_directory'])/'manifest.json'),
            sample_axis_sha256=_array_hash(population_ids),
            annotation_index=dict(promoter_intervals_file=entry.get('promoter_intervals')), jobs=jobs))
    from .phewas_cache import _analysis_defaults
    analysis_options = _analysis_defaults(None, options['covariance_block_size'],
        options['long_mask_threshold'], options['long_mask_rank'], options['seed'],
        long_mask_method=options.get('long_mask_method', 'full'))
    analysis_options.update(variant_type=options['gene_variant_type'])
    return dict(chromosomes=schedules, **settings, matmul_mode='tf32', analysis_options=analysis_options,
        resident_genotypes=True, local_mask_reuse=True, statistics_tail_optimization=True,
        weight_batch_optimization=True, stage_profile=True,
        individual_genotype_block_size=1024,
        individual_effective_block_size=options['individual_effective_block_size'],
        source_implementation_sha256=source)


def _covariate_selection(value, traits, covariates):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError('phenotype_covariates must be a phenotype mapping')
    result = dict(value)
    if set(result)-set(traits):
        raise ValueError('phenotype_covariates contains an unknown trait')
    for names in result.values():
        if (not isinstance(names, (list, tuple)) or any(not isinstance(name, str) for name in names)
                or len(set(names)) != len(names) or set(names)-set(covariates)):
            raise ValueError('phenotype_covariates must select distinct existing covariates')
    return {name: list(columns) for name, columns in result.items()}


def _cohort_pairs(root, manifest, population):
    sidecar = (manifest.get('phewas') or {}).get('cohort_rows')
    if sidecar is None:
        return population
    rows = np.load(run._cache_path(root, sidecar), allow_pickle=False)
    if rows.dtype.kind == 'b':
        if rows.shape != (len(population),):
            raise ValueError('prepared cohort mask has the wrong geometry')
        rows = np.flatnonzero(rows)
    if (rows.ndim != 1 or rows.dtype.kind not in 'iu' or not len(rows)
            or len(np.unique(rows)) != len(rows) or np.any(rows >= len(population)) or np.any(rows < 0)):
        raise ValueError('prepared cohort requires distinct logical population rows')
    # Retain the metadata's order, including for an unsorted supplied subset.
    return population[np.sort(rows)]


def _verify_model(target, expected, receipt):
    """Verify every saved array before accepting same-run model reuse."""
    if not target.is_dir():
        raise ValueError('existing model store lacks its immutable completion directory')
    metadata_sha = digest_file(target/'model.private.json')
    meta = json.loads((target/'model.private.json').read_text())
    fit = meta.get('fit', {})
    if (fit.get('prepared_association_binding') != expected
            or fit.get('source_implementation_sha256') != expected['source_implementation_sha256']
            or fit.get('trait_index') != expected['trait_index']):
        raise ValueError('existing model source/input/sample binding differs')
    if (meta.get('schema_version') != 1 or meta.get('normal_state') != 'normal'
            or meta.get('spa_state') != ('spa' if expected['family'] == 'binomial' else None)):
        raise ValueError('existing model state layout differs')
    had_receipt = receipt.is_file()
    entry = json.loads(receipt.read_text()) if had_receipt else dict(
        n=meta['n'], family=meta['family'], path=str(target), sha256=metadata_sha,
        trait_index=expected['trait_index'])
    if (entry.get('sha256') != metadata_sha or entry.get('path') != str(target)
            or entry.get('trait_index') != expected['trait_index']):
        raise ValueError('model completion receipt checksum differs')
    if meta.get('family') != expected['family'] or entry.get('family') != expected['family']:
        raise ValueError('existing model family differs')
    rows = np.load(target/'sample_rows.npy', allow_pickle=False)
    if (digest_file(target/'sample_rows.npy') != meta['sample_rows_sha256']
            or _array_hash(rows) != expected['sample_rows_sha256']
            or rows.dtype != np.dtype(np.int64) or rows.shape != (meta['n'],)
            or entry.get('n') != meta['n']):
        raise ValueError('existing model sample geometry differs')
    states = ['normal'] + (['spa'] if expected['family'] == 'binomial' else [])
    for name in states:
        path = target/name/'state.private.json'
        if digest_file(path) != meta['spa_state_sha256' if name == 'spa' else 'state_sha256']:
            raise ValueError('existing fitted state checksum differs')
        state = json.loads(path.read_text())
        if state.get('family') != expected['family'] or state.get('converged') is not True:
            raise ValueError('existing fitted state did not converge')
        if 'sample_ids' not in state.get('arrays', {}):
            raise ValueError('existing fitted state lacks its FID/IID identity')
        for field, spec in state['arrays'].items():
            filename = spec['file']
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError('model array escaped its immutable store')
            array_path = target/name/filename
            if digest_file(array_path) != spec['sha256']:
                raise ValueError('existing fitted array checksum differs')
            value = np.load(array_path, mmap_mode='r', allow_pickle=False)
            if list(value.shape) != spec['shape'] or value.dtype.str != spec['dtype']:
                raise ValueError('existing fitted array geometry differs')
            if field == 'sample_ids' and _array_hash(value) != expected['sample_ids_sha256']:
                raise ValueError('existing model FID/IID identity differs')
            del value
    if digest_file(target/'model.private.json') != metadata_sha:
        raise ValueError('existing model metadata changed during verification')
    if not had_receipt:
        # save_model_store commits the whole directory atomically. A crash can
        # precede its separate receipt; restore that receipt only after the
        # complete bound state and all its arrays have passed verification.
        atomic_json(receipt, entry)
    return entry


def _fit_models(plan, phenotypes, covariates, population, kinship, options, device):
    output, entries = Path(plan['output_directory']), []
    lookup = {key: i for i, key in enumerate(sample_keys(population))}
    cohort = _cohort_pairs(Path(plan['input_binding']['prepared_directory']), plan['dataset'], population)
    alignment_cache = run.AlignedInputCache(phenotypes, covariates, cohort)
    structures = FitStructureCache()
    verifier = (_BinaryInputVerification(plan['input_binding'],
                    loaded_input_stats=options.get('_loaded_input_stats')) if plan.get('binary_only') else None)
    def verify_fit_inputs():
        if verifier is None:
            _verify_inputs(plan['input_binding'])
        else:
            verifier.check()
    try:
        for index, name in enumerate(phenotypes.names):
            family = plan['families'][name]
            ids, y, x, names, counts = run.aligned_phenotype(phenotypes, covariates, cohort, name,
                covariate_names=plan['covariate_selection'].get(name), reduce_redundant=True,
                rm_nan=plan.get('rm_nan', False), family=family,
                alignment_cache=alignment_cache, binary_only=plan.get('binary_only',False))
            if plan.get('binary_only'):
                cases, controls = int(np.count_nonzero(y==1)), int(np.count_nonzero(y==0))
                if cases+controls!=len(y) or cases==0 or controls==0:
                    raise ValueError(f'binary phenotype {name} requires both disease cases and controls after alignment')
                counts.update(analysis_cases=cases, analysis_controls=controls,
                    imputed_controls=counts['imputed_n'], available_cases=cases,
                    available_controls=controls-counts['imputed_n'])
            rows = np.asarray([lookup[key] for key in ids], dtype=np.int64)
            expected = dict(input_manifest_sha256=plan['scientific_binding_sha256'],
                source_implementation_sha256=plan['source_implementation_sha256'], trait_index=index,
                family=family, rm_nan=plan.get('rm_nan', False), sample_rows_sha256=_array_hash(rows), sample_ids_sha256=_array_hash(ids),
                phenotype_sha256=_array_hash(y), design_sha256=_array_hash(x), covariate_names=list(names),
                covariate_design=counts['covariate_design'])
            if plan.get('binary_only'):
                expected['binary_only'] = True
            target = output/'models'/f'trait_{index:04d}'
            receipt = output/'models'/f'trait_{index:04d}.complete.private.json'
            if target.exists() or receipt.exists():
                if not options['resume']:
                    raise FileExistsError('existing fitted store requires resume=True')
                verify_fit_inputs()
                entry = _verify_model(target, expected, receipt)
            else:
                verify_fit_inputs()
                before = time.perf_counter()
                fitted = prepare_phewas_model(y, ids, x, family=family, kinship=kinship,
                    continuous_transform=plan['transform'] if family == 'gaussian' else 'none',
                    device=device, association_mode='tf32', use_spa=True, fit_structure_cache=structures)
                if not fitted.model.converged:
                    raise ArithmeticError('prepared phenotype null fit did not converge')
                selected = rows[fitted.sample_indices]
                if not np.array_equal(selected, rows):
                    raise ValueError('null fitter unexpectedly changed the aligned complete-case axis')
                metadata = dict(fitted.metadata, trait_index=index, phenotype=name,
                    covariate_names=list(names), alignment=counts,
                    covariate_design=counts['covariate_design'],
                    prepared_association_binding=expected, wall_seconds=time.perf_counter()-before,
                    source_implementation_sha256=plan['source_implementation_sha256'],
                    alignment_cache=alignment_cache.summary(), fit_structure_cache=structures.stats())
                if plan.get('binary_only'):
                    metadata.update(binary_only=True, **{key:counts[key] for key in
                        ('analysis_cases','analysis_controls','available_cases','available_controls','imputed_controls')})
                entry = save_model_store(fitted.model, target, sample_rows=selected,
                    fit_metadata=metadata, spa_model=fitted.spa_model)
                entry['trait_index'] = index
                verify_fit_inputs()
                atomic_json(receipt, entry)
                del fitted
                gc.collect()
            entries.append(entry)
        if verifier is not None:
            verifier.finish()
            atomic_json(output/'model_fit.runtime.private.json',dict(schema_version=1,complete=True,
                binary_only=True,trait_count=len(entries),input_verification=verifier.summary(),
                alignment_cache=alignment_cache.summary(),fit_structure_cache=structures.stats()))
    finally:
        structures.close()
    return entries


def _partition(configuration, worker_id, worker_count):
    result, chromosomes = dict(configuration), []
    for original in configuration['chromosomes']:
        jobs = [dict(job) for job in original['jobs']
                if job['checkpoint_job_index'] % worker_count == worker_id]
        if jobs:
            chromosomes.append(dict(original, jobs=jobs))
    result['chromosomes'] = chromosomes
    return result


def _cpu_resources(cpu_threads):
    """Bound the shared CPU pool by affinity and every visible cgroup quota."""
    capacity = detect_cpu_capacity()
    if cpu_threads > capacity['effective_cores']:
        raise ValueError('cpu_threads exceeds the effective affinity/cgroup CPU quota')
    visible = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(
        range(capacity['effective_cores']))
    return dict(cpu_capacity=capacity, cpu_pool=visible[:capacity['effective_cores']])


def _worker_cpu_selection(resources, worker_id):
    visible = set(os.sched_getaffinity(0))
    pool = resources['cpu_pool']
    if not pool or not set(pool) <= visible:
        raise ValueError('prepared worker CPU allocation escaped the caller affinity')
    # Quotas can become tighter after spawning. Keep the inherited pool inside
    # the current limit and reject a request that no longer fits it.
    effective = min(len(pool), detect_cpu_capacity()['effective_cores'])
    requested = resources['cpu_threads']
    if requested > effective:
        raise ValueError('cpu_threads exceeds the worker affinity/cgroup CPU quota')
    pool = pool[:effective]
    return sorted(pool[(worker_id*requested+i) % len(pool)] for i in range(requested))


def _thread_ids():
    tasks = Path('/proc/self/task')
    return [int(path.name) for path in tasks.iterdir()] if tasks.is_dir() else [0]


@contextmanager
def _inline_cpu_policy(resources):
    """Limit existing/helper threads temporarily, then restore caller policy."""
    previous_threads = torch.get_num_threads()
    can_bind = hasattr(os, 'sched_getaffinity') and hasattr(os, 'sched_setaffinity')
    prior, main_affinity, selected = {}, None, None
    if can_bind:
        main_affinity = sorted(os.sched_getaffinity(0))
        selected = _worker_cpu_selection(resources, 0)
        for thread_id in _thread_ids():
            try:
                prior[thread_id] = sorted(os.sched_getaffinity(thread_id))
            except ProcessLookupError:
                pass
    try:
        if can_bind:
            os.sched_setaffinity(0, selected)
            for thread_id in _thread_ids():
                try:
                    os.sched_setaffinity(thread_id, selected)
                except ProcessLookupError:
                    pass
        torch.set_num_threads(resources['cpu_threads'])
        yield
    finally:
        try:
            torch.set_num_threads(previous_threads)
        finally:
            if can_bind:
                # Restoring Torch can create helper threads. New threads take
                # the caller's main affinity; surviving threads retain their
                # exact earlier policy, including a narrower caller subset.
                os.sched_setaffinity(0, main_affinity)
                for thread_id in _thread_ids():
                    try:
                        os.sched_setaffinity(thread_id, prior.get(thread_id, main_affinity))
                    except ProcessLookupError:
                        pass


def _worker(plan, entries, resources, *, worker_id, worker_count, device, bind_affinity=False):
    """Run this invocation's own partition; parent affinity stays unchanged."""
    from .worker_file_resources import configure_worker_file_capacity
    file_admission = configure_worker_file_capacity(plan.get('trait_count', 0))
    from .phewas_trait_csv import PerTraitCheckpointCSVWriter
    from .phewas_runtime.runtime import run_batched_configuration
    from .phewas_resources import HostMemoryGate
    from .phewas_run import (_host_memory_policy, _close_association_resources,
                             _association_partition_strategy)
    output = Path(plan['output_directory'])
    prefix = output/f'worker_{worker_id}.attempt_{resources["attempt_id"]}'
    started, exit_code = time.perf_counter(), 1
    previous_threads = torch.get_num_threads()
    repository, writer = None, None
    try:
        if run._implementation_sha256() != plan['source_implementation_sha256']:
            raise ValueError('prepared association implementation changed after planning')
        _verify_inputs(plan['input_binding'])
        if bind_affinity and hasattr(os, 'sched_getaffinity'):
            selected = _worker_cpu_selection(resources, worker_id)
            os.sched_setaffinity(0, selected)
            for thread in Path('/proc/self/task').iterdir():
                try:
                    os.sched_setaffinity(int(thread.name), selected)
                except ProcessLookupError:
                    pass
        torch.set_num_threads(resources['cpu_threads'])
        from .phewas_partition import build_trait_dispatch, publish_trait_dispatch, dispatch_configuration
        dispatch = build_trait_dispatch(entries,worker_count,plan['configuration'],
            source_binding=dict(source_implementation_sha256=plan['source_implementation_sha256'],
                scientific_binding_sha256=plan['scientific_binding_sha256'],
                plan_sha256=digest_file(output/'plan.private.json')),
            strategy=_association_partition_strategy(dict(plan, csv_output_layout='per_trait_checkpoint')),
            model_cache_capacity_bytes=plan.get('association_model_cache_capacity_bytes'),
            trait_batch_size=resources['trait_batch_size'])
        publish_trait_dispatch(output,dispatch,resources['attempt_id'])
        assigned_traits = dispatch['workers'][worker_id]['trait_indices']
        repository = ModelRepository(entries,trait_indices=assigned_traits)
        binding = dict(source_implementation_sha256=plan['source_implementation_sha256'],
            input_manifest_sha256=plan['scientific_binding_sha256'],
            configuration_sha256=_hash(dict(configuration=plan['configuration'],
                                           trait_batch_size=resources['trait_batch_size'])),
            models=[dict(trait_index=entry['trait_index'], sha256=entry['sha256']) for entry in entries])
        if dispatch['strategy']=='whole_chromosomes':
            from .chromosome_csv import ChromosomeCheckpointCSVWriter
            writer = ChromosomeCheckpointCSVWriter(output/'csv', binding=binding,
                chromosomes=dispatch['workers'][worker_id]['assigned_chromosomes'],
                resume=resources['resume'], worker_id=str(worker_id),
                minimum_free_bytes=plan.get('csv_minimum_free_bytes', 0),
                file_descriptor_admission=file_admission)
        else:
            writer = PerTraitCheckpointCSVWriter(output/'csv', binding=binding,
                resume=resources['resume'], worker_id=str(worker_id),
                minimum_free_bytes=plan.get('csv_minimum_free_bytes', 0))
        gate = HostMemoryGate(output, **_host_memory_policy(plan))
        configuration = dispatch_configuration(plan['configuration'],dispatch,worker_id)
        from .phewas_partition import GeneticScheduleAudit
        schedule_audit = GeneticScheduleAudit(configuration)
        if configuration['chromosomes']:
            report = run_batched_configuration(configuration, model_repository=repository,
                reader_factory=prepared_reader, writer=writer, device=device,
                trait_batch_size=resources['trait_batch_size'], memory_limit_gib=resources['memory_limit_gib'],
                memory_reserve_bytes=int(resources['memory_reserve_gib']*2**30),
                cpu_threads=resources['cpu_threads'], device_cache_bytes=resources['device_cache_bytes'],
                compact_cache_bytes=resources['compact_cache_bytes'],
                metadata_cache_bytes=resources['metadata_cache_bytes'],
                emit=schedule_audit.observe,
                host_memory_lease_factory=gate.lease_factory)
        else:
            report = dict(association_rows=dict.fromkeys(_KINDS, 0), job_groups=0,
                          output=writer.finish(), total_seconds=0., assigned_genetic_jobs=0)
        _verify_inputs(plan['input_binding'])
        if run._implementation_sha256() != plan['source_implementation_sha256']:
            raise ValueError('prepared association source changed during execution')
        report.update(worker_id=worker_id, worker_count=worker_count,
            end_to_end_seconds=time.perf_counter()-started,
            source_implementation_sha256=plan['source_implementation_sha256'],
            assigned_chromosomes=[entry['name'] for entry in configuration['chromosomes']],
            file_descriptors=report['output'].get('file_descriptors',file_admission),
            assigned_trait_indices=assigned_traits,assigned_trait_tiles=dispatch['workers'][worker_id]['trait_tiles'],association_partition_strategy=dispatch['strategy'],
            association_dispatch_sha256=dispatch['dispatch_binding_sha256'],
            assigned_genetic_schedule_sha256=dispatch['workers'][worker_id]['genetic_schedule_sha256'])
        if dispatch['strategy']=='whole_chromosomes':
            report.update(schedule_audit.finish())
        atomic_json(prefix.with_name(prefix.name+'.report.private.json'), report)
        exit_code = 0
        return report
    except BaseException as error:
        try:
            atomic_json(prefix.with_name(prefix.name+'.failed.private.json'), dict(error_type=type(error).__name__,
                error=str(error), traceback=traceback.format_exc()))
        except BaseException:
            pass
        raise
    finally:
        original_error = sys.exc_info()[0] is not None
        cleanup_error = None
        try:
            _close_association_resources(writer, repository, failed=exit_code != 0)
        except BaseException as error:
            cleanup_error, exit_code = error, 1
        try:
            torch.set_num_threads(previous_threads)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
            exit_code = 1
        if cleanup_error is not None and not original_error:
            try:
                atomic_json(prefix.with_name(prefix.name+'.failed.private.json'),
                    dict(error_type=type(cleanup_error).__name__, error=str(cleanup_error),
                         traceback=''.join(traceback.format_exception(type(cleanup_error),cleanup_error,
                                                                     cleanup_error.__traceback__)),
                         phase='worker_cleanup'))
            except BaseException:
                pass
        try:
            atomic_json(prefix.with_name(prefix.name+'.exit.anonymous.json'), dict(worker_id=worker_id,
                exit_code=exit_code, elapsed_seconds=time.perf_counter()-started))
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if cleanup_error is not None and not original_error:
            raise cleanup_error


def _dispatch(plan, entries, resources, gpu_ids):
    from .phewas_partition import verify_dispatch_reports
    def verified(reports):
        path = Path(plan['output_directory'])/f'association_dispatch.{resources["attempt_id"]}.private.json'
        dispatch = json.loads(path.read_text())
        completion = verify_dispatch_reports(dispatch,reports)
        for report in reports:
            report['global_completion'] = completion
        return reports
    if len(gpu_ids) == 1:
        with _inline_cpu_policy(resources):
            return verified([_worker(plan, entries, resources, worker_id=0, worker_count=1,
                                     device=f'cuda:{gpu_ids[0]}')])
    context = mp.get_context('spawn')
    processes = [context.Process(target=_worker, kwargs=dict(plan=plan, entries=entries, resources=resources,
        worker_id=i, worker_count=len(gpu_ids), device=f'cuda:{device}', bind_affinity=True))
        for i, device in enumerate(gpu_ids)]
    names = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    previous = {name: os.environ.get(name) for name in names}
    owned = []
    try:
        for name in names:
            os.environ[name] = str(resources['cpu_threads'])
        for process in processes:
            process.start()
            owned.append(process)
    except BaseException:
        for process in owned:
            if process.is_alive():
                process.terminate()
        for process in owned:
            process.join()
        raise
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    while any(process.is_alive() for process in owned):
        if any(process.exitcode not in (None, 0) for process in owned):
            # These exact Process children belong solely to this invocation.
            for process in owned:
                if process.is_alive():
                    process.terminate()
            break
        for process in owned:
            process.join(timeout=.1)
    for process in owned:
        process.join()
    if any(process.exitcode != 0 for process in owned):
        raise RuntimeError('prepared association worker failed; inspect the private attempt receipts')
    output = Path(plan['output_directory'])
    return verified([json.loads((output/f'worker_{i}.attempt_{resources["attempt_id"]}.report.private.json').read_text())
                     for i in range(len(owned))])


def run_prepared_association(phenotype_csv, covariate_csv, prepared_directory, *, output_directory,
        gpu_ids, chromosomes=None, transform='none', null_fit_mode='fp64', **options):
    """Run every independent modern trait through the shared cache backend."""
    if null_fit_mode != 'fp64':
        raise ValueError('shared prepared PheWAS null fitting requires fp64')
    from .phewas_run import _validate_host_memory_limit
    if type(options.get('rm_nan', True)) is not bool:
        raise ValueError('rm_nan must be a boolean')
    if type(options.get('binary_only',False)) is not bool:
        raise ValueError('binary_only must be a boolean')
    options.setdefault('rm_nan', True)
    options['host_memory_limit_gib'] = _validate_host_memory_limit(options.get('host_memory_limit_gib'))
    cpu_resources = _cpu_resources(options['cpu_threads'])
    root, manifest_path, manifest, entries, population = run._dataset(prepared_directory, chromosomes)
    phenotype_csv, covariate_csv = Path(phenotype_csv).resolve(), Path(covariate_csv).resolve()
    if options.get('binary_only'):
        loaded_input_stats = {str(path):_input_file_identity(path) for path in (phenotype_csv,covariate_csv)}
    if options.get('binary_only'):
        phenotypes, covariates = run.read_csv_inputs(phenotype_csv, covariate_csv,
            host_memory_limit_gib=options['host_memory_limit_gib'])
    else:
        phenotypes, covariates = run.read_csv_inputs(phenotype_csv, covariate_csv)
    if options.get('binary_only'):
        if any(_input_file_identity(path)!=identity for path,identity in loaded_input_stats.items()):
            raise ValueError('binary input source changed while the phenotype matrices were read')
        options['_loaded_input_stats'] = loaded_input_stats
    binary_only = options.get('binary_only',False)
    families = (run._validate_binary_phenotypes(phenotypes) if binary_only else
        run._family_selection(options['phenotype_families'], phenotypes.names, phenotypes.values, rm_nan=options['rm_nan']))
    if binary_only and transform!='none':
        raise ValueError('binary WGS requires transform=none')
    selections = _covariate_selection(options['phenotype_covariates'], phenotypes.names, covariates.names)
    kinship_file, kinship = None, None
    if options['grm'] == 'sparse':
        sidecar = (manifest.get('phewas') or {}).get('kinship_npz')
        if options['grm_file'] is not None:
            kinship_file = Path(options['grm_file']).resolve()
        elif sidecar:
            kinship_file = run._cache_path(root, sidecar)
        else:
            raise ValueError('sparse GRM requires grm_file or the ID-bound phewas.kinship_npz sidecar')
        kinship = SparseKinshipData.load_npz(kinship_file)
    output = Path(output_directory).resolve()
    source = run._implementation_sha256()
    configuration = _configuration(manifest, entries, sample_keys(population), output, options, source)
    files = _source_files(root, manifest_path, manifest, entries, phenotype_csv, covariate_csv, kinship_file)
    input_binding = dict(prepared_directory=str(root), dataset_sha256=files[str(manifest_path)],
                         files=files, genotype_streams=_stream_bindings(entries))
    dictionary_sha = trait_index_sha256(phenotypes.names, families)
    scientific = dict(input_binding=input_binding, configuration=configuration, transform=transform,
        families=families, covariate_selection=selections, trait_names=list(phenotypes.names),
        grm=options['grm'], rm_nan=options['rm_nan'], binary_only=binary_only,
        trait_batch_size=options['trait_batch_size'], trait_index_sha256=dictionary_sha)
    scientific_sha = _hash(scientific)
    plan = dict(schema_version=1, workflow='prepared_fid_iid_shared_association',
        source_implementation_sha256=source, scientific_binding_sha256=scientific_sha,
        output_directory=str(output), input_binding=input_binding, configuration=configuration,
        families=families, covariate_selection=selections, transform=transform, rm_nan=options['rm_nan'],
        binary_only=binary_only, dataset=manifest,
        trait_count=len(phenotypes.names), trait_index_sha256=dictionary_sha,
        host_memory_limit_gib=options['host_memory_limit_gib'],
        association_partition_strategy=('whole_chromosomes' if binary_only else options.get('association_partition_strategy','cohort_tiles')),
        csv_output_layout='per_trait_checkpoint')
    from .phewas_partition import gpu_model_cache_capacity
    plan['association_model_cache_capacity_bytes'] = gpu_model_cache_capacity(gpu_ids,
        memory_limit_gib=options['memory_limit_gib'],reserve_bytes=int(options['memory_reserve_gib']*2**30))
    if output.exists() and any(output.iterdir()):
        if not options['resume']:
            raise FileExistsError('full association requires a fresh output directory; choose resume=True')
        existing = json.loads((output/'plan.private.json').read_text())
        resource_fields = {'host_memory_limit_gib','association_model_cache_capacity_bytes'}
        if ({key:value for key,value in existing.items() if key not in resource_fields} !=
                {key:value for key,value in plan.items() if key not in resource_fields}):
            raise ValueError('resume prepared association source/input/schedule binding differs')
        if existing != plan:
            atomic_json(output/f'plan.previous.{uuid.uuid4().hex}.private.json', existing)
            atomic_json(output/'plan.private.json', plan)
    else:
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        (output/'models').mkdir(mode=0o700)
        atomic_json(output/'plan.private.json', plan)
    _verify_inputs(input_binding)
    write_trait_index(output, phenotypes.names, families, expected_sha256=dictionary_sha)
    with _inline_cpu_policy(dict(cpu_resources, cpu_threads=options['cpu_threads'])):
        entries = _fit_models(plan, phenotypes, covariates, population, kinship, options, f'cuda:{gpu_ids[0]}')
    _verify_inputs(input_binding)
    del kinship, phenotypes, covariates
    gc.collect()
    resources = {key: options[key] for key in ('cpu_threads', 'memory_limit_gib', 'memory_reserve_gib', 'host_memory_limit_gib',
        'trait_batch_size', 'device_cache_bytes', 'compact_cache_bytes', 'metadata_cache_bytes', 'resume')}
    resources.update(cpu_resources)
    resources['attempt_id'] = uuid.uuid4().hex
    return _dispatch(plan, entries, resources, gpu_ids)
