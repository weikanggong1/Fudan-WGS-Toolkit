"""Ordered independent association analyses with shared verified cache IO.

Each analysis retains the single-trait pipeline, output groups and model state.
Single scans are frame-major; gene jobs are family-major across all models.
"""
from __future__ import annotations
from collections import Counter, OrderedDict
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
import json
import sys
import inspect
import math
import threading
import time
import traceback
import hashlib
import gc
import weakref
from types import MethodType, SimpleNamespace
import numpy as np
import torch

from .. import cli
from ..io import load_null_model, fit_prepared_input, save_null_model
from ..pipeline import AnalysisOptions, PheWASPipeline
from ..profiling import StageProfiler
from ..precision_audit import DenseProductAudit
from ..tf32 import configure_tf32, validate_split_k, execution_metadata as tf32_metadata
from ..statistics import statistics_execution_metadata
from ..cache_runtime.fast_container import Container
from ..cache_runtime.portable import PortableMetadataReader
from ..csv_output import write_association_batch
from .metadata import SharedMetadataReader
from .mask_limit import LimitedMaskPipeline
from .buffers import EffectiveBuffer
from .shared_state import SharedStateBroker, SharedTraitReader, _immutable, _readonly_source
from .single import (process_single_batches, process_single_shared_genotype,
                     execution_metadata as single_metadata)

_LOCK = threading.Lock()


def _clear_inactive_exception_frames(error):
    """Retain failure type, message and stack positions, releasing leaf locals."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend(value for value in (current.__cause__, current.__context__)
                       if value is not None)


def _gpu_memory_diagnostics(device, repository=None, broker=None):
    """Low-cost allocator/cache snapshots; reserved space is not a live leak."""
    target = torch.device(device)
    result = dict(device_type=target.type,
                  scope='owned process allocator/cache endpoint; not a peak or NVML card total')
    if target.type == 'cuda':
        allocated = int(torch.cuda.memory_allocated(target))
        reserved = int(torch.cuda.memory_reserved(target))
        result.update(allocated_bytes=allocated, reserved_bytes=reserved,
                      unused_reserved_bytes=max(0, reserved-allocated))
    if repository is not None:
        for name, attribute in (('resident_normal_models', 'models'),
                                ('resident_binary_sources', 'spa_sources'),
                                ('shared_design_entries', 'tensor_cache')):
            bank = getattr(repository, attribute, None)
            if bank is not None:
                result[name] = len(bank)
    if broker is not None:
        metrics = broker.metrics
        for key in ('current_gpu_csr_cache_bytes', 'current_sample_axis_device_bytes',
                    'current_compact_cache_bytes', 'current_sample_axis_host_bytes',
                    'current_raw_state_cache_bytes', 'current_raw_state_cache_slabs'):
            if key in metrics:
                result[key] = metrics[key]
    return result


def _resolved_budget(value, device):
    """Resolve None to device capacity; live free space remains an admission gate."""
    if value is not None:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError('memory_limit_gib must be positive finite or None')
        return float(value)
    return float(torch.cuda.get_device_properties(device).total_memory) / 2**30


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def validate_analyses(analyses):
    """Require aligned scientific schedules and separate original output files."""
    if not isinstance(analyses, (list, tuple)) or not analyses:
        raise ValueError('analyses must contain at least one standalone configuration')
    configs = [deepcopy(a) for a in analyses]
    signatures, outputs = [], {}
    input_paths = {str(Path(p[key]).resolve()) for c in configs for p in c.get('phenotypes', [])
                   for key in ('model', 'input', 'sample_indices_file') if key in p}
    shared_keys = ('qc_path', 'annotation_catalog', 'annotation_names', 'analysis_options',
        'statistics_tail_optimization', 'local_mask_reuse',
        'weight_batch_optimization', 'weighted_eigensolver', 'maximum_mask_variants')
    for trait, config in enumerate(configs):
        if len(config.get('phenotypes', [])) != 1:
            raise ValueError('every analysis must contain exactly one independent phenotype')
        if config.get('matmul_mode', 'tf32') != 'tf32' or config.get('precision_control', False):
            raise ValueError('PheWAS production uses native TF32 with FP32 storage')
        if config.get('statistics_execution', 'serial') != 'serial':
            raise ValueError('PheWAS schedules use serial independent cores')
        if config.get('resident_genotypes', True) is not True or config.get('single_batch_optimization', True) is not True:
            raise ValueError('shared PheWAS requires resident genotypes and effective Single batches')
        validate_split_k(config.get('tf32_split_k', 0))
        obsolete = {'tf32_binned_tile_shape', 'tf32_binned_fused_small'} & config.keys()
        if obsolete:raise ValueError('TF32 reconstruction parameters are obsolete')
        for flag in ('precision_control', 'local_mask_reuse', 'weight_batch_optimization',
                     'statistics_tail_optimization', 'stage_profile', 'validation_reference', 'require_single_continuous'):
            if flag in config and type(config[flag]) is not bool:
                raise ValueError(flag + ' must be a JSON boolean')
        cli._weighted_eigensolver_settings(config, 'tf32', 'cuda:0')
        options = config.setdefault('analysis_options', {})
        if options.get('wrapper_semantics', 'base') != 'base':
            raise ValueError('single-trait equivalence requires base wrapper semantics')
        options['wrapper_semantics'] = 'base'
        budget = options.setdefault('memory_limit_gib', 20)
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or not 0 < budget <= 20:
            raise ValueError('PheWAS memory_limit_gib must be finite and in (0, 20]')
        AnalysisOptions(**options)
        for field, default in [('individual_effective_block_size', 1024), ('individual_genotype_block_size', 1024)]:
            value = config.setdefault(field, default)
            if type(value) is not int or value < 1:
                raise ValueError(field + ' must be a positive integer')
        maximum = config.get('maximum_mask_variants')
        if maximum is not None and (type(maximum) is not int or maximum < 1):
            raise ValueError('maximum_mask_variants must be a positive integer or omitted')
        for phenotype in config['phenotypes']:
            if ('model' in phenotype) == ('input' in phenotype):
                raise ValueError('each phenotype requires exactly one model or prepared input')
            fit = dict(phenotype.get('fit_options', {}))
            for key in ('family', 'joint_mode', 'binary_mode'):
                if key in phenotype and key in fit and phenotype[key] != fit[key]:
                    raise ValueError('conflicting phenotype fitting modes')
            if fit.get('matmul_mode', 'tf32') != 'tf32':
                raise ValueError('phenotype matmul_mode conflicts with native TF32')
            validate_split_k(fit.get('tf32_split_k', 0))
            if phenotype.get('joint_mode', fit.get('joint_mode')) is not None:
                raise ValueError('independent PheWAS requires single-trait models')
            for key in ('save_model',):
                if key not in phenotype:continue
                if Path(phenotype[key]).suffix.lower() != '.npz':
                    raise ValueError('saved models must use the .npz format')
                path = str(Path(phenotype[key]).resolve())
                if path in outputs or path in input_paths:
                    raise ValueError('null/model outputs must be distinct from every input and output')
                outputs[path] = (trait, key, None)
        schedule = []
        for chromosome in config.get('chromosomes', []):
            jobs = []
            for job in chromosome.get('jobs', []):
                if job.get('kind') not in ('individual', 'coding', 'noncoding', 'ncrna') or job.get('layout', 'base') != 'base':
                    raise ValueError('PheWAS accepts base Individual, coding, noncoding and ncRNA jobs')
                arguments = job.get('arguments', {})
                if not isinstance(arguments, dict):raise ValueError('job arguments must be an object')
                signature_args = inspect.signature(getattr(PheWASPipeline, job['kind'])).parameters
                permitted = set(signature_args) - {'self', 'chromosome'}
                if job['kind'] == 'noncoding':permitted.add('promoter_intervals_file')
                if set(arguments) - permitted:raise ValueError('unknown association job arguments')
                if job['kind'] == 'individual':
                    for key, default in (('mac_cutoff',20), ('subset_variants_num',5000)):
                        value = arguments.get(key, default)
                        if type(value) is not int or value < 1:raise ValueError(key + ' must be a positive integer')
                    if arguments.get('variant_type', 'variant') not in ('variant', 'SNV', 'Indel'):
                        raise ValueError('invalid Single variant_type')
                    if (arguments.get('start') is None) != (arguments.get('end') is None):
                        raise ValueError('provide both Single endpoints or neither')
                if not isinstance(job.get('output'), str) or Path(job['output']).suffix.lower() != '.csv':
                    raise ValueError('association outputs must be CSV files')
                path = str(Path(job['output']).resolve())
                signature = (job['kind'], job.get('object_name'), job.get('layout', 'base'))
                if path in input_paths:
                    raise ValueError('association output cannot overwrite an input')
                if path in outputs:
                    owner, role, previous = outputs[path]
                    if owner != trait or role != 'association' or previous != signature or job['kind'] == 'individual':
                        raise ValueError('output collisions are allowed only for one trait and compatible gene batches')
                outputs[path] = (trait, 'association', signature)
                jobs.append({k: job.get(k) for k in ('name', 'kind', 'arguments', 'object_name', 'layout')})
            schedule.append({'name': chromosome['name'], 'genotype': str(Path(chromosome['genotype']).resolve()),
                             'annotation_index': chromosome.get('annotation_index'), 'jobs': jobs})
        if not schedule or not any(c['jobs'] for c in schedule):
            raise ValueError('provide a nonempty chromosome job schedule')
        signatures.append(_canonical({'settings': {k: config.get(k) for k in shared_keys}, 'schedule': schedule}))
    if len(set(signatures)) != 1:
        raise ValueError('independent analyses must share genotype, annotations, settings and ordered job arguments')
    return configs


class _CSVOutputs:
    def __init__(self, config):
        self.remaining = Counter(str(Path(j['output']).resolve()) for c in config['chromosomes'] for j in c['jobs'])
        self.groups, self.seconds, self.files = {}, 0., 0
        self.proofs = []

    def append(self, job, result):
        path = Path(job['output']);key = str(path.resolve())
        signature = (job['kind'], job.get('object_name'), job.get('layout', 'base'))
        group = self.groups.setdefault(key, {'path': path, 'signature': signature, 'results': []})
        if group['signature'] != signature:
            raise ValueError('one output file cannot combine different table layouts or kinds')
        if job['kind'] == 'individual' and group['results']:
            raise ValueError('Individual jobs require separate CSV files')
        group['results'].append(result);self.remaining[key] -= 1
        if self.remaining[key] == 0:
            start = time.perf_counter();path.parent.mkdir(parents=True, exist_ok=True)
            proof = write_association_batch(path, group['results'], kind=signature[0], layout=signature[2],
                exclude_columns=job.get('exclude_columns', ()), empty_columns=job.get('empty_columns'))
            self.proofs.append(proof)
            self.seconds += time.perf_counter()-start;self.files += 1
            group['results'].clear()

    def check(self):
        if any(self.remaining.values()) or any(g['results'] for g in self.groups.values()):
            raise RuntimeError('incomplete CSV output schedule')


def _load_models(configs, device):
    models, rows = [], []
    for config in configs:
        p = config['phenotypes'][0]
        if 'model' in p:
            model = load_null_model(p['model'], device=device, matmul_mode='tf32')
            row = np.load(p['sample_indices_file'], allow_pickle=False) if 'sample_indices_file' in p else None
        else:
            fit = dict(p.get('fit_options', {}))
            for key in ('family', 'binary_mode'):
                if key in p:fit[key] = p[key]
            fit.pop('tf32_split_k', None)
            family=fit.get('family', 'gaussian')
            if family != 'gaussian':raise ValueError('binary PheWAS requires a complete prefitted non-SPA state')
            fit['matmul_mode'] = 'tf32'
            model, row = fit_prepared_input(p['input'], device=device, transform=p.get('transform', 'none'), **fit)
        if model.n_pheno != 1 or model.use_spa or model.family not in ('gaussian', 'binomial'):
            raise ValueError('PheWAS requires independent single-trait Gaussian or non-SPA binary models')
        if model.family == 'binomial' and 'reference' in model.fit_method.lower() and not config.get('validation_reference', False):
            raise ValueError('a prefitted binary reference cache requires explicit validation_reference')
        if config.get('require_single_continuous', False) and model.family != 'gaussian':
            raise ValueError('require_single_continuous conflicts with a binary model')
        if model.x.dtype != torch.float32 or model.matmul_mode != 'tf32':
            raise RuntimeError('model storage must remain native FP32')
        models.append(model);rows.append(row)
    return models, rows


def _annotation_metadata_source(pipeline):
    genotype = pipeline.genotype
    return genotype._broker._reader if isinstance(genotype, SharedTraitReader) else genotype


def _shared_annotation_categories(self):
    """Publish the mature uint8 decode only after a complete successful scan."""
    shared = self._shared_annotation_state
    with shared['lock']:
        if shared['codes'] is None:
            self._category_codes = None
            try:
                codes = PheWASPipeline._annotation_categories(self)
            except BaseException:
                # The mature loader fills its array in place. A failed block
                # must never authorize a partial category array on a retry.
                self._category_codes = None
                raise
            codes.flags.writeable = False
            shared['codes'] = codes
        self._category_codes = shared['codes']
        return self._category_codes


def _share_annotations(pipelines):
    first = pipelines[0]
    shared = None
    if isinstance(first, PheWASPipeline):
        shared = getattr(first, '_shared_annotation_state', None)
        if shared is None:
            codes = first._category_codes
            if codes is not None:
                codes.flags.writeable = False
            shared = dict(source=_annotation_metadata_source(first), options=first.options,
                          catalog=dict(first.annotation_catalog), names=tuple(first.annotation_names),
                          codes=codes, lock=threading.Lock())
        for pipeline in pipelines:
            if (_annotation_metadata_source(pipeline) is not shared['source']
                    or pipeline.options != shared['options']
                    or pipeline.annotation_catalog != shared['catalog']
                    or tuple(pipeline.annotation_names) != shared['names']):
                raise ValueError('shared annotations require identical metadata source and options')
        for pipeline in pipelines:
            pipeline._shared_annotation_state = shared
            # A strong instance-bound method would create a cycle retaining
            # each short-lived selector's models/reader until cyclic GC.
            pipeline._annotation_categories = MethodType(_shared_annotation_categories, weakref.proxy(pipeline))
            pipeline._category_codes = shared['codes']
    for pipeline in pipelines[1:]:
        pipeline._base_masks = first._base_masks
        pipeline._category_codes = first._category_codes
        pipeline._annotation_indexes = first._annotation_indexes
        pipeline._coding_masks_cache = first._coding_masks_cache


def _reader_metadata_diagnostics(metadata):
    """Allowlisted endpoint counters; never copy reader paths or sample keys."""
    values = metadata.reader_metadata
    result = {}
    for key in ('metadata_reads', 'metadata_read_seconds', 'metadata_checked_files',
                'metadata_checksum_verification_seconds', 'metadata_checksums_enabled'):
        value = values.get(key)
        if isinstance(value, (int, float, bool)) and math.isfinite(value):
            result[key] = value
    return result


def _single(pipelines, broker, arguments, configs):
    chromosome=arguments['chromosome'];mac=arguments.get('mac_cutoff',20)
    subset=arguments.get('subset_variants_num',5000);variant_type=arguments.get('variant_type','variant')
    start, end = arguments.get('start'), arguments.get('end')
    if (start is None) != (end is None):raise ValueError('provide both Single endpoints or neither')
    first = pipelines[0]
    if start is None:
        selected = np.flatnonzero(first._base_mask(chromosome, variant_type))
    else:
        selected=first.region_indices(start,end)
        if variant_type == 'variant':selected=selected[first.qc[selected]=='PASS']
        else:
            from ..masks import variant_filter
            a=first.annotations(selected,include_weights=False,metadata='mask')
            selected=selected[variant_filter(a,variant_type)]
    buffers=[EffectiveBuffer(p.union_rows,c.get('individual_effective_block_size',1024),allocation_guard=broker._guard) for p,c in zip(pipelines,configs)]
    ordinals=[0]*len(pipelines);records=[[] for _ in pipelines]
    scan_step=configs[0]['individual_genotype_block_size']
    offset=0
    def consume(tail=False):
        nonlocal ordinals
        ready=[buffer.take(tail=tail) for buffer in buffers]
        if all(x is None for x in ready):return False
        result,ordinals=process_single_batches(pipelines,ready,ordinals,chromosome,mac,subset)
        for existing, block_rows in zip(records,result):existing.extend(block_rows)
        return True
    while offset < len(selected):
        frame=int(np.searchsorted(broker._starts,selected[offset],side='right')-1)
        stop=int(np.searchsorted(selected,broker._starts[frame]+broker._sizes[frame],side='left'))
        for begin in range(offset,stop,scan_step):
            raw=broker.read_states(selected[begin:min(begin+scan_step,stop)],minimum_mac_bound=mac)
            blocks=broker.trait_blocks(raw,[p.union_rows for p in pipelines],minimum_mac=mac)
            for block,buffer in zip(blocks,buffers):buffer.append(block)
            del blocks,block,raw
            while consume():pass
        offset=stop
    while consume(tail=True):pass
    return [p.individual_tables([rows]) for p,rows in zip(pipelines,records)]


def _descriptor_groups(repository):
    """Group exactly equal ordered cohorts; fitted projection keys are not reused."""
    descriptors = list(repository.descriptors())
    if not descriptors:
        raise ValueError('model_repository must expose at least one fitted phenotype')
    groups, by_digest, seen = [], {}, set()
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise ValueError('model descriptors must be dictionaries')
        index = descriptor.get('trait_index')
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError('trait_index must be a distinct nonnegative integer')
        seen.add(index)
        raw = np.asanyarray(descriptor.get('ordered_rows'))
        if (raw.ndim != 1 or raw.dtype.kind not in 'iu' or not len(raw)
                or descriptor.get('n') != len(raw) or descriptor.get('family') not in ('gaussian', 'binomial')):
            raise ValueError('descriptor requires n/family and a nonempty ordered integer sample axis')
        rows = raw if raw.dtype == np.dtype(np.int64) else np.asarray(raw, dtype=np.int64)
        row_bytes = memoryview(rows).cast('B') if rows.flags.c_contiguous else rows.tobytes()
        key = (len(rows), hashlib.sha256(row_bytes).digest())
        del row_bytes
        group = by_digest.get(key)
        if group is None or not np.array_equal(rows, group['rows']):
            # Verified read-only model-store mmaps retain their source binding.
            # Other callers get one owned immutable copy; a reversible readonly
            # flag alone never authorizes a scan-free axis alias.
            group = dict(rows=(rows if _readonly_source(rows) is not None else _immutable(rows)), descriptors=[])
            groups.append(group)
            by_digest[key] = group
        group['descriptors'].append(descriptor)
    return descriptors, groups


def _thin_pipeline(gds, model, rows, options, configuration, structural=None):
    """Use mature pipeline kernels without repeatedly rebuilding large sample maps.

    The broker has range/uniqueness-bound ``rows`` and the repository model is
    independently verified below. Single-model row order is retained directly.
    Only model-independent structural annotations are shared across tiles.
    """
    p = object.__new__(LimitedMaskPipeline)
    p.genotype, p.models, p.options = gds, [model], options
    p.annotation_catalog = configuration.get('_resolved_annotation_catalog', {})
    p.annotation_names = list(configuration.get('annotation_names', []))
    p.union_ids = None
    p.union_rows = rows
    if isinstance(gds, SharedTraitReader):
        binding = gds._single_axis_binding()
        canonical = binding.axis.samples
        if rows is not canonical and gds._broker._axis(rows) is not binding.axis:
            raise ValueError('thin pipeline rows differ from its verified reader axis')
        # Structural gene selectors carry only device/use_spa metadata. Their
        # dimension comes from the verified reader; real model dimensions are
        # still matched here and their IDs are checked by _check_model.
        binding.validate(gds, canonical, binding.axis.identity_rows, getattr(model, 'n', len(canonical)))
        p.union_rows = canonical
        p._single_axis_binding = binding
        p.trait_rows = [binding.axis.identity_rows]
    else:
        if not hasattr(gds, '_identity_rows'):
            gds._identity_rows = _immutable(np.arange(len(rows), dtype=np.int64))
        p.trait_rows = [gds._identity_rows]
    p.position = (np.asarray(gds.read_field('position'), dtype=np.int64)
                  if structural is None else structural.position)
    p.qc_path = configuration.get('qc_path', 'annotation/filter')
    p.qc = gds.read_field(p.qc_path) if structural is None else structural.qc
    p.skipped_sets = []
    p._position_sorted = (bool(np.all(p.position[1:] >= p.position[:-1]))
                          if structural is None else structural._position_sorted)
    p._base_masks, p._annotation_indexes, p._coding_masks_cache = {}, {}, {}
    p._category_codes = None
    p._test_set_cache = OrderedDict()
    p.batch_diagnostics, p.covariance_diagnostics = [], []
    p.statistics_execution = 'serial'
    p.statistics_tail_optimization = configuration.get('statistics_tail_optimization', True)
    p.weight_batch_optimization = configuration.get('weight_batch_optimization', True)
    p.local_mask_reuse = False
    p.local_mask_reuse_counters = dict.fromkeys(('families', 'union_score_calls', 'reused_masks',
        'input_variant_columns', 'union_variant_columns', 'prepared_union_variant_columns',
        'fallback_memory', 'fallback_unsupported', 'fallback_mapping', 'cache_hits',
        'duplicate_mask_hits', 'single_mask_paths', 'fallback_cuda_oom', 'fallback_geometry',
        'fallback_host_reused_masks', 'geometry_checks', 'geometry_union_covariance_cells',
        'geometry_mask_covariance_cells'), 0)
    p.statistics_tail_optimization_calls = 0
    p.profiler = configuration.get('_stage_profiler') or StageProfiler(model.device, enabled=configuration.get('stage_profile', False))
    p.resident_genotypes = True
    p.single_batch_optimization = True
    p.individual_effective_block_size = configuration.get('individual_effective_block_size', 1024)
    p.host_memory_guard = configuration.get('host_memory_guard')
    p.host_memory_set_future = configuration.get('_host_memory_set_future')
    p.hybrid_mask_sizes, p.hybrid_union_M = [], 0
    p.maximum_mask_variants = configuration.get('maximum_mask_variants')
    if structural is not None:
        _share_annotations([structural, p])
    # Borrow one worker-lifecycle owner across traits/cohorts/chromosomes.
    # It is lazy: Single, binary and ordinary masks create no CUDA streams.
    owner = configuration.get('_association_lobpcg_owner')
    if owner is not None:
        p._enable_fastskat_lobpcg(owner=owner)
    return p


def _check_model(model, descriptor, metadata, verified):
    index = descriptor['trait_index']
    if (model.n != descriptor['n'] or model.n_pheno != 1 or model.family != descriptor['family']
            or model.use_spa or model.matmul_mode != 'tf32' or model.x.dtype != torch.float32):
        raise ValueError('acquired model differs from its independent native FP32 descriptor')
    if index not in verified:
        rows = descriptor['ordered_rows']
        identifiers = np.asarray(metadata.sample_ids())[rows]
        actual = np.asarray(getattr(model, 'genotype_sample_ids', getattr(model, 'gds_sample_ids', model.sample_ids)), dtype=str)
        if not np.array_equal(np.asarray(identifiers, dtype=str), actual):
            raise ValueError('fitted model IDs differ from the portable ordered sample binding')
        verified.add(index)


def _prepare_host_guard(pipeline, model, selected_m, materialized_bytes=0):
    """Replace future preparation bytes using only initialized owned arrays."""
    guard = getattr(pipeline, 'host_memory_guard', None)
    if guard is None:
        return
    if callable(getattr(pipeline, 'host_memory_set_future', None)):
        guard(n=model.n, m=int(selected_m), phase='prepare',
              materialized_bytes=int(materialized_bytes))
    else:
        # Injected/older guards retain their original three-argument contract.
        guard(n=model.n, m=int(selected_m), phase='prepare')


def _model_cache_live_fraction(configuration, descriptors):
    """Retain a binary fitted sweep while keeping live workspace admission.

    A pure binary sweep uses small burden covariances rather than variant
    covariances. Its independent fitted states can therefore keep a larger
    optional LRU. Explicit caller bounds still win, and allocation continues
    to evict unpinned entries whenever actual workspace needs more memory.
    """
    if 'model_cache_live_fraction' in configuration:
        return configuration['model_cache_live_fraction']
    return .75 if descriptors and all(d['family'] == 'binomial' for d in descriptors) else .5


def _binary_gene_workspace_new_bytes(n, m, q, annotations, *, resident_genotype=True):
    """Conservative NEW resident burden/SPA bytes; existing G is counted live.

    B=2*(K+1) burden columns determine projection and covariance geometry.
    Native burdens may select a second FP32 G (4*N*M). Selective exact SPA
    owns at most one 512-variant gather plus its FP64 cast (12*N*panel),
    FP64 weights, exact/projected/scaled burden arrays, active-column gathers,
    derivative/cumulant temporaries, and bounded small covariance products.
    Twenty-four FP64 N*B arrays conservatively cover the original SPA peak;
    four FP32 N*B arrays cover native burdens and weighted projections.
    No variant M*M covariance is allocated by this inference route. Fitted
    bank/source uploads retain their separate repository admission.
    """
    b = 2*(annotations+1)
    panel = min(m, 512)
    # Host input streams the original FP32 burden panel; it never needs a
    # full CUDA G. Retaining that panel alongside exact SPA is conservative.
    selected_genotype = 4*n*(m if resident_genotype else panel)
    spa_panel = 12*n*panel
    native_and_spa_burdens = (4*4 + 24*8)*n*b
    # Retained FP32 weights, selected FP64 weights, annotation copies and
    # selection vectors; include covariate/burden and B*B temporary products.
    weights_and_selection = 12*m*b + 16*m*annotations + 64*m
    projections_and_covariance = 192*q*b + 32*b*b
    return (selected_genotype + spa_panel + native_and_spa_burdens
            + weights_and_selection + projections_and_covariance + 256*2**20)


def _binary_single_spa_workspace_new_bytes(n, m, q, *, variant_tile_size=512):
    """NEW statistical SPA storage, separate from fitted sidecar admission.

    Only one selected variant tile is live. Its FP32 gather, original FP64
    columns, projected/scaled columns, active-column gathers and Bernoulli
    cumulant/root/tail temporaries are covered by 24 FP64 sample slabs. G,
    ordinary score/variance vectors and the fitted SPA model are already owned
    or admitted by their respective caller/repository. The original 512-column
    algorithm is retained; no probability, precision or scientific filter changes.
    """
    width = min(m, variant_tile_size)
    return (4*n*width + 24*8*n*width + 64*q*width
            + 512*width + 256*2**20)


@contextmanager
def _binary_single_spa_acquire(repository, broker, trait_index, *, n, m, q):
    """Evict cold owned state before loading and running selected Single SPA.

    The binary kernel invokes this factory only for a nonempty selected set.
    A workspace lease surrounds sidecar loading and the complete correction,
    so fitted-source/projection admission cannot consume statistical headroom.
    """
    required = _binary_single_spa_workspace_new_bytes(n, m, q)
    with _gene_workspace_scope(repository, broker, required) as workspace:
        with repository.acquire_spa(trait_index, str(broker.device)) as sidecar:
            workspace.ensure_free(phase='single_binary_spa')
            yield sidecar


def _prepare_shared_gene_device(pipeline, indices, broker, phred, names):
    """Keep one small cohort/mask on device, preserving the original F layout.

    This route is optional memory admission, never a scientific filter. A
    rejected resident workspace uses the established bounded host route. Each
    cohort still owns its allele summaries, eligibility and imputed dosages.
    """
    model, options = pipeline.models[0], pipeline.options
    n, candidate_m = model.n, len(indices)
    # Preserve the mature linear bound for raw slabs, final G, indexing/count
    # scratch and cMAC. Binary inference projects only burden columns, so its
    # preparation must not reserve an unused variant covariance.
    binary_only = getattr(pipeline, '_binary_burden_only',
                          getattr(model, 'family', None) == 'binomial')
    covariance_bytes = 0 if binary_only else 32*candidate_m*candidate_m
    broker._guard(12*n*candidate_m + covariance_bytes + 256*2**20)
    parts, physical, frequencies, groups = [], [], [], []
    prefiltered = 0
    for offset in range(0, candidate_m, options.genotype_block_size):
        selected = indices[offset:offset+options.genotype_block_size]
        raw = broker.read_states(selected)
        block = broker.trait_block(raw, pipeline.union_rows)
        del raw
        alt = 1-block.union_ref_af
        source_maf = np.where(block.union_ref_af >= alt, alt, block.union_ref_af)
        prefilter = np.isfinite(source_maf) & (source_maf > 0) & (source_maf < options.rare_maf_cutoff)
        prefiltered += int(prefilter.sum())
        if prefiltered >= options.rv_num_cutoff_max_prefilter:
            raise ValueError('union-prefilter variant count reaches rv_num_cutoff_max_prefilter')
        maf = block.trait_summary(pipeline.trait_rows[0], options.imputation, frequency_mode='reference')[0]
        keep = prefilter & np.isfinite(maf) & (maf > 0) & (maf < options.rare_maf_cutoff)
        columns = np.flatnonzero(keep)
        if len(columns):
            physical.extend((offset+columns).tolist())
            frequencies.extend(maf[columns])
            missing = block.allele_missing_rate()
            grouping = np.where(alt > .5, 2, np.where((source_maf >= .01) | (missing >= .01), 1, 0))
            groups.extend(grouping[columns])
            parts.append((block, columns))
        else:
            del block
    count = len(physical)
    if count < options.rv_num_cutoff:
        return None
    if count >= options.rv_num_cutoff_max:
        raise ValueError('rare variant count reaches rv_num_cutoff_max')
    maximum = pipeline.maximum_mask_variants
    if maximum is not None and count > maximum:
        pipeline.skipped_sets.append(dict(eligible_variants=count, reason='eligible_M_exceeds_configured_limit'))
        return None
    group = np.asarray(groups, dtype=np.int64)
    order = np.argsort(group, kind='stable')
    inverse = np.empty(count, dtype=np.int64); inverse[order] = np.arange(count)
    # Baseline NumPy F-contiguous -> CUDA F-strided G fixes native GEMV order.
    genotype = torch.empty((count, n), dtype=torch.float32, device=broker.device).T
    begin = 0
    for block, columns in parts:
        end = begin+len(columns)
        block.trait_dense_columns(pipeline.trait_rows[0], columns, options.imputation,
            frequency_mode='reference', dtype=torch.float32, out=genotype, out_columns=inverse[begin:end])
        begin = end
    parts.clear(); del block
    physical = np.asarray(physical, dtype=np.int64)[order]
    maf = np.asarray(frequencies, dtype=np.float64)[order]
    from ..pipeline import _cmac_scalar_sum
    return dict(_genotype_device=genotype, _variant_indices=indices[physical], cmac=_cmac_scalar_sum(genotype),
        _extraction_groups=group[order], maf=maf, mac=np.rint(maf*2*n), annotations=phred[physical], names=names,
        acat_calibration='chi2', rare_maf_cutoff=options.rare_maf_cutoff,
        rv_num_cutoff=options.rv_num_cutoff, rv_num_cutoff_max=options.rv_num_cutoff_max)


def _prepare_shared_gene(pipeline, indices, broker):
    """Prepare one cohort/mask once, retaining bounded CPU uint8 parts.

    Source allele summaries, cohort rare filtering, and extraction groups use
    the mature helpers. A sparse binary precision never invokes a variant M²
    covariance merely to prepare G. No full candidate dosage remains on CUDA.
    """
    from ..masks import annotation_phred_matrix
    indices = np.asarray(indices, dtype=np.int64)
    model, options = pipeline.models[0], pipeline.options
    annotations = pipeline.annotations(indices, metadata='weights')
    phred, names = annotation_phred_matrix(annotations.annotations, pipeline.annotation_names,
        variant_type=options.variant_type, number_variants=len(indices))
    # Candidate geometry bounds raw ownership before counts are known. Large
    # masks retain the old tiled host path; no eligible mask is skipped.
    if (torch.device(broker.device).type == 'cuda' and len(indices) <= options.long_mask_threshold
            and pipeline.resident_genotypes):
        try:
            return _prepare_shared_gene_device(pipeline, indices, broker, phred, names)
        except (MemoryError, torch.cuda.OutOfMemoryError):
            # Exception-local tensors lose their owners here. Recompute only
            # preparation using the unchanged bounded CPU route.
            pass
    parts, physical, frequencies, groups = [], [], [], []
    prefiltered = 0
    for offset in range(0, len(indices), options.genotype_block_size):
        selected = indices[offset:offset+options.genotype_block_size]
        raw = broker.read_states(selected)
        block = broker.trait_block(raw, pipeline.union_rows)
        del raw
        alt = 1 - block.union_ref_af
        source_maf = np.where(block.union_ref_af >= alt, alt, block.union_ref_af)
        prefilter = np.isfinite(source_maf) & (source_maf > 0) & (source_maf < options.rare_maf_cutoff)
        prefiltered += int(prefilter.sum())
        if prefiltered >= options.rv_num_cutoff_max_prefilter:
            raise ValueError('union-prefilter variant count reaches rv_num_cutoff_max_prefilter')
        maf = block.trait_summary(pipeline.trait_rows[0], options.imputation,
                                  frequency_mode='reference')[0]
        keep = prefilter & np.isfinite(maf) & (maf > 0) & (maf < options.rare_maf_cutoff)
        columns = np.flatnonzero(keep)
        if len(columns):
            # Compact CPU parts are already host allocations; obtain/extend
            # the owner lease before retaining them, not only before dense G.
            _prepare_host_guard(pipeline, model, len(physical)+len(columns),
                sum(part.dosage.numel()*part.dosage.element_size() for part in parts))
            compact = block.select_columns(columns)
            # This handoff contains exact six-state-derived minor dosage, not
            # phenotype/model data. CPU conversion/imputation retains FP32.
            compact = compact.to_host()
            physical.extend((offset+columns).tolist())
            frequencies.extend(maf[columns])
            allele_missing = block.allele_missing_rate()
            group = np.where(alt > .5, 2, np.where((source_maf >= .01) | (allele_missing >= .01), 1, 0))
            groups.extend(group[columns])
            parts.append(compact)
        del block
    count = len(physical)
    if count < options.rv_num_cutoff:
        return None
    if count >= options.rv_num_cutoff_max:
        raise ValueError('rare variant count reaches rv_num_cutoff_max')
    maximum = pipeline.maximum_mask_variants
    if maximum is not None and count > maximum:
        pipeline.skipped_sets.append(dict(eligible_variants=count, reason='eligible_M_exceeds_configured_limit'))
        return None
    parts_bytes = sum(part.dosage.numel()*part.dosage.element_size() for part in parts)
    _prepare_host_guard(pipeline, model, count, parts_bytes)
    host = np.empty((model.n, count), dtype=np.float32, order='F')
    group = np.asarray(groups, dtype=np.int64)
    order = np.argsort(group, kind='stable')
    inverse = np.empty(count, dtype=np.int64)
    inverse[order] = np.arange(count)
    begin = 0
    host_tensor = torch.from_numpy(host)
    for part in parts:
        # The host lease includes one 128-column conversion, never a full
        # physical frame. Only filled host columns qualify as materialized.
        for local in range(0, part.shape[1], 128):
            stop = min(local+128, part.shape[1])
            end = begin+stop-local
            part.trait_dense_columns(pipeline.trait_rows[0],
                np.arange(local, stop, dtype=np.int64), options.imputation,
                frequency_mode='reference', dtype=torch.float32,
                out=host_tensor, out_columns=inverse[begin:end])
            begin = end
            _prepare_host_guard(pipeline, model, count, parts_bytes+4*model.n*begin)
    physical = np.asarray(physical, dtype=np.int64)[order]
    maf = np.asarray(frequencies, dtype=np.float64)[order]
    from ..pipeline import _cmac_scalar_sum
    return dict(_genotype_host=host, _variant_indices=indices[physical], cmac=_cmac_scalar_sum(host),
        _extraction_groups=group[order], maf=maf, mac=np.rint(maf*2*model.n),
        annotations=phred[physical], names=names, acat_calibration='chi2',
        rare_maf_cutoff=options.rare_maf_cutoff, rv_num_cutoff=options.rv_num_cutoff,
        rv_num_cutoff_max=options.rv_num_cutoff_max)


def _shared_gene_scores(genotype, models, broker, variant_tile_size=512):
    from .scores import fp32_score_columns
    n, m = genotype.shape
    resident = isinstance(genotype, torch.Tensor) and genotype.is_cuda
    panel_new = 0 if resident else n*min(m, variant_tile_size)
    # G/fitted vectors already belong to their owners. Admit mandatory panels
    # and outputs here; the Score helper admits optional N x traits SIMT stack
    # against the same live cap, or reduces to mature GEMV under pressure.
    traits = len(models)
    temporary_output = (m if resident else min(m, variant_tile_size))*traits
    broker._guard(4*(panel_new + m*traits + temporary_output) + 64*2**20)
    residuals = [model.scaled_residuals for model in models]
    result = torch.empty((m, len(models)), dtype=torch.float32, device=models[0].device)
    if isinstance(genotype, torch.Tensor) and genotype.is_cuda:
        fp32_score_columns(genotype, residuals, out=result)
    else:
        for begin in range(0, m, variant_tile_size):
            end = min(begin+variant_tile_size, m)
            block = torch.as_tensor(genotype[:, begin:end], dtype=torch.float32, device=models[0].device)
            fp32_score_columns(block, residuals, out=result[begin:end])
            del block
    return result


def _gene_normal_score_extreme(score, covariance):
    """Identify sensitive ordinary normal tails using this model's own V."""
    from ..pipeline import _individual_log_probabilities
    log_probability = _individual_log_probabilities(score, covariance.diagonal())
    return bool((log_probability > 100*np.log(10)).any())


def _gene_probability_extreme(result):
    """Check emitted native probabilities without inventing a zero-P floor."""
    if not isinstance(result, dict):
        return False
    for name, value in result.items():
        if not name.startswith(('SKAT(', 'Burden(', 'ACAT-', 'WGS-', 'STAAR-')):
            continue
        if name.endswith(('_log10', '-log10p')):
            if value is not None and np.isfinite(value) and value > 100:
                return True
        elif name.endswith('_log'):
            if value is not None and np.isfinite(value) and value > 100*np.log(10):
                return True
        elif value is not None and np.isfinite(value) and 0 <= value < 1e-100:
            return True
    return False


def _evaluate_gaussian_gene_score(pipeline, payload, model, genotype, broker,
                                  *, shared_simt, report):
    """Retain the original full trait column when FP32 tail sensitivity is high.

    IEEE FP32 matrix and vector reductions can differ slightly. Above native
    -log10(P)=100, or when an aggregate gene tail reaches that range, evaluate
    the original whole resident-G / 512 host-panel GEMV column with this same
    fitted residual and V. A failed first evaluation remains a real failure.
    The threshold selects an arithmetic fallback, never a mask/sample filter.
    """
    original = False
    def restore(reason):
        nonlocal original
        started = time.perf_counter()
        with pipeline.profiler.measure('gene_score_original_fallback', gpu=True):
            payload['score'] = _shared_gene_scores(genotype, [model], broker)[:, 0]
        original = True
        report['gene_score_original_fallback_columns'] = report.get('gene_score_original_fallback_columns', 0)+1
        name = 'gene_score_original_fallback_'+reason
        report[name] = report.get(name, 0)+1
        report['gene_score_original_fallback_submission_seconds'] = report.get('gene_score_original_fallback_submission_seconds', 0.)+time.perf_counter()-started
        report['gene_score_fallback_wall_scope'] = 'CPU submission wall; CUDA time is the nested profiler phase'
    if shared_simt:
        report['gene_score_normal_tail_checks'] = report.get('gene_score_normal_tail_checks', 0)+1
        if payload['score'].is_cuda:
            report['gene_score_control_d2h_batches'] = report.get('gene_score_control_d2h_batches', 0)+1
            report['gene_score_control_d2h_bytes'] = report.get('gene_score_control_d2h_bytes', 0)+1
        if _gene_normal_score_extreme(payload['score'], payload['covariance']):
            restore('normal_tail')
    report['gene_score_tail_evaluations'] = report.get('gene_score_tail_evaluations', 0)+1
    result = pipeline._evaluate_prepared(payload, model)
    if shared_simt and not original and _gene_probability_extreme(result):
        # First-tail dictionaries may contain native CUDA values. Release them
        # before the original-score tail, while keeping this same V resident.
        del result
        restore('gene_tail')
        report['gene_score_additional_tail_evaluations'] = report.get('gene_score_additional_tail_evaluations', 0)+1
        report['gene_score_tail_evaluations'] += 1
        started = time.perf_counter()
        result = pipeline._evaluate_prepared(payload, model)
        report['gene_score_additional_tail_submission_seconds'] = report.get('gene_score_additional_tail_submission_seconds', 0.)+time.perf_counter()-started
    return result


def _gaussian_covariance(model, genotype):
    """Original fitted covariance formula, with the shared score supplied separately."""
    from ..tf32 import matmul
    rotated = model.spectrum.rotate(genotype, matmul_mode='tf32')
    cross = matmul(model.precision_x.T, genotype, mode='tf32')
    covariance = (matmul(rotated.T, model.inverse_variance[:, None]*rotated, mode='tf32')
                  - matmul(matmul(cross.T, model.fixed_effect_covariance, mode='tf32'), cross, mode='tf32'))
    return (covariance+covariance.T)/2


def _blocked_gaussian_covariance(model, host, broker, variant_tile_size=512,
                                *, profiler=None, report=None):
    """Bounded exact fitted kinship-block formula for a host genotype mask.

    Each panel uses the model's original spectral rotation and inverse weights,
    then its own fixed-effect projection. Both directed FP32 TF32 covariance
    products are averaged, retaining the full covariance for ordinary/full
    spectra or FastSKAT. No diagonal approximation replaces relatedness blocks.
    CUDA panel geometry differs from one monolithic GEMM and requires the real
    numerical comparison; this function makes no bitwise equality claim.
    """
    if torch.device(model.device).type == 'cuda':
        from .._blocked_covariance import blocked_gaussian_covariance
        return blocked_gaussian_covariance(model, host, broker,
            variant_tile_size=variant_tile_size, profiler=profiler, report=report)
    from ..tf32 import matmul
    n, m = host.shape
    width = min(m, variant_tile_size)
    scratch = 20*n*width + 16*width**2 + 256*2**20
    broker._guard(4*m*m + scratch)
    covariance = torch.empty((m, m), dtype=torch.float32, device=model.device)
    for begin in range(0, m, width):
        end = min(begin+width, m)
        left = torch.as_tensor(host[:, begin:end], dtype=torch.float32, device=model.device)
        left_rotated = model.spectrum.rotate(left, matmul_mode='tf32')
        left_cross = matmul(model.precision_x.T, left, mode='tf32')
        projected_left = matmul(left_cross.T, model.fixed_effect_covariance, mode='tf32')
        for other in range(begin, m, width):
            stop = min(other+width, m)
            if other == begin:
                right, right_rotated, right_cross = left, left_rotated, left_cross
            else:
                right = torch.as_tensor(host[:, other:stop], dtype=torch.float32, device=model.device)
                right_rotated = model.spectrum.rotate(right, matmul_mode='tf32')
                right_cross = matmul(model.precision_x.T, right, mode='tf32')
            block = (matmul(left_rotated.T, model.inverse_variance[:, None]*right_rotated, mode='tf32')
                     - matmul(projected_left, right_cross, mode='tf32'))
            if other == begin:
                block = (block+block.T)/2
            else:
                reverse = (matmul(right_rotated.T, model.inverse_variance[:, None]*left_rotated, mode='tf32')
                           - matmul(matmul(right_cross.T, model.fixed_effect_covariance, mode='tf32'), left_cross, mode='tf32'))
                block = (block+reverse.T)/2
                del reverse
            covariance[begin:end, other:stop].copy_(block)
            if other != begin:
                covariance[other:stop, begin:end].copy_(block.T)
            del right, right_rotated, right_cross, block
        del left, left_rotated, left_cross, projected_left
    return covariance


def _gene_tail_new_bytes(n, m, options):
    """Whole STAAR stage NEW bytes, excluding its input covariance.

    FastSKAT's own guard starts after STAAR's rare-mask/symmetrization and
    ACAT-V covariance copies. Reuse the existing pipeline spectrum allowance,
    rather than reserve only its later 8*M*M weighted-matrix estimate.
    """
    if getattr(options, "long_mask_method", "full") != "full" and m > options.long_mask_threshold:
        return 16*m*m + 64*m*min(options.long_mask_rank, m) + 4*n
    return 32*m*m + 4*n + 256*2**20


def _gene_host_covariance_new_bytes(n, m, q, options):
    """Reserve the larger bounded original kinship or minimum cached panel."""
    width = min(m, options.variant_tile_size or 512)
    blocked = 4*m*m + 20*n*width + 16*width*width + 256*2**20
    from .._cached_covariance import cached_workspace_estimate
    cached = cached_workspace_estimate(n, m, q, variant_tile_size=512,
                                      panel_variant_size=512)['new_storage_bytes']
    return max(blocked, cached)


class _DirectWorkspace:
    """Compatibility for injected CPU contract repositories without a bank."""
    def __init__(self, broker, required):
        self.broker, self.required_bytes = broker, required

    def set_required(self, required, *, phase='workspace'):
        self.broker._guard(required)
        self.required_bytes = required

    def ensure_free(self, additional_bytes=0, *, phase='workspace'):
        self.broker._guard(max(self.required_bytes, additional_bytes))


@contextmanager
def _gene_workspace_scope(repository, broker, required):
    reserve = getattr(repository, 'reserve_workspace', None)
    if not callable(reserve):
        workspace = _DirectWorkspace(broker, required)
        workspace.ensure_free()
        yield workspace
        return
    with reserve(str(broker.device), required, reserve_bytes=broker._reserve,
                 cache_release=broker.release_device_cache) as workspace:
        with broker.workspace_guard(workspace.ensure_free):
            yield workspace


def _trait_tiles(group, trait_batch_size):
    descriptors = group['descriptors']
    fixed = group.get('trait_tiles')
    if fixed is not None:
        pending = {d['trait_index'] for d in descriptors}
        for tile in fixed:
            ids = {d['trait_index'] for d in tile}
            remaining = pending.intersection(ids)
            if remaining and remaining != ids:
                raise ValueError('a fixed cohort tile cannot be partially resumed')
            if remaining:
                yield tile
        return
    for begin in range(0, len(descriptors), trait_batch_size):
        yield descriptors[begin:begin+trait_batch_size]


def _bind_trait_tiles(groups, configured, trait_batch_size):
    """Retain global tile boundaries when GPU ownership or resume changes."""
    if configured is None:
        return
    if not isinstance(configured, list) or not configured:
        raise ValueError('fixed cohort tiles require a nonempty bound list')
    by_index = {d['trait_index']: (group, d)
                for group in groups for d in group['descriptors']}
    seen, bound = set(), {id(group): [] for group in groups}
    for record in configured:
        keys = ('cohort_index', 'tile_index', 'ordered_sample_rows_sha256',
                'n', 'trait_indices', 'trait_batch_size')
        if not isinstance(record, dict) or any(key not in record for key in keys):
            raise ValueError('fixed cohort tile scientific binding is incomplete')
        scientific = {key: record[key] for key in keys}
        digest = hashlib.sha256(json.dumps(scientific, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        ids = record['trait_indices']
        if (record.get('tile_binding_sha256') != digest
                or record['trait_batch_size'] != trait_batch_size
                or not isinstance(ids, list) or not ids
                or any(type(i) is not int or i not in by_index or i in seen for i in ids)
                or ids != sorted(set(ids)) or len(ids) > trait_batch_size):
            raise ValueError('fixed cohort tile IDs or scientific binding changed')
        group = by_index[ids[0]][0]
        tile = []
        for index in ids:
            owner, descriptor = by_index[index]
            if (owner is not group or descriptor['n'] != record['n']
                    or descriptor.get('sample_rows_sha256') != record['ordered_sample_rows_sha256']):
                raise ValueError('fixed cohort tile differs from its verified ordered sample axis')
            tile.append(descriptor)
        seen.update(ids)
        bound[id(group)].append(tile)
    if seen != set(by_index):
        raise ValueError('fixed cohort tiles do not cover all assigned traits')
    for group in groups:
        group['trait_tiles'] = bound[id(group)]


def _record_checkpoint_rows(report, receipt):
    kind = receipt['kind']
    report['association_rows'][kind] += receipt['rows']
    for trait, rows in receipt['trait_rows'].items():
        report['trait_rows'][int(trait)][kind] += rows
    report['binary_spa_selected'] += receipt.get('binary_spa_selected', 0)
    report['binary_spa_failed'] += receipt.get('binary_spa_failed', 0)
    report['resumed_checkpoint_blocks'] = report.get('resumed_checkpoint_blocks', 0)+1


def _pending_gene_groups(groups, writer, report, chromosome, job_index, kind, trait_batch_size):
    if not getattr(writer, 'checkpoint_tiles', False):
        return groups
    pending = []
    for group in groups:
        descriptors = []
        for tile in _trait_tiles(group, trait_batch_size):
            receipt = writer.checkpoint_tile(kind, chromosome, job_index,
                [d['trait_index'] for d in tile])
            if receipt is None:
                descriptors.extend(tile)
            else:
                _record_checkpoint_rows(report, receipt)
        if descriptors:
            pending.append(dict(group, descriptors=descriptors))
    return pending


def _batched_single(groups, repository, broker, metadata, options, configuration, arguments,
                    writer, report, verified, chromosome, job_index, trait_batch_size, structural):
    first = structural
    mac, subset = arguments.get('mac_cutoff', 20), arguments.get('subset_variants_num', 5000)
    start, end = arguments.get('start'), arguments.get('end')
    variant_type = arguments.get('variant_type', 'variant')
    if (start is None) != (end is None):
        raise ValueError('provide both Single endpoints or neither')
    if start is None:
        selected = np.flatnonzero(first._base_mask(chromosome, variant_type))
    else:
        selected = first.region_indices(start, end)
        if variant_type == 'variant':
            selected = selected[first.qc[selected] == 'PASS']
        else:
            from ..masks import variant_filter
            selected = selected[variant_filter(first.annotations(selected, include_weights=False, metadata='mask'), variant_type)]
    ordinals = [0]*len(groups)
    started = time.perf_counter()
    last_progress = started
    frame_count = 0
    progress = configuration.get('_batched_emit')
    frame_interval = configuration.get('single_progress_frame_interval', 100)
    seconds_interval = configuration.get('single_progress_seconds', 60.)
    checkpoints = bool(getattr(writer, 'checkpoint_tiles', False))
    from ..cache_runtime.single_batches import _physical_requests
    for block_index, request in enumerate(_physical_requests(
            broker, selected, configuration.get('individual_genotype_block_size', 1024))):
        block_identity = hashlib.sha256(np.asarray(request, dtype='<i8').tobytes()).hexdigest()
        cached_groups = []
        all_complete = checkpoints
        for gi, group in enumerate(groups):
            cached = []
            for tile in _trait_tiles(group, trait_batch_size):
                receipt = (writer.checkpoint_tile('individual', chromosome, job_index,
                    [d['trait_index'] for d in tile], block_index=block_index,
                    block_identity=block_identity, ordinal_before=ordinals[gi]) if checkpoints else None)
                cached.append(receipt)
                if receipt is not None:
                    _record_checkpoint_rows(report, receipt)
                else:
                    all_complete = False
            cached_groups.append(cached)
        if all_complete:
            for gi, cached in enumerate(cached_groups):
                after = {r['ordinal_after'] for r in cached}
                if len(after) != 1:
                    raise ValueError('Single checkpoint cohort ordinal advancement differs across trait tiles')
                ordinals[gi] = after.pop()
            report['resumed_single_physical_requests'] = report.get('resumed_single_physical_requests', 0)+1
            continue
        raw = broker.read_states(request, minimum_mac_bound=mac)
        for gi, group in enumerate(groups):
            cached = cached_groups[gi]
            if checkpoints and all(r is not None for r in cached):
                after = {r['ordinal_after'] for r in cached}
                if len(after) != 1:
                    raise ValueError('Single checkpoint cohort ordinal advancement differs across trait tiles')
                ordinals[gi] = after.pop()
                continue
            axis = broker.reader_view(group['rows'])
            rows = axis._axis.samples
            block, prepared = None, None
            ordinal_before = ordinals[gi]
            try:
                # A rejected block still owns the reader's pinned sample axis.
                # Enter its cleanup scope before any allocating preparation.
                block = broker.trait_block(raw, rows, minimum_mac=mac)
                for tile_index, tile in enumerate(_trait_tiles(group, trait_batch_size)):
                    if cached[tile_index] is not None:
                        continue
                    models, model, state, outputs = None, None, None, None
                    pipelines = []
                    try:
                        from ..phewas_resources import host_output_lease_scope
                        with host_output_lease_scope(configuration.get('_single_host_lease_factory'),
                                rows=len(request), traits=len(tile),
                                chromosome_index=configuration.get('_chromosome_index', 0), job_index=job_index,
                                on_wait=configuration.get('_batched_emit')) as output_reservation:
                            with repository.acquire([d['trait_index'] for d in tile], str(broker.device)) as models:
                                pipelines = []
                                for model, descriptor in zip(models, tile):
                                    _check_model(model, descriptor, metadata, verified)
                                    pipelines.append(_thin_pipeline(axis, model, rows, options, configuration, structural))
                                if len(models) != len(tile):
                                    raise ValueError('repository acquire must retain the requested model order/count')
                                if prepared is None:
                                    state, ordinals[gi] = pipelines[0]._prepare_individual_block(block, ordinal_before, mac_cutoff=mac)
                                    prepared = (None if state is None else
                                        pipelines[0]._prepare_individual_trait(state, 0, mac_cutoff=mac))
                                    state = None
                                    if prepared is not None:
                                        prepared.pop('model')
                                    for receipt in cached:
                                        if receipt is not None and receipt['ordinal_after'] != ordinals[gi]:
                                            raise ValueError('Single checkpoint ordinal differs from the current verified genetic block')
                                if prepared is None:
                                    outputs = [[] for _ in tile]
                                else:
                                    g = prepared['genotype']
                                    m, t = g.shape[1], len(models)
                                    # One G is already materialized; three NEW
                                    # genotype/projection buffers remain in the
                                    # mature single-model estimate. Variance is
                                    # serial, but packed tails/corrections span
                                    # the trait tile and need separate storage.
                                    core = max(p._workspace_estimate(model, m, individual=True)
                                               for p, model in zip(pipelines, models))
                                    resident_g = g.numel()*g.element_size()
                                    binary_traits = sum(model.family == 'binomial' for model in models)
                                    tile_tail = 128*m*t + 264*m*binary_traits
                                    broker._guard(max(0, core-resident_g) + tile_tail)
                                    del g
                                    # Optional IEEE Score N x traits stack is
                                    # admitted by its own native live-budget
                                    # gate, with mature GEMV pressure fallback.
                                    def correction(index, model, genotype, score, variance):
                                        from ..binary import binary_single_phewas
                                        return binary_single_phewas(model, genotype,
                                            spa_acquire=lambda: _binary_single_spa_acquire(
                                                repository, broker, tile[index]['trait_index'],
                                                n=model.n, m=genotype.shape[1], q=model.x.shape[1]),
                                            normal_score=score, normal_variance=variance,
                                            p_filter_cutoff=options.p_filter_cutoff,
                                            tol=options.spa_tol, max_iter=options.spa_max_iter)
                                    outputs, _ = process_single_shared_genotype(pipelines, None, ordinals[gi], chromosome,
                                        mac, subset, prepared_state=(prepared, ordinals[gi]), binary_correction=correction)
                                output_reservation.materialized()
                                if checkpoints:
                                    writer.write_tile('individual', chromosome,
                                        [(d['trait_index'], result) for d, result in zip(tile, outputs)],
                                        job_index=job_index, block_index=block_index,
                                        block_identity=block_identity, ordinal_before=ordinal_before,
                                        ordinal_after=ordinals[gi])
                                else:
                                    for descriptor, result in zip(tile, outputs):
                                        writer.write('individual', descriptor['trait_index'], chromosome, result, job_index=job_index)
                                for descriptor, result in zip(tile, outputs):
                                    report['association_rows']['individual'] += len(result)
                                    report['trait_rows'][descriptor['trait_index']]['individual'] += len(result)
                                    if descriptor['family'] == 'binomial':
                                        report['binary_spa_selected'] += sum(row['spa_selected'] for row in result)
                                        report['binary_spa_failed'] += sum(row['spa_failed'] for row in result)
                    finally:
                        for pipeline in pipelines:
                            pipeline.models = []
                        pipelines.clear()
                        models, model, state, outputs = None, None, None, None
                        descriptor, result, pipeline = None, None, None
            finally:
                del prepared, block
                axis.close()
        del raw
        frame_count += 1
        now = time.perf_counter()
        if progress and (frame_count % frame_interval == 0 or now-last_progress >= seconds_interval):
            progress(dict(event='single_progress', job_index=job_index, physical_requests=frame_count,
                elapsed_seconds=now-started, completed_rows=report['association_rows']['individual'],
                exact_ordered_cohorts=len(groups), reader_cumulative=broker.metrics,
                reader_metrics_scope='reader lifetime; not this job delta'))
            last_progress = now


@contextmanager
def _job_host_memory_scope(configuration, factory, *, kind, chromosome_index, job_index):
    """Lease host workspace for one accepted gene cohort and release on all exits.

    A nonblocking growth rejection propagates to the dispatcher's job retry.
    Gene preparation/statistics precede every writer call, so a rejected
    preparation never publishes partial cohort output. Retrying writer failures
    requires the writer's own transactional contract and is not done here.
    """
    if kind == 'individual' or factory is None:
        yield configuration
        return
    with factory(kind=kind, chromosome_index=chromosome_index, job_index=job_index) as lease:
        guard = lease if callable(lease) else getattr(lease, 'guard', None)
        if not callable(guard):
            raise TypeError('host memory lease must provide a callable guard')
        current = dict(configuration)
        current['host_memory_guard'] = guard
        set_future = getattr(lease, 'set_future', None)
        if callable(set_future):
            def replace_future(required_bytes, *, phase):
                # A rejected real lease is already removed or retired by its
                # gate. Idempotent cleanup must not mask the original pressure
                # exception; every new nonzero request remains strictly checked.
                if required_bytes == 0 and getattr(lease, 'closed', False):
                    return 0.
                return set_future(required_bytes, phase=phase)
            current['_host_memory_set_future'] = replace_future
        try:
            yield current
        finally:
            if callable(set_future):
                replace_future(0, phase='job_released')


class _StagedGeneWriter:
    """Keep one cohort private until its preparation and computation succeed."""
    def __init__(self, writer=None):
        self.pending = []
        self.checkpoint_tiles = bool(getattr(writer, 'checkpoint_tiles', False))

    def write(self, *args, **kwargs):
        self.pending.append(('write', args, kwargs))

    def write_tile(self, *args, **kwargs):
        self.pending.append(('write_tile', args, kwargs))

    def publish(self, writer):
        for method, args, kwargs in self.pending:
            getattr(writer, method)(*args, **kwargs)
        self.pending.clear()


def _scientific_counter_snapshot(report, trait_indices=None):
    snapshot = {key: deepcopy(report[key]) for key in
                ('association_rows', 'binary_spa_selected', 'binary_spa_failed', 'job_groups') if key in report}
    if 'trait_rows' in report:
        selected = report['trait_rows'] if trait_indices is None else trait_indices
        snapshot['trait_rows'] = {key: deepcopy(report['trait_rows'][key]) for key in selected}
    return snapshot


def _execute_batched_job(attempt, *, configuration, host_memory_lease_factory,
                         kind, chromosome_index, job_index, writer, report, emit=None, trait_indices=None,
                         host_cache_recovery=None):
    """Retry temporary gene admission only after every owning context unwinds.

    Results and accepted-row counters are rolled back between failed attempts.
    IO, TF32, spectral and repository counters retain all actual attempted work.
    Publishing occurs outside the retry handler: writer errors cannot duplicate
    a partially published job. Single streams directly and has no host-G lease.
    """
    from ..phewas_resources import HostMemoryDeferred
    if kind == 'individual':
        attempt(configuration, writer)
        return
    if host_cache_recovery is not None and not callable(host_cache_recovery):
        raise TypeError('host_cache_recovery must be a context factory or None')
    baseline = _scientific_counter_snapshot(report, trait_indices)
    count = 0
    with ExitStack() as recovery:
        while True:
            staged = _StagedGeneWriter(writer)
            deferred = None
            try:
                with _job_host_memory_scope(configuration, host_memory_lease_factory, kind=kind,
                        chromosome_index=chromosome_index, job_index=job_index) as current:
                    attempt(current, staged)
            except HostMemoryDeferred as error:
                # Keep only anonymous integers. The exception and its traceback
                # would otherwise retain the partial genotype/model stack.
                deferred = dict(required_bytes=error.required_bytes, available_bytes=error.available_bytes)
                # Expected pressure is reported through integers, not retained
                # exception frames containing partially prepared CPU/GPU arrays.
                error.__traceback__ = error.__context__ = error.__cause__ = None
                for key, value in baseline.items():
                    if key == 'trait_rows' and trait_indices is not None:
                        for trait, counters in value.items():
                            report['trait_rows'][trait] = deepcopy(counters)
                    else:
                        report[key] = deepcopy(value)
                staged.pending.clear()
            if deferred is None:
                staged.publish(writer)
                return
            # This point is outside the except block and the lease context.
            # Release partial arrays first, then optional cache owners. Keep
            # metadata LRU admission suspended across this cohort's retries so
            # repeated annotation reads cannot recreate the original pressure.
            del staged
            gc.collect()
            if count == 0 and host_cache_recovery is not None:
                recovery.enter_context(host_cache_recovery())
            count += 1
            report['host_memory_deferrals'] = report.get('host_memory_deferrals', 0)+1
            delay = min(configuration.get('host_memory_retry_seconds', 5.)*2**min(count-1, 8),
                        configuration.get('host_memory_retry_max_seconds', 60.))
            if emit:
                emit(dict(event='host_memory_deferred', attempt=count, retry_after_seconds=delay,
                          **deferred))
            wait_started = time.perf_counter()
            time.sleep(delay)
            report['host_memory_retry_wait_seconds'] = (report.get('host_memory_retry_wait_seconds', 0.)
                                                       +time.perf_counter()-wait_started)


def _phewas_result_fields(value):
    """Keep private PheWAS CSV labels stable at the public-result boundary.

    Mature public kernels return WGS labels. This changes only dictionary keys;
    probabilities, their precision and all diagnostic/metadata values are kept.
    """
    if value is None:
        return None
    result = {}
    for key, field in value.items():
        target = 'STAAR-' + key[4:] if key.startswith('WGS-') else key
        if target in result:
            raise ValueError('conflicting public and private association labels')
        result[target] = field
    return result


def _gene_preparation_geometry(repository, descriptor, broker, metadata, verified):
    """Copy only verified dimensions; preparation never uses fitted values.

    One cohort's masks share its ordered sample axis. Acquiring the same
    anchor for every mask repeats model IO/admission after the tile bank has
    evicted it. Keep only n/device/q for this evaluation, releasing the real
    model before genotype preparation and mandatory workspace admission.
    Every numerical trait still acquires and validates its own fitted model.
    """
    models = model = None
    try:
        with repository.acquire([descriptor['trait_index']], str(broker.device)) as models:
            if len(models) != 1:
                raise ValueError('repository acquire must retain requested model count')
            model = models[0]
            _check_model(model, descriptor, metadata, verified)
            return SimpleNamespace(n=model.n, device=model.device, use_spa=False), int(model.x.shape[1])
    except BaseException as error:
        _clear_inactive_exception_frames(error)
        raise
    finally:
        models = model = None


def _batched_gene(groups, repository, broker, metadata, options, configuration, arguments,
                  writer, report, verified, chromosome, job_index, trait_batch_size, structural, kind):
    descriptors = [descriptor for group in groups for descriptor in group['descriptors']]
    positions = {descriptor['trait_index']: i for i, descriptor in enumerate(descriptors)}
    binary_counts = {d['trait_index']: dict(selected=0, failed=0) for d in descriptors}
    selector = _thin_pipeline(structural.genotype, structural.models[0], structural.union_rows,
                              options, configuration, structural)
    selector.models = [SimpleNamespace(use_spa=d['family'] == 'binomial') for d in descriptors]
    def evaluate(index_sets):
        masks = list(index_sets)
        result = [[None]*len(descriptors) for _ in masks]
        for group in groups:
            axis = broker.reader_view(group['rows'])
            try:
                if not masks:
                    continue
                preparation_model, q = _gene_preparation_geometry(repository,
                    group['descriptors'][0], broker, metadata, verified)
                for mi, indices in enumerate(masks):
                    prepared, prototype, host, genotype, burdens = None, None, None, None, None
                    covariance = payload = scores = diagnostic = p = value = model = models = lobpcg_future = None
                    try:
                        prototype = _thin_pipeline(axis, preparation_model, axis._axis.samples,
                                                   options, configuration, structural)
                        # Preparation retains geometry only. A mixed cohort
                        # still needs each Gaussian variant covariance later.
                        prototype._binary_burden_only = all(
                            d['family'] == 'binomial' for d in group['descriptors'])
                        with prototype.profiler.measure('gene_read_decode_cohort_prepare', gpu=True):
                            with _gene_workspace_scope(repository, broker, 0):
                                prepared = _prepare_shared_gene(prototype, indices, broker)
                        prototype.models = []
                        set_future = configuration.get('_host_memory_set_future')
                        if callable(set_future):
                            set_future(0, phase='gene_host_ready')
                        if prepared is None:
                            continue
                        host = prepared.get('_genotype_device', prepared.get('_genotype_host'))
                        genotype, burdens = host, None
                        n, m = host.shape
                        genotype_elements = n*m
                        memory_sample = configuration.get('_memory_sample')
                        if callable(memory_sample):
                            memory_sample('gene_prepared', genotype=host)
                        has_gaussian = any(d['family'] == 'gaussian' for d in group['descriptors'])
                        resident_input = isinstance(host, torch.Tensor) and host.is_cuda
                        score_traits = min(trait_batch_size, len(group['descriptors']))
                        host_panel = min(m, 512)
                        # Keep the host route's panel allowance independently of
                        # where G currently lives: a rejected resident route must
                        # still admit its bounded host score/variance work.
                        host_score_new = 4*(n*host_panel+(m+host_panel)*score_traits)+64*2**20
                        device_score_new = 8*m*score_traits+64*2**20
                        tail_new = _gene_tail_new_bytes(n, m, options) if has_gaussian else 0
                        host_peak = (max(_gene_host_covariance_new_bytes(n, m, q, options),
                                         4*m*m+tail_new, host_score_new) if has_gaussian else
                            _binary_gene_workspace_new_bytes(n, m, q, len(prepared['names']),
                                                             resident_genotype=False))
                        device_peak = (max(8*genotype_elements+32*m*m+256*2**20,
                                           4*m*m+tail_new, device_score_new) if has_gaussian else
                            _binary_gene_workspace_new_bytes(n, m, q, len(prepared['names'])))
                        # Existing device G is already counted by live admission.
                        # Start with zero NEW bytes, then attempt the optional
                        # resident workspace inside the host-fallback handler.
                        # Host inputs retain their mandatory bounded workspace.
                        initial_peak = 0 if resident_input else host_peak
                        with _gene_workspace_scope(repository, broker, initial_peak) as workspace:
                            # Materialize only if the live device can admit original
                            # fitted ordinary covariance products plus this one G.
                            if resident_input or (m <= options.long_mask_threshold and torch.device(broker.device).type == 'cuda'):
                                # Relatedness may vary between traits in one cohort;
                                # two genotype copies cover the original block path.
                                try:
                                    workspace.set_required(device_peak + (0 if isinstance(host, torch.Tensor) else 4*genotype_elements),
                                                           phase='gene_genotype_upload')
                                except MemoryError:
                                    # A resident preparation must release its CUDA
                                    # owner before genuinely selecting the host path.
                                    # Admit NEW host storage before that transfer.
                                    if isinstance(host, torch.Tensor):
                                        _prepare_host_guard(prototype, SimpleNamespace(n=n), m)
                                        host = host.detach().cpu().numpy()
                                        prepared.pop('_genotype_device')
                                        prepared['_genotype_host'] = host
                                        genotype = host
                                        if callable(set_future):
                                            set_future(0, phase='gene_host_fallback_materialized')
                                    workspace.set_required(host_peak, phase='gene_host_fallback')
                                else:
                                    genotype = torch.as_tensor(host, device=broker.device)
                                    workspace.set_required(device_peak, phase='gene_genotype_resident')
                            tile_peak = (device_peak if isinstance(genotype, torch.Tensor) and genotype.is_cuda
                                         else host_peak)
                            for begin in range(0, len(group['descriptors']), trait_batch_size):
                                workspace.set_required(tile_peak, phase='gene_trait_tile')
                                tile = group['descriptors'][begin:begin+trait_batch_size]
                                with repository.acquire([d['trait_index'] for d in tile], str(broker.device)) as models:
                                    if len(models) != len(tile):
                                        raise ValueError('repository acquire must retain requested model count')
                                    for model, descriptor in zip(models, tile):
                                        _check_model(model, descriptor, metadata, verified)
                                    gaussian_indices = [i for i, model in enumerate(models) if model.family == 'gaussian']
                                    from .scores import execution_metadata as score_metadata
                                    before_simt = score_metadata()['fp32_simt_products']
                                    with prototype.profiler.measure('gene_shared_score', gpu=True):
                                        scores = (_shared_gene_scores(genotype, [models[i] for i in gaussian_indices], broker)
                                                  if gaussian_indices else None)
                                    shared_simt = score_metadata()['fp32_simt_products'] > before_simt
                                    gauss_columns = {i: j for j, i in enumerate(gaussian_indices)}
                                    for ti, (descriptor, model) in enumerate(zip(tile, models)):
                                        p = _thin_pipeline(axis, model, axis._axis.samples, options, configuration, structural)
                                        if model.family == 'binomial':
                                            from ..binary import prepare_binary_burdens, staar_binary_phewas
                                            if burdens is None:
                                                # Model-tile admission precedes binding/checking its
                                                # fitted states. Recheck the remaining owner budget at
                                                # the actual burden allocation boundary; G is already
                                                # resident and never credited as free storage.
                                                workspace.ensure_free(phase='gene_binary_burden_prepare')
                                                with p.profiler.measure('gene_binary_shared_burdens', gpu=True):
                                                    burdens = prepare_binary_burdens(model, genotype, prepared['maf'],
                                                        prepared['annotations'], prepared['names'], rare_maf_cutoff=options.rare_maf_cutoff,
                                                        rv_num_cutoff=options.rv_num_cutoff, rv_num_cutoff_max=options.rv_num_cutoff_max)
                                            with p.profiler.measure('gene_binary_nominal_and_spa', gpu=True):
                                                value, diagnostic = staar_binary_phewas(model, genotype, prepared['maf'],
                                                    prepared['annotations'], prepared['names'],
                                                    spa_acquire=lambda: repository.acquire_spa(descriptor['trait_index'], str(broker.device)),
                                                    prepared_burdens=burdens, rare_maf_cutoff=options.rare_maf_cutoff,
                                                    rv_num_cutoff=options.rv_num_cutoff, rv_num_cutoff_max=options.rv_num_cutoff_max,
                                                    p_filter_cutoff=options.p_filter_cutoff, tol=options.spa_tol,
                                                    max_iter=options.spa_max_iter, return_diagnostics=True)
                                            selected = int(diagnostic['spa_selected'].sum())
                                            failed = int(diagnostic['spa'].failed.sum())
                                            report['binary_spa_selected'] += selected
                                            report['binary_spa_failed'] += failed
                                            binary_counts[descriptor['trait_index']]['selected'] += selected
                                            binary_counts[descriptor['trait_index']]['failed'] += failed
                                            if diagnostic['variant_covariance_constructed']:
                                                raise RuntimeError('binary burden path must not construct a variant covariance')
                                            del diagnostic
                                        else:
                                            cov_new = (4*genotype_elements*(2 if model.spectrum.blocks else 1)
                                                       +32*m*m+256*2**20
                                                       if isinstance(genotype, torch.Tensor) and genotype.is_cuda
                                                       else _gene_host_covariance_new_bytes(n, m, model.x.shape[1], options))
                                            workspace.set_required(max(cov_new, 4*m*m+tail_new), phase='gene_covariance')
                                            if isinstance(genotype, torch.Tensor) and genotype.is_cuda:
                                                copies = 2 if model.spectrum.blocks else 1
                                                broker._guard(4*genotype_elements*copies + 32*m*m + 256*2**20)
                                                with p.profiler.measure('gene_covariance', gpu=True):
                                                    covariance = _gaussian_covariance(model, genotype)
                                            else:
                                                # Cached/tiled covariance is fitted-model specific;
                                                # only G IO and phenotype-score GEMM are shared.
                                                with p.profiler.measure('gene_covariance', gpu=True):
                                                    if p._supports_long_tf32_products(model):
                                                        redundant_score, covariance = p._long_mask_products(model, host, compute_score=False)
                                                        del redundant_score
                                                    else:
                                                        covariance = _blocked_gaussian_covariance(model, host, broker,
                                                            variant_tile_size=options.variant_tile_size or 512,
                                                            profiler=p.profiler, report=report)
                                            if callable(memory_sample):
                                                memory_sample('gene_covariance_ready', genotype=genotype, covariance=covariance, score=scores)
                                            # Its input covariance is now allocated.
                                            # Reserve only NEW STAAR/tail bytes here.
                                            workspace.set_required(tail_new, phase='gene_tail')
                                            payload = {key: value for key, value in prepared.items() if not key.startswith('_')}
                                            payload.update(score=scores[:, gauss_columns[ti]], covariance=covariance,
                                                           cmac=prepared['cmac'])
                                            from .._fastskat_lobpcg_tiles import lobpcg_workspace_scope
                                            def lobpcg_future(additional, *, phase):
                                                # V and prepared G are already resident. Preserve
                                                # the whole original STAAR reservation and add only
                                                # lookahead/parallel NEW excess before allocation.
                                                workspace.set_required(tail_new+additional, phase=phase)
                                                return True
                                            with lobpcg_workspace_scope(callback=lobpcg_future):
                                                value = _evaluate_gaussian_gene_score(p, payload, model, genotype, broker,
                                                    shared_simt=shared_simt, report=report)
                                            lobpcg_future = None
                                            del covariance, payload
                                        result[mi][positions[descriptor['trait_index']]] = _phewas_result_fields(value)
                                        del p, model, value
                                    del scores, models
                        del prepared, host, genotype, burdens
                    except BaseException as error:
                        # Completed leaf frames can otherwise retain a model,
                        # G and V even after the cache and leases are closed.
                        _clear_inactive_exception_frames(error)
                        raise
                    finally:
                        if prototype is not None:
                            prototype.models = []
                        prepared, prototype, host, genotype, burdens = None, None, None, None, None
                        # A retained caller exception must not extend the last
                        # model lease or keep a completed covariance alive.
                        covariance = payload = scores = diagnostic = p = value = model = models = lobpcg_future = None
                        set_future = configuration.get('_host_memory_set_future')
                        if callable(set_future):
                            set_future(0, phase='gene_mask_released')
            finally:
                axis.close()
        return result
    selector._run_mask_sets = evaluate
    outputs = getattr(selector, kind)(chromosome=chromosome, **arguments)
    for number, descriptor in enumerate(descriptors):
        if isinstance(outputs, dict):
            result = {category: rows[number] for category, rows in outputs.items()}
        else:
            result = outputs[number]
        if not getattr(writer, 'checkpoint_tiles', False):
            writer.write(kind, descriptor['trait_index'], chromosome, result, job_index=job_index)
        count = sum(len(rows) for rows in result.values()) if isinstance(result, dict) else len(result)
        report['association_rows'][kind] += count
        report['trait_rows'][descriptor['trait_index']][kind] += count
    if getattr(writer, 'checkpoint_tiles', False):
        for group in groups:
            for tile in _trait_tiles(group, trait_batch_size):
                values = []
                for descriptor in tile:
                    number = positions[descriptor['trait_index']]
                    result = ({category: rows[number] for category, rows in outputs.items()}
                              if isinstance(outputs, dict) else outputs[number])
                    values.append((descriptor['trait_index'], result))
                with selector.profiler.measure('gene_checkpoint_write'):
                    writer.write_tile(kind, chromosome, values, job_index=job_index,
                        binary_counts={name: sum(binary_counts[d['trait_index']][name] for d in tile)
                                       for name in ('selected', 'failed')})


def run_batched_configuration(configuration, *, model_repository, reader_factory, writer,
                              device='cuda:0', trait_batch_size=32, memory_limit_gib=None,
                              cpu_threads=1, cache_specs=None, device_cache_bytes=512*2**20,
                              compact_cache_bytes=64*2**20, metadata_cache_bytes=256*2**20,
                              sample_axis_cache_bytes=64*2**20,
                              sample_axis_host_cache_bytes=64*2**20, job_state_cache_bytes=0,
                              job_state_cache_live_fraction=.25, device_cache_live_fraction=None,
                              memory_reserve_bytes=256*2**20, emit=None,
                              host_memory_lease_factory=None):
    """Run a cache-only frame/job-major independent multi-phenotype analysis.

    The repository exposes descriptors/acquire/acquire_spa, and the injected
    reader factory context returns metadata/container/portable_axis. The writer
    streams every result immediately. GPU null states are bounded by repository
    admission and trait tiles; neither all-trait G nor M² covariances are stacked.
    None removes artificial GiB caps while retaining live-memory reservation.
    Binary formal probabilities use on-demand fitted FP64 SPA sidecars. Shared
    IEEE FP32 CUDA Score tiles retain mature vector fallback for sensitive tails;
    covariance/projection preserve the fitted native TF32 route.
    host_memory_lease_factory receives kind/chromosome_index/job_index and yields
    a guard(n,m,phase) or a lease exposing guard. Shared-host dispatchers supply
    a nonblocking lease, retry deferred jobs after this context releases state,
    and retain the host budget independently of the GPU physical-memory budget.
    """
    if type(trait_batch_size) is not int or trait_batch_size < 1 or type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError('trait_batch_size and cpu_threads must be positive integers')
    for name, value in (('memory_reserve_bytes', memory_reserve_bytes),
                        ('device_cache_bytes', device_cache_bytes),
                        ('compact_cache_bytes', compact_cache_bytes),
                        ('metadata_cache_bytes', metadata_cache_bytes),
                        ('sample_axis_cache_bytes', sample_axis_cache_bytes),
                        ('sample_axis_host_cache_bytes', sample_axis_host_cache_bytes),
                        ('job_state_cache_bytes', job_state_cache_bytes)):
        if type(value) is not int or value < 0:
            raise ValueError(name+' must be a nonnegative integer')
    for name, value in (('job_state_cache_live_fraction', job_state_cache_live_fraction),
                        ('device_cache_live_fraction', device_cache_live_fraction)):
        if value is None and name == 'device_cache_live_fraction':
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(name+' must be positive, finite and at most 1')
    if not callable(reader_factory) or not callable(getattr(writer, 'write', None)):
        raise ValueError('portable reader_factory and streaming writer are required')
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('PheWAS context is already running')
    previous_threads = torch.get_num_threads()
    repository_policy = ExitStack()
    lobpcg_owner = report = None
    try:
        if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('native batched PheWAS production requires CUDA')
        torch.set_num_threads(cpu_threads)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
        budget = _resolved_budget(memory_limit_gib, device)
        configuration = dict(configuration)
        from .._fastskat_lobpcg_tiles import lobpcg_pool_scope
        lobpcg_owner = repository_policy.enter_context(lobpcg_pool_scope(
            max_workers=2, owner=configuration.get('_association_lobpcg_owner')))
        configuration['_association_lobpcg_owner'] = lobpcg_owner
        if host_memory_lease_factory is not None and not callable(host_memory_lease_factory):
            raise TypeError('host_memory_lease_factory must be callable or None')
        configuration['_batched_emit'] = emit
        configuration['_single_host_lease_factory'] = host_memory_lease_factory
        profiler = StageProfiler(device, enabled=configuration.get('stage_profile', False))
        configuration['_stage_profiler'] = profiler
        from ..phewas_resources import ResourceTelemetry, tensor_storage_footprint
        telemetry = ResourceTelemetry(configuration.get('memory_telemetry_seconds', 30.))
        last_sample = [None]
        memory_samples = []
        def memory_sample(phase, **owners):
            sample = telemetry.sample(device=device, limit_bytes=(None if memory_limit_gib is None else int(budget*2**30)),
                                      reserve_bytes=memory_reserve_bytes, include_pss=True)
            # Deep process mapping inspection is throttled independently of
            # computational phases. Retain numerical summaries, never tensors.
            token = sample['sampled_at']
            if token == last_sample[0]:
                return
            last_sample[0] = token
            sample.update(phase=phase, explicit_workspace_owners=tensor_storage_footprint(owners),
                          fitted_state_owners=getattr(model_repository, 'resident_storage_summary', lambda: {})())
            sample['fastskat_lobpcg_owner'] = lobpcg_owner.metadata()
            memory_samples.append(sample)
            if len(memory_samples) > 120:
                del memory_samples[0]
            if emit:
                emit(dict(event='resource_sample', memory=sample))
        configuration['_memory_sample'] = memory_sample
        if configuration.get('matmul_mode', 'tf32') != 'tf32' or configuration.get('precision_control', False):
            raise ValueError('batched PheWAS uses native FP32 storage and TF32 products')
        raw_options = dict(configuration.get('analysis_options', {}))
        if raw_options.get('wrapper_semantics', 'base') != 'base':
            raise ValueError('independent batched PheWAS requires base wrapper semantics')
        for name in ('individual_genotype_block_size', 'individual_effective_block_size'):
            value = configuration.get(name, 1024)
            if type(value) is not int or value < 1:
                raise ValueError(name+' must be a positive integer')
        progress_frames = configuration.get('single_progress_frame_interval', 100)
        progress_seconds = configuration.get('single_progress_seconds', 60.)
        if type(progress_frames) is not int or progress_frames < 1:
            raise ValueError('single_progress_frame_interval must be a positive integer')
        if isinstance(progress_seconds, bool) or not isinstance(progress_seconds, (int, float)) or not math.isfinite(progress_seconds) or progress_seconds <= 0:
            raise ValueError('single_progress_seconds must be positive and finite')
        gene_progress = configuration.get('gene_progress_seconds', 60.)
        raw_capacity = configuration.get('job_state_cache_bytes',
            configuration.get('gene_raw_state_cache_bytes', job_state_cache_bytes))
        raw_fraction = configuration.get('job_state_cache_live_fraction',
            configuration.get('gene_raw_state_cache_live_fraction', job_state_cache_live_fraction))
        if isinstance(gene_progress, bool) or not isinstance(gene_progress, (int, float)) or not math.isfinite(gene_progress) or gene_progress <= 0:
            raise ValueError('gene_progress_seconds must be positive and finite')
        if type(raw_capacity) is not int or raw_capacity < 0:
            raise ValueError('gene_raw_state_cache_bytes must be a nonnegative integer')
        if isinstance(raw_fraction, bool) or not isinstance(raw_fraction, (int, float)) or not math.isfinite(raw_fraction) or not 0 < raw_fraction <= 1:
            raise ValueError('gene_raw_state_cache_live_fraction must be positive, finite and at most 1')
        for name, default in (('host_memory_retry_seconds', 5.), ('host_memory_retry_max_seconds', 60.)):
            delay = configuration.get(name, default)
            if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not math.isfinite(delay) or not 0 < delay <= 60:
                raise ValueError(name+' must be positive, finite and at most 60 seconds')
        maximum = configuration.get('maximum_mask_variants')
        if maximum is not None and (type(maximum) is not int or maximum < 1):
            raise ValueError('maximum_mask_variants must be a positive integer or None')
        raw_options.update(wrapper_semantics='base', memory_limit_gib=budget)
        options = AnalysisOptions(**raw_options)
        catalog = configuration.get('annotation_catalog', {})
        if isinstance(catalog, str):
            catalog = json.loads(Path(catalog).read_text())
        configuration['_resolved_annotation_catalog'] = catalog
        descriptors, groups = _descriptor_groups(model_repository)
        _bind_trait_tiles(groups, configuration.get('_association_trait_tiles'), trait_batch_size)
        for descriptor in descriptors:
            if descriptor['family'] == 'binomial' and not callable(getattr(model_repository, 'acquire_spa', None)):
                raise ValueError('formal binary analysis requires a fitted FP64 SPA repository')
        configured = configure_tf32(split_k=0, memory_limit_gib=budget,
                                    memory_reserve_bytes=memory_reserve_bytes)
        policy = getattr(model_repository, 'memory_policy', None)
        if callable(policy):
            repository_policy.enter_context(policy(device, limit_bytes=(None if memory_limit_gib is None else int(budget*2**30)),
                                                  reserve_bytes=memory_reserve_bytes,
                                                  cache_live_fraction=_model_cache_live_fraction(configuration, descriptors)))
        tf32_metadata(reset=True); single_metadata(reset=True); statistics_execution_metadata(reset=True)
        start = time.perf_counter()
        kinds = ('individual', 'coding', 'noncoding', 'ncrna')
        report = dict(schema_version=2, execution='frame/job-major-native-TF32-phenotype-score-GEMM',
            traits=len(descriptors), exact_ordered_cohorts=len(groups), trait_batch_size=trait_batch_size,
            association_rows=dict.fromkeys(kinds, 0),
            trait_rows={d['trait_index']: dict.fromkeys(kinds, 0) for d in descriptors},
            job_groups=0, chromosome_readers=[], binary_spa_selected=0, binary_spa_failed=0,
            covariance_shared=False, fitted_projection_reuse=False, source_gds_required=False,
            host_memory_lease_active=host_memory_lease_factory is not None,
            host_memory_deferrals=0, host_memory_retry_wait_seconds=0.,
            execution_counter_scope='worker lifecycle including retried work; accepted-row counters exclude deferrals')
        from .. import _weighted_spectra
        solver = cli._weighted_eigensolver_settings(configuration, 'tf32', device)
        with _weighted_spectra.eigensolver_context(solver['effective'], memory_limit=int(budget*2**30)) as solver_state:
            with DenseProductAudit(forced=True) as audit:
                for ci, reference in enumerate(configuration.get('chromosomes', [])):
                    source = reference.get('genotype', reference.get('gds', reference.get('cache', reference.get('name'))))
                    spec = None if cache_specs is None else cache_specs.get(source)
                    with reader_factory(reference, spec) as resources:
                        metadata = resources['metadata']
                        if not isinstance(metadata, SharedMetadataReader):
                            metadata = SharedMetadataReader(metadata, capacity_bytes=metadata_cache_bytes)
                        metadata.pin_field('position'); metadata.pin_field(configuration.get('qc_path', 'annotation/filter'))
                        index_settings = reference.get('annotation_index') or {}
                        interval_file = index_settings.get('promoter_intervals_file')
                        chromosome_promoters = (cli._read_promoter_intervals(interval_file)
                                                if interval_file is not None else None)
                        with SharedStateBroker(metadata, resources['container'], device=device, memory_limit_gib=budget,
                                device_cache_bytes=device_cache_bytes, compact_cache_bytes=compact_cache_bytes,
                                device_cache_live_fraction=device_cache_live_fraction,
                                sample_axis_cache_bytes=sample_axis_cache_bytes,
                                sample_axis_host_cache_bytes=sample_axis_host_cache_bytes,
                                own_reader=False, own_container=False, portable_axis=resources.get('portable_axis', False),
                                memory_reserve_bytes=memory_reserve_bytes) as broker:
                            verified = set()
                            first_descriptor = descriptors[0]
                            axis = broker.reader_view(first_descriptor['ordered_rows'])
                            with model_repository.acquire([first_descriptor['trait_index']], str(broker.device)) as models:
                                _check_model(models[0], first_descriptor, metadata, verified)
                                structural = _thin_pipeline(axis, models[0], axis._axis.samples, options, configuration)
                            # A lightweight descriptor is enough for mask selection;
                            # don't pin the first GPU null during all later jobs.
                            structural.models = [SimpleNamespace(device=broker.device, use_spa=False)]
                            del models
                            for ji, job in enumerate(reference.get('jobs', [])):
                                configuration['_chromosome_index'] = ci
                                kind = job.get('kind')
                                if kind not in kinds:
                                    raise ValueError('unknown association job kind')
                                arguments = dict(job.get('arguments', {}))
                                arguments.pop('chromosome', None)
                                filename = arguments.pop('promoter_intervals_file', None)
                                if filename:
                                    arguments['promoter_intervals'] = cli._read_promoter_intervals(filename)
                                elif kind == 'noncoding' and chromosome_promoters is not None:
                                    arguments.setdefault('promoter_intervals', chromosome_promoters)
                                before = time.perf_counter()
                                memory_sample('job_started')
                                if emit:
                                    emit(dict(event='started', chromosome_index=ci, job_index=ji, kind=kind))
                                function = _batched_single if kind == 'individual' else _batched_gene
                                checkpoint_job_index = job.get('checkpoint_job_index', ji)
                                pending_groups = (groups if kind == 'individual' else
                                    _pending_gene_groups(groups, writer, report, reference['name'],
                                                         checkpoint_job_index, kind, trait_batch_size))
                                with ExitStack() as job_resources:
                                    raw_cache = getattr(broker, 'raw_state_cache', None)
                                    if kind != 'individual' and callable(raw_cache):
                                        job_resources.enter_context(raw_cache(
                                            raw_capacity, live_fraction=raw_fraction))
                                    last_cohort_progress = before
                                    cohort_groups = ([pending_groups] if kind == 'individual' else
                                                     [[group] for group in pending_groups])
                                    for cohort_index, current_groups in enumerate(cohort_groups):
                                        if not current_groups:
                                            continue
                                        def attempt(job_configuration, current_writer):
                                            args = (current_groups, model_repository, broker, metadata, options, job_configuration,
                                                arguments, current_writer, report, verified, reference['name'],
                                                checkpoint_job_index, trait_batch_size, structural)
                                            if kind == 'individual':
                                                function(*args)
                                            else:
                                                function(*args, kind)
                                        _execute_batched_job(attempt, configuration=configuration,
                                            host_memory_lease_factory=host_memory_lease_factory, kind=kind,
                                            chromosome_index=ci, job_index=ji, writer=writer, report=report, emit=emit,
                                            trait_indices=[d['trait_index'] for group in current_groups
                                                           for d in group['descriptors']],
                                            host_cache_recovery=metadata.pressure_recovery)
                                        now = time.perf_counter()
                                        if (emit and kind != 'individual' and
                                                now-last_cohort_progress >= gene_progress):
                                            emit(dict(event='cohort_finished', chromosome_index=ci, job_index=ji,
                                                      kind=kind, cohort_index=cohort_index,
                                                      pending_cohorts=len(cohort_groups),
                                                      completed_rows=report['association_rows'][kind],
                                                      gpu_memory=_gpu_memory_diagnostics(device, model_repository, broker)))
                                            last_cohort_progress = now
                                report['job_groups'] += 1
                                profiler.flush()
                                memory_sample('job_released')
                                if emit:
                                    emit(dict(event='finished', chromosome_index=ci, job_index=ji, kind=kind,
                                              wall_seconds=time.perf_counter()-before,
                                              gpu_memory=_gpu_memory_diagnostics(device, model_repository, broker)))
                            report['chromosome_readers'].append(dict(broker=broker.metrics,
                                metadata=dict(metadata.metrics),
                                reader_metadata=_reader_metadata_diagnostics(metadata)))
                            axis.close()
                report['dense_product_audit'] = audit.report()
        if report['job_groups'] == 0:
            raise ValueError('batched PheWAS requires a nonempty chromosome/job schedule')
        report.update(total_seconds=time.perf_counter()-start, memory_limit_gib=budget,
            memory_limit_source='physical device total' if memory_limit_gib is None else 'explicit caller budget',
            live_memory_reserve_bytes=memory_reserve_bytes, single_execution=single_metadata(),
            statistics_execution_metadata=statistics_execution_metadata(), tf32_execution=tf32_metadata(),
            tf32_configuration=configured, weighted_eigensolver_execution=dict(solver, **solver_state),
            peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20,
            peak_gpu_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20,
            gpu_memory=_gpu_memory_diagnostics(device, model_repository),
            cpu_thread_control=dict(requested=cpu_threads, effective=torch.get_num_threads(), previous=previous_threads,
                                    restored=True, scope='pytorch_intraop'),
            output=writer.finish(), stage_profile=profiler.report(), resource_samples=memory_samples,
            resource_samples_scope='bounded 120 throttled owned-process endpoints; not additive with allocator or cgroup usage')
        if callable(getattr(model_repository, 'summary', None)):
            report['model_repository'] = model_repository.summary()
        return report
    except BaseException as error:
        abort = getattr(writer, 'abort', None)
        if callable(abort):
            try:
                abort()
            except BaseException as cleanup_error:
                if hasattr(error, 'add_note'):
                    error.add_note('CSV abort also failed: '+type(cleanup_error).__name__)
        raise
    finally:
        try:
            from .._lobpcg_stream_pool import close_preserving_error
            try:
                close_preserving_error(repository_policy.close, sys.exc_info()[1])
            finally:
                if report is not None and lobpcg_owner is not None:
                    report['fastskat_lobpcg_owner'] = lobpcg_owner.metadata()
        finally:
            try:
                torch.set_num_threads(previous_threads)
            finally:
                _LOCK.release()


def run_configuration(analyses, *, cache_specs, device='cuda:0', device_cache_bytes=512*2**20,
                      compact_cache_bytes=64*2**20, metadata_cache_bytes=256*2**20,
                      cpu_threads=2):
    """Run standalone-compatible configs using existing verified genotype caches.

    ``analyses`` is a sequence of individual association configurations with
    identical ordered scientific jobs and separate original output paths.
    ``cache_specs`` maps each prepared metadata directory to the existing CacheSpec with a
    current source proof. Missingness and fitted model axes remain independent.
    ``cpu_threads`` is a positive integer controlling PyTorch CPU intra-op
    threads during this call, default 2. The previous process setting is restored
    on success or failure. This does not configure cache worker processes,
    inter-op threads, environment variables, or GPU scheduling.
    """
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError('cpu_threads must be a positive integer')
    if not _LOCK.acquire(blocking=False):raise RuntimeError('PheWAS context is already running')
    previous_threads = None
    try:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(cpu_threads)
        effective_threads = torch.get_num_threads()
        if effective_threads != cpu_threads:
            raise RuntimeError('requested PyTorch CPU intra-op thread count was not applied')
        report = _run(analyses,cache_specs=cache_specs,device=device,device_cache_bytes=device_cache_bytes,
                      compact_cache_bytes=compact_cache_bytes,metadata_cache_bytes=metadata_cache_bytes)
        report['cpu_thread_control'] = dict(requested=cpu_threads, effective=effective_threads,
            previous=previous_threads, restored=True, scope='pytorch_intraop')
    finally:
        try:
            if previous_threads is not None:
                torch.set_num_threads(previous_threads)
        finally:
            _LOCK.release()
    return report


def _run(analyses, *, cache_specs, device, device_cache_bytes, compact_cache_bytes, metadata_cache_bytes):
    configs=validate_analyses(analyses)
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('native PheWAS production requires CUDA')
    started=time.perf_counter();torch.cuda.init();torch.cuda.reset_peak_memory_stats(device)
    budget=_resolved_budget(configs[0]['analysis_options'].get('memory_limit_gib'), device)
    for config in configs:
        config['analysis_options']['memory_limit_gib'] = budget
    tf32_config=configure_tf32(split_k=0,memory_limit_gib=budget)
    tf32_metadata(reset=True);single_metadata(reset=True);statistics_execution_metadata(reset=True)
    before=time.perf_counter();models,rows=_load_models(configs,device)
    model_seconds=time.perf_counter()-before
    outputs=[_CSVOutputs(c) for c in configs]
    report={'schema_version':1,'matmul_mode':'tf32','models':[{'index':i,'n':m.n,'covariates':m.x.shape[1],
        'family':m.family,'use_spa':m.use_spa} for i,m in enumerate(models)],'jobs':[], 'job_groups':[], 'shared_readers':[]}
    specs={str(Path(p).resolve()):s for p,s in cache_specs.items()}
    interval_cache={};saved=False;setup_seconds=0.;annotation_seconds=0.
    from .. import _weighted_spectra
    solver=cli._weighted_eigensolver_settings(configs[0],'tf32',device)
    with _weighted_spectra.eigensolver_context(solver['effective'],memory_limit=int(budget*2**30)) as solver_state:
        with DenseProductAudit(forced=True) as audit:
            for ci,reference in enumerate(configs[0]['chromosomes']):
                source=str(Path(reference['genotype']).resolve())
                if source not in specs:raise ValueError('every source genotype requires an explicit cache spec')
                spec=specs[source];expected=json.loads(json.dumps(spec.expected_binding,sort_keys=True,allow_nan=False))
                if not callable(spec.source_proof) or spec.source_proof()!=expected:
                    raise ValueError('current input/source proof differs from cache binding')
                begin=time.perf_counter()
                with ExitStack() as stack:
                    prepared=stack.enter_context(PortableMetadataReader(reference['genotype'],
                        container_directory=spec.directory))
                    metadata=SharedMetadataReader(prepared,capacity_bytes=metadata_cache_bytes)
                    container=stack.enter_context(Container(spec.directory,expected,spec.expected_samples))
                    broker=stack.enter_context(SharedStateBroker(metadata,container,device=device,memory_limit_gib=budget,
                        device_cache_bytes=device_cache_bytes,compact_cache_bytes=compact_cache_bytes,own_reader=False,own_container=False))
                    def verify_after():
                        if spec.source_proof() != expected:raise ValueError('input/source binding changed during PheWAS')
                    stack.callback(verify_after)
                    metadata.pin_field('position')
                    metadata.pin_field(configs[0].get('qc_path','annotation/filter'))
                    pipelines=[]
                    catalog=configs[0].get('annotation_catalog',{})
                    if isinstance(catalog,str):catalog=json.loads(Path(catalog).read_text())
                    for i,(model,row,config) in enumerate(zip(models,rows,configs)):
                        physical=cli._bind_genotype_samples(metadata,model,row,config['phenotypes'][0].get('sample_id_rule','exact'))
                        view=broker.reader_view(physical)
                        p=LimitedMaskPipeline(view,[model],qc_path=config.get('qc_path','annotation/filter'),
                            annotation_catalog=catalog,annotation_names=config.get('annotation_names',[]),
                            genotype_sample_indices=[physical],options=AnalysisOptions(**config['analysis_options']))
                        p.resident_genotypes=True;p.single_batch_optimization=True
                        p.individual_effective_block_size=config['individual_effective_block_size']
                        p.statistics_execution='serial';p.statistics_tail_optimization=config.get('statistics_tail_optimization',True)
                        p.local_mask_reuse=config.get('local_mask_reuse',True);p.weight_batch_optimization=config.get('weight_batch_optimization',True)
                        p.maximum_mask_variants=config.get('maximum_mask_variants')
                        p.profiler=StageProfiler(device,enabled=config.get('stage_profile',False))
                        pipelines.append(p)
                    _share_annotations(pipelines)
                    setup_seconds+=time.perf_counter()-begin
                    categories=cli._scheduled_index_categories(reference['jobs'])
                    if categories:
                        index=reference.get('annotation_index',{})
                        needs_promoters=any(x.startswith('promoter_') for x in categories)
                        filename=index.get('promoter_intervals_file') if needs_promoters else None
                        if needs_promoters and filename is None:
                            raise ValueError('promoter jobs require exact intervals')
                        if filename and filename not in interval_cache:interval_cache[filename]=cli._read_promoter_intervals(filename)
                        begin=time.perf_counter();pipelines[0].prepare_annotation_index(reference['name'],
                            promoter_intervals=interval_cache.get(filename),include_ncrna=False,categories=categories)
                        _share_annotations(pipelines);annotation_seconds+=time.perf_counter()-begin
                    if not saved:
                        for model,config in zip(models,configs):
                            p=config['phenotypes'][0]
                            if 'save_model' in p:
                                Path(p['save_model']).parent.mkdir(parents=True,exist_ok=True);save_null_model(model,p['save_model'])
                        saved=True
                    for ji,job in enumerate(reference['jobs']):
                        kind=job['kind'];arguments=dict(job.get('arguments',{}));arguments['chromosome']=reference['name']
                        filename=arguments.pop('promoter_intervals_file',None)
                        if filename:
                            if filename not in interval_cache:interval_cache[filename]=cli._read_promoter_intervals(filename)
                            arguments['promoter_intervals']=interval_cache[filename]
                        print(json.dumps({'event':'started','chromosome_index':ci,'job_index':ji,'kind':kind}),flush=True)
                        begin=time.perf_counter()
                        if kind=='individual':
                            results=_single(pipelines,broker,arguments,configs)
                            computed=time.perf_counter()-begin
                            for ti,result in enumerate(results):
                                target=configs[ti]['chromosomes'][ci]['jobs'][ji]
                                outputs[ti].append(target,result)
                                report['jobs'].append({'trait_index':ti,'chromosome_index':ci,'job_index':ji,'kind':kind,
                                    'eligible_association_tests':cli._association_result_row_count(result,kind=kind)})
                            del results
                        else:
                            for ti,p in enumerate(pipelines):
                                result=getattr(p,kind)(**arguments)
                                outputs[ti].append(configs[ti]['chromosomes'][ci]['jobs'][ji],result)
                                report['jobs'].append({'trait_index':ti,'chromosome_index':ci,'job_index':ji,'kind':kind,
                                    'eligible_association_tests':cli._association_result_row_count(result,kind=kind)})
                                del result
                            computed=time.perf_counter()-begin
                        torch.cuda.synchronize(device)
                        report['job_groups'].append({'chromosome_index':ci,'job_index':ji,'kind':kind,
                            'compute_and_output_seconds':time.perf_counter()-begin})
                        print(json.dumps({'event':'finished','chromosome_index':ci,'job_index':ji,'kind':kind,'seconds':computed}),flush=True)
                    for p in pipelines:p.profiler.flush()
                    report['shared_readers'].append({'broker':broker.metrics,'metadata':dict(metadata.metrics),
                        'stage_profiles':[p.profiler.report() for p in pipelines],
                        'skipped_masks':[list(p.skipped_sets) for p in pipelines],
                        'local_mask_reuse':[dict(p.local_mask_reuse_counters) for p in pipelines],
                        'covariance_diagnostics':[list(getattr(p, 'covariance_diagnostics', [])) for p in pipelines]})
                del pipelines, p, metadata, prepared, container, broker
        report['dense_product_audit']=audit.report()
    for output in outputs:output.check()
    report.update(total_seconds=time.perf_counter()-started,model_seconds=model_seconds,setup_seconds=setup_seconds,
        annotation_seconds=annotation_seconds,csv_output_seconds=sum(o.seconds for o in outputs),
        csv_files=sum(o.files for o in outputs), csv_outputs=[o.proofs for o in outputs],single_execution=single_metadata(),tf32_execution=tf32_metadata(),
        statistics_execution_metadata=statistics_execution_metadata(),
        peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20,peak_gpu_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20,
        memory_limit_gib=budget,tf32_configuration=tf32_config,weighted_eigensolver_execution=dict(solver,**solver_state))
    report['eligible_association_tests']=sum(j['eligible_association_tests'] for j in report['jobs'])
    report['native_execution_status']=cli._native_execution_status(report['tf32_execution'],report['jobs'],planned_jobs=sum(len(c['jobs']) for config in configs for c in config['chromosomes']))
    if report['peak_gpu_mib']>budget*1024:raise MemoryError('observed PheWAS CUDA allocation exceeded configured budget')
    return report
