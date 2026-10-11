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
import inspect
import math
import threading
import time
import hashlib
import gc
from types import SimpleNamespace
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


def _share_annotations(pipelines):
    first = pipelines[0]
    for pipeline in pipelines[1:]:
        pipeline._base_masks = first._base_masks
        pipeline._category_codes = first._category_codes
        pipeline._annotation_indexes = first._annotation_indexes
        pipeline._coding_masks_cache = first._coding_masks_cache


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
        raw = np.asarray(descriptor.get('ordered_rows'))
        if (raw.ndim != 1 or raw.dtype.kind not in 'iu' or not len(raw)
                or descriptor.get('n') != len(raw) or descriptor.get('family') not in ('gaussian', 'binomial')):
            raise ValueError('descriptor requires n/family and a nonempty ordered integer sample axis')
        rows = np.asarray(raw, dtype=np.int64)
        key = (len(rows), hashlib.sha256(rows.tobytes()).digest())
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
    """Use mature pipeline kernels without repeatedly rebuilding large eid maps.

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
    p.profiler = StageProfiler(model.device, enabled=configuration.get('stage_profile', False))
    p.resident_genotypes = True
    p.single_batch_optimization = True
    p.individual_effective_block_size = configuration.get('individual_effective_block_size', 1024)
    p.host_memory_guard = configuration.get('host_memory_guard')
    p.hybrid_mask_sizes, p.hybrid_union_M = [], 0
    p.maximum_mask_variants = configuration.get('maximum_mask_variants')
    if structural is not None:
        _share_annotations([structural, p])
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
            pipeline._hybrid_host_guard(model, len(physical)+len(columns))
            compact = block.select_columns(columns)
            # This handoff contains exact six-state-derived minor dosage, not
            # phenotype/model data. CPU conversion/imputation retains FP32.
            compact.dosage = compact.dosage.detach().cpu()
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
    pipeline._hybrid_host_guard(model, count)
    host = np.empty((model.n, count), dtype=np.float32, order='F')
    group = np.asarray(groups, dtype=np.int64)
    order = np.argsort(group, kind='stable')
    inverse = np.empty(count, dtype=np.int64)
    inverse[order] = np.arange(count)
    begin = 0
    for part in parts:
        end = begin + part.shape[1]
        dense = part.trait_dense(pipeline.trait_rows[0], options.imputation,
                                frequency_mode='reference', dtype=torch.float32)[0]
        host[:, inverse[begin:end]] = dense.numpy()
        begin = end
        del dense
    physical = np.asarray(physical, dtype=np.int64)[order]
    maf = np.asarray(frequencies, dtype=np.float64)[order]
    from ..pipeline import _cmac_scalar_sum
    return dict(_genotype_host=host, _variant_indices=indices[physical], cmac=_cmac_scalar_sum(host),
        _extraction_groups=group[order], maf=maf, mac=np.rint(maf*2*model.n),
        annotations=phred[physical], names=names, acat_calibration='chi2',
        rare_maf_cutoff=options.rare_maf_cutoff, rv_num_cutoff=options.rv_num_cutoff,
        rv_num_cutoff_max=options.rv_num_cutoff_max)


def _shared_gene_scores(genotype, models, broker, variant_tile_size=512):
    from ..tf32 import matmul
    n, m = genotype.shape
    broker._guard(4*(n*min(m, variant_tile_size) + n*len(models) + m*len(models)) + 64*2**20)
    residuals = torch.stack([model.scaled_residuals for model in models], 1)
    result = torch.empty((m, len(models)), dtype=torch.float32, device=models[0].device)
    if isinstance(genotype, torch.Tensor) and genotype.is_cuda:
        result.copy_(matmul(genotype.T, residuals, mode='tf32'))
    else:
        for begin in range(0, m, variant_tile_size):
            end = min(begin+variant_tile_size, m)
            block = torch.as_tensor(genotype[:, begin:end], dtype=torch.float32, device=models[0].device)
            result[begin:end] = matmul(block.T, residuals, mode='tf32')
            del block
    return result


def _gaussian_covariance(model, genotype):
    """Original fitted covariance formula, with the shared score supplied separately."""
    from ..tf32 import matmul
    rotated = model.spectrum.rotate(genotype, matmul_mode='tf32')
    cross = matmul(model.precision_x.T, genotype, mode='tf32')
    covariance = (matmul(rotated.T, model.inverse_variance[:, None]*rotated, mode='tf32')
                  - matmul(matmul(cross.T, model.fixed_effect_covariance, mode='tf32'), cross, mode='tf32'))
    return (covariance+covariance.T)/2


def _blocked_gaussian_covariance(model, host, broker, variant_tile_size=512):
    """Bounded exact fitted kinship-block formula for a host genotype mask.

    Each panel uses the model's original spectral rotation and inverse weights,
    then its own fixed-effect projection. Both directed FP32 TF32 covariance
    products are averaged, retaining the full covariance for ordinary/full
    spectra or FastSKAT. No diagonal approximation replaces relatedness blocks.
    CUDA panel geometry differs from one monolithic GEMM and requires the real
    numerical comparison; this function makes no bitwise equality claim.
    """
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
    if m > options.long_mask_threshold:
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
    from ..cache_runtime.single_batches import _physical_requests
    for request in _physical_requests(broker, selected, configuration.get('individual_genotype_block_size', 1024)):
        raw = broker.read_states(request, minimum_mac_bound=mac)
        for gi, group in enumerate(groups):
            axis = broker.reader_view(group['rows'])
            rows = axis._axis.samples
            block = broker.trait_block(raw, rows, minimum_mac=mac)
            prepared = None
            try:
                for begin in range(0, len(group['descriptors']), trait_batch_size):
                    tile = group['descriptors'][begin:begin+trait_batch_size]
                    with repository.acquire([d['trait_index'] for d in tile], str(broker.device)) as models:
                        pipelines = []
                        for model, descriptor in zip(models, tile):
                            _check_model(model, descriptor, metadata, verified)
                            pipelines.append(_thin_pipeline(axis, model, rows, options, configuration, structural))
                        if len(models) != len(tile):
                            raise ValueError('repository acquire must retain the requested model order/count')
                        if prepared is None:
                            state, ordinals[gi] = pipelines[0]._prepare_individual_block(block, ordinals[gi], mac_cutoff=mac)
                            if state is None:
                                break
                            prepared = pipelines[0]._prepare_individual_trait(state, 0, mac_cutoff=mac)
                            if prepared is None:
                                break
                            prepared.pop('model')
                        broker._guard(pipelines[0]._workspace_estimate(models[0], prepared['genotype'].shape[1], individual=True)
                                      + 4*models[0].n*len(models))
                        def correction(index, model, genotype, score, variance):
                            from ..binary import binary_single_phewas
                            return binary_single_phewas(model, genotype,
                                spa_acquire=lambda: repository.acquire_spa(tile[index]['trait_index'], str(broker.device)),
                                normal_score=score, normal_variance=variance,
                                p_filter_cutoff=options.p_filter_cutoff,
                                tol=options.spa_tol, max_iter=options.spa_max_iter)
                        outputs, _ = process_single_shared_genotype(pipelines, None, ordinals[gi], chromosome,
                            mac, subset, prepared_state=(prepared, ordinals[gi]), binary_correction=correction)
                        for descriptor, result in zip(tile, outputs):
                            writer.write('individual', descriptor['trait_index'], chromosome, result, job_index=job_index)
                            report['association_rows']['individual'] += len(result)
                            report['trait_rows'][descriptor['trait_index']]['individual'] += len(result)
                            if descriptor['family'] == 'binomial':
                                report['binary_spa_selected'] += sum(row['spa_selected'] for row in result)
                                report['binary_spa_failed'] += sum(row['spa_failed'] for row in result)
                        del outputs, pipelines, models, model
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
    """Lease host workspace for one complete gene job and release on all exits.

    A nonblocking growth rejection propagates to the dispatcher's job retry.
    Gene preparation/statistics precede every writer call, so a rejected
    preparation never publishes partial gene output. Retrying writer failures
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
        yield current


class _StagedGeneWriter:
    """Keep one already assembled gene job private until preparation succeeds."""
    def __init__(self):
        self.pending = []

    def write(self, *args, **kwargs):
        self.pending.append((args, kwargs))

    def publish(self, writer):
        for args, kwargs in self.pending:
            writer.write(*args, **kwargs)
        self.pending.clear()


def _scientific_counter_snapshot(report):
    return {key: deepcopy(report[key]) for key in
            ('association_rows', 'trait_rows', 'binary_spa_selected', 'binary_spa_failed', 'job_groups')
            if key in report}


def _execute_batched_job(attempt, *, configuration, host_memory_lease_factory,
                         kind, chromosome_index, job_index, writer, report, emit=None):
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
    baseline = _scientific_counter_snapshot(report)
    count = 0
    while True:
        staged = _StagedGeneWriter()
        deferred = None
        try:
            with _job_host_memory_scope(configuration, host_memory_lease_factory, kind=kind,
                    chromosome_index=chromosome_index, job_index=job_index) as current:
                attempt(current, staged)
        except HostMemoryDeferred as error:
            # Keep only anonymous integers. The exception and its traceback
            # would otherwise retain the partial genotype/model stack.
            deferred = dict(required_bytes=error.required_bytes, available_bytes=error.available_bytes)
            for key, value in baseline.items():
                report[key] = deepcopy(value)
            staged.pending.clear()
        if deferred is None:
            staged.publish(writer)
            return
        # This point is outside the except block and the lease context. Release
        # cyclic frames before waiting; verified reader/model caches may remain.
        del staged
        gc.collect()
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


def _batched_gene(groups, repository, broker, metadata, options, configuration, arguments,
                  writer, report, verified, chromosome, job_index, trait_batch_size, structural, kind):
    descriptors = [descriptor for group in groups for descriptor in group['descriptors']]
    positions = {descriptor['trait_index']: i for i, descriptor in enumerate(descriptors)}
    selector = _thin_pipeline(structural.genotype, structural.models[0], structural.union_rows,
                              options, configuration, structural)
    selector.models = [SimpleNamespace(use_spa=d['family'] == 'binomial') for d in descriptors]
    def evaluate(index_sets):
        masks = list(index_sets)
        result = [[None]*len(descriptors) for _ in masks]
        for group in groups:
            axis = broker.reader_view(group['rows'])
            try:
                for mi, indices in enumerate(masks):
                    anchor = group['descriptors'][0]
                    with repository.acquire([anchor['trait_index']], str(broker.device)) as models:
                        _check_model(models[0], anchor, metadata, verified)
                        prototype = _thin_pipeline(axis, models[0], axis._axis.samples, options, configuration, structural)
                        prepared = _prepare_shared_gene(prototype, indices, broker)
                        q = models[0].x.shape[1]
                    prototype.models = []
                    del models
                    if prepared is None:
                        continue
                    host = prepared['_genotype_host']
                    genotype, burdens = host, None
                    n, m = host.shape
                    has_gaussian = any(d['family'] == 'gaussian' for d in group['descriptors'])
                    score_new = 4*(n*min(m, 512)+n*min(trait_batch_size, len(group['descriptors']))
                                   +m*min(trait_batch_size, len(group['descriptors']))) + 64*2**20
                    tail_new = _gene_tail_new_bytes(n, m, options) if has_gaussian else 0
                    host_peak = (max(_gene_host_covariance_new_bytes(n, m, q, options),
                                     4*m*m+tail_new, score_new) if has_gaussian else score_new)
                    with _gene_workspace_scope(repository, broker, host_peak) as workspace:
                        # Materialize only if the live device can admit original
                        # fitted ordinary covariance products plus this one G.
                        if m <= options.long_mask_threshold and torch.device(broker.device).type == 'cuda':
                            # Relatedness may vary between traits in one cohort;
                            # two genotype copies cover the original block path.
                            device_peak = max(8*host.size+32*m*m+256*2**20,
                                              4*m*m+tail_new, score_new)
                            try:
                                workspace.set_required(4*host.size+device_peak, phase='gene_genotype_upload')
                            except MemoryError:
                                # Preserve the established bounded-host route
                                # when full G plus pinned state cannot fit.
                                pass
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
                                scores = (_shared_gene_scores(genotype, [models[i] for i in gaussian_indices], broker)
                                          if gaussian_indices else None)
                                gauss_columns = {i: j for j, i in enumerate(gaussian_indices)}
                                for ti, (descriptor, model) in enumerate(zip(tile, models)):
                                    p = _thin_pipeline(axis, model, axis._axis.samples, options, configuration, structural)
                                    if model.family == 'binomial':
                                        from ..binary import prepare_binary_burdens, staar_binary_phewas
                                        if burdens is None:
                                            burdens = prepare_binary_burdens(model, genotype, prepared['maf'],
                                                prepared['annotations'], prepared['names'], rare_maf_cutoff=options.rare_maf_cutoff,
                                                rv_num_cutoff=options.rv_num_cutoff, rv_num_cutoff_max=options.rv_num_cutoff_max)
                                        value, diagnostic = staar_binary_phewas(model, genotype, prepared['maf'],
                                            prepared['annotations'], prepared['names'],
                                            spa_acquire=lambda: repository.acquire_spa(descriptor['trait_index'], str(broker.device)),
                                            prepared_burdens=burdens, rare_maf_cutoff=options.rare_maf_cutoff,
                                            rv_num_cutoff=options.rv_num_cutoff, rv_num_cutoff_max=options.rv_num_cutoff_max,
                                            p_filter_cutoff=options.p_filter_cutoff, tol=options.spa_tol,
                                            max_iter=options.spa_max_iter, return_diagnostics=True)
                                        report['binary_spa_selected'] += int(diagnostic['spa_selected'].sum())
                                        report['binary_spa_failed'] += int(diagnostic['spa'].failed.sum())
                                        if diagnostic['variant_covariance_constructed']:
                                            raise RuntimeError('binary burden path must not construct a variant covariance')
                                    else:
                                        cov_new = (4*host.size*(2 if model.spectrum.blocks else 1)
                                                   +32*m*m+256*2**20
                                                   if isinstance(genotype, torch.Tensor) and genotype.is_cuda
                                                   else _gene_host_covariance_new_bytes(n, m, model.x.shape[1], options))
                                        workspace.set_required(max(cov_new, 4*m*m+tail_new), phase='gene_covariance')
                                        if isinstance(genotype, torch.Tensor) and genotype.is_cuda:
                                            copies = 2 if model.spectrum.blocks else 1
                                            broker._guard(4*host.size*copies + 32*host.shape[1]**2 + 256*2**20)
                                            covariance = _gaussian_covariance(model, genotype)
                                        else:
                                            # Cached/tiled covariance is fitted-model specific;
                                            # only G IO and phenotype-score GEMM are shared.
                                            if p._supports_long_tf32_products(model):
                                                _, covariance = p._long_mask_products(model, host)
                                            else:
                                                covariance = _blocked_gaussian_covariance(model, host, broker,
                                                    variant_tile_size=options.variant_tile_size or 512)
                                        # Its input covariance is now allocated.
                                        # Reserve only NEW STAAR/tail bytes here.
                                        workspace.set_required(tail_new, phase='gene_tail')
                                        payload = {key: value for key, value in prepared.items() if not key.startswith('_')}
                                        payload.update(score=scores[:, gauss_columns[ti]], covariance=covariance,
                                                       cmac=prepared['cmac'])
                                        value = p._evaluate_prepared(payload, model)
                                        del covariance, payload
                                    result[mi][positions[descriptor['trait_index']]] = _phewas_result_fields(value)
                                    del p, model
                                del scores, models
                    del prepared, host, genotype, burdens
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
        writer.write(kind, descriptor['trait_index'], chromosome, result, job_index=job_index)
        count = sum(len(rows) for rows in result.values()) if isinstance(result, dict) else len(result)
        report['association_rows'][kind] += count
        report['trait_rows'][descriptor['trait_index']][kind] += count


def run_batched_configuration(configuration, *, model_repository, reader_factory, writer,
                              device='cuda:0', trait_batch_size=32, memory_limit_gib=None,
                              cpu_threads=1, cache_specs=None, device_cache_bytes=512*2**20,
                              compact_cache_bytes=64*2**20, metadata_cache_bytes=256*2**20,
                              memory_reserve_bytes=256*2**20, emit=None,
                              host_memory_lease_factory=None):
    """Run a cache-only frame/job-major independent multi-phenotype analysis.

    The repository exposes descriptors/acquire/acquire_spa, and the injected
    reader factory context returns metadata/container/portable_axis. The writer
    streams every result immediately. GPU null states are bounded by repository
    admission and trait tiles; neither all-trait G nor M² covariances are stacked.
    None removes artificial GiB caps while retaining live-memory reservation.
    Binary formal probabilities use on-demand fitted FP64 SPA sidecars. Shared
    TF32 score GEMM is a new arithmetic route, not claimed reference-equivalent.
    host_memory_lease_factory receives kind/chromosome_index/job_index and yields
    a guard(n,m,phase) or a lease exposing guard. Shared-host dispatchers supply
    a nonblocking lease, retry deferred jobs after this context releases state,
    and retain the host budget independently of the GPU physical-memory budget.
    """
    if type(trait_batch_size) is not int or trait_batch_size < 1 or type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError('trait_batch_size and cpu_threads must be positive integers')
    if type(memory_reserve_bytes) is not int or memory_reserve_bytes < 0:
        raise ValueError('memory_reserve_bytes must be a nonnegative integer')
    if not callable(reader_factory) or not callable(getattr(writer, 'write', None)):
        raise ValueError('portable reader_factory and streaming writer are required')
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('PheWAS context is already running')
    previous_threads = torch.get_num_threads()
    repository_policy = ExitStack()
    try:
        if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('native batched PheWAS production requires CUDA')
        torch.set_num_threads(cpu_threads)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
        budget = _resolved_budget(memory_limit_gib, device)
        configuration = dict(configuration)
        if host_memory_lease_factory is not None and not callable(host_memory_lease_factory):
            raise TypeError('host_memory_lease_factory must be callable or None')
        configuration['_batched_emit'] = emit
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
        for descriptor in descriptors:
            if descriptor['family'] == 'binomial' and not callable(getattr(model_repository, 'acquire_spa', None)):
                raise ValueError('formal binary analysis requires a fitted FP64 SPA repository')
        configured = configure_tf32(split_k=0, memory_limit_gib=budget,
                                    memory_reserve_bytes=memory_reserve_bytes)
        policy = getattr(model_repository, 'memory_policy', None)
        if callable(policy):
            repository_policy.enter_context(policy(device, limit_bytes=(None if memory_limit_gib is None else int(budget*2**30)),
                                                  reserve_bytes=memory_reserve_bytes))
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
                                if emit:
                                    emit(dict(event='started', chromosome_index=ci, job_index=ji, kind=kind))
                                function = _batched_single if kind == 'individual' else _batched_gene
                                def attempt(job_configuration, current_writer):
                                    args = (groups, model_repository, broker, metadata, options, job_configuration, arguments,
                                        current_writer, report, verified, reference['name'], ji, trait_batch_size, structural)
                                    if kind == 'individual':
                                        function(*args)
                                    else:
                                        function(*args, kind)
                                _execute_batched_job(attempt, configuration=configuration,
                                    host_memory_lease_factory=host_memory_lease_factory, kind=kind,
                                    chromosome_index=ci, job_index=ji, writer=writer, report=report, emit=emit)
                                report['job_groups'] += 1
                                if emit:
                                    emit(dict(event='finished', chromosome_index=ci, job_index=ji, kind=kind,
                                              wall_seconds=time.perf_counter()-before))
                            report['chromosome_readers'].append(dict(broker=broker.metrics, metadata=dict(metadata.metrics)))
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
            cpu_thread_control=dict(requested=cpu_threads, effective=torch.get_num_threads(), previous=previous_threads,
                                    restored=True, scope='pytorch_intraop'),
            output=writer.finish())
        if callable(getattr(model_repository, 'summary', None)):
            report['model_repository'] = model_repository.summary()
        return report
    except BaseException:
        abort = getattr(writer, 'abort', None)
        if callable(abort):
            abort()
        raise
    finally:
        try:
            repository_policy.close()
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
