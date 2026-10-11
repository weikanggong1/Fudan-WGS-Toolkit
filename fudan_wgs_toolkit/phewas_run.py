"""One-submission, cache-only multi-phenotype WGS execution.

Phenotype CSV + covariate CSV + portable disk cache are the scientific inputs.
The portable dataset declares its ID-bound GRM/cohort sidecars and gene catalogs.
Deployment plans and participant/trait identifiers stay in the private output.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
import re
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from .phewas_inputs import PhewasInputs, prepare_phewas_inputs
from .phewas_models import SparseKinshipData, prepare_phewas_model
from .phewas_storage import ModelRepository, StreamingCSVWriter, atomic_json, digest_file, save_model_store
from .run import _implementation_sha256
from .phewas_cache import _dataset, _analysis_defaults


def _file_stat(path):
    value = Path(path).stat()
    # Device numbers differ between mounts on the two consumers. These four
    # properties plus the bound SHA retain the original immutable-file proof.
    return dict(ino=value.st_ino, size=value.st_size,
                mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _bound_json(path, expected):
    if not _hash(expected):
        raise ValueError('reuse receipt requires a SHA-256 binding')
    path = Path(path)
    before = _file_stat(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected or _file_stat(path) != before:
        raise ValueError('reuse receipt checksum or immutable identity differs')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('reuse receipt must be an object')
    return value


def _reuse_context(plan):
    """Validate separate original-fit and current-consumer provenance.

    Mathematical compatibility is an independent source-code receipt. The
    import inventory additionally binds actual inputs, all original files and
    successful original fit processes. Neither receipt changes model metadata.
    """
    reuse = plan.get('reuse_models_only', False)
    if type(reuse) is not bool:
        raise ValueError('reuse_models_only must be a boolean')
    consumer = plan['source_implementation_sha256']
    fit_source = plan.get('model_fit_source_implementation_sha256', consumer)
    if not reuse:
        if fit_source != consumer or any(key in plan for key in (
                'null_fit_compatibility_receipt', 'null_fit_compatibility_sha256',
                'model_import_receipt', 'model_import_sha256')):
            raise ValueError('separate fit provenance requires immutable reuse-only mode')
        return None
    if not _hash(consumer) or not _hash(fit_source) or _implementation_sha256() != consumer:
        raise ValueError('reuse consumer/source implementation binding differs')
    if type(plan.get('trait_count')) is not int or plan['trait_count'] < 1:
        raise ValueError('reuse requires a positive complete phenotype count')
    proof = _bound_json(plan['null_fit_compatibility_receipt'], plan['null_fit_compatibility_sha256'])
    if (proof.get('schema_version') != 1 or proof.get('complete') is not True
            or proof.get('compatible_null_fit_math') is not True
            or proof.get('source_unchanged_during_review') is not True
            or proof.get('scientific_model_input_changes') is not False
            or proof.get('fitted_math_or_parameter_changes') is not False
            or proof.get('model_fit_source_implementation_sha256') != fit_source
            or proof.get('consumer_source_implementation_sha256') != consumer):
        raise ValueError('null-fit compatibility receipt does not authorize this reuse')
    components = proof.get('components', {})
    required = {'binary_null.py:complete_module', 'phewas_models.py:complete_module',
        'phewas_inputs.py:complete_module', 'rint.py:complete_module', 'numerics.py:complete_module',
        'tensor_validation.py:complete_module',
        'null_model.py:fit_gaussian_null', 'null_model.py:KinshipSpectrum',
        'null_model.py:rank_inverse_normal', 'null_model.py:_r_sum',
        'null_model.py:GaussianNullModel.dataclass_fields',
        'null_model.py:GaussianNullModel.set_matmul_mode_except_new_cache_clear', 'phewas_run.py:_design'}
    if not isinstance(components, dict) or not required <= components.keys():
        raise ValueError('null-fit compatibility receipt lacks required scientific components')
    if 'tf32.py:complete_module' not in components:
        policy = proof.get('tf32_resource_policy_compatibility', {})
        if ('tf32.py:numerical_component' not in components
                or not isinstance(policy, dict)
                or policy.get('kind') != 'configure_tf32_reserve_only'
                or policy.get('resource_policy_only') is not True
                or policy.get('numerical_component_ast_identical') is not True
                or policy.get('configure_tf32_resource_additions_verified') is not True
                or policy.get('complete_tf32_module_identical') is not False
                or policy.get('changed_tf32_definitions') != ['configure_tf32']
                or not _hash(policy.get('baseline_null_fit_compatibility_sha256'))
                or not _hash(policy.get('baseline_consumer_source_implementation_sha256'))
                or policy.get('model_fit_source_implementation_sha256') != fit_source
                or policy.get('consumer_source_implementation_sha256') != consumer):
            raise ValueError('null-fit compatibility receipt lacks strict TF32 numerical/resource-policy proof')
    joined = hashlib.sha256()
    for name, record in sorted(components.items()):
        if (not isinstance(record, dict) or record.get('identical') is not True
                or not _hash(record.get('old_fingerprint'))
                or record.get('new_fingerprint') != record['old_fingerprint']):
            raise ValueError('null-fit scientific component fingerprints differ')
        joined.update(name.encode())
        joined.update(bytes.fromhex(record['old_fingerprint']))
    if (proof.get('old_null_fit_component_fingerprint') != joined.hexdigest()
            or proof.get('new_null_fit_component_fingerprint') != joined.hexdigest()):
        raise ValueError('null-fit aggregate fingerprint differs')
    qualification = proof.get('existing_models_reuse', {})
    for flag in ('scientifically_compatible', 'original_model_metadata_and_array_hashes_must_be_preserved',
                 'consumer_and_fit_source_binding_must_be_separate', 'missing_reuse_target_must_hard_fail',
                 'no_new_fits_may_be_labelled_as_old_source'):
        if qualification.get(flag) is not True:
            raise ValueError('null-fit compatibility receipt lacks strict reuse qualification')
    inventory = _bound_json(plan['model_import_receipt'], plan['model_import_sha256'])
    bindings = dict(model_fit_source_implementation_sha256=fit_source,
        consumer_source_implementation_sha256=consumer,
        null_fit_compatibility_sha256=plan['null_fit_compatibility_sha256'],
        prepared_inputs_manifest_sha256=plan['prepared_inputs_manifest_sha256'],
        kinship_sha256=plan['kinship_sha256'], cohort_sha256=plan.get('cohort_sha256'),
        continuous_transform=plan.get('continuous_transform', 'paper'), model_count=plan['trait_count'])
    if (inventory.get('schema_version') != 1 or inventory.get('status') != 'verified'
            or any(inventory.get(key) != value for key, value in bindings.items())
            or inventory.get('original_model_stores_modified') is not False
            or type(inventory.get('models_refitted')) is not int or inventory['models_refitted'] != 0
            or any(inventory.get(key) is not True for key in (
                'all_fit_supervisor_exits_zero', 'full_array_hashes_verified', 'full_array_geometry_verified'))):
        raise ValueError('model import receipt scientific/source bindings differ')
    old_run = Path(inventory['from_run_directory']).resolve()
    original = old_run/'outputs/models'
    destination = Path(plan['output_directory']).resolve()/'models'
    if (Path(inventory['from_models_directory']) != original
            or Path(inventory['models_directory']) != destination or original == destination):
        raise ValueError('model import source/destination directories differ')
    records = inventory.get('models', [])
    if (not isinstance(records, list) or len(records) != plan['trait_count']
            or [record.get('trait_index') for record in records] != list(range(plan['trait_count']))):
        raise ValueError('model import coverage differs from the complete phenotype plan')
    by_index = {}
    for record in records:
        index = record['trait_index']
        target, link = original/f'trait_{index:04d}', destination/f'trait_{index:04d}'
        if (type(index) is not int or Path(record['target']) != target
                or Path(record['symlink']) != link or target.resolve() != target or target.is_symlink()
                or not _hash(record.get('metadata_sha256'))
                or record.get('family') not in ('gaussian', 'binomial')
                or type(record.get('n')) is not int or record['n'] < 1):
            raise ValueError('model import target/index/family geometry differs')
        by_index[index] = record
    count = inventory.get('source_fit_worker_count')
    keys = {str(i) for i in range(count)} if type(count) is int and count > 0 else set()
    exits, complete = inventory.get('fit_supervisor_exit_sha256', {}), inventory.get('fit_complete_receipt_sha256', {})
    if not keys or set(exits) != keys or set(complete) != keys:
        raise ValueError('model import original fit-process coverage differs')
    old_plan = _bound_json(old_run/'outputs/plan.private.json', inventory['from_plan_sha256'])
    old_bindings = dict(source_implementation_sha256=fit_source, trait_count=plan['trait_count'],
        prepared_inputs_manifest_sha256=plan['prepared_inputs_manifest_sha256'],
        kinship_sha256=plan['kinship_sha256'], cohort_sha256=plan.get('cohort_sha256'))
    if (any(old_plan.get(key) != value for key, value in old_bindings.items())
            or old_plan.get('continuous_transform', 'paper') != plan.get('continuous_transform', 'paper')):
        raise ValueError('original fit plan scientific/source bindings differ')
    for i in range(count):
        exit_receipt = _bound_json(old_run/'outputs'/f'worker_{i}.fit.supervisor_exit.anonymous.json', exits[str(i)])
        if (type(exit_receipt.get('worker_id')) is not int or exit_receipt['worker_id'] != i
                or exit_receipt.get('phase') != 'fit'
                or type(exit_receipt.get('exit_code')) is not int or exit_receipt['exit_code'] != 0
                or exit_receipt.get('source_implementation_sha256') != fit_source
                or exit_receipt.get('plan_sha256') != inventory['from_plan_sha256']):
            raise ValueError('original fit process did not successfully exit under the bound source')
        receipt = _bound_json(old_run/'outputs'/f'fit_worker_{i}.complete.private.json', complete[str(i)])
        expected_indices = list(range(i, plan['trait_count'], count))
        if (type(receipt.get('worker_id')) is not int or receipt['worker_id'] != i or receipt.get('errors') != []
                or receipt.get('assigned') != len(expected_indices)
                or [entry.get('trait_index') for entry in receipt.get('models', [])] != expected_indices):
            raise ValueError('original fit receipt model coverage differs')
        for entry in receipt['models']:
            record = by_index[entry['trait_index']]
            if (Path(entry['path']) != Path(record['target']) or entry.get('sha256') != record['metadata_sha256']
                    or entry.get('family') != record['family'] or entry.get('n') != record['n']):
                raise ValueError('original fitted-model receipts differ from the import inventory')
    return dict(fit_source=fit_source, inventory=inventory, models=by_index)


def _verify_reused_model(plan, context, index):
    record = context['models'][index]
    link, original = Path(record['symlink']), Path(record['target'])
    if not link.is_symlink() or not original.is_dir() or link.resolve() != original:
        raise ValueError('reuse-only model target is missing or its link binding differs')
    files = record.get('files', [])
    bound = {}
    for item in files:
        path = Path(item['path'])
        if (path in bound or path.resolve() != path or path.is_symlink()
                or not path.is_relative_to(original) or not _hash(item.get('sha256'))
                or _file_stat(path) != item.get('stat')):
            raise ValueError('immutable imported model file identity differs')
        bound[path] = item['sha256']
    metadata = _bound_json(original/'model.private.json', record['metadata_sha256'])
    expected = dict(input_manifest_sha256=plan['prepared_inputs_manifest_sha256'],
        kinship_sha256=plan['kinship_sha256'], source_implementation_sha256=context['fit_source'],
        cohort_sha256=plan.get('cohort_sha256'), continuous_transform=plan.get('continuous_transform', 'paper'))
    if (any(metadata['fit'].get(key) != value for key, value in expected.items())
            or metadata['fit'].get('trait_index') != index or metadata.get('n') != record['n']
            or metadata.get('family') != record['family'] or metadata.get('normal_state') != 'normal'
            or metadata.get('spa_state') != ('spa' if record['family'] == 'binomial' else None)):
        raise ValueError('reused model metadata/index/scientific bindings differ')
    expected_files = {original/'model.private.json': record['metadata_sha256'],
                      original/'sample_rows.npy': metadata['sample_rows_sha256']}
    for kind, key in (('normal', 'state_sha256'), ('spa', 'spa_state_sha256')):
        if metadata.get(key) is None:
            continue
        state_path = original/kind/'state.private.json'
        state = _bound_json(state_path, metadata[key])
        if (state.get('family') != record['family'] or state.get('converged') is not True
                or state.get('has_kinship') is not True
                or state.get('matmul_mode') != ('fp64' if kind == 'spa' else 'tf32')):
            raise ValueError('reused normal/SPA state is not a converged bound mixed model')
        expected_files[state_path] = metadata[key]
        for spec in state['arrays'].values():
            name = spec['file']
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError('reused state array path escaped its directory')
            expected_files[original/kind/name] = spec['sha256']
    if bound != expected_files:
        raise ValueError('import inventory does not bind every fitted-state array')
    # The import helper fully hashed arrays and checked their geometry. Exact
    # unchanged identities preserve that proof here; the storage loader also
    # independently hashes every array it materializes for association.
    return dict(trait_index=index, path=str(link), family=record['family'],
                n=record['n'], sha256=record['metadata_sha256'])


def _fit_binding(plan, *, _context=None):
    context = _reuse_context(plan) if _context is None else _context
    return dict(kinship_sha256=plan['kinship_sha256'],
                source_implementation_sha256=(plan['source_implementation_sha256'] if context is None
                                              else context['fit_source']),
                cohort_sha256=plan.get('cohort_sha256'),
                continuous_transform=plan.get('continuous_transform', 'paper'))


def _design(raw, names):
    """Add intercept, standardize numerical fixed effects and drop redundancies.

    Standardization preserves the exact fixed-effect span. Removing a constant
    sex column in a sex-specific trait preserves that span rather than excluding
    the entire phenotype. Retained/dropped columns and transformations are saved.
    """
    raw = np.asarray(raw, dtype=np.float64)
    center, scale = raw.mean(0), raw.std(0)
    candidates = np.column_stack((np.ones(len(raw)),
        (raw-center)/np.where(scale > 0, scale, 1)))
    labels = ['Intercept', *names]
    retained, orthogonal = [], []
    for index in range(candidates.shape[1]):
        value = candidates[:, index].copy()
        original = np.linalg.norm(value)
        for vector in orthogonal:
            value -= vector * np.dot(vector, value)
        # A second pass avoids false admission of a nearly redundant column.
        for vector in orthogonal:
            value -= vector * np.dot(vector, value)
        norm = np.linalg.norm(value)
        if original and norm > original * 1e-10:
            retained.append(index)
            orthogonal.append(value/norm)
    return candidates[:, retained], dict(covariate_names=[labels[i] for i in retained],
        dropped_covariate_names=[labels[i] for i in range(len(labels)) if i not in retained],
        raw_covariate_names=list(names), center=center.tolist(), scale=scale.tolist())


def build_configuration(cache_directory, *, chromosomes=None, maximum_single_variants=None):
    """Build one chromosome/genetic schedule, independent of phenotype count."""
    root, manifest_path, manifest, entries, ids = _dataset(cache_directory, chromosomes)
    from .phewas_cache import _cache_path
    catalog = manifest.get('annotation_catalog') or {}
    if isinstance(catalog, str):
        catalog = str(_cache_path(root, catalog))
    elif not isinstance(catalog, dict):
        raise ValueError('annotation_catalog must be a field mapping or relative JSON path')
    schedules = []
    for entry in entries:
        jobs = [dict(kind='individual', arguments={})]
        if maximum_single_variants is not None:
            jobs[0]['arguments']['end'] = maximum_single_variants
            jobs[0]['arguments']['start'] = 1
        genes = json.loads(Path(entry['gene_catalog']).read_text())
        if isinstance(genes, dict):
            genes = genes.get('jobs', genes.get('genes'))
        for gene in genes:
            kind = gene.get('kind')
            if kind not in ('coding', 'noncoding', 'ncrna'):
                continue
            arguments = dict(gene.get('arguments', {}))
            arguments.pop('chromosome', None)
            arguments.pop('promoter_intervals_file', None)
            for field in ('gene_name', 'start', 'end', 'category', 'include_ptv', 'include_ncrna'):
                if field in gene:
                    arguments[field] = gene[field]
            if kind == 'coding':
                # The reusable one-trait catalogs can contain five categories.
                # This PheWAS schedule includes all seven published categories.
                arguments.update(category='all_categories_incl_ptv', include_ptv=True)
            jobs.append(dict(kind=kind, arguments=arguments))
        # Complete gene-based outputs first, then the much larger Single CSVs.
        jobs = jobs[1:] + jobs[:1]
        schedules.append(dict(name=entry['name'], gds=entry['container_directory'],
            container_directory=entry['container_directory'], metadata_directory=entry['metadata_directory'],
            annotation_index=dict(promoter_intervals_file=entry.get('promoter_intervals')),
            jobs=jobs))
    options = _analysis_defaults(80., 4096, 5000, 512, 1729)
    options['memory_limit_gib'] = None
    return dict(chromosomes=schedules, qc_path=manifest.get('qc_path', 'annotation/info/QC_label'),
        annotation_catalog=catalog,
        annotation_names=manifest.get('annotation_names'),
        matmul_mode='tf32', analysis_options=options, resident_genotypes=True,
        local_mask_reuse=True, statistics_tail_optimization=True, weight_batch_optimization=True,
        individual_effective_block_size=1024, individual_genotype_block_size=1024,
        stage_profile=True, source_implementation_sha256=_implementation_sha256())


@contextmanager
def portable_reader(reference, spec=None):
    """Verify physical cache/metadata axes and expose logical population rows."""
    from .cache_runtime.portable import PortableMetadataReader
    from .cache_runtime.fast_container import Container
    if isinstance(reference, str):
        directory = Path(reference)
        metadata_directory = Path(spec['metadata_directory']) if spec and spec.get('metadata_directory') else directory/'metadata'
    else:
        directory = Path(reference['container_directory'])
        metadata_directory = Path(reference['metadata_directory'])
    metadata = PortableMetadataReader(metadata_directory, directory, verify_checksums=True,
                                      legacy_numeric=True)
    container = None
    try:
        container = Container(directory)
        if not np.array_equal(container.samples, metadata._array(metadata.manifest['source_sample_rows'])):
            raise ValueError('cache source sample rows differ from portable metadata')
        yield dict(metadata=metadata, container=container, portable_axis=True)
    finally:
        metadata.close()
        if container is not None:
            container.close()


def fit_models(plan, *, worker_id=0, worker_count=1, device='cuda:0', trait_indices=None):
    """Fit every assigned phenotype once; publish immutable model receipts."""
    context = _reuse_context(plan)
    fit_binding = _fit_binding(plan, _context=context)
    output = Path(plan['output_directory'])
    models = output/'models'
    models.mkdir(exist_ok=True)
    inputs = PhewasInputs(plan['prepared_inputs_directory'])
    input_sha = digest_file(inputs.directory/'manifest.private.json')
    if input_sha != plan.get('prepared_inputs_manifest_sha256'):
        inputs.close()
        raise ValueError('prepared input manifest differs from the immutable dispatch plan')
    cohort_sha = digest_file(plan['cohort_rows']) if plan.get('cohort_rows') else None
    if cohort_sha != plan.get('cohort_sha256') or digest_file(plan['kinship_npz']) != plan['kinship_sha256']:
        raise ValueError('GRM/cohort checksum differs from the immutable dispatch plan')
    # Reuse mode verifies saved models only. It never enters a fitting path or
    # loads a fresh GRM object that could later be used to label a new fit old.
    kinship = None if context is not None else SparseKinshipData.load_npz(plan['kinship_npz'])
    cohort = None if context is not None else (np.load(plan['cohort_rows'], allow_pickle=False)
                                              if plan.get('cohort_rows') else None)
    assigned = list(range(worker_id, inputs.manifest['trait_count'], worker_count)) if trait_indices is None else list(trait_indices)
    started, successes, errors = time.time(), [], []
    try:
        for index in assigned:
            target = models/f'trait_{index:04d}'
            if context is not None:
                successes.append(_verify_reused_model(plan, context, index))
                atomic_json(output/f'fit_worker_{worker_id}.progress.anonymous.json', dict(
                    assigned=len(assigned), completed=len(successes), failed=0,
                    last_trait_index=index, elapsed_seconds=time.time()-started,
                    mode='verified_reuse', source_implementation_sha256=plan['source_implementation_sha256'],
                    model_fit_source_implementation_sha256=context['fit_source']))
                continue
            if target.exists():
                meta = json.loads((target/'model.private.json').read_text())
                expected = dict(input_manifest_sha256=input_sha, **fit_binding)
                if any(meta['fit'].get(key) != value for key, value in expected.items()):
                    raise ValueError('existing null model is bound to another input/source/cohort')
                successes.append(dict(trait_index=index, path=str(target), family=meta['family'],
                                      n=meta['n'], sha256=digest_file(target/'model.private.json')))
                continue
            before = time.perf_counter()
            try:
                trait = inputs.trait(int(index), cohort_indices=cohort)
                x, design = _design(trait['covariates'], trait['covariate_names'])
                fitted = prepare_phewas_model(trait['y_raw'], trait['sample_ids'], x,
                    family=trait['family'], kinship=kinship, continuous_transform=plan.get('continuous_transform', 'paper'),
                    device=device, association_mode='tf32', use_spa=True)
                if not fitted.model.converged:
                    raise ArithmeticError('null fit did not converge')
                rows = trait['sample_indices'][fitted.sample_indices]
                if digest_file(inputs.directory/'manifest.private.json') != input_sha:
                    raise ValueError('prepared input manifest changed during null fitting')
                metadata = dict(fitted.metadata, **design, trait_index=index, phenotype=trait['name'],
                    profile=trait['profile'], input_manifest_sha256=input_sha,
                    **fit_binding,
                    wall_seconds=time.perf_counter()-before)
                entry = save_model_store(fitted.model, target, sample_rows=rows, fit_metadata=metadata,
                                         spa_model=fitted.spa_model)
                entry['trait_index'] = index
                successes.append(entry)
                del fitted, trait, x
                gc.collect()
            except Exception as error:
                record = dict(trait_index=index, error_type=type(error).__name__, error=str(error),
                              traceback=traceback.format_exc(), wall_seconds=time.perf_counter()-before)
                errors.append(record)
                atomic_json(models/f'trait_{index:04d}.failed.private.json', record)
            atomic_json(output/f'fit_worker_{worker_id}.progress.anonymous.json', dict(
                assigned=len(assigned), completed=len(successes), failed=len(errors),
                last_trait_index=index, elapsed_seconds=time.time()-started,
                source_implementation_sha256=plan['source_implementation_sha256']))
        result = dict(worker_id=worker_id, assigned=len(assigned), models=successes, errors=errors,
                      elapsed_seconds=time.time()-started,
                      mode='verified_reuse' if context is not None else 'fresh_fit',
                      model_fit_source_implementation_sha256=fit_binding['source_implementation_sha256'],
                      consumer_source_implementation_sha256=plan['source_implementation_sha256'],
                      null_fit_compatibility_sha256=plan.get('null_fit_compatibility_sha256'),
                      model_import_sha256=plan.get('model_import_sha256'))
        atomic_json(output/f'fit_worker_{worker_id}.complete.private.json', result)
        return result
    finally:
        inputs.close()


def _all_models(plan, worker_count):
    context = _reuse_context(plan)
    receipts = [Path(plan['output_directory'])/f'fit_worker_{i}.complete.private.json'
                for i in range(worker_count)]
    while not all(path.exists() for path in receipts):
        for i in range(worker_count):
            root = Path(plan['output_directory'])
            if (root/f'worker_{i}.failed.private.json').exists():
                raise RuntimeError('a distributed worker failed before the model barrier')
            exit_file = root/f'worker_{i}.exit.anonymous.json'
            if exit_file.exists() and json.loads(exit_file.read_text()).get('exit_code', 0) != 0:
                raise RuntimeError('a distributed worker exited before the model barrier')
        time.sleep(10)
    results = [json.loads(path.read_text()) for path in receipts]
    if context is not None:
        for i, result in enumerate(results):
            if (type(result.get('worker_id')) is not int or result['worker_id'] != i
                    or result.get('mode') != 'verified_reuse' or result.get('errors') != []
                    or result.get('model_fit_source_implementation_sha256') != context['fit_source']
                    or result.get('consumer_source_implementation_sha256') != plan['source_implementation_sha256']
                    or result.get('null_fit_compatibility_sha256') != plan['null_fit_compatibility_sha256']
                    or result.get('model_import_sha256') != plan['model_import_sha256']
                    or result.get('assigned') != len(range(i, plan['trait_count'], worker_count))
                    or [entry.get('trait_index') for entry in result.get('models', [])]
                        != list(range(i, plan['trait_count'], worker_count))):
                raise ValueError('current reuse verification receipt worker/provenance coverage differs')
    if any(result['errors'] for result in results):
        raise RuntimeError('one or more null fits failed; full association cannot silently omit phenotypes')
    models = sorted((entry for result in results for entry in result['models']), key=lambda entry: entry['trait_index'])
    if [entry['trait_index'] for entry in models] != list(range(plan['trait_count'])):
        raise RuntimeError('fitted-model coverage differs from the complete phenotype plan')
    expected = dict(input_manifest_sha256=plan['prepared_inputs_manifest_sha256'],
                    **_fit_binding(plan, _context=context))
    if digest_file(Path(plan['prepared_inputs_directory'])/'manifest.private.json') != expected['input_manifest_sha256']:
        raise ValueError('prepared input manifest changed before the model barrier')
    for entry in models:
        if context is not None:
            verified = _verify_reused_model(plan, context, entry['trait_index'])
            if any(entry.get(key) != value for key, value in verified.items()):
                raise ValueError('current fitted-model receipt differs from the immutable import inventory')
        model_path = Path(entry['path'])/'model.private.json'
        if digest_file(model_path) != entry['sha256']:
            raise ValueError('fitted model receipt checksum differs')
        metadata = json.loads(model_path.read_text())
        if any(metadata['fit'].get(key) != value for key,value in expected.items()):
            raise ValueError('fitted models do not share the bound input/source/cohort')
    return models


def _execute_worker(plan, *, worker_id=0, worker_count=1, device='cuda:0', fit_only=False):
    """One GPU worker: fixed eight-CPU affinity, then genetic-major association."""
    affinity = sorted(os.sched_getaffinity(0))
    cpu_offset = int(plan.get('cpu_offsets', {}).get(str(worker_id), 0))
    selected = affinity[cpu_offset:cpu_offset+8]
    if len(selected) != 8:
        raise ValueError('worker must have eight assigned CPUs')
    os.sched_setaffinity(0, selected)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('production PheWAS requires the configured CUDA device')
    torch.cuda.set_device(device)
    # Libraries imported before this entry point may already own helper
    # threads. Restrict them too, then all later loader/CUDA threads inherit
    # the same worker allocation. The node cgroup remains the shared limit.
    for _ in range(2):
        for thread in Path('/proc/self/task').iterdir():
            try:
                os.sched_setaffinity(int(thread.name), selected)
            except ProcessLookupError:
                pass
    thread_affinities = {}
    for thread in Path('/proc/self/task').iterdir():
        try:
            thread_affinities[thread.name] = sorted(os.sched_getaffinity(int(thread.name)))
        except ProcessLookupError:
            pass
    if any(cpus != selected for cpus in thread_affinities.values()):
        raise RuntimeError('worker helper thread escaped the assigned CPU affinity')
    started = time.time()
    output = Path(plan['output_directory'])
    atomic_json(output/f'worker_{worker_id}.identity.private.json', dict(pid=os.getpid(),
        start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split(') ')[1].split()[19],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), cpus=selected,
        source_implementation_sha256=_implementation_sha256(), device=str(device),
        all_thread_affinities_verified=True, thread_affinities=thread_affinities))
    if _implementation_sha256() != plan['source_implementation_sha256']:
        raise ValueError('frozen implementation SHA differs from the dispatch plan')
    result = fit_models(plan, worker_id=worker_id, worker_count=worker_count, device=device)
    if result['errors']:
        raise RuntimeError('assigned null fits failed; retain the private fit receipts and block association')
    if fit_only:
        return result
    entries = _all_models(plan, worker_count)
    from .phewas_runtime.runtime import run_batched_configuration
    from .phewas_resources import HostMemoryGate
    repository = ModelRepository(entries)
    writer = StreamingCSVWriter(output/'csv', worker_id=str(worker_id))
    host_limit = plan.get('host_limit_gib_by_worker', {}).get(str(worker_id),
        plan.get('host_limit_gib', 200))
    gate = HostMemoryGate(output, host_limit_gib=host_limit,
                         reserve_gib=plan.get('host_reserve_gib', 20))
    config = dict(plan['configuration'])
    config['chromosomes'] = [entry for i, entry in enumerate(config['chromosomes']) if i % worker_count == worker_id]
    completed_jobs = output/f'worker_{worker_id}.completed_jobs.private.jsonl'
    if completed_jobs.exists():
        raise FileExistsError('association refuses to overwrite a previous job completion journal')
    def event(item):
        atomic_json(output/f'worker_{worker_id}.progress.anonymous.json',
                    dict(item, worker_id=worker_id, observed_at=time.time()))
        if item.get('event') == 'finished':
            # Append only after the job's staged rows have been accepted. This
            # proves the ordered schedule independently of aggregate counts.
            with completed_jobs.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(dict(item, worker_id=worker_id,
                    observed_at=time.time()), allow_nan=False)+'\n')
    try:
        report = run_batched_configuration(config, model_repository=repository,
            reader_factory=portable_reader, writer=writer, device=device,
            trait_batch_size=plan.get('trait_batch_size', 32), memory_limit_gib=None, cpu_threads=4,
            emit=event, host_memory_lease_factory=gate.lease_factory)
        report.update(end_to_end_seconds=time.time()-started, model_repository=repository.summary(),
                      source_implementation_sha256=plan['source_implementation_sha256'],
                      worker_id=worker_id, worker_count=worker_count,
                      assigned_chromosomes=[c['name'] for c in config['chromosomes']],
                      completed_jobs_sha256=digest_file(completed_jobs))
        atomic_json(output/f'worker_{worker_id}.report.private.json', report)
        return report
    except Exception as error:
        atomic_json(output/f'worker_{worker_id}.failed.private.json', dict(error_type=type(error).__name__,
            error=str(error), traceback=traceback.format_exc(), end_to_end_seconds=time.time()-started))
        raise


def execute_worker(plan, *, worker_id=0, worker_count=1, device='cuda:0', fit_only=False):
    """Publish an exit receipt even when preparation fails before the barrier."""
    started, code = time.time(), 1
    output = Path(plan['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = _execute_worker(plan, worker_id=worker_id, worker_count=worker_count,
                                 device=device, fit_only=fit_only)
        code = 0
        return result
    except BaseException as error:
        atomic_json(output/f'worker_{worker_id}.failed.private.json',
                    dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        atomic_json(output/f'worker_{worker_id}.exit.anonymous.json',
                    dict(worker_id=worker_id, exit_code=code, elapsed_seconds=time.time()-started))



def prepare_phewas_run(phenotype_csv, covariate_csv, cache_directory, *, output_directory,
                       chromosomes=None, trait_batch_size=32, continuous_transform='paper'):
    """Prepare one complete multi-trait plan from the same three scientific inputs.

    The cache_dataset.json ``phewas`` sidecar declares a relative ``kinship_npz``
    and, optionally, ``cohort_rows`` and ``covariate_profiles``. Without an
    explicit ID-bound kinship the mixed workflow refuses to substitute a GLM.
    Phenotype/covariate CSVs start with eid and contain numerical values or NA.
    Each trait keeps its own complete cases and covariate profile. Complete
    outputs are private CSV.gz shards plus a trait dictionary and receipts.
    """
    from .phewas_cache import _cache_path
    if continuous_transform not in ('paper', 'none', 'rint'):
        raise ValueError('continuous_transform must be paper, none or rint')
    root, manifest_path, manifest, entries, sample_ids = _dataset(cache_directory, chromosomes)
    sidecars = manifest.get('phewas', {})
    if 'kinship_npz' not in sidecars:
        raise ValueError('portable cache must declare its ID-bound phewas.kinship_npz sidecar')
    kinship = _cache_path(root, sidecars['kinship_npz'])
    cohort = _cache_path(root, sidecars['cohort_rows']) if 'cohort_rows' in sidecars else None
    profiles = (_cache_path(root, sidecars['covariate_profiles']) if 'covariate_profiles' in sidecars
                else Path(covariate_csv).with_suffix('.profiles.csv'))
    if 'covariate_profiles' in sidecars and not profiles.is_file():
        raise FileNotFoundError('declared covariate profile is missing')
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = output/'inputs'
    if not prepared.exists():
        prepare_phewas_inputs(phenotypes=phenotype_csv, covariates=covariate_csv,
            profiles=profiles if profiles.exists() else None, sample_ids=root/manifest.get('sample_ids', 'sample_ids.npy'), output=prepared)
    inputs = PhewasInputs(prepared)
    for name, path in (('phenotypes', phenotype_csv), ('covariates', covariate_csv)):
        binding = inputs.manifest['sources'][name]
        if digest_file(path) != binding['sha256']:
            raise ValueError('existing prepared matrix is bound to another input CSV')
    recorded_profile = inputs.manifest['sources'].get('profiles')
    if bool(recorded_profile) != profiles.is_file():
        raise ValueError('prepared covariate profile presence differs from the requested inputs')
    if profiles.is_file() and recorded_profile.get('sha256') != digest_file(profiles):
        raise ValueError('existing prepared matrix is bound to another covariate profile')
    if not np.array_equal(inputs.sample_ids, sample_ids):
        raise ValueError('existing prepared matrix has another cache sample axis')
    count = inputs.manifest['trait_count']
    inputs.close()
    config = build_configuration(cache_directory, chromosomes=chromosomes)
    plan = dict(schema_version=1, output_directory=str(output), prepared_inputs_directory=str(prepared),
        phenotype_csv=str(Path(phenotype_csv).resolve()), covariate_csv=str(Path(covariate_csv).resolve()),
        cache_directory=str(root), kinship_npz=str(kinship), kinship_sha256=digest_file(kinship),
        cohort_rows=None if cohort is None else str(cohort),
        cohort_sha256=None if cohort is None else digest_file(cohort), trait_count=count,
        continuous_transform=continuous_transform,
        trait_batch_size=trait_batch_size, source_implementation_sha256=_implementation_sha256(),
        prepared_inputs_manifest_sha256=digest_file(prepared/'manifest.private.json'),
        configuration=config, strict_logp_delta_required=False,
        comparison_significance_p=1e-5, output_all_pvalues=True)
    atomic_json(output/'plan.private.json', plan)
    return plan


def run_phewas(phenotype_csv, covariate_csv, cache_directory, *, output_directory,
               gpu_ids=(0,), chromosomes=None, trait_batch_size=32, continuous_transform='paper'):
    """One call for all traits and chromosomes on the selected local GPUs.

    Remote deployments use the identical private plan on each host. Genetic
    schedules are partitioned across workers; traits never trigger new complete
    genome scans. All workers use exactly eight assigned CPU affinity cores.
    """
    import multiprocessing as mp
    gpu_ids = tuple(int(value) for value in gpu_ids)
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('gpu_ids must contain distinct selected devices')
    plan = prepare_phewas_run(phenotype_csv, covariate_csv, cache_directory,
        output_directory=output_directory, chromosomes=chromosomes, trait_batch_size=trait_batch_size,
        continuous_transform=continuous_transform)
    plan['cpu_offsets'] = {str(i):8*i for i in range(len(gpu_ids))}
    atomic_json(Path(plan['output_directory'])/'plan.private.json', plan)
    context = mp.get_context('spawn')
    workers = [context.Process(target=execute_worker, kwargs=dict(plan=plan, worker_id=i,
                   worker_count=len(gpu_ids), device=f'cuda:{gpu}')) for i, gpu in enumerate(gpu_ids)]
    started_workers = []
    try:
        for worker in workers:
            worker.start()
            started_workers.append(worker)
    except BaseException:
        for worker in started_workers:
            if worker.is_alive():
                worker.terminate()
        for worker in started_workers:
            worker.join(timeout=10)
        raise
    while any(worker.is_alive() for worker in workers):
        failed = [(i, worker.exitcode) for i,worker in enumerate(workers) if worker.exitcode not in (None, 0)]
        if failed:
            # Children waiting at the barrier observe this even for SIGKILL.
            for i, code in failed:
                atomic_json(Path(plan['output_directory'])/f'worker_{i}.exit.anonymous.json',
                            dict(worker_id=i, exit_code=code, detected_by='owning_local_parent'))
            for worker in workers:
                worker.join(timeout=10)
            # These are this call's exact Process children. Preserve unfinished
            # shards, and bound cleanup after a peer cannot reach the barrier.
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            for worker in workers:
                worker.join(timeout=10)
            for worker in workers:
                if worker.is_alive():
                    worker.kill()
            for worker in workers:
                worker.join(timeout=5)
            raise RuntimeError('a PheWAS worker exited; inspect private exit receipts')
        for worker in workers:
            worker.join(timeout=.2)
    if any(worker.exitcode != 0 for worker in workers):
        raise RuntimeError('a PheWAS worker failed; inspect private failure receipts')
    return [json.loads((Path(plan['output_directory'])/f'worker_{i}.report.private.json').read_text())
            for i in range(len(workers))]

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', help='Private prepared deployment plan; no public data')
    parser.add_argument('--phenotype-csv')
    parser.add_argument('--covariate-csv')
    parser.add_argument('--cache-directory')
    parser.add_argument('--output-directory')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0])
    parser.add_argument('--worker-id', type=int, default=0)
    parser.add_argument('--worker-count', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fit-only', action='store_true')
    parser.add_argument('--continuous-transform', choices=['paper','none','rint'], default='paper')
    args = parser.parse_args(argv)
    if args.plan is None:
        if any(value is None for value in (args.phenotype_csv, args.covariate_csv, args.cache_directory, args.output_directory)):
            parser.error('provide phenotype CSV, covariate CSV, cache directory and output directory')
        return run_phewas(args.phenotype_csv, args.covariate_csv, args.cache_directory,
                          output_directory=args.output_directory, gpu_ids=args.gpu_ids,
                          continuous_transform=args.continuous_transform)
    plan = json.loads(Path(args.plan).read_text())
    execute_worker(plan, worker_id=args.worker_id, worker_count=args.worker_count,
                   device=args.device, fit_only=args.fit_only)


if __name__ == '__main__':
    main()
