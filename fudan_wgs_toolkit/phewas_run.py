"""One-submission, cache-only multi-phenotype WGS execution.

Phenotype CSV + covariate CSV + portable disk cache are the scientific inputs.
The portable dataset declares its ID-bound GRM/cohort sidecars and gene catalogs.
Deployment plans and participant/trait identifiers stay in the private output.
"""
from __future__ import annotations
import argparse
import ast
import copy
from collections.abc import Mapping
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from .phewas_inputs import PhewasInputs, prepare_phewas_inputs
from .phewas_models import SparseKinshipData, prepare_phewas_model
from ._fit_structure_cache import FitStructureCache
from .phewas_storage import (ModelRepository, StreamingCSVWriter, atomic_json, digest_file,
                             save_model_store, trait_index_sha256, verify_file, write_trait_index)
from .run import _implementation_sha256
from .phewas_cache import _dataset, _analysis_defaults, _stream_bindings, _verify_stream_bindings


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


def _existing_prepared_input_components(plan, proof):
    """Permit changed preparation code only for certified, unmodified inputs.

    Consumption has its own exact AST proof. This narrower route cannot fit a
    new phenotype or reinterpret a prepared artifact under changed CSV rules.
    The original plan, input manifest and all saved models are still verified
    by the common reuse path below.
    """
    if 'prepared_input_reuse_scope' not in proof:
        return {'phewas_inputs.py:complete_module'}
    scope = proof['prepared_input_reuse_scope']
    bindings = dict(model_fit_source_implementation_sha256=plan.get(
        'model_fit_source_implementation_sha256', plan['source_implementation_sha256']),
        consumer_source_implementation_sha256=plan['source_implementation_sha256'],
        prepared_inputs_manifest_sha256=plan['prepared_inputs_manifest_sha256'],
        cohort_sha256=plan.get('cohort_sha256'), kinship_sha256=plan['kinship_sha256'],
        continuous_transform=plan.get('continuous_transform', 'paper'))
    if (not isinstance(scope, dict) or scope.get('kind') != 'fixed_existing_prepared_inputs'
            or plan.get('reuse_models_only') is not True
            or not set(bindings) <= scope.keys()
            or any(scope.get(key) != value for key, value in bindings.items())
            or not _hash(scope.get('prepared_inputs_manifest_sha256'))
            or any(value is not None and not _hash(value)
                   for value in (scope.get('cohort_sha256'), scope.get('kinship_sha256')))
            or any(scope.get(key) is not True for key in ('input_preparation_disabled',
                'consumption_components_identical', 'complete_input_preparation_module_changed'))
            or type(scope.get('models_refitted')) is not int or scope['models_refitted'] != 0):
        raise ValueError('prepared-input reuse scope does not bind these existing scientific inputs')
    change = proof.get('inputs_module_change')
    if (not isinstance(change, dict) or change.get('identical') is not False
            or any(not _hash(change.get(key)) for key in
                   ('old_fingerprint', 'new_fingerprint', 'old_sha256', 'new_sha256'))
            or change['old_fingerprint'] == change['new_fingerprint']
            or change['old_sha256'] == change['new_sha256']
            or not isinstance(proof.get('components'), dict)
            or 'phewas_inputs.py:complete_module' in proof['components']):
        raise ValueError('prepared-input reuse must explicitly report changed preparation code')
    return {'phewas_inputs.py:PhewasInputs', 'phewas_inputs.py:sha256',
            'phewas_inputs.py:_discard', 'phewas_inputs.py:consumer_imports'}


def _consumption_component_fingerprints():
    """Fingerprint actual stored-state consumers; fitting factories are excluded."""
    selections = {
        'phewas_storage.py': ('load_model_store', 'load_spa_source', 'verify_file', 'digest_file', 'consumer_imports'),
        'phewas_inputs.py': ('PhewasInputs', 'sha256', '_discard', 'consumer_imports'),
        'null_model.py': ('GaussianNullModel.dataclass_fields', 'GaussianNullModel.set_matmul_mode',
                          'KinshipSpectrum:consumption_except_from_sparse', 'consumer_imports'),
        'binary_null.py': ('BinaryNullModel.dataclass_fields', 'BinaryNullModel.set_matmul_mode',
            'consumer_imports', 'ImmutableFittedBinarySource', '_compact_fitted_arrays',
            'binary_prefitted_state', 'compact_binary_normal_from_fitted_arrays',
            'compact_spa_state_from_fitted_arrays'),
        'tensor_validation.py': ('complete_module',),
    }
    values = {}
    for filename, names in selections.items():
        tree = ast.parse((Path(__file__).parent/filename).read_text())
        for name in names:
            if name == 'complete_module':
                node = tree
            elif name == 'consumer_imports':
                node = ast.Module(body=[item for item in tree.body
                    if isinstance(item, (ast.Import, ast.ImportFrom))], type_ignores=[])
            else:
                definition = name.split(':', 1)[0].split('.', 1)[0]
                node = next(item for item in tree.body if getattr(item, 'name', None) == definition)
                if name.endswith(':consumption_except_from_sparse'):
                    node = copy.deepcopy(node)
                    node.body = [item for item in node.body
                        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) or item.name != 'from_sparse']
                elif name.endswith('.dataclass_fields'):
                    node = ast.Module(body=node.decorator_list+[item for item in node.body
                        if isinstance(item, ast.AnnAssign)], type_ignores=[])
                elif '.' in name:
                    node = next(item for item in node.body if getattr(item, 'name', None) == name.split('.', 1)[1])
            values[filename+':'+name] = hashlib.sha256(
                ast.dump(node, include_attributes=False).encode()).hexdigest()
    return values


def _stored_model_consumption_proof(plan, proof, fit_source, consumer):
    """Authorize fixed stored models without asserting unchanged fitting math."""
    bindings = dict(model_fit_source_implementation_sha256=fit_source,
        consumer_source_implementation_sha256=consumer,
        prepared_inputs_manifest_sha256=plan['prepared_inputs_manifest_sha256'],
        cohort_sha256=plan.get('cohort_sha256'), kinship_sha256=plan['kinship_sha256'],
        continuous_transform=plan.get('continuous_transform', 'paper'), model_count=plan['trait_count'])
    if (type(proof.get('schema_version')) is not int or proof['schema_version'] != 2
            or proof.get('kind') != 'immutable_fitted_model_consumption'
            or proof.get('complete') is not True
            or proof.get('source_unchanged_during_review') is not True
            or proof.get('compatible_null_fit_math') is not False
            or proof.get('fitting_implementation_equivalence_verified') is not False
            or proof.get('scientific_model_input_changes') is not False
            or any(proof.get(key) is not True for key in ('model_fitting_disabled',
                'input_preparation_disabled', 'kinship_spectrum_from_sparse_disabled'))
            or type(proof.get('models_refitted')) is not int or proof['models_refitted'] != 0
            or not set(bindings) <= proof.keys()
            or any(proof.get(key) != value for key, value in bindings.items())
            or any(key in proof for key in ('old_null_fit_component_fingerprint',
                'new_null_fit_component_fingerprint', 'fitted_math_or_parameter_changes'))):
        raise ValueError('stored-model consumption compatibility receipt does not authorize this reuse')
    actual = _consumption_component_fingerprints()
    components = proof.get('components')
    if not isinstance(components, dict) or set(components) != set(actual):
        raise ValueError('stored-model consumption compatibility requires every actual consumer component')
    joined = hashlib.sha256()
    for name, current in sorted(actual.items()):
        record = components[name]
        if (not isinstance(record, dict) or record.get('identical') is not True
                or record.get('old_fingerprint') != current or record.get('new_fingerprint') != current):
            raise ValueError('stored-model consumption component fingerprints differ from actual source')
        joined.update(name.encode()); joined.update(bytes.fromhex(current))
    if (proof.get('old_model_consumption_component_fingerprint') != joined.hexdigest()
            or proof.get('new_model_consumption_component_fingerprint') != joined.hexdigest()):
        raise ValueError('stored-model consumption aggregate fingerprint differs')
    qualification = proof.get('existing_models_reuse', {})
    if not isinstance(qualification, dict) or any(qualification.get(key) is not True for key in (
            'scientifically_compatible', 'original_model_metadata_and_array_hashes_must_be_preserved',
            'consumer_and_fit_source_binding_must_be_separate', 'missing_reuse_target_must_hard_fail',
            'no_new_fits_may_be_labelled_as_old_source')):
        raise ValueError('stored-model consumption compatibility lacks strict reuse qualification')


def _model_inventory_fingerprint(inventory):
    # The compatibility checksum is the sole exclusion: including it would
    # create a hash cycle between proof and inventory. All model files, stats,
    # original fit receipts and any inherited write provenance remain bound.
    payload = {key: value for key, value in inventory.items()
               if key != 'null_fit_compatibility_sha256'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _gaussian_consumption_component_fingerprints():
    """Unchanged consumers actually reached by selected Gaussian models.

    The changed prepared-input consumer is certified separately, on the fixed
    inputs. Binary and SPA factories are excluded only by the family/selection
    gates below, never by asserting that their implementations are identical.
    """
    return {name: value for name, value in _consumption_component_fingerprints().items()
            if not name.startswith('binary_null.py:')
            and name not in ('phewas_storage.py:load_spa_source', 'phewas_inputs.py:PhewasInputs')}


def _gaussian_stored_consumption_proof(plan, proof, fit_source, consumer):
    """Authorize a fixed Gaussian selection; leave schema 1/2 gates intact."""
    selected = plan.get('selected_trait_indices')
    bindings = dict(model_fit_source_implementation_sha256=fit_source,
        consumer_source_implementation_sha256=consumer,
        prepared_inputs_manifest_sha256=plan['prepared_inputs_manifest_sha256'],
        cohort_sha256=plan.get('cohort_sha256'), kinship_sha256=plan['kinship_sha256'],
        continuous_transform=plan.get('continuous_transform', 'paper'),
        model_count=plan['trait_count'], rm_nan=plan.get('rm_nan', False))
    if (type(proof.get('schema_version')) is not int or proof['schema_version'] != 3
            or proof.get('kind') != 'immutable_fitted_gaussian_consumption'
            or not isinstance(selected, list) or not selected
            or any(type(index) is not int or not 0 <= index < plan['trait_count'] for index in selected)
            or selected != sorted(set(selected)) or proof.get('selected_trait_indices') != selected
            or type(plan.get('analysis_trait_count')) is not int
            or plan['analysis_trait_count'] != len(selected)
            or type(proof.get('analysis_trait_count')) is not int
            or proof.get('analysis_trait_count') != len(selected)
            or type(plan.get('rm_nan', False)) is not bool or type(proof.get('rm_nan')) is not bool
            or plan.get('input_contract', 'mixed') != 'mixed' or plan.get('binary_only', False)
            or proof.get('complete') is not True or proof.get('source_unchanged_during_review') is not True
            or any(proof.get(key) is not False for key in ('compatible_null_fit_math',
                'fitting_implementation_equivalence_verified', 'scientific_model_input_changes'))
            or any(proof.get(key) is not True for key in ('model_fitting_disabled',
                'input_preparation_disabled', 'kinship_spectrum_from_sparse_disabled',
                'binary_consumption_disabled', 'spa_consumption_disabled'))
            or type(proof.get('models_refitted')) is not int or proof['models_refitted'] != 0
            or not set(bindings) <= proof.keys()
            or any(proof.get(key) != value for key, value in bindings.items())):
        raise ValueError('Gaussian stored-model consumption selection or scientific bindings differ')
    parent = _bound_json(proof['parent_consumption_receipt'], proof['parent_consumption_sha256'])
    if (parent.get('schema_version') != 2 or parent.get('kind') != 'immutable_fitted_model_consumption'
            or parent.get('complete') is not True
            or parent.get('model_fit_source_implementation_sha256') != fit_source
            or parent.get('consumer_source_implementation_sha256') != proof.get('previous_consumer_source_implementation_sha256')
            or any(parent.get(key) != value for key, value in bindings.items()
                   if key not in ('consumer_source_implementation_sha256', 'rm_nan'))):
        raise ValueError('Gaussian consumption parent proof/source/input binding differs')
    actual = _gaussian_consumption_component_fingerprints()
    components = proof.get('components')
    if not isinstance(components, dict) or set(components) != set(actual):
        raise ValueError('Gaussian consumption requires the complete actual Gaussian consumer closure')
    joined = hashlib.sha256()
    for name, value in sorted(actual.items()):
        record = components[name]; prior = parent.get('components', {}).get(name, {})
        if (not isinstance(record, dict) or record.get('identical') is not True
                or record.get('old_fingerprint') != value or record.get('new_fingerprint') != value
                or prior.get('identical') is not True or prior.get('new_fingerprint') != value
                or prior.get('old_fingerprint') != value):
            raise ValueError('Gaussian consumption unchanged component differs from parent or actual source')
        joined.update(name.encode()); joined.update(bytes.fromhex(value))
    if proof.get('gaussian_consumption_component_fingerprint') != joined.hexdigest():
        raise ValueError('Gaussian consumption aggregate fingerprint differs')
    change = proof.get('prepared_input_consumer_change', {})
    current_input = _consumption_component_fingerprints()['phewas_inputs.py:PhewasInputs']
    old_input = parent.get('components', {}).get('phewas_inputs.py:PhewasInputs', {}).get('new_fingerprint')
    if (not isinstance(change, dict) or change.get('identical') is not False
            or not _hash(old_input) or old_input == current_input
            or change.get('old_fingerprint') != old_input or change.get('new_fingerprint') != current_input):
        raise ValueError('Gaussian input consumer change must report the genuine different ASTs')
    semantic = _bound_json(proof['input_semantic_receipt'], proof['input_semantic_sha256'])
    required = dict(schema_version=1, kind='fixed_existing_gaussian_inputs_semantic_equality',
        old_consumer_source_implementation_sha256=proof['previous_consumer_source_implementation_sha256'],
        new_consumer_source_implementation_sha256=consumer,
        parent_consumption_sha256=proof['parent_consumption_sha256'],
        prepared_inputs_manifest_sha256=bindings['prepared_inputs_manifest_sha256'],
        cohort_sha256=bindings['cohort_sha256'], kinship_sha256=bindings['kinship_sha256'],
        continuous_transform=bindings['continuous_transform'], rm_nan=bindings['rm_nan'],
        old_input_consumer_fingerprint=old_input, new_input_consumer_fingerprint=current_input,
        new_input_module_sha256=hashlib.sha256((Path(__file__).parent/'phewas_inputs.py').read_bytes()).hexdigest(),
        selected_trait_indices=selected, compared_trait_count=len(selected),
        fields=['sample_indices', 'sample_ids', 'y_raw', 'covariates', 'covariate_names',
                'missing_policy', 'available_n', 'mother_n', 'imputed_n', 'imputation_mean',
                'recoded_nonbinary_n'],
        complete=True, exact_values_equal=True, ordered_sample_axes_equal=True,
        new_design_arrays_readonly=True, repeated_design_values_equal=True,
        model_fitting_disabled=True, input_preparation_disabled=True, models_refitted=0)
    if (any(semantic.get(key) != value for key, value in required.items())
            or type(semantic.get('schema_version')) is not int
            or type(semantic.get('compared_trait_count')) is not int
            or type(semantic.get('models_refitted')) is not int
            or not _hash(semantic.get('old_input_module_sha256'))
            or not isinstance(semantic.get('python_version'), list)
            or semantic['python_version'][:2] != [3, 12]
            or not isinstance(semantic.get('traits'), list)
            or any(not isinstance(item, dict) for item in semantic.get('traits', []))
            or [item.get('trait_index') for item in semantic['traits'] if isinstance(item, dict)] != selected
            or any(item.get('family') != 'gaussian' or item.get('exact_values_equal') is not True
                   or not _hash(item.get('values_sha256')) for item in semantic['traits'])):
        raise ValueError('Gaussian fixed-input semantic comparison is incomplete or unbound')
    root = Path(plan['prepared_inputs_directory'])
    manifest = _bound_json(root/'manifest.private.json', bindings['prepared_inputs_manifest_sha256'])
    artifacts = manifest.get('artifacts', {})
    if not isinstance(artifacts, dict) or any(not isinstance(name, str) or Path(name).name != name
                                              for name in artifacts):
        raise ValueError('Gaussian fixed-input artifact closure differs')
    files = [root/'manifest.private.json', *(root/name for name in artifacts)]
    for path_key, hash_key in (('cohort_rows', 'cohort_sha256'), ('kinship_npz', 'kinship_sha256')):
        path = plan.get(path_key)
        if (path is None) != (plan.get(hash_key) is None):
            raise ValueError('Gaussian fixed-input cohort/kinship paths differ')
        if path is not None:
            files.append(Path(path))
    identities = semantic.get('fixed_input_file_identities')
    actual_identities = {str(path.resolve()): _file_stat(path) for path in files}
    if not isinstance(identities, dict) or identities != actual_identities:
        raise ValueError('Gaussian fixed-input identities changed after actual semantic comparison')
    qualification = proof.get('existing_models_reuse', {})
    if not isinstance(qualification, dict) or any(qualification.get(key) is not True for key in (
            'scientifically_compatible', 'original_model_metadata_and_array_hashes_must_be_preserved',
            'consumer_and_fit_source_binding_must_be_separate', 'missing_reuse_target_must_hard_fail',
            'no_new_fits_may_be_labelled_as_old_source')):
        raise ValueError('Gaussian stored consumption lacks strict immutable reuse qualification')


def _stored_model_consumption_schema(plan, proof, context):
    """Check complete serialized field geometry without fitting or array copies."""
    if proof.get('immutable_inventory_fingerprint') != _model_inventory_fingerprint(context['inventory']):
        raise ValueError('stored-model consumption immutable inventory fingerprint differs')
    contract = proof.get('serialized_model_contract')
    if (not isinstance(contract, dict) or type(contract.get('schema_version')) is not int
            or contract['schema_version'] != 1 or contract.get('complete') is not True
            or contract.get('geometry_verified') is not True
            or type(contract.get('model_count')) is not int or contract['model_count'] != plan['trait_count']
            or not isinstance(contract.get('models'), list)
            or any(not isinstance(item, dict) for item in contract['models'])
            or [item.get('trait_index') for item in contract['models']] != list(range(plan['trait_count']))):
        raise ValueError('stored-model consumption serialized model schema coverage differs')
    for item in contract['models']:
        index = item['trait_index']
        _verify_reused_model(plan, context, index)
        record = context['models'][index]; root = Path(record['target']); n = record['n']
        metadata = _bound_json(root/'model.private.json', record['metadata_sha256'])
        expected_rows = dict(shape=[n], dtype='<i8', sha256=metadata['sample_rows_sha256'])
        labels = ('normal', 'spa') if record['family'] == 'binomial' else ('normal',)
        if (type(index) is not int or item.get('metadata_sha256') != record['metadata_sha256']
                or item.get('sample_rows') != expected_rows
                or not isinstance(item.get('states'), dict) or set(item['states']) != set(labels)):
            raise ValueError('stored-model consumption serialized model schema binding differs')
        _stored_array_header(root/'sample_rows.npy', expected_rows)
        for label in labels:
            state = _bound_json(root/label/'state.private.json', metadata[
                'state_sha256' if label == 'normal' else 'spa_state_sha256'])
            if item['states'][label] != state:
                raise ValueError('stored-model consumption complete serialized state differs')
            _stored_state_geometry(root/label, state, n, record['family'], label)


def _stored_array_header(path, spec):
    before = _file_stat(path)
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        if (list(value.shape) != spec['shape'] or value.dtype.str != spec['dtype']
                or value.dtype.hasobject or _file_stat(path) != before):
            raise ValueError('stored-model consumption array header geometry differs')
    finally:
        mapping = getattr(value, '_mmap', None)
        if mapping is not None:
            mapping.close()


def _stored_state_geometry(root, state, n, family, label):
    arrays = state.get('arrays')
    if (not isinstance(arrays, dict) or not arrays
            or any(not isinstance(spec, dict) for spec in arrays.values())
            or type(state.get('iterations')) is not int or state['iterations'] < 0
            or state.get('use_spa') is not (label == 'spa')
            or not isinstance(state.get('fit_method'), str) or not state['fit_method']):
        raise ValueError('stored-model consumption serialized state lacks fitted fields')
    width = arrays.get('x', {}).get('shape')
    if (not isinstance(width, list) or len(width) != 2 or width[0] != n
            or type(width[1]) is not int or width[1] < 1):
        raise ValueError('stored-model consumption serialized design geometry differs')
    k = width[1]
    shapes = dict(sample_ids=[n], x=[n, k], scaled_residuals=[n], coefficients=[k],
                  fixed_effect_covariance=[k, k], precision_x=[n, k])
    if family == 'gaussian':
        shapes.update(theta=[2], precision_theta=[2], inverse_variance=[n], eigenvalues=[n])
        blocks = state.get('blocks')
        if (not isinstance(blocks, list) or any(type(i) is not int or i < 0 for i in blocks)
                or blocks != list(range(len(blocks)))):
            raise ValueError('stored-model consumption kinship block schema differs')
        for i in blocks:
            rows = arrays.get(f'block_{i}_rows', {}).get('shape')
            if (not isinstance(rows, list) or len(rows) != 1
                    or type(rows[0]) is not int or not 0 < rows[0] <= n):
                raise ValueError('stored-model consumption kinship block geometry differs')
            shapes[f'block_{i}_rows'] = rows
            shapes[f'block_{i}_rotation'] = [rows[0], rows[0]]
    else:
        shapes['fitted_probability'] = [n]
        if 'precision' in arrays:
            shapes['precision'] = [n]
        else:
            values = arrays.get('precision_values', {}).get('shape')
            if (not isinstance(values, list) or len(values) != 1
                    or type(values[0]) is not int or values[0] < 1
                    or state.get('precision_shape') != [n, n]):
                raise ValueError('stored-model consumption sparse precision geometry differs')
            shapes.update(precision_values=values, precision_indices=[2, values[0]])
        if label == 'spa':
            shapes.update(xw=[k, n], projection_left=[n, k], phenotype=[n], working_phenotype=[n])
    if set(arrays) != set(shapes):
        raise ValueError('stored-model consumption serialized state fields differ')
    dtype = '<f8' if label == 'spa' else '<f4'
    for name, shape in shapes.items():
        spec = arrays[name]
        expected_dtype = '<i8' if name.endswith('_rows') or name == 'precision_indices' else dtype
        try:
            valid_dtype = np.dtype(spec.get('dtype')).kind in 'US' if name == 'sample_ids' else spec.get('dtype') == expected_dtype
        except (TypeError, ValueError):
            valid_dtype = False
        if (not isinstance(spec, dict) or spec.get('shape') != shape or not valid_dtype
                or not _hash(spec.get('sha256')) or not isinstance(spec.get('file'), str)
                or spec['file'] != name+'.npy'):
            raise ValueError('stored-model consumption serialized field geometry or dtype differs')
        _stored_array_header(root/spec['file'], spec)


def _reuse_context(plan):
    """Validate separate original-fit and current-consumer provenance.

    A versioned source receipt binds fitting equivalence or fixed stored-state
    consumption. The inventory binds actual inputs, all original files and
    successful original fit processes. The receipts preserve model metadata.
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
    if proof.get('schema_version') == 3:
        _gaussian_stored_consumption_proof(plan, proof, fit_source, consumer)
    elif proof.get('schema_version') == 2:
        _stored_model_consumption_proof(plan, proof, fit_source, consumer)
    else:
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
            'rint.py:complete_module', 'numerics.py:complete_module',
            'tensor_validation.py:complete_module',
            'null_model.py:fit_gaussian_null', 'null_model.py:KinshipSpectrum',
            'null_model.py:rank_inverse_normal', 'null_model.py:_r_sum',
            'null_model.py:GaussianNullModel.dataclass_fields',
            'null_model.py:GaussianNullModel.set_matmul_mode_except_new_cache_clear', 'phewas_run.py:_design'}
        required.update(_existing_prepared_input_components(plan, proof))
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
    context = dict(fit_source=fit_source, inventory=inventory, models=by_index)
    if proof.get('schema_version') in (2, 3):
        _stored_model_consumption_schema(plan, proof, context)
        context['reuse_receipt_kind'] = proof['kind']
        if proof.get('schema_version') == 3:
            selected = proof['selected_trait_indices']
            if any(by_index[index]['family'] != 'gaussian' for index in selected):
                raise ValueError('Gaussian consumption selection contains a binary model')
            context['selected_trait_indices'] = selected
    return context


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
    result = dict(kinship_sha256=plan['kinship_sha256'],
                source_implementation_sha256=(plan['source_implementation_sha256'] if context is None
                                              else context['fit_source']),
                cohort_sha256=plan.get('cohort_sha256'),
                continuous_transform=plan.get('continuous_transform', 'paper'))
    if plan.get('binary_only'):
        result['binary_only'] = True
    return result


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


def _selected_analyses(analyses):
    if (not isinstance(analyses, (list, tuple)) or not analyses
            or any(kind not in ('individual', 'coding', 'noncoding', 'ncrna') for kind in analyses)
            or len(set(analyses)) != len(analyses)):
        raise ValueError('analyses must contain distinct individual, coding, noncoding or ncrna entries')
    return tuple(analyses)


def _selected_chromosomes(chromosomes):
    if chromosomes is None or chromosomes == 'all':
        return None
    if not isinstance(chromosomes, (list, tuple)) or not chromosomes:
        raise ValueError('chromosomes must be all or a nonempty list of chromosome numbers')
    selected = []
    for value in chromosomes:
        name = str(value).removeprefix('chr')
        if isinstance(value, bool) or not re.fullmatch(r'[1-9][0-9]*', name) or not 1 <= int(name) <= 22:
            raise ValueError('chromosome numbers must be from 1 to 22')
        selected.append(name)
    if len(set(selected)) != len(selected):
        raise ValueError('chromosomes must be distinct')
    return selected


def _validate_host_memory_limit(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError('host_memory_limit_gib must be None or a positive finite number')
    return value


def _host_memory_policy(plan):
    """One task policy for all same-node peers, with explicit public precedence."""
    if 'host_memory_limit_gib' in plan:
        limit = _validate_host_memory_limit(plan['host_memory_limit_gib'])
        return dict(host_limit_gib=limit, reserve_gib=plan.get('host_memory_reserve_gib'))
    # Older reviewed deployment plans use a shared host_limit_gib/reserve.
    # Legacy per-worker values cannot generate conflicting same-node ledgers:
    # retain the most restrictive explicit common policy for all workers.
    limits = list(plan.get('host_limit_gib_by_worker', {}).values())
    if 'host_limit_gib' in plan:
        limits.append(plan['host_limit_gib'])
    limits = [_validate_host_memory_limit(value) for value in limits]
    limit = min((value for value in limits if value is not None), default=None)
    return dict(host_limit_gib=limit, reserve_gib=plan.get('host_reserve_gib'))


def _close_association_resources(writer, repository, *, failed, gate=None):
    """Always close model mappings; abort only unfinished exceptional writers.

    An original worker error remains the raised error even if cleanup fails.
    On a successful association, a cleanup error is still an execution error.
    """
    first_error = None
    operations = ([writer.abort] if failed and writer is not None else [])
    if repository is not None:
        operations.append(repository.close)
    for operation in operations:
        try:
            operation()
        except BaseException as error:
            if first_error is None:
                first_error = error
    close_gate = getattr(gate,'close',None)
    if first_error is None and callable(close_gate):
        try:
            close_gate()
        except BaseException as error:
            first_error = error
    if first_error is not None and not failed:
        raise first_error


def _resource_options(memory_limit_gib, memory_reserve_gib, cpu_threads, trait_batch_size, host_memory_limit_gib=None):
    _validate_host_memory_limit(host_memory_limit_gib)
    if memory_limit_gib is not None and (isinstance(memory_limit_gib, bool)
            or not isinstance(memory_limit_gib, (int, float)) or not math.isfinite(memory_limit_gib)
            or memory_limit_gib <= 0):
        raise ValueError('memory_limit_gib must be None or a positive finite number')
    if (isinstance(memory_reserve_gib, bool) or not isinstance(memory_reserve_gib, (int, float))
            or not math.isfinite(memory_reserve_gib) or memory_reserve_gib < 0):
        raise ValueError('memory_reserve_gib must be a nonnegative finite number')
    if memory_limit_gib is not None and memory_reserve_gib >= memory_limit_gib:
        raise ValueError('memory_reserve_gib must be below the requested memory limit')
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError('cpu_threads must be a positive integer')
    if type(trait_batch_size) is not int or trait_batch_size < 1:
        raise ValueError('trait_batch_size must be a positive integer')



def _legacy_single_batch_configuration(configuration):
    """Canonicalize only the known absent/False legacy Single policy.

    Fresh legacy plans omit the new key, preserving their historical JSON
    shape. Resume callers compare this view but retain the previous original
    configuration and its checkpoint hash; no receipt is migrated. Every
    other key/value, and the explicit True policy, remains significant.
    """
    if not isinstance(configuration, dict):
        raise ValueError('association configuration must be a dictionary')
    value = configuration.get('single_cross_frame_batches', False)
    if type(value) is not bool:
        raise ValueError('single_cross_frame_batches must be a boolean')
    result = dict(configuration)
    if not value:
        result.pop('single_cross_frame_batches', None)
    return result

def build_configuration(cache_directory, *, chromosomes=None, maximum_single_variants=None,
                        analyses=('individual', 'coding', 'noncoding', 'ncrna'),
                        gene_variant_type='SNV', memory_limit_gib=None,
                        covariance_block_size=4096, long_mask_threshold=5000, long_mask_method="full",
                        long_mask_rank=512, seed=1729, single_mac_cutoff=20,
                        single_group_variants=5000, single_region_size=10_000_000,
                        individual_effective_block_size=1024, single_cross_frame_batches=False):
    """Build one chromosome/genetic schedule, independent of phenotype count."""
    if type(single_cross_frame_batches) is not bool:
        raise ValueError('single_cross_frame_batches must be a boolean')
    analyses = _selected_analyses(analyses)
    if long_mask_method not in ('full', 'fastskat', 'liu'):
        raise ValueError('long_mask_method must be full, fastskat or liu')
    if gene_variant_type not in ('SNV', 'Indel', 'variant'):
        raise ValueError('gene_variant_type must be SNV, Indel or variant')
    for key, value in dict(covariance_block_size=covariance_block_size,
            long_mask_threshold=long_mask_threshold, long_mask_rank=long_mask_rank,
            single_mac_cutoff=single_mac_cutoff, single_group_variants=single_group_variants,
            single_region_size=single_region_size,
            individual_effective_block_size=individual_effective_block_size).items():
        if type(value) is not int or value < 1:
            raise ValueError(key+' must be a positive integer')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    chromosomes = _selected_chromosomes(chromosomes)
    root, manifest_path, manifest, entries, ids = _dataset(cache_directory, chromosomes)
    from .phewas_cache import _cache_path
    catalog = manifest.get('annotation_catalog') or {}
    if isinstance(catalog, str):
        catalog = str(_cache_path(root, catalog))
    elif not isinstance(catalog, dict):
        raise ValueError('annotation_catalog must be a field mapping or relative JSON path')
    schedules = []
    for entry in entries:
        jobs = []
        if 'individual' in analyses:
            maximum = maximum_single_variants if maximum_single_variants is not None else entry.get('_maximum_position')
            if maximum is None:
                from .cache_runtime.portable import PortableMetadataReader
                with PortableMetadataReader(entry['metadata_directory'], entry['container_directory'], legacy_numeric=True) as metadata:
                    maximum = int(np.asarray(metadata.read_field('position'), dtype=np.int64).max(initial=0))
            for start in range(1, int(maximum)+1, single_region_size):
                jobs.append(dict(kind='individual', arguments=dict(start=start,
                    end=min(start+single_region_size-1, int(maximum)),
                    mac_cutoff=single_mac_cutoff, subset_variants_num=single_group_variants,
                    variant_type='variant')))
        genes = (json.loads(Path(entry['gene_catalog']).read_text())
                 if set(analyses) - {'individual'} else [])
        if isinstance(genes, dict):
            genes = genes.get('jobs', genes.get('genes'))
        for gene in genes:
            kind = gene.get('kind')
            if kind not in analyses or kind not in ('coding', 'noncoding', 'ncrna'):
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
        jobs = [job for job in jobs if job['kind'] != 'individual'] + [job for job in jobs if job['kind'] == 'individual']
        for index, job in enumerate(jobs):
            job['checkpoint_job_index'] = index
        schedules.append(dict(name=entry['name'], gds=entry['container_directory'],
            container_directory=entry['container_directory'], metadata_directory=entry['metadata_directory'],
            annotation_index=dict(promoter_intervals_file=entry.get('promoter_intervals')),
            jobs=jobs))
    options = _analysis_defaults(memory_limit_gib, covariance_block_size, long_mask_threshold, long_mask_rank, seed,
                                long_mask_method=long_mask_method)
    options.update(memory_limit_gib=memory_limit_gib, variant_type=gene_variant_type)
    return _legacy_single_batch_configuration(dict(chromosomes=schedules, qc_path=manifest.get('qc_path', 'annotation/info/QC_label'),
        annotation_catalog=catalog,
        annotation_names=manifest.get('annotation_names'),
        matmul_mode='tf32', analysis_options=options, resident_genotypes=True,
        local_mask_reuse=True, statistics_tail_optimization=True, weight_batch_optimization=True,
        individual_effective_block_size=individual_effective_block_size, individual_genotype_block_size=1024,
        single_cross_frame_batches=single_cross_frame_batches,
        stage_profile=True, source_implementation_sha256=_implementation_sha256()))


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
        settings = reference if isinstance(reference, dict) else (spec or {})
        processes = settings.get('frame_prefetch_processes', 0)
        if type(processes) is not int or not 0 <= processes <= 32:
            raise ValueError('frame_prefetch_processes must be in 0..32')
        if processes:
            container.enable_process_prefetch(processes,
                settings.get('frame_prefetch_memory_bytes', 2*2**30),
                settings.get('frame_prefetch_directory'))
        else:
            container.enable_prefetch(settings.get('frame_prefetch_bytes', 64*2**20))
        if not np.array_equal(container.samples, metadata._array(metadata.manifest['source_sample_rows'])):
            raise ValueError('cache source sample rows differ from portable metadata')
        yield dict(metadata=metadata, container=container, portable_axis=True)
    finally:
        primary = sys.exc_info()[1]
        cleanup_error = None
        for resource in (metadata, container):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
                if primary is not None and hasattr(primary, 'add_note'):
                    primary.add_note('portable reader cleanup also failed: '+type(error).__name__)
        if primary is None and cleanup_error is not None:
            raise cleanup_error


def _verify_stored_model(target, metadata, *, cache_sample_ids=None):
    """Verify immutable saved arrays on CPU without materializing model tensors.

    Completed association tiles cannot hide a missing/corrupt fitted state.
    Checksums include both normal and binary SPA arrays; mmap headers retain
    shape/dtype and the stored sample rows must address the original cache axis.
    """
    target = Path(target).resolve()
    count = 0
    def checked_array(path, expected, shape=None, dtype=None):
        nonlocal count
        verify_file(path, expected)
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        try:
            if ((shape is not None and list(array.shape) != list(shape))
                    or (dtype is not None and array.dtype.str != dtype)):
                raise ValueError('stored null-model array shape/dtype differs')
            value = np.asarray(array).copy() if path.name in ('sample_rows.npy', 'sample_ids.npy') else None
            count += 1
            return value
        finally:
            mapping = getattr(array, '_mmap', None)
            if mapping is not None:
                mapping.close()
    n = metadata.get('n')
    if type(n) is not int or n < 1 or metadata.get('family') not in ('gaussian', 'binomial'):
        raise ValueError('stored null-model sample count or family differs')
    rows = checked_array(target/'sample_rows.npy', metadata['sample_rows_sha256'], shape=[n], dtype='<i8')
    if (rows.ndim != 1 or np.any(rows < 0) or len(np.unique(rows)) != n
            or cache_sample_ids is not None and np.any(rows >= len(cache_sample_ids))):
        raise ValueError('stored null-model sample rows differ from the cache axis')
    for label, digest_key in (('normal', 'state_sha256'), ('spa', 'spa_state_sha256')):
        state_name = metadata.get('normal_state' if label == 'normal' else 'spa_state')
        if state_name is None:
            if label == 'normal' or metadata['family'] == 'binomial':
                raise ValueError('stored null model lacks required normal/FP64 SPA state')
            continue
        state_root = (target/state_name).resolve()
        if not state_root.is_relative_to(target):
            raise ValueError('stored null-model state must remain inside its model directory')
        state_path = state_root/'state.private.json'
        verify_file(state_path, metadata[digest_key])
        state = json.loads(state_path.read_text())
        if state.get('family') != metadata['family'] or not isinstance(state.get('arrays'), dict):
            raise ValueError('stored null-model state family/array descriptors differ')
        sample_ids = None
        for name, record in state['arrays'].items():
            path = (state_root/record['file']).resolve()
            if not path.is_relative_to(state_root):
                raise ValueError('stored null-model array must remain inside its state directory')
            values = checked_array(path, record['sha256'], shape=record['shape'], dtype=record['dtype'])
            if name == 'sample_ids':
                sample_ids = values
        if (sample_ids is None or sample_ids.shape != (n,)
                or cache_sample_ids is not None and not np.array_equal(
                    sample_ids.astype(str), np.asarray(cache_sample_ids)[rows].astype(str))):
            raise ValueError('stored null-model IDs differ from their cache sample rows')
        for name in ('x', 'scaled_residuals', 'precision_x'):
            if name not in state['arrays'] or state['arrays'][name]['shape'][0] != n:
                raise ValueError('stored null-model fitted sample geometry differs')
    return dict(model_array_files_verified=count, model_arrays_materialized=False)


def fit_models(plan, *, worker_id=0, worker_count=1, device='cuda:0', trait_indices=None):
    """Fit every assigned phenotype once; publish immutable model receipts."""
    context = _reuse_context(plan)
    fit_binding = _fit_binding(plan, _context=context)
    output = Path(plan['output_directory'])
    models = output/'models'
    models.mkdir(exist_ok=True)
    inputs = (PhewasInputs(plan['prepared_inputs_directory'], verify_arrays=False)
              if context is not None and context.get('reuse_receipt_kind') == 'immutable_fitted_gaussian_consumption'
              else PhewasInputs(plan['prepared_inputs_directory']))
    if plan.get('rm_nan', False) != inputs.manifest.get('rm_nan', False):
        inputs.close()
        raise ValueError('prepared phenotype missing policy differs from the immutable plan')
    if plan.get('binary_only',False) != inputs.manifest.get('binary_only',False):
        inputs.close()
        raise ValueError('prepared phenotype binary-only policy differs from the immutable plan')
    input_sha = digest_file(inputs.directory/'manifest.private.json')
    if input_sha != plan.get('prepared_inputs_manifest_sha256'):
        inputs.close()
        raise ValueError('prepared input manifest differs from the immutable dispatch plan')
    contract=plan.get('input_contract','mixed')
    if (contract not in ('mixed','continuous')
            or inputs.manifest.get('trait_options',{}).get('input_contract','mixed')!=contract):
        inputs.close()
        raise ValueError('prepared input contract differs from the immutable dispatch plan')
    if contract=='continuous':
        from .continuous_run import validate_continuous_metadata
        try:
            validate_continuous_metadata(inputs.metadata)
        except BaseException:
            inputs.close()
            raise
    cohort_sha = digest_file(plan['cohort_rows']) if plan.get('cohort_rows') else None
    kinship_sha = digest_file(plan['kinship_npz']) if plan.get('kinship_npz') else None
    if cohort_sha != plan.get('cohort_sha256') or kinship_sha != plan['kinship_sha256']:
        raise ValueError('GRM/cohort checksum differs from the immutable dispatch plan')
    # Reuse mode verifies saved models only. It never enters a fitting path or
    # loads a fresh GRM object that could later be used to label a new fit old.
    kinship = (SparseKinshipData.load_npz(plan['kinship_npz'])
               if context is None and plan.get('kinship_npz') else None)
    cohort = None if context is not None else (np.load(plan['cohort_rows'], allow_pickle=False)
                                              if plan.get('cohort_rows') else None)
    selected = None if context is None else context.get('selected_trait_indices')
    expected_assignment = (list(range(worker_id, inputs.manifest['trait_count'], worker_count))
                           if selected is None else selected[worker_id::worker_count])
    assigned = expected_assignment if trait_indices is None else list(trait_indices)
    if selected is not None and assigned != expected_assignment:
        inputs.close()
        raise ValueError('Gaussian reuse verification must retain its exact selected worker assignment')
    started, successes, errors = time.time(), [], []
    structures = FitStructureCache() if context is None else None
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
                verification = _verify_stored_model(target, meta, cache_sample_ids=inputs.sample_ids)
                successes.append(dict(trait_index=index, path=str(target), family=meta['family'],
                                      n=meta['n'], sha256=digest_file(target/'model.private.json'), **verification))
                continue
            before = time.perf_counter()
            try:
                trait = inputs.trait(int(index), cohort_indices=cohort)
                with structures.acquire_design(trait['sample_ids'], trait['covariates'],
                        trait['covariate_names'], _design) as (x, design):
                    fitted = prepare_phewas_model(trait['y_raw'], trait['sample_ids'], x,
                        family=trait['family'], kinship=kinship, continuous_transform=plan.get('continuous_transform', 'paper'),
                        device=device, association_mode='tf32', use_spa=True,
                        fit_structure_cache=structures)
                if not fitted.model.converged:
                    raise ArithmeticError('null fit did not converge')
                rows = trait['sample_indices'][fitted.sample_indices]
                if digest_file(inputs.directory/'manifest.private.json') != input_sha:
                    raise ValueError('prepared input manifest changed during null fitting')
                metadata = dict(fitted.metadata, **design, trait_index=index, phenotype=trait['name'],
                    profile=trait['profile'], input_manifest_sha256=input_sha,
                    rm_nan=trait.get('rm_nan', False), missing_policy=trait.get('missing_policy', 'available'),
                    available_n=trait.get('available_n', len(trait['y_raw'])),
                    mother_n=trait.get('mother_n', len(trait['y_raw'])),
                    imputed_n=trait.get('imputed_n', 0), imputation_mean=trait.get('imputation_mean'),
                    **fit_binding,
                    wall_seconds=time.perf_counter()-before)
                for key in ('analysis_cases','analysis_controls','available_cases','available_controls','imputed_controls'):
                    if key in trait:
                        metadata[key] = trait[key]
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
                      model_import_sha256=plan.get('model_import_sha256'),
                      fit_structure_cache=None if structures is None else structures.stats())
        if callable(getattr(inputs,'design_cache_summary',None)):
            result['prepared_design_cache'] = inputs.design_cache_summary()
        if context is not None and context.get('reuse_receipt_kind') in (
                'immutable_fitted_model_consumption', 'immutable_fitted_gaussian_consumption'):
            result.update(reuse_receipt_kind=context['reuse_receipt_kind'], compatible_null_fit_math=False,
                fitting_implementation_equivalence_verified=False, model_fitting_disabled=True,
                input_preparation_disabled=True, models_refitted=0)
        atomic_json(output/f'fit_worker_{worker_id}.complete.private.json', result)
        return result
    finally:
        if structures is not None:
            structures.close()
        inputs.close()


def _association_inputs(plan):
    """Validate completed prepared inputs without rereading their large arrays.

    The immutable manifest, trait metadata and mmap headers establish the
    consumption contract. Array checksums remain checked when the association
    storage loader actually materializes a fitted state.
    """
    if _implementation_sha256() != plan['source_implementation_sha256']:
        raise ValueError('association-only implementation binding differs from the plan')
    root = Path(plan['prepared_inputs_directory'])
    manifest = _bound_json(root/'manifest.private.json', plan['prepared_inputs_manifest_sha256'])
    if (manifest.get('status') != 'complete'
            or type(plan.get('trait_count')) is not int or plan['trait_count'] < 1
            or manifest.get('trait_count') != plan['trait_count']
            or type(manifest.get('sample_count')) is not int or manifest['sample_count'] < 1
            or type(plan.get('rm_nan', False)) is not bool
            or manifest.get('rm_nan', False) != plan.get('rm_nan', False)
            or manifest.get('binary_only',False)!=plan.get('binary_only',False)):
        raise ValueError('association-only prepared input count/status/missing policy differs')
    inputs = PhewasInputs(root, verify_arrays=False)
    try:
        contract=plan.get('input_contract','mixed')
        if (contract not in ('mixed','continuous')
                or manifest.get('trait_options',{}).get('input_contract','mixed')!=contract):
            raise ValueError('association-only prepared input contract differs from the plan')
        if contract=='continuous':
            from .continuous_run import validate_continuous_metadata
            validate_continuous_metadata(inputs.metadata)
        metadata = _bound_json(root/'traits.private.json',
                               manifest['artifacts']['traits.private.json']['sha256'])
        traits = metadata.get('traits', [])
        if (len(traits) != plan['trait_count']
                or [trait.get('index') for trait in traits] != list(range(plan['trait_count']))
                or any(trait.get('family') not in ('gaussian', 'binomial') for trait in traits)
                or (plan.get('binary_only') and any(trait.get('family')!='binomial' for trait in traits))):
            raise ValueError('association-only prepared trait metadata coverage differs')
        if 'trait_index_sha256' in plan:
            names = [trait['name'] for trait in traits]
            families = {trait['name']: trait['family'] for trait in traits}
            if trait_index_sha256(names, families) != plan['trait_index_sha256']:
                raise ValueError('association-only prepared trait dictionary binding differs')
        for path_key, hash_key in (('kinship_npz', 'kinship_sha256'), ('cohort_rows', 'cohort_sha256')):
            path = plan.get(path_key)
            expected = plan.get(hash_key)
            if (path is None) != (expected is None) or (path is not None and digest_file(path) != expected):
                raise ValueError('association-only GRM/cohort checksum differs from the plan')
        return dict(sample_count=manifest['sample_count'], traits=traits,
                    missing_policy=manifest.get('missing_policy'))
    finally:
        inputs.close()


def _association_model_metadata(plan, entry, metadata, prepared):
    """Check bound normal/SPA descriptors; defer large NPY hashes to loading."""
    index = entry['trait_index']
    trait = prepared['traits'][index]
    n = metadata.get('n')
    if (type(n) is not int or not 0 < n <= prepared['sample_count']
            or entry.get('n') != n or entry.get('family') != trait['family']
            or metadata.get('family') != trait['family']
            or metadata['fit'].get('trait_index') != index
            or metadata['fit'].get('phenotype') != trait['name']
            or metadata['fit'].get('profile') != trait['profile']
            or metadata['fit'].get('rm_nan', False) != plan.get('rm_nan', False)
            or metadata['fit'].get('binary_only',False)!=plan.get('binary_only',False)
            or (prepared['missing_policy'] is not None and 'missing_policy' in metadata['fit']
                and metadata['fit']['missing_policy'] != prepared['missing_policy'])
            or metadata.get('normal_state') != 'normal'
            or metadata.get('spa_state') != ('spa' if trait['family'] == 'binomial' else None)
            or not _hash(metadata.get('sample_rows_sha256'))):
        raise ValueError('association-only fitted model trait/sample/policy metadata differs')
    target = Path(entry['path']).resolve()
    if not (target/'sample_rows.npy').is_file():
        raise ValueError('association-only fitted model sample axis is missing')
    for label, digest_key in (('normal', 'state_sha256'), ('spa', 'spa_state_sha256')):
        if label == 'spa' and trait['family'] != 'binomial':
            continue
        state_root = target/label
        if not state_root.resolve().is_relative_to(target):
            raise ValueError('association-only fitted state escaped its model directory')
        state = _bound_json(state_root/'state.private.json', metadata.get(digest_key))
        arrays = state.get('arrays')
        if (state.get('family') != trait['family'] or state.get('converged') is not True
                or state.get('has_kinship') is not (plan.get('kinship_npz') is not None)
                or state.get('matmul_mode') != ('fp64' if label == 'spa' else 'tf32')
                or not isinstance(arrays, dict)
                or not {'sample_ids', 'x', 'scaled_residuals', 'precision_x'} <= arrays.keys()):
            raise ValueError('association-only normal/SPA state metadata differs')
        for name, record in arrays.items():
            filename = record.get('file')
            shape = record.get('shape')
            if (not isinstance(filename, str) or Path(filename).name != filename
                    or not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape)
                    or not isinstance(record.get('dtype'), str) or not _hash(record.get('sha256'))
                    or not (state_root/filename).is_file()
                    or not (state_root/filename).resolve().is_relative_to(state_root.resolve())):
                raise ValueError('association-only fitted array descriptor/path differs')
            if name in ('sample_ids', 'x', 'scaled_residuals', 'precision_x') and (not shape or shape[0] != n):
                raise ValueError('association-only fitted state sample geometry differs')
            try:
                if np.dtype(record['dtype']).hasobject:
                    raise ValueError('object arrays are not fitted numerical state')
            except (TypeError, ValueError) as error:
                raise ValueError('association-only fitted array dtype differs') from error
        if arrays['sample_ids']['shape'] != [n] or len(arrays['x']['shape']) != 2:
            raise ValueError('association-only fitted state sample/design geometry differs')


def _all_models(plan, worker_count, *, association_only=False):
    if type(association_only) is not bool:
        raise ValueError('association_only must be a boolean')
    if association_only and (type(worker_count) is not int or worker_count < 1):
        raise ValueError('association-only requires a positive fitted worker count')
    prepared = _association_inputs(plan) if association_only else None
    context = _reuse_context(plan)
    receipts = [Path(plan['output_directory'])/f'fit_worker_{i}.complete.private.json'
                for i in range(worker_count)]
    while not all(path.exists() for path in receipts):
        if association_only:
            raise FileNotFoundError('association-only requires every existing completed fit receipt')
        for i in range(worker_count):
            root = Path(plan['output_directory'])
            if (root/f'worker_{i}.failed.private.json').exists():
                raise RuntimeError('a distributed worker failed before the model barrier')
            exit_file = root/f'worker_{i}.exit.anonymous.json'
            if exit_file.exists() and json.loads(exit_file.read_text()).get('exit_code', 0) != 0:
                raise RuntimeError('a distributed worker exited before the model barrier')
        time.sleep(10)
    receipt_bindings = ([(path, _file_stat(path), digest_file(path)) for path in receipts]
                        if association_only else [])
    results = ([_bound_json(path, checksum) for path, _, checksum in receipt_bindings]
               if association_only else [json.loads(path.read_text()) for path in receipts])
    if association_only and context is None:
        for i, result in enumerate(results):
            if (type(result.get('worker_id')) is not int or result['worker_id'] != i
                    or result.get('mode') != 'fresh_fit' or result.get('errors') != []
                    or type(result.get('assigned')) is not int
                    or result.get('assigned') != len(range(i, plan['trait_count'], worker_count))
                    or result.get('model_fit_source_implementation_sha256') != plan['source_implementation_sha256']
                    or result.get('consumer_source_implementation_sha256') != plan['source_implementation_sha256']
                    or result.get('null_fit_compatibility_sha256') is not None
                    or result.get('model_import_sha256') is not None
                    or [entry.get('trait_index') for entry in result.get('models', [])]
                        != list(range(i, plan['trait_count'], worker_count))):
                raise ValueError('association-only existing fresh-fit worker/provenance coverage differs')
    if context is not None:
        for i, result in enumerate(results):
            assigned = (list(range(i, plan['trait_count'], worker_count))
                        if 'selected_trait_indices' not in context
                        else context['selected_trait_indices'][i::worker_count])
            if (type(result.get('worker_id')) is not int or result['worker_id'] != i
                    or result.get('mode') != 'verified_reuse' or result.get('errors') != []
                    or result.get('model_fit_source_implementation_sha256') != context['fit_source']
                    or result.get('consumer_source_implementation_sha256') != plan['source_implementation_sha256']
                    or result.get('null_fit_compatibility_sha256') != plan['null_fit_compatibility_sha256']
                    or result.get('model_import_sha256') != plan['model_import_sha256']
                    or result.get('assigned') != len(assigned)
                    or [entry.get('trait_index') for entry in result.get('models', [])]
                        != assigned):
                raise ValueError('current reuse verification receipt worker/provenance coverage differs')
            if context.get('reuse_receipt_kind') in ('immutable_fitted_model_consumption',
                    'immutable_fitted_gaussian_consumption') and (
                    result.get('reuse_receipt_kind') != context['reuse_receipt_kind']
                    or result.get('compatible_null_fit_math') is not False
                    or result.get('fitting_implementation_equivalence_verified') is not False
                    or result.get('model_fitting_disabled') is not True
                    or result.get('input_preparation_disabled') is not True
                    or type(result.get('models_refitted')) is not int or result['models_refitted'] != 0):
                raise ValueError('current stored-model consumption receipt claims fitting equivalence or preparation')
    if any(result['errors'] for result in results):
        raise RuntimeError('one or more null fits failed; full association cannot silently omit phenotypes')
    models = sorted((entry for result in results for entry in result['models']), key=lambda entry: entry['trait_index'])
    expected_indices = (context['selected_trait_indices'] if context is not None
                        and 'selected_trait_indices' in context else list(range(plan['trait_count'])))
    if [entry['trait_index'] for entry in models] != expected_indices:
        raise RuntimeError('fitted-model coverage differs from the complete phenotype plan')
    if context is not None and 'selected_trait_indices' in context and any(
            entry.get('family') != 'gaussian' for entry in models):
        raise ValueError('Gaussian association cannot consume binary or SPA models')
    if association_only and any(type(entry.get('trait_index')) is not int for entry in models):
        raise ValueError('association-only fitted trait identities must be integer indices')
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
        metadata = (_bound_json(model_path, entry['sha256']) if association_only
                    else json.loads(model_path.read_text()))
        if any(metadata['fit'].get(key) != value for key,value in expected.items()):
            raise ValueError('fitted models do not share the bound input/source/cohort')
        if association_only:
            _association_model_metadata(plan, entry, metadata, prepared)
    if association_only:
        if digest_file(Path(plan['prepared_inputs_directory'])/'manifest.private.json') != expected['input_manifest_sha256']:
            raise ValueError('association-only prepared input manifest changed during verification')
        if any(_file_stat(path) != identity or digest_file(path) != checksum
               for path, identity, checksum in receipt_bindings):
            raise ValueError('association-only completed fit receipt changed during verification')
    return models


def _partition_configuration(configuration, worker_id, worker_count):
    """Assign genetic jobs exactly once, retaining their chromosome-local IDs."""
    config, chromosomes, ordinal = dict(configuration), [], 0
    for original in configuration['chromosomes']:
        jobs = []
        for index, original_job in enumerate(original['jobs']):
            job = dict(original_job)
            job.setdefault('checkpoint_job_index', index)
            if ordinal % worker_count == worker_id:
                jobs.append(job)
            ordinal += 1
        if jobs:
            chromosome = dict(original, jobs=jobs)
            chromosomes.append(chromosome)
    config['chromosomes'] = chromosomes
    return config


def _association_partition_strategy(plan):
    """Give each canonical CSV path exactly one active scientific owner.

    Fixed cohort tiles retain their boundaries when GPU ownership changes on
    resume. Legacy mixed-shard plans keep their original genetic dispatch.
    """
    strategy = plan.get('association_partition_strategy', 'auto')
    if plan.get('input_contract','mixed')=='continuous':
        if plan.get('binary_only'):
            raise ValueError('continuous and binary WGS input contracts cannot be combined')
        if strategy not in ('auto','chromosomes'):
            raise ValueError('continuous WGS requires chromosomes GPU dispatch')
        return 'chromosomes'
    if plan.get('binary_only'):
        if strategy not in ('auto', 'whole_chromosomes'):
            raise ValueError('binary WGS requires whole_chromosomes GPU dispatch')
        return 'whole_chromosomes'
    canonical = (plan.get('public_api_options') or
                 plan.get('csv_output_layout') == 'per_trait_checkpoint')
    if canonical:
        if strategy == 'auto':
            return 'cohort_tiles'
        if strategy not in ('cohort_tiles','chromosomes','whole_chromosomes'):
            raise ValueError('per-trait checkpoint CSV requires cohort_tiles, chromosomes or whole_chromosomes dispatch')
    return strategy


def _execute_worker(plan, *, worker_id=0, worker_count=1, device='cuda:0', fit_only=False,
                    association_only=False, file_descriptor_admission=None):
    """One GPU worker: caller-bound CPU affinity, then genetic-major association."""
    from .worker_file_resources import configure_worker_file_capacity
    file_admission = (configure_worker_file_capacity(plan.get('trait_count', 0))
                      if file_descriptor_admission is None else file_descriptor_admission)
    if type(fit_only) is not bool or type(association_only) is not bool or fit_only and association_only:
        raise ValueError('fit_only and association_only must be mutually exclusive booleans')
    affinity = sorted(os.sched_getaffinity(0))
    if plan.get('public_api_options'):
        pool = plan.get('cpu_pool', affinity)
        if not set(pool) <= set(affinity):
            raise ValueError('the worker CPU allocation escaped the caller affinity')
        affinity = pool
    cpu_offset = int(plan.get('cpu_offsets', {}).get(str(worker_id), 0))
    cpu_threads = plan.get('cpu_threads', 8)
    if type(cpu_threads) is not int or cpu_threads < 1 or cpu_threads > len(affinity):
        raise ValueError('cpu_threads requires enough CPUs in the current process affinity')
    selected = sorted(affinity[(cpu_offset+index) % len(affinity)] for index in range(cpu_threads))
    os.sched_setaffinity(0, selected)
    torch.set_num_threads(cpu_threads if plan.get('public_api_options') else 4)
    if not plan.get('inline_worker') and torch.get_num_interop_threads() != 1:
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
        all_thread_affinities_verified=True, thread_affinities=thread_affinities,
        file_descriptors=file_admission))
    if _implementation_sha256() != plan['source_implementation_sha256']:
        raise ValueError('frozen implementation SHA differs from the dispatch plan')
    if association_only:
        entries = _all_models(plan, worker_count, association_only=True)
    else:
        result = fit_models(plan, worker_id=worker_id, worker_count=worker_count, device=device)
        if result['errors']:
            raise RuntimeError('assigned null fits failed; retain the private fit receipts and block association')
        if fit_only:
            return result
        entries = _all_models(plan, worker_count)
    from .phewas_runtime.runtime import run_batched_configuration
    from .phewas_resources import HostMemoryGate
    repository, writer, gate = None, None, None
    association_failed = True
    try:
        dispatch = None
        if (plan.get('public_api_options') or plan.get('csv_output_layout') == 'per_trait_checkpoint'
                or 'association_partition_strategy' in plan):
            from .phewas_partition import build_trait_dispatch, publish_trait_dispatch, dispatch_configuration
            dispatch = build_trait_dispatch(entries,worker_count,plan['configuration'],
                source_binding=dict(source_implementation_sha256=plan['source_implementation_sha256'],
                    prepared_inputs_manifest_sha256=plan['prepared_inputs_manifest_sha256'],
                    plan_sha256=digest_file(output/'plan.private.json')),
                strategy=_association_partition_strategy(plan),
                model_cache_capacity_bytes=plan.get('association_model_cache_capacity_bytes'),
                binary_normal_cache=plan.get('binary_normal_cache','derived_fp64'),
                trait_batch_size=plan.get('trait_batch_size',32))
            publish_trait_dispatch(output,dispatch,plan.get('attempt_id','production'))
        assigned_traits = None if dispatch is None else dispatch['workers'][worker_id]['trait_indices']
        repository = ModelRepository(entries, trait_indices=assigned_traits,
            binary_normal_cache=plan.get('binary_normal_cache', 'derived_fp64'))
        per_trait_output = (plan.get('public_api_options') or
                            plan.get('csv_output_layout') == 'per_trait_checkpoint')
        if per_trait_output:
            from .phewas_async_writer import AsyncPerTraitCheckpointCSVWriter
            binding = dict(source_implementation_sha256=plan['source_implementation_sha256'],
                input_manifest_sha256=plan['prepared_inputs_manifest_sha256'],
                trait_batch_size=plan.get('trait_batch_size', 32), checkpoint_schema_version=2,
                configuration_sha256=hashlib.sha256(json.dumps(plan['configuration'], sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                models=[dict(trait_index=entry['trait_index'], sha256=entry['sha256']) for entry in entries])
            if dispatch is not None and dispatch['strategy']=='whole_chromosomes':
                from .chromosome_csv import ChromosomeCheckpointCSVWriter
                writer = ChromosomeCheckpointCSVWriter(output/'csv', binding=binding,
                    asynchronous=True,
                    chromosomes=dispatch['workers'][worker_id]['assigned_chromosomes'],
                    resume=plan.get('resume', False), worker_id=str(worker_id),
                    minimum_free_bytes=plan.get('csv_minimum_free_bytes', 0),
                    file_descriptor_admission=file_admission)
            else:
                writer = AsyncPerTraitCheckpointCSVWriter(output/'csv', binding=binding,
                    resume=plan.get('resume', False), worker_id=str(worker_id),
                    minimum_free_bytes=plan.get('csv_minimum_free_bytes', 0),
                    ownership_scope=('chromosome' if dispatch is not None
                        and dispatch['strategy']=='chromosomes' else 'trait'))
        else:
            writer = StreamingCSVWriter(output/'csv', worker_id=str(worker_id))
        gate = HostMemoryGate(output, **_host_memory_policy(plan))
        config = (dispatch_configuration(plan['configuration'],dispatch,worker_id) if dispatch is not None
                  else dict(plan['configuration']))
        if dispatch is None:
            config['chromosomes'] = [entry for i, entry in enumerate(config['chromosomes']) if i % worker_count == worker_id]
        # A resumed invocation replays the verified schedule while the writer
        # skips its committed scientific tiles. Keep each invocation's journal
        # independently instead of overwriting history or appending duplicates.
        journal_suffix = (f'.attempt_{plan.get("attempt_id", "production")}.{time.time_ns()}'
                          if per_trait_output else '')
        completed_jobs = output/f'worker_{worker_id}{journal_suffix}.completed_jobs.private.jsonl'
        if completed_jobs.exists():
            raise FileExistsError('association refuses to overwrite a previous job completion journal')
        from .phewas_partition import GeneticScheduleAudit
        schedule_audit = (GeneticScheduleAudit(config)
                          if dispatch is not None and dispatch['strategy']=='whole_chromosomes' else None)
        def event(item):
            if schedule_audit is not None:
                schedule_audit.observe(item)
            atomic_json(output/f'worker_{worker_id}.progress.anonymous.json',
                        dict(item, worker_id=worker_id, observed_at=time.time()))
            if item.get('event') == 'finished':
                # Append only after the job's staged rows have been accepted. This
                # proves the ordered schedule independently of aggregate counts.
                with completed_jobs.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(dict(item, worker_id=worker_id,
                        observed_at=time.time()), allow_nan=False)+'\n')
        try:
            if not config['chromosomes']:
                # More selected GPUs than genetic jobs is legal. An idle worker
                # participates in the full model barrier and publishes an empty,
                # explicit report rather than inventing completed associations.
                report = dict(job_groups=0, association_rows=dict.fromkeys(('individual', 'coding', 'noncoding', 'ncrna'), 0),
                              trait_rows={}, total_seconds=0.,
                              output=writer.finish(), assigned_genetic_jobs=0)
                completed_jobs.touch(exist_ok=False)
            else:
                report = run_batched_configuration(config, model_repository=repository,
                reader_factory=portable_reader, writer=writer, device=device,
                trait_batch_size=plan.get('trait_batch_size', 32), memory_limit_gib=plan.get('memory_limit_gib'),
                memory_reserve_bytes=plan.get('memory_reserve_bytes', 256*2**20),
                cpu_threads=cpu_threads if plan.get('public_api_options') else 4,
                device_cache_bytes=plan.get('device_cache_bytes', 512*2**20),
                compact_cache_bytes=plan.get('compact_cache_bytes', 64*2**20),
                metadata_cache_bytes=plan.get('metadata_cache_bytes', 256*2**20),
                sample_axis_cache_bytes=plan.get('sample_axis_cache_bytes', 64*2**20),
                sample_axis_host_cache_bytes=plan.get('sample_axis_host_cache_bytes', 64*2**20),
                job_state_cache_bytes=plan.get('job_state_cache_bytes', 0),
                job_state_cache_live_fraction=plan.get('job_state_cache_live_fraction', .25),
                device_cache_live_fraction=plan.get('device_cache_live_fraction'),
                emit=event, host_memory_lease_factory=gate.lease_factory)
            report.update(end_to_end_seconds=time.time()-started, model_repository=repository.summary(),
                          source_implementation_sha256=plan['source_implementation_sha256'],
                          worker_id=worker_id, worker_count=worker_count,
                          assigned_chromosomes=[c['name'] for c in config['chromosomes']],
                          completed_jobs_sha256=digest_file(completed_jobs),
                          completed_jobs_file=completed_jobs.name)
            gate_metadata = getattr(gate,'metadata',None)
            if callable(gate_metadata):
                report['host_memory_admission'] = gate_metadata()
            if dispatch is not None and dispatch['strategy']=='whole_chromosomes':
                report.update(schedule_audit.finish())
            report['file_descriptors'] = report['output'].get('file_descriptors',file_admission)
            if association_only:
                report.update(association_only=True, models_refitted=0,
                              existing_fit_receipts_preserved=True)
            if dispatch is not None:
                report.update(assigned_trait_indices=assigned_traits,assigned_trait_tiles=dispatch['workers'][worker_id]['trait_tiles'],
                    association_partition_strategy=dispatch['strategy'],
                    association_dispatch_sha256=dispatch['dispatch_binding_sha256'],
                    assigned_genetic_schedule_sha256=dispatch['workers'][worker_id]['genetic_schedule_sha256'])
            atomic_json(output/f'worker_{worker_id}.report.private.json', report)
            association_failed = False
            return report
        except Exception as error:
            try:
                atomic_json(output/f'worker_{worker_id}.failed.private.json', dict(error_type=type(error).__name__,
                    error=str(error), traceback=traceback.format_exc(), end_to_end_seconds=time.time()-started))
            except BaseException:
                pass  # A diagnostic write cannot replace the calculation error.
            raise
    finally:
        _close_association_resources(writer, repository, failed=association_failed, gate=gate)


def _execute_worker_bound(plan, *, worker_id=0, worker_count=1, device='cuda:0', fit_only=False,
                          association_only=False):
    """Publish a worker exit; optionally consume already certified null fits.

    ``association_only=True`` requires every fit receipt and its scientific
    bindings to remain valid. It never fits models or rewrites fit receipts.
    The default still fits assigned null models before association.
    """
    if type(fit_only) is not bool or type(association_only) is not bool or fit_only and association_only:
        raise ValueError('fit_only and association_only must be mutually exclusive booleans')
    from .worker_file_resources import configure_worker_file_capacity
    file_admission = configure_worker_file_capacity(plan.get('trait_count', 0))
    started, code = time.time(), 1
    output = Path(plan['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    try:
        streams = plan.get('genotype_streams')
        if streams is None and not plan.get('public_api_options'):
            if plan.get('resume'):
                raise ValueError('older deployment plans need persistent stream bindings before resume')
            # Existing non-resumable deployments have no stored cross-attempt
            # proof. Preserve their API while observing this invocation's
            # actual streams before and after all work.
            streams = _stream_bindings(plan['configuration']['chromosomes'])
        _verify_stream_bindings(streams)
        result = _execute_worker(plan, worker_id=worker_id, worker_count=worker_count,
                                 device=device, fit_only=fit_only, association_only=association_only,
                                 file_descriptor_admission=file_admission)
        _verify_stream_bindings(streams)
        code = 0
        return result
    except BaseException as error:
        try:
            atomic_json(output/f'worker_{worker_id}.failed.private.json',
                        dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
        except BaseException:
            pass
        raise
    finally:
        original_error = sys.exc_info()[0] is not None
        try:
            atomic_json(output/f'worker_{worker_id}.exit.anonymous.json',
                        dict(worker_id=worker_id, exit_code=code, elapsed_seconds=time.time()-started))
        except BaseException:
            if not original_error:
                raise



def execute_worker(plan, *, worker_id=0, worker_count=1, device='cuda:0', fit_only=False,
                   association_only=False, invocation_lease=None):
    """Keep public output ownership alive until this spawned worker exits.

    The optional descriptor is runtime-only and never enters the private plan
    or scientific binding. Standalone existing-plan workers retain their API.
    """
    from .phewas_trait_csv import held_invocation_lease
    with held_invocation_lease(invocation_lease):
        return _execute_worker_bound(plan,worker_id=worker_id,worker_count=worker_count,
            device=device,fit_only=fit_only,association_only=association_only)


def prepare_phewas_run(phenotype_csv, covariate_csv, cache_directory, *, output_directory,
                       chromosomes='all', trait_batch_size=32, continuous_transform='paper',
                       analyses=('individual', 'coding', 'noncoding', 'ncrna'),
                       phenotype_families='auto', phenotype_covariates=None, rm_nan=True, binary_only=False,
                       gene_variant_type='SNV', memory_limit_gib=None, memory_reserve_gib=1., host_memory_limit_gib=None,
                       cpu_threads=8, grm='sparse', grm_file=None, resume=False,
                       covariance_block_size=4096, long_mask_threshold=5000, long_mask_method="full",
                       long_mask_rank=512, seed=1729, single_mac_cutoff=20,
                       single_group_variants=5000, single_region_size=10_000_000,
                       individual_effective_block_size=1024, single_cross_frame_batches=False,
                       device_cache_bytes=512*2**20, compact_cache_bytes=64*2**20,
                       metadata_cache_bytes=256*2**20, input_contract='mixed',
                       association_partition_strategy='auto'):
    """Prepare one complete multi-trait plan from the same three scientific inputs.

    Sparse GRM is the default: ``grm_file`` or the dataset's ``kinship_npz``
    sidecar supplies its explicitly bound sample IDs. ``grm='none'`` fits
    independent Gaussian/logistic models without kinship. Covariate profiles
    come only from the arguments; neighboring or dataset profile files never
    change the default of all supplied covariates.
    Phenotype/covariate CSVs use FID/IID identities and numerical values or NA;
    the legacy numeric cache accepts its explicitly preserved numeric ID axis.
    rm_nan=True fills missing phenotypes after checking selected covariates;
    False retains only available phenotype/covariate cases. The policy is
    bound to the prepared input manifest and null models. Complete
    outputs are independent trait CSV.gz directories, a trait dictionary and
    bound logical-block receipts. Resume verifies committed compressed ranges
    before skipping their association and recovers only uncommitted tails.
    """
    from .phewas_cache import _cache_path
    if type(single_cross_frame_batches) is not bool:
        raise ValueError('single_cross_frame_batches must be a boolean')
    _resource_options(memory_limit_gib, memory_reserve_gib, cpu_threads, trait_batch_size, host_memory_limit_gib)
    for name, value in dict(device_cache_bytes=device_cache_bytes, compact_cache_bytes=compact_cache_bytes,
                            metadata_cache_bytes=metadata_cache_bytes).items():
        if type(value) is not int or value < 0:
            raise ValueError(name+' must be a nonnegative integer')
    if type(rm_nan) is not bool:
        raise ValueError('rm_nan must be a boolean')
    if type(binary_only) is not bool:
        raise ValueError('binary_only must be a boolean')
    if binary_only and continuous_transform != 'none':
        raise ValueError('binary WGS requires continuous_transform=none')
    if type(resume) is not bool:
        raise ValueError('resume must be a boolean')
    if input_contract not in ('mixed','continuous'):
        raise ValueError('input_contract must be mixed or continuous')
    association_partition_strategy = _association_partition_strategy(dict(binary_only=binary_only,
        input_contract=input_contract, association_partition_strategy=association_partition_strategy,
        public_api_options=True))
    if grm not in ('sparse', 'none') or (grm == 'none' and grm_file is not None):
        raise ValueError('grm must be sparse or none; grm_file requires sparse mode')
    if continuous_transform not in ('paper', 'none', 'rint'):
        raise ValueError('continuous_transform must be paper, none or rint')
    chromosomes = _selected_chromosomes(chromosomes)
    root, manifest_path, manifest, entries, sample_ids = _dataset(cache_directory, chromosomes)
    genotype_streams = _stream_bindings(entries)
    sidecars = manifest.get('phewas', {})
    kinship = None
    if grm == 'sparse':
        if grm_file is not None:
            kinship = Path(grm_file).resolve()
        elif 'kinship_npz' in sidecars:
            kinship = _cache_path(root, sidecars['kinship_npz'])
        else:
            raise ValueError('sparse GRM requires grm_file or the ID-bound phewas.kinship_npz sidecar')
        # Validate the ID-bound sparse representation now; the mature fitter
        # requires coverage of each trait's actual complete-case sample subset.
        SparseKinshipData.load_npz(kinship)
    cohort = _cache_path(root, sidecars['cohort_rows']) if 'cohort_rows' in sidecars else None
    config = build_configuration(cache_directory, chromosomes=chromosomes, analyses=analyses,
        gene_variant_type=gene_variant_type, memory_limit_gib=memory_limit_gib,
        covariance_block_size=covariance_block_size, long_mask_threshold=long_mask_threshold,
        long_mask_method=long_mask_method, long_mask_rank=long_mask_rank, seed=seed, single_mac_cutoff=single_mac_cutoff,
        single_group_variants=single_group_variants, single_region_size=single_region_size,
        individual_effective_block_size=individual_effective_block_size,
        single_cross_frame_batches=single_cross_frame_batches)
    if input_contract == 'continuous':
        from .continuous_run import _continuous_performance_configuration
        config = _continuous_performance_configuration(config)
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError('output directory already contains an analysis; use resume=True or a new directory')
    output.mkdir(parents=True, exist_ok=True)
    prepared = output/'inputs'
    if not prepared.exists():
        prepare_phewas_inputs(phenotypes=phenotype_csv, covariates=covariate_csv,
            phenotype_families=phenotype_families, phenotype_covariates=phenotype_covariates,
            rm_nan=rm_nan, binary_only=binary_only,
            sample_ids=root/manifest.get('sample_ids', 'sample_ids.npy'), output=prepared,
            input_contract=input_contract)
    inputs = PhewasInputs(prepared)
    try:
        for name, path in (('phenotypes', phenotype_csv), ('covariates', covariate_csv)):
            binding = inputs.manifest['sources'][name]
            if digest_file(path) != binding['sha256']:
                raise ValueError('existing prepared matrix is bound to another input CSV')
        requested = dict(phenotype_families=(dict(phenotype_families) if isinstance(phenotype_families, Mapping) else 'auto'),
            phenotype_covariates={name: list(columns) for name, columns in (phenotype_covariates or {}).items()},
            rm_nan=rm_nan, input_contract=input_contract, binary_only=binary_only)
        recorded = dict(inputs.manifest.get('trait_options', dict(phenotype_families='auto', phenotype_covariates={})))
        # Inputs created before the explicit policy used available cases.
        recorded.setdefault('rm_nan', False)
        recorded.setdefault('binary_only', False)
        recorded.setdefault('input_contract', 'mixed')
        if 'profiles' in inputs.manifest['sources'] or recorded != requested:
            raise ValueError('existing prepared matrix is bound to another trait family/covariate selection')
        if not np.array_equal(inputs.sample_ids, sample_ids):
            raise ValueError('existing prepared matrix has another cache sample axis')
        if input_contract=='continuous':
            from .continuous_run import validate_continuous_metadata
            validate_continuous_metadata(inputs.metadata)
        count = inputs.manifest['trait_count']
        trait_names = [trait['name'] for trait in inputs.metadata['traits']]
        assigned_families = {trait['name']: trait['family'] for trait in inputs.metadata['traits']}
        dictionary_sha = trait_index_sha256(trait_names, assigned_families)
    finally:
        inputs.close()
    plan = dict(schema_version=1, output_directory=str(output), prepared_inputs_directory=str(prepared),
        phenotype_csv=str(Path(phenotype_csv).resolve()), covariate_csv=str(Path(covariate_csv).resolve()),
        cache_directory=str(root), kinship_npz=None if kinship is None else str(kinship),
        genotype_streams=genotype_streams,
        kinship_sha256=None if kinship is None else digest_file(kinship), grm=grm,
        cohort_rows=None if cohort is None else str(cohort),
        cohort_sha256=None if cohort is None else digest_file(cohort), trait_count=count,
        trait_index_sha256=dictionary_sha,
        continuous_transform=continuous_transform, rm_nan=rm_nan, binary_only=binary_only,
        input_contract=input_contract,
        trait_batch_size=trait_batch_size, cpu_threads=cpu_threads,
        memory_limit_gib=memory_limit_gib, memory_reserve_bytes=int(memory_reserve_gib*2**30),
        host_memory_limit_gib=host_memory_limit_gib,
        device_cache_bytes=device_cache_bytes, compact_cache_bytes=compact_cache_bytes,
        metadata_cache_bytes=metadata_cache_bytes, resume=resume, public_api_options=True,
        csv_output_layout='per_trait_checkpoint',
        association_partition_strategy=_association_partition_strategy(dict(binary_only=binary_only,
            input_contract=input_contract, association_partition_strategy=association_partition_strategy,
            public_api_options=True)),
        attempt_id=str(time.time_ns()), source_implementation_sha256=_implementation_sha256(),
        prepared_inputs_manifest_sha256=digest_file(prepared/'manifest.private.json'),
        configuration=config, strict_logp_delta_required=False,
        comparison_significance_p=1e-5, output_all_pvalues=True)
    if input_contract == 'continuous':
        # Derived from the already bound and validated prepared metadata.
        plan['_continuous_spawn_families'] = list(assigned_families.values())
    previous_path = output/'plan.private.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        for key in ('source_implementation_sha256', 'prepared_inputs_manifest_sha256', 'kinship_sha256',
                    'cohort_sha256', 'continuous_transform', 'rm_nan', 'configuration', 'trait_count', 'trait_batch_size',
                    'genotype_streams', 'trait_index_sha256', 'csv_output_layout', 'binary_only',
                    'input_contract', 'association_partition_strategy'):
            default = {'rm_nan':False, 'input_contract':'mixed',
                       'binary_only':False,
                       'association_partition_strategy':'cohort_tiles'}.get(key)
            if key == 'configuration':
                if (_legacy_single_batch_configuration(previous.get(key)) !=
                        _legacy_single_batch_configuration(plan.get(key))):
                    raise ValueError('resume plan scientific/source bindings differ; use a new output directory')
                # A legacy missing key and explicit False execute identically,
                # but old writers hashed their original raw JSON. Keep that
                # exact representation rather than rewriting checkpoint proof.
                plan[key] = previous[key]
            elif previous.get(key,default) != plan.get(key):
                raise ValueError('resume plan scientific/source bindings differ; use a new output directory')
        # Retain the original plan as audit evidence across resource/device changes.
        atomic_json(output/f'plan.previous.{plan["attempt_id"]}.private.json', previous)
    _verify_stream_bindings(genotype_streams)
    atomic_json(previous_path, plan)
    write_trait_index(output, trait_names, assigned_families, expected_sha256=dictionary_sha)
    return plan


def run_phewas(phenotype_csv, covariate_csv, cache_directory, *, output_directory,
               gpu_ids=(0,), chromosomes='all', trait_batch_size=32, continuous_transform='paper',
               analyses=('individual', 'coding', 'noncoding', 'ncrna'),
               phenotype_families='auto', phenotype_covariates=None, rm_nan=True, binary_only=False,
               gene_variant_type='SNV', memory_limit_gib=None, memory_reserve_gib=1., host_memory_limit_gib=None,
               cpu_threads=8, grm='sparse', grm_file=None, resume=False,
               covariance_block_size=4096, long_mask_threshold=5000, long_mask_method="full",
               long_mask_rank=512, seed=1729, single_mac_cutoff=20,
               single_group_variants=5000, single_region_size=10_000_000,
               individual_effective_block_size=1024, single_cross_frame_batches=False,
               device_cache_bytes=512*2**20, compact_cache_bytes=64*2**20,
               metadata_cache_bytes=256*2**20, input_contract='mixed',
               association_partition_strategy='auto'):
    """One call for all traits and chromosomes on the selected local GPUs.

    Remote deployments use the identical private plan on each host. Genetic
    jobs are partitioned across workers when the fitted bank fits one GPU's
    cache. Larger banks assign whole ordered cohorts to workers; each worker
    scans the genetic schedule for its own disjoint traits. CPU affinity is
    bounded per worker; the node quota is shared.
    """
    import multiprocessing as mp
    gpu_ids = tuple(gpu_ids)
    if (not gpu_ids or any(type(value) is not int or value < 0 for value in gpu_ids)
            or len(set(gpu_ids)) != len(gpu_ids)):
        raise ValueError('gpu_ids must contain distinct selected devices')
    plan = prepare_phewas_run(phenotype_csv, covariate_csv, cache_directory,
        output_directory=output_directory, chromosomes=chromosomes, trait_batch_size=trait_batch_size,
        continuous_transform=continuous_transform, analyses=analyses,
        phenotype_families=phenotype_families, phenotype_covariates=phenotype_covariates,
        rm_nan=rm_nan, binary_only=binary_only,
        gene_variant_type=gene_variant_type, memory_limit_gib=memory_limit_gib,
        memory_reserve_gib=memory_reserve_gib, host_memory_limit_gib=host_memory_limit_gib,
        cpu_threads=cpu_threads, grm=grm,
        grm_file=grm_file, resume=resume, covariance_block_size=covariance_block_size,
        long_mask_threshold=long_mask_threshold, long_mask_method=long_mask_method, long_mask_rank=long_mask_rank, seed=seed,
        single_mac_cutoff=single_mac_cutoff, single_group_variants=single_group_variants,
        single_region_size=single_region_size, individual_effective_block_size=individual_effective_block_size,
        single_cross_frame_batches=single_cross_frame_batches,
        device_cache_bytes=device_cache_bytes, compact_cache_bytes=compact_cache_bytes,
        metadata_cache_bytes=metadata_cache_bytes, input_contract=input_contract,
        association_partition_strategy=association_partition_strategy)
    from .cpu_budget import detect_cpu_capacity
    capacity = detect_cpu_capacity()
    if cpu_threads > capacity['effective_cores']:
        raise ValueError('cpu_threads exceeds the effective affinity/cgroup CPU quota')
    plan['cpu_pool'] = sorted(os.sched_getaffinity(0))[:capacity['effective_cores']]
    plan['cpu_capacity'] = capacity
    plan['cpu_offsets'] = {str(i):(cpu_threads*i) % capacity['effective_cores'] for i in range(len(gpu_ids))}
    from .phewas_partition import gpu_model_cache_capacity, verify_dispatch_reports
    plan['association_partition_strategy'] = _association_partition_strategy(dict(plan,
        binary_only=binary_only, input_contract=input_contract,
        association_partition_strategy=association_partition_strategy))
    plan['association_model_cache_capacity_bytes'] = gpu_model_cache_capacity(gpu_ids,
        memory_limit_gib=memory_limit_gib,reserve_bytes=int(memory_reserve_gib*2**30),
        binary_only=binary_only)
    output = Path(plan['output_directory'])
    if resume:
        prior = [path for pattern in ('worker_*.exit.anonymous.json', 'worker_*.failed.private.json',
                'worker_*.identity.private.json', 'worker_*.progress.anonymous.json',
                'worker_*.report.private.json', 'fit_worker_*.complete.private.json',
                'fit_worker_*.progress.anonymous.json') for path in output.glob(pattern)]
        if prior:
            history = output/'attempts'/plan['attempt_id']
            history.mkdir(parents=True, exist_ok=False)
            for path in prior:
                path.rename(history/path.name)
    atomic_json(Path(plan['output_directory'])/'plan.private.json', plan)
    if len(gpu_ids) == 1:
        # A single device runs in this call's process, including notebook use.
        # Restore thread/affinity policy before returning to the caller.
        plan['inline_worker'] = True
        previous_threads = torch.get_num_threads()
        previous_affinity = {}
        for path in Path('/proc/self/task').iterdir():
            try:
                previous_affinity[int(path.name)] = sorted(os.sched_getaffinity(int(path.name)))
            except ProcessLookupError:
                pass
        main_affinity = sorted(os.sched_getaffinity(0))
        try:
            reports = [execute_worker(plan, worker_id=0, worker_count=1, device=f'cuda:{gpu_ids[0]}')]
            dispatch = json.loads((output/f'association_dispatch.{plan["attempt_id"]}.private.json').read_text())
            completion = verify_dispatch_reports(dispatch,reports)
            for report in reports:
                report['global_completion'] = completion
            return reports
        finally:
            torch.set_num_threads(previous_threads)
            for path in Path('/proc/self/task').iterdir():
                try:
                    os.sched_setaffinity(int(path.name), previous_affinity.get(int(path.name), main_affinity))
                except ProcessLookupError:
                    pass
    context = mp.get_context('spawn')
    from .phewas_trait_csv import invocation_lease_for_spawn, close_pending_invocation_lease, _note
    workers = []
    started_workers = []
    thread_environment=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')
    from ._continuous_spawn_allocator import continuous_spawn_allocator_scope
    with continuous_spawn_allocator_scope(
            input_contract=plan.get('input_contract', 'mixed'),
            families=plan.get('_continuous_spawn_families', ()),
            worker_count=len(gpu_ids)) as allocator_policy:
        previous_environment={name:os.environ.get(name) for name in thread_environment}
        try:
            # Spawn imports numerical libraries before entering execute_worker.
            # Give those imports the same bounded CPU pool rather than allowing
            # one BLAS helper pool per visible host CPU. The parent policy returns
            # immediately after all owned children have inherited this environment.
            for name in thread_environment:
                os.environ[name]=str(cpu_threads)
            for index,gpu in enumerate(gpu_ids):
                handle=invocation_lease_for_spawn()
                worker=None
                try:
                    worker=context.Process(target=execute_worker,kwargs=dict(plan=plan,worker_id=index,
                        worker_count=len(gpu_ids),device=f'cuda:{gpu}',invocation_lease=handle))
                    worker.start()
                except BaseException as error:
                    if worker is not None and getattr(worker,'pid',None) is not None:
                        # A rare start error after the OS child exists still owns
                        # that exact child; include it in the cleanup below.
                        started_workers.append(worker)
                    else:
                        try:
                            close_pending_invocation_lease(handle)
                        except BaseException as cleanup:
                            _note(error, f'invocation transfer cleanup also failed: {cleanup}')
                    raise
                workers.append(worker)
                started_workers.append(worker)
        except BaseException:
            for worker in started_workers:
                if worker.is_alive():
                    worker.terminate()
            for worker in started_workers:
                worker.join(timeout=10)
            raise
        finally:
            for name,value in previous_environment.items():
                if value is None:
                    os.environ.pop(name,None)
                else:
                    os.environ[name]=value
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
    atomic_json(output/f'spawn_allocator.{plan["attempt_id"]}.private.json', allocator_policy)
    reports = [json.loads((Path(plan['output_directory'])/f'worker_{i}.report.private.json').read_text())
               for i in range(len(workers))]
    dispatch = json.loads((output/f'association_dispatch.{plan["attempt_id"]}.private.json').read_text())
    completion = verify_dispatch_reports(dispatch,reports)
    for report in reports:
        report['global_completion'] = completion
    return reports

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
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument('--fit-only', action='store_true')
    phases.add_argument('--association-only', action='store_true',
                        help='consume existing completed null fits without refitting or rewriting fit receipts')
    parser.add_argument('--host-memory-limit-gib', type=float, default=None,
                        help='shared host task budget in GiB; default detects all finite limits')
    parser.add_argument('--rm-nan', action=argparse.BooleanOptionalAction, default=None,
                        help='fill phenotypes by default; --no-rm-nan selects available cases')
    parser.add_argument('--long-mask-method', choices=['full','fastskat','liu'], default=None,
                        help='full native FP32 spectra by default; existing plans keep their bound policy')
    parser.add_argument('--single-cross-frame-batches', action=argparse.BooleanOptionalAction, default=None,
                        help='merge bounded Single requests across frames; an existing plan keeps its bound policy')
    parser.add_argument('--continuous-transform', choices=['paper','none','rint'], default='paper')
    args = parser.parse_args(argv)
    if args.plan is None:
        if args.association_only:
            parser.error('--association-only requires an existing --plan')
        if any(value is None for value in (args.phenotype_csv, args.covariate_csv, args.cache_directory, args.output_directory)):
            parser.error('provide phenotype CSV, covariate CSV, cache directory and output directory')
        return run_phewas(args.phenotype_csv, args.covariate_csv, args.cache_directory,
                          output_directory=args.output_directory, gpu_ids=args.gpu_ids,
                          continuous_transform=args.continuous_transform,
                          long_mask_method='full' if args.long_mask_method is None else args.long_mask_method,
                          rm_nan=True if args.rm_nan is None else args.rm_nan,
                          host_memory_limit_gib=args.host_memory_limit_gib,
                          single_cross_frame_batches=False if args.single_cross_frame_batches is None else args.single_cross_frame_batches)
    plan = json.loads(Path(args.plan).read_text())
    if (args.long_mask_method is not None and args.long_mask_method !=
            plan['configuration']['analysis_options'].get('long_mask_method', 'full')):
        parser.error('long_mask_method cannot change an existing prepared plan; use a new output directory')
    if (args.single_cross_frame_batches is not None and args.single_cross_frame_batches !=
            plan['configuration'].get('single_cross_frame_batches', False)):
        parser.error('single_cross_frame_batches cannot change an existing prepared plan; use a new output directory')
    if args.rm_nan is not None and args.rm_nan != plan.get('rm_nan', False):
        parser.error('rm_nan cannot change an existing prepared plan; use a new output directory')
    if args.host_memory_limit_gib is not None:
        plan['host_memory_limit_gib'] = _validate_host_memory_limit(args.host_memory_limit_gib)
    execute_worker(plan, worker_id=args.worker_id, worker_count=args.worker_count,
                   device=args.device, fit_only=args.fit_only, association_only=args.association_only)


if __name__ == '__main__':
    main()
