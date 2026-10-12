"""Fail-closed Gaussian selection receipts; anonymous fixtures, not benchmarks."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from fudan_wgs_toolkit import phewas_run as run
from test_phewas_model_reuse import (OLD, NEW, consumption_reuse, reuse,
                                   refresh_inventory, write)


def refresh(plan, inventory, proof, semantic):
    plan['null_fit_compatibility_sha256'] = write(Path(plan['null_fit_compatibility_receipt']), proof)
    inventory['null_fit_compatibility_sha256'] = plan['null_fit_compatibility_sha256']
    proof['immutable_inventory_fingerprint'] = run._model_inventory_fingerprint(inventory)
    proof['input_semantic_sha256'] = write(Path(proof['input_semantic_receipt']), semantic)
    plan['null_fit_compatibility_sha256'] = write(Path(plan['null_fit_compatibility_receipt']), proof)
    inventory['null_fit_compatibility_sha256'] = plan['null_fit_compatibility_sha256']
    refresh_inventory(plan, inventory)


@pytest.fixture
def gaussian_reuse(consumption_reuse, monkeypatch):
    plan, inventory, prior = consumption_reuse
    existing_inputs = run.PhewasInputs
    monkeypatch.setattr(run, 'PhewasInputs', lambda *args, **kwargs: existing_inputs(*args))
    parent = copy.deepcopy(prior)
    previous = 'd' * 64
    parent['consumer_source_implementation_sha256'] = previous
    parent['components']['phewas_inputs.py:PhewasInputs'] = dict(
        identical=True, old_fingerprint='f' * 64, new_fingerprint='f' * 64)
    parent_path = Path(plan['output_directory']) / 'parent_consumption.json'
    parent_sha = write(parent_path, parent)
    selected = [0]
    plan.update(selected_trait_indices=selected, analysis_trait_count=1)
    actual = run._gaussian_consumption_component_fingerprints()
    joined = hashlib.sha256()
    for name, value in sorted(actual.items()):
        joined.update(name.encode()); joined.update(bytes.fromhex(value))
    proof = copy.deepcopy(prior)
    proof.update(schema_version=3, kind='immutable_fitted_gaussian_consumption',
        previous_consumer_source_implementation_sha256=previous,
        parent_consumption_receipt=str(parent_path), parent_consumption_sha256=parent_sha,
        selected_trait_indices=selected, analysis_trait_count=1, rm_nan=False,
        binary_consumption_disabled=True, spa_consumption_disabled=True,
        components={name: dict(identical=True, old_fingerprint=value, new_fingerprint=value)
                    for name, value in actual.items()},
        gaussian_consumption_component_fingerprint=joined.hexdigest(),
        prepared_input_consumer_change=dict(identical=False, old_fingerprint='f' * 64,
            new_fingerprint=run._consumption_component_fingerprints()['phewas_inputs.py:PhewasInputs']),
        input_semantic_receipt=str(Path(plan['output_directory']) / 'semantic.json'))
    semantic = dict(schema_version=1, kind='fixed_existing_gaussian_inputs_semantic_equality',
        old_consumer_source_implementation_sha256=previous,
        new_consumer_source_implementation_sha256=NEW, parent_consumption_sha256=parent_sha,
        prepared_inputs_manifest_sha256=plan['prepared_inputs_manifest_sha256'],
        cohort_sha256=plan['cohort_sha256'], kinship_sha256=plan['kinship_sha256'],
        continuous_transform=plan['continuous_transform'], rm_nan=False,
        old_input_consumer_fingerprint='f' * 64,
        new_input_consumer_fingerprint=proof['prepared_input_consumer_change']['new_fingerprint'],
        old_input_module_sha256='c' * 64,
        new_input_module_sha256=hashlib.sha256((Path(run.__file__).parent / 'phewas_inputs.py').read_bytes()).hexdigest(),
        selected_trait_indices=selected, compared_trait_count=1,
        fields=['sample_indices', 'sample_ids', 'y_raw', 'covariates', 'covariate_names',
                'missing_policy', 'available_n', 'mother_n', 'imputed_n', 'imputation_mean',
                'recoded_nonbinary_n'],
        complete=True, exact_values_equal=True, ordered_sample_axes_equal=True,
        new_design_arrays_readonly=True, repeated_design_values_equal=True,
        model_fitting_disabled=True, input_preparation_disabled=True, models_refitted=0,
        python_version=[3, 12, 1],
        traits=[dict(trait_index=0, family='gaussian', exact_values_equal=True, values_sha256='e' * 64)])
    semantic['fixed_input_file_identities'] = {str(Path(path).resolve()): run._file_stat(Path(path))
        for path in [Path(plan['prepared_inputs_directory'])/'manifest.private.json',
                     plan['cohort_rows'], plan['kinship_npz']]}
    refresh(plan, inventory, proof, semantic)
    return plan, inventory, proof, semantic


def test_gaussian_scope_verifies_only_selected_models_and_preserves_full_bank(gaussian_reuse):
    plan, inventory, _, _ = gaussian_reuse
    context = run._reuse_context(plan)
    assert len(context['models']) == 2
    assert context['selected_trait_indices'] == [0]
    before = {file['path']: Path(file['path']).read_bytes()
              for record in inventory['models'] for file in record['files']}
    result = run.fit_models(plan, device='cpu')
    assert result['mode'] == 'verified_reuse'
    assert result['assigned'] == 1 and result['models'][0]['family'] == 'gaussian'
    assert result['reuse_receipt_kind'] == 'immutable_fitted_gaussian_consumption'
    assert result['models_refitted'] == 0 and result['compatible_null_fit_math'] is False
    assert [item['trait_index'] for item in run._all_models(plan, 1)] == [0]
    assert {path: Path(path).read_bytes() for path in before} == before


@pytest.mark.parametrize('key,value', [
    ('selected_trait_indices', [1]), ('selected_trait_indices', [True]),
    ('selected_trait_indices', [0, 0]), ('selected_trait_indices', []),
    ('analysis_trait_count', 2), ('input_contract', 'continuous'), ('binary_only', True),
    ('continuous_transform', 'none'), ('rm_nan', True),
])
def test_gaussian_scope_rejects_plan_selection_or_input_relabel(gaussian_reuse, key, value):
    plan, _, _, _ = gaussian_reuse
    plan[key] = value
    with pytest.raises(ValueError):
        run._reuse_context(plan)


@pytest.mark.parametrize('key,value', [
    ('binary_consumption_disabled', False), ('spa_consumption_disabled', False),
    ('model_fitting_disabled', False), ('input_preparation_disabled', False),
    ('compatible_null_fit_math', True), ('fitting_implementation_equivalence_verified', True),
    ('models_refitted', 1), ('models_refitted', False),
    ('analysis_trait_count', True), ('rm_nan', 0),
    ('previous_consumer_source_implementation_sha256', OLD),
])
def test_gaussian_scope_rejects_inflated_or_unsafe_proof(gaussian_reuse, key, value):
    plan, inventory, proof, semantic = gaussian_reuse
    proof[key] = value; refresh(plan, inventory, proof, semantic)
    with pytest.raises(ValueError):
        run._reuse_context(plan)


@pytest.mark.parametrize('key,value', [
    ('exact_values_equal', False), ('new_design_arrays_readonly', False),
    ('repeated_design_values_equal', False), ('compared_trait_count', 0),
    ('selected_trait_indices', [1]), ('new_input_module_sha256', 'c' * 64),
    ('old_input_consumer_fingerprint', 'c' * 64), ('python_version', [3, 10]),
    ('prepared_inputs_manifest_sha256', 'c' * 64), ('fields', ['sample_indices']),
    ('traits', []), ('models_refitted', 1),
    ('models_refitted', False), ('compared_trait_count', True),
])
def test_gaussian_scope_requires_bound_complete_actual_input_semantics(gaussian_reuse, key, value):
    plan, inventory, proof, semantic = gaussian_reuse
    semantic[key] = value; refresh(plan, inventory, proof, semantic)
    with pytest.raises(ValueError, match='semantic comparison'):
        run._reuse_context(plan)


@pytest.mark.parametrize('problem', ['missing_component', 'changed_component', 'fake_input_identical',
                                      'binary_selected', 'bank_file_changed', 'parent_changed'])
def test_gaussian_scope_retains_component_family_and_original_identity_gates(gaussian_reuse, problem):
    plan, inventory, proof, semantic = gaussian_reuse
    if problem == 'missing_component':
        proof['components'].pop(next(iter(proof['components'])))
    elif problem == 'changed_component':
        proof['components'][next(iter(proof['components']))]['new_fingerprint'] = 'c' * 64
    elif problem == 'fake_input_identical':
        proof['prepared_input_consumer_change']['identical'] = True
    elif problem == 'binary_selected':
        plan['selected_trait_indices'] = proof['selected_trait_indices'] = [1]
        semantic['selected_trait_indices'] = [1]; semantic['traits'][0]['trait_index'] = 1
    elif problem == 'bank_file_changed':
        Path(inventory['models'][1]['target'], 'sample_rows.npy').write_bytes(b'changed')
    else:
        Path(proof['parent_consumption_receipt']).write_text('{}')
    refresh(plan, inventory, proof, semantic)
    with pytest.raises(ValueError):
        run._reuse_context(plan)


def test_gaussian_scope_rejects_changed_actual_prepared_identity(gaussian_reuse):
    plan, _, _, _ = gaussian_reuse
    Path(plan['cohort_rows']).touch()
    with pytest.raises(ValueError, match='fixed-input identities'):
        run._reuse_context(plan)


def test_gaussian_scope_cannot_verify_unselected_binary_via_trait_override(gaussian_reuse):
    plan, _, _, _ = gaussian_reuse
    with pytest.raises(ValueError, match='exact selected worker assignment'):
        run.fit_models(plan, device='cpu', trait_indices=[1])


def test_gaussian_scope_barrier_rejects_binary_or_missing_selected_receipt(gaussian_reuse):
    plan, _, _, _ = gaussian_reuse
    run.fit_models(plan, device='cpu')
    path = Path(plan['output_directory']) / 'fit_worker_0.complete.private.json'
    receipt = json.loads(path.read_text()); receipt['models'][0]['trait_index'] = 1
    write(path, receipt)
    with pytest.raises(ValueError, match='worker/provenance coverage'):
        run._all_models(plan, 1)
