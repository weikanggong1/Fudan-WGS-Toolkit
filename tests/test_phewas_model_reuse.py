"""Original-fit/current-consumer provenance and fail-closed reuse contracts.

Anonymous file fixtures exercise gates; they are not statistical benchmarks.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fudan_wgs_toolkit import phewas_run as run
from fudan_wgs_toolkit.phewas_storage import digest_file

OLD, NEW = 'a'*64, 'b'*64


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return digest_file(path)


def refresh_inventory(plan, inventory):
    plan['model_import_sha256'] = write(Path(plan['model_import_receipt']), inventory)


@pytest.fixture
def reuse(tmp_path, monkeypatch):
    old_run, new_output = tmp_path/'old', tmp_path/'new/outputs'
    new_output.mkdir(parents=True)
    prepared = tmp_path/'inputs'
    input_sha = write(prepared/'manifest.private.json', {'trait_count': 2})
    kinship = tmp_path/'kinship.npz';kinship.write_bytes(b'anonymous bound kinship fixture')
    cohort = tmp_path/'cohort.npy';np.save(cohort, np.arange(4, dtype=np.int64))
    plan = dict(output_directory=str(new_output), prepared_inputs_directory=str(prepared), trait_count=2,
        prepared_inputs_manifest_sha256=input_sha, kinship_npz=str(kinship), kinship_sha256=digest_file(kinship),
        cohort_rows=str(cohort), cohort_sha256=digest_file(cohort), continuous_transform='paper',
        source_implementation_sha256=NEW, model_fit_source_implementation_sha256=OLD, reuse_models_only=True)
    components = {name: dict(old_fingerprint='c'*64, new_fingerprint='c'*64, identical=True) for name in (
        'binary_null.py:complete_module', 'phewas_models.py:complete_module', 'phewas_inputs.py:complete_module',
        'rint.py:complete_module', 'numerics.py:complete_module', 'tf32.py:complete_module',
        'tensor_validation.py:complete_module', 'null_model.py:fit_gaussian_null',
        'null_model.py:KinshipSpectrum', 'null_model.py:rank_inverse_normal', 'null_model.py:_r_sum',
        'null_model.py:GaussianNullModel.dataclass_fields',
        'null_model.py:GaussianNullModel.set_matmul_mode_except_new_cache_clear', 'phewas_run.py:_design')}
    joined = hashlib.sha256()
    for name, record in sorted(components.items()):
        joined.update(name.encode());joined.update(bytes.fromhex(record['old_fingerprint']))
    proof = dict(schema_version=1, complete=True, compatible_null_fit_math=True,
        source_unchanged_during_review=True, scientific_model_input_changes=False,
        fitted_math_or_parameter_changes=False, model_fit_source_implementation_sha256=OLD,
        consumer_source_implementation_sha256=NEW, components=components,
        old_null_fit_component_fingerprint=joined.hexdigest(), new_null_fit_component_fingerprint=joined.hexdigest(),
        existing_models_reuse={key: True for key in ('scientifically_compatible',
            'original_model_metadata_and_array_hashes_must_be_preserved',
            'consumer_and_fit_source_binding_must_be_separate', 'missing_reuse_target_must_hard_fail',
            'no_new_fits_may_be_labelled_as_old_source')})
    proof_path = tmp_path/'compatibility.json'
    plan.update(null_fit_compatibility_receipt=str(proof_path), null_fit_compatibility_sha256=write(proof_path, proof))
    original_models, imported = old_run/'outputs/models', new_output/'models'
    imported.mkdir()
    records, entries = [], []
    for index, family in enumerate(('gaussian', 'binomial')):
        target = original_models/f'trait_{index:04d}';target.mkdir(parents=True)
        np.save(target/'sample_rows.npy', np.arange(4, dtype=np.int64))
        state_hashes = {}
        for kind in ('normal', 'spa') if family == 'binomial' else ('normal',):
            state_dir = target/kind;state_dir.mkdir()
            values = dict(sample_ids=np.asarray(['301', '303', '307', '309']), x=np.ones((4, 1), dtype=np.float64))
            arrays = {}
            for name, value in values.items():
                file = state_dir/(name+'.npy');np.save(file, value)
                arrays[name] = dict(file=file.name, sha256=digest_file(file), shape=list(value.shape), dtype=value.dtype.str)
            state = dict(family=family, has_kinship=True, converged=True, matmul_mode='fp64' if kind == 'spa' else 'tf32', arrays=arrays)
            state_hashes[kind] = write(state_dir/'state.private.json', state)
        binding = dict(input_manifest_sha256=input_sha, kinship_sha256=plan['kinship_sha256'],
            source_implementation_sha256=OLD, cohort_sha256=plan['cohort_sha256'], continuous_transform='paper', trait_index=index)
        metadata = dict(schema_version=1, fit=binding, n=4, family=family, normal_state='normal',
            spa_state='spa' if family == 'binomial' else None, state_sha256=state_hashes['normal'],
            spa_state_sha256=state_hashes.get('spa'), sample_rows_sha256=digest_file(target/'sample_rows.npy'))
        metadata_sha = write(target/'model.private.json', metadata)
        link = imported/target.name;link.symlink_to(target, target_is_directory=True)
        files = [dict(path=str(file), sha256=digest_file(file), stat=run._file_stat(file))
                 for file in sorted(target.rglob('*')) if file.is_file()]
        records.append(dict(trait_index=index, target=str(target), symlink=str(link), family=family,
            n=4, metadata_sha256=metadata_sha, files=files))
        entries.append(dict(trait_index=index, path=str(target), family=family, n=4, sha256=metadata_sha))
    old_plan = dict(source_implementation_sha256=OLD, trait_count=2,
        prepared_inputs_manifest_sha256=input_sha, kinship_sha256=plan['kinship_sha256'],
        cohort_sha256=plan['cohort_sha256'], continuous_transform='paper')
    old_plan_sha = write(old_run/'outputs/plan.private.json', old_plan)
    old_exit = dict(worker_id=0, phase='fit', exit_code=0, source_implementation_sha256=OLD, plan_sha256=old_plan_sha)
    exit_sha = write(old_run/'outputs/worker_0.fit.supervisor_exit.anonymous.json', old_exit)
    complete_sha = write(old_run/'outputs/fit_worker_0.complete.private.json', dict(worker_id=0, errors=[], assigned=2, models=entries))
    inventory = dict(schema_version=1, status='verified', model_fit_source_implementation_sha256=OLD,
        consumer_source_implementation_sha256=NEW, null_fit_compatibility_sha256=plan['null_fit_compatibility_sha256'],
        prepared_inputs_manifest_sha256=input_sha, kinship_sha256=plan['kinship_sha256'],
        cohort_sha256=plan['cohort_sha256'], continuous_transform='paper', model_count=2,
        from_run_directory=str(old_run), from_models_directory=str(original_models), models_directory=str(imported),
        from_plan_sha256=old_plan_sha, source_fit_worker_count=1,
        fit_supervisor_exit_sha256={'0': exit_sha}, fit_complete_receipt_sha256={'0': complete_sha},
        all_fit_supervisor_exits_zero=True, full_array_hashes_verified=True, full_array_geometry_verified=True,
        original_model_stores_modified=False, models_refitted=0, models=records)
    plan['model_import_receipt'] = str(tmp_path/'import.private.json');refresh_inventory(plan, inventory)
    monkeypatch.setattr(run, '_implementation_sha256', lambda: NEW)
    class Inputs:
        directory = prepared
        manifest = dict(trait_count=2)
        closed = False
        def trait(self, *args, **kwargs):
            raise AssertionError('reuse must not prepare any trait for a new fit')
        def close(self): self.closed = True
    monkeypatch.setattr(run, 'PhewasInputs', lambda *args: Inputs())
    monkeypatch.setattr(run, 'prepare_phewas_model', lambda *args, **kwargs: pytest.fail('reuse refitted a model'))
    monkeypatch.setattr(run.SparseKinshipData, 'load_npz', lambda *args: pytest.fail('reuse loaded a fresh fitting GRM'))
    return plan, inventory, proof


def test_reuse_verifies_old_source_without_refit_or_metadata_rewrite(reuse):
    plan, inventory, _ = reuse
    before = {Path(file['path']): digest_file(file['path']) for record in inventory['models'] for file in record['files']}
    assert run._fit_binding(plan)['source_implementation_sha256'] == OLD
    result = run.fit_models(plan, device='cpu')
    assert result['mode'] == 'verified_reuse'
    assert result['model_fit_source_implementation_sha256'] == OLD
    assert result['consumer_source_implementation_sha256'] == NEW
    assert len(run._all_models(plan, 1)) == 2
    assert {path: digest_file(path) for path in before} == before


@pytest.mark.parametrize('problem', ['missing', 'wrong_target'])
def test_reuse_missing_or_rebound_link_hardfails_before_any_fit(reuse, problem):
    plan, inventory, _ = reuse
    link = Path(inventory['models'][0]['symlink']);link.unlink()
    if problem == 'wrong_target':
        link.symlink_to(inventory['models'][1]['target'], target_is_directory=True)
    with pytest.raises(ValueError, match='missing|link binding'):
        run.fit_models(plan, device='cpu')
    assert not (Path(plan['output_directory'])/'fit_worker_0.complete.private.json').exists()


@pytest.mark.parametrize('problem', ['consumer', 'fingerprint', 'aggregate', 'component', 'qualification'])
def test_reuse_rejects_wrong_scientific_proof_even_with_rebound_receipt_sha(reuse, problem):
    plan, inventory, proof = reuse
    if problem == 'consumer': proof['consumer_source_implementation_sha256'] = 'd'*64
    elif problem == 'fingerprint': proof['components']['binary_null.py:complete_module']['new_fingerprint'] = 'd'*64
    elif problem == 'aggregate': proof['old_null_fit_component_fingerprint'] = 'd'*64
    elif problem == 'component': proof['components'].pop('phewas_inputs.py:complete_module')
    else: proof['existing_models_reuse']['missing_reuse_target_must_hard_fail'] = False
    plan['null_fit_compatibility_sha256'] = write(Path(plan['null_fit_compatibility_receipt']), proof)
    inventory['null_fit_compatibility_sha256'] = plan['null_fit_compatibility_sha256'];refresh_inventory(plan, inventory)
    with pytest.raises(ValueError, match='compatibility|fingerprint'):
        run._fit_binding(plan)


@pytest.mark.parametrize('problem', ['input', 'fit_source', 'cohort', 'refitted', 'coverage', 'destination'])
def test_reuse_rejects_wrong_inventory_with_valid_json_hash(reuse, problem):
    plan, inventory, _ = reuse
    if problem == 'input': inventory['prepared_inputs_manifest_sha256'] = 'd'*64
    elif problem == 'fit_source': inventory['model_fit_source_implementation_sha256'] = NEW
    elif problem == 'cohort': inventory['cohort_sha256'] = None
    elif problem == 'refitted': inventory['models_refitted'] = 1
    elif problem == 'coverage': inventory['models'].pop()
    else: inventory['models_directory'] = inventory['from_models_directory']
    refresh_inventory(plan, inventory)
    with pytest.raises(ValueError, match='binding|coverage|directories'):
        run._fit_binding(plan)


@pytest.mark.parametrize('exit_code', [1, False, '0', None])
def test_nonzero_original_fit_exit_cannot_be_hidden_by_success_flag(reuse, exit_code):
    plan, inventory, _ = reuse
    path = Path(inventory['from_run_directory'])/'outputs/worker_0.fit.supervisor_exit.anonymous.json'
    receipt = json.loads(path.read_text());receipt['exit_code'] = exit_code
    inventory['fit_supervisor_exit_sha256']['0'] = write(path, receipt);refresh_inventory(plan, inventory)
    with pytest.raises(ValueError, match='successfully exit'):
        run._fit_binding(plan)


@pytest.mark.parametrize('problem', ['array_mutated', 'array_omitted'])
def test_reuse_array_inventory_is_complete_and_immutable(reuse, problem):
    plan, inventory, _ = reuse
    file = next(file for file in inventory['models'][0]['files'] if file['path'].endswith('/x.npy'))
    if problem == 'array_mutated':
        path = Path(file['path']);raw = bytearray(path.read_bytes());raw[-1] ^= 1;path.write_bytes(raw)
    else:
        inventory['models'][0]['files'].remove(file);refresh_inventory(plan, inventory)
    with pytest.raises(ValueError, match='file identity|every fitted-state array'):
        run.fit_models(plan, device='cpu')


def test_current_receipt_cannot_relabel_or_omit_a_reused_trait(reuse):
    plan, _, _ = reuse
    run.fit_models(plan, device='cpu')
    path = Path(plan['output_directory'])/'fit_worker_0.complete.private.json'
    receipt = json.loads(path.read_text());receipt['model_fit_source_implementation_sha256'] = NEW
    write(path, receipt)
    with pytest.raises(ValueError, match='provenance coverage'):
        run._all_models(plan, 1)


def test_override_requires_reuse_and_current_source_hash(reuse, monkeypatch):
    plan, _, _ = reuse
    plan['reuse_models_only'] = False
    with pytest.raises(ValueError, match='reuse-only'):
        run._fit_binding(plan)
    plan['reuse_models_only'] = True
    monkeypatch.setattr(run, '_implementation_sha256', lambda: 'd'*64)
    with pytest.raises(ValueError, match='implementation binding'):
        run._fit_binding(plan)


def test_compatibility_checksum_changes_are_not_trusted(reuse):
    plan, _, _ = reuse
    path = Path(plan['null_fit_compatibility_receipt']);path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='checksum'):
        run.fit_models(plan, device='cpu')


def test_original_fit_receipt_missing_trait_cannot_authorize_reuse(reuse):
    plan, inventory, _ = reuse
    path = Path(inventory['from_run_directory'])/'outputs/fit_worker_0.complete.private.json'
    receipt = json.loads(path.read_text());receipt['models'].pop()
    inventory['fit_complete_receipt_sha256']['0'] = write(path, receipt);refresh_inventory(plan, inventory)
    with pytest.raises(ValueError, match='model coverage'):
        run._fit_binding(plan)


def test_model_metadata_source_and_trait_are_not_relabelled_by_import(reuse):
    plan, inventory, _ = reuse
    path = Path(inventory['models'][0]['target'])/'model.private.json'
    # Retain valid inventory binding while changing the actual stored metadata:
    # immutable identity and metadata SHA independently reject the mutation.
    value = json.loads(path.read_text());value['fit']['source_implementation_sha256'] = NEW
    write(path, value)
    with pytest.raises(ValueError, match='file identity|checksum'):
        run.fit_models(plan, device='cpu')


def test_current_receipt_wrong_model_path_or_coverage_cannot_pass_barrier(reuse):
    plan, inventory, _ = reuse
    run.fit_models(plan, device='cpu')
    path = Path(plan['output_directory'])/'fit_worker_0.complete.private.json'
    receipt = json.loads(path.read_text())
    receipt['models'][0]['path'] = inventory['models'][1]['symlink']
    write(path, receipt)
    with pytest.raises(ValueError, match='import inventory'):
        run._all_models(plan, 1)


def test_unchanged_fresh_fit_binding_remains_default():
    plan = dict(kinship_sha256='kinship', source_implementation_sha256='current', cohort_sha256=None)
    assert run._fit_binding(plan) == dict(kinship_sha256='kinship', source_implementation_sha256='current',
                                        cohort_sha256=None, continuous_transform='paper')


def refresh_scoped_proof(plan, inventory, proof):
    joined = hashlib.sha256()
    for name, record in sorted(proof['components'].items()):
        joined.update(name.encode())
        joined.update(bytes.fromhex(record['old_fingerprint']))
    proof['old_null_fit_component_fingerprint'] = joined.hexdigest()
    proof['new_null_fit_component_fingerprint'] = joined.hexdigest()
    plan['null_fit_compatibility_sha256'] = write(Path(plan['null_fit_compatibility_receipt']), proof)
    inventory['null_fit_compatibility_sha256'] = plan['null_fit_compatibility_sha256']
    refresh_inventory(plan, inventory)


def existing_input_scope(reuse):
    plan, inventory, proof = reuse
    proof['components'].pop('phewas_inputs.py:complete_module')
    for name in ('PhewasInputs', 'sha256', '_discard', 'consumer_imports'):
        proof['components']['phewas_inputs.py:'+name] = dict(
            old_fingerprint='c'*64, new_fingerprint='c'*64, identical=True)
    proof['prepared_input_reuse_scope'] = dict(
        kind='fixed_existing_prepared_inputs',
        model_fit_source_implementation_sha256=OLD,
        consumer_source_implementation_sha256=NEW,
        prepared_inputs_manifest_sha256=plan['prepared_inputs_manifest_sha256'],
        cohort_sha256=plan['cohort_sha256'], kinship_sha256=plan['kinship_sha256'],
        continuous_transform=plan['continuous_transform'], input_preparation_disabled=True,
        models_refitted=0, consumption_components_identical=True,
        complete_input_preparation_module_changed=True)
    proof['inputs_module_change'] = dict(old_fingerprint='d'*64, new_fingerprint='e'*64,
        old_sha256='f'*64, new_sha256='1'*64, identical=False)
    refresh_scoped_proof(plan, inventory, proof)
    return plan, inventory, proof


def test_fixed_existing_inputs_reuse_phase_never_prepares_refits_or_rewrites_models(reuse, monkeypatch):
    plan, inventory, _ = existing_input_scope(reuse)
    monkeypatch.setattr(run, 'prepare_phewas_inputs', lambda **kwargs: pytest.fail('reuse prepared inputs'))
    before = {Path(file['path']): digest_file(file['path'])
              for record in inventory['models'] for file in record['files']}
    # This is the fit-only phase called by a dispatched worker; the fixture
    # forbids trait preparation, fresh GRM loading and every fitting function.
    result = run.fit_models(plan, device='cpu')
    assert result['mode'] == 'verified_reuse' and result['errors'] == []
    assert result['model_fit_source_implementation_sha256'] == OLD
    assert result['consumer_source_implementation_sha256'] == NEW
    assert len(run._all_models(plan, 1)) == 2
    assert {path: digest_file(path) for path in before} == before


@pytest.mark.parametrize('key,value', [
    ('kind', 'general_changed_preparation'),
    ('model_fit_source_implementation_sha256', NEW),
    ('consumer_source_implementation_sha256', OLD),
    ('prepared_inputs_manifest_sha256', 'd'*64),
    ('cohort_sha256', None), ('kinship_sha256', 'd'*64),
    ('continuous_transform', 'none'), ('input_preparation_disabled', False),
    ('models_refitted', 1), ('models_refitted', False),
    ('consumption_components_identical', False),
    ('complete_input_preparation_module_changed', False),
])
def test_fixed_existing_input_scope_rejects_shifted_or_reinterpreted_inputs(reuse, key, value):
    plan, inventory, proof = existing_input_scope(reuse)
    proof['prepared_input_reuse_scope'][key] = value
    refresh_scoped_proof(plan, inventory, proof)
    with pytest.raises(ValueError, match='prepared-input reuse scope'):
        run.fit_models(plan, device='cpu')
    assert not (Path(plan['output_directory'])/'fit_worker_0.complete.private.json').exists()


@pytest.mark.parametrize('problem', ['null', 'missing_binding', 'missing_change', 'same_ast',
    'same_sha', 'claimed_identical', 'bad_ast_digest', 'bad_source_digest', 'full_module_claim'])
def test_fixed_existing_input_proof_requires_explicit_scoped_changed_module_evidence(reuse, problem):
    plan, inventory, proof = existing_input_scope(reuse)
    if problem == 'null':
        proof['prepared_input_reuse_scope'] = None
    elif problem == 'missing_binding':
        proof['prepared_input_reuse_scope'].pop('cohort_sha256')
    elif problem == 'missing_change':
        proof.pop('inputs_module_change')
    elif problem == 'same_ast':
        proof['inputs_module_change']['new_fingerprint'] = proof['inputs_module_change']['old_fingerprint']
    elif problem == 'same_sha':
        proof['inputs_module_change']['new_sha256'] = proof['inputs_module_change']['old_sha256']
    elif problem == 'claimed_identical':
        proof['inputs_module_change']['identical'] = True
    elif problem == 'bad_ast_digest':
        proof['inputs_module_change']['old_fingerprint'] = 'not-a-hash'
    elif problem == 'bad_source_digest':
        proof['inputs_module_change']['new_sha256'] = 'z'*64
    else:
        proof['components']['phewas_inputs.py:complete_module'] = dict(
            old_fingerprint='c'*64, new_fingerprint='c'*64, identical=True)
    refresh_scoped_proof(plan, inventory, proof)
    with pytest.raises(ValueError, match='prepared-input reuse'):
        run._fit_binding(plan)


@pytest.mark.parametrize('component', ['PhewasInputs', 'sha256', '_discard', 'consumer_imports'])
def test_fixed_existing_input_proof_requires_every_unchanged_consumption_component(reuse, component):
    plan, inventory, proof = existing_input_scope(reuse)
    proof['components'].pop('phewas_inputs.py:'+component)
    refresh_scoped_proof(plan, inventory, proof)
    with pytest.raises(ValueError, match='required scientific components'):
        run._fit_binding(plan)


def test_fixed_existing_input_proof_rejects_changed_consumer_even_with_rebound_receipts(reuse):
    plan, inventory, proof = existing_input_scope(reuse)
    proof['components']['phewas_inputs.py:PhewasInputs']['new_fingerprint'] = 'd'*64
    refresh_scoped_proof(plan, inventory, proof)
    with pytest.raises(ValueError, match='scientific component fingerprints differ'):
        run._fit_binding(plan)


@pytest.mark.parametrize('key', ['prepared_inputs_manifest_sha256', 'cohort_sha256', 'kinship_sha256'])
def test_fixed_existing_input_scope_cannot_rebind_original_fit_plan_to_new_inputs(reuse, key):
    plan, inventory, proof = existing_input_scope(reuse)
    plan[key] = inventory[key] = proof['prepared_input_reuse_scope'][key] = '2'*64
    refresh_scoped_proof(plan, inventory, proof)
    with pytest.raises(ValueError, match='original fit plan scientific/source bindings'):
        run._fit_binding(plan)


@pytest.mark.parametrize('file', ['manifest', 'cohort', 'kinship'])
def test_fixed_existing_input_scope_still_verifies_current_artifact_checksums(reuse, file):
    plan, _, _ = existing_input_scope(reuse)
    path = (Path(plan['prepared_inputs_directory'])/'manifest.private.json' if file == 'manifest'
            else Path(plan['cohort_rows' if file == 'cohort' else 'kinship_npz']))
    path.write_bytes(path.read_bytes()+b'corrupted')
    with pytest.raises(ValueError, match='prepared input manifest|GRM/cohort checksum'):
        run.fit_models(plan, device='cpu')


def test_fixed_existing_input_scope_cannot_authorize_new_model_fits(reuse):
    plan, _, _ = existing_input_scope(reuse)
    plan['reuse_models_only'] = False
    with pytest.raises(ValueError, match='reuse-only'):
        run.fit_models(plan, device='cpu')
