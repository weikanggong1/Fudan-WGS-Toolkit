"""Actual plan/default binding contracts, without GPU or model fitting."""
import copy
import csv
import inspect
import json

import pytest

from fudan_wgs_toolkit import continuous_run, phewas_run, prepared_association
from fudan_wgs_toolkit.phewas_trait_csv import PerTraitCheckpointCSVWriter
from test_phewas_public_options import api_inputs


PREPARATION_FLAGS = ('_continuous_shared_panels', '_continuous_resident_preparation',
                     '_continuous_covariance_tiles', '_continuous_live_resident')
FULL_FLAGS = ('_continuous_tail_groups', '_continuous_complete_spectrum_batch',
              '_continuous_spectrum_reuse')
PARAMETERS = '_continuous_complete_spectrum_batch_parameters'


def assert_defaults(configuration, method):
    assert all(configuration[key] is True for key in PREPARATION_FLAGS)
    assert all(configuration[key] is (method == 'full') for key in FULL_FLAGS)
    assert configuration[PARAMETERS] == (dict(maximum_variants=1536, weight_batch_size=32,
        relation_workspace_bytes=256*2**10) if method == 'full' else None)
    assert configuration['analysis_options']['long_mask_method'] == method
    assert inspect.signature(PerTraitCheckpointCSVWriter).parameters['parallel_fsync'].default is False


@pytest.mark.parametrize('method', ['full', 'fastskat', 'liu'])
def test_execution_copy_preserves_scientific_schedule_filters_options_and_order(method):
    original = dict(analysis_options=dict(long_mask_method=method, long_mask_threshold=5000,
        long_mask_rank=512, seed=91, variant_type='variant', rv_num_cutoff=2),
        chromosomes=[dict(name='22', jobs=[dict(kind='coding', arguments=dict(category='plof'), checkpoint_job_index=7)]),
                     dict(name='21', jobs=[dict(kind='ncrna', arguments=dict(start=3, end=9), checkpoint_job_index=8)])],
        annotation_names=['annotation_a'], filters={'pass': True}, qc_path='annotation/filter',
        matmul_mode='tf32', source_implementation_sha256='source')
    frozen = copy.deepcopy(original)
    actual = continuous_run._continuous_performance_configuration(original)
    assert actual is not original and original == frozen
    for key in original:
        assert actual[key] is original[key]  # No reordering/copy of the immutable scientific tree.
    assert_defaults(actual, method)
    assert continuous_run._continuous_performance_configuration(actual) == actual
    if method == 'full':
        second = continuous_run._continuous_performance_configuration(original)
        actual[PARAMETERS]['maximum_variants'] = 1
        assert second[PARAMETERS]['maximum_variants'] == 1536 and PARAMETERS not in original


def continuous_csv(inputs):
    with inputs['phenotype_csv'].open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'trait_a', 'trait_b'])
        writer.writerows([[301+i, .25+i, 7.5-i*.7] for i in range(6)])


@pytest.mark.parametrize('contract', ['continuous', 'mixed'])
@pytest.mark.parametrize('method', ['full', 'fastskat', 'liu'])
def test_population_plan_applies_defaults_only_to_explicit_continuous_contract(
        api_inputs, monkeypatch, contract, method):
    if contract == 'continuous': continuous_csv(api_inputs)
    monkeypatch.setattr(phewas_run, '_implementation_sha256', lambda: 'fixed_contract_source')
    expected = phewas_run.build_configuration(api_inputs['cache_directory'], analyses=('coding',),
        long_mask_method=method, long_mask_threshold=9001, gene_variant_type='variant')
    plan = phewas_run.prepare_phewas_run(**api_inputs, analyses=('coding',), input_contract=contract,
        long_mask_method=method, long_mask_threshold=9001, gene_variant_type='variant')
    scientific = {key: value for key, value in plan['configuration'].items()
                  if key not in (*PREPARATION_FLAGS, *FULL_FLAGS, PARAMETERS)}
    assert scientific == expected
    assert plan['configuration']['analysis_options']['long_mask_threshold'] == 9001
    if contract == 'continuous': assert_defaults(plan['configuration'], method)
    else: assert all(key not in plan['configuration'] for key in (*PREPARATION_FLAGS, *FULL_FLAGS, PARAMETERS))
    persisted = json.loads((api_inputs['output_directory']/'plan.private.json').read_text())
    assert persisted['configuration'] == plan['configuration']


def test_population_resume_rejects_changed_execution_flags_in_existing_binding(api_inputs, monkeypatch):
    continuous_csv(api_inputs)
    monkeypatch.setattr(phewas_run, '_implementation_sha256', lambda: 'fixed_contract_source')
    options = dict(api_inputs, analyses=('coding',), input_contract='continuous')
    first = phewas_run.prepare_phewas_run(**options)
    resumed = phewas_run.prepare_phewas_run(**options, resume=True)
    assert first['configuration'] == resumed['configuration']
    path = api_inputs['output_directory']/'plan.private.json'
    previous = json.loads(path.read_text())
    previous['configuration']['_continuous_live_resident'] = False
    path.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match='resume plan scientific/source bindings differ'):
        phewas_run.prepare_phewas_run(**options, resume=True)


class PlanReachedBeforeFitting(Exception):
    pass


@pytest.mark.parametrize('contract', ['continuous', 'mixed'])
@pytest.mark.parametrize('method', ['full', 'fastskat'])
def test_prepared_plan_actual_entry_binds_defaults_before_any_model_fit_or_dispatch(
        tmp_path, monkeypatch, contract, method):
    from test_continuous_prepared import continuous_fixture, cpu_single_control, call
    from fudan_wgs_toolkit.run import _run_wgs_association
    root, phenotype, covariate, _ = continuous_fixture(tmp_path)
    cpu_single_control(monkeypatch)  # Inject only the unavailable CUDA accounting boundary.
    output, reached = tmp_path/'output', []
    def stop(plan, *args, **kwargs):
        reached.append(copy.deepcopy(plan))
        raise PlanReachedBeforeFitting
    monkeypatch.setattr(prepared_association, '_fit_models', stop)
    options = dict(rm_nan=False, long_mask_method=method, long_mask_threshold=9001)
    api = continuous_run.run_WGS_continuous if contract == 'continuous' else _run_wgs_association
    with pytest.raises(PlanReachedBeforeFitting):
        call(api, root, phenotype, covariate, output, **options)
    plan = json.loads((output/'plan.private.json').read_text())
    assert reached == [plan] and not list((output/'models').iterdir())
    assert plan['configuration']['analysis_options']['long_mask_threshold'] == 9001
    if contract == 'continuous': assert_defaults(plan['configuration'], method)
    else: assert all(key not in plan['configuration'] for key in (*PREPARATION_FLAGS, *FULL_FLAGS, PARAMETERS))
    with pytest.raises(PlanReachedBeforeFitting):
        call(api, root, phenotype, covariate, output, **options, resume=True)
    assert len(reached) == 2
    if contract == 'continuous':
        plan['configuration']['_continuous_resident_preparation'] = False
        (output/'plan.private.json').write_text(json.dumps(plan))
        with pytest.raises(ValueError, match='resume prepared association source/input/schedule binding differs'):
            call(api, root, phenotype, covariate, output, **options, resume=True)
        assert len(reached) == 2  # Mismatched defaults reject before fitting or launch.
