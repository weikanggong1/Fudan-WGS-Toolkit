"""Binary API, identity reuse and actual fitted-sample count contracts."""
import inspect
import json

import numpy as np
import pytest

import fudan_wgs_toolkit as package
from fudan_wgs_toolkit import run


def inputs():
    pairs = np.asarray([['0', str(i)] for i in range(8)])
    p = run.NumericCSV(pairs, ('trait_a', 'trait_b'), np.asarray([
        [1, 0], [0, 1], [np.nan, 0], [1, np.nan],
        [0, 1], [1, 0], [0, 1], [0, 0]], dtype=np.float64))
    c = run.NumericCSV(pairs, ('age',), np.arange(8, dtype=np.float64)[:, None])
    return pairs, p, c


def test_old_generic_entry_is_removed_and_new_entry_is_binary_only():
    assert not hasattr(package, 'run_WGS_all') and not hasattr(run, 'run_WGS_all')
    signature = inspect.signature(package.run_WGS_all_binanry)
    assert signature.parameters['impute_missing'].default is True
    assert signature.parameters['phenotype_covariates'].default is None
    assert signature.parameters['trait_batch_size'].default == 32
    for retired in ('rm_nan', 'phenotype_families', 'transform', 'continuous_transform',
                    'long_mask_method', 'null_fit_mode', 'gpu_ids'):
        assert retired not in signature.parameters
        with pytest.raises(TypeError):
            package.run_WGS_all_binanry('p', 'c', 'g', output_directory='out', **{retired: None})


@pytest.mark.parametrize('impute', [True, False])
def test_alignment_cache_reuses_only_the_exact_sample_design(impute):
    pairs, p, c = inputs()
    cache = run.AlignedInputCache(p, c, pairs)
    results = []
    for name in p.names:
        options = dict(reduce_redundant=True, rm_nan=impute, family='binomial', binary_only=True)
        expected = run.aligned_phenotype(p, c, pairs, name, **options)
        actual = run.aligned_phenotype(p, c, pairs, name, alignment_cache=cache, **options)
        for i in (0, 1, 2):
            np.testing.assert_array_equal(actual[i], expected[i])
        assert actual[3:] == expected[3:]
        assert actual[4]['analysis_cases'] + actual[4]['analysis_controls'] == len(actual[1])
        results.append(actual)
    assert cache.summary()['design_builds'] == (1 if impute else 2)
    assert cache.summary()['retained_bytes'] <= cache.summary()['capacity_bytes']
    if impute:
        assert results[0][2] is results[1][2]
    else:
        assert not np.array_equal(results[0][0], results[1][0])
    assert not results[0][2].flags.writeable


def test_alignment_rejects_foreign_cache_order_and_is_bounded():
    pairs, p, c = inputs()
    cache = run.AlignedInputCache(p, c, pairs, max_bytes=1)
    run.aligned_phenotype(p, c, pairs, 'trait_a', rm_nan=True,
                          family='binomial', binary_only=True, alignment_cache=cache)
    assert cache.summary()['retained_bytes'] <= 1
    with pytest.raises(ValueError, match='different inputs'):
        run.aligned_phenotype(p, c, pairs.copy(), 'trait_b', alignment_cache=cache)


def test_complete_report_counts_are_fitted_counts_and_written_to_csv(tmp_path):
    for index, counts in enumerate(((2, 4, 1), (3, 3, 0))):
        cases, controls, imputed = counts
        directory = tmp_path / 'models' / f'trait_{index:04d}'
        directory.mkdir(parents=True)
        fit = dict(phenotype=f'trait_{index}', analysis_cases=cases, analysis_controls=controls,
                   available_cases=cases, available_controls=controls-imputed, imputed_controls=imputed)
        (directory / 'model.private.json').write_text(json.dumps(dict(family='binomial', n=6, fit=fit)))
    (tmp_path / 'plan.private.json').write_text(json.dumps(dict(trait_count=2)))
    report = run._binary_sample_counts(tmp_path)
    assert [row['samples'] for row in report] == [6, 6]
    assert report[0]['controls'] == 4 and report[0]['imputed_controls'] == 1
    assert (tmp_path / 'phenotype_sample_counts.csv').is_file()
    broken = tmp_path / 'models' / 'trait_0000' / 'model.private.json'
    data = json.loads(broken.read_text())
    data['fit']['analysis_controls'] = 2
    broken.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='actual analysis samples'):
        run._binary_sample_counts(tmp_path)


@pytest.mark.parametrize('bad', [None, 0, 1, 'true', {}])
def test_missing_policy_is_strict_boolean_before_inputs(tmp_path, bad):
    with pytest.raises(ValueError, match='impute_missing'):
        package.run_WGS_all_binanry('missing', 'missing', 'missing',
            output_directory=tmp_path / 'out', impute_missing=bad)
    assert not (tmp_path / 'out').exists()
