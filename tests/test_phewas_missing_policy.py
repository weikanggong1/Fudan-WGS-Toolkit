"""Bound phenotype filling and available-sample contracts on real CSV parsing."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).parents[1] / "fudan_wgs_toolkit" / "phewas_inputs.py"
spec = importlib.util.spec_from_file_location("phewas_missing_policy_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _table(path, rows):
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)


def _csv_inputs(tmp_path):
    axis, phenotypes, covariates = (tmp_path / name for name in ("axis.npy", "p.csv", "c.csv"))
    np.save(axis, [105, 101, 104, 102, 106, 103, 107, 108], allow_pickle=False)
    _table(phenotypes, [["FID", "IID", "a", "b"],
        [0, 101, 2, 1], [0, 102, "", ""], [0, 103, 100, 2],
        [0, 104, 4, 0], [0, 105, "", -3], [0, 106, 8, 1], [0, 107, "", ""]])
    _table(covariates, [["FID", "IID", "age", "extra"],
        [0, 101, 10, 1], [0, 102, 20, 2], [0, 103, "", 3],
        [0, 104, 40, 4], [0, 105, 50, ""], [0, 106, 60, 6], [0, 108, 80, 8]])
    return dict(phenotypes=phenotypes, covariates=covariates, sample_ids=axis,
        output=tmp_path / "prepared", chunk_rows=2,
        phenotype_families={"a": "gaussian", "b": "binomial"},
        phenotype_covariates={"a": ["age"], "b": ["extra"]})


def test_default_fill_uses_each_selected_covariate_population_and_retains_raw(tmp_path):
    args = _csv_inputs(tmp_path)
    summary = module.prepare_phewas_inputs(**args)
    with_rows = module.PhewasInputs(args["output"])
    try:
        a, b = with_rows.trait(0), with_rows.trait(1)
        assert summary["rm_nan"] is True
        assert a["sample_ids"].tolist() == [105, 101, 104, 102, 106]
        assert a["available_n"] == 3 and a["mother_n"] == 5 and a["imputed_n"] == 2
        assert a["imputation_mean"] == 14 / 3
        np.testing.assert_array_equal(a["y_raw"], [14 / 3, 2, 4, 14 / 3, 8])
        assert a["covariates"].tolist() == [[50], [10], [40], [20], [60]]
        assert np.isnan(with_rows.phenotypes[0, 0]) and np.isnan(with_rows.phenotypes[3, 0])
        # Missing unused age does not exclude this trait; missing selected
        # extra excludes 105; a numeric non-1 label becomes a control.
        assert b["sample_ids"].tolist() == [101, 104, 102, 106, 103]
        assert b["y_raw"].tolist() == [1, 0, 0, 1, 0]
        assert b["available_n"] == 4 and b["mother_n"] == 5 and b["imputed_n"] == 1
        assert b["recoded_nonbinary_n"] == 1
        quality = with_rows.metadata["quality"]
        assert quality[0]["available_n"] == 3 and quality[0]["imputation_mean"] == 14 / 3
        assert quality[1]["analysis_cases"] == 2 and quality[1]["analysis_controls"] == 3
        assert with_rows.manifest["trait_options"]["rm_nan"] is True
        assert with_rows.manifest["missing_policy"] == a["missing_policy"]
        assert summary["numeric_csv_parses_per_input"] == 1
    finally:
        with_rows.close()


def test_selected_cohort_recomputes_mean_and_keeps_cache_order(tmp_path):
    args = _csv_inputs(tmp_path)
    module.prepare_phewas_inputs(**args)
    data = module.PhewasInputs(args["output"])
    try:
        a = data.trait(0, cohort_indices=np.array([3, 2, 0]))
        assert a["sample_ids"].tolist() == [105, 104, 102]
        assert a["available_n"] == 1 and a["mother_n"] == 3 and a["imputed_n"] == 2
        assert a["imputation_mean"] == 4
        assert a["y_raw"].tolist() == [4, 4, 4]
        with pytest.raises(ValueError, match="finite eligible phenotype mean"):
            data.trait(0, cohort_indices=np.array([3]))
    finally:
        data.close()


def test_empty_covariates_do_not_require_absent_covariate_rows_or_create_absent_phenotypes(tmp_path):
    args = _csv_inputs(tmp_path)
    args["phenotype_covariates"]["b"] = []
    module.prepare_phewas_inputs(**args)
    data = module.PhewasInputs(args["output"])
    try:
        b = data.trait(1)
        assert b["sample_ids"].tolist() == [105, 101, 104, 102, 106, 103, 107]
        assert b["y_raw"].tolist() == [0, 1, 0, 0, 1, 0, 0]
        assert b["covariates"].shape == (7, 0)
        assert b["mother_n"] == 7 and b["imputed_n"] == 2
        assert b["recoded_nonbinary_n"] == 2
        assert 108 not in b["sample_ids"]
    finally:
        data.close()


def test_available_policy_drops_only_own_missing_values_and_preserves_finite_values(tmp_path):
    args = _csv_inputs(tmp_path)
    args["rm_nan"] = False
    args["phenotype_families"]["b"] = "gaussian"
    module.prepare_phewas_inputs(**args)
    data = module.PhewasInputs(args["output"])
    try:
        a, b = data.trait(0), data.trait(1)
        assert a["sample_ids"].tolist() == [101, 104, 106]
        assert a["y_raw"].tolist() == [2, 4, 8]
        assert b["sample_ids"].tolist() == [101, 104, 106, 103]
        assert b["y_raw"].tolist() == [1, 0, 1, 2]
        assert a["imputed_n"] == 0 and a["imputation_mean"] is None
        assert a["available_n"] == 3 and a["rm_nan"] is False
    finally:
        data.close()


@pytest.mark.parametrize("rm_nan", [False, True])
@pytest.mark.parametrize("family", ["gaussian", "binomial"])
@pytest.mark.parametrize("invalid", [np.inf, -np.inf])
def test_infinity_is_not_a_nan_or_a_control(rm_nan, family, invalid):
    with pytest.raises(ValueError, match="Infinite phenotype"):
        module.apply_phenotype_missing_policy([1, invalid], family=family, rm_nan=rm_nan,
            covariate_valid=np.array([True, False]))


@pytest.mark.parametrize("invalid", [None, 0, 1, "true"])
def test_rm_nan_requires_explicit_boolean(invalid, tmp_path):
    args = _csv_inputs(tmp_path)
    with pytest.raises(ValueError, match="rm_nan must be a boolean"):
        module.prepare_phewas_inputs(**args, rm_nan=invalid)
    assert not args["output"].exists()


def test_binary_recode_and_available_exact01_validation_do_not_mutate_input():
    values = np.array([np.nan, -3, 1, 0, 2], dtype=np.float64)
    original = values.copy()
    selected = np.array([True, True, True, True, False])
    policy = module.apply_phenotype_missing_policy(values, family="binomial", covariate_valid=selected)
    assert policy["values"].tolist() == [0, 0, 1, 0]
    assert policy["sample_indices"].tolist() == [0, 1, 2, 3]
    assert policy["available_n"] == 3 and policy["recoded_nonbinary_n"] == 1
    np.testing.assert_array_equal(values, original)
    with pytest.raises(ValueError, match="rm_nan=False"):
        module.apply_phenotype_missing_policy(values, family="binomial", rm_nan=False,
            covariate_valid=selected)


def test_auto_family_is_decided_before_recode_and_explicit_gaussian_is_retained(tmp_path):
    args = _csv_inputs(tmp_path)
    args["phenotype_families"] = "auto"
    module.prepare_phewas_inputs(**args)
    data = module.PhewasInputs(args["output"])
    try:
        # Non-01 source b, including a value outside its covariate population,
        # prevents inference as binary. Filling does not redefine the family.
        assert data.trait(1)["family"] == "gaussian"
        assert 2 in data.trait(1)["y_raw"]
    finally:
        data.close()


def test_auto_exact01_with_nan_remains_binary_when_filling(tmp_path):
    args = _csv_inputs(tmp_path)
    args["phenotype_families"] = "auto"
    _table(args["phenotypes"], [["FID", "IID", "a", "b"],
        [0, 101, 2, 1], [0, 102, "", ""], [0, 103, 100, 0],
        [0, 104, 4, 0], [0, 105, "", 0], [0, 106, 8, 1]])
    module.prepare_phewas_inputs(**args)
    data = module.PhewasInputs(args["output"])
    try:
        b = data.trait(1)
        assert b["family"] == "binomial"
        assert b["y_raw"].tolist() == [1, 0, 0, 1, 0]
        assert b["imputed_n"] == 1 and b["imputation_mean"] is None
    finally:
        data.close()


def test_explicit_gaussian_exact01_column_is_filled_with_mean_not_reclassified(tmp_path):
    args = _csv_inputs(tmp_path)
    args["phenotype_families"]["b"] = "gaussian"
    _table(args["phenotypes"], [["FID", "IID", "a", "b"],
        [0, 101, 2, 1], [0, 102, "", ""], [0, 103, 100, 0],
        [0, 104, 4, 0], [0, 105, "", 0], [0, 106, 8, 1]])
    module.prepare_phewas_inputs(**args)
    data = module.PhewasInputs(args["output"])
    try:
        b = data.trait(1)
        assert b["family"] == "gaussian" and b["imputation_mean"] == .5
        assert b["y_raw"].tolist() == [1, 0, .5, 1, 0]
    finally:
        data.close()


def test_explicit_binary_all_nan_is_controls_without_fabricated_gaussian_mean(tmp_path):
    args = _csv_inputs(tmp_path)
    _table(args["phenotypes"], [["FID", "IID", "a", "b"], [0, 101, 2, ""]])
    module.prepare_phewas_inputs(**args)
    data = module.PhewasInputs(args["output"])
    try:
        b = data.trait(1)
        assert b["family"] == "binomial" and b["y_raw"].tolist() == [0]
        assert b["available_n"] == 0 and b["mother_n"] == 1 and b["imputed_n"] == 1
        assert b["imputation_mean"] is None
    finally:
        data.close()


def test_all_nan_gaussian_has_no_fabricated_mean_or_completion(tmp_path):
    args = _csv_inputs(tmp_path)
    _table(args["phenotypes"], [["FID", "IID", "a", "b"], [0, 101, "", 1]])
    with pytest.raises(ValueError, match="finite eligible phenotype mean"):
        module.prepare_phewas_inputs(**args)
    assert not (args["output"] / "manifest.private.json").exists()
    assert json.loads((args["output"] / "status.anonymous.json").read_text())["status"] == "failed"


def test_empty_eligible_population_returns_empty_without_mean():
    policy = module.apply_phenotype_missing_policy([np.nan, 2], family="gaussian",
        covariate_valid=np.array([False, False]))
    assert policy["values"].size == 0 and policy["sample_indices"].size == 0
    assert policy["available_n"] == 0 and policy["mother_n"] == 0
    assert policy["imputation_mean"] is None


def _rewrite_manifest(output, manifest):
    path = output / "manifest.private.json"
    path.write_text(json.dumps(manifest))
    status = json.loads((output / "status.anonymous.json").read_text())
    status["manifest_sha256"] = module.sha256(path)
    (output / "status.anonymous.json").write_text(json.dumps(status))


def test_missing_policy_binding_rejects_reinterpreting_existing_raw_inputs(tmp_path):
    args = _csv_inputs(tmp_path)
    module.prepare_phewas_inputs(**args)
    manifest = json.loads((args["output"] / "manifest.private.json").read_text())
    manifest["rm_nan"] = False
    _rewrite_manifest(args["output"], manifest)
    with pytest.raises(ValueError, match="policy binding"):
        module.PhewasInputs(args["output"])


def test_legacy_manifest_without_missing_policy_keeps_available_sample_semantics(tmp_path):
    args = _csv_inputs(tmp_path)
    args["rm_nan"] = False
    args["phenotype_families"]["b"] = "gaussian"
    module.prepare_phewas_inputs(**args)
    output = args["output"]
    metadata_path = output / "traits.private.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["rm_nan"], metadata["missing_policy"]
    metadata_path.write_text(json.dumps(metadata))
    manifest = json.loads((output / "manifest.private.json").read_text())
    del manifest["rm_nan"], manifest["missing_policy"], manifest["trait_options"]["rm_nan"]
    manifest["artifacts"]["traits.private.json"] = {"bytes": metadata_path.stat().st_size,
        "sha256": module.sha256(metadata_path)}
    _rewrite_manifest(output, manifest)
    data = module.PhewasInputs(output)
    try:
        assert data.rm_nan is False
        assert data.trait(0)["sample_ids"].tolist() == [101, 104, 106]
        assert data.trait(0)["imputed_n"] == 0
    finally:
        data.close()
