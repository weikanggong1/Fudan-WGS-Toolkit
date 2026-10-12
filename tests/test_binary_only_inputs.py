"""CSV contracts for disease columns and bounded shared covariate preparation."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).parents[1] / "fudan_wgs_toolkit" / "phewas_inputs.py"
spec = importlib.util.spec_from_file_location("binary_only_inputs_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _write(path, rows):
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)


def _inputs(tmp_path):
    axis, phenotypes, covariates = [tmp_path / name for name in ("axis.npy", "traits.csv", "covariates.csv")]
    np.save(axis, np.arange(101, 107), allow_pickle=False)
    _write(phenotypes, [["FID", "IID", "disease_a", "disease_b"],
        [0, 101, 1, 0], [0, 102, 0, 1], [0, 103, "", 1],
        [0, 104, 0, ""], [0, 105, 1, 0]])
    _write(covariates, [["FID", "IID", "age", "sex"],
        [0, 101, 20, 0], [0, 102, 30, 1], [0, 103, 40, 0],
        [0, 104, 50, 1], [0, 105, "", 0], [0, 106, 60, 1]])
    return dict(phenotypes=phenotypes, covariates=covariates, sample_ids=axis,
                output=tmp_path / "prepared", binary_only=True, chunk_rows=2)


def test_default_binary_fill_counts_and_shared_immutable_design(tmp_path):
    args = _inputs(tmp_path)
    summary = module.prepare_phewas_inputs(**args)
    inputs = module.PhewasInputs(args["output"])
    try:
        left, right = inputs.trait(0), inputs.trait(1)
        assert summary["binary_only"] is True and summary["rm_nan"] is True
        assert summary["family_assignment"] == "strict_binary_only"
        assert left["sample_ids"].tolist() == [101, 102, 103, 104]
        assert right["sample_ids"].tolist() == [101, 102, 103, 104]
        assert 106 not in left["sample_ids"]  # No phenotype row: never invent a control.
        np.testing.assert_array_equal(left["y_raw"], [1, 0, 0, 0])
        np.testing.assert_array_equal(right["y_raw"], [0, 1, 1, 0])
        assert (left["analysis_cases"], left["analysis_controls"], left["available_controls"]) == (1, 3, 2)
        assert (right["analysis_cases"], right["analysis_controls"], right["available_controls"]) == (2, 2, 1)
        assert left["available_n"] == right["available_n"] == 3
        assert left["imputed_controls"] == right["imputed_controls"] == 1
        for name in ("sample_indices", "sample_ids", "covariates"):
            assert left[name] is right[name]
            with pytest.raises(ValueError):
                left[name].setflags(write=True)
        assert left["y_raw"] is not right["y_raw"]
        stats = inputs.design_cache_summary()
        assert stats["hits"] == 1 and stats["misses"] == 1 and stats["entries"] == 1
        assert stats["resident_bytes"] <= stats["limit_bytes"]
        assert not stats["shared_fitted_models"]
        np.testing.assert_equal(inputs.phenotypes[2, 0], np.nan)
        assert inputs.metadata["quality"][0]["imputed_controls"] == 1
        assert inputs.manifest["trait_options"]["binary_only"] is True
    finally:
        inputs.close()
    assert inputs.design_cache_summary()["resident_bytes"] == 0


def test_no_fill_is_per_trait_and_design_lru_is_bounded(tmp_path):
    args = _inputs(tmp_path)
    module.prepare_phewas_inputs(**args, rm_nan=False)
    # Each 3-sample, 2-column design owns 96 bytes including its two axes.
    inputs = module.PhewasInputs(args["output"], design_cache_limit_bytes=96)
    try:
        left, right = inputs.trait(0), inputs.trait(1)
        assert left["sample_ids"].tolist() == [101, 102, 104]
        assert right["sample_ids"].tolist() == [101, 102, 103]
        assert left["covariates"] is not right["covariates"]
        assert left["analysis_cases"] == 1 and left["analysis_controls"] == 2
        assert right["analysis_cases"] == 2 and right["analysis_controls"] == 1
        assert left["imputed_controls"] == right["imputed_controls"] == 0
        np.testing.assert_array_equal(left["covariates"], [[20, 0], [30, 1], [50, 1]])
        np.testing.assert_array_equal(right["covariates"], [[20, 0], [30, 1], [40, 0]])
        stats = inputs.design_cache_summary()
        assert stats["entries"] == 1 and stats["evictions"] == 1 and stats["resident_bytes"] == 96
        inputs.trait(0)
        assert inputs.design_cache_summary()["evictions"] == 2
    finally:
        inputs.close()


@pytest.mark.parametrize("fill", [False, True])
@pytest.mark.parametrize("bad", [2, -1, .25, "inf", "-inf"])
def test_nonbinary_values_rejected_even_outside_cache_and_under_fill(tmp_path, fill, bad):
    args = _inputs(tmp_path)
    with args["phenotypes"].open("a", newline="") as stream:
        csv.writer(stream).writerow([0, 999, bad, 0])
    with pytest.raises(ValueError, match="zero, one or missing|infinite"):
        module.prepare_phewas_inputs(**args, rm_nan=fill)
    assert not (args["output"] / "manifest.private.json").exists()
    assert json.loads((args["output"] / "status.anonymous.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("values", [[0, 0, 0], [1, 1, 1], ["", "", ""]])
def test_requires_analysis_cases_and_controls(tmp_path, values):
    args = _inputs(tmp_path)
    _write(args["phenotypes"], [["FID", "IID", "disease_a", "disease_b"],
        *[[0, 101 + i, value, i % 2] for i, value in enumerate(values)]])
    with pytest.raises(ValueError, match="one analysis case and one control"):
        module.prepare_phewas_inputs(**args)


def test_single_column_and_gaussian_override_rejected_before_output(tmp_path):
    args = _inputs(tmp_path)
    with pytest.raises(ValueError, match="only binomial"):
        module.prepare_phewas_inputs(**args, phenotype_families={"disease_a": "gaussian"})
    assert not args["output"].exists()
    _write(args["phenotypes"], [["FID", "IID", "only_disease"], [0, 101, 1], [0, 102, 0]])
    with pytest.raises(ValueError, match="at least two"):
        module.prepare_phewas_inputs(**args)
    assert not args["output"].exists()


def test_optional_profiles_and_subset_still_check_actual_case_control_counts(tmp_path):
    args = _inputs(tmp_path)
    module.prepare_phewas_inputs(**args, phenotype_covariates={"disease_a": ["sex"]})
    inputs = module.PhewasInputs(args["output"])
    try:
        left, right = inputs.trait(0), inputs.trait(1)
        assert left["sample_ids"].tolist() == [101, 102, 103, 104, 105]
        assert left["covariate_names"] == ["sex"]
        assert right["covariate_names"] == ["age", "sex"]
        assert left["analysis_cases"] == 2 and left["analysis_controls"] == 3
        with pytest.raises(ValueError, match="one analysis case and one control"):
            inputs.trait(0, cohort_indices=np.asarray([1, 2, 3]))
    finally:
        inputs.close()


def test_manifest_cannot_reinterpret_strict_binary_data_as_generic(tmp_path):
    args = _inputs(tmp_path)
    module.prepare_phewas_inputs(**args)
    path = args["output"] / "manifest.private.json"
    manifest = json.loads(path.read_text())
    manifest["binary_only"] = False
    path.write_text(json.dumps(manifest))
    status = args["output"] / "status.anonymous.json"
    status.write_text(json.dumps({"status": "complete", "manifest_sha256": module.sha256(path)}))
    with pytest.raises(ValueError, match="binary-only input binding"):
        module.PhewasInputs(args["output"])
