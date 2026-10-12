"""Continuous input, dispatch and resume options on small anonymous inputs.

The execution-contract tests replace the shared executor. Input-policy tests
parse real temporary CSVs but do not claim real-data accuracy or throughput.
"""
import csv
import inspect
import json

import numpy as np
import pytest

from fudan_wgs_toolkit import run_WGS_continuous
from fudan_wgs_toolkit import continuous_run as module
from fudan_wgs_toolkit.phewas_inputs import PhewasInputs, prepare_phewas_inputs


def metadata():
    return dict(traits=[dict(index=index, name=f"trait_{index}", family="gaussian",
                            profile_index=0, covariate_indices=[0, 1]) for index in (0, 1)],
                quality=[dict(index=index, available_n=3, minimum=2., maximum=7.) for index in (0, 1)],
                covariate_names=["age", "sex"], profile_covariate_indices=[[0, 1]])


def test_package_export_defaults_and_single_executor(monkeypatch):
    from fudan_wgs_toolkit import run
    called = []
    sentinel = {"completed": True}
    monkeypatch.setattr(run, "_run_wgs_association", lambda *args, **kwargs:
                        called.append((args, kwargs)) or sentinel)
    assert run_WGS_continuous("traits.csv", "covariates.csv", "prepared", output_directory="results") is sentinel
    positional, options = called[0]
    assert positional == ("traits.csv", "covariates.csv", "prepared")
    assert options["input_contract"] == "continuous"
    assert options["association_partition_strategy"] == "chromosomes"
    assert options["phenotype_families"] == "auto" and options["phenotype_covariates"] is None
    assert options["rm_nan"] is True and options["continuous_transform"] == "none"
    assert options["long_mask_method"] == "full" and options["trait_batch_size"] == 32
    defaults = inspect.signature(run_WGS_continuous).parameters
    assert defaults["memory_limit_gib"].default is None
    assert defaults["host_memory_limit_gib"].default is None
    assert defaults["resume"].default is False


def test_execution_and_available_options_reach_common_executor(monkeypatch):
    from fudan_wgs_toolkit import run
    called = []
    monkeypatch.setattr(run, "_run_wgs_association", lambda *args, **kwargs: called.append(kwargs) or {})
    run_WGS_continuous("p", "c", "g", output_directory="r", rm_nan=False,
                       device=["cuda:4", "cuda:5", "cuda:6", "cuda:7"], chromosomes=[1, 22],
                       memory_limit_gib=20, host_memory_limit_gib=60, cpu_threads=4,
                       resume=True, grm="none", continuous_transform="rint")
    options = called[0]
    assert options["rm_nan"] is False and options["resume"] is True
    assert options["device"] == ["cuda:4", "cuda:5", "cuda:6", "cuda:7"]
    assert options["chromosomes"] == [1, 22]
    assert options["memory_limit_gib"] == 20 and options["host_memory_limit_gib"] == 60
    assert options["cpu_threads"] == 4 and options["grm"] == "none"
    assert options["continuous_transform"] == "rint"


@pytest.mark.parametrize("override", ["phenotype_families", "phenotype_covariates", "input_contract",
                                     "association_partition_strategy", "transform", "binary"])
def test_continuous_public_api_has_no_mixed_override(override):
    with pytest.raises(TypeError, match="unexpected keyword"):
        run_WGS_continuous("p", "c", "g", output_directory="r", **{override: {}})


def test_metadata_accepts_shared_covariates_and_eligible_extrema():
    source = metadata()
    source["quality"][0].update(minimum=3., maximum=3., eligible_minimum=2., eligible_maximum=4.)
    module.validate_continuous_metadata(source)


@pytest.mark.parametrize("change", ["one_trait", "binary", "empty", "constant", "eligible_constant",
                                   "infinite", "out_of_order", "second_profile", "subset_profile",
                                   "trait_subset", "duplicate_name", "invalid_available"])
def test_continuous_metadata_fails_closed(change):
    source = metadata()
    if change == "one_trait":
        source["traits"].pop(); source["quality"].pop()
    elif change == "binary":
        source["traits"][1]["family"] = "binomial"
    elif change == "empty":
        source["quality"][1]["available_n"] = 0
    elif change == "constant":
        source["quality"][1]["maximum"] = 2.
    elif change == "eligible_constant":
        source["quality"][1].update(eligible_minimum=5., eligible_maximum=5.)
    elif change == "infinite":
        source["quality"][1]["maximum"] = float("inf")
    elif change == "out_of_order":
        source["quality"].reverse()
    elif change == "second_profile":
        source["profile_covariate_indices"].append([0])
    elif change == "subset_profile":
        source["profile_covariate_indices"] = [[0]]
    elif change == "trait_subset":
        source["traits"][1]["covariate_indices"] = [0]
    elif change == "duplicate_name":
        source["traits"][1]["name"] = source["traits"][0]["name"]
    elif change == "invalid_available":
        source["quality"][1]["available_n"] = True
    with pytest.raises(ValueError):
        module.validate_continuous_metadata(source)


@pytest.mark.parametrize("invalid", ["binary", "empty", "constant", "one_finite", "infinite", "one_trait"])
def test_preparsed_continuous_validation_rejects_invalid_columns(invalid):
    names = ["trait_a", "trait_b"]
    values = np.asarray([[2., 4.], [3., 5.], [np.nan, 6.]])
    if invalid == "binary":
        values[:, 1] = [0., 1., np.nan]
    elif invalid == "empty":
        values[:, 1] = np.nan
    elif invalid == "constant":
        values[:, 1] = 4.
    elif invalid == "one_finite":
        values[:, 1] = [np.nan, 5., np.nan]
    elif invalid == "infinite":
        values[0, 1] = np.inf
    elif invalid == "one_trait":
        values = values[:, :1]; names.pop()
    with pytest.raises(ValueError):
        module.validate_continuous_values(names, values)


def test_preparsed_values_preserve_nan_and_do_not_modify_input():
    values = np.asarray([[2., np.nan], [4., 7.], [np.nan, 9.]])
    before = values.copy()
    families, quality = module.validate_continuous_values(["trait_a", "trait_b"], values)
    assert families == {"trait_a": "gaussian", "trait_b": "gaussian"}
    assert [record["available_n"] for record in quality] == [2, 2]
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("rm_nan", [True, False])
def test_shared_covariates_mean_or_available_samples_after_one_parse(tmp_path, rm_nan):
    axis = tmp_path / "axis.npy"
    np.save(axis, [4, 2, 1, 3, 5, 6, 7], allow_pickle=False)
    phenotypes, covariates = tmp_path / "p.csv", tmp_path / "c.csv"
    with phenotypes.open("w", newline="") as handle:
        csv.writer(handle).writerows([["sample_id", "trait_a", "trait_b"],
                                     [1, 1, 2], [2, 3, ""], [3, "", 6],
                                     [4, 7, 8], [5, 9, 10], [6, 11, 12]])
    with covariates.open("w", newline="") as handle:
        csv.writer(handle).writerows([["sample_id", "age"], [6, 26], [5, 25], [4, ""],
                                     [3, 23], [2, 22], [1, 21], [7, 27]])
    result = prepare_phewas_inputs(phenotypes=phenotypes, covariates=covariates,
        sample_ids=axis, output=tmp_path / "inputs", rm_nan=rm_nan, chunk_rows=2)
    assert result["numeric_csv_parses_per_input"] == 1
    assert result["unique_covariate_profiles"] == 1
    inputs = PhewasInputs(tmp_path / "inputs")
    try:
        module.validate_continuous_metadata(inputs.metadata)
        first, second = inputs.trait(0), inputs.trait(1)
        assert first["covariate_names"] == second["covariate_names"] == ["age"]
        assert first["available_n"] == second["available_n"] == 4
        if rm_nan:
            assert first["sample_indices"].tolist() == second["sample_indices"].tolist() == [1, 2, 3, 4, 5]
            assert first["y_raw"].tolist() == [3., 1., 6., 9., 11.]
            assert second["y_raw"].tolist() == [7.5, 2., 6., 10., 12.]
            assert first["imputed_n"] == second["imputed_n"] == 1
        else:
            assert first["sample_indices"].tolist() == [1, 2, 4, 5]
            assert second["sample_indices"].tolist() == [2, 3, 4, 5]
            assert first["y_raw"].tolist() == [3., 1., 9., 11.]
            assert second["y_raw"].tolist() == [2., 6., 10., 12.]
            assert first["imputed_n"] == second["imputed_n"] == 0
        assert 6 not in first["sample_indices"] and 6 not in second["sample_indices"]
    finally:
        inputs.close()


def test_cli_preserves_available_path_and_device_count(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(module, "run_WGS_continuous", lambda **kwargs:
                        called.append(kwargs) or dict(completed=True))
    module.main(["p.csv", "c.csv", "prepared", "--output-directory", "out",
                 "--device", "4", "--chromosomes", "21", "22", "--no-rm-nan", "--resume"])
    options = called[0]
    assert options["device"] == 4 and options["chromosomes"] == ["21", "22"]
    assert options["rm_nan"] is False and options["resume"] is True
    assert options["continuous_transform"] == "none" and options["long_mask_method"] == "full"
    assert json.loads(capsys.readouterr().out)["completed"] is True


@pytest.mark.parametrize("flag", ["--phenotype-families", "--phenotype-covariates"])
def test_cli_rejects_mixed_trait_overrides(flag):
    with pytest.raises(SystemExit) as error:
        module.main(["p", "c", "g", "--output-directory", "out", flag, "overrides.json"])
    assert error.value.code == 2
