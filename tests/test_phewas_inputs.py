"""Meaningful input alignment and phenotype-wise missingness contracts."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).parents[1] / "fudan_wgs_toolkit" / "phewas_inputs.py"
spec = importlib.util.spec_from_file_location("phewas_inputs_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def table(path, rows):
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)


def inputs(tmp_path, *, binary_bad=False, duplicate=False):
    ids = tmp_path / "axis.npy"
    np.save(ids, [1000003, 1000001, 1000002, 1000004], allow_pickle=False)
    p, c, profiles = [tmp_path / name for name in ("p.csv", "c.csv", "profiles.csv")]
    table(p, [["sample_id", "continuous", "binary"],
              [1000001, "0.10000000000000002", 1],
              [1000002, "", 0], [1000003, 3, 2 if binary_bad else 1],
              [1000001 if duplicate else 1000005, 5, 0]])
    table(c, [["sample_id", "age", "extra"], [1000002, 20, 2],
              [1000003, 30, ""], [1000001, 10, 1], [1000005, 50, 5]])
    # Deliberately different profile order: align by exact phenotype label.
    table(profiles, [["phenotype", "model_family", "profile", "covariate_columns"],
                     ["binary", "binomial", "extra_profile", "age;extra"],
                     ["continuous", "gaussian", "base_profile", "age"]])
    return dict(phenotypes=p, covariates=c, profiles=profiles, sample_ids=ids,
                output=tmp_path / "prepared", chunk_rows=2, rm_nan=False)


def test_cache_order_float_roundtrip_and_individual_profiles(tmp_path):
    args = inputs(tmp_path)
    result = module.prepare_phewas_inputs(**args)
    assert result["numeric_csv_parses_per_input"] == 1
    assert result["families"]["gaussian"]["min_analysis_complete_samples"] == 2
    assert result["families"]["binomial"]["min_analysis_complete_samples"] == 2
    loaded = module.PhewasInputs(args["output"])
    assert loaded.sample_ids.tolist() == [1000003, 1000001, 1000002, 1000004]
    continuous, binary = loaded.trait(0), loaded.trait(1)
    assert continuous["sample_indices"].tolist() == [0, 1]
    assert continuous["y_raw"].tolist() == [3, float("0.10000000000000002")]
    assert continuous["covariates"].shape == (2, 1)
    assert binary["sample_indices"].tolist() == [1, 2]
    assert binary["y_raw"].tolist() == [1, 0]
    assert binary["covariates"].tolist() == [[10, 1], [20, 2]]
    assert not continuous["intercept_added"]
    # Arbitrary caller request order is normalized to cache physical order.
    assert loaded.trait(0, cohort_indices=np.array([1, 0]))["sample_indices"].tolist() == [0, 1]
    assert loaded.trait(0, cohort_indices=np.array([True, False, True, False]))["sample_indices"].tolist() == [0]
    with pytest.raises(ValueError):
        loaded.trait(0, cohort_indices=np.array([0, 0]))
    loaded.close()


@pytest.mark.parametrize("which", ["binary_bad", "duplicate"])
def test_invalid_source_never_publishes_completion(tmp_path, which):
    args = inputs(tmp_path, **{which: True})
    with pytest.raises(ValueError):
        module.prepare_phewas_inputs(**args)
    assert not (args["output"] / "manifest.private.json").exists()
    assert json.loads((args["output"] / "status.anonymous.json").read_text())["status"] == "failed"


def test_hash_corruption_and_no_overwrite(tmp_path):
    args = inputs(tmp_path)
    module.prepare_phewas_inputs(**args)
    with pytest.raises(FileExistsError):
        module.prepare_phewas_inputs(**args)
    path = args["output"] / "phenotypes.npy"
    data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    with pytest.raises(ValueError):
        module.PhewasInputs(args["output"])


def test_dictionary_family_and_source_manifest_binding(tmp_path):
    args = inputs(tmp_path)
    dictionary = tmp_path / "dictionary.csv"
    table(dictionary, [["column_index", "phenotype", "model_family"],
                       [2, "continuous", "gaussian"], [3, "binary", "binomial"]])
    pm = tmp_path / "manifest.json"
    pm.write_text(json.dumps({"status": "complete", "output": str(args["phenotypes"]),
                             "participant_rows": 4,
                             "output_sha256": module.sha256(args["phenotypes"])}))
    args.update(phenotype_dictionary=dictionary, phenotype_manifest=pm)
    module.prepare_phewas_inputs(**args)
    loaded = module.PhewasInputs(args["output"])
    assert len(loaded.metadata["phenotype_dictionary"]) == 2
    loaded.close()


def test_optional_profiles_use_all_covariates_and_single_parse_family_inference(tmp_path):
    args = inputs(tmp_path)
    del args["profiles"]
    summary = module.prepare_phewas_inputs(**args)
    assert summary["unique_covariate_profiles"] == 1
    assert summary["numeric_csv_parses_per_input"] == 1
    assert summary["covariate_profile_scope"] == "all_covariates_generic"
    assert summary["paper_specific_covariate_profile_applied"] is False
    loaded = module.PhewasInputs(args["output"])
    assert "profiles" not in loaded.manifest["sources"]
    assert loaded.trait(0)["family"] == "gaussian"
    assert loaded.trait(1)["family"] == "binomial"
    # Missing extra covariate excludes cache participant 0 for BOTH traits;
    # missing continuous phenotype excludes participant 2 only for that trait.
    assert loaded.trait(0)["sample_indices"].tolist() == [1]
    assert loaded.trait(1)["sample_indices"].tolist() == [1, 2]
    assert loaded.trait(1)["covariate_names"] == ["age", "extra"]
    loaded.close()


def test_optional_profiles_inference_covers_noncache_rows_and_empty_columns(tmp_path):
    args = inputs(tmp_path)
    args["profiles"] = None
    table(args["phenotypes"], [["sample_id", "only_binary_in_cache", "empty"],
              [1000001, 0, ""], [1000002, 1, ""], [1000005, 2, ""]])
    summary = module.prepare_phewas_inputs(**args)
    assert summary["families"]["gaussian"]["traits"] == 2
    assert summary["phenotype_parse"]["entirely_missing_source_traits"] == 1
    loaded = module.PhewasInputs(args["output"])
    assert loaded.trait(0)["family"] == "gaussian"
    assert loaded.trait(1)["y_raw"].size == 0
    loaded.close()


def test_numeric_cache_standard_pairs_preserve_covariates_and_trait_rows(tmp_path):
    args = inputs(tmp_path)
    for name in ('phenotypes', 'covariates'):
        with args[name].open(newline='') as stream:
            rows = list(csv.reader(stream))
        table(args[name], [['FID', 'IID', *rows[0][1:]],
                           *[['0', row[0], *row[1:]] for row in rows[1:]]])
    module.prepare_phewas_inputs(**args)
    loaded = module.PhewasInputs(args['output'])
    assert loaded.trait(0)['sample_indices'].tolist() == [0, 1]
    assert loaded.trait(1)['sample_indices'].tolist() == [1, 2]
    assert loaded.trait(0)['covariate_names'] == ['age']
    loaded.close()


def test_identity_only_covariate_file_supports_intercept_only_models(tmp_path):
    args = inputs(tmp_path)
    args['profiles'] = None
    table(args['covariates'], [['FID', 'IID'], ['0', '1000001'], ['0', '1000002'],
                              ['0', '1000003']])
    result = module.prepare_phewas_inputs(**args)
    assert result['covariate_count'] == 0
    loaded = module.PhewasInputs(args['output'])
    assert loaded.trait(0)['sample_indices'].tolist() == [0, 1]
    assert loaded.trait(1)['sample_indices'].tolist() == [0, 1, 2]
    assert loaded.trait(1)['covariates'].shape == (3, 0)
    assert loaded.trait(1)['covariate_names'] == []
    loaded.close()


@pytest.mark.parametrize('fid,iid', [('family_a', '1000001'), ('0', '01000001')])
def test_numeric_axis_never_discards_family_or_leading_zero_identity(tmp_path, fid, iid):
    args = inputs(tmp_path)
    table(args['phenotypes'], [['FID', 'IID', 'continuous', 'binary'], [fid, iid, 3, 1]])
    with pytest.raises(ValueError):
        module.prepare_phewas_inputs(**args)
    assert not (args['output']/'manifest.private.json').exists()
