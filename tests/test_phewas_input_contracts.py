"""Continuous/binary prepared input bindings across the API merge."""
import csv
import json

import numpy as np
import pytest

from fudan_wgs_toolkit import phewas_inputs as module


def data(tmp_path):
    phenotype, covariate, axis = [tmp_path / name for name in ("p.csv", "c.csv", "axis.npy")]
    np.save(axis, np.arange(1, 7, dtype=np.int64), allow_pickle=False)
    with phenotype.open("w", newline="") as stream:
        csv.writer(stream).writerows([["FID", "IID", "trait_a", "trait_b"],
            [0, 1, 2, -1], [0, 2, "NA", 3], [0, 3, 6, "NA"], [0, 4, 7, 9], [0, 5, 10, 12]])
    with covariate.open("w", newline="") as stream:
        csv.writer(stream).writerows([["FID", "IID", "age"], [0, 1, 20], [0, 2, 21],
            [0, 3, 22], [0, 4, 23], [0, 5, "NA"], [0, 6, 25]])
    return dict(phenotypes=phenotype, covariates=covariate, sample_ids=axis,
                output=tmp_path / "prepared", input_contract="continuous", chunk_rows=2)


def rewrite_binding(directory, manifest, metadata=None):
    if metadata is not None:
        path = directory / "traits.private.json"
        path.write_text(json.dumps(metadata))
        manifest["artifacts"][path.name] = dict(bytes=path.stat().st_size, sha256=module.sha256(path))
    path = directory / "manifest.private.json"
    path.write_text(json.dumps(manifest))
    (directory / "status.anonymous.json").write_text(json.dumps(
        dict(status="complete", manifest_sha256=module.sha256(path))))


@pytest.mark.parametrize("fill", [True, False])
def test_continuous_input_contract_binds_families_and_shared_readonly_design(tmp_path, fill):
    args = data(tmp_path)
    summary = module.prepare_phewas_inputs(**args, rm_nan=fill)
    assert summary["input_contract"] == "continuous" and summary["binary_only"] is False
    inputs = module.PhewasInputs(args["output"])
    try:
        assert inputs.input_contract == inputs.manifest["trait_options"]["input_contract"] == "continuous"
        assert inputs.metadata["input_contract"] == "continuous"
        assert [trait["family"] for trait in inputs.metadata["traits"]] == ["gaussian", "gaussian"]
        first, second = inputs.trait(0), inputs.trait(1)
        assert first["analysis_cases"] is second["analysis_controls"] is None
        assert first["available_n"] == second["available_n"] == 3
        assert first["input_contract"] == second["input_contract"] == "continuous"
        if fill:
            assert first["sample_ids"].tolist() == second["sample_ids"].tolist() == [1, 2, 3, 4]
            np.testing.assert_allclose(first["y_raw"], [2, 5, 6, 7])
            np.testing.assert_allclose(second["y_raw"], [-1, 3, 11 / 3, 9])
            assert first["covariates"] is second["covariates"]
        else:
            assert first["sample_ids"].tolist() == [1, 3, 4]
            assert second["sample_ids"].tolist() == [1, 2, 4]
            assert first["covariates"] is not second["covariates"]
        with pytest.raises(ValueError):
            first["covariates"].setflags(write=True)
        assert inputs.design_cache_summary()["resident_bytes"] <= 128 << 20
    finally:
        inputs.close()


@pytest.mark.parametrize("override", [dict(binary_only=True), dict(input_contract="invalid"),
    dict(phenotype_families={"trait_a": "gaussian"}), dict(phenotype_covariates={})])
def test_invalid_or_conflicting_contract_rejected_before_output_creation(tmp_path, override):
    args = data(tmp_path)
    args.update(override)
    with pytest.raises(ValueError):
        module.prepare_phewas_inputs(**args)
    assert not args["output"].exists()


def test_continuous_contract_rejects_binary_column(tmp_path):
    args = data(tmp_path)
    args["phenotypes"].write_text("FID,IID,trait_a,trait_b\n0,1,0,2\n0,2,1,4\n0,3,0,6\n0,4,1,8\n")
    with pytest.raises(ValueError, match="continuous phenotypes only|Gaussian phenotypes only"):
        module.prepare_phewas_inputs(**args)
    assert not (args["output"] / "manifest.private.json").exists()


@pytest.mark.parametrize("changed", ["manifest", "trait_options", "metadata"])
def test_completed_input_contract_mismatch_cannot_be_reinterpreted(tmp_path, changed):
    args = data(tmp_path)
    module.prepare_phewas_inputs(**args)
    directory = args["output"]
    manifest = json.loads((directory / "manifest.private.json").read_text())
    metadata = json.loads((directory / "traits.private.json").read_text())
    if changed == "manifest":
        manifest["input_contract"] = "mixed"
    elif changed == "trait_options":
        manifest["trait_options"]["input_contract"] = "mixed"
    else:
        metadata["input_contract"] = "mixed"
    rewrite_binding(directory, manifest, metadata)
    with pytest.raises(ValueError, match="input contract binding"):
        module.PhewasInputs(directory)


def test_original_continuous_manifest_trait_options_binding_remains_readable(tmp_path):
    args = data(tmp_path)
    module.prepare_phewas_inputs(**args)
    directory = args["output"]
    manifest = json.loads((directory / "manifest.private.json").read_text())
    metadata = json.loads((directory / "traits.private.json").read_text())
    # Incoming continuous release stored only trait_options.input_contract;
    # binary_only was also introduced independently on the other branch.
    for document in (manifest, metadata):
        document.pop("input_contract")
        document.pop("binary_only")
    manifest["trait_options"].pop("binary_only")
    rewrite_binding(directory, manifest, metadata)
    inputs = module.PhewasInputs(directory)
    try:
        assert inputs.input_contract == "continuous" and inputs.binary_only is False
        assert inputs.trait(0)["sample_ids"].tolist() == [1, 2, 3, 4]
    finally:
        inputs.close()
