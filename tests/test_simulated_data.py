"""Direct synthetic input contracts, without real cohorts or raw genotypes.

Tests read actual prepared frames/metadata and the two production input
contracts. They establish reproducibility and compatibility, not official
scientific agreement or GPU throughput.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fudan_wgs_toolkit.cache_runtime import sparse_codec_fast as csr, store
from fudan_wgs_toolkit.cache_runtime.portable import PortableMetadataReader
from fudan_wgs_toolkit.continuous_run import run_WGS_continuous, validate_continuous_values
from fudan_wgs_toolkit.identity import sample_keys
from fudan_wgs_toolkit.run import (_analysis_settings, _annotation_coverage, _dataset,
    _jobs, _validate_binary_phenotypes, aligned_phenotype, read_csv_inputs,
    run_WGS_all_binanry)
from fudan_wgs_toolkit.simulate import generate_simulated_data
from fudan_wgs_toolkit.simulation_profile import small_profile


def _csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return tuple(reader.fieldnames), list(reader)


def _input_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(root) / "inputs").rglob("*") if path.is_file()}


def _states(directory):
    reader = store.Container(directory)
    assert reader.verify_streams()
    result = np.zeros((reader.manifest["m"], reader.manifest["n"]), dtype=np.uint8)
    for index in range(len(reader.index)):
        frame = reader.read_frame(index)
        arrays = (frame["offsets"], frame["sample_index"], frame["state"])
        counts = csr.integer_counts(*arrays, frame["m"], frame["n"])
        for name in ("reference_alleles", "called_alleles", "half_missing_samples"):
            np.testing.assert_array_equal(frame[name], counts[name])
        for local in range(frame["m"]):
            first, stop = frame["offsets"][local:local + 2]
            result[frame["start"] + local, frame["sample_index"][first:stop]] = frame["state"][first:stop]
    return result


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_direct") / "data"
    report = generate_simulated_data(root, preset="small", n=1000,
        chromosomes=[21, 22], n_binary_traits=3, n_continuous_traits=3,
        seed=90210, phenotype_missing_rate=0.12, cpu_threads=2,
        chunk_size=32, memory_limit_gib=2)
    return root, report


def test_no_raw_interchange_files_and_actual_prepared_readers(generated, tmp_path):
    root, report = generated
    assert report["completed"] is True and report["synthetic"] is True
    assert report["schema_version"] == 2
    assert {path.name for path in root.iterdir()} == {"prepared", "inputs", "simulation.json"}
    assert not (root / "simulation.pending.json").exists()
    forbidden = {".bed", ".bim", ".fam", ".gds", ".rds", ".rdata"}
    assert not any(path.suffix.lower() in forbidden for path in root.rglob("*"))
    _, _, manifest, entries, pairs = _dataset(report["prepared_directory"])
    assert pairs.shape == (1000, 2)
    assert [entry["name"] for entry in entries] == ["21", "22"]
    for entry in entries:
        coverage = _annotation_coverage(entry, gene_variant_type="variant",
            analyses=("individual", "coding", "noncoding", "ncrna"),
            expected_sample_keys=sample_keys(pairs), expected_analysis=_analysis_settings(manifest))
        assert coverage["n_variants"] == coverage["matched_variants"] == 1024
        assert coverage["missing_snv"] == coverage["missing_non_snv"] == 0
        reader = store.Container(entry["container_directory"])
        assert reader.manifest["source_bytes"] is None
        assert reader.manifest["binding"]["synthetic"] is True
        assert reader.manifest["storage_budget_kind"] == "explicit_synthetic_bytes"
        states = _states(entry["container_directory"])
        assert states.shape == (1024, 1000) and set(np.unique(states)) <= set(range(6))
        with PortableMetadataReader(entry["metadata_directory"], entry["container_directory"]) as metadata:
            np.testing.assert_array_equal(metadata.sample_ids(), sample_keys(pairs))
            positions = metadata.read_field("position")
            assert np.all(positions > 0) and np.all(np.diff(positions) >= 0)
            assert metadata.read_field("annotation_available").all()
        jobs = _jobs(entry, ("individual", "coding", "noncoding", "ncrna"), tmp_path,
            single_mac_cutoff=20, single_group_variants=5000, single_region_size=10_000_000)
        assert {job["kind"] for job in jobs} == {"individual", "coding", "noncoding", "ncrna"}


def test_returned_arguments_bind_both_real_public_api_signatures(generated):
    _, report = generated
    for name, function, phenotype in (("binary_kwargs", run_WGS_all_binanry, "binary_phenotype_csv"),
                                     ("continuous_kwargs", run_WGS_continuous, "continuous_phenotype_csv")):
        options = dict(report[name])
        inspect.signature(function).bind(**options)
        assert options["phenotype_csv"] == report[phenotype]
        assert options["prepared_directory"] == report["prepared_directory"]
        assert options["covariate_csv"] == report["covariate_csv"]
        assert options["seed"] == report["seed"]
        assert options["grm"] == "none" and options["gene_variant_type"] == "variant"
        assert options.get("impute_missing", options.get("rm_nan")) is False
        assert tuple(options["analyses"]) == ("individual", "coding", "noncoding", "ncrna")


def test_causal_truth_is_bound_to_the_actual_prepared_locus(generated):
    _, report = generated
    preparation = report["preparation"]
    causal = preparation["causal_variant"]
    directory = Path(report["prepared_directory"]) / ("chr" + causal["chromosome"])
    states = _states(directory)[causal["index"]]
    called = csr.CALLED[states]
    alternate = called - csr.REF[states]
    frequency = float(alternate.sum() / called.sum())
    dosage = alternate + (2 - called) * frequency
    np.testing.assert_array_equal(np.load(preparation["causal_dosage_path"], allow_pickle=False), dosage)
    with PortableMetadataReader(directory / "metadata", directory) as metadata:
        assert int(metadata.read_field("position", [causal["index"]])[0]) == causal["position"]
        reference, alternate_allele = metadata.read_ref_alt([causal["index"]])
        assert reference[0] == causal["reference"] and alternate_allele[0] == causal["alternate"]
    _, truth = _csv(report["ground_truth_csv"])
    assert len(truth) == len(report["trait_reports"])
    for row in truth:
        assert row["chromosome"] == causal["chromosome"]
        assert int(row["position"]) == causal["position"]
        assert float(row["dosage_mean"]) == float(dosage.mean())
        assert float(row["dosage_sd"]) == float(dosage.std())


def test_trait_missingness_and_complete_covariate_samples_are_independent(generated):
    _, report = generated
    pairs = np.load(Path(report["prepared_directory"]) / "sample_pairs.npy", allow_pickle=False)
    fields, covariate_rows = _csv(report["covariate_csv"])
    assert fields == ("FID", "IID", "age", "sex", *(f"PC{i}" for i in range(1, 21)))
    assert len(covariate_rows) == len(pairs)
    for family, name in (("binomial", "binary_phenotype_csv"), ("gaussian", "continuous_phenotype_csv")):
        traits, covariates = read_csv_inputs(report[name], report["covariate_csv"])
        np.testing.assert_array_equal(traits.sample_pairs, pairs)
        np.testing.assert_array_equal(covariates.sample_pairs, pairs)
        assert traits.values.shape == (1000, 3)
        missing = ~np.isfinite(traits.values)
        assert np.all(missing.sum(axis=0) > 0)
        assert not np.array_equal(missing[:, 0], missing[:, 1])
        if family == "binomial":
            assert _validate_binary_phenotypes(traits) == dict.fromkeys(traits.names, family)
        else:
            assert validate_continuous_values(traits.names, traits.values)[0] == dict.fromkeys(traits.names, family)
        for index, trait in enumerate(traits.names):
            ids, y, x, _, counts = aligned_phenotype(traits, covariates, pairs, trait,
                rm_nan=False, family=family, binary_only=family == "binomial")
            eligible = np.isfinite(traits.values[:, index]) & np.isfinite(covariates.values).all(axis=1)
            np.testing.assert_array_equal(ids, sample_keys(pairs[eligible]))
            np.testing.assert_array_equal(y, traits.values[eligible, index])
            assert x.shape[0] == eligible.sum() and counts["imputed_n"] == 0


def test_seed_reproduces_input_bytes_across_worker_and_chunk_changes(tmp_path):
    first, second, other = (tmp_path / name for name in ("serial", "parallel", "different"))
    common = dict(preset="small", chromosomes=[21, 22], n=131, seed=9901,
                  phenotype_missing_rate=0.1, memory_limit_gib=2)
    one = generate_simulated_data(first, chunk_size=7, cpu_threads=1, **common)
    two = generate_simulated_data(second, chunk_size=32, cpu_threads=2, **common)
    assert _input_hashes(first) == _input_hashes(second)
    for chromosome in ("21", "22"):
        np.testing.assert_array_equal(_states(Path(one["prepared_directory"]) / ("chr" + chromosome)),
                                      _states(Path(two["prepared_directory"]) / ("chr" + chromosome)))
    generate_simulated_data(other, chunk_size=7, cpu_threads=1, **dict(common, seed=9902))
    assert (first / "inputs/covariates.csv").read_bytes() != (other / "inputs/covariates.csv").read_bytes()


def test_default_requires_real_aggregate_profile_without_silent_small_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("FUDAN_WGS_SIMULATION_PROFILE", raising=False)
    monkeypatch.setenv("FUDAN_WGS_SIMULATION_CONFIG", str(tmp_path / "no_configuration.json"))
    output = tmp_path / "unconfigured"
    with pytest.raises(FileNotFoundError, match="profile"):
        generate_simulated_data(output)
    assert not output.exists()


def test_dry_run_reports_compressed_final_and_peak_without_publishing(tmp_path, monkeypatch):
    import fudan_wgs_toolkit.simulate as implementation
    monkeypatch.setattr(implementation.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    output = tmp_path/"planned"
    report = generate_simulated_data(output, preset="small", n=1000, chromosomes=[21],
                                     cpu_threads=1, memory_limit_gib=2, dry_run=True)
    assert report["dry_run"] and not report["completed"] and not report["output_created"]
    assert not output.exists()
    resources = report["resource_plan"]
    assert resources["estimated_total_peak_bytes"] > resources["estimated_total_disk_bytes"]
    assert not resources["generation_fits_available_disk"]
    with pytest.raises(ValueError, match="available disk"):
        generate_simulated_data(output, preset="small", n=1000, chromosomes=[21],
                                cpu_threads=1, memory_limit_gib=2)


def test_n_defaults_to_profile_reference_and_no_private_identifiers_are_preserved(tmp_path):
    profile = small_profile()
    profile["chromosomes"] = [entry for entry in profile["chromosomes"] if entry["name"] == "21"]
    # Unexpected identifying labels are deliberately discarded by normalization.
    profile["private_label"] = "NEVER_COPY_REAL_LABEL"
    report = generate_simulated_data(tmp_path / "explicit_profile", calibration_profile=profile,
                                    chromosomes=[21], memory_limit_gib=2, cpu_threads=1)
    assert report["n_samples"] == profile["reference_n"] and report["calibration"]["calibrated"] is True
    assert report["resource_plan"]["total_variants"] == 1024
    for directory in (Path(report["prepared_directory"]), Path(report["covariate_csv"]).parent):
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".csv"):
                assert "NEVER_COPY_REAL_LABEL" not in path.read_text()


@pytest.mark.parametrize("family,expected,absent", [("binary", "binary_phenotype_csv", "continuous_phenotype_csv"),
                                                   ("continuous", "continuous_phenotype_csv", "binary_phenotype_csv")])
def test_only_requested_family_is_written(tmp_path, family, expected, absent):
    report = generate_simulated_data(tmp_path / family, preset="small", n=131,
        chromosomes=[21], phenotype_type=family, cpu_threads=1, memory_limit_gib=2)
    assert Path(report[expected]).is_file() and report[absent] is None
    assert not (Path(report["covariate_csv"]).parent / ("continuous_phenotypes.csv" if family == "binary" else "binary_phenotypes.csv")).exists()


def test_covariate_nan_excludes_only_actual_unavailable_subjects(tmp_path):
    report = generate_simulated_data(tmp_path / "covariate_nan", preset="small", n=1000,
        chromosomes=[21], covariate_missing_rate=0.015, phenotype_missing_rate=0.1,
        cpu_threads=1, memory_limit_gib=2)
    pairs = np.load(Path(report["prepared_directory"]) / "sample_pairs.npy", allow_pickle=False)
    traits, covariates = read_csv_inputs(report["binary_phenotype_csv"], report["covariate_csv"])
    eligible = np.isfinite(covariates.values).all(axis=1)
    assert 26 < eligible.sum() < len(pairs)
    assert eligible.sum() == report["complete_covariate_samples"]
    for index, trait in enumerate(traits.names):
        identifiers, y, x, _, counts = aligned_phenotype(traits, covariates, pairs, trait,
            rm_nan=False, family="binomial", binary_only=True)
        valid = eligible & np.isfinite(traits.values[:, index])
        np.testing.assert_array_equal(identifiers, sample_keys(pairs[valid]))
        assert np.isfinite(x).all() and counts["analysis_samples"] == valid.sum()


@pytest.mark.parametrize("invalid", [
    {"n": 127}, {"n": True}, {"n": 128.5}, {"chromosomes": []}, {"chromosomes": [21, "21"]},
    {"chromosomes": [23]}, {"phenotype_type": "other"}, {"preset": "other"},
    {"n_binary_traits": 1}, {"n_continuous_traits": 1}, {"seed": -1}, {"seed": True},
    {"cpu_threads": 0}, {"chunk_size": 0}, {"chunk_size": 1025},
    {"variant_scale": 0}, {"variant_scale": float("nan")},
    {"phenotype_missing_rate": 1}, {"covariate_missing_rate": -0.1},
    {"case_fraction": 0}, {"case_fraction": 1}, {"causal_effect": float("inf")},
    {"memory_limit_gib": float("inf")}, {"storage_limit_gib": 0}, {"dry_run": "yes"},
])
def test_invalid_options_fail_before_output_creation(tmp_path, invalid):
    root = tmp_path / "invalid"
    options = dict(preset="small", chromosomes=[21], n=131, cpu_threads=1)
    options.update(invalid)
    with pytest.raises((TypeError, ValueError, MemoryError)):
        generate_simulated_data(root, **options)
    assert not root.exists()


def test_impossible_memory_or_disk_budget_rejects_without_creating_output(tmp_path, monkeypatch):
    root = tmp_path / "memory"
    with pytest.raises((ValueError, MemoryError), match="memory|host"):
        generate_simulated_data(root, preset="small", chromosomes=[21], n=131,
                                memory_limit_gib=0.00001, cpu_threads=1)
    assert not root.exists()
    root = tmp_path / "storage"
    with pytest.raises((ValueError, MemoryError), match="storage"):
        generate_simulated_data(root, preset="small", chromosomes=[21], n=131,
                                storage_limit_gib=0.00001, cpu_threads=1)
    assert not root.exists()
    from fudan_wgs_toolkit import simulate
    monkeypatch.setattr(simulate.shutil, "disk_usage", lambda parent: SimpleNamespace(free=100))
    root = tmp_path / "full_disk"
    with pytest.raises((ValueError, MemoryError, OSError), match="disk|storage"):
        generate_simulated_data(root, preset="small", chromosomes=[21], n=131, cpu_threads=1)
    assert not root.exists()


def test_existing_output_is_never_modified(tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    sentinel = root / "retain.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        generate_simulated_data(root, preset="small", chromosomes=[21], n=131)
    assert list(root.iterdir()) == [sentinel] and sentinel.read_text() == "keep"


def test_public_import_exposes_only_the_new_direct_generator():
    from fudan_wgs_toolkit import generate_simulated_data as function
    assert function is generate_simulated_data
    parameters = inspect.signature(function).parameters
    assert parameters["n"].default is None and parameters["chromosomes"].default == "all"
    assert parameters["preset"].default == "reference"
    assert not {"prepare", "variants_per_gene", "variants_per_chromosome", "genotype_missing_rate"} & parameters.keys()
