"""Prepared continuous entry integration with genuine small CPU controls.

The fixture contains six anonymous samples encoded as a real PLINK BED. Null
models, cache decoding, Single filtering/variance/tails and CSV checkpoints
are executed. The CUDA production executor is replaced with an explicit
serial CPU FP64 control, so these tests establish input/output/resume contracts
and do not establish GPU accuracy, real-data performance or a speedup.
"""
import csv
import gzip
import json

import numpy as np
import pytest

from fudan_wgs_toolkit import run_WGS_continuous
from fudan_wgs_toolkit import prepared_association as prepared
from fudan_wgs_toolkit.phewas_models import prepare_phewas_model
from fudan_wgs_toolkit.run import run_WGS_all


def continuous_fixture(tmp_path, *, sparse_observations=False):
    from test_prepared_association import fixture
    root, phenotype, covariate, pairs = fixture(tmp_path)
    first = ["NA", 3.2, 1.1, 5.3, 2.8, 4.7]
    second = [-1.4, "NA", 3.8, 6.1, 4.2, 2.9]
    if sparse_observations:
        first = ["NA", 2., "NA", "NA", 7., "NA"]
        second = ["NA", 4., "NA", "NA", 9., "NA"]
    with phenotype.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["FID", "IID", "trait_a", "trait_b"])
        writer.writerows([*pair, first[index], second[index]] for index, pair in enumerate(pairs))
    if sparse_observations:
        with covariate.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["FID", "IID", "age", "age_squared"])
            writer.writerows([*pair, "NA" if index == 0 else index,
                              "NA" if index == 0 else index*index]
                             for index, pair in reversed(list(enumerate(pairs))))
    return root, phenotype, covariate, pairs


def cpu_single_control(monkeypatch):
    from fudan_wgs_toolkit import run
    from fudan_wgs_toolkit.cache_runtime.adapter_fast import CachedGenotypeAdapter
    from fudan_wgs_toolkit.pipeline import AnalysisOptions, PheWASPipeline
    from fudan_wgs_toolkit.phewas_runtime import runtime
    calls = dict(fits=[], computed=[], descriptors=[])
    monkeypatch.setattr(run.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(run.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(run.torch.cuda, "mem_get_info", lambda *args: (8*2**30, 8*2**30))

    def fit(*args, **kwargs):
        calls["fits"].append(kwargs.copy())
        kwargs["device"] = "cpu"
        return prepare_phewas_model(*args, **kwargs)

    monkeypatch.setattr(prepared, "prepare_phewas_model", fit)

    def associate(configuration, *, model_repository, reader_factory, writer, **kwargs):
        descriptors = model_repository.descriptors()
        calls["descriptors"].append([
            dict(index=entry["trait_index"], n=entry["n"], rows=entry["ordered_rows"].tolist())
            for entry in descriptors])
        counts = dict.fromkeys(("individual", "coding", "noncoding", "ncrna"), 0)
        trait_counts = {str(entry["trait_index"]): dict(counts) for entry in descriptors}
        jobs = 0
        for reference in configuration["chromosomes"]:
            for job in reference["jobs"]:
                assert job["kind"] == "individual", "this unit control covers Single only"
                jobs += 1
                ordinal = job["checkpoint_job_index"]
                for descriptor in descriptors:
                    trait = descriptor["trait_index"]
                    key = dict(block_index=0, block_identity=f"cpu-control-single-{ordinal}", ordinal_before=0)
                    record = writer.checkpoint_tile("individual", reference["name"], ordinal, [trait], **key)
                    if record is None:
                        with model_repository.acquire([trait], "cpu") as models:
                            model = models[0]
                            model.set_matmul_mode("fp64")
                            with reader_factory(reference) as resources:
                                adapter = CachedGenotypeAdapter(resources["metadata"], resources["container"],
                                    device="cpu", own_reader=False)
                                try:
                                    pipeline = PheWASPipeline(adapter, [model],
                                        qc_path=configuration["qc_path"],
                                        annotation_catalog=configuration["annotation_catalog"],
                                        annotation_names=configuration["annotation_names"],
                                        options=AnalysisOptions(**dict(configuration["analysis_options"],
                                                                      memory_limit_gib=8.)))
                                    arguments = dict(job["arguments"])
                                    result = pipeline.individual(chromosome=reference["name"], **arguments)[0]
                                finally:
                                    adapter.close()
                        calls["computed"].append((ordinal, trait))
                        record = writer.write_tile("individual", reference["name"], [(trait, result)],
                            job_index=ordinal, ordinal_after=len(result), **key)
                    counts["individual"] += record["rows"]
                    trait_counts[str(trait)]["individual"] += record["rows"]
        return dict(association_rows=counts, trait_rows=trait_counts, job_groups=jobs,
                    output=writer.finish(), unit_execution="serial_cpu_fp64_control")

    monkeypatch.setattr(runtime, "run_batched_configuration", associate)
    return calls


def call(api, root, phenotype, covariate, output, *, rm_nan, **options):
    return api(phenotype, covariate, root, output_directory=output,
        analyses=("individual",), chromosomes=[21], device=["cuda:0"], grm="none",
        cpu_threads=1, single_mac_cutoff=1, single_region_size=1_000_000,
        continuous_transform="none", rm_nan=rm_nan, **options)


def csv_rows(output, report):
    files = {}
    for receipt in report["workers"][0]["output"]["receipts"]:
        with gzip.open(output/"csv"/receipt["csv_file"], "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            files[receipt["csv_file"]] = (reader.fieldnames, list(reader))
    return files


@pytest.mark.parametrize("rm_nan", [True, False])
def test_continuous_prepared_single_matches_mixed_entry_and_bound_resume(tmp_path, monkeypatch, rm_nan):
    root, phenotype, covariate, pairs = continuous_fixture(tmp_path)
    calls = cpu_single_control(monkeypatch)
    standalone, mixed = tmp_path/"standalone", tmp_path/"mixed"
    report = call(run_WGS_continuous, root, phenotype, covariate, standalone, rm_nan=rm_nan)
    reference = call(run_WGS_all, root, phenotype, covariate, mixed, rm_nan=rm_nan)
    assert report["completed"] and reference["completed"]
    assert report["association_rows"] == reference["association_rows"]
    assert report["association_rows"]["individual"] > 0
    assert csv_rows(standalone, report) == csv_rows(mixed, reference)
    assert all(entry["family"] == "gaussian" and entry["continuous_transform"] == "none"
               for entry in calls["fits"])
    expected = [dict(index=0, n=5, rows=[1, 2, 3, 4, 5]),
                dict(index=1, n=5 if rm_nan else 4,
                     rows=[1, 2, 3, 4, 5] if rm_nan else [2, 3, 4, 5])]
    assert calls["descriptors"] == [expected, expected]
    plan = json.loads((standalone/"plan.private.json").read_text())
    assert plan["input_contract"] == "continuous"
    assert plan["association_partition_strategy"] == "chromosomes"
    assert plan["covariate_selection"] == {}
    before = list(calls["computed"])
    again = call(run_WGS_continuous, root, phenotype, covariate, standalone, rm_nan=rm_nan, resume=True)
    assert again["completed"] and calls["computed"] == before
    assert len(calls["fits"]) == 4
    assert again["workers"][0]["output"]["reused_blocks"] == 2
    assert csv_rows(standalone, again) == csv_rows(mixed, reference)
    for _, (_, rows) in csv_rows(standalone, again).items():
        assert rows and len(rows) == len({json.dumps(row, sort_keys=True) for row in rows})


def test_mean_preflight_accepts_two_observations_with_larger_valid_shared_design(tmp_path, monkeypatch):
    root, phenotype, covariate, _ = continuous_fixture(tmp_path, sparse_observations=True)
    calls = cpu_single_control(monkeypatch)
    report = call(run_WGS_continuous, root, phenotype, covariate, tmp_path/"mean", rm_nan=True)
    assert report["completed"]
    assert [item["n"] for item in calls["descriptors"][0]] == [5, 5]
    assert len(calls["fits"]) == 2
    for path in (tmp_path/"mean"/"models").glob("trait_*/model.private.json"):
        saved = json.loads(path.read_text())
        assert saved["n"] == 5
    with pytest.raises(ValueError, match="insufficient complete-case samples"):
        call(run_WGS_continuous, root, phenotype, covariate, tmp_path/"available", rm_nan=False)
    assert len(calls["fits"]) == 2


@pytest.mark.parametrize("invalid", ["binary", "eligible_constant", "all_missing"])
def test_bad_continuous_csv_rejected_before_actual_model_fit(tmp_path, monkeypatch, invalid):
    root, phenotype, covariate, pairs = continuous_fixture(tmp_path)
    calls = cpu_single_control(monkeypatch)
    second = [0, "NA", 1, 0, 1, 0] if invalid == "binary" else [-1.4, "NA", 3.8, 6.1, 4.2, 2.9]
    first = [3.2, 5., 5., 5., 5., 5.] if invalid == "eligible_constant" else ["NA", 3.2, 1.1, 5.3, 2.8, 4.7]
    if invalid == "all_missing":
        first = ["NA"]*6
    with phenotype.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["FID", "IID", "trait_a", "trait_b"])
        writer.writerows([*pair, first[index], second[index]] for index, pair in enumerate(pairs))
    with pytest.raises(ValueError):
        call(run_WGS_continuous, root, phenotype, covariate, tmp_path/"bad", rm_nan=True)
    assert not calls["fits"] and not calls["computed"]
    assert not (tmp_path/"bad"/"models").exists()
