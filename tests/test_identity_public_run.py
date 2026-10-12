"""Public scheduling and fitted identity contracts on small anonymous inputs.

The shared compute call is replaced to inspect its inputs. These checks make
no performance or scientific-equivalence claim.
"""
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

from fudan_wgs_toolkit.identity import sample_keys, pairs_from_keys
from fudan_wgs_toolkit.io import load_null_model
from fudan_wgs_toolkit.run import run_WGS_all


def commit_dataset(root, value):
    raw = json.dumps(value).encode()
    (root / "dataset.json").write_bytes(raw)
    (root / "COMPLETE").write_text(hashlib.sha256(raw).hexdigest())


def prepared_inputs(tmp_path):
    root = tmp_path / "prepared"
    metadata = root / "chr21/metadata"
    metadata.mkdir(parents=True)
    container = metadata.parent
    pairs = np.asarray([["0", "001"], ["f", "001"], ["0", "003"], ["0", "004"],
                        ["0", "005"], ["0", "006"], ["0", "007"], ["0", "008"]])
    np.save(root / "sample_pairs.npy", pairs)
    np.save(root / "sample_ids.npy", sample_keys(pairs))
    np.save(container / "samples.npy", np.arange(len(pairs), dtype=np.int64))
    for stream in ('data.bin', 'headers.bin', 'counts.bin'):
        (container / stream).write_bytes(b'fixture')
    raw = json.dumps({"binding": {"source_kind": "bed_fixture"}}).encode()
    (container / "manifest.json").write_bytes(raw)
    (container / "COMPLETE").write_text(hashlib.sha256(raw).hexdigest())
    (metadata / "manifest.json").write_text("{}")
    (metadata / "genes.json").write_text(json.dumps([
        {"kind": "coding", "gene_name": "GENE_A", "start": 1, "end": 20},
        {"kind": "noncoding", "gene_name": "GENE_A"},
        {"kind": "ncrna", "gene_name": "RNA_A"}]))
    (metadata / "promoters.json").write_text("[]")
    commit_dataset(root, {"schema_version": 2,
        "annotation_catalog": {}, "annotation_names": [], "qc_path": "annotation/filter",
        "sample_pairs": "sample_pairs.npy", "sample_ids": "sample_ids.npy", "chromosomes": [
            {"name": "21", "container_directory": "chr21", "metadata_directory": "chr21/metadata",
             "gene_catalog": "chr21/metadata/genes.json", "promoter_intervals": "chr21/metadata/promoters.json"}]})
    phenotype, covariate = tmp_path / "phenotypes.csv", tmp_path / "covariates.csv"
    with phenotype.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["FID", "IID", "trait_a", "trait_b"])
        for index, pair in enumerate(pairs):
            writer.writerow([*pair, "NaN" if index == 0 else [1, 3, 8, 2, 6, 4, 9, 5][index],
                             "NaN" if index in (1, 2) else [8, 2, 3, 7, 4, 9, 1, 6][index]])
    with covariate.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["FID", "IID", "adjustment"])
        for index, pair in reversed(list(enumerate(pairs))):
            writer.writerow([*pair, "NaN" if index == 7 else index])
    return root, phenotype, covariate, pairs


class PositionMetadata:
    opens = 0
    n_variants = 2
    manifest = {"analysis":dict(annotation_catalog={},annotation_names=[],qc_path="annotation/filter"),
        "annotation_coverage": dict(n_variants=2,matched_variants=2,missing_snv=0,
        missing_non_snv=0,allow_missing_non_snv=False,available_field="annotation_available")}

    def __init__(self, *args, **kwargs):
        type(self).opens += 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read_field(self, name):
        if name == "annotation_available":
            return np.ones(2, dtype=bool)
        assert name == "position"
        return np.asarray([10, 20], dtype=np.int64)

    def sample_ids(self):
        return sample_keys(np.asarray([["0", "001"], ["f", "001"], ["0", "003"], ["0", "004"],
                        ["0", "005"], ["0", "006"], ["0", "007"], ["0", "008"]]))


def test_public_run_shares_schedule_and_retains_independent_samples(tmp_path, monkeypatch):
    from fudan_wgs_toolkit import run as module, prepared_association
    from fudan_wgs_toolkit.phewas_models import prepare_phewas_model
    from fudan_wgs_toolkit.cache_runtime import portable
    from fudan_wgs_toolkit.phewas_runtime import runtime
    root, phenotype, covariate, pairs = prepared_inputs(tmp_path)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(module.torch.cuda, "mem_get_info", lambda *args: (8*2**30, 8*2**30))
    monkeypatch.setattr(portable, "PortableMetadataReader", PositionMetadata)
    fits, seen = [], []
    def fit(*args, **kwargs):
        fits.append(kwargs.copy())
        kwargs["device"] = "cpu"
        return prepare_phewas_model(*args, **kwargs)
    monkeypatch.setattr(prepared_association, "prepare_phewas_model", fit)
    def shared(configuration, *, model_repository, writer, **kwargs):
        seen.append(configuration)
        assert kwargs["cpu_threads"] == 8
        assert configuration["analysis_options"]["variant_type"] == "SNV"
        descriptors = model_repository.descriptors()
        assert [entry["n"] for entry in descriptors] == [6, 5]
        for index, descriptor in enumerate(descriptors):
            with model_repository.acquire([index], "cpu") as models:
                np.testing.assert_array_equal(models[0].sample_ids,
                    sample_keys(pairs[descriptor["ordered_rows"]]))
        counts = dict.fromkeys(("individual", "coding", "noncoding", "ncrna"), 0)
        for chromosome in configuration["chromosomes"]:
            assert {job["kind"] for job in chromosome["jobs"]} == set(counts)
            for job in chromosome["jobs"]:
                args = (dict(block_identity="identity-contract", ordinal_before=0)
                        if job["kind"] == "individual" else {})
                existing = writer.checkpoint_tile(job["kind"], chromosome["name"],
                    job["checkpoint_job_index"], [0, 1], **args)
                if existing is None:
                    existing = writer.write_tile(job["kind"], chromosome["name"],
                        [(i, [{"pvalue": .25, "pvalue_log10": .6020599913279624}]) for i in (0, 1)],
                        job_index=job["checkpoint_job_index"],
                        **(dict(ordinal_after=1) if job["kind"] == "individual" else {}), **args)
                counts[job["kind"]] += existing["rows"]
                if job["kind"] == "individual":
                    assert job["arguments"]["variant_type"] == "variant"
        return dict(association_rows=counts, output=writer.finish(),
            job_groups=sum(len(chromosome['jobs']) for chromosome in configuration['chromosomes']),
            trait_rows={str(index):{kind: count//2 for kind,count in counts.items()} for index in (0,1)})
    monkeypatch.setattr(runtime, "run_batched_configuration", shared)
    PositionMetadata.opens = 0
    output = tmp_path / "results"
    report = run_WGS_all(phenotype, covariate, root, output_directory=output,
        single_region_size=10, grm="none", rm_nan=False)
    assert report["completed"] and len(seen) == 1 and PositionMetadata.opens == 1
    assert report["eligible_association_tests"] == 10 and len(fits) == 2
    reused = run_WGS_all(phenotype, covariate, root, output_directory=output,
        single_region_size=10, grm="none", rm_nan=False, resume=True)
    assert reused["association_rows"] == report["association_rows"]
    assert len(seen) == 2 and len(fits) == 2
    assert reused["workers"][0]["output"]["reused_blocks"] == 5


def test_snv_gene_default_keeps_all_annotation_weight_columns(tmp_path, monkeypatch):
    import inspect
    from fudan_wgs_toolkit.masks import annotation_phred_matrix, association_weights
    kind = inspect.signature(run_WGS_all).parameters["gene_variant_type"].default
    assert kind == "SNV"
    names = ["CADD", "LINSIGHT", "FATHMM-XF", "aPC.Conservation", "aPC.ProteinFunction",
             "aPC.Epigenetics", "aPC.Distance", "aPC.TF", "aPC.LocalDiversity",
             "aPC.Mappability", "aPC.Deleteriousness"]
    annotations = {name: np.asarray([float(index+1), float(index+2)]) for index,name in enumerate(names)}
    phred, labels = annotation_phred_matrix(annotations, names, variant_type=kind)
    assert phred.shape == (2, 12) and "aPC.LocalDiversity(-)" in labels
    weights = association_weights([.001, .002], phred)
    assert all(value.shape == (2, 26) for value in weights.values())


@pytest.mark.parametrize("changed", ["annotation_catalog", "annotation_names", "qc_path"])
def test_root_annotation_settings_cannot_be_resigned_independently_of_chromosome(tmp_path, monkeypatch, changed):
    from fudan_wgs_toolkit import run as module
    from fudan_wgs_toolkit.cache_runtime import portable
    root, phenotype, covariate, pairs = prepared_inputs(tmp_path)
    dataset = json.loads((root / "dataset.json").read_text())
    dataset["annotation_catalog"] = {"weight_a": "weights/a", "weight_b": "weights/b"}
    dataset["annotation_names"] = ["weight_a", "weight_b"]
    commit_dataset(root, dataset)
    settings = {name:deepcopy(dataset[name]) for name in ("annotation_catalog", "annotation_names", "qc_path")}
    class BoundMetadata(PositionMetadata):
        manifest = dict(deepcopy(PositionMetadata.manifest), analysis=settings)
    if changed == "annotation_catalog":
        dataset[changed]["weight_a"] = "weights/b"
    elif changed == "annotation_names":
        dataset[changed].reverse()
    else:
        dataset[changed] = "another/filter"
    # A self-consistent root hash must not be able to change the chromosome's
    # independently bound weight mapping/order or QC scientific configuration.
    commit_dataset(root, dataset)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(portable, "PortableMetadataReader", BoundMetadata)
    with pytest.raises(ValueError, match="differs from chromosome"):
        run_WGS_all(phenotype,covariate,root,output_directory=tmp_path/"results",grm="none")
    assert not (tmp_path/"results").exists()


@pytest.mark.parametrize("missing", ["annotation_catalog", "annotation_names", "qc_path"])
def test_missing_chromosome_annotation_settings_are_rejected(tmp_path,monkeypatch,missing):
    from fudan_wgs_toolkit import run as module
    from fudan_wgs_toolkit.cache_runtime import portable
    root, phenotype, covariate, pairs = prepared_inputs(tmp_path)
    analysis=deepcopy(PositionMetadata.manifest["analysis"])
    del analysis[missing]
    class Missing(PositionMetadata):
        manifest=dict(deepcopy(PositionMetadata.manifest),analysis=analysis)
    monkeypatch.setattr(module.torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(module.torch.cuda,"device_count",lambda:1)
    monkeypatch.setattr(portable,"PortableMetadataReader",Missing)
    with pytest.raises(ValueError,match="requires explicit"):
        run_WGS_all(phenotype,covariate,root,output_directory=tmp_path/"results",grm="none")


def test_root_sample_axis_cannot_be_reordered_independently_of_chromosome(tmp_path,monkeypatch):
    from fudan_wgs_toolkit import run as module
    from fudan_wgs_toolkit.cache_runtime import portable
    root, phenotype, covariate, pairs=prepared_inputs(tmp_path)
    np.save(root/"sample_pairs.npy",pairs[::-1])
    np.save(root/"sample_ids.npy",sample_keys(pairs[::-1]))
    monkeypatch.setattr(module.torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(module.torch.cuda,"device_count",lambda:1)
    monkeypatch.setattr(portable,"PortableMetadataReader",PositionMetadata)
    with pytest.raises(ValueError,match="chromosome sample keys differ"):
        run_WGS_all(phenotype,covariate,root,output_directory=tmp_path/"results",grm="none")
    assert not (tmp_path/"results").exists()


@pytest.mark.parametrize("kind,analyses,allowed", [("SNV",("individual","coding"),True),
    ("Indel",("individual",),True), ("variant",("coding",),False), ("Indel",("noncoding",),False)])
def test_partial_non_snv_annotations_are_only_allowed_for_covered_gene_types(tmp_path,monkeypatch,kind,analyses,allowed):
    from fudan_wgs_toolkit.cache_runtime import portable
    from fudan_wgs_toolkit.run import _annotation_coverage
    class Partial(PositionMetadata):
        manifest={"annotation_coverage":dict(n_variants=2,matched_variants=1,missing_snv=0,
            missing_non_snv=1,allow_missing_non_snv=True,available_field="annotation_available")}
        def read_field(self,name):
            return np.asarray([True,False]) if name=="annotation_available" else super().read_field(name)
    monkeypatch.setattr(portable,"PortableMetadataReader",Partial)
    entry=dict(metadata_directory=tmp_path,container_directory=tmp_path)
    if allowed:
        assert _annotation_coverage(entry,gene_variant_type=kind,analyses=analyses)["missing_non_snv"]==1
    else:
        with pytest.raises(ValueError,match="requires complete"):
            _annotation_coverage(entry,gene_variant_type=kind,analyses=analyses)
