"""Real-source preparation storage, cleanup boundaries and legacy resumes."""
import hashlib
import json

import numpy as np
import pytest

import fudan_wgs_toolkit.prepare as prepare_module
from fudan_wgs_toolkit.cache_runtime.portable import PortableMetadataReader
from fudan_wgs_toolkit.cache_runtime.store import Container
from fudan_wgs_toolkit.prepare import prepare_WGS_data, _cleanup_committed_staging
from fudan_wgs_toolkit.storage_plan import directory_storage
from test_plink_prepare import source_fixture, dense_cache, interrupt_root_publication


@pytest.mark.parametrize("hardlink", [False, True])
def test_completed_preparation_releases_only_staging_and_reports_measured_bytes(tmp_path, hardlink):
    source, annotations, _, expected = source_fixture(tmp_path)
    originals = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                 for directory in (source, annotations) for path in directory.iterdir()}
    output = tmp_path / "prepared"
    result = prepare_WGS_data(source, output, annotation_directory=annotations,
        cpu_threads=1, chunk_size=2, hardlink_annotations=hardlink)
    report = result["storage"]
    assert report == json.loads((output / "storage.report.json").read_text())
    assert report["prepared"] == directory_storage(output, exclude=("storage.report.json",))
    assert report["cleanup"] == dict(completed=True,
        removed=[".preflight", ".raw_annotations"], errors=[])
    assert not (output / ".preflight").exists()
    assert not (output / ".raw_annotations").exists()
    assert report["components"]["raw_annotation_staging"]["files"] == 0
    assert report["phase_boundary_before_cleanup"]["unique_allocated_bytes"] > report["prepared"]["unique_allocated_bytes"]
    assert report["peak"]["actual_bytes"] is None
    assert report["original_inputs_retained"]
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in originals.items())
    np.testing.assert_array_equal(dense_cache(output / "chr21"), expected)
    with PortableMetadataReader(output / "chr21/metadata", output / "chr21") as reader:
        np.testing.assert_array_equal(reader.read_field("position"), [10, 20, 20, 30])
        np.testing.assert_equal(reader.read_field("annotation/cadd"), [12.25, 20, 30, np.nan])
    cache = Container(output / "chr21")
    proof = output / "chr21/metadata/annotation.source.json"
    assert hashlib.sha256(proof.read_bytes()).hexdigest() == cache.manifest["binding"]["raw_annotation_manifest_sha256"]
    with PortableMetadataReader(output / "chr21/metadata", output / "chr21") as reader:
        assert reader.manifest["preparation"]["raw_annotation_manifest"] == proof.name
        assert reader.manifest["files"][proof.name]["sha256"] == hashlib.sha256(proof.read_bytes()).hexdigest()
    assert cache.manifest["journal_format"] == "jsonl-v1"
    assert cache.manifest["sample_encoding"] == "row-delta"
    assert cache.manifest["storage_cap_bytes"] == max(int(cache.manifest["source_bytes"] * .1), 2**20)
    assert "storage_budget_kind" not in cache.manifest
    assert report["genotype_cache_admission"]["includes_metadata"] is False


def test_checkpoint_keeps_required_staging_and_final_resume_releases_it(tmp_path):
    source, annotations, _, expected = source_fixture(tmp_path)
    output = tmp_path / "prepared"
    options = dict(annotation_directory=annotations, cpu_threads=1, chunk_size=2)
    checkpoint = prepare_WGS_data(source, output, max_frames=1, **options)
    assert checkpoint["storage"]["completed"] is False
    assert checkpoint["storage"]["cleanup"]["removed"] == []
    assert (output / ".raw_annotations").is_dir()
    assert (output / ".preflight").is_dir()
    assert not (output / "COMPLETE").exists()
    result = prepare_WGS_data(source, output, resume=True, **options)
    assert result["storage"]["cleanup"]["completed"]
    np.testing.assert_array_equal(dense_cache(output / "chr21"), expected)


def test_root_publication_interruption_keeps_staging_then_resumes(tmp_path, monkeypatch):
    source, annotations, _, expected = source_fixture(tmp_path)
    output = tmp_path / "prepared"
    options = dict(annotation_directory=annotations, cpu_threads=1, chunk_size=2)
    interrupt_root_publication(monkeypatch, source, output, **options)
    original_proof = (output / ".raw_annotations/chr21/manifest.json").read_bytes()
    result = prepare_WGS_data(source, output, resume=True, **options)
    assert result["storage"]["cleanup"]["completed"]
    assert (output / "chr21/metadata/annotation.source.json").read_bytes() == original_proof
    np.testing.assert_array_equal(dense_cache(output / "chr21"), expected)


def test_cleanup_requires_the_exact_durable_root_commit(tmp_path):
    output = tmp_path / "prepared"
    output.mkdir()
    for name in (".raw_annotations", ".preflight"):
        (output / name).mkdir()
        (output / name / "witness").write_text("retain this checkpoint")
    raw = b'{"completed":true}'
    (output / "dataset.json").write_bytes(raw)
    (output / "COMPLETE").write_text("0" * 64)
    with pytest.raises(ValueError, match="commitment differs"):
        _cleanup_committed_staging(output, raw)
    assert all((output / name / "witness").is_file()
               for name in (".raw_annotations", ".preflight"))


def test_existing_absolute_legacy_journal_resumes_without_reencoding(tmp_path, monkeypatch):
    source, annotations, _, expected = source_fixture(tmp_path)
    output = tmp_path / "prepared"
    options = dict(annotation_directory=annotations, cpu_threads=1, chunk_size=2)
    writer = prepare_module.store.Writer
    def legacy_writer(*args, **kwargs):
        kwargs.update(streaming=False, sample_encoding="absolute")
        return writer(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(prepare_module.store, "Writer", legacy_writer)
        checkpoint = prepare_WGS_data(source, output, max_frames=1, **options)
    assert not checkpoint["completed"]
    original_payload = (output / "chr21/data.bin").read_bytes()
    assert (output / "chr21/journal.json").exists()
    pending = json.loads((output / "chr21/manifest.pending.json").read_text())
    assert "journal_format" not in pending and "sample_encoding" not in pending
    result = prepare_WGS_data(source, output, resume=True, **options)
    assert result["completed"]
    assert (output / "chr21/data.bin").read_bytes().startswith(original_payload)
    final = Container(output / "chr21")
    assert "sample_encoding" not in final.manifest and "journal_format" not in final.manifest
    np.testing.assert_array_equal(dense_cache(output / "chr21"), expected)
