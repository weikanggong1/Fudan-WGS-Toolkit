"""CPU preparation scheduling and atomic chromosome commit contracts.

Small anonymous PLINK inputs exercise real spawned processes and exact
diploid state/count preservation. They are correctness tests, not benchmarks.
"""
import hashlib
import json
import os
import shutil

import numpy as np
import pytest

from fudan_wgs_toolkit.cache_runtime.portable import PortableMetadataReader
from fudan_wgs_toolkit.cache_runtime.store import Container
from fudan_wgs_toolkit.prepare import (
    prepare_WGS_data, _preparation_schedule, _annotation_sources,
    _source_descriptor, _preflight_chromosome, _convert_chromosome,
)
from test_plink_prepare import source_fixture, dense_cache, interrupt_root_publication


def multiple_sources(tmp_path):
    source, annotations, pairs, states = source_fixture(tmp_path)
    first, second = source / "chromosome21", source / "chromosome22"
    for extension in (".bed", ".fam"):
        shutil.copyfile(str(first) + extension, str(second) + extension)
    second.with_suffix(".bim").write_text(first.with_suffix(".bim").read_text().replace("21 ", "22 "))
    (annotations / "chromosome22.csv").write_text(
        (annotations / "chromosome21.csv").read_text().replace("21,", "22,"))
    path = annotations / "annotations.json"
    manifest = json.loads(path.read_text())
    manifest["chromosomes"].append(dict(name="22", variants="chromosome22.csv"))
    path.write_text(json.dumps(manifest))
    return source, annotations, pairs, states


def assert_states_counts_identical(first, second):
    a, b = Container(first), Container(second)
    np.testing.assert_array_equal(a.samples, b.samples)
    assert a.manifest["m"] == b.manifest["m"]
    for index in range(len(a.index)):
        x, y = a.read_frame(index), b.read_frame(index)
        for name in ("offsets", "sample_index", "state", "reference_alleles",
                     "called_alleles", "half_missing_samples"):
            np.testing.assert_array_equal(x[name], y[name])


def test_actual_processes_share_cpu_budget_and_keep_requested_order(tmp_path):
    source, annotations, pairs, states = multiple_sources(tmp_path)
    output = tmp_path / "parallel"
    selected = pairs[[5, 1, 2]]
    original_environment = {name: os.environ.get(name) for name in
        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    result = prepare_WGS_data(source, output, chromosomes=["22", "21"],
        annotation_directory=annotations, sample_pairs=selected, chunk_size=2, cpu_threads=4)
    schedule = result["parallel_execution"]
    assert schedule["process_start_method"] == "spawn"
    assert schedule["preflight_workers"] == schedule["conversion_workers"] == 2
    assert schedule["conversion_cpu_threads_per_worker"] == 2
    assert schedule["conversion_workers"] * schedule["conversion_cpu_threads_per_worker"] <= 4
    assert schedule["preflight_workspace_aggregate_bound_bytes"] <= schedule["memory_limit_bytes"]
    assert schedule["conversion_workspace_aggregate_bound_bytes"] <= schedule["memory_limit_bytes"]
    assert {metric["conversion_pid"] for metric in result["chromosomes"]}.isdisjoint({os.getpid()})
    assert len({metric["conversion_pid"] for metric in result["chromosomes"]}) == 2
    assert len({metric["preflight_pid"] for metric in result["chromosomes"]}) == 2
    assert all(set(metric["native_thread_environment"].values()) == {"2"}
        for metric in result["chromosomes"])
    assert {name: os.environ.get(name) for name in original_environment} == original_environment
    dataset = json.loads((output / "dataset.json").read_text())
    assert [entry["name"] for entry in dataset["chromosomes"]] == ["22", "21"]
    assert (output / "COMPLETE").read_text() == hashlib.sha256((output / "dataset.json").read_bytes()).hexdigest()
    np.testing.assert_array_equal(np.load(output / "sample_pairs.npy"), selected)
    assert not (output / ".preflight").exists()
    for chromosome in ("22", "21"):
        np.testing.assert_array_equal(dense_cache(output / ("chr" + chromosome)), states[:, [5, 1, 2]])


def test_single_cpu_and_multicpu_preserve_states_counts_and_annotation_bytes(tmp_path):
    source, annotations, pairs, _ = multiple_sources(tmp_path)
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    options = dict(chromosomes=["21", "22"], annotation_directory=annotations,
        sample_pairs=pairs[[2, 0, 5]], variant_indices=np.asarray([0, 2, 3]),
        chunk_size=2, hardlink_annotations=False)
    first = prepare_WGS_data(source, serial, cpu_threads=1, **options)
    second = prepare_WGS_data(source, parallel, cpu_threads=4, **options)
    assert first["parallel_execution"]["process_start_method"] == "none"
    assert first["parallel_execution"]["conversion_workers"] == 1
    assert all(metric["conversion_pid"] == os.getpid() for metric in first["chromosomes"])
    assert first["committed_frames"] == second["committed_frames"] == 4
    for chromosome in ("21", "22"):
        left, right = serial / ("chr" + chromosome), parallel / ("chr" + chromosome)
        assert_states_counts_identical(left, right)
        with PortableMetadataReader(left / "metadata", left) as a, PortableMetadataReader(right / "metadata", right) as b:
            assert a.manifest["analysis"] == b.manifest["analysis"]
            # The exact source witness includes independent preflight progress
            # timings; array/annotation payloads must still be byte-identical.
            assert {name: proof for name, proof in a.manifest["files"].items()
                    if name != "annotation.source.json"} == {
                   name: proof for name, proof in b.manifest["files"].items()
                    if name != "annotation.source.json"}
            assert a.manifest["fields"] == b.manifest["fields"]
            assert a.manifest["annotation_coverage"] == b.manifest["annotation_coverage"]


def test_sample_order_mismatch_rejected_before_annotation_or_genotype_work(tmp_path):
    source, annotations, _, _ = multiple_sources(tmp_path)
    path = source / "chromosome22.fam"
    rows = path.read_text().splitlines()
    rows[0], rows[1] = rows[1], rows[0]
    path.write_text("\n".join(rows) + "\n")
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="order differs"):
        prepare_WGS_data(source, output, annotation_directory=annotations, cpu_threads=4)
    assert not (output / ".raw_annotations").exists()
    assert not (output / "preparation.pending.json").exists()
    assert not list(output.glob("chr*/data.bin"))
    assert not (output / "dataset.json").exists()


def test_raw_annotation_failure_prevents_all_genotype_writers(tmp_path):
    source, annotations, _, _ = multiple_sources(tmp_path)
    path = annotations / "chromosome22.csv"
    path.write_text(path.read_text().replace("22,10,A,C,12.25,PASS\n", ""))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="no exact"):
        prepare_WGS_data(source, output, annotation_directory=annotations, cpu_threads=4)
    assert not list(output.glob("chr*/data.bin"))
    assert not (output / "preparation.pending.json").exists()
    assert not (output / "dataset.json").exists()
    assert not (output / "COMPLETE").exists()


def test_invalid_source_header_rejected_before_any_annotation_worker(tmp_path):
    source, annotations, _, _ = multiple_sources(tmp_path)
    path = source / "chromosome22.bed"
    raw = path.read_bytes()
    path.write_bytes(b"\x6c\x1b\x00" + raw[3:])
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="SNP-major"):
        prepare_WGS_data(source, output, annotation_directory=annotations, cpu_threads=4)
    assert not (output / ".raw_annotations").exists()
    assert not list(output.glob("chr*/data.bin"))
    assert not (output / "dataset.json").exists()


@pytest.mark.parametrize("frame_limit", [2, 3])
def test_global_frame_checkpoint_is_exact_and_resumes_with_processes(tmp_path, frame_limit):
    source, annotations, _, states = multiple_sources(tmp_path)
    output = tmp_path / "checkpoint"
    options = dict(chromosomes=["22", "21"], annotation_directory=annotations,
        chunk_size=2, cpu_threads=4)
    checkpoint = prepare_WGS_data(source, output, max_frames=frame_limit, **options)
    assert checkpoint["committed_frames"] == frame_limit and not checkpoint["completed"]
    assert checkpoint["parallel_execution"]["preflight_workers"] == 2
    assert checkpoint["parallel_execution"]["conversion_workers"] == 1
    assert "global_frame_checkpoint_serial_conversion" in checkpoint["parallel_execution"]["limiting_reasons"]
    assert not (output / "dataset.json").exists()
    journals = [[json.loads(line) for line in path.read_text().splitlines()]
                for path in output.glob("chr*/journal.jsonl")]
    assert sum(map(len, journals)) == frame_limit
    final = prepare_WGS_data(source, output, resume=True, **options)
    assert final["completed"] and final["committed_frames"] == 4 - frame_limit
    assert final["parallel_execution"]["conversion_workers"] == 2
    for chromosome in ("22", "21"):
        np.testing.assert_array_equal(dense_cache(output / ("chr" + chromosome)), states)


def test_shared_original_annotation_file_and_hardlink_fields_keep_source_unchanged(tmp_path):
    source, annotations, _, _ = multiple_sources(tmp_path)
    first = (annotations / "chromosome21.csv").read_text()
    second = (annotations / "chromosome22.csv").read_text().splitlines(keepends=True)
    shared = annotations / "shared.csv"
    shared.write_text(first + "".join(second[1:]))
    stat = shared.stat()
    before = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    manifest_path = annotations / "annotations.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["chromosomes"]:
        entry["variants"] = shared.name
    manifest_path.write_text(json.dumps(manifest))
    result = prepare_WGS_data(source, tmp_path / "prepared", annotation_directory=annotations,
        cpu_threads=4, chunk_size=2, hardlink_annotations=True)
    assert result["completed"]
    stat = shared.stat()
    assert (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == before


def test_workspace_admission_reduces_processes_without_exceeding_total_limit():
    contexts = [dict(conversion_workspace_bytes=8 * 2**20,
        preflight_workspace_base_bytes=2 * 2**20, annotation_row_workspace_bytes=1024,
        annotation_spec={}) for _ in range(4)]
    schedule = _preparation_schedule(contexts, 8, 16 / 1024, max_frames=None)
    assert schedule["requested_chromosome_workers"] == 4
    assert schedule["preflight_workers"] == schedule["conversion_workers"] == 2
    assert "aggregate_workspace_admission" in schedule["limiting_reasons"]
    assert schedule["preflight_workspace_aggregate_bound_bytes"] <= 16 * 2**20
    assert schedule["conversion_workspace_aggregate_bound_bytes"] <= 16 * 2**20


def test_single_chromosome_keeps_total_budget_for_threaded_decode(tmp_path):
    source, annotations, _, _ = source_fixture(tmp_path)
    result = prepare_WGS_data(source, tmp_path / "single", annotation_directory=annotations,
        chunk_size=2, cpu_threads=8)
    assert result["parallel_execution"]["conversion_workers"] == 1
    assert result["parallel_execution"]["conversion_cpu_threads_per_worker"] == 8
    assert result["chromosomes"][0]["decode_cpu_threads"] == 8
    assert result["chromosomes"][0]["conversion_pid"] == os.getpid()


@pytest.mark.parametrize("damaged_axis", ["sample_pairs.npy", "sample_ids.npy"])
def test_resume_rejects_changed_root_axis_before_republishing(tmp_path, monkeypatch, damaged_axis):
    source, annotations, _, _ = source_fixture(tmp_path)
    output = tmp_path / "prepared"
    options = dict(annotation_directory=annotations, chunk_size=2, cpu_threads=1)
    interrupt_root_publication(monkeypatch, source, output, **options)
    path = output / damaged_axis
    axis = np.load(path, allow_pickle=False)
    axis[[0, 1]] = axis[[1, 0]]
    np.save(path, axis, allow_pickle=False)
    original_stream = (output / "chr21/data.bin").read_bytes()
    with pytest.raises(ValueError, match="root sample pairs or identity keys differ"):
        prepare_WGS_data(source, output, resume=True, **options)
    assert (output / "chr21/data.bin").read_bytes() == original_stream
    assert not (output / "dataset.json").exists()
    assert not (output / "COMPLETE").exists()


@pytest.mark.parametrize("damaged_stream", ["data.bin", "counts.bin", "headers.bin"])
def test_resume_checks_all_completed_stream_hashes_before_root_commit(tmp_path, monkeypatch, damaged_stream):
    source, annotations, _, _ = source_fixture(tmp_path)
    output = tmp_path / "prepared"
    options = dict(annotation_directory=annotations, chunk_size=2, cpu_threads=1)
    interrupt_root_publication(monkeypatch, source, output, **options)
    path = output / "chr21" / damaged_stream
    content = bytearray(path.read_bytes())
    content[len(content) // 2] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="Container file checksum mismatch"):
        prepare_WGS_data(source, output, resume=True, **options)
    assert not (output / "dataset.json").exists()
    assert not (output / "COMPLETE").exists()


def test_interprocess_sample_descriptor_is_checked_before_annotation_work(tmp_path):
    source, annotations, _, _ = source_fixture(tmp_path)
    output = tmp_path / "prepared"
    context, _ = _source_descriptor("21", source / "chromosome21",
        _annotation_sources(annotations, ["21"])["21"], output, None, None, 2)
    _preparation_schedule([context], 1, 8, max_frames=None)
    path = context["sample_rows_file"]
    rows = np.load(path, allow_pickle=False)
    rows[[0, 1]] = rows[[1, 0]]
    np.save(path, rows, allow_pickle=False)
    with pytest.raises(RuntimeError, match="descriptor changed"):
        _preflight_chromosome(context)
    assert not (output / ".raw_annotations").exists()
    assert not (output / "chr21").exists()


def test_raw_orientation_checksum_is_verified_before_genotype_writer(tmp_path):
    source, annotations, _, _ = source_fixture(tmp_path)
    output = tmp_path / "prepared"
    context, _ = _source_descriptor("21", source / "chromosome21",
        _annotation_sources(annotations, ["21"])["21"], output, None, None, 2)
    _preparation_schedule([context], 1, 8, max_frames=None)
    context = _preflight_chromosome(context)
    path = output / ".raw_annotations/chr21/reference_is_a1.npy"
    orientation = np.load(path, allow_pickle=False)
    orientation[0] ^= 1
    np.save(path, orientation, allow_pickle=False)
    with pytest.raises(ValueError, match="checksum or size differs"):
        _convert_chromosome(context, cpu_threads=1, hardlink_annotations=False, chunk_size=2)
    assert not (output / "chr21").exists()
    assert not (output / "dataset.json").exists()
