"""Direct synthetic preparation retains the production storage proofs.

All inputs are invented. These checks exercise byte compatibility, six-state
counts, corruption rejection and the public prepared-data reader contracts;
they are not scientific or performance benchmarks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fudan_wgs_toolkit.cache_runtime import codec, sparse_codec_fast as csr, store
from fudan_wgs_toolkit.cache_runtime.portable import export_metadata, PortableMetadataReader
from fudan_wgs_toolkit.identity import sample_keys
from fudan_wgs_toolkit.run import _annotation_coverage, _analysis_settings, _dataset


def six_state_calls():
    return np.asarray([
        [0, 1, 2, 3, 4, 5, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [3, 3, 3, 3, 3, 3, 3],
        [5, 4, 3, 2, 1, 0, 5],
    ], dtype=np.uint8)


def independent_counts(calls):
    # Count observed alleles directly by the six-state semantic definitions,
    # independently of the production CSR prefix-sum implementation.
    reference = np.asarray([2, 1, 0, 0, 1, 0], dtype=np.int64)
    called = np.asarray([2, 2, 2, 0, 1, 1], dtype=np.int64)
    return dict(reference_alleles=reference[calls].sum(axis=1, dtype=np.int64),
                called_alleles=called[calls].sum(axis=1, dtype=np.int64),
                half_missing_samples=(calls >= 4).sum(axis=1, dtype=np.int64))


@pytest.mark.parametrize("level", [1, 3])
def test_sparse_encoder_matches_existing_dense_payload_and_original_raw_layout(level):
    calls = six_state_calls()
    arrays = csr.compact(calls)
    encoded, header = csr.encode_compact(*arrays, *map(int, calls.shape), level=level)
    dense_payload, dense_header = csr.encode(calls, level=level)
    assert encoded == dense_payload and header == dense_header
    # These literal arrays describe the original on-disk CSR layout, including
    # the empty second row and the fully missing third row.
    offsets = np.asarray([0, 5, 5, 12, 18], dtype="<u4")
    samples = np.asarray([1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6,
                          0, 1, 2, 3, 4, 6], dtype="<u2")
    values = np.asarray([1, 2, 3, 4, 5, 3, 3, 3, 3, 3, 3, 3,
                         5, 4, 3, 2, 1, 5], dtype=np.uint8)
    raw = offsets.tobytes() + samples.tobytes() + values.tobytes()
    assert encoded == codec._zstd().ZstdCompressor(level=level, write_content_size=True).compress(raw)
    assert header["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    for actual, expected in zip(csr.decode_compact(encoded, header), (offsets, samples, values)):
        np.testing.assert_array_equal(actual, expected)
    assert independent_counts(calls)["called_alleles"].tolist() == [10, 14, 0, 9]


@pytest.mark.parametrize("n", [65535, 65536, 65537])
def test_direct_encoder_uses_correct_sample_index_width_without_dense_allocation(n):
    dtype = "<u2" if n < 65536 else "<u4"
    arrays = (np.asarray([0, 0, 2], dtype="<u4"),
              np.asarray([0, n - 1], dtype=dtype), np.asarray([4, 5], dtype=np.uint8))
    payload, header = csr.encode_compact(*arrays, 2, n)
    assert header["sample_dtype"] == np.dtype(dtype).str
    counts = csr.integer_counts(*csr.decode_compact(payload, header), 2, n)
    assert counts["reference_alleles"].tolist() == [2*n, 2*n - 3]
    assert counts["called_alleles"].tolist() == [2*n, 2*n - 2]
    assert counts["half_missing_samples"].tolist() == [0, 2]


@pytest.mark.parametrize("change", ["zero_state", "duplicate_sample", "bad_offset", "wrong_width", "sample_bounds", "negative_geometry", "byte_budget"])
def test_direct_encoder_rejects_invalid_mutable_csr(change):
    offsets, samples, values = [value.copy() for value in csr.compact(six_state_calls())]
    m, n, options = 4, 7, {}
    if change == "zero_state":
        values[0] = 0
    elif change == "duplicate_sample":
        samples[1] = samples[0]
    elif change == "bad_offset":
        offsets[-1] -= 1
    elif change == "wrong_width":
        samples = samples.astype("<u4")
    elif change == "sample_bounds":
        samples[-1] = n
    elif change == "negative_geometry":
        m = -1
    else:
        options["max_raw_bytes"] = 1
    with pytest.raises((ValueError, MemoryError)):
        csr.encode_compact(offsets, samples, values, m, n, **options)


def _write_calls(path, calls, *, compact, synthetic=False, chunk=2, streaming=False):
    options = (dict(source_bytes=None, storage_limit_bytes=8*2**20)
               if synthetic else dict(source_bytes=100_000_000))
    writer = store.Writer(path, dict(synthetic=True, generator="test"),
                          np.arange(calls.shape[1], dtype=np.int64), len(calls),
                          chunk=chunk, streaming=streaming, **options)
    for first in range(0, len(calls), chunk):
        block = calls[first:first + chunk]
        counts = independent_counts(block)
        if compact:
            writer.append_compact(*csr.compact(block), counts)
        else:
            writer.append(block, counts)
    return writer.finish()


def test_direct_writer_preserves_dense_stream_bytes_and_all_six_state_counts(tmp_path):
    calls = six_state_calls()
    dense, direct = tmp_path / "dense", tmp_path / "direct"
    _write_calls(dense, calls, compact=False)
    _write_calls(direct, calls, compact=True)
    for name in ("data.bin", "headers.bin", "counts.bin", "index.npy", "samples.npy", "journal.json"):
        assert (dense / name).read_bytes() == (direct / name).read_bytes()
    reader = store.Container(direct)
    assert reader.verify_streams()
    for index in range(2):
        frame = reader.read_frame(index)
        expected = independent_counts(calls[index * 2:index * 2 + 2])
        for name, value in expected.items():
            np.testing.assert_array_equal(frame[name], value)
        rebuilt = np.zeros((frame["m"], frame["n"]), dtype=np.uint8)
        for row in range(frame["m"]):
            first, stop = frame["offsets"][row:row + 2]
            rebuilt[row, frame["sample_index"][first:stop]] = frame["state"][first:stop]
        np.testing.assert_array_equal(rebuilt, calls[index * 2:index * 2 + 2])


def test_direct_writer_checks_counts_before_commit_and_resumes_exact_binding(tmp_path):
    path, calls = tmp_path / "resume", six_state_calls()
    binding = dict(synthetic=True, generator="test")
    options = dict(source_bytes=None, storage_limit_bytes=8*2**20, chunk=2)
    samples = np.arange(calls.shape[1], dtype=np.int64)
    writer = store.Writer(path, binding, samples, len(calls), **options)
    wrong = independent_counts(calls[:2])
    wrong["reference_alleles"][0] += 1
    with pytest.raises(ValueError, match="integer summary"):
        writer.append_compact(*csr.compact(calls[:2]), wrong)
    assert writer.next_start == 0 and json.loads((path / "journal.json").read_text()) == []
    assert all((path / name).stat().st_size == 0 for name in ("data.bin", "counts.bin", "headers.bin"))
    writer.append_compact(*csr.compact(calls[:2]), independent_counts(calls[:2]))
    with (path / "data.bin").open("ab") as stream:
        stream.write(b"uncommitted tail")
    resumed = store.Writer(path, binding, samples, len(calls), **options)
    assert resumed.next_start == 2
    resumed.append_compact(*csr.compact(calls[2:]), independent_counts(calls[2:]))
    manifest = resumed.finish()
    assert manifest["source_bytes"] is None
    assert manifest["storage_budget_kind"] == "explicit_synthetic_bytes"
    assert manifest["storage_cap_bytes"] == 8*2**20
    assert store.Container(path).verify_streams()
    with pytest.raises(FileExistsError):
        store.Writer(path, binding, samples, len(calls), **options)


@pytest.mark.parametrize("binding,limit", [({}, 2**20), ({"synthetic": False}, 2**20),
                                         ({"synthetic": "true"}, 2**20),
                                         ({"synthetic": True}, 0),
                                         ({"synthetic": True}, True)])
def test_explicit_storage_budget_is_only_admitted_for_synthetic_sources(tmp_path, binding, limit):
    path = tmp_path / "invalid"
    with pytest.raises(ValueError):
        store.Writer(path, binding, np.arange(7, dtype=np.int64), 2,
                     storage_limit_bytes=limit)
    assert not path.exists()


def test_production_storage_fraction_default_is_unchanged_and_cannot_be_bypassed(tmp_path):
    writer = store.Writer(tmp_path / "production", {}, np.arange(7, dtype=np.int64), 2,
                          source_bytes=100_000_000)
    assert writer.spec["storage_cap_bytes"] == 10_000_000
    assert writer.spec["source_bytes"] == 100_000_000
    assert "storage_budget_kind" not in writer.spec
    with pytest.raises(ValueError):
        store.Writer(tmp_path / "no_source", {}, np.arange(7, dtype=np.int64), 2)
    bounded = store.Writer(tmp_path / "bounded", {"synthetic": True},
                           np.arange(7, dtype=np.int64), 2, source_bytes=None,
                           storage_limit_bytes=1, chunk=2)
    with pytest.raises(MemoryError):
        bounded.append_compact(*csr.compact(six_state_calls()[:2]),
                               independent_counts(six_state_calls()[:2]))


def test_direct_frame_corruption_is_rejected_by_unchanged_reader(tmp_path):
    path = tmp_path / "corrupted"
    _write_calls(path, six_state_calls(), compact=True, synthetic=True)
    with (path / "data.bin").open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 1]))
    reader = store.Container(path)
    with pytest.raises(ValueError, match="checksum"):
        reader.verify_streams()
    with pytest.raises(ValueError, match="SHA"):
        reader.read_frame(0)


def test_streaming_writer_preserves_readable_frame_and_index_bytes_without_row_list(tmp_path):
    calls = six_state_calls()
    ordinary, streamed = tmp_path / "ordinary", tmp_path / "streamed"
    _write_calls(ordinary, calls, compact=True, synthetic=True)
    manifest = _write_calls(streamed, calls, compact=True, synthetic=True, streaming=True)
    assert manifest["journal_format"] == "jsonl-v1"
    assert not (streamed / "journal.json").exists()
    rows = [json.loads(line) for line in (streamed / "journal.jsonl").read_bytes().splitlines()]
    assert [row["start"] for row in rows] == [0, 2]
    for name in ("data.bin", "headers.bin", "counts.bin", "index.npy", "samples.npy"):
        assert (ordinary / name).read_bytes() == (streamed / name).read_bytes()
    reader = store.Container(streamed)
    assert isinstance(reader.index, np.memmap) and not reader.index.flags.writeable
    assert reader.verify_streams()
    for index in range(2):
        np.testing.assert_array_equal(reader.read_frame(index)["called_alleles"],
                                      independent_counts(calls[index*2:index*2+2])["called_alleles"])


@pytest.mark.parametrize("torn", [b'{"start":', b'{"start":2}'])
def test_streaming_recovery_discards_only_unterminated_tail_and_uncommitted_streams(tmp_path, torn):
    path, calls = tmp_path / "resume_streamed", six_state_calls()
    binding = dict(synthetic=True, generator="streaming_test")
    samples = np.arange(calls.shape[1], dtype=np.int64)
    options = dict(storage_limit_bytes=8*2**20, chunk=2, streaming=True)
    writer = store.Writer(path, binding, samples, len(calls), **options)
    writer.append_compact(*csr.compact(calls[:2]), independent_counts(calls[:2]))
    assert writer.rows is None and writer.frame_count == 1
    original_journal = (path / "journal.jsonl").read_bytes()
    original_sizes = {name: (path / name).stat().st_size
                      for name in ("data.bin", "counts.bin", "headers.bin")}
    for name in original_sizes:
        with (path / name).open("ab") as stream:
            stream.write(b"uncommitted frame tail")
    with (path / "journal.jsonl").open("ab") as stream:
        stream.write(torn)
    resumed = store.Writer(path, binding, samples, len(calls), **options)
    assert resumed.next_start == 2 and resumed.frame_count == 1 and resumed.rows is None
    assert (path / "journal.jsonl").read_bytes() == original_journal
    assert all((path / name).stat().st_size == size for name, size in original_sizes.items())
    resumed.append_compact(*csr.compact(calls[2:]), independent_counts(calls[2:]))
    manifest = resumed.finish()
    assert manifest["frames"] == 2 and store.Container(path).verify_streams()
    with pytest.raises(FileExistsError):
        store.Writer(path, binding, samples, len(calls), **options)


@pytest.mark.parametrize("corruption", ["invalid_json", "wrong_keys", "frame_order", "payload"])
def test_streaming_recovery_rejects_complete_committed_record_or_payload_corruption(tmp_path, corruption):
    path, calls = tmp_path / "committed_corruption", six_state_calls()
    samples, binding = np.arange(calls.shape[1], dtype=np.int64), dict(synthetic=True)
    options = dict(storage_limit_bytes=8*2**20, chunk=2, streaming=True)
    writer = store.Writer(path, binding, samples, len(calls), **options)
    writer.append_compact(*csr.compact(calls[:2]), independent_counts(calls[:2]))
    if corruption == "payload":
        with (path / "data.bin").open("r+b") as stream:
            value = stream.read(1)
            stream.seek(0)
            stream.write(bytes([value[0] ^ 1]))
    else:
        row = json.loads((path / "journal.jsonl").read_bytes())
        if corruption == "invalid_json":
            raw = b"not-json\n"
        elif corruption == "wrong_keys":
            row["unknown"] = 1
            raw = store.canonical(row) + b"\n"
        else:
            row["start"] = 1
            raw = store.canonical(row) + b"\n"
        (path / "journal.jsonl").write_bytes(raw)
    with pytest.raises(ValueError):
        store.Writer(path, binding, samples, len(calls), **options)
    assert not (path / "COMPLETE").exists()


def test_streaming_resume_rebuilds_only_unpublished_index_artifacts(tmp_path):
    calls, path = six_state_calls(), tmp_path / "interrupted_finalization"
    samples, binding = np.arange(calls.shape[1], dtype=np.int64), dict(synthetic=True)
    options = dict(storage_limit_bytes=8*2**20, chunk=2, streaming=True)
    writer = store.Writer(path, binding, samples, len(calls), **options)
    for first in range(0, len(calls), 2):
        writer.append_compact(*csr.compact(calls[first:first+2]), independent_counts(calls[first:first+2]))
    (path / "index.npy.tmp").write_bytes(b"torn temporary index")
    (path / "index.npy").write_bytes(b"renamed but unpublished index")
    resumed = store.Writer(path, binding, samples, len(calls), **options)
    assert resumed.next_start == len(calls) and resumed.frame_count == 2
    assert not (path / "index.npy.tmp").exists() and not (path / "index.npy").exists()
    resumed.finish()
    assert store.Container(path).verify_streams()


def test_streaming_empty_physical_axis_still_has_an_ordinary_valid_index(tmp_path):
    path = tmp_path / "empty_axis"
    writer = store.Writer(path, dict(synthetic=True), np.arange(7, dtype=np.int64), 0,
                         storage_limit_bytes=2**20, streaming=True)
    manifest = writer.finish()
    assert manifest["frames"] == 0 and writer.rows is None
    reader = store.Container(path)
    assert len(reader.index) == 0 and reader.verify_streams()


@pytest.mark.parametrize("binding,streaming", [(None, True), ({"synthetic": True}, 1),
    ({"synthetic": True}, "true")])
def test_streaming_path_requires_explicit_binding_and_boolean_option(tmp_path, binding, streaming):
    path = tmp_path / "unadmitted"
    with pytest.raises(ValueError, match="streaming|Streaming|synthetic"):
        store.Writer(path, binding, np.arange(7, dtype=np.int64), 2,
                     source_bytes=100_000_000, streaming=streaming)
    assert not path.exists()


@pytest.mark.parametrize("n", [7, 65535, 65536, 2**32])
def test_row_delta_roundtrip_preserves_absolute_indices_states_and_counts(n):
    dtype = "<u2" if n < 65536 else "<u4"
    offsets = np.asarray([0, 0, 3, 3, 4, 7], dtype="<u4")
    indices = np.asarray([0, 2, n-1, n-1, 1, 3, n-1], dtype=dtype)
    states = np.asarray([1, 2, 3, 4, 5, 1, 2], dtype=np.uint8)
    originals = [value.copy() for value in (offsets, indices, states)]
    payload, header = csr.encode_compact(offsets, indices, states, 5, n,
                                         sample_encoding="row-delta")
    assert header["version"] == 2 and header["sample_encoding"] == "row-delta"
    actual = csr.decode_compact(payload, header)
    for result, expected, source in zip(actual, originals, (offsets, indices, states)):
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(source, expected)
        assert not result.flags.writeable
    expected = csr.integer_counts(*originals, 5, n)
    for name, counts in csr.integer_counts(*actual, 5, n).items():
        np.testing.assert_array_equal(counts, expected[name])


def test_row_delta_is_optional_and_reduces_a_frequent_variant_index_stream():
    n, m = 100000, 4
    rng = np.random.default_rng(1729)
    indices = np.concatenate([np.sort(rng.choice(n, size=20000, replace=False))
                              for _ in range(m)]).astype("<u4")
    offsets = np.arange(m+1, dtype="<u4")*20000
    states = np.ones(len(indices), dtype=np.uint8)
    old, original = csr.encode_compact(offsets, indices, states, m, n, level=3)
    new, transformed = csr.encode_compact(offsets, indices, states, m, n, level=3,
                                         sample_encoding="row-delta")
    assert original["version"] == 1 and "sample_encoding" not in original
    assert len(new) < len(old)
    for first, second in zip(csr.decode_compact(old, original),
                             csr.decode_compact(new, transformed)):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("damage", ["zero_delta", "wrapped_index", "bad_offsets", "unknown_encoding"])
def test_row_delta_rejects_authenticated_but_semantically_invalid_payload(damage):
    offsets = np.asarray([0, 3], dtype="<u4")
    indices = np.asarray([0, 2, 6], dtype="<u2")
    states = np.asarray([1, 4, 5], dtype=np.uint8)
    payload, header = csr.encode_compact(offsets, indices, states, 1, 7,
                                         sample_encoding="row-delta")
    raw = bytearray(codec._zstd().ZstdDecompressor().decompress(payload))
    if damage == "zero_delta":
        raw[10:12] = np.asarray([0], dtype="<u2").tobytes()
    elif damage == "wrapped_index":
        raw[10:12] = np.asarray([65535], dtype="<u2").tobytes()
    elif damage == "bad_offsets":
        raw[:4] = np.asarray([1], dtype="<u4").tobytes()
    else:
        header["sample_encoding"] = "invented-lossy-format"
    payload = codec._zstd().ZstdCompressor(level=1, write_content_size=True).compress(raw)
    header.update(raw_sha256=hashlib.sha256(raw).hexdigest(),
                  compressed_sha256=hashlib.sha256(payload).hexdigest(),
                  compressed_bytes=len(payload))
    header["header_sha256"] = csr._header_sha(header)
    with pytest.raises(ValueError):
        csr.decode_compact(payload, header)


def test_real_source_streaming_row_delta_preserves_fraction_cap_and_resume(tmp_path):
    calls, path = six_state_calls(), tmp_path / "real_source"
    binding = dict(source_format="invented retained source for test")
    samples = np.arange(calls.shape[1], dtype=np.int64)
    options = dict(source_bytes=100_000_000, streaming=True,
                   sample_encoding="row-delta", chunk=2)
    writer = store.Writer(path, binding, samples, len(calls), **options)
    writer.append(calls[:2], independent_counts(calls[:2]))
    assert writer.rows is None and writer.spec["storage_cap_bytes"] == 10_000_000
    assert "storage_budget_kind" not in writer.spec
    with pytest.raises(ValueError, match="binding"):
        store.Writer(path, binding, samples, len(calls), source_bytes=100_000_000,
                     streaming=True, chunk=2)
    resumed = store.Writer(path, binding, samples, len(calls), **options)
    assert resumed.next_start == 2
    resumed.append_compact(*csr.compact(calls[2:]), independent_counts(calls[2:]))
    manifest = resumed.finish()
    assert manifest["sample_encoding"] == "row-delta"
    reader = store.Container(path, expected_source_binding=binding,
                             expected_samples=samples)
    assert isinstance(reader.index, np.memmap) and reader.verify_streams()
    for frame in range(2):
        expected = independent_counts(calls[2*frame:2*frame+2])
        for name, value in expected.items():
            np.testing.assert_array_equal(reader.read_frame(frame)[name], value)
    report = resumed.storage_report()
    assert report["current"] == store.usage(path)
    assert all(report["peak_observed"][name] >= value
               for name, value in report["current"].items())
    (path / "metadata").mkdir()
    (path / "metadata" / "separate-scope.bin").write_bytes(b"metadata")
    assert resumed.storage_report()["current"] == report["current"]


def test_legacy_pending_writer_still_resumes_original_binding_and_absolute_layout(tmp_path):
    calls, path = six_state_calls(), tmp_path / "legacy"
    binding, samples = {}, np.arange(7, dtype=np.int64)
    options = dict(source_bytes=100_000_000, chunk=2)
    writer = store.Writer(path, binding, samples, len(calls), **options)
    writer.append(calls[:2], independent_counts(calls[:2]))
    pending = json.loads((path / "manifest.pending.json").read_text())
    assert "sample_encoding" not in pending and "journal_format" not in pending
    writer = store.Writer(path, binding, samples, len(calls), **options)
    writer.append(calls[2:], independent_counts(calls[2:]))
    assert "sample_encoding" not in writer.finish()
    assert store.Container(path).verify_streams()


def test_disk_free_reserve_rejects_before_creation_and_preserves_committed_frames(tmp_path, monkeypatch):
    from collections import namedtuple
    usage = namedtuple("DiskUsage", "total used free")
    path, calls = tmp_path / "disk_admission", six_state_calls()
    samples, binding = np.arange(7, dtype=np.int64), dict(synthetic=True)
    options = dict(storage_limit_bytes=8*2**20, streaming=True, chunk=2,
                   free_disk_reserve_bytes=64*2**20)
    monkeypatch.setattr(store.shutil, "disk_usage", lambda path: usage(2**30, 2**30, 0))
    with pytest.raises(OSError, match="reserved free"):
        store.Writer(path, binding, samples, len(calls), **options)
    assert not path.exists()
    monkeypatch.setattr(store.shutil, "disk_usage", lambda path: usage(2**30, 0, 2**30))
    writer = store.Writer(path, binding, samples, len(calls), **options)
    writer.append(calls[:2], independent_counts(calls[:2]))
    committed = (path / "journal.jsonl").read_bytes()
    sizes = store.usage(path)
    monkeypatch.setattr(store.shutil, "disk_usage", lambda path: usage(2**30, 2**30, 0))
    with pytest.raises(OSError, match="reserved free"):
        writer.append(calls[2:], independent_counts(calls[2:]))
    assert writer.next_start == 2 and store.usage(path) == sizes
    assert (path / "journal.jsonl").read_bytes() == committed
    assert not (path / "COMPLETE").exists()


class SyntheticMetadata:
    def __init__(self, n=7, availability=None):
        self.n_samples, self.n_variants = n, 4
        self._pairs = np.asarray([["SIMF", "SIMI" + str(index)] for index in range(n)])
        self.fields = {
            "position": np.asarray([101, 202, 303, 404], dtype=np.int64),
            "chromosome": np.asarray(["21"] * 4),
            "variant.id": np.arange(1, 5, dtype=np.int64),
            "allele": np.asarray(["A,C", "C,G", "G,T", "T,A"]),
            "annotation/filter": np.asarray(["PASS"] * 4),
            "annotation/cadd": np.asarray([3., 6., 9., 12.]),
            "annotation_available": np.ones(4, dtype=bool) if availability is None else availability,
        }

    def sample_pairs(self):
        return self._pairs

    def read_field(self, path, indices):
        return self.fields[path][indices]


def complete_coverage():
    return dict(n_variants=4, matched_variants=4, missing_snv=0, missing_non_snv=0,
                allow_missing_non_snv=False, available_field="annotation_available")


def test_direct_metadata_publishes_coverage_once_and_passes_real_dataset_reader(tmp_path):
    root, source = tmp_path / "prepared", SyntheticMetadata()
    chromosome = root / "chr21"
    _write_calls(chromosome, six_state_calls(), compact=True, synthetic=True)
    settings = dict(annotation_catalog={"CADD": "annotation/cadd"},
                    annotation_names=["CADD"], qc_path="annotation/filter")
    exported = export_metadata(source, chromosome, chromosome / "metadata", list(source.fields),
                               block_size=2, annotation_coverage=complete_coverage(), **settings)
    assert exported["annotation_coverage"] == complete_coverage()
    np.save(root / "sample_pairs.npy", source.sample_pairs(), allow_pickle=False)
    np.save(root / "sample_ids.npy", sample_keys(source.sample_pairs()), allow_pickle=False)
    manifest = dict(schema_version=2, sample_identifier_format="fid_iid_json",
                    sample_pairs="sample_pairs.npy", sample_ids="sample_ids.npy", **settings,
                    chromosomes=[dict(name="21", container_directory="chr21", metadata_directory="chr21/metadata")],
                    preparation=dict(completed=True, source_format="direct-synthetic-six-state"))
    document = store.canonical(manifest)
    store.atomic(root / "dataset.json", document)
    store.atomic(root / "COMPLETE", hashlib.sha256(document).hexdigest().encode())
    _, _, loaded, entries, pairs = _dataset(root)
    coverage = _annotation_coverage(entries[0], gene_variant_type="SNV", analyses=("individual",),
                    expected_sample_keys=sample_keys(pairs), expected_analysis=_analysis_settings(loaded))
    assert coverage == complete_coverage()
    with PortableMetadataReader(chromosome / "metadata", chromosome) as reader:
        for field, expected in source.fields.items():
            np.testing.assert_array_equal(reader.read_field(field), expected)


@pytest.mark.parametrize("change", ["false_available", "numeric_available", "wrong_counts", "boolean_count", "missing_field", "allow_missing"])
def test_metadata_rejects_unproven_coverage_before_atomic_publication(tmp_path, change):
    source, coverage = SyntheticMetadata(), complete_coverage()
    if change == "false_available":
        source.fields["annotation_available"][2] = False
    elif change == "numeric_available":
        source.fields["annotation_available"] = np.ones(4, dtype=np.int64)
    elif change == "wrong_counts":
        coverage["matched_variants"] = 3
    elif change == "boolean_count":
        coverage["missing_snv"] = False
    elif change == "missing_field":
        del source.fields["annotation_available"]
    else:
        coverage["allow_missing_non_snv"] = True
    chromosome, metadata = tmp_path / "chr21", tmp_path / "metadata"
    _write_calls(chromosome, six_state_calls(), compact=True, synthetic=True)
    with pytest.raises(ValueError, match="annotation|coverage|availability"):
        export_metadata(source, chromosome, metadata, list(source.fields), block_size=2,
                        annotation_catalog={"CADD": "annotation/cadd"}, annotation_names=["CADD"],
                        annotation_coverage=coverage)
    assert not metadata.exists()
    assert not metadata.with_name(metadata.name + ".partial-" + str(__import__("os").getpid())).exists()


def test_legacy_metadata_without_coverage_keeps_its_previous_schema(tmp_path):
    source, chromosome = SyntheticMetadata(), tmp_path / "legacy"
    _write_calls(chromosome, six_state_calls(), compact=False)
    manifest = export_metadata(source, chromosome, chromosome / "metadata", list(source.fields),
                               annotation_catalog={"CADD": "annotation/cadd"}, annotation_names=["CADD"])
    assert "annotation_coverage" not in manifest
    raw = (chromosome / "metadata/manifest.json").read_bytes()
    assert (chromosome / "metadata/COMPLETE").read_text() == hashlib.sha256(raw).hexdigest()
