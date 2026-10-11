"""Anonymous cache-only reader contracts; fixtures are not benchmarks."""
import builtins
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fudan_wgs_toolkit.cache_runtime import sparse_codec_fast, sparse_decode, store
from fudan_wgs_toolkit.cache_runtime.portable import (
    PortableGenotypeReader, PortableMetadataReader, export_metadata)
from fudan_wgs_toolkit.identity import sample_keys, validate_sample_pairs


class NativeMetadataFixture:
    n_samples = 8
    n_variants = 6

    def __init__(self):
        self.calls = []
        self.fields = {
            "position": np.asarray([10, 20, 20, 40, 50, 60], dtype=np.int64),
            "chromosome": np.asarray(["1"] * 6),
            "variant.id": np.arange(11, 17, dtype=np.int32),
            "allele": np.asarray(["A,C", "C,T,G", "G,A", "AG,A", "T,TC", "C,G"]),
            "annotation/filter": np.asarray(["PASS", None, "PASS", "OTHER", "PASS", "PASS"], dtype=object),
            "annotation/info/Category": np.asarray(["coding", "", None, "untranslated", "coding", "intergenic"], dtype=object),
            "annotation/info/weight": np.asarray([.5, np.nan, 0., 2., 4., 1.]),
            "annotation/info/vector": np.arange(12, dtype=np.float64).reshape(6, 2),
            "annotation/info/ragged": [np.asarray(["a", "b"]), np.asarray([], dtype="U1"),
                np.asarray(["longer_name"]), np.asarray(["c", "d", "e"]),
                np.asarray([], dtype="U1"), np.asarray(["f"])],
            "annotation/info/bytes": np.asarray([b"A", b"BC", b"D", b"E", b"FG", b"H"]),
            "annotation/info/unicode": np.asarray([["α", "脑"], [None, ""],
                ["é", "🧬"], ["á", "短"], ["λγ", None], ["", "終"]], dtype=object),
            "annotation/info/numeric_ragged": [np.asarray([1, 2], dtype=np.int32),
                np.asarray([], dtype=float), np.asarray([3], dtype=np.int32),
                np.asarray([], dtype=float), np.asarray([4, 5], dtype=np.int32),
                np.asarray([6], dtype=np.int32)],
        }

    def sample_ids(self):
        return sample_keys(self.sample_pairs())

    def sample_pairs(self):
        return np.asarray([["0", "%d_%d" % (101 + i, 101 + i)] for i in range(8)])

    def read_field(self, path, indices):
        self.calls.append((path, np.asarray(indices).copy()))
        values = self.fields[path]
        return [values[int(i)] for i in indices] if isinstance(values, list) else values[indices]


@pytest.fixture
def portable(tmp_path):
    native = NativeMetadataFixture()
    # Larger source population, smaller reordered cached population. Public
    # indices must become 0,1,2,3; no original physical rows leak into lookup.
    source_rows = np.asarray([7, 1, 5, 3], dtype=np.int64)
    raw = np.asarray([[0, 1, 2, 3], [4, 0, 5, 1], [0, 0, 0, 0],
                      [2, 1, 4, 0], [1, 5, 0, 3], [0, 1, 1, 0]], dtype=np.uint8)
    cache = tmp_path / "cache"
    writer = store.Writer(cache, {"source": "unavailable-source.genotype"}, source_rows, 6, source_bytes=10**8)
    offsets, rows, states = sparse_codec_fast.compact(raw)
    counts = sparse_codec_fast.integer_counts(offsets, rows, states, *raw.shape)
    writer.append(raw, counts)
    writer.finish()
    metadata = cache / "metadata"
    manifest = export_metadata(native, cache, metadata, list(native.fields), block_size=2,
        annotation_catalog={"Category": "annotation/info/Category", "weight": "annotation/info/weight"},
        annotation_names=["weight"])
    return native, cache, metadata, manifest, raw, source_rows


def test_family_and_individual_identity_is_lossless_and_pair_unique():
    values = np.asarray([["0", "00101"], ["0", "101_102"], ["a", "same"], ["b", "same"]])
    np.testing.assert_array_equal(validate_sample_pairs(values), values)
    assert len(np.unique(sample_keys(values))) == 4
    for values in ([["0", ""]], [["", "a"]], [["0", "a b"]], [["0", "a"], ["0", "a"]]):
        with pytest.raises(ValueError):
            validate_sample_pairs(values)


def test_metadata_chunked_roundtrip_fixed_ragged_missing_and_multiallelic(portable):
    native, cache, metadata, manifest, _, source_rows = portable
    assert manifest["n_samples"] == 4 and manifest["n_source_samples"] == 8
    assert manifest["preparation"]["genotype_read"] is False
    assert all(len(indices) <= 2 for _, indices in native.calls)
    with PortableMetadataReader(metadata, cache) as reader:
        assert reader.n_samples == 4 and reader.n_variants == 6
        np.testing.assert_array_equal(reader.sample_ids(), native.sample_ids()[source_rows])
        np.testing.assert_array_equal(reader.sample_indices(sample_keys([["0", "106_106"], ["0", "108_108"]])), [2, 0])
        selected = np.asarray([5, 1, 3, 0], dtype=np.int64)
        for name, expected in native.fields.items():
            for indices in (None, selected, np.asarray([], dtype=np.int64)):
                observed = reader.read_field(name, indices)
                request = np.arange(6) if indices is None else indices
                if isinstance(expected, list):
                    assert isinstance(observed, list) and len(observed) == len(request)
                    for a, i in zip(observed, request):
                        np.testing.assert_array_equal(a, expected[i])
                else:
                    np.testing.assert_array_equal(observed, expected[request])
        assert isinstance(reader.read_field("position"), np.memmap)
        ref, alt = reader.read_ref_alt([1, 3])
        np.testing.assert_array_equal(ref, ["C", "AG"])
        np.testing.assert_array_equal(alt, ["T,G", "A"])
        np.testing.assert_array_equal(reader.read_field("$num_allele", [1, 3]), [3, 2])
        np.testing.assert_array_equal(reader._array(manifest["source_sample_rows"]), source_rows)
        assert reader.reader_metadata["original_genotype_required"] is False
    with pytest.raises(ValueError, match="closed"):
        reader.read_field("position")


def test_cache_only_adapter_never_imports_sdk_or_stats_original_source(portable, monkeypatch):
    _, cache, _, _, raw, _ = portable
    original_import = builtins.__import__
    def checked_import(name, *args, **kwargs):
        if name == "pygenotype" or name.startswith("pygenotype."):
            raise AssertionError("cache-only runtime imported native SDK")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked_import)
    original_stat = Path.stat
    def checked_stat(path, *args, **kwargs):
        if path.suffix == ".genotype":
            raise AssertionError("cache-only runtime statted original source")
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", checked_stat)
    with PortableGenotypeReader(cache, device="cpu") as reader:
        np.testing.assert_array_equal(reader.sample_indices(sample_keys([["0", "106_106"], ["0", "108_108"], ["0", "102_102"]])), [2, 0, 1])
        variants, samples = np.asarray([5, 1, 3, 0]), np.asarray([2, 0, 1])
        actual = reader.minor_block(variants, samples)
        offsets, rows, states = sparse_codec_fast.compact(raw)
        counts = sparse_codec_fast.integer_counts(offsets, rows, states, *raw.shape)
        control = sparse_decode.prepare(offsets, rows, states,
            counts["reference_alleles"], counts["called_alleles"], 4, variants, samples)
        expected = sparse_decode.dosage_numpy(control).astype(np.float64)
        expected[expected == 3] = np.nan
        observed, _, _, _, _ = actual.trait_dense(np.arange(3), "minor")
        np.testing.assert_array_equal(observed, np.nan_to_num(expected))
        np.testing.assert_array_equal(actual.variant_indices, variants)
        np.testing.assert_array_equal(actual.sample_indices, samples)
        assert hasattr(reader, "iter_effective_minor_blocks")
        assert reader.reader_metadata["analysis_cache"]["genotype_sdk_fallback_count"] == 0
        with pytest.raises(IndexError, match="outside"):
            reader.minor_block([0], [4])


def test_fail_closed_missing_samples_duplicates_incomplete_and_field_mutation(portable):
    _, cache, metadata, manifest, _, _ = portable
    reader = PortableMetadataReader(metadata, cache)
    for values in (["101"], ["108", "108"], [["108"]]):
        with pytest.raises(ValueError):
            reader.sample_indices(values)
    with pytest.raises(KeyError):
        reader.read_field("not_exported")
    with pytest.raises(ValueError, match="duplicate"):
        reader.read_field("position", [0, 0])
    field = metadata / manifest["fields"]["position"]["file"]
    reader.read_field("position")
    with field.open("r+b") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="changed"):
        reader.read_field("position")
    fresh = PortableMetadataReader(metadata, cache)
    with pytest.raises(ValueError, match="checksum"):
        fresh.read_field("position")
    (metadata / "COMPLETE").write_text("bad")
    with pytest.raises(ValueError, match="marker"):
        PortableMetadataReader(metadata, cache)


def test_exact_genotype_manifest_binding_no_silent_reuse(portable):
    native, cache, metadata, manifest, _, _ = portable
    with pytest.raises(FileExistsError):
        export_metadata(native, cache, metadata, list(native.fields))
    document = json.loads((metadata / "manifest.json").read_text())
    document["genotype_manifest_sha256"] = "0" * 64
    raw = json.dumps(document).encode()
    (metadata / "manifest.json").write_bytes(raw)
    (metadata / "COMPLETE").write_text(hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="another"):
        PortableMetadataReader(metadata, cache)


@pytest.mark.parametrize("field", ["annotation/info/Category", "annotation/info/bytes",
                                   "annotation/info/unicode"])
def test_utf8_scalar_decode_uses_readonly_zero_copy_ndarray_views(portable, monkeypatch, field):
    native, cache, metadata, manifest, _, _ = portable
    with PortableMetadataReader(metadata, cache) as reader:
        spec = manifest["fields"][field]
        offsets = reader._array(spec["offsets"]["file"])
        valid = reader._array(spec["valid"]["file"])
        assert isinstance(offsets, np.memmap) and isinstance(valid, np.memmap)
        assert not offsets.flags.writeable and not valid.flags.writeable
        original_asarray = np.asarray
        original_getitem = np.memmap.__getitem__
        views = []

        def checked_asarray(value, *args, **kwargs):
            result = original_asarray(value, *args, **kwargs)
            if value is offsets or value is valid:
                assert type(result) is np.ndarray
                assert not result.flags.writeable and np.shares_memory(result, value)
                assert result.__array_interface__["data"][0] == value.__array_interface__["data"][0]
                views.append(result)
            return result

        def checked_getitem(value, index):
            if value is offsets or value is valid:
                raise AssertionError("UTF8 decoding used a scalar/slice memmap subclass lookup")
            return original_getitem(value, index)

        monkeypatch.setattr(np, "asarray", checked_asarray)
        monkeypatch.setattr(np.memmap, "__getitem__", checked_getitem)
        axes = (None, np.asarray([5, 1, 3, 0]), np.arange(6)[::2][::-1],
                np.asarray([], dtype=np.int64))
        for rows in axes:
            requested = np.arange(6) if rows is None else rows
            observed = reader.read_field(field, rows)
            np.testing.assert_array_equal(observed, native.fields[field][requested])
        assert len(views) == 2*len(axes)
        for view in views:
            with pytest.raises(ValueError, match="read-only"):
                view[0] = view[0]


@pytest.mark.parametrize("component", ["file", "offsets", "valid"])
def test_utf8_views_keep_payload_and_index_checksum_validation(portable, component):
    _, cache, metadata, manifest, _, _ = portable
    spec = manifest["fields"]["annotation/info/Category"]
    name = spec["file"] if component == "file" else spec[component]["file"]
    path = metadata/name
    reader = PortableMetadataReader(metadata, cache)
    reader.read_field("annotation/info/Category", [3, 0])
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match="changed"):
        reader.read_field("annotation/info/Category", [0, 3])
    with PortableMetadataReader(metadata, cache) as fresh:
        with pytest.raises(ValueError, match="checksum"):
            fresh.read_field("annotation/info/Category")
