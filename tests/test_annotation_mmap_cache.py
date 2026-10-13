"""Exact-selector and bounded derived-cache contracts; not benchmark data."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from fudan_wgs_toolkit.annotation_cache import AnnotationCache
from fudan_wgs_toolkit.annotation_index import CandidateAnnotationIndex
from fudan_wgs_toolkit.cache_runtime.portable import PortableMetadataReader, export_metadata
from fudan_wgs_toolkit.cache_runtime.store import Writer
from fudan_wgs_toolkit.cache_runtime import sparse_codec_fast
from fudan_wgs_toolkit.identity import sample_keys
from fudan_wgs_toolkit.pipeline import AnalysisOptions, PheWASPipeline
from fudan_wgs_toolkit.masks import VariantAnnotations, noncoding_masks, ncRNA_mask, promoter_overlaps


class Source:
    n_samples = 2
    n_variants = 9
    def __init__(self, *, empty=False):
        self.fields = {
            "position": np.asarray([10, 20, 30, 30, 50, 60, 70, 80, 90], dtype=np.int64),
            "variant.id": np.arange(9, dtype=np.int64),
            "qc": np.asarray(["PASS"] * 8 + ["FAIL"]),
            "chromosome": np.asarray(["21", "chr21", "21", "21", "21", "21", "21", "22", "21"]),
            "allele": np.asarray(["A,G"] * 5 + ["A,G,T", "A,C", "A,G", "A,G"]),
            "GENCODE.Category": np.asarray(["upstream", "downstream", "UTR3", "ncRNA_exonic", "intergenic", "UTR5", "ncRNA_splicing", "UTR5", "UTR3"]),
            "GENCODE.Info": np.asarray(["A,B", "B,A", "A(detail)", "A,B,C,D(detail);rest", "A-rest", "A(detail)", "B(detail);rest", "A(detail)", "A(detail)"]),
            "GeneHancer": np.asarray(["", "", "a=b=c=A;detail", "", "a=b=c=B;detail", "", "", "", ""]),
            "CAGE": np.asarray(["", "yes", "yes", "", "yes", "yes", "", "yes", "yes"]),
            "DHS": np.asarray(["", "", "yes", "yes", "", "", "yes", "yes", "yes"]),
        }
        if empty:
            self.fields["GENCODE.Info"][:] = ""
            self.fields["GeneHancer"][:] = ""
    def sample_pairs(self):
        return np.asarray([["a", "s1"], ["a", "s2"]])
    def sample_ids(self):
        return sample_keys(self.sample_pairs())
    def read_field(self, field, rows):
        return self.fields[field][rows]


def prepare(tmp_path, *, empty=False):
    source = Source(empty=empty)
    directory = tmp_path / "prepared"
    writer = Writer(directory, {"source": "unused.genotype"}, np.arange(2, dtype=np.int64), 9, source_bytes=10**8)
    states = np.zeros((9, 2), dtype=np.uint8)
    compact = sparse_codec_fast.compact(states)
    writer.append(states, sparse_codec_fast.integer_counts(*compact, *states.shape))
    writer.finish()
    catalog = {name:name for name in ("GENCODE.Category", "GENCODE.Info", "GeneHancer", "CAGE", "DHS")}
    export_metadata(source, directory, directory / "metadata", list(source.fields), block_size=3,
                    annotation_catalog=catalog, qc_path="qc")
    return source, directory, catalog


def pipeline(directory, catalog, *, verify=True):
    reader = PortableMetadataReader(directory / "metadata", directory, verify_checksums=verify)
    model = SimpleNamespace(n=2, sample_ids=Source().sample_ids(), family="gaussian", n_pheno=1, use_spa=False)
    pipe = PheWASPipeline(reader, [model], qc_path="qc", annotation_catalog=catalog,
                         options=AnalysisOptions(annotation_block_size=3))
    return pipe, reader


def assert_same(left, right):
    assert list(left._groups) == list(right._groups)
    assert left.prepared_categories == right.prepared_categories
    assert left.promoter_signature == right.promoter_signature
    for gene, groups in left._groups.items():
        assert right.categories_for(gene) == tuple(groups)
        for category, rows in groups.items():
            np.testing.assert_array_equal(rows, right.indices(gene, category))
            assert not right.indices(gene, category).flags.writeable


INTERVALS = [("21", 15, 45), ("chr21", 40, 75)]


def test_automatic_hit_skips_decoding_and_matches_all_original_selectors(tmp_path, monkeypatch):
    source, directory, catalog = prepare(tmp_path)
    first, reader = pipeline(directory, catalog)
    expected = first.prepare_annotation_index("21", promoter_intervals=INTERVALS)
    assert reader.reader_metadata["annotation_cache"]["publishes"] == 3
    assert reader.reader_metadata["annotation_cache"]["builds"] == 3
    assert reader.reader_metadata["annotation_cache"]["published_bytes"] > 0
    ref = np.asarray(["A"] * 9)
    alt = np.asarray(["G"] * 5 + ["G,T", "C", "G", "G"])
    raw = VariantAnnotations(source.fields["position"], source.fields["qc"],
                             {k:source.fields[k] for k in catalog}, ref=ref, alt=alt,
                             chromosome=source.fields["chromosome"])
    overlap = promoter_overlaps(raw.position, raw.chromosome, INTERVALS)
    for gene in ("A", "B", "C", "D"):
        masks = noncoding_masks(raw, gene, chromosome="21", promoter_overlap=overlap)
        masks["ncRNA"] = ncRNA_mask(raw, gene, chromosome="21")
        for category, rows in masks.items():
            np.testing.assert_array_equal(expected.indices(gene, category), rows)
    second, other = pipeline(directory, catalog)
    def reject(*args, **kwargs):
        raise AssertionError("persistent index hit decoded original annotation")
    monkeypatch.setattr(other, "_read", reject)
    result = second.prepare_annotation_index("chr21", promoter_intervals=INTERVALS)
    assert_same(expected, result)
    assert other.reader_metadata["annotation_cache"]["hits"] == 1
    rows = [a for groups in result._groups.values() for a in groups.values()]
    assert rows and all(isinstance(a, np.memmap) for a in rows)
    # Every gene view borrows one flat backing map, not another file descriptor.
    assert len({id(a._mmap) for a in rows}) == 1


def test_numeric_arrays_are_readonly_mmaps_and_keep_file_authentication(tmp_path, monkeypatch):
    _, directory, catalog = prepare(tmp_path)
    first, reader = pipeline(directory, catalog)
    base = first._base_mask("21", "SNV")
    codes = first._annotation_categories()
    second, other = pipeline(directory, catalog)
    monkeypatch.setattr(other, "_read", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("decoded source")))
    restored_base = second._base_mask("chr21", "SNV")
    restored_codes = second._annotation_categories()
    np.testing.assert_array_equal(restored_base, base)
    np.testing.assert_array_equal(restored_codes, codes)
    assert isinstance(restored_base, np.memmap) and isinstance(restored_codes, np.memmap)
    assert not restored_base.flags.writeable and not restored_codes.flags.writeable
    assert other.reader_metadata["annotation_cache"]["hits"] == 2


def test_incremental_categories_and_exact_promoter_reference(tmp_path):
    _, directory, catalog = prepare(tmp_path)
    first, _ = pipeline(directory, catalog)
    first.prepare_annotation_index("21", categories=["UTR"], include_ncrna=False)
    first.prepare_annotation_index("21", categories=["ncRNA", "upstream"], include_ncrna=False)
    expected = first.prepare_annotation_index("21", categories=["promoter_CAGE"], include_ncrna=False,
                                              promoter_intervals=INTERVALS)
    second, reader = pipeline(directory, catalog)
    second.prepare_annotation_index("21", categories=["UTR"], include_ncrna=False)
    second.prepare_annotation_index("21", categories=["ncRNA", "upstream"], include_ncrna=False)
    actual = second.prepare_annotation_index("21", categories=["promoter_CAGE"], include_ncrna=False,
                                            promoter_intervals=INTERVALS)
    assert_same(expected, actual)
    assert reader.reader_metadata["annotation_cache"]["hits"] == 3
    with pytest.raises(ValueError, match="different promoter"):
        second.prepare_annotation_index("21", categories=["promoter_CAGE"], include_ncrna=False,
                                        promoter_intervals=[("21", 1, 100)])
    third, other = pipeline(directory, catalog)
    third.prepare_annotation_index("21", categories=["promoter_CAGE"], include_ncrna=False,
                                   promoter_intervals=[("21", 1, 100)])
    assert other.reader_metadata["annotation_cache"].get("publishes", 0) >= 1


def test_corrupt_optional_cache_rebuilds_but_altered_source_fails(tmp_path):
    _, directory, catalog = prepare(tmp_path)
    first, _ = pipeline(directory, catalog)
    expected = first.prepare_annotation_index("21", categories=["UTR"], include_ncrna=False)
    paths = list((directory / "metadata" / ".candidate_annotations").glob("*/rows.npy"))
    assert len(paths) == 1
    with paths[0].open("r+b") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 1]))
    second, reader = pipeline(directory, catalog)
    actual = second.prepare_annotation_index("21", categories=["UTR"], include_ncrna=False)
    assert_same(expected, actual)
    assert reader.reader_metadata["annotation_cache"]["fallback_invalid"] == 1
    # The invalid entry was atomically repaired, so the next reader hits.
    third, other = pipeline(directory, catalog)
    assert_same(expected, third.prepare_annotation_index("21", categories=["UTR"], include_ncrna=False))
    assert other.reader_metadata["annotation_cache"]["hits"] == 1
    name = other.manifest["fields"]["GENCODE.Info"]["file"]
    field = directory / "metadata" / name
    with field.open("r+b") as stream:
        byte = stream.read(1)
        stream.seek(0)
        stream.write(bytes([byte[0] ^ 1]))
    fresh, _ = pipeline(directory, catalog)
    with pytest.raises(ValueError, match="checksum"):
        fresh.prepare_annotation_index("21", categories=["UTR"], include_ncrna=False)


def test_changed_selector_source_binding_and_unverified_reader_never_hit(tmp_path, monkeypatch):
    import fudan_wgs_toolkit.pipeline as module
    _, directory, catalog = prepare(tmp_path)
    first, _ = pipeline(directory, catalog)
    expected = first.prepare_annotation_index("21", categories=["UTR"], include_ncrna=False)
    monkeypatch.setattr(module, "_annotation_semantics", lambda: {"rule": "changed"})
    second, reader = pipeline(directory, catalog)
    assert_same(expected, second.prepare_annotation_index("21", categories=["UTR"], include_ncrna=False))
    assert reader.reader_metadata["annotation_cache"].get("hits", 0) == 0
    assert reader.reader_metadata["annotation_cache"]["publishes"] == 3
    unchecked, other = pipeline(directory, catalog, verify=False)
    assert_same(expected, unchecked.prepare_annotation_index("21", categories=["UTR"], include_ncrna=False))
    assert other.reader_metadata["annotation_cache"]["disabled_unverified_source"] is True
    assert other.reader_metadata["annotation_cache"].get("hits", 0) == 0


def test_empty_candidate_index_roundtrip(tmp_path):
    _, directory, catalog = prepare(tmp_path, empty=True)
    first, _ = pipeline(directory, catalog)
    expected = first.prepare_annotation_index("21", categories=["UTR"], include_ncrna=False)
    assert expected.genes == ()
    second, reader = pipeline(directory, catalog)
    assert_same(expected, second.prepare_annotation_index("21", categories=["UTR"], include_ncrna=False))
    assert reader.reader_metadata["annotation_cache"]["hits"] == 1


def test_streamed_publication_does_not_concatenate_and_lock_is_nonblocking(tmp_path, monkeypatch):
    import fcntl
    source = dict(n_variants=100)
    metrics, admissions = {}, []
    cache = AnnotationCache(tmp_path / "derived", source, lambda: None, metrics,
                            future=lambda size, **kw: admissions.append(size))
    rules = dict(chromosome="21", variant_type="SNV", categories=["UTR"], promoter_signature=None)
    index = CandidateAnnotationIndex("21", "SNV", prepared_categories={"UTR"})
    index._groups = {"A":{"UTR":np.array([1, 2], np.int64)}, "B":{"UTR":np.array([2, 3], np.int64)}}
    monkeypatch.setattr(np, "concatenate", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("full rows copy")))
    cache.save_index(rules, index)
    assert_same(index, cache.load_index(rules))
    assert admissions and max(admissions) <= 80 * 2**20 and admissions[-1] == 0
    lock_rules = dict(chromosome="22")
    target = cache._path(cache._binding("array:blocked", lock_rules))
    lock_path = cache.directory / (target.name + ".lock")
    with lock_path.open("a+b") as stream:
        lock_path.chmod(0o600)
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache.save_array("blocked", lock_rules, np.array([1], dtype=np.uint8))
    assert metrics["lock_busy"] == 1 and not target.exists()


def test_readonly_cache_directory_safely_falls_back(tmp_path, monkeypatch):
    _, directory, catalog = prepare(tmp_path)
    original_mkdir = Path.mkdir
    def readonly(path, *args, **kwargs):
        if path.name == ".candidate_annotations":
            raise PermissionError("readonly prepared source")
        return original_mkdir(path, *args, **kwargs)
    monkeypatch.setattr(Path, "mkdir", readonly)
    pipe, reader = pipeline(directory, catalog)
    result = pipe.prepare_annotation_index("21", categories=["UTR"], include_ncrna=False)
    assert result.indices("A", "UTR").tolist() == [2]
    assert reader.reader_metadata["annotation_cache"]["fallback_write"] == 3
    assert reader.reader_metadata["annotation_cache"]["fallback_read_only"] == 3
    assert reader.reader_metadata["annotation_cache"]["builds"] == 3
    assert not (directory / "metadata" / ".candidate_annotations").exists()


def test_valid_changed_annotation_manifest_cannot_reuse_old_candidates(tmp_path):
    _, directory, catalog = prepare(tmp_path)
    first, reader = pipeline(directory, catalog)
    first.prepare_annotation_index("21", categories=["UTR"], include_ncrna=False)
    manifest_path = directory / "metadata" / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    name = manifest["fields"]["GENCODE.Info"]["file"]
    field = directory / "metadata" / name
    payload = field.read_bytes().replace(b"A(detail)", b"Z(detail)", 1)
    field.write_bytes(payload)
    manifest["files"][name]["sha256"] = hashlib.sha256(payload).hexdigest()
    raw = json.dumps(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(raw)
    (manifest_path.parent / "COMPLETE").write_text(hashlib.sha256(raw).hexdigest())
    second, other = pipeline(directory, catalog)
    actual = second.prepare_annotation_index("21", categories=["UTR"], include_ncrna=False)
    assert actual.indices("Z", "UTR").tolist() == [2]
    assert actual.indices("A", "UTR").tolist() == []
    assert other.reader_metadata["annotation_cache"].get("hits", 0) == 0


def test_cache_identity_is_rechecked_after_row_validation(tmp_path, monkeypatch):
    import fudan_wgs_toolkit.annotation_cache as module
    metrics = {}
    cache = AnnotationCache(tmp_path / "derived", dict(n_variants=100), lambda: None, metrics)
    rules = dict(chromosome="21", variant_type="SNV", categories=["UTR"], promoter_signature=None)
    index = CandidateAnnotationIndex("21", "SNV", prepared_categories={"UTR"})
    index._groups = {"A":{"UTR":np.array([1, 2], np.int64)}}
    cache.save_index(rules, index)
    original = module._validate_vector
    path = cache._path(cache._binding("index", rules)) / "rows.npy"
    changed = False
    def mutate_after_validation(array, **kwargs):
        nonlocal changed
        original(array, **kwargs)
        if kwargs.get("ordered") and not changed:
            changed = True
            with path.open("r+b") as stream:
                stream.seek(-1, 2)
                byte = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([byte[0] ^ 1]))
    monkeypatch.setattr(module, "_validate_vector", mutate_after_validation)
    assert cache.load_index(rules) is None
    assert metrics["fallback_invalid"] == 1 and metrics.get("hits", 0) == 0


def test_semantically_invalid_entry_is_repaired_even_with_matching_checksums(tmp_path):
    metrics = {}
    cache = AnnotationCache(tmp_path / "derived", dict(n_variants=100), lambda: None, metrics)
    rules = dict(chromosome="21", variant_type="SNV", categories=["UTR"], promoter_signature=None)
    index = CandidateAnnotationIndex("21", "SNV", prepared_categories={"UTR"})
    index._groups = {"A":{"UTR":np.array([1, 2], np.int64)}}
    cache.save_index(rules, index)
    directory = cache._path(cache._binding("index", rules))
    array = np.load(directory / "rows.npy", mmap_mode="r+")
    array[:] = [2, 1]
    array.flush()
    del array
    manifest = json.loads((directory / "manifest.json").read_bytes())
    manifest["files"]["rows.npy"]["sha256"] = hashlib.sha256((directory / "rows.npy").read_bytes()).hexdigest()
    raw = json.dumps(manifest, sort_keys=True).encode()
    (directory / "manifest.json").write_bytes(raw)
    (directory / "COMPLETE").write_text(hashlib.sha256(raw).hexdigest())
    assert cache.load_index(rules) is None
    cache.save_index(rules, index)
    assert_same(index, cache.load_index(rules))
    assert metrics["publishes"] == 2
