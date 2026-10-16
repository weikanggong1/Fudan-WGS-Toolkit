"""Preparation contracts: real PLINK encoding, identity and original tables."""
import csv
import gzip
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from fudan_wgs_toolkit.prepare import prepare_WGS_data
from fudan_wgs_toolkit.cache_runtime.store import Container
from fudan_wgs_toolkit.cache_runtime.portable import PortableMetadataReader
from fudan_wgs_toolkit.identity import sample_keys
import fudan_wgs_toolkit.prepare as prepare_module


def source_fixture(tmp_path, *, archive=False):
    source = tmp_path / "source"
    source.mkdir()
    prefix = source / "chromosome21"
    pairs = np.asarray([["0", "001_001"], ["A", "same"], ["B", "same"],
                        ["0", "four"], ["0", "five"], ["0", "six"]])
    prefix.with_suffix(".fam").write_text("".join(" ".join([*pair, "0", "0", "0", "-9"]) + "\n" for pair in pairs))
    # The first locus has A1=ALT. The second has A1=REF; both are decoded
    # against explicit raw reference alleles rather than a fixed A1/A2 rule.
    prefix.with_suffix(".bim").write_text("21 first 0 10 C A\n21 second 0 20 G T\n21 third 0 20 A T\n21 fourth 0 30 T C\n")
    codes = np.asarray([[0, 1, 2, 3, 0, 3], [0, 1, 2, 3, 2, 0],
                        [3, 2, 1, 0, 3, 3], [1, 3, 0, 2, 3, 3]], dtype=np.uint8)
    padded = np.zeros((len(codes), 8), dtype=np.uint8)
    padded[:, :6] = codes
    packed = np.sum(padded.reshape(len(codes), 2, 4) << np.asarray([0, 2, 4, 6]), axis=2).astype(np.uint8)
    prefix.with_suffix(".bed").write_bytes(b"\x6c\x1b\x01" + packed.tobytes())
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    rows = [["21", "1", "A", "C", "0", "PASS"],
            ["21", "10", "A", "C", "12.25", "PASS"],
            ["21", "20", "G", "T", "20", "PASS"],
            ["21", "20", "T", "A", "30", "FAIL"],
            ["21", "30", "C", "T", "", "PASS"],
            ["21", "50", "A", "T", "55", "PASS"]]
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(["CHROM", "POS", "REF", "ALT", "WEIGHT", "QC"])
    writer.writerows(rows)
    if archive:
        with tarfile.open(annotations / "chromosome21.tar.gz", "w:gz") as bundle:
            payload = text.getvalue().encode()
            member = tarfile.TarInfo("part1.csv")
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))
        variants = "chromosome21.tar.gz"
    else:
        (annotations / "chromosome21.csv").write_text(text.getvalue())
        variants = "chromosome21.csv"
    manifest = dict(schema_version=1, format="raw-wgs-annotations-v1",
        annotation_catalog={"CADD": "annotation/cadd"}, annotation_names=["CADD"],
        qc_path="annotation/qc", column_mapping=dict(chromosome="CHROM", position="POS",
            reference="REF", alternate="ALT", **{"annotation/cadd": "WEIGHT", "annotation/qc": "QC"}),
        chromosomes=[dict(name="21", variants=variants)])
    (annotations / "annotations.json").write_text(json.dumps(manifest))
    expected = np.asarray([[2, 3, 1, 0, 2, 0], [0, 3, 1, 2, 1, 0],
                           [0, 1, 3, 2, 0, 0], [3, 0, 2, 1, 0, 0]], dtype=np.uint8)
    return source, annotations, pairs, expected


def dense_cache(path):
    container = Container(path)
    result = np.zeros((container.manifest["m"], container.manifest["n"]), dtype=np.uint8)
    for frame_id in range(len(container.index)):
        frame = container.read_frame(frame_id)
        for local in range(frame["m"]):
            begin, end = frame["offsets"][local:local + 2]
            result[frame["start"] + local, frame["sample_index"][begin:end]] = frame["state"][begin:end]
    return result


def interrupt_root_publication(monkeypatch, source, output, **options):
    """Leave the genuine pre-publication crash state, including resume inputs."""
    atomic = prepare_module.store.atomic
    def fail_publication(path, content):
        if Path(path) == output / "dataset.json":
            raise InterruptedError("injected root publication interruption")
        return atomic(path, content)
    with monkeypatch.context() as patch:
        patch.setattr(prepare_module.store, "atomic", fail_publication)
        with pytest.raises(InterruptedError, match="root publication interruption"):
            prepare_WGS_data(source, output, **options)
    assert not (output / "dataset.json").exists()
    assert not (output / "COMPLETE").exists()
    assert (output / ".raw_annotations").is_dir()


@pytest.mark.parametrize("archive", [False, True])
def test_original_tables_orientation_identity_and_missingness(tmp_path, archive):
    source, annotations, pairs, expected = source_fixture(tmp_path, archive=archive)
    output = tmp_path / "prepared"
    result = prepare_WGS_data(source, output, chromosomes=["21"], annotation_directory=annotations,
                              chunk_size=2, cpu_threads=2)
    assert result["completed"]
    assert np.array_equal(dense_cache(output / "chr21"), expected)
    assert np.array_equal(np.load(output / "sample_pairs.npy"), pairs)
    assert np.array_equal(np.load(output / "sample_ids.npy"), sample_keys(pairs))
    manifest = json.loads((output / "dataset.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["preparation"]["variant_subset"] is False
    with PortableMetadataReader(output / "chr21/metadata", output / "chr21") as reader:
        assert reader.sample_axis_kind == "prepared_population"
        assert np.array_equal(reader.read_field("position"), [10, 20, 20, 30])
        assert np.array_equal(reader.sample_indices(sample_keys(pairs[[2, 1]])), [2, 1])
        weight = reader.read_field("annotation/cadd")
        np.testing.assert_equal(weight, [12.25, 20.0, 30.0, np.nan])
        assert list(reader.read_field("annotation/qc")) == ["PASS", "PASS", "FAIL", "PASS"]
    assert all(path.suffix in (".json", ".jsonl", ".npy", ".bin", ".utf8", "")
               for path in output.rglob("*") if path.is_file())


def test_partial_checkpoint_resume_and_sample_variant_subsets(tmp_path):
    source, annotations, pairs, expected = source_fixture(tmp_path)
    output = tmp_path / "prepared"
    options = dict(chromosomes=["21"], annotation_directory=annotations,
                   sample_pairs=pairs[[5, 1, 2]], variant_indices=np.asarray([0, 2, 3]),
                   chunk_size=2, cpu_threads=1)
    first = prepare_WGS_data(source, output, max_frames=1, **options)
    assert not first["completed"] and not (output / "dataset.json").exists()
    final = prepare_WGS_data(source, output, resume=True, **options)
    assert final["completed"]
    assert np.array_equal(dense_cache(output / "chr21"), expected[[0, 2, 3]][:, [5, 1, 2]])
    dataset = json.loads((output / "dataset.json").read_text())
    assert dataset["preparation"]["variant_subset"] is True
    with PortableMetadataReader(output / "chr21/metadata", output / "chr21") as reader:
        assert reader.manifest["n_source_samples"] == 6
        assert list(reader.read_field("position")) == [10, 20, 30]
    with pytest.raises(FileExistsError, match="immutable"):
        prepare_WGS_data(source, output, resume=True, **options)


@pytest.mark.parametrize("corruption", [None, "completion_marker", "field_bytes"])
def test_resume_verifies_previously_committed_metadata(tmp_path, monkeypatch, corruption):
    source, annotations, _, _ = source_fixture(tmp_path)
    output = tmp_path / "prepared"
    options = dict(annotation_directory=annotations, chunk_size=2,
                   hardlink_annotations=False)
    interrupt_root_publication(monkeypatch, source, output, **options)
    metadata = output / "chr21/metadata"
    if corruption is None:
        result = prepare_WGS_data(source, output, resume=True, **options)
        assert result["completed"] and (output / "dataset.json").exists()
        return
    if corruption == "completion_marker":
        (metadata / "COMPLETE").write_text("0" * 64)
        message = "completion marker differs"
    else:
        manifest = json.loads((metadata / "manifest.json").read_text())
        weight = metadata / manifest["fields"]["annotation/cadd"]["file"]
        changed = bytearray(weight.read_bytes())
        changed[-1] ^= 1
        weight.write_bytes(changed)
        message = "checksum or size differs"
    with pytest.raises(ValueError, match=message):
        prepare_WGS_data(source, output, resume=True, **options)
    assert not (output / "dataset.json").exists()


def explicit_bim_identity(source, annotations, *, missing_indel=False):
    prefix = source / "chromosome21"
    rows = [line.split() for line in prefix.with_suffix(".bim").read_text().splitlines()]
    references = [("A", "C"), ("G", "T"), ("T", "A"), ("C", "T")]
    if missing_indel:
        rows[3][4] = "TCC"
        references[3] = ("C", "TCC")
    for row, (reference, alternate) in zip(rows, references):
        row[1] = f"21:{row[3]}:{reference}:{alternate}"
    prefix.with_suffix(".bim").write_text("".join(" ".join(row)+"\n" for row in rows))
    path = annotations / "annotations.json"
    manifest = json.loads(path.read_text())
    manifest["reference_allele_source"] = "bim_variant_id"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_declared_bim_reference_resolves_opposite_raw_orientation(tmp_path):
    source, annotations, _, expected = source_fixture(tmp_path)
    explicit_bim_identity(source, annotations)
    raw = annotations / "chromosome21.csv"
    raw.write_text(raw.read_text().replace("21,10,A,C,12.25,PASS", "21,10,A,C,12.25,PASS\n21,10,C,A,99,PASS"))
    output = tmp_path / "prepared"
    prepare_WGS_data(source, output, annotation_directory=annotations, chunk_size=2)
    np.testing.assert_array_equal(dense_cache(output / "chr21"), expected)
    with PortableMetadataReader(output / "chr21/metadata", output / "chr21") as reader:
        np.testing.assert_equal(reader.read_field("annotation/cadd"), [12.25, 20, 30, np.nan])
    manifest = json.loads((annotations / "annotations.json").read_text())
    del manifest["reference_allele_source"]
    (annotations / "annotations.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="ambiguous"):
        prepare_WGS_data(source, tmp_path / "undeclared", annotation_directory=annotations)


def test_selected_annotation_axis_does_not_require_unselected_variants(tmp_path):
    source, annotations, _, expected = source_fixture(tmp_path)
    raw = annotations / "chromosome21.csv"
    raw.write_text(raw.read_text().replace("21,30,C,T,,PASS\n", ""))
    output = tmp_path / "selected"
    result = prepare_WGS_data(source, output, annotation_directory=annotations,
                              variant_indices=np.asarray([0, 2]), chunk_size=2)
    np.testing.assert_array_equal(dense_cache(output / "chr21"), expected[[0, 2]])
    assert result["chromosomes"][0]["source_n_variants"] == 4
    validation = result["chromosomes"][0]["axis_validation"]
    assert validation["variants"] == 2 and validation["source_variants"] == 4
    assert validation["full_variant_axis_checked"]
    assert not validation["full_annotation_axis_checked"]
    with PortableMetadataReader(output / "chr21/metadata", output / "chr21") as reader:
        np.testing.assert_array_equal(reader.read_field("variant.id"), [1, 3])
        np.testing.assert_array_equal(reader.read_field("position"), [10, 20])


def test_opt_in_missing_non_snv_retains_calls_and_nan_annotations(tmp_path):
    source, annotations, _, expected = source_fixture(tmp_path)
    path, manifest = explicit_bim_identity(source, annotations, missing_indel=True)
    with pytest.raises(ValueError, match="no exact"):
        prepare_WGS_data(source, tmp_path / "strict", annotation_directory=annotations)
    manifest.update(allow_missing_non_snv=True, default_qc="PASS")
    path.write_text(json.dumps(manifest))
    output = tmp_path / "allowed"
    result = prepare_WGS_data(source, output, annotation_directory=annotations, chunk_size=2)
    np.testing.assert_array_equal(dense_cache(output / "chr21"), expected)
    assert result["chromosomes"][0]["axis_validation"]["missing_non_snv"] == 1
    with PortableMetadataReader(output / "chr21/metadata", output / "chr21") as reader:
        np.testing.assert_array_equal(reader.read_field("annotation_available"), [True, True, True, False])
        np.testing.assert_equal(reader.read_field("annotation/cadd"), [12.25, 20, 30, np.nan])
        assert reader.read_field("allele")[-1] == "C,TCC"
    raw = annotations / "chromosome21.csv"
    raw.write_text(raw.read_text().replace("21,10,A,C,12.25,PASS\n", ""))
    with pytest.raises(ValueError, match="no exact"):
        prepare_WGS_data(source, tmp_path / "missing_snv", annotation_directory=annotations)


@pytest.mark.parametrize("chromosome", ["X", "23", "01", "chr01"])
def test_unsupported_chromosome_is_rejected_before_io(tmp_path, chromosome):
    with pytest.raises(ValueError, match="canonical autosomes"):
        prepare_WGS_data(output_directory=tmp_path / "prepared", annotation_directory=tmp_path,
                         prefixes={chromosome: tmp_path / "absent_source"})


def test_scalar_chromosome_and_explicit_variant_mapping_keep_requested_axis(tmp_path):
    source, annotations, _, expected = source_fixture(tmp_path)
    output = tmp_path / "prepared"
    prepare_WGS_data(source, output, chromosomes="21", annotation_directory=annotations,
                     variant_indices={21: np.asarray([1, 3])}, chunk_size=2)
    np.testing.assert_array_equal(dense_cache(output / "chr21"), expected[[1, 3]])


def test_duplicate_annotation_chromosome_identity_is_rejected(tmp_path):
    source, annotations, _, _ = source_fixture(tmp_path)
    path = annotations / "annotations.json"
    manifest = json.loads(path.read_text())
    manifest["chromosomes"].append(dict(manifest["chromosomes"][0], name="chr21"))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="chromosome names must be unique"):
        prepare_WGS_data(source, tmp_path / "prepared", annotation_directory=annotations)


def test_prepared_annotation_input_and_missing_columns_are_rejected(tmp_path):
    source, annotations, _, _ = source_fixture(tmp_path)
    (annotations / "annotations.json").write_text(json.dumps(dict(schema_version=1, format="portable-six-state-metadata-v1")))
    with pytest.raises(ValueError, match="prepared metadata is not accepted"):
        prepare_WGS_data(source, tmp_path / "output", annotation_directory=annotations)


def test_exact_annotation_join_does_not_impute_unknown_weights(tmp_path):
    source, annotations, _, _ = source_fixture(tmp_path)
    path = annotations / "chromosome21.csv"
    path.write_text(path.read_text().replace("21,10,A,C,12.25", "21,10,A,G,12.25"))
    with pytest.raises(ValueError, match="no exact"):
        prepare_WGS_data(source, tmp_path / "output", annotation_directory=annotations)


def test_bed_padding_and_memory_limit_are_enforced(tmp_path):
    source, annotations, _, _ = source_fixture(tmp_path)
    with pytest.raises(MemoryError, match="workspace"):
        prepare_WGS_data(source, tmp_path / "too_small", annotation_directory=annotations,
                         memory_limit_gib=1e-9)
    bed = source / "chromosome21.bed"
    raw = bytearray(bed.read_bytes())
    raw[4] |= 0xc0
    bed.write_bytes(raw)
    with pytest.raises(ValueError, match="padding"):
        prepare_WGS_data(source, tmp_path / "padding", annotation_directory=annotations,
                         chunk_size=2)


def test_source_change_blocks_resume(tmp_path):
    source, annotations, pairs, _ = source_fixture(tmp_path)
    output = tmp_path / "prepared"
    prepare_WGS_data(source, output, annotation_directory=annotations, chunk_size=2, max_frames=1)
    bed = source / "chromosome21.bed"
    raw = bytearray(bed.read_bytes())
    raw[3] ^= 3
    bed.write_bytes(raw)
    with pytest.raises(ValueError, match="binding differs"):
        prepare_WGS_data(source, output, annotation_directory=annotations, chunk_size=2, resume=True)


def test_independent_gene_and_promoter_csvs_preserve_job_order(tmp_path):
    source, annotations, _, _ = source_fixture(tmp_path)
    (annotations / "genes.csv").write_text("hgnc_symbol,chromosome_name,start_position,end_position\nGENE_B,21,20,30\nGENE_A,21,1.0,10.0\nOTHER,22,1,2\n")
    (annotations / "rna.csv").write_text("chr,ncRNA\n21,RNA_B\n22,OTHER_RNA\n21,RNA_A\n")
    (annotations / "promoters.tsv").write_text("chromosome\tstart\tend\tstrand\tgene_id\n21\t1\t3\t+\tGENE_A\n22\t1\t5\t+\tOTHER\n")
    path = annotations / "annotations.json"
    spec = json.loads(path.read_text())
    spec["chromosomes"][0].update(gene_reference="genes.csv", ncrna_reference="rna.csv",
                                   promoter_intervals="promoters.tsv")
    path.write_text(json.dumps(spec))
    output = tmp_path / "prepared"
    prepare_WGS_data(source, output, annotation_directory=annotations, chunk_size=2)
    genes = json.loads((output / "catalogs/chr21.genes.json").read_text())["jobs"]
    assert [job["kind"] for job in genes] == ["coding", "coding", "noncoding", "noncoding", "ncrna", "ncrna"]
    assert [job["arguments"]["gene_name"] for job in genes] == ["GENE_B", "GENE_A", "GENE_B", "GENE_A", "RNA_B", "RNA_A"]
    assert genes[0]["arguments"]["include_ptv"] is False
    assert "start" not in genes[2]["arguments"] and "end" not in genes[-1]["arguments"]
    assert json.loads((output / "catalogs/chr21.promoters.json").read_text()) == [["21", 1, 3]]
