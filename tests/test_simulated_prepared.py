"""Synthetic prepared-format and calibration invariants, not speed benchmarks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fudan_wgs_toolkit.cache_runtime import sparse_codec_fast as csr, store
from fudan_wgs_toolkit.cache_runtime.portable import PortableMetadataReader
from fudan_wgs_toolkit.masks import (
    VariantAnnotations, coding_masks, noncoding_masks, ncRNA_mask, promoter_overlaps,
)
from fudan_wgs_toolkit.simulation_profile import (
    PROFILE_FORMAT, normalize_profile, resolve_profile, small_profile,
)
from fudan_wgs_toolkit.simulate_prepared import (
    ANNOTATION_CATALOG, ANNOTATION_NAMES, QC_PATH, _FrequencyStream, _Metadata, _counts,
    _mask_plan, _position, _variant_exceptions, build_simulated_prepared,
    plan_simulated_prepared,
)


@pytest.fixture(scope="module")
def direct_prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("direct-prepared")
    report = build_simulated_prepared(root, small_profile(), 1000, ["21"],
        cpu_threads=1, memory_limit_gib=1, chunk_size=73)
    return root, report


def _read_variants(metadata):
    rows = np.arange(metadata.n_variants)
    ref, alt = metadata.read_ref_alt(rows)
    return VariantAnnotations(
        position=metadata.read_field("position"), qc=metadata.read_field(QC_PATH),
        chromosome=metadata.read_field("chromosome"), ref=ref, alt=alt,
        annotations={key: metadata.read_field(path) for key, path in ANNOTATION_CATALOG.items()},
    )


def test_profile_requires_explicit_reference_and_small_is_wholly_artificial(monkeypatch, tmp_path):
    monkeypatch.delenv("FUDAN_WGS_SIMULATION_PROFILE", raising=False)
    monkeypatch.setenv("FUDAN_WGS_SIMULATION_CONFIG", str(tmp_path/"absent.json"))
    with pytest.raises(FileNotFoundError, match="reference calibration"):
        resolve_profile()
    profile, provenance = resolve_profile(preset="small")
    assert [record["name"] for record in profile["chromosomes"]] == list(map(str, range(1, 23)))
    assert profile["reference_n"] == 1000
    assert {entry["variants"] for entry in profile["chromosomes"]} == {1024}
    assert all(len(entry["masks"]) == 15 for entry in profile["chromosomes"])
    assert provenance["kind"] == "artificial_small_preset" and not provenance["calibrated"]
    with pytest.raises(ValueError, match="cannot also"):
        resolve_profile(profile, "small")


def test_profile_resolution_preserves_private_binding_and_relative_user_configuration(monkeypatch, tmp_path):
    profile = small_profile()
    data = json.dumps(profile).encode()
    (tmp_path/"calibration.json").write_bytes(data)
    (tmp_path/"configuration.json").write_text(json.dumps({"calibration_profile": "calibration.json"}))
    monkeypatch.delenv("FUDAN_WGS_SIMULATION_PROFILE", raising=False)
    monkeypatch.setenv("FUDAN_WGS_SIMULATION_CONFIG", str(tmp_path/"configuration.json"))
    observed, provenance = resolve_profile()
    assert observed == profile
    assert provenance["profile_sha256"] == hashlib.sha256(data).hexdigest()
    assert provenance["resolved_from"] == "user_configuration"


@pytest.mark.parametrize("change", [
    lambda p: p.update(reference_n=True),
    lambda p: p["chromosomes"].append(deepcopy(p["chromosomes"][0])),
    lambda p: p["chromosomes"][0]["frequency_bins"][0].update(count=599),
    lambda p: p["chromosomes"][0]["frequency_bins"][0].update(maf=float("nan")),
    lambda p: p["chromosomes"][0]["frequency_bins"][0].update(missing_rate=1.1),
    lambda p: p["chromosomes"][0]["frequency_bins"][0].update(called_alleles_mean=2001),
    lambda p: p["chromosomes"][0]["frequency_bins"][0].update(called_alleles_mean=100, minor_ac_mean=51),
    lambda p: p["chromosomes"][0]["frequency_bins"][0].update(minor_ac_low=6, minor_ac_high=5),
    lambda p: p["chromosomes"][0]["masks"][0].update(m=1),
    lambda p: p["chromosomes"][0]["masks"][0].update(group="real-gene"),
    lambda p: p["chromosomes"][0]["masks"][0].update(category="unknown"),
    lambda p: p["chromosomes"][0]["masks"].append(deepcopy(p["chromosomes"][0]["masks"][0])),
    lambda p: p["chromosomes"][0].update(position_span=10),
    lambda p: p["chromosomes"][0]["frequency_bins"].append("not-a-bin"),
    lambda p: p["chromosomes"][0]["masks"].append("not-a-mask"),
])
def test_invalid_calibration_fails_closed(change):
    profile = small_profile()
    change(profile)
    with pytest.raises(ValueError):
        normalize_profile(profile)


def test_exact_frequency_inventory_is_frame_size_independent():
    counts = [1600, 700, 13, 1200]
    allowed = [True, True, False, False]
    first = _FrequencyStream(counts, allowed, 2100, 123)
    a = np.concatenate([first.take(73) for _ in range(48)] + [first.take(9)])
    second = _FrequencyStream(counts, allowed, 2100, 123)
    b = second.take(sum(counts))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(np.bincount(a, minlength=4), counts)
    assert np.isin(a[:2100], [0, 1]).all()


def test_low_sample_size_preserves_maf_target_instead_of_forcing_mask_mac():
    entry = small_profile()["chromosomes"][0]
    called, minor = _counts(entry, 128, 1000)
    np.testing.assert_array_equal(called, [256]*4)
    np.testing.assert_array_equal(minor, [0, 1, 5, 51])
    assert minor[0] == 0  # Reference singleton bin does not become invented MAC>=2.
    realized = minor/called
    target = np.asarray([item["maf"] for item in entry["frequency_bins"]])
    assert np.max(np.abs(realized-target)) <= .5/256


@pytest.mark.parametrize("n", [128, 65536])
@pytest.mark.parametrize("called,minor", [(256, 0), (255, 7), (254, 30), (0, 0), (1, 0), (3, 1)])
def test_sparse_exception_allele_counts_cover_half_and_full_missing(n, called, minor):
    # Small absolute allele totals also exercise an overwhelmingly missing axis.
    rows, states = _variant_exceptions(n, called, minor, np.random.default_rng(15))
    offsets = np.asarray([0, len(rows)], dtype="<u4")
    counts = csr.integer_counts(offsets, rows, states, 1, n)
    assert rows.dtype == np.dtype("<u2" if n < 65536 else "<u4")
    assert int(counts["called_alleles"][0]) == called
    assert int(counts["reference_alleles"][0]) == called-minor
    assert int(counts["half_missing_samples"][0]) == (2*n-called) % 2


def test_all_fifteen_masks_match_the_mature_selectors_including_indel_ptv(direct_prepared):
    root, report = direct_prepared
    prepared = Path(report["prepared_directory"])
    with PortableMetadataReader(prepared/"chr21/metadata", prepared/"chr21") as metadata:
        variants = _read_variants(metadata)
    promoters = json.loads((prepared/"catalogs/chr21.promoters.json").read_text())
    overlap = promoter_overlaps(variants.position, variants.chromosome, promoters)
    jobs = json.loads((prepared/"catalogs/chr21.genes.json").read_text())["jobs"]
    observed = {}
    for job in jobs:
        args = job["arguments"]
        gene = args["gene_name"]
        group = int(gene.rsplit("_", 1)[1])
        if job["kind"] == "coding":
            masks = coding_masks(variants, gene, args["start"], args["end"],
                variant_type="variant", include_ptv=True, chromosome="21")
            snv_ptv = coding_masks(variants, gene, args["start"], args["end"],
                variant_type="SNV", include_ptv=True, chromosome="21")["ptv"]
            assert len(masks["ptv"]) > len(snv_ptv)
        elif job["kind"] == "noncoding":
            masks = noncoding_masks(variants, gene, promoter_overlap=overlap,
                variant_type="variant", chromosome="21")
        else:
            masks = {"ncRNA": ncRNA_mask(variants, gene, variant_type="variant", chromosome="21")}
        for category, rows in masks.items():
            observed[(job["kind"], group, category)] = len(rows)
    targets = small_profile()["chromosomes"][20]["masks"]
    assert len(observed) == len(targets) == 15
    for target in targets:
        assert observed[(target["kind"], target["group"], target["category"])] == target["m"]
    assert report["mask_counts_exact"] and report["frequency_counts_exact"]


def test_direct_container_is_complete_without_raw_genotypes(direct_prepared):
    root, report = direct_prepared
    dataset_path = Path(report["prepared_directory"])
    assert {p.name for p in root.iterdir()} == {"prepared", "inputs"}
    assert not any(path.suffix in (".bed", ".bim", ".fam", ".gds") for path in root.rglob("*"))
    dataset = json.loads((dataset_path/"dataset.json").read_text())
    assert dataset["schema_version"] == 2
    assert (dataset_path/"COMPLETE").read_text() == hashlib.sha256((dataset_path/"dataset.json").read_bytes()).hexdigest()
    container = store.Container(dataset_path/"chr21")
    assert container.verify_streams()
    assert container.manifest["source_bytes"] is None
    assert container.manifest["storage_budget_kind"] == "explicit_synthetic_bytes"
    assert container.manifest["m"] == 1024 and container.manifest["n"] == 1000
    observed = []
    for i in range(len(container.index)):
        frame = container.read_frame(i)
        observed.extend((frame["called_alleles"]-frame["reference_alleles"]).tolist())
    unique, counts = np.unique(observed, return_counts=True)
    assert dict(zip(unique, counts)) == {1:600, 6:300, 40:100, 400:24}
    causal = np.load(report["causal_dosage_path"], allow_pickle=False)
    assert causal.shape == (1000,) and np.isfinite(causal).all() and causal.std() > 0
    chromosome_report = report["chromosome_reports"][0]
    assert chromosome_report["maf_rounding_max_absolute_error"] == 0


def test_eleven_invented_numeric_weight_channels_are_exported(direct_prepared):
    root, report = direct_prepared
    prepared = Path(report["prepared_directory"])
    assert len(ANNOTATION_NAMES) == 11
    assert ANNOTATION_NAMES[3:] == [f"Synthetic.Score{number:02d}" for number in range(1, 9)]
    with PortableMetadataReader(prepared/"chr21/metadata", prepared/"chr21") as metadata:
        assert metadata.manifest["analysis"]["annotation_names"] == ANNOTATION_NAMES
        columns = [metadata.read_field(ANNOTATION_CATALOG[name]) for name in ANNOTATION_NAMES]
        assert all(values.shape == (1024,) and values.dtype.kind == "f" and
                   np.isfinite(values).all() and (values >= .5).all() and (values < 35).all()
                   for values in columns)
        assert all(not np.array_equal(columns[0], values) for values in columns[1:])
    assert report["chromosome_reports"][0]["annotation_weight_channels"] == 11


def test_positions_are_monotone_unique_invented_and_within_calibrated_span():
    positions = _position(np.arange(10000), 10000, 48000000, 2**100, "21")
    assert positions.min() >= 1 and positions.max() <= 48000000
    assert np.all(np.diff(positions) > 0)
    assert not np.array_equal(positions, _position(np.arange(10000), 10000, 48000000, 1729, "21"))


def test_missing_published_rows_are_unknown_and_inferred_masks_are_reported():
    entry = dict(name="1", masks=[
        dict(kind="coding", group=7, category="plof_ds", m=2),
        dict(kind="coding", group=7, category="ptv_ds", m=2)])
    plan = _mask_plan(entry, 1)
    assert not plan["problems"]
    assert len(plan["targets"]) == 2
    assert plan["assigned_variants"] == 2
    assert any(row["category"] == "plof" and row["inferred_m"] == 2
               for row in plan["inferred_masks"])


def test_absent_disruptive_mask_may_have_more_than_one_variant():
    entry = dict(name="1", masks=[
        dict(kind="coding", group=7, category="plof", m=5),
        dict(kind="coding", group=7, category="plof_ds", m=12),
        dict(kind="coding", group=7, category="ptv_ds", m=20)])
    plan = _mask_plan(entry, 1)
    assert not plan["problems"]
    inferred = {row["category"]: row["inferred_m"] for row in plan["inferred_masks"]}
    assert inferred["disruptive_missense"] == 7
    assert inferred["missense"] == 7 and inferred["ptv"] == 13


@pytest.mark.parametrize("plof_overlap,ptv_overlap", [(3, 3), (3, 1)])
def test_disruptive_splice_overlap_matches_mature_coding_masks(plof_overlap, ptv_overlap):
    targets = dict(plof=10, ptv=9, disruptive_missense=7, missense=12,
        synonymous=4, plof_ds=17-plof_overlap, ptv_ds=16-ptv_overlap)
    entry = dict(name="1", masks=[dict(kind="coding", group=2, category=name, m=count)
                                  for name, count in targets.items()])
    plan = _mask_plan(entry, 1)
    assert not plan["problems"]
    m = plan["assigned_variants"]
    metadata = _Metadata("1", m, np.empty((128, 2), dtype="U1"), plan["segments"], 1729, 2*m)
    rows = np.arange(m)
    alleles = [value.split(",") for value in metadata.read_field("allele", rows)]
    variants = VariantAnnotations(position=100001+2*rows,
        qc=metadata.read_field(QC_PATH, rows), chromosome=["1"]*m,
        ref=[value[0] for value in alleles], alt=[value[1] for value in alleles],
        annotations={name: metadata.read_field(path, rows) for name, path in ANNOTATION_CATALOG.items()})
    job = plan["jobs"][0]["arguments"]
    selected = coding_masks(variants, job["gene_name"], job["start"], job["end"],
                            variant_type="variant", include_ptv=True)
    assert {name: len(indices) for name, indices in selected.items()} == targets


def test_inconsistent_coding_targets_are_audited_instead_of_claimed_exact():
    entry = dict(name="1", masks=[
        dict(kind="coding", group=0, category="plof", m=5),
        dict(kind="coding", group=0, category="disruptive_missense", m=3),
        dict(kind="coding", group=0, category="plof_ds", m=4)])
    assert _mask_plan(entry, 1)["problems"]


def test_resource_preflight_rejects_without_publishing_output(tmp_path):
    root = tmp_path/"not-created"
    with pytest.raises(MemoryError, match="memory_limit_gib"):
        plan_simulated_prepared(root, small_profile(), 1000, ["21"], memory_limit_gib=.001)
    with pytest.raises(MemoryError, match="storage_limit_bytes"):
        plan_simulated_prepared(root, small_profile(), 1000, ["21"], storage_limit_bytes=1)
    assert not root.exists()


def test_profile_without_polymorphic_loci_fails_before_mkdir(tmp_path):
    profile = small_profile()
    for entry in profile["chromosomes"]:
        entry["frequency_bins"] = [dict(count=1024, maf=0, missing_rate=0)]
    with pytest.raises(ValueError, match="no nonconstant locus"):
        build_simulated_prepared(tmp_path/"absent", profile, 1000, ["21"])
    assert not (tmp_path/"absent").exists()
