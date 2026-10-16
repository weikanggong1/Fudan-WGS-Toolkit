"""Synthetic input contracts using real PLINK preparation and mask readers.

These tests contain no cohort-derived inputs. They verify generated file formats,
identity alignment and production input readers; they are not scientific parity
or GPU throughput benchmarks.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from fudan_wgs_toolkit.identity import sample_keys
from fudan_wgs_toolkit.cache_runtime.portable import PortableMetadataReader
from fudan_wgs_toolkit.cache_runtime.store import Container
from fudan_wgs_toolkit.masks import (
    CODING_CATEGORIES_INCL_PTV,
    NONCODING_CATEGORIES,
    VariantAnnotations,
    coding_masks,
    ncRNA_mask,
    noncoding_masks,
    promoter_overlaps,
)
from fudan_wgs_toolkit.prepare import prepare_WGS_data
from fudan_wgs_toolkit.run import (
    _analysis_settings,
    _annotation_coverage,
    _dataset,
    _jobs,
    _validate_binary_phenotypes,
    aligned_phenotype,
    read_csv_inputs,
    run_WGS_all_binanry,
)
from fudan_wgs_toolkit.continuous_run import (
    run_WGS_continuous,
    validate_continuous_values,
)
from fudan_wgs_toolkit.simulate import generate_simulated_data


def _csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        return tuple(reader.fieldnames), list(reader)


def _plink(root, chromosome, n):
    """Decode standard BED two-bit states independently of production code."""
    prefix = Path(root) / 'raw' / f'chr{chromosome}'
    bim = [row.split() for row in prefix.with_suffix('.bim').read_text().splitlines()]
    fam = [row.split() for row in prefix.with_suffix('.fam').read_text().splitlines()]
    assert len(fam) == n
    assert all(len(row) == 6 for row in fam)
    assert all(len(row) == 6 for row in bim)
    assert len({tuple(row[:2]) for row in fam}) == n
    raw = prefix.with_suffix('.bed').read_bytes()
    assert raw[:3] == bytes([0x6c, 0x1b, 0x01])
    stride = (n + 3) // 4
    assert len(raw) == 3 + len(bim) * stride
    packed = np.frombuffer(raw[3:], dtype=np.uint8).reshape(len(bim), stride)
    codes = ((packed[:, :, None] >> np.asarray([0, 2, 4, 6], dtype=np.uint8)) & 3).reshape(len(bim), -1)
    assert np.all(codes[:, n:] == 0), 'BED sample padding must contain zero bits'
    # BIM A1/A2 direction is checked against the independent raw annotation.
    manifest = json.loads((Path(root) / 'annotations' / 'annotations.json').read_text())
    entry = next(item for item in manifest['chromosomes'] if str(item['name']) == str(chromosome))
    variant_path = entry['variants']
    assert isinstance(variant_path, str)
    names, rows = _csv(Path(root) / 'annotations' / variant_path)
    mapping = entry.get('column_mapping', manifest['column_mapping'])
    keys = [mapping[field] for field in ('chromosome', 'position', 'reference', 'alternate')]
    annotations = {(row[keys[0]], int(row[keys[1]]), row[keys[2]], row[keys[3]]): row for row in rows}
    result = np.empty((len(bim), n), dtype=np.uint8)
    for index, row in enumerate(bim):
        chrom, _, _, position, a1, a2 = row
        candidates = [key for key in annotations if key[0] == chrom and key[1] == int(position)
                      and {key[2], key[3]} == {a1, a2}]
        assert len(candidates) == 1
        alternate = candidates[0][3]
        # 00=A1/A1, 01=missing, 10=heterozygote, 11=A2/A2.
        lookup = [2, 3, 1, 0] if a1 == alternate else [0, 3, 1, 2]
        result[index] = np.asarray(lookup, dtype=np.uint8)[codes[index, :n]]
    return np.asarray([row[:2] for row in fam]), bim, result


def _prepared_dosages(directory):
    """Reconstruct all committed CSR frames without generator helper code."""
    container = Container(directory)
    values = np.zeros((container.manifest['m'], container.manifest['n']), dtype=np.uint8)
    for index in range(len(container.index)):
        frame = container.read_frame(index)
        for variant in range(frame['m']):
            begin, end = frame['offsets'][variant:variant + 2]
            values[frame['start'] + variant, frame['sample_index'][begin:end]] = frame['state'][begin:end]
    return values


def _artifact_hashes(root):
    # Generation receipts contain output paths and timings. Compare only the
    # actual scientific inputs to establish seed and scheduling independence.
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for subdirectory in ('raw', 'inputs', 'annotations')
            for path in (Path(root) / subdirectory).rglob('*') if path.is_file()}


@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic_input') / 'data'
    report = generate_simulated_data(root, n=1000, variants_per_chromosome={'21': 256, '22': 64},
        chromosomes=[21, 22], seed=90210, n_binary_traits=3, n_continuous_traits=3,
        phenotype_missing_rate=0.12, cpu_threads=2, chunk_size=32,
        genotype_missing_rate=0, memory_limit_gib=1)
    return root, report


def test_plink_padding_and_preparation_roundtrip_preserve_every_genotype(tmp_path):
    root = tmp_path / 'unaligned_samples'
    report = generate_simulated_data(root, n=131, variants_per_chromosome=67,
        genotype_missing_rate=0.04, cpu_threads=1, chunk_size=17,
        memory_limit_gib=1)
    pairs, bim, decoded = _plink(root, '21', 131)
    assert len(bim) == 67
    assert set(np.unique(decoded)) == {0, 1, 2, 3}
    assert all(int(row[3]) > 0 for row in bim)
    assert np.all(np.diff([int(row[3]) for row in bim]) > 0)
    assert all(len(row[4]) == len(row[5]) == 1 and row[4] != row[5] for row in bim)
    prepared = Path(report['prepared_directory'])
    np.testing.assert_array_equal(np.load(prepared / 'sample_pairs.npy', allow_pickle=False), pairs)
    np.testing.assert_array_equal(_prepared_dosages(prepared / 'chr21'), decoded)
    assert not any(path.suffix.lower() in {'.rds', '.rdata', '.gds'} for path in root.rglob('*'))


def test_seed_is_independent_of_chunk_size_and_cpu_scheduling(tmp_path):
    common = dict(n=131, variants_per_chromosome={'22': 35, '21': 67}, chromosomes=[22, 21],
        seed=9901, genotype_missing_rate=0.03, phenotype_missing_rate=0.1, prepare=False)
    first, second, changed = (tmp_path / name for name in ('serial', 'parallel', 'other_seed'))
    generate_simulated_data(first, chunk_size=7, cpu_threads=1, **common)
    generate_simulated_data(second, chunk_size=32, cpu_threads=2, **common)
    assert _artifact_hashes(first) == _artifact_hashes(second)
    generate_simulated_data(changed, chunk_size=7, cpu_threads=1, **dict(common, seed=9902))
    assert (first / 'raw/chr21.bed').read_bytes() != (changed / 'raw/chr21.bed').read_bytes()
    assert (first / 'inputs/covariates.csv').read_bytes() != (changed / 'inputs/covariates.csv').read_bytes()
    assert not (first / 'prepared').exists()


def test_separate_trait_families_and_available_case_alignment(generated):
    root, report = generated
    prepared = Path(report['prepared_directory'])
    pairs = np.load(prepared / 'sample_pairs.npy', allow_pickle=False)
    cov_fields, cov_rows = _csv(report['covariate_csv'])
    assert cov_fields == ('FID', 'IID', 'age', 'sex', *(f'PC{i}' for i in range(1, 21)))
    assert len(cov_rows) == 1000
    covariates = None
    for family, field in (('binomial', 'binary_phenotype_csv'), ('gaussian', 'continuous_phenotype_csv')):
        phenotypes, covariates = read_csv_inputs(report[field], report['covariate_csv'])
        np.testing.assert_array_equal(phenotypes.sample_pairs, pairs)
        np.testing.assert_array_equal(covariates.sample_pairs, pairs)
        assert phenotypes.values.shape == (1000, 3)
        assert covariates.values.shape == (1000, 22)
        assert np.isfinite(covariates.values).all()
        missing = ~np.isfinite(phenotypes.values)
        assert np.all(missing.sum(axis=0) > 0)
        assert not np.array_equal(missing[:, 0], missing[:, 1])
        if family == 'binomial':
            assert _validate_binary_phenotypes(phenotypes) == dict.fromkeys(phenotypes.names, family)
            for column in phenotypes.values.T:
                assert set(column[np.isfinite(column)]) == {0, 1}
        else:
            assert validate_continuous_values(phenotypes.names, phenotypes.values)[0] == dict.fromkeys(phenotypes.names, family)
            assert np.any(phenotypes.values[np.isfinite(phenotypes.values)] % 1 != 0)
        for index, trait in enumerate(phenotypes.names):
            identifiers, y, x, _, counts = aligned_phenotype(
                phenotypes, covariates, pairs, trait, rm_nan=False,
                family=family, binary_only=family == 'binomial')
            keep = np.isfinite(phenotypes.values[:, index])
            np.testing.assert_array_equal(identifiers, sample_keys(pairs[keep]))
            np.testing.assert_array_equal(y, phenotypes.values[keep, index])
            assert x.shape[0] == keep.sum()
            assert counts['imputed_n'] == 0
            assert counts['analysis_samples'] == int(keep.sum())


def test_prepared_data_passes_both_workflows_actual_reader_contracts(generated, tmp_path):
    root, report = generated
    assert report['completed'] is True
    for name, function, phenotype in (
        ('binary_kwargs', run_WGS_all_binanry, 'binary_phenotype_csv'),
        ('continuous_kwargs', run_WGS_continuous, 'continuous_phenotype_csv'),
    ):
        arguments = dict(report[name])
        assert Path(arguments['phenotype_csv']) == Path(report[phenotype])
        assert Path(arguments['covariate_csv']) == Path(report['covariate_csv'])
        assert Path(arguments['prepared_directory']) == Path(report['prepared_directory'])
        assert arguments['grm'] == 'none'
        assert arguments.get('impute_missing', arguments.get('rm_nan')) is False
        arguments.setdefault('output_directory', tmp_path / name)
        inspect.signature(function).bind(**arguments)
    prepared, _, manifest, entries, pairs = _dataset(report['prepared_directory'])
    assert len(pairs) == 1000
    assert [entry['name'] for entry in entries] == ['21', '22']
    assert manifest['preparation']['source_format'] == 'PLINK-BED-BIM-FAM'
    assert manifest['preparation']['completed'] is True
    settings = _analysis_settings(manifest)
    for entry, expected_m in zip(entries, (256, 64)):
        coverage = _annotation_coverage(entry, gene_variant_type='SNV',
            analyses=('individual', 'coding', 'noncoding', 'ncrna'),
            expected_sample_keys=sample_keys(pairs), expected_analysis=settings)
        assert coverage['matched_variants'] == coverage['n_variants'] == expected_m
        assert coverage['missing_snv'] == coverage['missing_non_snv'] == 0
        jobs = _jobs(entry, ('individual', 'coding', 'noncoding', 'ncrna'), tmp_path,
            single_mac_cutoff=20, single_group_variants=5000, single_region_size=10_000_000)
        assert {job['kind'] for job in jobs} == {'individual', 'coding', 'noncoding', 'ncrna'}
        _, _, decoded = _plink(root, entry['name'], len(pairs))
        np.testing.assert_array_equal(_prepared_dosages(Path(entry['container_directory'])), decoded)
        with PortableMetadataReader(entry['metadata_directory'], entry['container_directory']) as reader:
            np.testing.assert_array_equal(reader.sample_ids(), sample_keys(pairs))
            np.testing.assert_array_equal(reader.read_field('annotation_available'), np.ones(expected_m, dtype=bool))


def test_all_fifteen_masks_retain_usable_rare_variants_and_single_common_sites(generated):
    _, report = generated
    _, _, manifest, entries, _ = _dataset(report['prepared_directory'])
    entry = entries[0]
    dosages = _prepared_dosages(entry['container_directory'])
    called = 2 * np.sum(dosages != 3, axis=1)
    alternate = np.where(dosages == 3, 0, dosages).sum(axis=1)
    mac = np.minimum(alternate, called - alternate)
    maf = mac / called
    rare = (mac >= 2) & (maf < 0.01)
    assert rare.any()
    assert np.count_nonzero((mac >= 20) & (maf > 0.01)) >= 2
    with PortableMetadataReader(entry['metadata_directory'], entry['container_directory']) as reader:
        ref, alt = reader.read_ref_alt(np.arange(reader.n_variants))
        variants = VariantAnnotations(reader.read_field('position'), reader.read_field(manifest['qc_path']),
            {name: reader.read_field(path) for name, path in manifest['annotation_catalog'].items()},
            ref=ref, alt=alt, chromosome=reader.read_field('chromosome'))
        for name in manifest['annotation_names']:
            assert np.isfinite(np.asarray(variants.annotations[name], dtype=float)).all()
    catalog = json.loads(Path(entry['gene_catalog']).read_text())
    genes = catalog.get('jobs', catalog) if isinstance(catalog, dict) else catalog
    intervals = json.loads(Path(entry['promoter_intervals']).read_text())
    overlap = promoter_overlaps(variants.position, variants.chromosome, intervals)
    collected = {name: set() for name in (*CODING_CATEGORIES_INCL_PTV, *NONCODING_CATEGORIES, 'ncRNA')}
    for gene in genes:
        arguments = dict(gene.get('arguments', {}))
        arguments.update({key: gene[key] for key in ('gene_name', 'start', 'end') if key in gene})
        if gene['kind'] == 'coding':
            selections = coding_masks(variants, arguments['gene_name'], arguments['start'], arguments['end'],
                                      include_ptv=True, chromosome=entry['name'])
        elif gene['kind'] == 'noncoding':
            selections = noncoding_masks(variants, arguments['gene_name'], promoter_overlap=overlap,
                                         chromosome=entry['name'])
        else:
            selections = {'ncRNA': ncRNA_mask(variants, arguments['gene_name'], chromosome=entry['name'])}
        for category, rows in selections.items():
            collected[category].update(int(row) for row in rows if rare[row])
    assert set(collected) == set(CODING_CATEGORIES_INCL_PTV) | set(NONCODING_CATEGORIES) | {'ncRNA'}
    assert all(len(rows) >= 2 for rows in collected.values()), {key: len(value) for key, value in collected.items()}


@pytest.mark.parametrize('phenotype_type,expected,absent', [
    ('binary', 'binary_phenotype_csv', 'continuous_phenotype_csv'),
    ('continuous', 'continuous_phenotype_csv', 'binary_phenotype_csv'),
])
def test_single_family_generation_has_only_selected_inputs(tmp_path, phenotype_type, expected, absent):
    report = generate_simulated_data(tmp_path / phenotype_type, n=131, variants_per_chromosome=32,
        phenotype_type=phenotype_type, cpu_threads=1, prepare=False)
    assert Path(report[expected]).is_file()
    assert report.get(absent) is None
    assert report['prepared_directory'] is None
    assert report['configuration']['rare_fraction'] == 0.999
    assert report['chromosome_reports'][0]['common_variants'] == 2
    assert report['chromosome_reports'][0]['rare_variants'] == 30


@pytest.mark.parametrize('invalid', [
    {'n': 127}, {'n': True}, {'n': 128.5}, {'variants_per_chromosome': 31},
    {'variants_per_chromosome': True}, {'chromosomes': [21, '21']}, {'chromosomes': [23]},
    {'chromosomes': []}, {'variants_per_chromosome': {'21': 64, '22': 64}, 'chromosomes': [21]},
    {'phenotype_type': 'unknown'}, {'n_binary_traits': 1}, {'n_continuous_traits': 1},
    {'genotype_missing_rate': -0.1}, {'phenotype_missing_rate': 1.0}, {'covariate_missing_rate': float('nan')},
    {'case_fraction': 0}, {'case_fraction': 1}, {'rare_fraction': 1.1}, {'ld_strength': -0.1},
    {'seed': -1}, {'seed': True}, {'cpu_threads': 0}, {'chunk_size': 0},
    {'memory_limit_gib': float('inf')}, {'prepare': 'yes'},
])
def test_invalid_parameters_are_rejected_before_output_creation(tmp_path, invalid):
    output = tmp_path / 'invalid'
    with pytest.raises((TypeError, ValueError, MemoryError)):
        generate_simulated_data(output, **{"n": 131, "variants_per_chromosome": 32,
            "cpu_threads": 1, "prepare": False, **invalid})
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    root = tmp_path / 'once'
    options = dict(n=131, variants_per_chromosome=32, prepare=False, cpu_threads=1)
    generate_simulated_data(root, **options)
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}
    with pytest.raises(FileExistsError):
        generate_simulated_data(root, **options)
    assert {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()} == before


def test_raw_only_generation_can_be_prepared_by_public_converter(tmp_path):
    root = tmp_path / 'raw_only'
    report = generate_simulated_data(root, n=131, variants_per_chromosome=35,
        prepare=False, cpu_threads=1)
    output = tmp_path / 'independent_preparation'
    result = prepare_WGS_data(root / 'raw', output, chromosomes=['21'],
        annotation_directory=root / 'annotations', cpu_threads=1, memory_limit_gib=1, chunk_size=13)
    assert result['completed'] is True
    _, _, _, entries, pairs = _dataset(output)
    assert len(pairs) == 131 and len(entries) == 1
    np.testing.assert_array_equal(_prepared_dosages(entries[0]['container_directory']), _plink(root, '21', 131)[2])


def test_annotation_counts_and_ld_assignments_match_actual_diploid_calls(generated):
    root, report = generated
    _, _, decoded = _plink(root, '21', 1000)
    _, rows = _csv(root / 'annotations/chr21.csv')
    assert len(rows) == len(decoded) == 256
    chromosome_report = next(item for item in report['chromosome_reports'] if item['chromosome'] == '21')
    common = sum(float(row['target_maf']) >= 0.05 for row in rows)
    rare = sum(float(row['target_maf']) < 0.01 for row in rows)
    assert common == chromosome_report['common_variants'] == 2
    assert rare == chromosome_report['rare_variants'] == 254
    called_alleles = 2 * np.sum(decoded != 3, axis=1)
    alternate_alleles = np.where(decoded == 3, 0, decoded).sum(axis=1)
    observed_maf = np.minimum(alternate_alleles, called_alleles - alternate_alleles) / called_alleles
    assert np.count_nonzero(observed_maf >= 0.01) == common
    assert np.count_nonzero((observed_maf > 0) & (observed_maf < 0.01)) == rare
    assert all(row['synthetic'] == 'true' for row in rows)
    assert len(set(np.diff([int(row['position']) for row in rows]))) > 1
    copied = 0
    for index, row in enumerate(rows):
        called = decoded[index] != 3
        assert int(row['called_samples']) == int(called.sum())
        assert int(row['observed_alt_count']) == int(decoded[index, called].sum())
        target = float(row['target_maf'])
        assert 0 < target < 0.5
        if row['ld_copied_from']:
            source = int(row['ld_copied_from'])
            assert 0 <= source < index
            np.testing.assert_array_equal(decoded[index], decoded[source])
            copied += 1
    assert copied > 0
    assert copied == report['chromosome_reports'][0]['copied_ld_variants']


def test_covariate_missingness_excludes_only_actual_unavailable_samples(tmp_path):
    report = generate_simulated_data(tmp_path / 'covariate_missing', n=131, variants_per_chromosome=32,
        genotype_missing_rate=0, covariate_missing_rate=0.015,
        phenotype_missing_rate=0.1, cpu_threads=1, chunk_size=16)
    pairs = np.load(Path(report['prepared_directory']) / 'sample_pairs.npy', allow_pickle=False)
    phenotypes, covariates = read_csv_inputs(report['binary_phenotype_csv'], report['covariate_csv'])
    covariate_valid = np.isfinite(covariates.values).all(axis=1)
    assert 26 < covariate_valid.sum() < len(pairs)
    assert int(covariate_valid.sum()) == report['complete_covariate_samples']
    for index, trait in enumerate(phenotypes.names):
        identifiers, values, design, _, counts = aligned_phenotype(
            phenotypes, covariates, pairs, trait, rm_nan=False,
            family='binomial', binary_only=True)
        eligible = covariate_valid & np.isfinite(phenotypes.values[:, index])
        np.testing.assert_array_equal(identifiers, sample_keys(pairs[eligible]))
        np.testing.assert_array_equal(values, phenotypes.values[eligible, index])
        assert np.isfinite(design).all()
        assert counts['analysis_samples'] == int(eligible.sum())
        assert counts['imputed_n'] == 0


def test_declared_memory_budget_rejects_impossible_generation_without_files(tmp_path):
    root = tmp_path / 'impossible_budget'
    with pytest.raises((ValueError, MemoryError), match='memory_limit_gib'):
        generate_simulated_data(root, n=131, variants_per_chromosome=32,
            cpu_threads=1, memory_limit_gib=0.00001, prepare=False)
    assert not root.exists()


def test_public_import_exposes_the_generator():
    from fudan_wgs_toolkit import generate_simulated_data as public_generate
    assert public_generate is generate_simulated_data
