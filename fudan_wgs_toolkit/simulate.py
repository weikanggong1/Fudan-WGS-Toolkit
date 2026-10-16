"""Generate wholly synthetic inputs for reproducible software benchmarks.

No source cohort, reference genome, downloaded annotation or network service is
read. Coordinates, genes, annotation values and effects are invented. These
fixtures exercise file and numerical contracts; they do not model a population
or establish scientific agreement with another implementation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np


_ANNOTATIONS = {
    "CADD": "annotation/cadd",
    "aPC.LocalDiversity": "annotation/local_diversity",
    "aPC.Epigenetics": "annotation/epigenetics",
    "GENCODE.Category": "annotation/category",
    "GENCODE.EXONIC.Category": "annotation/exonic_category",
    "GENCODE.Info": "annotation/gene_info",
    "MetaSVM": "annotation/meta_svm",
    "CAGE": "annotation/cage",
    "DHS": "annotation/dhs",
    "GeneHancer": "annotation/enhancer",
}
_WEIGHT_NAMES = ["CADD", "aPC.LocalDiversity", "aPC.Epigenetics"]
_COLUMNS = ["chromosome", "position", "reference", "alternate", "qc",
            "CADD", "aPC.LocalDiversity", "aPC.Epigenetics",
            "GENCODE.Category", "GENCODE.EXONIC.Category", "GENCODE.Info",
            "MetaSVM", "CAGE", "DHS", "GeneHancer", "synthetic", "target_maf",
            "observed_alt_count", "called_samples", "ld_copied_from"]
_CLASSES = (
    ("exonic", "stopgain", "", "", "", ""),
    ("exonic", "stoploss", "", "", "", ""),
    ("splicing", "", "", "", "", ""),
    ("exonic", "nonsynonymous SNV", "D", "", "", ""),
    ("exonic", "nonsynonymous SNV", "T", "", "", ""),
    ("exonic", "synonymous SNV", "", "", "", ""),
    ("upstream", "", "", "", "", ""),
    ("downstream", "", "", "", "", ""),
    ("UTR3", "", "", "", "", ""),
    ("UTR5", "", "", "", "", ""),
    ("intronic", "", "", "SIM_CAGE", "", ""),
    ("intronic", "", "", "", "SIM_DHS", ""),
    ("intergenic", "", "", "SIM_CAGE", "", "enhancer"),
    ("intergenic", "", "", "", "SIM_DHS", "enhancer"),
    ("ncRNA_exonic", "", "", "", "", ""),
    ("ncRNA_splicing", "", "", "", "", ""),
)


def _positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _rate(value, name, *, upper=1.0, closed=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (value > upper if closed else value >= upper):
        bracket = "]" if closed else ")"
        raise ValueError(f"{name} must be in [0, {upper}{bracket}")
    return float(value)


def _chromosomes(values):
    if isinstance(values, str) and values == "all":
        return [str(value) for value in range(1, 23)]
    if isinstance(values, (str, int)):
        values = [values]
    result = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("chromosomes must contain unique autosomes 1 through 22")
        text = str(value).removeprefix("chr")
        if text not in {str(number) for number in range(1, 23)} or text in result:
            raise ValueError("chromosomes must contain unique autosomes 1 through 22")
        result.append(text)
    if not result:
        raise ValueError("chromosomes must not be empty")
    return result


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _conditioned_count(rng, n, probability, rare):
    upper = max(2, int(math.floor(2 * n * 0.009))) if rare else 2 * n - 2
    for _ in range(10000):
        value = int(rng.binomial(2 * n, probability))
        if 2 <= value <= upper:
            return value
    raise RuntimeError("conditional allele-count sampling failed; choose a larger n")


def _generate_chromosome(options):
    chromosome, n, m = options["chromosome"], options["n"], options["m"]
    root = Path(options["root"])
    rng = np.random.default_rng(np.random.SeedSequence([options["seed"], int(chromosome), 100]))
    prefix, annotation_path = root / "raw" / ("chr" + chromosome), root / "annotations" / ("chr" + chromosome + ".csv")
    variants_per_gene = options["variants_per_gene"]
    genes = (m + variants_per_gene - 1) // variants_per_gene
    # Positive exponential gaps are sampled around wholly fictional loci.
    positions = 100000 + np.cumsum(1 + rng.exponential(80, size=m).astype(np.int64))
    references = rng.integers(0, 4, size=m)
    alternates = (references + rng.integers(1, 4, size=m)) % 4
    bases = np.asarray(list("ACGT"))
    rare = np.ones(m, dtype=bool)
    # The complete first mask cycle stays rare, except redundant stoploss
    # records reserved for at least two common Single calibration variants.
    common_count = min(m, max(2, int(round(m * (1 - options["rare_fraction"])))))
    common_order = np.concatenate((np.asarray([2, 3]), rng.permutation(np.arange(32, m))))
    rare[common_order[:common_count]] = False
    lower = min(0.002, 1 / n)
    maf = np.exp(rng.uniform(math.log(lower), math.log(0.008), size=m))
    maf[~rare] = rng.uniform(0.05, 0.45, size=int((~rare).sum()))
    bytes_per_variant = (n + 3) // 4
    bed_codes = np.asarray([3, 2, 0], dtype=np.uint8)
    previous = {True: None, False: None}
    previous_index = {True: None, False: None}
    causal_index = int(np.flatnonzero(~rare)[0])
    causal_dosage = None
    alt_total = missing_total = copied_total = 0
    with prefix.with_suffix(".bed").open("wb") as bed, prefix.with_suffix(".bim").open("w", encoding="utf-8") as bim, annotation_path.open("w", encoding="utf-8", newline="") as annotations:
        bed.write(b"\x6c\x1b\x01")
        writer = csv.writer(annotations)
        writer.writerow(_COLUMNS)
        for first in range(0, m, options["generation_chunk_size"]):
            count = min(options["generation_chunk_size"], m - first)
            calls = np.zeros((count, n), dtype=np.uint8)
            metadata = []
            for local in range(count):
                index = first + local
                row = calls[local]
                is_rare = bool(rare[index])
                copied_from = ""
                if previous[is_rare] is not None and rng.random() < options["ld_strength"]:
                    row[:] = previous[is_rare]
                    copied_from = str(previous_index[is_rare])
                    copied_total += 1
                else:
                    alt_count = _conditioned_count(rng, n, float(maf[index]), is_rare)
                    haplotypes = rng.choice(2 * n, size=alt_count, replace=False)
                    sample_rows, dosage = np.unique(haplotypes // 2, return_counts=True)
                    row[sample_rows] = dosage
                # Copy before adding locus-specific missing calls.
                previous[is_rare] = row.copy()
                previous_index[is_rare] = index
                missing_count = int(rng.binomial(n, options["genotype_missing_rate"]))
                if missing_count:
                    row[rng.choice(n, size=missing_count, replace=False)] = 255
                called = row != 255
                observed_alt = int(row[called].sum(dtype=np.int64))
                called_count = int(called.sum())
                alt_total += observed_alt
                missing_total += missing_count
                if index == causal_index:
                    causal_dosage = row.astype(np.float64)
                    causal_dosage[~called] = (observed_alt / called_count) if called_count else 0
                gene_index = index // variants_per_gene
                gene = f"SIMGENE_{chromosome}_{gene_index:06d}"
                rna = f"SIMRNA_{chromosome}_{gene_index:06d}"
                category, exonic, meta_svm, cage, dhs, enhancer = _CLASSES[((index % variants_per_gene) // 2) % len(_CLASSES)]
                info = rna + "(synthetic)" if category.startswith("ncRNA") else gene + "(synthetic)" if category.startswith("UTR") else gene
                genehancer = f"synthetic=region=gene={gene};invented" if enhancer else ""
                reference, alternate = bases[references[index]], bases[alternates[index]]
                position = int(positions[index])
                variant = f"{chromosome}:{position}:{reference}:{alternate}"
                bim.write(f"{chromosome}\t{variant}\t0\t{position}\t{alternate}\t{reference}\n")
                weights = rng.uniform(0.5, 35, size=3)
                metadata.append([chromosome, position, reference, alternate, "PASS", *weights,
                    category, exonic, info, meta_svm, cage, dhs, genehancer, "true", float(maf[index]),
                    observed_alt, called_count, copied_from])
            writer.writerows(metadata)
            # PLINK variant-major two-bit calls, with A1 explicitly ALT:
            # ALT dosage 0/1/2/missing -> 11/10/00/01; padding bits are zero.
            packed = np.zeros((count, bytes_per_variant), dtype=np.uint8)
            for offset in range(4):
                selected = calls[:, offset::4]
                code = np.ones(selected.shape, dtype=np.uint8)
                called = selected != 255
                code[called] = bed_codes[selected[called]]
                packed[:, :selected.shape[1]] |= code << (2 * offset)
            bed.write(packed.tobytes())
    np.save(root / "raw" / ("chr" + chromosome + ".causal.npy"), causal_dosage, allow_pickle=False)
    gene_path, rna_path = root / "annotations" / ("chr" + chromosome + ".genes.csv"), root / "annotations" / ("chr" + chromosome + ".ncrna.csv")
    promoter_path = root / "annotations" / ("chr" + chromosome + ".promoters.tsv")
    with gene_path.open("w", newline="", encoding="utf-8") as gf, rna_path.open("w", newline="", encoding="utf-8") as rf, promoter_path.open("w", newline="", encoding="utf-8") as pf:
        gw, rw, pw = csv.writer(gf), csv.writer(rf), csv.writer(pf, delimiter="\t")
        gw.writerow(["hgnc_symbol", "chromosome_name", "start_position", "end_position"])
        rw.writerow(["chr", "ncRNA"])
        pw.writerow(["chromosome", "start", "end"])
        for gene_index in range(genes):
            start = gene_index * variants_per_gene
            stop = min(m, start + variants_per_gene)
            gw.writerow([f"SIMGENE_{chromosome}_{gene_index:06d}", chromosome, int(positions[start]), int(positions[stop - 1])])
            rw.writerow([chromosome, f"SIMRNA_{chromosome}_{gene_index:06d}"])
            # Exact invented promoter intervals cover only promoter-class loci.
            for index in range(start, stop):
                if ((index - start) // 2) % 16 in (10, 11):
                    pw.writerow([chromosome, int(positions[index]), int(positions[index])])
    return dict(chromosome=chromosome, variants=m, genes=genes, rare_variants=int(rare.sum()),
                common_variants=int((~rare).sum()), copied_ld_variants=copied_total,
                missing_genotypes=missing_total, observed_alt_count=alt_total,
                causal_variant=dict(index=causal_index, position=int(positions[causal_index]),
                    reference=str(bases[references[causal_index]]), alternate=str(bases[alternates[causal_index]])))


def _numeric_csv(path, pairs, columns, values):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["FID", "IID", *columns])
        for pair, row in zip(pairs, values):
            writer.writerow([*pair, *("" if not np.isfinite(value) else format(float(value), ".17g") for value in row)])


def generate_simulated_data(output_directory, *, n=1000, variants_per_chromosome=1024,
                            chromosomes=("21",), phenotype_type="both",
                            n_binary_traits=2, n_continuous_traits=2, seed=1729,
                            phenotype_missing_rate=0.02, covariate_missing_rate=0.0,
                            genotype_missing_rate=1e-5, case_fraction=0.2,
                            causal_effect=0.35, rare_fraction=0.999, ld_strength=0.2,
                            variants_per_gene=256, chunk_size=256, cpu_threads=8,
                            memory_limit_gib=8, prepare=True):
    """Write invented PLINK, annotations and phenotype/covariate CSV inputs.

    ``n`` controls subjects (at least 128); ``variants_per_chromosome`` is a
    count or autosome-to-count dictionary (at least 32 each). ``chromosomes``
    is an autosome list or ``'all'``. ``phenotype_type`` is ``binary``,
    ``continuous`` or ``both``; each selected family contains at least two
    traits. CSVs use literal string FID/IID and numeric columns, with blank
    missing entries. Covariates are age, sex and PC1 through PC20. Each trait
    independently loses a seeded subset of subjects; covariates are never
    filled. Returned kwargs select the available-sample analysis policy and
    ordinary null models (``grm='none'``).

    Rare target frequencies are log-uniform between min(.002,1/n) and .008;
    common frequencies are uniform .05 through .45. Diploid alternate counts
    follow Binomial(2*n, frequency), conditioned on MAC >= 2 and, for rare
    sites, MAC/(2*n) <= .009. Alternate haplotypes are uniformly sampled.
    ``ld_strength`` is the probability of copying the preceding locus of the
    same rare/common class; it is not a target LD correlation. All loci are
    biallelic SNVs. Positive exponential position gaps and the synthetic mask
    cycle have no relationship to a reference genome. ``rare_fraction`` is a
    target count proportion, rounded to a whole number of loci and constrained
    by the first annotation cycle's rare representatives and at least two
    common calibration loci; report counts give the realized exact mixture.

    ``causal_effect`` applies a standardized dosage coefficient to independent
    normal residuals or logistic disease probabilities. Ground truth records
    generating coefficients, not expected estimated effects or P values.
    ``cpu_threads`` caps chromosome generation workers and is passed to the
    existing preparation function. ``memory_limit_gib`` is an estimated host
    scheduling budget, not an RSS quota; dense genotype blocks are bounded
    by that estimate. ``prepare=False`` retains raw inputs only. Existing
    output directories are never overwritten. No real dataset is accessed.
    """
    n = _positive_int(n, "n", 128)
    cpu_threads = _positive_int(cpu_threads, "cpu_threads")
    variants_per_gene = _positive_int(variants_per_gene, "variants_per_gene", 32)
    chunk_size = _positive_int(chunk_size, "chunk_size")
    if chunk_size > 1024:
        raise ValueError("chunk_size must not exceed 1024")
    _positive_int(seed, "seed", 0)
    chromosome_names = _chromosomes(chromosomes)
    if phenotype_type not in ("binary", "continuous", "both"):
        raise ValueError("phenotype_type must be binary, continuous or both")
    binary_count = _positive_int(n_binary_traits, "n_binary_traits", 2) if phenotype_type in ("binary", "both") else 0
    continuous_count = _positive_int(n_continuous_traits, "n_continuous_traits", 2) if phenotype_type in ("continuous", "both") else 0
    for name, value in (("phenotype_missing_rate", phenotype_missing_rate), ("covariate_missing_rate", covariate_missing_rate), ("genotype_missing_rate", genotype_missing_rate)):
        _rate(value, name)
    _rate(rare_fraction, "rare_fraction", closed=True)
    _rate(ld_strength, "ld_strength", closed=True)
    if not 0 < _rate(case_fraction, "case_fraction") < 1:
        raise ValueError("case_fraction must be strictly between 0 and 1")
    if isinstance(causal_effect, bool) or not isinstance(causal_effect, (int, float)) or not math.isfinite(causal_effect):
        raise ValueError("causal_effect must be finite")
    if isinstance(memory_limit_gib, bool) or not isinstance(memory_limit_gib, (int, float)) or not math.isfinite(memory_limit_gib) or memory_limit_gib <= 0:
        raise ValueError("memory_limit_gib must be positive and finite")
    if type(prepare) is not bool:
        raise ValueError("prepare must be a boolean")
    if isinstance(variants_per_chromosome, dict):
        counts = {str(key).removeprefix("chr"): _positive_int(value, "variants_per_chromosome", 32)
                  for key, value in variants_per_chromosome.items()}
        if set(counts) != set(chromosome_names):
            raise ValueError("variant-count dictionary must match the requested chromosomes exactly")
    else:
        count = _positive_int(variants_per_chromosome, "variants_per_chromosome", 32)
        counts = dict.fromkeys(chromosome_names, count)
    root = Path(output_directory).expanduser().resolve()
    if root.exists():
        raise FileExistsError("synthetic output directory already exists; choose a fresh directory")
    budget_bytes = int(memory_limit_gib * 2 ** 30)
    # Conservative estimates include Python sample strings, all sample vectors,
    # both covariate matrices, the largest phenotype-family matrix, rank-check
    # scratch, chromosome axes and concurrent genotype packing/choice scratch.
    # Preparation has its own existing scheduler and memory estimate.
    global_estimate = 128 * 2 ** 20 + n * (1536 + 8 * max(binary_count, continuous_count))
    chromosome_axis_estimate = max(counts.values()) * 128
    worker_fixed_estimate = 64 * 2 ** 20 + n * 80 + chromosome_axis_estimate
    per_variant_estimate = n * 6 + 8192
    remaining_bytes = budget_bytes - global_estimate
    minimum_worker_bytes = worker_fixed_estimate + per_variant_estimate
    if remaining_bytes < minimum_worker_bytes:
        raise ValueError("memory_limit_gib is too small for estimated sample/trait vectors and one bounded genotype block")
    workers = min(cpu_threads, len(chromosome_names), remaining_bytes // minimum_worker_bytes)
    worker_bytes = remaining_bytes // workers
    generation_chunk = min(chunk_size, (worker_bytes - worker_fixed_estimate) // per_variant_estimate)
    if generation_chunk < 1:
        raise ValueError("memory_limit_gib is too small for one generation block")
    root.mkdir(parents=True, mode=0o700)
    for name in ("raw", "annotations", "inputs"):
        (root / name).mkdir(mode=0o700)
    started = time.perf_counter()
    configuration = dict(n=n, chromosomes=chromosome_names, variants_per_chromosome=counts,
        phenotype_type=phenotype_type, n_binary_traits=binary_count, n_continuous_traits=continuous_count,
        seed=seed, phenotype_missing_rate=phenotype_missing_rate, covariate_missing_rate=covariate_missing_rate,
        genotype_missing_rate=genotype_missing_rate, case_fraction=case_fraction, causal_effect=causal_effect,
        rare_fraction=rare_fraction, ld_strength=ld_strength, variants_per_gene=variants_per_gene,
        chunk_size=chunk_size, cpu_threads=cpu_threads, memory_limit_gib=memory_limit_gib, prepare=prepare)
    _write_json(root / "simulation.pending.json", dict(synthetic=True, configuration=configuration))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0, 200]))
    pairs = [(f"SIMF{index:09d}", f"SIMI{index:09d}") for index in range(n)]
    age = rng.uniform(40, 80, n)
    sex = rng.integers(0, 2, n)
    pcs = rng.normal(size=(n, 20))
    covariates = np.column_stack((age, sex, pcs))
    covariate_observed = rng.random(covariates.shape) >= covariate_missing_rate
    eligible = covariate_observed.all(axis=1)
    if eligible.sum() < 26:
        raise ValueError("too few samples have complete covariates; reduce covariate_missing_rate")
    for chromosome in chromosome_names:
        with (root / "raw" / ("chr" + chromosome + ".fam")).open("w", encoding="utf-8") as stream:
            for pair, individual_sex in zip(pairs, sex):
                stream.write(" ".join([*pair, "0", "0", str(int(individual_sex) + 1), "-9"]) + "\n")
    options = [dict(root=str(root), chromosome=chromosome, n=n, m=counts[chromosome],
        seed=seed, rare_fraction=rare_fraction, ld_strength=ld_strength,
        genotype_missing_rate=genotype_missing_rate, variants_per_gene=variants_per_gene,
        generation_chunk_size=int(generation_chunk)) for chromosome in chromosome_names]
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            chromosome_reports = list(pool.map(_generate_chromosome, options))
    else:
        chromosome_reports = [_generate_chromosome(option) for option in options]
    mapping = dict(chromosome="chromosome", position="position", reference="reference", alternate="alternate")
    mapping.update({value: key for key, value in _ANNOTATIONS.items()})
    mapping["annotation/qc"] = "qc"
    raw_manifest = dict(format="raw-wgs-annotations-v1", schema_version=1,
        synthetic=True, annotation_catalog=_ANNOTATIONS, annotation_names=_WEIGHT_NAMES,
        qc_path="annotation/qc", column_mapping=mapping,
        source_provenance=dict(synthetic=True, generator="generate_simulated_data", seed=seed,
                               reference_genome="none; invented positive coordinates"),
        chromosomes=[dict(name=chromosome, variants="chr" + chromosome + ".csv",
            gene_reference="chr" + chromosome + ".genes.csv",
            ncrna_reference="chr" + chromosome + ".ncrna.csv",
            promoter_intervals="chr" + chromosome + ".promoters.tsv") for chromosome in chromosome_names])
    _write_json(root / "annotations" / "annotations.json", raw_manifest)
    dosage = np.load(root / "raw" / ("chr" + chromosome_names[0] + ".causal.npy"), allow_pickle=False)
    dosage_mean, dosage_sd = float(dosage.mean()), float(dosage.std())
    if dosage_sd == 0:
        raise RuntimeError("synthetic causal calibration locus is constant")
    standardized_dosage = (dosage - dosage_mean) / dosage_sd
    linear_covariates = 0.015 * (age - 60) + 0.15 * (sex - 0.5) + 0.08 * pcs[:, 0]
    # Every trait has its own seed, observation subset and residual draws.
    ground_truth, trait_reports = [], []
    phenotype_paths = {}
    for family, number in (("binary", binary_count), ("continuous", continuous_count)):
        if not number:
            phenotype_paths[family] = None
            continue
        values = np.empty((n, number), dtype=np.float64)
        names = [f"sim_{family}_{index + 1:03d}" for index in range(number)]
        for index, name in enumerate(names):
            trait_rng = np.random.default_rng(np.random.SeedSequence([seed, 300 if family == "binary" else 400, index]))
            effect = float(causal_effect) * (1 if index % 2 == 0 else -1)
            eta = linear_covariates + effect * standardized_dosage
            intercept = 0.0
            if family == "binary":
                # Calibrate mean probability, then sample Bernoulli outcomes.
                lower_intercept, upper_intercept = -50.0, 50.0
                for _ in range(70):
                    intercept = (lower_intercept + upper_intercept) / 2
                    probability = 1 / (1 + np.exp(-np.clip(eta + intercept, -700, 700)))
                    if float(probability.mean()) < case_fraction:
                        lower_intercept = intercept
                    else:
                        upper_intercept = intercept
                result = trait_rng.binomial(1, probability).astype(np.float64)
            else:
                result = eta + trait_rng.normal(size=n)
            result[trait_rng.random(n) < phenotype_missing_rate] = np.nan
            finite = np.isfinite(result)
            analysis_rows = finite & eligible
            if analysis_rows.sum() < 26 or (family == "binary" and len(np.unique(result[analysis_rows])) != 2):
                raise ValueError("sample size/missingness/case fraction produced an unusable trait; increase n or reduce missingness")
            design = np.column_stack((np.ones(int(analysis_rows.sum())), covariates[analysis_rows]))
            if np.linalg.matrix_rank(design) != design.shape[1]:
                raise ValueError("synthetic trait has a rank-deficient covariate design; increase n or reduce missingness")
            del design
            values[:, index] = result
            trait_reports.append(dict(name=name, family=family, observed=int(finite.sum()),
                missing=int((~finite).sum()), available_for_analysis=int(analysis_rows.sum()),
                cases=int(result[analysis_rows].sum()) if family == "binary" else None,
                controls=int(analysis_rows.sum() - result[analysis_rows].sum()) if family == "binary" else None))
            causal = chromosome_reports[0]["causal_variant"]
            ground_truth.append([name, family, chromosome_names[0], causal["position"], causal["reference"],
                causal["alternate"], effect, dosage_mean, dosage_sd, intercept - 0.975, 0.015, 0.15, 0.08])
        path = root / "inputs" / (family + "_phenotypes.csv")
        _numeric_csv(path, pairs, names, values)
        phenotype_paths[family] = str(path)
        del values
    covariates[~covariate_observed] = np.nan
    covariate_path = root / "inputs" / "covariates.csv"
    _numeric_csv(covariate_path, pairs, ["age", "sex", *[f"PC{index}" for index in range(1, 21)]], covariates)
    truth_path = root / "inputs" / "ground_truth.csv"
    with truth_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trait", "family", "chromosome", "position", "reference", "alternate",
            "effect_per_dosage_sd", "dosage_mean", "dosage_sd", "intercept", "age_effect", "sex_effect", "PC1_effect"])
        writer.writerows(ground_truth)
    raw_seconds = time.perf_counter() - started
    prepared_path = root / "prepared"
    preparation_report = None
    if prepare:
        from .prepare import prepare_WGS_data
        preparation_report = prepare_WGS_data(root / "raw", prepared_path,
            annotation_directory=root / "annotations", chromosomes=chromosome_names,
            cpu_threads=cpu_threads, memory_limit_gib=memory_limit_gib, chunk_size=chunk_size)
    common_kwargs = dict(covariate_csv=str(covariate_path), prepared_directory=str(prepared_path),
        chromosomes=chromosome_names, analyses=("individual", "coding", "noncoding", "ncrna"),
        cpu_threads=cpu_threads, grm="none")
    binary_kwargs = dict(common_kwargs, phenotype_csv=phenotype_paths["binary"],
        output_directory=str(root / "results_binary"), impute_missing=False) if binary_count else None
    continuous_kwargs = dict(common_kwargs, phenotype_csv=phenotype_paths["continuous"],
        output_directory=str(root / "results_continuous"), rm_nan=False) if continuous_count else None
    files = {str(path.relative_to(root)): dict(size_bytes=path.stat().st_size, sha256=_file_sha(path))
             for directory in (root / "raw", root / "annotations", root / "inputs")
             for path in sorted(directory.rglob("*")) if path.is_file()}
    report = dict(completed=True, synthetic=True, schema_version=1, seed=seed, n_samples=n,
        source_directory=str(root / "raw"), annotation_directory=str(root / "annotations"),
        binary_phenotype_csv=phenotype_paths["binary"], continuous_phenotype_csv=phenotype_paths["continuous"],
        covariate_csv=str(covariate_path), ground_truth_csv=str(truth_path),
        prepared_directory=str(prepared_path) if prepare else None,
        binary_kwargs=binary_kwargs if prepare else None, continuous_kwargs=continuous_kwargs if prepare else None,
        configuration=configuration, chromosome_reports=chromosome_reports, trait_reports=trait_reports,
        complete_covariate_samples=int(eligible.sum()), preparation=preparation_report,
        ground_truth_definition="eta = intercept + effect_per_dosage_sd * (ALT_dosage - dosage_mean) / dosage_sd + age_effect * age + sex_effect * sex + PC1_effect * PC1; binary: Bernoulli(logistic(eta)); continuous: eta + N(0,1)",
        generation_seconds=raw_seconds, total_seconds=time.perf_counter() - started,
        resource_plan=dict(cpu_threads=cpu_threads, chromosome_generation_workers=workers,
            generation_chunk_size=int(generation_chunk), memory_limit_gib=memory_limit_gib,
            memory_budget_kind="estimated scheduling budget, not a hard RSS quota",
            estimated_global_vector_bytes=int(global_estimate),
            estimated_worker_fixed_bytes=int(worker_fixed_estimate),
            estimated_generation_peak_bytes=int(global_estimate + workers * (worker_fixed_estimate + generation_chunk * per_variant_estimate)),
            expected_plink_bed_bytes=sum(3 + ((n + 3) // 4) * count for count in counts.values())),
        limitations=["Entirely invented coordinates, genes, annotations and effects; no real cohort data.",
            "Conditional allele counts and copy-based LD exercise software; this is not a population simulator.",
            "Synthetic checks cannot replace real-data agreement with an official implementation.",
            "Preparation retains its normal compression storage cap; dense user settings can exceed it."],
        files=files, manifest=str(root / "simulation.json"))
    _write_json(root / "simulation.json", report)
    (root / "simulation.pending.json").unlink()
    return report


def main():
    """Command-line entry point for :func:`generate_simulated_data`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--variants-per-chromosome", type=int, default=1024)
    parser.add_argument("--chromosomes", nargs="+", default=["21"])
    parser.add_argument("--phenotype-type", choices=("binary", "continuous", "both"), default="both")
    parser.add_argument("--n-binary-traits", type=int, default=2)
    parser.add_argument("--n-continuous-traits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--phenotype-missing-rate", type=float, default=0.02)
    parser.add_argument("--covariate-missing-rate", type=float, default=0.0)
    parser.add_argument("--genotype-missing-rate", type=float, default=1e-5)
    parser.add_argument("--case-fraction", type=float, default=0.2)
    parser.add_argument("--causal-effect", type=float, default=0.35)
    parser.add_argument("--rare-fraction", type=float, default=0.999)
    parser.add_argument("--ld-strength", type=float, default=0.2)
    parser.add_argument("--variants-per-gene", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--memory-limit-gib", type=float, default=8)
    parser.add_argument("--raw-only", action="store_true", help="keep PLINK/CSV inputs without preparing genetic data")
    arguments = vars(parser.parse_args())
    arguments["prepare"] = not arguments.pop("raw_only")
    if arguments["chromosomes"] == ["all"]:
        arguments["chromosomes"] = "all"
    print(json.dumps(generate_simulated_data(**arguments), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
