"""Generate synthetic WGS inputs directly in the validated prepared format.

Only aggregate calibration statistics are consumed. No participant identity,
original coordinate, allele call, gene name or association result is copied.
The small preset is entirely invented; reference runs require an explicit or
privately configured aggregate profile. No raw genotype interchange is written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .storage_plan import directory_storage


def _positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _rate(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError(f"{name} must be a finite number in [0, 1)")
    return float(value)


def _finite_positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _chromosomes(values, available):
    if isinstance(values, str) and values == "all":
        return list(available)
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
    if not result or set(result) - set(available):
        raise ValueError("requested chromosomes must be present in the calibration profile")
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


def _numeric_csv(path, pairs, columns, values):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["FID", "IID", *columns])
        for pair, row in zip(pairs, values):
            writer.writerow([*pair, *("" if not np.isfinite(value) else format(float(value), ".17g") for value in row)])


def _nearest_existing_parent(path):
    current = Path(path).parent
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise ValueError("no accessible filesystem parent for output_directory")
        current = parent
    return current


def _resource_preflight(root, profile, chromosomes, n, variant_scale, binary_count,
                        continuous_count, cpu_threads, chunk_size, memory_limit_gib,
                        storage_limit_gib, prepared_plan, *, dry_run=False):
    """Bound active vectors/frames; estimate full metadata and payload on disk.

    No n-by-whole-chromosome allocation is admitted. The stored metadata size
    grows with m; its host estimate grows only with a bounded metadata block.
    These are explicit scheduling estimates rather than operating-system caps.
    """
    selected = [entry for entry in profile["chromosomes"] if str(entry["name"]) in chromosomes]
    counts = {str(entry["name"]): max(1, int(round(entry["variants"] * variant_scale))) for entry in selected}
    maximum_m = max(counts.values())
    total_masks = sum(len(entry.get("masks", [])) for entry in selected)
    global_bytes = 128 * 2**20 + n * (1536 + 8 * max(binary_count, continuous_count))
    metadata_block_bytes = min(maximum_m, 65536) * 128
    catalog_bytes = total_masks * 1024
    # One bounded compact frame may require original arrays, compressed and
    # decompressed bytes, validation scratch and integer prefix counts.
    frame_bytes = min(6 * 256 * 2**20, 64 * 2**20 + n * min(chunk_size, maximum_m) * 24)
    minimum_peak = global_bytes + metadata_block_bytes + catalog_bytes + frame_bytes
    memory_budget = int(memory_limit_gib * 2**30)
    if minimum_peak > memory_budget:
        raise ValueError("memory_limit_gib is too small for estimated sample vectors, metadata block and one sparse frame")
    total_m = sum(counts.values())
    # The builder owns the on-disk format and compression estimate. A second
    # raw-CSR formula here previously replaced its estimate with a multi-TiB
    # worst-case bound; all callers now consume this single component plan.
    prepared_disk_bytes = int(prepared_plan["estimated_prepared_bytes"])
    prepared_peak_bytes = int(prepared_plan["estimated_prepared_peak_bytes"])
    csv_disk_bytes = n * (binary_count + continuous_count + 22) * 32 + n * 128
    estimated_disk_bytes = prepared_disk_bytes + csv_disk_bytes
    estimated_peak_bytes = prepared_peak_bytes+csv_disk_bytes
    parent = _nearest_existing_parent(root)
    free_bytes = shutil.disk_usage(parent).free
    reserve_bytes = min(2**30, max(64 * 2**20, free_bytes // 20))
    usable_bytes = max(0, free_bytes - reserve_bytes)
    if not dry_run and estimated_peak_bytes > usable_bytes:
        raise ValueError("estimated synthetic data storage exceeds available disk space with reserve")
    if storage_limit_gib is not None:
        requested = int(storage_limit_gib * 2**30)
        if requested < prepared_peak_bytes:
            raise ValueError("storage_limit_gib is smaller than the prepared peak planning upper estimate")
        if not dry_run and requested + csv_disk_bytes > usable_bytes:
            raise ValueError("storage_limit_gib exceeds available disk space with reserve")
        storage_limit_bytes = requested
    else:
        storage_limit_bytes = prepared_peak_bytes
    return dict(memory_limit_gib=memory_limit_gib, memory_budget_kind="estimated scheduling budget, not a hard RSS quota",
        estimated_frontend_vector_bytes=int(global_bytes), estimated_metadata_block_bytes=int(metadata_block_bytes),
        estimated_catalog_bytes=int(catalog_bytes), estimated_sparse_frame_bytes=int(frame_bytes),
        estimated_minimum_host_bytes=int(minimum_peak), total_variants=int(total_m),
        variants_per_chromosome=counts, estimated_prepared_disk_bytes=int(prepared_disk_bytes),
        estimated_prepared_peak_bytes=prepared_peak_bytes,
        estimated_total_disk_bytes=int(estimated_disk_bytes), estimated_total_peak_bytes=int(estimated_peak_bytes),
        generation_fits_available_disk=estimated_peak_bytes <= usable_bytes,
        storage_estimate_kind="compressed final estimate plus explicit temporary peak and planning margins; not a measured full dataset",
        available_disk_bytes=int(free_bytes),
        disk_reserve_bytes=int(reserve_bytes), storage_limit_bytes=int(storage_limit_bytes),
        cpu_threads=cpu_threads)


def _phenotype_inputs(root, pairs, dosage, causal, *, n, seed, binary_count,
                      continuous_count, phenotype_missing_rate, covariate_missing_rate,
                      case_fraction, causal_effect):
    """Retain independent family seeds and the validated covariate/effect model."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0, 200]))
    age = rng.uniform(40, 80, n)
    sex = rng.integers(0, 2, n)
    pcs = rng.normal(size=(n, 20))
    covariates = np.column_stack((age, sex, pcs))
    covariate_observed = rng.random(covariates.shape) >= covariate_missing_rate
    eligible = covariate_observed.all(axis=1)
    if eligible.sum() < 26:
        raise ValueError("too few samples have complete covariates; reduce covariate_missing_rate")
    dosage_mean, dosage_sd = float(dosage.mean()), float(dosage.std())
    if dosage.shape != (n,) or not np.isfinite(dosage).all() or dosage_sd == 0:
        raise RuntimeError("synthetic causal calibration locus must be finite and nonconstant")
    standardized_dosage = (dosage - dosage_mean) / dosage_sd
    linear_covariates = 0.015 * (age - 60) + 0.15 * (sex - 0.5) + 0.08 * pcs[:, 0]
    ground_truth, trait_reports, phenotype_paths = [], [], {}
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
            ground_truth.append([name, family, causal["chromosome"], causal["position"], causal["reference"],
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
    return dict(phenotype_paths=phenotype_paths, covariate_csv=str(covariate_path),
                ground_truth_csv=str(truth_path), trait_reports=trait_reports,
                complete_covariate_samples=int(eligible.sum()))


def generate_simulated_data(output_directory, *, n=None, calibration_profile=None,
                            preset="reference", chromosomes="all", variant_scale=1.0,
                            phenotype_type="both", n_binary_traits=2, n_continuous_traits=2,
                            seed=1729, phenotype_missing_rate=0.02,
                            covariate_missing_rate=0.0, case_fraction=0.2,
                            causal_effect=0.35, chunk_size=256, cpu_threads=8,
                            memory_limit_gib=8, storage_limit_gib=None, dry_run=False):
    """Generate direct prepared inputs and independent numeric phenotype CSVs.

    Reference calibration consumes aggregate statistics only and is required
    by default; absent private configuration fails before output creation.
    ``preset='small'`` selects an invented, reproducible software-test profile.
    ``n=None`` uses its reference sample size. Changing n preserves target MAF
    while integer MAC changes with the simulated denominator. ``variant_scale``
    scales chromosome counts and mask sizes; 1 retains the calibrated counts.

    CSVs contain literal FID/IID and at least two selected binary or continuous
    columns. Missingness is independently sampled per trait; covariates are
    never imputed. Returned kwargs use ordinary nulls and available samples,
    with ``gene_variant_type='variant'`` to include synthetic non-SNV masks.
    Prepared gene catalogs request all seven coding masks including PTV, all
    seven noncoding masks and ncRNA wherever the profile declares those jobs.
    The frontend and direct builder estimate memory/disk before publication.
    ``cpu_threads`` controls generation workers, not already initialized BLAS
    threads. No analysis or background run is started by this function.
    ``dry_run=True`` returns the bounded compression/storage plan without
    creating output files and may inspect a scale larger than free disk.
    """
    _positive_int(cpu_threads, "cpu_threads")
    _positive_int(chunk_size, "chunk_size")
    if chunk_size > 1024:
        raise ValueError("chunk_size must not exceed 1024")
    _positive_int(seed, "seed", 0)
    _finite_positive(variant_scale, "variant_scale")
    _finite_positive(memory_limit_gib, "memory_limit_gib")
    if storage_limit_gib is not None:
        _finite_positive(storage_limit_gib, "storage_limit_gib")
    if n is not None:
        _positive_int(n, "n", 128)
    if type(dry_run) is not bool:
        raise ValueError("dry_run must be a boolean")
    if preset not in ("reference", "small"):
        raise ValueError("preset must be reference or small")
    if phenotype_type not in ("binary", "continuous", "both"):
        raise ValueError("phenotype_type must be binary, continuous or both")
    binary_count = _positive_int(n_binary_traits, "n_binary_traits", 2) if phenotype_type in ("binary", "both") else 0
    continuous_count = _positive_int(n_continuous_traits, "n_continuous_traits", 2) if phenotype_type in ("continuous", "both") else 0
    _rate(phenotype_missing_rate, "phenotype_missing_rate")
    _rate(covariate_missing_rate, "covariate_missing_rate")
    if not 0 < _rate(case_fraction, "case_fraction") < 1:
        raise ValueError("case_fraction must be strictly between 0 and 1")
    if isinstance(causal_effect, bool) or not isinstance(causal_effect, (int, float)) or not math.isfinite(causal_effect):
        raise ValueError("causal_effect must be finite")
    root = Path(output_directory).expanduser().resolve()
    if root.exists() and not dry_run:
        raise FileExistsError("synthetic output directory already exists; choose a fresh directory")
    from .simulation_profile import resolve_profile
    from .simulate_prepared import build_simulated_prepared, plan_simulated_prepared
    profile, provenance = resolve_profile(calibration_profile, preset)
    selected = _chromosomes(chromosomes, [str(entry["name"]) for entry in profile["chromosomes"]])
    if n is None:
        n = profile["reference_n"]
    _positive_int(n, "n", 128)
    prepared_plan = plan_simulated_prepared(root, profile, n, selected, seed,
        cpu_threads, memory_limit_gib, chunk_size,
        None if storage_limit_gib is None else int(storage_limit_gib * 2**30),
        variant_scale=variant_scale)
    resources = _resource_preflight(root, profile, selected, n, variant_scale, binary_count,
        continuous_count, cpu_threads, chunk_size, memory_limit_gib, storage_limit_gib,
        prepared_plan, dry_run=dry_run)
    resources["prepared_preflight"] = {key: value for key, value in prepared_plan.items() if key != "entries"}
    del prepared_plan
    configuration = dict(n=n, preset=preset, chromosomes=selected, variant_scale=variant_scale,
        phenotype_type=phenotype_type, n_binary_traits=binary_count, n_continuous_traits=continuous_count,
        seed=seed, phenotype_missing_rate=phenotype_missing_rate,
        covariate_missing_rate=covariate_missing_rate, case_fraction=case_fraction,
        causal_effect=causal_effect, chunk_size=chunk_size, cpu_threads=cpu_threads,
        memory_limit_gib=memory_limit_gib, storage_limit_gib=storage_limit_gib)
    if dry_run:
        return dict(completed=False, dry_run=True, synthetic=True, n_samples=n,
            configuration=configuration, calibration=provenance, resource_plan=resources,
            output_created=False,
            scope="bounded invented compression probes only; no genotype dataset, phenotype or analysis was written")
    root.mkdir(parents=True, mode=0o700)
    (root / "inputs").mkdir(mode=0o700)
    started = time.perf_counter()
    _write_json(root / "simulation.pending.json", dict(synthetic=True, configuration=configuration,
        calibration=provenance, resource_plan=resources))
    prepared = build_simulated_prepared(root, profile, n, selected, seed, cpu_threads,
        memory_limit_gib, chunk_size, resources["storage_limit_bytes"], variant_scale=variant_scale)
    from .identity import validate_sample_pairs
    pairs = validate_sample_pairs(np.load(prepared["sample_pairs_path"], allow_pickle=False),
                                  where="simulated prepared population")
    if pairs.shape != (n, 2):
        raise RuntimeError("direct prepared sample axis differs from requested n")
    dosage = np.load(prepared["causal_dosage_path"], allow_pickle=False)
    inputs = _phenotype_inputs(root, pairs, dosage, prepared["causal_variant"], n=n, seed=seed,
        binary_count=binary_count, continuous_count=continuous_count,
        phenotype_missing_rate=phenotype_missing_rate, covariate_missing_rate=covariate_missing_rate,
        case_fraction=case_fraction, causal_effect=causal_effect)
    common = dict(covariate_csv=inputs["covariate_csv"], prepared_directory=prepared["prepared_directory"],
        chromosomes=selected, analyses=("individual", "coding", "noncoding", "ncrna"),
        cpu_threads=cpu_threads, seed=seed, grm="none", gene_variant_type="variant")
    binary_kwargs = dict(common, phenotype_csv=inputs["phenotype_paths"]["binary"],
        output_directory=str(root / "results_binary"), impute_missing=False) if binary_count else None
    continuous_kwargs = dict(common, phenotype_csv=inputs["phenotype_paths"]["continuous"],
        output_directory=str(root / "results_continuous"), rm_nan=False) if continuous_count else None
    files = {str(path.relative_to(root)): dict(size_bytes=path.stat().st_size, sha256=_file_sha(path))
             for path in sorted((root / "inputs").rglob("*")) if path.is_file()}
    report = dict(completed=True, synthetic=True, schema_version=2, seed=seed, n_samples=n,
        prepared_directory=prepared["prepared_directory"],
        binary_phenotype_csv=inputs["phenotype_paths"]["binary"],
        continuous_phenotype_csv=inputs["phenotype_paths"]["continuous"],
        covariate_csv=inputs["covariate_csv"], ground_truth_csv=inputs["ground_truth_csv"],
        binary_kwargs=binary_kwargs, continuous_kwargs=continuous_kwargs,
        configuration=configuration, calibration=provenance,
        chromosome_reports=prepared.get("chromosome_reports", []),
        mask_audit={str(entry["chromosome"]): entry.get("mask_audit", [])
                    for entry in prepared.get("chromosome_reports", [])},
        mask_counts_exact=prepared.get("mask_counts_exact"),
        frequency_counts_exact=prepared.get("frequency_counts_exact"),
        frequency_policy=prepared.get("frequency_policy"), mask_policy=prepared.get("mask_policy"),
        trait_reports=inputs["trait_reports"], complete_covariate_samples=inputs["complete_covariate_samples"],
        preparation=prepared, resource_plan=resources,
        ground_truth_definition="eta = intercept + effect_per_dosage_sd * (ALT_dosage - dosage_mean) / dosage_sd + age_effect * age + sex_effect * sex + PC1_effect * PC1; binary: Bernoulli(logistic(eta)); continuous: eta + N(0,1)",
        generation_seconds=time.perf_counter() - started, total_seconds=time.perf_counter() - started,
        measured_storage=directory_storage(root, exclude=("simulation.pending.json", "simulation.json")),
        limitations=["Coordinates, samples, calls, gene labels and weights are synthetic; calibration uses aggregate statistics only.",
            "Target MAF is retained when n changes; integer MAC and observed mask membership can change.",
            "Synthetic comparison cannot replace same-input scientific validation against an official implementation.",
            "Memory and disk estimates are admission estimates; the prepared writer also enforces explicit byte budgets."],
        files=files, manifest=str(root / "simulation.json"))
    _write_json(root / "simulation.json", report)
    (root / "simulation.pending.json").unlink()
    return report


def main():
    """Command-line entry point for direct synthetic prepared-data generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True,
                        help="fresh output directory for prepared/, inputs/ and simulation.json")
    parser.add_argument("--n", type=int,
                        help="synthetic sample count >=128; default: calibration reference_n; MAF is retained as n changes")
    parser.add_argument("--calibration-profile", type=Path,
                        help="private aggregate calibration JSON; otherwise use environment or private user configuration")
    parser.add_argument("--preset", choices=("reference", "small"), default="reference",
                        help="reference requires calibrated aggregates; small is an explicitly artificial software-test preset")
    parser.add_argument("--chromosomes", nargs="+", default=["all"],
                        help="all calibrated autosomes (default), or selected chromosome numbers")
    parser.add_argument("--variant-scale", type=float, default=1.0,
                        help="physical variant and marginal mask workload scale; 1 retains the calibrated counts (default)")
    parser.add_argument("--phenotype-type", choices=("binary", "continuous", "both"), default="both",
                        help="phenotype family to write; default: both")
    parser.add_argument("--n-binary-traits", type=int, default=2,
                        help="number of binary 0/1 columns, at least 2 when requested (default: 2)")
    parser.add_argument("--n-continuous-traits", type=int, default=2,
                        help="number of continuous columns, at least 2 when requested (default: 2)")
    parser.add_argument("--seed", type=int, default=1729,
                        help="nonnegative reproducibility seed (default: 1729)")
    parser.add_argument("--phenotype-missing-rate", type=float, default=0.02,
                        help="independent per-trait missing fraction in [0,1) (default: 0.02)")
    parser.add_argument("--covariate-missing-rate", type=float, default=0.0,
                        help="independent per-covariate missing fraction in [0,1) (default: 0)")
    parser.add_argument("--case-fraction", type=float, default=0.2,
                        help="expected binary case fraction strictly between 0 and 1 (default: 0.2)")
    parser.add_argument("--causal-effect", type=float, default=0.35,
                        help="effect per causal ALT-dosage SD; binary uses log odds (default: 0.35)")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="requested maximum variants per bounded sparse frame, 1..1024 (default: 256)")
    parser.add_argument("--cpu-threads", type=int, default=8,
                        help="maximum chromosome generation workers (default: 8); existing BLAS pools are unchanged")
    parser.add_argument("--memory-limit-gib", type=float, default=8,
                        help="positive host-memory admission estimate in GiB, not a kernel quota (default: 8)")
    parser.add_argument("--storage-limit-gib", type=float,
                        help="optional explicit prepared-data byte budget in GiB; full metadata and free disk are checked")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report bounded compression/final/peak storage estimates; create no output")
    arguments = vars(parser.parse_args())
    if arguments["chromosomes"] == ["all"]:
        arguments["chromosomes"] = "all"
    print(json.dumps(generate_simulated_data(**arguments), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
