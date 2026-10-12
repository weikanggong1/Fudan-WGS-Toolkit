"""Single and multiple phenotype association on prepared genetic data."""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid
import numpy as np
import torch
from .identity import sample_keys, validate_sample_pairs, pairs_from_keys

_MISSING = {"", "na", "nan", "null", "none", "n/a"}
_KINDS = ("individual", "coding", "noncoding", "ncrna")
_DEFAULT_MEMORY_LIMIT = None
_DEFAULT_GENE_VARIANT = "SNV"


def _normalise_chromosomes(value):
    """All supplied autosomes, or a distinct explicitly requested subset."""
    if value is None or value == "all" or isinstance(value, (list, tuple)) and list(value) == ["all"]:
        return None
    if isinstance(value, (str, int)):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("chromosomes must be 'all' or a nonempty chromosome list")
    selected = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise ValueError("chromosomes must contain integers from 1 to 22")
        text = str(item).removeprefix("chr")
        if not text.isdecimal() or not 1 <= int(text) <= 22:
            raise ValueError("chromosomes must contain integers from 1 to 22")
        name = str(int(text))
        if name in selected:
            raise ValueError("chromosomes must not contain duplicates")
        selected.append(name)
    return selected


def _normalise_devices(device, gpu_ids=None):
    """A CUDA name list selects devices; integer n selects cuda:0..n-1."""
    if gpu_ids is not None:
        if device not in (("cuda:0",), "cuda:0", ["cuda:0"]):
            raise ValueError("use device or gpu_ids, rather than both")
        if not isinstance(gpu_ids, (list, tuple)) or any(type(i) is not int or i < 0 for i in gpu_ids):
            raise ValueError("gpu_ids must contain nonnegative integers")
        device = [f"cuda:{i}" for i in gpu_ids]
    if type(device) is int:
        if device < 1:
            raise ValueError("device GPU count must be positive")
        device = [f"cuda:{i}" for i in range(device)]
    elif isinstance(device, (str, torch.device)):
        device = [str(device)]
    if not isinstance(device, (list, tuple)) or not device:
        raise ValueError("device must be a CUDA device list or positive GPU count")
    indices = []
    for value in device:
        if not isinstance(value, str):
            raise ValueError("device lists must contain CUDA names, such as cuda:0")
        try:
            selected = torch.device(value)
        except (ValueError, RuntimeError) as error:
            raise ValueError("invalid CUDA device name") from error
        if selected.type != "cuda":
            raise ValueError("run_WGS_all requires CUDA devices")
        indices.append(0 if selected.index is None else selected.index)
    if len(set(indices)) != len(indices):
        raise ValueError("device must select distinct GPUs")
    if not torch.cuda.is_available() or max(indices) >= torch.cuda.device_count():
        raise ValueError("requested CUDA devices are unavailable")
    return tuple(indices)


def _family_selection(value, names, values, *, rm_nan=False):
    from collections.abc import Mapping
    if value is None or value == "auto":
        overrides = {}
    elif isinstance(value, Mapping):
        overrides = dict(value)
    else:
        raise ValueError("phenotype_families must be 'auto' or a phenotype mapping")
    if set(overrides) - set(names) or any(v not in ("gaussian", "binomial", "binary", "auto") for v in overrides.values()):
        raise ValueError("phenotype_families must map existing traits to gaussian or binomial")
    result = {}
    for index, name in enumerate(names):
        family = overrides.get(name, "auto")
        finite = values[:, index][np.isfinite(values[:, index])]
        if family == "auto":
            family = "binomial" if finite.size and np.all((finite == 0) | (finite == 1)) else "gaussian"
        if family == "binary":
            family = "binomial"
        if family == "binomial" and not rm_nan and np.any((finite != 0) & (finite != 1)):
            raise ValueError("binary phenotype values must be zero, one or missing")
        result[name] = family
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_sha256():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted((*root.rglob("*.py"), *root.rglob("*.cpp"))):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _json_write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class NumericCSV:
    sample_pairs: np.ndarray
    names: tuple[str, ...]
    values: np.ndarray

    @property
    def sample_ids(self):
        return sample_keys(self.sample_pairs)


def _read_csv(path, *, require_columns):
    path = Path(path)
    pairs, rows = [], []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            raise ValueError(f"{path.name}: missing CSV header")
        header = [name.strip() for name in header]
        if any(not name for name in header) or len(set(header)) != len(header):
            raise ValueError(f"{path.name}: column names must be nonempty and unique")
        if not {"FID", "IID"} <= set(header):
            raise ValueError(f"{path.name}: FID and IID columns are required")
        fid, iid = header.index("FID"), header.index("IID")
        columns = [i for i, name in enumerate(header) if name not in ("FID", "IID")]
        if require_columns and not columns:
            raise ValueError(f"{path.name}: provide at least one named phenotype column")
        for line, row in enumerate(reader, 2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                raise ValueError(f"{path.name}:{line}: number of columns differs from the header")
            pairs.append([row[fid], row[iid]])
            values = []
            for column in columns:
                token = row[column].strip()
                if token.lower() in _MISSING:
                    value = float("nan")
                else:
                    try:
                        value = float(token)
                    except ValueError as error:
                        raise ValueError(f"{path.name}:{line}: {header[column]} must contain numeric or missing values") from error
                    if not np.isfinite(value):
                        raise ValueError(f"{path.name}:{line}: infinity is not a valid numeric value")
                values.append(value)
            rows.append(values)
    if not rows:
        raise ValueError(f"{path.name}: no participant rows")
    pairs = validate_sample_pairs(np.asarray(pairs, dtype=str), where=path.name)
    names = tuple(header[column] for column in columns)
    return NumericCSV(pairs, names, np.asarray(rows, dtype=np.float64).reshape(len(rows), len(names)))


def read_csv_inputs(phenotype_csv, covariate_csv):
    """Read numeric values using exact FID/IID strings without imputing values."""
    return _read_csv(phenotype_csv, require_columns=True), _read_csv(covariate_csv, require_columns=False)


def aligned_phenotype(phenotypes, covariates, cache_pairs, trait, *, covariate_names=None,
                      reduce_redundant=False, rm_nan=False, family="gaussian"):
    """Align this trait and apply its explicit missing policy in cache order.

    This lower-level alignment helper keeps the historical available-case
    default. Public association entry points pass their rm_nan=True default.
    """
    if trait not in phenotypes.names:
        raise ValueError("unknown phenotype column")
    if covariate_names is not None:
        if (not isinstance(covariate_names, (list, tuple)) or
                any(not isinstance(name, str) for name in covariate_names) or
                len(set(covariate_names)) != len(covariate_names)):
            raise ValueError("selected covariate names must be a sequence of distinct column names")
        if set(covariate_names) - set(covariates.names):
            raise ValueError("unknown selected covariate column")
        columns = [covariates.names.index(name) for name in covariate_names]
        covariates = NumericCSV(covariates.sample_pairs, tuple(covariate_names), covariates.values[:, columns])
    cached = validate_sample_pairs(cache_pairs, where="prepared sample index")
    keys = sample_keys(cached)
    pheno_lookup = {key: i for i, key in enumerate(phenotypes.sample_ids)}
    cov_lookup = {key: i for i, key in enumerate(covariates.sample_ids)}
    uses_covariates = bool(covariates.names)
    matched = [(row, pheno_lookup[key], cov_lookup.get(key, -1)) for row, key in enumerate(keys)
               if key in pheno_lookup and (not uses_covariates or key in cov_lookup)]
    cache_rows = np.asarray([item[0] for item in matched], dtype=np.int64)
    pheno_rows = np.asarray([item[1] for item in matched], dtype=np.int64)
    cov_rows = np.asarray([item[2] for item in matched], dtype=np.int64)
    y = phenotypes.values[pheno_rows, phenotypes.names.index(trait)]
    x = covariates.values[cov_rows] if uses_covariates else np.empty((len(y), 0), dtype=np.float64)
    from .phewas_inputs import apply_phenotype_missing_policy
    policy = apply_phenotype_missing_policy(y, family=family, rm_nan=rm_nan,
        covariate_valid=np.isfinite(x).all(axis=1))
    selected = policy['sample_indices']
    keep = np.zeros(len(y), dtype=bool)
    keep[selected] = True
    identifiers, y, x = keys[cache_rows][selected], policy['values'], x[selected]
    has_intercept = bool(x.shape[1] and np.any(np.all(x == 1, axis=0)))
    names = covariates.names
    design = None
    if reduce_redundant:
        if not len(y):
            raise ValueError(f"phenotype {trait}: insufficient complete-case samples for the covariate design")
        from .phewas_run import _design
        x, design = _design(x, names)
        names = tuple(design['covariate_names'])
    elif not has_intercept:
        x = np.column_stack((np.ones(len(y)), x))
        names = ("Intercept", *names)
    if len(y) <= x.shape[1]:
        raise ValueError(f"phenotype {trait}: insufficient complete-case samples for the covariate design")
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError(f"phenotype {trait}: covariate design is rank deficient")
    cache_set = set(keys)
    counts = dict(phenotype_rows=len(phenotypes.values), covariate_rows=len(covariates.values),
        prepared_samples=len(cached), matched_rows_before_missing=len(matched),
        excluded_missing_rows=int((~keep).sum()), analysis_samples=len(y),
        phenotype_rows_absent_from_prepared=sum(key not in cache_set for key in pheno_lookup),
        phenotype_rows_absent_from_covariates=sum(key not in cov_lookup for key in pheno_lookup),
        intercept_added=not has_intercept, design_columns=x.shape[1],
        rm_nan=rm_nan, missing_policy=policy['missing_policy'],
        available_n=policy['available_n'], mother_n=policy['mother_n'],
        imputed_n=policy['imputed_n'], imputation_mean=policy['imputation_mean'])
    if design is not None:
        counts['covariate_design'] = design
    return identifiers, y, x, names, counts


def _cache_path(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("prepared dataset paths must be nonempty relative paths")
    path = Path(os.path.abspath(root / value))
    if not path.is_relative_to(root):
        raise ValueError("prepared dataset paths must remain within the prepared directory")
    return path


def _dataset(prepared_directory, chromosomes=None):
    root = Path(prepared_directory).resolve()
    manifest_path = root / "dataset.json"
    raw = manifest_path.read_bytes()
    try:
        complete = (root / "COMPLETE").read_text(encoding="utf-8").strip()
    except FileNotFoundError as error:
        raise ValueError("prepared dataset lacks its COMPLETE integrity proof") from error
    if complete != hashlib.sha256(raw).hexdigest():
        raise ValueError("prepared dataset COMPLETE hash differs from dataset.json")
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 2:
        raise ValueError("unsupported dataset.json schema_version; prepare the genetic data first")
    _analysis_settings(manifest)
    raw_entries = manifest.get("chromosomes")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("dataset.json must declare chromosome entries")
    if isinstance(chromosomes, (str, int)):
        chromosomes = [chromosomes]
    wanted = None if chromosomes is None else {str(value).removeprefix("chr") for value in chromosomes}
    entries, seen = [], set()
    for raw in raw_entries:
        entry = dict(raw)
        name = str(entry.get("name", "")).removeprefix("chr")
        if not name.isdecimal() or str(int(name)) != name or not 1 <= int(name) <= 22 or name in seen:
            raise ValueError("chromosome entries must have distinct names from 1 to 22")
        seen.add(name)
        if wanted is not None and name not in wanted:
            continue
        entry["name"] = name
        for key in ("container_directory", "metadata_directory", "gene_catalog", "promoter_intervals"):
            if key in entry:
                entry[key] = str(_cache_path(root, entry[key]))
        if not {"container_directory", "metadata_directory"} <= entry.keys():
            raise ValueError("each chromosome requires genotype and metadata directories")
        entries.append(entry)
    if wanted is not None and wanted - seen:
        raise ValueError("requested chromosomes are absent from the prepared dataset")
    if not entries:
        raise ValueError("no chromosomes selected")
    entries.sort(key=lambda entry: int(entry["name"]))
    pairs = validate_sample_pairs(np.load(_cache_path(root, manifest.get("sample_pairs", "sample_pairs.npy")),
                                        allow_pickle=False), where="prepared sample index")
    if "sample_ids" in manifest:
        stored = np.load(_cache_path(root, manifest["sample_ids"]), allow_pickle=False)
        if stored.ndim != 1 or not np.array_equal(stored, sample_keys(pairs)):
            raise ValueError("prepared sample keys disagree with the FID/IID sample axis")
    return root, manifest_path, manifest, entries, pairs


def _analysis_settings(manifest):
    """Require the prepared root's explicit scientific annotation settings."""
    required = ("annotation_catalog", "annotation_names", "qc_path")
    if any(name not in manifest for name in required):
        raise ValueError("prepared dataset requires annotation_catalog, annotation_names and qc_path")
    catalog, names, qc_path = (manifest[name] for name in required)
    if (not isinstance(catalog, dict) or
            any(not isinstance(key, str) or not key or not isinstance(value, str) or not value
                for key, value in catalog.items())):
        raise ValueError("prepared annotation_catalog must map nonempty names to nonempty field paths")
    if (not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or set(names) - set(catalog)):
        raise ValueError("prepared annotation_names must be distinct catalog names in weight order")
    if not isinstance(qc_path, str) or not qc_path:
        raise ValueError("prepared qc_path must be a nonempty field path")
    return dict(annotation_catalog=dict(catalog), annotation_names=list(names), qc_path=qc_path)


def _safe_name(name):
    if (not isinstance(name, str) or not name or name in (".", "..")
            or name in ("models", "plan.private.json", "report.private.json")
            or any(character in name for character in ("/", "\\", "\0"))
            or any(character.isspace() and character not in (" ",) for character in name)
            or name != name.strip() or Path(name).name != name):
        raise ValueError("phenotype names must be safe single-directory names")
    return name


def _proof(directory, expected_sha256):
    directory = Path(directory)
    raw = (directory / "manifest.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256 or (directory / "COMPLETE").read_text().strip() != digest:
        raise ValueError("immutable prepared genotype manifest changed")
    return json.loads(raw)["binding"]


def _annotation_coverage(entry, *, gene_variant_type, analyses, expected_sample_keys=None,
                         expected_analysis=None):
    """Validate annotation availability on the selected, bound variant axis."""
    from .cache_runtime.portable import PortableMetadataReader
    with PortableMetadataReader(entry["metadata_directory"], entry["container_directory"]) as reader:
        if expected_analysis is not None:
            analysis = reader.manifest.get("analysis")
            if not isinstance(analysis, dict) or any(name not in analysis for name in expected_analysis):
                raise ValueError("chromosome metadata requires explicit annotation_catalog, annotation_names and qc_path")
            if any(analysis[name] != value for name, value in expected_analysis.items()):
                raise ValueError("prepared dataset annotation mapping, weight order or QC differs from chromosome metadata")
        if expected_sample_keys is not None and not np.array_equal(reader.sample_ids(), expected_sample_keys):
            raise ValueError("chromosome sample keys differ from the prepared dataset FID/IID axis")
        coverage = reader.manifest.get("annotation_coverage")
        if not isinstance(coverage, dict):
            raise ValueError("prepared metadata lacks selected-axis annotation coverage")
        for name in ("n_variants", "matched_variants", "missing_snv", "missing_non_snv"):
            if type(coverage.get(name)) is not int or coverage[name] < 0:
                raise ValueError("invalid selected-axis annotation coverage counts")
        if (coverage["n_variants"] != reader.n_variants or
                coverage["matched_variants"] + coverage["missing_snv"] + coverage["missing_non_snv"] != reader.n_variants):
            raise ValueError("annotation coverage does not match the selected variant axis")
        availability = np.asarray(reader.read_field(coverage.get("available_field", "annotation_available")))
        if availability.shape != (reader.n_variants,) or availability.dtype.kind != "b":
            raise ValueError("annotation availability must be a boolean selected-variant vector")
        if int(availability.sum()) != coverage["matched_variants"]:
            raise ValueError("annotation availability differs from the coverage proof")
        if coverage["missing_snv"]:
            raise ValueError("selected SNVs lack required functional annotations")
        if coverage["missing_non_snv"]:
            if coverage.get("allow_missing_non_snv") is not True:
                raise ValueError("missing non-SNV annotations require an explicit preparation policy")
            if gene_variant_type != "SNV" and any(kind != "individual" for kind in analyses):
                raise ValueError("Indel or all-variant gene analysis requires complete non-SNV annotations")
        if "individual" in analyses:
            entry["_maximum_position"] = int(np.asarray(reader.read_field("position"), dtype=np.int64).max(initial=0))
        return dict(coverage)


def _jobs(entry, analyses, output, *, single_mac_cutoff, single_group_variants, single_region_size):
    jobs, chromosome = [], entry["name"]
    if "individual" in analyses:
        maximum = entry.get("_maximum_position")
        if maximum is None:
            from .cache_runtime.portable import PortableMetadataReader
            with PortableMetadataReader(entry["metadata_directory"], entry["container_directory"]) as reader:
                maximum = int(np.asarray(reader.read_field("position"), dtype=np.int64).max(initial=0))
        for block, start in enumerate(range(1, maximum + 1, single_region_size), 1):
            jobs.append(dict(name=f"single_{block:04d}", kind="individual", layout="base",
                output=str(output / f"chr{chromosome}_single_{block:04d}.csv"),
                arguments=dict(start=start, end=min(start + single_region_size - 1, maximum),
                    mac_cutoff=single_mac_cutoff, subset_variants_num=single_group_variants, variant_type="variant")))
    gene_kinds = set(analyses) - {"individual"}
    if gene_kinds:
        if "gene_catalog" not in entry:
            raise ValueError("gene analyses require an annotated gene_catalog for each chromosome")
        genes = json.loads(Path(entry["gene_catalog"]).read_text(encoding="utf-8"))
        if isinstance(genes, dict):
            genes = genes.get("jobs", genes.get("genes"))
        if not isinstance(genes, list):
            raise ValueError("gene_catalog must contain an ordered list of gene jobs")
        for ordinal, gene in enumerate(genes):
            kind = gene.get("kind")
            if kind not in gene_kinds:
                continue
            arguments = dict(gene.get("arguments", {}))
            arguments.pop("chromosome", None)
            arguments.pop("promoter_intervals_file", None)
            for key in ("gene_name", "start", "end", "category", "include_ptv", "include_ncrna"):
                if key in gene:
                    arguments[key] = gene[key]
            if not isinstance(arguments.get("gene_name"), str) or not arguments["gene_name"]:
                raise ValueError("gene jobs require a nonempty gene_name")
            if kind == "coding" and not {"start", "end"} <= arguments.keys():
                raise ValueError("coding jobs require inclusive start and end coordinates")
            if kind == "noncoding":
                if "promoter_intervals" not in entry:
                    raise ValueError("noncoding analyses require exact promoter intervals")
                arguments["promoter_intervals_file"] = entry["promoter_intervals"]
            jobs.append(dict(name=f"{kind}_{ordinal:06d}", kind=kind, layout="base", arguments=arguments,
                             output=str(output / f"chr{chromosome}_{kind}.csv")))
    if not jobs:
        raise ValueError("no association jobs found for the requested chromosome and analyses")
    return jobs


def _run_wgs_association_unlocked(phenotype_csv, covariate_csv, prepared_directory, *, output_directory,
                chromosomes="all", analyses=_KINDS, device=("cuda:0",), cpu_threads=8,
                memory_limit_gib=None, memory_reserve_gib=1.0, host_memory_limit_gib=None,
                phenotype_families="auto", phenotype_covariates=None, rm_nan=True,
                transform="none", null_fit_mode="fp64", gene_variant_type="SNV",
                covariance_block_size=4096, long_mask_threshold=5000, long_mask_method="full", long_mask_rank=512, seed=1729,
                single_mac_cutoff=20, single_group_variants=5000, single_region_size=10_000_000,
                individual_effective_block_size=1024, device_cache_bytes=512*2**20,
                compact_cache_bytes=64*2**20, metadata_cache_bytes=256*2**20, resume=False,
                gpu_ids=None, trait_batch_size=16, continuous_transform=None,
                grm="sparse", grm_file=None, input_contract="mixed",
                association_partition_strategy="cohort_tiles"):
    """Associate every numeric phenotype independently with existing WGS caches.

    chromosomes='all' consumes all declared autosomes; a list selects a subset.
    device=['cuda:0'] selects one GPU, while integer n selects the first n GPUs.
    None uses physical GPU capacity with memory_reserve_gib kept for workspace.
    host_memory_limit_gib bounds the whole node task, shared across GPU workers;
    None detects all finite cgroup ancestor limits and physical capacity.
    Exact finite 0/1 columns are binomial in auto mode; other columns Gaussian.
    rm_nan=True fills Gaussian missing phenotypes with their eligible mean and
    maps binomial values other than one, including missing, to zero. False
    selects complete cases. Selected covariates are never imputed. Unlisted
    traits use all supplied covariates; changing rm_nan requires new null fits.
    grm='sparse' requires an ID-bound GRM; grm='none' selects ordinary nulls.
    resume verifies immutable per-block CSV/model/input/source proofs and fills
    missing blocks. A new full run (resume=False) requires a fresh output folder.
    Both supported cache formats use genetic-major GPU phenotype tiles and
    emit complete compressed text CSV shards and a private trait dictionary.
    Large fitted banks partition ordered cohorts across GPUs; when too few
    cohorts fit the workers or a cohort exceeds cache capacity, fixed global
    phenotype tiles divide that cohort. Smaller banks partition genetic jobs.
    Unique jobs and worker scan visits are
    reported separately, with global trait IDs and model bindings preserved.
    long_mask_method="full" computes complete native FP32 gene spectra with
    live GPU admission. fastskat/liu explicitly retain threshold-based
    approximations; long_mask_threshold is not the default spectrum switch.
    Genetic preparation is never invoked by this association entry point.
    """
    if input_contract not in ("mixed", "continuous"):
        raise ValueError("unknown phenotype input contract")
    if association_partition_strategy not in ("cohort_tiles", "chromosomes"):
        raise ValueError("unknown association partition strategy")
    if input_contract == "continuous" and (phenotype_families != "auto" or phenotype_covariates is not None):
        raise ValueError("continuous analysis requires shared covariates and automatic continuous input validation")
    chromosomes = _normalise_chromosomes(chromosomes)
    if type(rm_nan) is not bool:
        raise ValueError("rm_nan must be a boolean")
    if type(resume) is not bool:
        raise ValueError("resume must be a boolean")
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    for name, value, allow_none in (("memory_limit_gib", memory_limit_gib, True),
                                     ("host_memory_limit_gib", host_memory_limit_gib, True),
                                     ("memory_reserve_gib", memory_reserve_gib, False)):
        if allow_none and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (allow_none and value == 0):
            raise ValueError(name + " must be finite and positive" + (" or None" if allow_none else " or zero"))
    if memory_limit_gib is not None and memory_reserve_gib >= memory_limit_gib:
        raise ValueError("memory_reserve_gib must be below the GPU memory limit")
    if not isinstance(analyses, (list, tuple)):
        raise ValueError("analyses must be a sequence of analysis names")
    kinds = tuple(dict.fromkeys(analyses))
    if not kinds or any(k not in _KINDS for k in kinds):
        raise ValueError("analyses must select individual, coding, noncoding or ncrna")
    positive = (covariance_block_size, long_mask_threshold, long_mask_rank, single_mac_cutoff,
                single_group_variants, single_region_size, individual_effective_block_size, trait_batch_size)
    if any(type(v) is not int or v < 1 for v in positive):
        raise ValueError("block sizes, trait tiles and MAC cutoff must be positive integers")
    if long_mask_method not in ("full", "fastskat", "liu"):
        raise ValueError("long_mask_method must be full, fastskat or liu")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if any(type(v) is not int or v < 0 for v in (device_cache_bytes, compact_cache_bytes)) or type(metadata_cache_bytes) is not int or metadata_cache_bytes < 1:
        raise ValueError("cache capacities must be nonnegative; metadata capacity must be positive")
    if transform not in ("none", "rint") or null_fit_mode not in ("fp64", "tf32"):
        raise ValueError("transform must be none or rint; null_fit_mode must be fp64 or tf32")
    if continuous_transform not in (None, "paper", "none", "rint"):
        raise ValueError("continuous_transform must be None, paper, none or rint")
    if gene_variant_type not in ("SNV", "Indel", "variant"):
        raise ValueError("gene_variant_type must be SNV, Indel or variant")
    if grm not in ("sparse", "none") or (grm == "none" and grm_file is not None):
        raise ValueError("grm must be sparse or none; grm_file requires sparse mode")
    population = (Path(prepared_directory)/"cache_dataset.json").is_file()
    if population and (Path(prepared_directory)/"dataset.json").exists():
        raise ValueError("prepared directory contains ambiguous old and new dataset roots")
    if transform != "none":
        if continuous_transform not in (None, transform):
            raise ValueError("conflicting phenotype transformations")
        continuous_transform = transform
    if continuous_transform is None:
        continuous_transform = "paper" if population else "none"
    devices = _normalise_devices(device, gpu_ids)
    options = dict(analyses=kinds, cpu_threads=cpu_threads, memory_limit_gib=memory_limit_gib,
        memory_reserve_gib=float(memory_reserve_gib), host_memory_limit_gib=host_memory_limit_gib, phenotype_families=phenotype_families,
        phenotype_covariates=phenotype_covariates, rm_nan=rm_nan, gene_variant_type=gene_variant_type,
        covariance_block_size=covariance_block_size, long_mask_threshold=long_mask_threshold,
        long_mask_method=long_mask_method, long_mask_rank=long_mask_rank, seed=seed, single_mac_cutoff=single_mac_cutoff,
        single_group_variants=single_group_variants, single_region_size=single_region_size,
        individual_effective_block_size=individual_effective_block_size,
        device_cache_bytes=device_cache_bytes, compact_cache_bytes=compact_cache_bytes,
        metadata_cache_bytes=metadata_cache_bytes, resume=resume, grm=grm, grm_file=grm_file,
        trait_batch_size=trait_batch_size, input_contract=input_contract,
        association_partition_strategy=association_partition_strategy)
    started = time.perf_counter()
    output = Path(output_directory).resolve()
    attempt_id = uuid.uuid4().hex
    if resume and (output/"plan.private.json").is_file():
        previous = output/"report.private.json"
        if previous.exists():
            previous.replace(output/f"report.previous.{attempt_id}.private.json")
        _json_write(output/"status.private.json", dict(status="running", attempt_id=attempt_id))
    try:
        if population:
            if null_fit_mode != "fp64":
                raise ValueError("mixed PheWAS null fitting requires fp64")
            from .phewas_run import run_phewas
            workers = run_phewas(phenotype_csv, covariate_csv, prepared_directory,
                output_directory=output_directory, gpu_ids=devices, chromosomes=chromosomes,
                continuous_transform=continuous_transform, **options)
        else:
            from .prepared_association import run_prepared_association
            workers = run_prepared_association(phenotype_csv, covariate_csv, prepared_directory,
                output_directory=output_directory, gpu_ids=devices, chromosomes=chromosomes,
                transform=continuous_transform, null_fit_mode=null_fit_mode, **options)
        if not isinstance(workers, list) or len(workers) != len(devices):
            raise RuntimeError("WGS execution returned incomplete worker reports")
        rows = dict.fromkeys(_KINDS, 0)
        for worker in workers:
            counts = worker.get("association_rows", {})
            if any(type(counts.get(kind)) is not int or counts[kind] < 0 for kind in _KINDS):
                raise RuntimeError("worker report has invalid association row counts")
            for kind in _KINDS:
                rows[kind] += counts[kind]
        dictionary_sha = None
        if (output/"plan.private.json").is_file():
            expected = json.loads((output/"plan.private.json").read_text())["trait_index_sha256"]
            dictionary_sha = _sha256(output/"trait_index.private.csv")
            if dictionary_sha != expected:
                raise ValueError("completed phenotype dictionary differs from the bound input order")
        report = dict(completed=True, schema_version=3, workflow="shared_cache_phewas",
            input_contract=input_contract, association_partition_strategy=association_partition_strategy,
            attempt_id=attempt_id,
            trait_index_sha256=dictionary_sha,
            eligible_association_tests=sum(rows.values()), association_rows=rows, workers=workers,
            gpu_ids=list(devices), end_to_end_seconds=time.perf_counter()-started,
            output_directory=str(output))
        completions = [worker.get('global_completion') for worker in workers]
        if any(completion is not None for completion in completions):
            if (any(completion is None for completion in completions)
                    or any(completion!=completions[0] for completion in completions)):
                raise RuntimeError('worker global completion bindings differ')
            report.update(completions[0])
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        _json_write(output/"report.private.json", report)
        _json_write(output/"status.private.json", dict(status="complete", attempt_id=attempt_id,
            report_sha256=_sha256(output/"report.private.json")))
        return report
    except BaseException as error:
        current = output/"report.private.json"
        if current.exists():
            published = json.loads(current.read_text())
            if published.get("attempt_id") == attempt_id:
                current.replace(output/f"report.failed.{attempt_id}.private.json")
        if output.is_dir() and (resume or not current.exists()):
            _json_write(output/"status.private.json", dict(status="failed", attempt_id=attempt_id,
                error_type=type(error).__name__))
        raise


def _run_wgs_association(*args, **kwargs):
    """Own this output attempt before changing any plan or resume receipt."""
    from .phewas_trait_csv import association_output_lease
    with association_output_lease(kwargs['output_directory']):
        return _run_wgs_association_unlocked(*args, **kwargs)


def run_WGS_all(phenotype_csv, covariate_csv, prepared_directory, *, output_directory,
                chromosomes="all", analyses=_KINDS, device=("cuda:0",), cpu_threads=8,
                memory_limit_gib=None, memory_reserve_gib=1.0, host_memory_limit_gib=None,
                phenotype_families="auto", phenotype_covariates=None, rm_nan=True,
                transform="none", null_fit_mode="fp64", gene_variant_type="SNV",
                covariance_block_size=4096, long_mask_threshold=5000, long_mask_method="full", long_mask_rank=512, seed=1729,
                single_mac_cutoff=20, single_group_variants=5000, single_region_size=10_000_000,
                individual_effective_block_size=1024, device_cache_bytes=512*2**20,
                compact_cache_bytes=64*2**20, metadata_cache_bytes=256*2**20, resume=False,
                gpu_ids=None, trait_batch_size=16, continuous_transform=None,
                grm="sparse", grm_file=None):
    """Run independent continuous and binary traits on an existing WGS cache.

    This compatibility entry retains the mixed-trait options and dispatch.
    See ``run_WGS_continuous`` for the continuous-only chromosome workflow.
    """
    return _run_wgs_association(**locals())


def main(argv=None):
    def positive_integer(value):
        try:
            result = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be a positive integer") from error
        if result < 1:
            raise argparse.ArgumentTypeError("must be a positive integer")
        return result
    parser = argparse.ArgumentParser(description="Run WGS associations from FID/IID CSVs and prepared genetic data")
    for name in ("phenotype_csv","covariate_csv","prepared_directory"):parser.add_argument(name)
    parser.add_argument("--output-directory",required=True)
    parser.add_argument("--chromosomes",nargs="+",default="all",help="all or chromosome numbers from 1 to 22")
    parser.add_argument("--analyses",nargs="+",choices=_KINDS,default=_KINDS)
    parser.add_argument("--device",nargs="+",default=["cuda:0"],help="CUDA names, or one integer selecting that many GPUs")
    parser.add_argument("--gpu-ids",type=int,nargs="+",
                        help="local CUDA device IDs for an existing GRM-bound population cache")
    parser.add_argument("--trait-batch-size",type=positive_integer,default=16)
    parser.add_argument("--continuous-transform",choices=("paper","none","rint"),default=None)
    parser.add_argument("--cpu-threads",type=positive_integer,default=8)
    parser.add_argument("--memory-limit-gib",type=float,default=_DEFAULT_MEMORY_LIMIT)
    parser.add_argument("--memory-reserve-gib",type=float,default=1.0)
    parser.add_argument("--host-memory-limit-gib",type=float,default=None,
                        help="shared host task budget in GiB; default detects physical and all finite cgroup limits")
    parser.add_argument("--phenotype-families",default="auto",help="auto, or a JSON file mapping phenotype names to gaussian or binomial")
    parser.add_argument("--phenotype-covariates",help="JSON file mapping phenotype names to selected covariate column names")
    parser.add_argument("--transform",choices=("none","rint"),default="none")
    parser.add_argument("--null-fit-mode",choices=("fp64","tf32"),default="fp64")
    parser.add_argument("--gene-variant-type", choices=("SNV", "Indel", "variant"), default=_DEFAULT_GENE_VARIANT)
    for name,default in (("covariance-block-size",4096),("long-mask-threshold",5000),("long-mask-rank",512),
        ("seed",1729),("single-mac-cutoff",20),("single-group-variants",5000),("single-region-size",10_000_000),
        ("individual-effective-block-size",1024),("device-cache-bytes",512*2**20),
        ("compact-cache-bytes",64*2**20),("metadata-cache-bytes",256*2**20)):
        parser.add_argument("--"+name,type=int,default=default)
    parser.add_argument("--long-mask-method", choices=("full", "fastskat", "liu"), default="full",
                        help="full native FP32 spectra (default), or explicit legacy approximation")
    parser.add_argument("--rm-nan", action=argparse.BooleanOptionalAction, default=True,
        help="fill missing phenotypes (default); --no-rm-nan selects available complete cases")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--grm",choices=("sparse","none"),default="sparse")
    parser.add_argument("--grm-file",help="ID-bound sparse GRM NPZ; otherwise use the prepared dataset sidecar")
    arguments = vars(parser.parse_args(argv))
    if len(arguments["device"]) == 1 and arguments["device"][0].isdecimal():
        arguments["device"] = int(arguments["device"][0])
    if arguments["phenotype_families"] != "auto":
        arguments["phenotype_families"] = json.loads(Path(arguments["phenotype_families"]).read_text(encoding="utf-8"))
    if arguments["phenotype_covariates"] is not None:
        arguments["phenotype_covariates"] = json.loads(Path(arguments["phenotype_covariates"]).read_text(encoding="utf-8"))
    report = run_WGS_all(**arguments)
    print(json.dumps({key:report.get(key) for key in ("completed","eligible_association_tests","end_to_end_seconds","output_directory")},indent=2))


if __name__=="__main__":main()
