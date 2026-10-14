"""Continuous-only multi-phenotype WGS on an existing genetic cache."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from numbers import Integral, Real

import numpy as np

from .run import _KINDS


def _continuous_performance_configuration(configuration):
    """Copy execution defaults for the verified continuous-only workflow.

    Scientific options and the ordered genetic schedule are borrowed without
    mutation. All added fields belong to the returned top-level mapping, so
    both plan builders include the actual execution choices in their existing
    source/configuration and resume bindings. Batched spectra bound execution
    geometry only; larger masks retain the complete sequential FULL spectrum.
    Writer fsync keeps its existing serial default.
    """
    result = dict(configuration)
    full = result.get('analysis_options', {}).get('long_mask_method', 'full') == 'full'
    result.update(_continuous_shared_panels=True,
                  _continuous_resident_preparation=True,
                  _continuous_covariance_tiles=True,
                  _continuous_live_resident=True,
                  _continuous_tail_groups=full,
                  _continuous_complete_spectrum_batch=full,
                  _continuous_spectrum_reuse=full,
                  _continuous_complete_spectrum_batch_parameters=(dict(
                      maximum_variants=1536, weight_batch_size=32,
                      relation_workspace_bytes=256*2**10) if full else None))
    return result


def validate_continuous_traits(families, quality):
    """Validate parsed traits without rereading the phenotype CSV.

    ``families`` retains input column order. ``quality`` contains the same
    ordered trait summaries. Eligible finite extrema take precedence over raw
    extrema, so values excluded by shared covariate missingness cannot make a
    constant analysis population appear valid.
    """
    if not isinstance(families, Mapping) or len(families) < 2:
        raise ValueError("run_WGS_continuous requires at least two continuous phenotype columns")
    if not isinstance(quality, (list, tuple)) or len(quality) != len(families):
        raise ValueError("continuous phenotype quality summaries must match the input columns")
    for index, ((name, family), summary) in enumerate(zip(families.items(), quality)):
        if not isinstance(name, str) or not name or family != "gaussian":
            raise ValueError("run_WGS_continuous accepts continuous phenotypes only; exact 0/1 columns are binary")
        if not isinstance(summary, Mapping) or summary.get("index", index) != index:
            raise ValueError("continuous phenotype quality summaries are out of input order")
        available = summary.get("available_n")
        if isinstance(available, bool) or not isinstance(available, Integral) or available < 2:
            raise ValueError(f"phenotype {name}: at least two eligible finite values are required")
        minimum = summary.get("eligible_minimum", summary.get("minimum"))
        maximum = summary.get("eligible_maximum", summary.get("maximum"))
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
               for value in (minimum, maximum)):
            raise ValueError(f"phenotype {name}: eligible finite phenotype bounds are required")
        if minimum >= maximum:
            raise ValueError(f"phenotype {name}: the eligible continuous phenotype is constant")


def validate_continuous_metadata(metadata):
    """Validate immutable prepared input metadata for this public workflow."""
    if not isinstance(metadata, Mapping):
        raise ValueError("continuous input metadata must be a mapping")
    traits = metadata.get("traits")
    if not isinstance(traits, list) or any(not isinstance(trait, Mapping) for trait in traits):
        raise ValueError("continuous input metadata must contain ordered trait records")
    if any(trait.get("index") != index for index, trait in enumerate(traits)):
        raise ValueError("continuous trait metadata is out of input order")
    names = [trait.get("name") for trait in traits]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("continuous phenotype column names must be nonempty and unique")
    families = {trait["name"]: trait.get("family") for trait in traits}
    validate_continuous_traits(families, metadata.get("quality"))
    covariates = metadata.get("covariate_names")
    profiles = metadata.get("profile_covariate_indices")
    if covariates is not None or profiles is not None:
        if not isinstance(covariates, list) or profiles != [list(range(len(covariates)))]:
            raise ValueError("continuous phenotypes must share all supplied covariate columns")
        if any(trait.get("profile_index") != 0 or trait.get("covariate_indices") != profiles[0]
               for trait in traits):
            raise ValueError("continuous phenotypes must share one covariate profile")


def validate_continuous_values(names, values):
    """Validate a previously parsed numeric matrix with bounded column work."""
    from .run import _family_selection
    if not isinstance(names, (tuple, list)) or len(set(names)) != len(names):
        raise ValueError("continuous phenotype column names must be distinct")
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != len(names) or array.dtype.kind not in "fiu":
        raise ValueError("continuous phenotypes require an aligned numeric matrix")
    families = _family_selection("auto", names, array)
    quality = []
    for index in range(len(names)):
        column = array[:, index]
        if np.isinf(column).any():
            raise ValueError("infinity is not a valid continuous phenotype value")
        finite = column[np.isfinite(column)]
        quality.append({"index": index, "available_n": int(len(finite)),
                        "minimum": float(finite.min()) if len(finite) else None,
                        "maximum": float(finite.max()) if len(finite) else None})
    validate_continuous_traits(families, quality)
    return families, quality


def run_WGS_continuous(phenotype_csv, covariate_csv, prepared_directory, *, output_directory,
                      chromosomes="all", analyses=_KINDS, device=("cuda:0",), cpu_threads=8,
                      memory_limit_gib=None, memory_reserve_gib=1.0, host_memory_limit_gib=None,
                      rm_nan=True, continuous_transform="none", null_fit_mode="fp64",
                      gene_variant_type="SNV", covariance_block_size=4096,
                      long_mask_threshold=5000, long_mask_method="full", long_mask_rank=512,
                      seed=1729, single_mac_cutoff=20, single_group_variants=5000,
                      single_region_size=10_000_000, individual_effective_block_size=1024,
                      device_cache_bytes=512*2**20, compact_cache_bytes=64*2**20,
                      metadata_cache_bytes=256*2**20, resume=False, gpu_ids=None,
                      trait_batch_size=32, grm="sparse", grm_file=None):
    """Run two or more continuous phenotype columns using shared covariates.

    The three scientific inputs are the numeric phenotype CSV, numeric
    covariate CSV and existing genetic cache. Exact 0/1, entirely missing and
    constant phenotype columns are rejected. Every trait uses all covariate
    columns; covariates are never imputed. ``rm_nan=True`` fills phenotype
    NaNs with that trait's finite mean among its eligible samples. ``False``
    retains each trait's available samples, including its own order and
    genotype statistics. No transformation is applied unless explicitly
    requested through ``continuous_transform``.

    Multiple GPUs own disjoint complete chromosomes, with all traits tiled
    within each worker. Single physical frames, identical ordered sample
    axes, model-independent preparation and bounded caches use the shared
    association engine. Different sample axes retain independent filtering,
    projections and null models. Explicit GPU and whole-task host budgets,
    live admission, CSV checkpoint integrity and resume remain enforced.
    ``long_mask_method='full'`` uses complete native FP32 spectra at every
    mask size. Genetic data preparation is never invoked here.
    """
    from .run import _run_wgs_association
    return _run_wgs_association(
        phenotype_csv, covariate_csv, prepared_directory,
        output_directory=output_directory, chromosomes=chromosomes, analyses=analyses,
        device=device, cpu_threads=cpu_threads, memory_limit_gib=memory_limit_gib,
        memory_reserve_gib=memory_reserve_gib, host_memory_limit_gib=host_memory_limit_gib,
        phenotype_families="auto", phenotype_covariates=None, rm_nan=rm_nan,
        transform="none", null_fit_mode=null_fit_mode, gene_variant_type=gene_variant_type,
        covariance_block_size=covariance_block_size, long_mask_threshold=long_mask_threshold,
        long_mask_method=long_mask_method, long_mask_rank=long_mask_rank, seed=seed,
        single_mac_cutoff=single_mac_cutoff, single_group_variants=single_group_variants,
        single_region_size=single_region_size,
        individual_effective_block_size=individual_effective_block_size,
        device_cache_bytes=device_cache_bytes, compact_cache_bytes=compact_cache_bytes,
        metadata_cache_bytes=metadata_cache_bytes, resume=resume, gpu_ids=gpu_ids,
        trait_batch_size=trait_batch_size, continuous_transform=continuous_transform,
        grm=grm, grm_file=grm_file, input_contract="continuous",
        association_partition_strategy="chromosomes")


def main(argv=None):
    """Command line interface with no family or per-trait covariate overrides."""
    parser = argparse.ArgumentParser(
        description="Run continuous-only multi-phenotype WGS on prepared genetic data")
    for name in ("phenotype_csv", "covariate_csv", "prepared_directory"):
        parser.add_argument(name)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--chromosomes", nargs="+", default="all",
                        help="all, or selected chromosome numbers from 1 to 22")
    parser.add_argument("--analyses", nargs="+", choices=_KINDS, default=_KINDS)
    parser.add_argument("--device", nargs="+", default=["cuda:0"],
                        help="CUDA device names, or one integer selecting that many GPUs")
    parser.add_argument("--gpu-ids", type=int, nargs="+")
    for name, default in (("cpu-threads", 8), ("trait-batch-size", 32),
                          ("covariance-block-size", 4096), ("long-mask-threshold", 5000),
                          ("long-mask-rank", 512), ("seed", 1729), ("single-mac-cutoff", 20),
                          ("single-group-variants", 5000), ("single-region-size", 10_000_000),
                          ("individual-effective-block-size", 1024),
                          ("device-cache-bytes", 512*2**20), ("compact-cache-bytes", 64*2**20),
                          ("metadata-cache-bytes", 256*2**20)):
        parser.add_argument("--"+name, type=int, default=default)
    parser.add_argument("--memory-limit-gib", type=float, default=None)
    parser.add_argument("--memory-reserve-gib", type=float, default=1.0)
    parser.add_argument("--host-memory-limit-gib", type=float, default=None)
    parser.add_argument("--rm-nan", action=argparse.BooleanOptionalAction, default=True,
                        help="fill eligible phenotype NaNs with their mean; --no-rm-nan uses available samples")
    parser.add_argument("--continuous-transform", choices=("none", "rint", "paper"), default="none")
    parser.add_argument("--null-fit-mode", choices=("fp64", "tf32"), default="fp64")
    parser.add_argument("--gene-variant-type", choices=("SNV", "Indel", "variant"), default="SNV")
    parser.add_argument("--long-mask-method", choices=("full", "fastskat", "liu"), default="full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--grm", choices=("sparse", "none"), default="sparse")
    parser.add_argument("--grm-file")
    arguments = vars(parser.parse_args(argv))
    if len(arguments["device"]) == 1 and arguments["device"][0].isdecimal():
        arguments["device"] = int(arguments["device"][0])
    report = run_WGS_continuous(**arguments)
    print(json.dumps({key: report.get(key) for key in
                     ("completed", "eligible_association_tests", "end_to_end_seconds", "output_directory")},
                     indent=2))


if __name__ == "__main__":
    main()
