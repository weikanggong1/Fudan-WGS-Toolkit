"""Analyze every binary phenotype column using an existing genetic directory.

The input CSV must contain FID, IID and at least two phenotype columns.
Only 0, 1 and missing labels are accepted. This example prepares no new
genetic data and writes no simulated scientific results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fudan_wgs_toolkit import run_WGS_all_binanry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phenotype_csv", type=Path)
    parser.add_argument("covariate_csv", type=Path)
    parser.add_argument("prepared_directory", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--chromosomes", nargs="+", default=["all"])
    parser.add_argument("--device", nargs="+", default=["cuda:0"])
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--trait-batch-size", type=int, default=32)
    parser.add_argument("--memory-limit-gib", type=float)
    parser.add_argument("--host-memory-limit-gib", type=float)
    parser.add_argument("--phenotype-covariates", type=Path)
    parser.add_argument("--impute-missing", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--grm", choices=("sparse", "none"), default="sparse")
    parser.add_argument("--grm-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()

    chromosomes = (
        "all" if arguments.chromosomes == ["all"]
        else [int(chromosome) for chromosome in arguments.chromosomes]
    )
    device = (
        int(arguments.device[0])
        if len(arguments.device) == 1 and arguments.device[0].isdecimal()
        else arguments.device
    )
    phenotype_covariates = (
        None if arguments.phenotype_covariates is None
        else json.loads(arguments.phenotype_covariates.read_text(encoding="utf-8"))
    )
    # The same covariate set applies to every phenotype unless explicitly mapped.
    # Multiple GPUs own distinct whole chromosomes and consume all phenotypes.
    association_report = run_WGS_all_binanry(
        phenotype_csv=arguments.phenotype_csv,
        covariate_csv=arguments.covariate_csv,
        prepared_directory=arguments.prepared_directory,
        output_directory=arguments.output_directory,
        chromosomes=chromosomes,
        analyses=("individual", "coding", "noncoding", "ncrna"),
        device=device,
        cpu_threads=arguments.cpu_threads,
        trait_batch_size=arguments.trait_batch_size,
        memory_limit_gib=arguments.memory_limit_gib,
        host_memory_limit_gib=arguments.host_memory_limit_gib,
        phenotype_covariates=phenotype_covariates,
        impute_missing=arguments.impute_missing,
        gene_variant_type="SNV",
        grm=arguments.grm,
        grm_file=arguments.grm_file,
        resume=arguments.resume,
    )
    print(json.dumps({"completed": association_report["completed"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
