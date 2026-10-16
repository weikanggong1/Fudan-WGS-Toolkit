"""Stream invented sparse genotypes and annotations directly into prepared data."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from .cache_runtime import sparse_codec_fast as csr
from .cache_runtime import store
from .cache_runtime.portable import export_metadata
from .identity import sample_keys
from .masks import CODING_CATEGORIES_INCL_PTV, NONCODING_CATEGORIES
from .simulation_profile import normalize_profile

ANNOTATION_CATALOG = {
    "CADD": "annotation/cadd", "aPC.LocalDiversity": "annotation/local_diversity",
    "aPC.Epigenetics": "annotation/epigenetics", "GENCODE.Category": "annotation/category",
    "GENCODE.EXONIC.Category": "annotation/exonic_category", "GENCODE.Info": "annotation/gene_info",
    "MetaSVM": "annotation/meta_svm", "CAGE": "annotation/cage", "DHS": "annotation/dhs",
    "GeneHancer": "annotation/enhancer"}
ANNOTATION_NAMES = ["CADD", "aPC.LocalDiversity", "aPC.Epigenetics"] + [
    f"Synthetic.Score{number:02d}" for number in range(1, 9)]
# Preserve the reference channel count for an SNV weighting workload, without
# copying additional database field names or claiming functional annotation.
ANNOTATION_CATALOG.update({f"Synthetic.Score{number:02d}": f"annotation/synthetic_score{number:02d}"
                           for number in range(1, 9)})
QC_PATH = "annotation/qc"


def _json(path, value):
    store.atomic(path, store.canonical(value))


def _position(rows, m, span, seed, chromosome):
    """Invent one distinct coordinate per integer stratum, within a span."""
    rows = np.asarray(rows, dtype=np.int64)
    first = rows*span//m
    width = (rows+1)*span//m-first
    state = rows.astype(np.uint64)+np.uint64((seed+int(chromosome)*31) % 2**64)
    state = (state ^ (state >> np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
    state = (state ^ (state >> np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    return first+(state % width.astype(np.uint64)).astype(np.int64)+1


def _counts(entry, n, reference_n):
    called, minor = [], []
    for value in entry["frequency_bins"]:
        an = round(value.get("called_alleles_mean", 2*n*(1-value["missing_rate"]))) if n == reference_n else round(2*n*(1-value["missing_rate"]))
        an = min(2*n, max(0, an))
        ac = round(value.get("minor_ac_mean", an*value["maf"])) if n == reference_n else round(an*value["maf"])
        called.append(an)
        minor.append(min(an//2, max(0, ac)))
    return np.asarray(called, dtype=np.int64), np.asarray(minor, dtype=np.int64)


def _apportion(bins, m, original_m):
    raw = np.asarray([value["count"] * m / original_m for value in bins], dtype=np.float64)
    counts = np.floor(raw).astype(np.int64)
    remainder = m-int(counts.sum())
    if remainder:
        counts[np.argsort(-(raw-counts), kind="stable")[:remainder]] += 1
    return counts


def _coding_parameters(known):
    """Infer coding marginal sets, including disruptive splice overlaps.

    A splice category can coexist with a nonsynonymous exonic annotation.
    Disruptive variants in coding splice categories belong to both pLoF and
    PTV. ncRNA splice categories belong only to pLoF. Therefore the two
    disruptive intersections may differ, but PTV's must be a subset of pLoF's.
    Unpublished masks are unknown, not zero. Infer a deterministic feasible
    extension of the observed marginals, choosing the smallest disruptive set.
    """
    lower_d = max([0, *(known[union]-known[base]
        for base, union in (("plof", "plof_ds"), ("ptv", "ptv_ds"))
        if base in known and union in known)])
    d = known.get("disruptive_missense", lower_d)
    valid_d = d >= lower_d and all(d <= known[name]
        for name in ("missense", "plof_ds", "ptv_ds") if name in known)

    def interval(base, union):
        if base in known and union in known:
            overlap = known[base]+d-known[union]
            return overlap, overlap, known[base]
        if base in known:
            return 0, min(d, known[base]), known[base]
        # If union is known then base = union-D+overlap, otherwise the
        # smallest compatible unobserved base is exactly its overlap.
        return 0, d, None

    xl_min, xl_max, l = interval("plof", "plof_ds")
    xp_min, xp_max, p = interval("ptv", "ptv_ds")
    xp, xl = xp_min, max(xl_min, xp_min)
    if l is None:
        l = known.get("plof_ds", d)-d+xl
    if p is None:
        p = known.get("ptv_ds", d)-d+xp
    t, s = known.get("missense", d), known.get("synonymous", 0)
    predicted = dict(plof=l, ptv=p, disruptive_missense=d, missense=t, synonymous=s,
                     plof_ds=l+d-xl, ptv_ds=p+d-xp)
    if (valid_d and 0 <= xp_min <= xp_max and 0 <= xl_min <= xl_max and
            0 <= xp <= xl <= d and xl <= xl_max and xl <= l and xp <= p and
            all(predicted[name] == count for name, count in known.items())):
        return (l, p, d, t, s, xl, xp), True
    # Retain a deterministic fixture and explicitly expose incompatible targets.
    d = known.get("disruptive_missense", 0)
    l = known.get("plof", max(0, known.get("plof_ds", d)-d))
    p = known.get("ptv", max(0, known.get("ptv_ds", d)-d))
    return (l, p, d, max(d, known.get("missense", d)), known.get("synonymous", 0), 0, 0), False


def _mask_plan(entry, scale):
    grouped = {}
    for target in entry["masks"]:
        grouped.setdefault((target["kind"], target["group"]), {})[target["category"]] = target["m"]
    segments, jobs, targets, problems, inferred = [], [], [], [], []
    cursor = 0
    chromosome = entry["name"]
    def add(count, kind, group, annotation, memberships):
        nonlocal cursor
        if count:
            segments.append(dict(start=cursor, stop=cursor+count, kind=kind, group=group,
                                 annotation=annotation, memberships=memberships))
            cursor += count
    for (kind, group), known in sorted(grouped.items()):
        gene = f"SIM{kind.upper()}_{chromosome}_{group:07d}"
        start = cursor
        for category, m in sorted(known.items()):
            targets.append(dict(kind=kind, group=group, category=category, source_m=m,
                                target_m=int(round(m*scale))))
        if kind == "coding":
            parameters, valid = _coding_parameters(known)
            if not valid:
                problems.append(dict(kind=kind, group=group, reason="published coding counts violate feasible splice/disruptive mask relationships"))
            l, p, d, t, s, xl, xp = (max(0, int(round(count*scale))) for count in parameters)
            expanded = dict(plof=l, ptv=p, disruptive_missense=d, missense=t,
                synonymous=s, plof_ds=l+d-xl, ptv_ds=p+d-xp)
            inferred.extend(dict(kind=kind, group=group, category=category,
                                 inferred_m=count, published_target_available=False)
                            for category, count in expanded.items() if category not in known)
            # Rounding preserves these subset inequalities. Marginal combined
            # mask rounding can differ by one at non-unit variant_scale and is
            # checked in the generated-mask audit rather than hidden.
            both = min(l-xl, p-xp)
            common = {"GENCODE.Info": gene}
            add(both, kind, group, dict(common, **{"GENCODE.Category": "exonic", "GENCODE.EXONIC.Category": "stopgain"}), ["plof", "plof_ds", "ptv", "ptv_ds"])
            add(max(0, l-xl-(p-xp)), kind, group, {"GENCODE.Category": "ncRNA_splicing", "GENCODE.Info": gene}, ["plof", "plof_ds"])
            add(max(0, p-xp-(l-xl)), kind, group, dict(common, **{"GENCODE.Category": "exonic", "GENCODE.EXONIC.Category": "frameshift insertion", "allele": "A,AT"}), ["ptv", "ptv_ds"])
            disruptive = dict(common, **{"GENCODE.EXONIC.Category": "nonsynonymous SNV", "MetaSVM": "D"})
            add(xp, kind, group, dict(disruptive, **{"GENCODE.Category": "splicing"}), ["plof", "plof_ds", "ptv", "ptv_ds", "missense", "disruptive_missense"])
            add(xl-xp, kind, group, dict(disruptive, **{"GENCODE.Category": "ncRNA_splicing"}), ["plof", "plof_ds", "ptv_ds", "missense", "disruptive_missense"])
            add(d-xl, kind, group, dict(disruptive, **{"GENCODE.Category": "exonic"}), ["missense", "disruptive_missense", "plof_ds", "ptv_ds"])
            add(max(0, t-d), kind, group, dict(common, **{"GENCODE.Category": "exonic", "GENCODE.EXONIC.Category": "nonsynonymous SNV", "MetaSVM": "T"}), ["missense"])
            add(s, kind, group, dict(common, **{"GENCODE.Category": "exonic", "GENCODE.EXONIC.Category": "synonymous SNV"}), ["synonymous"])
            if cursor > start:
                jobs.append(dict(kind=kind, arguments=dict(gene_name=gene, start=100001+2*start,
                    end=100001+2*(cursor-1), category="all_categories_incl_ptv", include_ptv=True)))
        elif kind == "noncoding":
            for category in NONCODING_CATEGORIES:
                count = max(0, int(round(known.get(category, 0)*scale)))
                annotation = {"GENCODE.Info": gene}
                if category in ("upstream", "downstream"):
                    annotation["GENCODE.Category"] = category
                elif category == "UTR":
                    annotation.update({"GENCODE.Category": "UTR3", "GENCODE.Info": gene+"(synthetic)"})
                elif category.startswith("promoter_"):
                    annotation.update({"GENCODE.Category": "intronic", category.split("_")[1]: "synthetic", "promoter": True})
                else:
                    annotation.update({"GENCODE.Category": "intergenic", "GENCODE.Info": "", category.split("_")[1]: "synthetic", "GeneHancer": "synthetic=region=gene="+gene+";invented"})
                add(count, kind, group, annotation, [category])
            if cursor > start:
                jobs.append(dict(kind=kind, arguments=dict(gene_name=gene, category="all_categories")))
        else:
            add(max(0, int(round(known.get("ncRNA", 0)*scale))), kind, group,
                {"GENCODE.Category": "ncRNA_exonic", "GENCODE.Info": gene+"(synthetic)"}, ["ncRNA"])
            if cursor > start:
                jobs.append(dict(kind=kind, arguments=dict(gene_name=gene)))
    return dict(segments=segments, assigned_variants=cursor, jobs=jobs,
                targets=targets, problems=problems, inferred_masks=inferred)


def plan_simulated_prepared(root, profile, n, chromosomes, seed=1729, cpu_threads=8,
                            memory_limit_gib=8, chunk_size=256, storage_limit_bytes=None,
                            variant_scale=1.0, variants_per_chromosome=None):
    """Read-only scheduling/storage preflight, also callable before mkdir."""
    profile = normalize_profile(profile)
    if type(n) is not int or not 128 <= n <= 2**32 or type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("n must be in 128..2**32 and cpu_threads must be positive")
    if type(seed) is not int or seed < 0 or type(chunk_size) is not int or not 1 <= chunk_size <= 1024:
        raise ValueError("invalid seed or frame chunk_size")
    if isinstance(memory_limit_gib, bool) or not isinstance(memory_limit_gib, (int, float)) or not math.isfinite(memory_limit_gib) or memory_limit_gib <= 0:
        raise ValueError("memory_limit_gib must be positive and finite")
    if isinstance(variant_scale, bool) or not isinstance(variant_scale, (int, float)) or not math.isfinite(variant_scale) or variant_scale <= 0:
        raise ValueError("variant_scale must be positive and finite")
    if isinstance(chromosomes, str) and chromosomes == "all":
        selected = [entry["name"] for entry in profile["chromosomes"]]
    else:
        if isinstance(chromosomes, (str, int)):
            chromosomes = [chromosomes]
        selected = [str(value).removeprefix("chr") for value in chromosomes]
    if len(selected) != len(set(selected)) or not selected:
        raise ValueError("chromosomes must be nonempty and unique")
    source = {entry["name"]: entry for entry in profile["chromosomes"]}
    if not set(selected) <= source.keys():
        raise ValueError("a requested chromosome is absent from the calibration profile")
    if variants_per_chromosome is not None:
        overrides = ({str(key).removeprefix("chr"): value for key, value in variants_per_chromosome.items()}
                     if isinstance(variants_per_chromosome, dict) else dict.fromkeys(selected, variants_per_chromosome))
        if set(overrides) != set(selected) or any(type(value) is not int or value < 1 for value in overrides.values()):
            raise ValueError("variant overrides must be positive counts for selected chromosomes")
    else:
        overrides = {}
    # Worst-case frame includes N exceptions/variant. Compact buffers, integer
    # count prefix arrays, compression and roundtrip scratch are all budgeted.
    sample_bytes = 512*n + 128*2**20
    # Python profile/segment/job/audit dictionaries remain global while
    # chromosomes stream. Their count follows the finite aggregate mask
    # catalog, not physical variants; admit them explicitly rather than
    # pretending all generation memory is in sparse arrays.
    plan_bytes = sum(len(entry["masks"])*4096+len(entry["frequency_bins"])*1024
                     for entry in profile["chromosomes"])
    global_bytes = sample_bytes+plan_bytes
    metadata_work = 65536*256 + 16*2**20
    budget = int(memory_limit_gib*2**30)
    if budget < global_bytes+metadata_work+64*2**20+n*80:
        raise MemoryError("memory_limit_gib cannot admit the synthetic sample axis and one sparse frame")
    max_workers = min(cpu_threads, len(selected))
    workers = max(1, min(max_workers, (budget-global_bytes)//(metadata_work+64*2**20+n*80)))
    per_worker = (budget-global_bytes)//workers
    raw_budget = min(csr.DEFAULT_MAX_BYTES, max(1, (per_worker-metadata_work-64*2**20)//8))
    actual_chunk = min(chunk_size, max(1, (raw_budget-4)//(n*(5 if n >= 65536 else 3)+4)))
    if csr._size(actual_chunk, n, actual_chunk*n) > raw_budget:
        raise MemoryError("memory_limit_gib cannot admit one worst-case sparse variant")
    entries = []
    total_disk = 0
    has_variable_locus = False
    for chromosome in selected:
        entry = source[chromosome]
        m = overrides.get(chromosome, max(1, int(round(entry["variants"]*variant_scale))))
        masks = _mask_plan(entry, m/entry["variants"])
        span = entry.get("position_span", max(m, 2*m))
        if span < m:
            raise ValueError("position_span cannot hold the requested number of distinct invented loci")
        if m*span > np.iinfo(np.int64).max:
            raise ValueError("requested synthetic coordinate strata exceed int64 arithmetic")
        for job in masks["jobs"]:
            if job["kind"] == "coding":
                arguments = job["arguments"]
                for endpoint in ("start", "end"):
                    ordinal = (arguments[endpoint]-100001)//2
                    arguments[endpoint] = int(_position([ordinal], m, span, seed, chromosome)[0])
        if masks["assigned_variants"] > m:
            raise ValueError("scaled disjoint mask atoms exceed chromosome variant count")
        frequency_counts = _apportion(entry["frequency_bins"], m, entry["variants"])
        called, minor = _counts(entry, n, profile["reference_n"])
        has_variable_locus |= any(count and 0 < ac < an for count, ac, an in zip(frequency_counts, minor, called))
        missing_samples = (2*n-called+1)//2
        nnz_bound = sum(int(count)*min(n, int(ac+missing)) for count, ac, missing in zip(frequency_counts, minor, missing_samples))
        frames = (m+actual_chunk-1)//actual_chunk
        genotype_cap = 8*n + nnz_bound*(5 if n >= 65536 else 3)*2 + frames*16384 + 2*2**20
        metadata_cap = 768*m + 384*n + 2048*len(masks["jobs"]) + 8*2**20
        total_disk += genotype_cap+metadata_cap
        entries.append(dict(name=chromosome, variants=m, scale=m/entry["variants"], position_span=span,
            genotype_storage_limit_bytes=int(genotype_cap), estimated_metadata_bytes=int(metadata_cap),
            estimated_nnz_upper_bound=int(nnz_bound), mask_plan=masks,
            frequency_counts=frequency_counts.tolist(), called=called.tolist(), minor=minor.tolist()))
    if not has_variable_locus:
        raise ValueError("profile and n yield no nonconstant locus for phenotype generation")
    if storage_limit_bytes is not None:
        if type(storage_limit_bytes) is not int or storage_limit_bytes <= 0:
            raise ValueError("storage_limit_bytes must be a positive integer")
        if total_disk > storage_limit_bytes:
            raise MemoryError("synthetic prepared storage estimate exceeds storage_limit_bytes")
    ancestor = Path(root).expanduser().resolve()
    while not ancestor.exists():
        ancestor = ancestor.parent
    if total_disk > shutil.disk_usage(ancestor).free:
        raise OSError("insufficient free disk space for the synthetic prepared estimate")
    return dict(entries=entries, cpu_threads=cpu_threads, chromosome_workers=int(workers),
        frame_chunk_size=int(actual_chunk), estimated_global_sample_bytes=sample_bytes,
        estimated_global_catalog_plan_bytes=int(plan_bytes),
        estimated_per_worker_bytes=int(metadata_work+64*2**20+raw_budget*8),
        estimated_prepared_bytes=int(total_disk), storage_limit_bytes=storage_limit_bytes,
        memory_limit_gib=float(memory_limit_gib), budgets_are_estimates=True)


class _FrequencyStream:
    """Exact bin totals with fixed internal batches, independent of frame size."""
    def __init__(self, counts, allowed, assigned, seed):
        self.remaining = np.asarray(counts, dtype=np.int64)
        self.allowed = np.asarray(allowed, dtype=bool)
        self.assigned_left = int(assigned)
        self.rng = np.random.default_rng(seed)
        self.buffer = np.empty(0, dtype=np.int64)
        self.offset = 0
        if int(self.remaining[self.allowed].sum()) < assigned:
            # Preserve the physical frequency spectrum rather than inventing
            # nonzero MACs. The observed-M audit exposes this incompatibility.
            self.allowed[:] = True

    def take(self, count):
        pieces = []
        while count:
            if self.offset == len(self.buffer):
                mask = self.allowed if self.assigned_left else np.ones(len(self.remaining), dtype=bool)
                number = min(1024, self.assigned_left if self.assigned_left else int(self.remaining.sum()))
                positions = np.flatnonzero(mask)
                chosen = self.rng.multivariate_hypergeometric(self.remaining[positions], number, method="marginals")
                self.remaining[positions] -= chosen
                self.buffer = np.repeat(positions, chosen)
                self.rng.shuffle(self.buffer)
                self.offset = 0
                self.assigned_left -= min(self.assigned_left, number)
            length = min(count, len(self.buffer)-self.offset)
            pieces.append(self.buffer[self.offset:self.offset+length])
            self.offset += length
            count -= length
        return np.concatenate(pieces) if len(pieces) > 1 else pieces[0].copy()


def _variant_exceptions(n, called, minor, rng):
    """Construct MAC/called exactly, retaining complete and half-missing calls."""
    missing_alleles = 2*n-int(called)
    full, half = divmod(missing_alleles, 2)
    missing = rng.choice(n, size=full+half, replace=False) if full+half else np.empty(0, dtype=np.int64)
    excluded = np.sort(missing)
    complete = n-full-half
    haplotypes = rng.choice(2*complete+half, size=int(minor), replace=False) if minor else np.empty(0, dtype=np.int64)
    ordinary = haplotypes[haplotypes < 2*complete]
    ranks, dosage = np.unique(ordinary//2, return_counts=True)
    adjusted = ranks+np.searchsorted(excluded-np.arange(len(excluded)), ranks, side="right")
    arrays = [adjusted, missing[:full]]
    values = [dosage.astype(np.uint8), np.full(full, 3, dtype=np.uint8)]
    if half:
        arrays.append(missing[-1:])
        values.append(np.asarray([5 if np.any(haplotypes == 2*complete) else 4], dtype=np.uint8))
    indices = np.concatenate(arrays)
    states = np.concatenate(values)
    order = np.argsort(indices)
    return indices[order].astype(csr._sample_dtype(n)), states[order]


class _Metadata:
    """Field-at-a-time metadata; no chromosome-sized object/string arrays."""
    def __init__(self, chromosome, m, pairs, segments, seed, span):
        self.chromosome, self.n_variants, self.n_samples = chromosome, m, len(pairs)
        self.pairs, self.segments, self.seed = pairs, segments, seed
        self.span = span
        self.starts = np.asarray([value["start"] for value in segments], dtype=np.int64)
        self.stops = np.asarray([value["stop"] for value in segments], dtype=np.int64)

    def sample_pairs(self):
        return self.pairs

    def read_field(self, path, indices):
        rows = np.asarray(indices, dtype=np.int64)
        if path == "position":
            return _position(rows, self.n_variants, self.span, self.seed, self.chromosome)
        if path == "chromosome":
            return np.full(len(rows), self.chromosome)
        if path == "variant.id":
            return np.asarray([f"SIMV{self.chromosome}_{int(row):012d}" for row in rows], dtype=object)
        if path == "annotation_available":
            return np.ones(len(rows), dtype=bool)
        if path == QC_PATH:
            return np.full(len(rows), "PASS")
        reverse = {value: key for key, value in ANNOTATION_CATALOG.items()}
        name = reverse.get(path, path)
        if name in ANNOTATION_NAMES:
            state = rows.astype(np.uint64)+np.uint64((self.seed+int(self.chromosome)*31+ANNOTATION_NAMES.index(name)*12289) % 2**64)
            state = (state ^ (state >> np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
            state = (state ^ (state >> np.uint64(27)))*np.uint64(0x94d049bb133111eb)
            return .5+(state % np.uint64(34500)).astype(np.float64)/1000
        default = "A,C" if path == "allele" else "intergenic" if name == "GENCODE.Category" else ""
        result = np.full(len(rows), default, dtype=object)
        if not len(self.segments):
            return result
        locations = np.searchsorted(self.stops, rows, side="right")
        for segment in np.unique(locations[locations < len(self.segments)]):
            value = self.segments[int(segment)]
            selected = (locations == segment) & (rows >= value["start"])
            result[selected] = value["annotation"].get(name, value["annotation"].get(path, default))
        return result


def _generate_one(prepared, inputs, source, entry, pairs, n, reference_n, seed, chunk):
    started = time.perf_counter()
    chromosome, m = entry["name"], entry["variants"]
    mask_plan = entry["mask_plan"]
    segments = mask_plan["segments"]
    directory = prepared/("chr"+chromosome)
    binding = dict(synthetic=True, source_format="synthetic-generated", seed=seed,
                   chromosome=chromosome, profile_sha256=hashlib.sha256(store.canonical(source)).hexdigest())
    writer = store.Writer(directory, binding, np.arange(n, dtype=np.int64), m,
        source_bytes=None, storage_limit_bytes=entry["genotype_storage_limit_bytes"],
        chunk=chunk, streaming=True)
    called, minor = np.asarray(entry["called"], dtype=np.int64), np.asarray(entry["minor"], dtype=np.int64)
    target_maf = np.asarray([value["maf"] for value in source["frequency_bins"]])
    allowed = (target_maf > 0) & (target_maf < .01)
    observed_rare = (minor > 0) & (minor < called*.01)
    if n == reference_n and sum(np.asarray(entry["frequency_counts"])[allowed & observed_rare]) >= mask_plan["assigned_variants"]:
        allowed &= observed_rare
    frequencies = _FrequencyStream(entry["frequency_counts"], allowed,
        mask_plan["assigned_variants"], np.random.SeedSequence([seed, int(chromosome), 700]))
    observed_bins = np.zeros(len(called), dtype=np.int64)
    observed_masks = {}
    segment_stops = np.asarray([value["stop"] for value in segments], dtype=np.int64)
    causal, causal_path, causal_priority = None, None, 0
    half_missing_total = missing_total = alt_total = 0
    for start in range(0, m, chunk):
        size = min(chunk, m-start)
        bin_indices = frequencies.take(size)
        offsets = np.zeros(size+1, dtype="<u4")
        index_parts, value_parts = [], []
        for local, bin_index in enumerate(bin_indices):
            ordinal = start+local
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(chromosome), ordinal, 800]))
            indices, states = _variant_exceptions(n, int(called[bin_index]), int(minor[bin_index]), rng)
            index_parts.append(indices)
            value_parts.append(states)
            offsets[local+1] = int(offsets[local])+len(indices)
            observed_bins[bin_index] += 1
            half_missing_total += int((2*n-called[bin_index]) % 2)
            missing_total += int((2*n-called[bin_index])//2)
            alt_total += int(minor[bin_index])
            if ordinal < mask_plan["assigned_variants"] and observed_rare[bin_index]:
                segment = segments[int(np.searchsorted(segment_stops, ordinal, side="right"))]
                for category in segment["memberships"]:
                    key = segment["kind"], segment["group"], category
                    observed_masks[key] = observed_masks.get(key, 0)+1
            priority = 2 if minor[bin_index] >= 20 and minor[bin_index] >= called[bin_index]*.01 else 1
            if priority > causal_priority and 0 < minor[bin_index] < called[bin_index]:
                dosage = np.zeros(n, dtype=np.float64)
                p = minor[bin_index]/max(1, called[bin_index])
                dosage[indices] = np.asarray([0, 1, 2, 2*p, p, 1+p])[states]
                if dosage.std() > 0:
                    causal_path = inputs/("causal_chr"+chromosome+".npy")
                    np.save(causal_path, dosage, allow_pickle=False)
                    reference, alternate = ("A", "AT") if ordinal < mask_plan["assigned_variants"] and segments[int(np.searchsorted(segment_stops, ordinal, side="right"))]["annotation"].get("allele") == "A,AT" else ("A", "C")
                    causal = dict(chromosome=chromosome, index=ordinal, position=int(_position([ordinal], m, entry["position_span"], seed, chromosome)[0]),
                                  reference=reference, alternate=alternate)
                    causal_priority = priority
        indices = np.concatenate(index_parts) if index_parts else np.empty(0, dtype=csr._sample_dtype(n))
        states = np.concatenate(value_parts) if value_parts else np.empty(0, dtype=np.uint8)
        counts = csr.integer_counts(offsets, indices, states, size, n)
        expected_called, expected_minor = called[bin_indices], minor[bin_indices]
        if not (np.array_equal(counts["called_alleles"], expected_called) and
                np.array_equal(counts["called_alleles"]-counts["reference_alleles"], expected_minor)):
            raise ValueError("synthetic exception construction differs from target allele counts")
        writer.append_compact(offsets, indices, states, counts)
    container_manifest = writer.finish()
    metadata = _Metadata(chromosome, m, pairs, segments, seed, entry["position_span"])
    fields = ["position", "chromosome", "variant.id", "allele", QC_PATH,
              "annotation_available", *ANNOTATION_CATALOG.values()]
    coverage = dict(n_variants=m, matched_variants=m, missing_snv=0, missing_non_snv=0,
                    available_field="annotation_available", allow_missing_non_snv=False)
    export_metadata(metadata, directory, directory/"metadata", fields,
        annotation_catalog=ANNOTATION_CATALOG, annotation_names=ANNOTATION_NAMES,
        qc_path=QC_PATH, sample_pairs=pairs, annotation_coverage=coverage)
    catalogs = prepared/"catalogs"
    catalogs.mkdir(exist_ok=True)
    gene_path = catalogs/("chr"+chromosome+".genes.json")
    _json(gene_path, dict(jobs=mask_plan["jobs"], synthetic=True))
    promoter_path = catalogs/("chr"+chromosome+".promoters.json")
    promoters = [[chromosome, int(_position([segment["start"]], m, entry["position_span"], seed, chromosome)[0]),
                  int(_position([segment["stop"]-1], m, entry["position_span"], seed, chromosome)[0])]
                 for segment in segments if segment["annotation"].get("promoter")]
    _json(promoter_path, promoters)
    audit = []
    for target in mask_plan["targets"]:
        actual = observed_masks.get((target["kind"], target["group"], target["category"]), 0)
        audit.append(dict(target, observed_rare_m=actual, matched=actual == target["target_m"]))
    inferred_audit = [dict(target, observed_rare_m=observed_masks.get(
        (target["kind"], target["group"], target["category"]), 0))
        for target in mask_plan["inferred_masks"]]
    realized_maf = np.divide(minor, called, out=np.zeros(len(called), dtype=np.float64), where=called > 0)
    total_variants = max(1, int(observed_bins.sum()))
    maf_error = np.abs(realized_maf-target_maf)
    report = dict(chromosome=chromosome, variants=m, assigned_mask_variants=mask_plan["assigned_variants"],
        annotation_weight_channels=len(ANNOTATION_NAMES), annotation_weights="invented scores; functional database values are not reproduced",
        source_reference_n=reference_n, generated_n=n, gene_jobs=len(mask_plan["jobs"]),
        position_span=entry["position_span"], coordinates="invented stratified coordinates; no original loci retained",
        frequency_counts=observed_bins.tolist(), target_frequency_counts=entry["frequency_counts"],
        frequency_counts_exact=observed_bins.tolist() == entry["frequency_counts"],
        mask_audit=audit, mask_counts_exact=all(value["matched"] for value in audit) and not mask_plan["problems"],
        mask_counts_exact_scope="published target rows on the full generated sample axis only; absent published rows are unknown; trait-specific missingness can change M",
        inferred_mask_audit=inferred_audit,
        additional_inferred_eligible_mask_rows=sum(value["observed_rare_m"] >= 2 for value in inferred_audit),
        incompatible_targets=mask_plan["problems"], source_allele_count_bins=source["frequency_bins"],
        generated_bin_called_alleles=called.tolist(), generated_bin_minor_ac=minor.tolist(),
        generated_bin_maf=realized_maf.tolist(),
        maf_rounding_max_absolute_error=float(maf_error[observed_bins > 0].max(initial=0)),
        maf_rounding_weighted_mean_absolute_error=float(np.dot(observed_bins, maf_error)/total_variants),
        physical_rare_variants=int(observed_bins[observed_rare].sum()),
        monomorphic_variants=int(observed_bins[minor == 0].sum()),
        full_missing_calls=missing_total, half_missing_calls=half_missing_total,
        alternate_alleles=alt_total, causal_variant=causal,
        causal_dosage_path=str(causal_path) if causal_path else None,
        storage_bytes=store.usage(directory)["logical_bytes"],
        storage_limit_bytes=entry["genotype_storage_limit_bytes"], seconds=time.perf_counter()-started)
    result_entry = dict(name=chromosome, container_directory="chr"+chromosome,
        metadata_directory="chr"+chromosome+"/metadata",
        gene_catalog=str(gene_path.relative_to(prepared)), promoter_intervals=str(promoter_path.relative_to(prepared)),
        source_binding=binding, genotype_manifest_sha256=hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest())
    return result_entry, report


def build_simulated_prepared(root, profile, n, chromosomes, seed=1729, cpu_threads=8,
                             memory_limit_gib=8, chunk_size=256, storage_limit_bytes=None,
                             variant_scale=1.0, variants_per_chromosome=None):
    """Create only inputs/ and prepared/; no retained raw genotype source."""
    started = time.perf_counter()
    profile = normalize_profile(profile)
    plan = plan_simulated_prepared(root, profile, n, chromosomes, seed, cpu_threads,
        memory_limit_gib, chunk_size, storage_limit_bytes, variant_scale, variants_per_chromosome)
    root = Path(root).expanduser().resolve()
    prepared, inputs = root/"prepared", root/"inputs"
    if prepared.exists():
        raise FileExistsError("synthetic prepared data already exists")
    prepared.mkdir(parents=True, mode=0o700)
    inputs.mkdir(exist_ok=True, mode=0o700)
    width = max(9, len(str(n-1)))
    pairs = np.asarray([[f"SIMF{index:0{width}d}", f"SIMI{index:0{width}d}"] for index in range(n)])
    pair_path = prepared/"sample_pairs.npy"
    np.save(pair_path, pairs, allow_pickle=False)
    np.save(prepared/"sample_ids.npy", sample_keys(pairs), allow_pickle=False)
    source_entries = {entry["name"]: entry for entry in profile["chromosomes"]}
    def generate(entry):
        return _generate_one(prepared, inputs, source_entries[entry["name"]], entry, pairs, n,
                             profile["reference_n"], seed, plan["frame_chunk_size"])
    if plan["chromosome_workers"] > 1:
        with ThreadPoolExecutor(max_workers=plan["chromosome_workers"]) as pool:
            results = list(pool.map(generate, plan["entries"]))
    else:
        results = [generate(entry) for entry in plan["entries"]]
    entries, reports = [value[0] for value in results], [value[1] for value in results]
    dataset = dict(schema_version=2, sample_identifier_format="fid_iid_json",
        sample_pairs="sample_pairs.npy", sample_ids="sample_ids.npy", chromosomes=entries,
        annotation_catalog=ANNOTATION_CATALOG, annotation_names=ANNOTATION_NAMES, qc_path=QC_PATH,
        preparation=dict(source_format="synthetic-generated", synthetic=True, completed=True,
            original_genotype_source_retained=False, cpu_threads=cpu_threads,
            memory_limit_gib=memory_limit_gib, seconds=time.perf_counter()-started))
    raw = store.canonical(dataset)
    store.atomic(prepared/"dataset.json", raw)
    store.atomic(prepared/"COMPLETE", hashlib.sha256(raw).hexdigest().encode())
    causal_report = next((report for report in reports if report["causal_dosage_path"]), None)
    if causal_report is None:
        raise ValueError("generated profile has no nonconstant locus for phenotype generation")
    public_plan = {key: value for key, value in plan.items() if key != "entries"}
    return dict(completed=True, prepared_directory=str(prepared), sample_pairs_path=str(pair_path),
        causal_dosage_path=causal_report["causal_dosage_path"], causal_variant=causal_report["causal_variant"],
        chromosome_reports=reports, mask_counts_exact=all(report["mask_counts_exact"] for report in reports),
        mask_counts_exact_scope="published target rows on the full generated sample axis only; absent published rows are unknown; trait-specific missingness can change M",
        additional_inferred_eligible_mask_rows=sum(report["additional_inferred_eligible_mask_rows"] for report in reports),
        frequency_counts_exact=all(report["frequency_counts_exact"] for report in reports),
        resource_plan=public_plan, seconds=time.perf_counter()-started,
        frequency_policy="preserve profile MAF; integer MAC/called rounding changes low-count eligibility when n changes",
        mask_policy="published marginal M and feasible within-coding overlaps are calibrated; absent published masks are inferred unknowns and may emit additional rows; noncoding atoms are disjoint; biological LD and original variant intersections are not reproduced")
