"""Prepare and read phenotype-wise, cache-aligned, numeric PheWAS inputs.

CSV files are parsed once each.  Missingness is applied per phenotype and its
declared covariate profile.  This module preserves the supplied cache sample
axis and makes no ancestry selection or phenotype transformation.  All outputs
contain private participant/phenotype metadata and belong in a private folder.
"""
from __future__ import annotations

import collections
from collections.abc import Mapping
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
        _discard(stream.fileno())
    return digest.hexdigest()


def _discard(fd):
    if hasattr(os, "posix_fadvise"):
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def _stat(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns]


def _binding(path, expected=None):
    before = _stat(path)
    digest = sha256(path)
    if before != _stat(path) or (expected is not None and digest != expected):
        raise ValueError("Input SHA or file identity does not match")
    return {"path": str(Path(path).resolve()), "stat": before, "sha256": digest}


def _unchanged(binding):
    path = binding["path"]
    if _stat(path) != binding["stat"] or sha256(path) != binding["sha256"]:
        raise ValueError("An input changed during preparation")


def _json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".partial")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.chmod(0o600)
    os.replace(pending, path)


def _header(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        value = next(csv.reader(stream), None)
    if not value or any(not name for name in value) or len(value) != len(set(value)):
        raise ValueError("CSV requires a unique nonempty header")
    if ("FID" in value or "IID" in value) and value[:2] != ["FID", "IID"]:
        raise ValueError("Numeric-axis CSV identity columns must start with FID,IID")
    return value


def _identity_width(header):
    """New CSVs use FID/IID; old numeric single-ID exports stay readable."""
    return 2 if header[:2] == ["FID", "IID"] else 1


def _ids(values):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iuUS":
        raise ValueError("Sample identifiers must be a one-dimensional integer/string array")
    strings = values.astype(str)
    if not np.all(np.char.isdigit(strings)):
        raise ValueError("Sample identifiers must be canonical positive integer IDs")
    try:
        result = strings.astype(np.int64)
    except (ValueError, OverflowError) as error:
        raise ValueError("Sample identifiers exceed the supported integer range") from error
    if not np.all(result > 0) or not np.array_equal(result.astype(str), strings):
        raise ValueError("Sample identifiers must be canonical positive integer IDs")
    return result


def _lookup(cache_ids, values, index=None):
    """Return chunk row and cache-axis indices without changing either order."""
    if index is None:
        order = np.argsort(cache_ids)
        sorted_ids = cache_ids[order]
    else:
        order, sorted_ids = index
    index = np.searchsorted(sorted_ids, values)
    inside = index < len(sorted_ids)
    inside[inside] &= sorted_ids[index[inside]] == values[inside]
    rows = np.flatnonzero(inside)
    return rows, order[index[inside]]


def _source_manifest(path, output, digest_key):
    if path is None:
        return None, None
    value = json.loads(Path(path).read_text())
    if value.get("status") != "complete" or value.get("limited_probe") is True:
        raise ValueError("Source CSV manifest is incomplete or a limited probe")
    if Path(value["output"]).resolve() != Path(output).resolve():
        raise ValueError("Source manifest is bound to another CSV")
    return value, value[digest_key]


def _metadata(profiles, names, covariate_names, dictionary=None, *,
              phenotype_families="auto", phenotype_covariates=None):
    """Resolve only caller-declared trait options; omitted entries stay generic."""
    if phenotype_families == "auto" or phenotype_families is None:
        families = {}
    elif isinstance(phenotype_families, Mapping):
        families = dict(phenotype_families)
    else:
        raise ValueError("phenotype_families must be auto or a trait-to-family mapping")
    if phenotype_covariates is None:
        selected_covariates = {}
    elif isinstance(phenotype_covariates, Mapping):
        selected_covariates = dict(phenotype_covariates)
    else:
        raise ValueError("phenotype_covariates must be a trait-to-column-list mapping")
    if (set(families) | set(selected_covariates)) - set(names):
        raise ValueError("trait options name a column absent from the phenotype CSV")
    for name, family in families.items():
        if family not in ("auto", "gaussian", "binomial", "binary"):
            raise ValueError("phenotype families must be auto, gaussian or binomial")
        families[name] = None if family == "auto" else ("binomial" if family == "binary" else family)
    for name, columns in selected_covariates.items():
        if (not isinstance(columns, (list, tuple)) or any(not isinstance(c, str) for c in columns)
                or len(columns) != len(set(columns)) or not set(columns) <= set(covariate_names)):
            raise ValueError("selected covariates must be distinct column names from the covariate CSV")
        selected_covariates[name] = list(columns)
    if profiles is not None and (families or selected_covariates):
        raise ValueError("explicit profile files cannot be combined with trait option mappings")
    if profiles is None:
        records = [{"phenotype": name, "model_family": families.get(name),
                    "profile": "explicit_trait_covariates" if name in selected_covariates else "all_covariates_generic",
                    "covariate_columns": ";".join(selected_covariates.get(name, covariate_names))} for name in names]
    else:
        with Path(profiles).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["phenotype", "model_family", "profile", "covariate_columns"]:
                raise ValueError("Unsupported covariate-profile schema")
            records = list(reader)
    if len(records) != len(names) or {r["phenotype"] for r in records} != set(names):
        raise ValueError("Every phenotype requires one covariate profile")
    by_name = {r["phenotype"]: r for r in records}
    if len(by_name) != len(records):
        raise ValueError("Duplicate phenotype-profile record")
    groups, traits = {}, []
    for index, name in enumerate(names):
        row = by_name[name]
        if row["model_family"] not in ("gaussian", "binomial") and profiles is not None:
            raise ValueError("Unsupported phenotype model family")
        columns = row["covariate_columns"].split(";") if row["covariate_columns"] else []
        if ((not columns and profiles is not None) or len(columns) != len(set(columns))
                or not set(columns) <= set(covariate_names)):
            raise ValueError("Invalid selected covariate columns")
        indices = [covariate_names.index(column) for column in columns]
        key = tuple(indices)
        if key not in groups:
            groups[key] = len(groups)
        traits.append({"index": index, "name": name, "family": row["model_family"],
                       "profile": row["profile"], "profile_index": groups[key],
                       "covariate_indices": indices})
    dictionary_rows = None
    if dictionary is not None:
        with Path(dictionary).open(newline="", encoding="utf-8-sig") as stream:
            dictionary_rows = list(csv.DictReader(stream))
        if len(dictionary_rows) != len(traits):
            raise ValueError("Phenotype dictionary coverage differs")
        for index, row in enumerate(dictionary_rows):
            if (row.get("phenotype") != names[index] or int(row["column_index"]) != index + 2
                    or (profiles is not None and row.get("model_family") != traits[index]["family"])):
                raise ValueError("Phenotype dictionary order/family differs")
    return traits, [list(key) for key in groups], dictionary_rows


def _parse(path, cache_ids, target, *, families=None, infer_families=False,
           chunk_rows=8192, emit=None, phase=None, rm_nan=False, binary_only=False):
    import pandas as pd  # Existing project dependency; only needed for preparation.
    header = _header(path)
    identity_width = _identity_width(header)
    dtypes = {name: np.float64 for name in header[identity_width:]}
    dtypes.update({name: str for name in header[:identity_width]})
    present = np.zeros(len(cache_ids), dtype=bool)
    seen = set()
    lookup_order = np.argsort(cache_ids)
    lookup_index = lookup_order, cache_ids[lookup_order]
    rows = 0
    binary_indices = [] if families is None else [i for i, f in enumerate(families) if f == "binomial"]
    any_nonmissing = np.zeros(target.shape[1], dtype=bool)
    binary_compatible = np.ones(target.shape[1], dtype=bool)
    # Keep pandas' round-trip conversion rather than its faster approximate
    # float parser: the exported CSV is the actual model input.
    with Path(path).open("rb") as stream:
        reader = pd.read_csv(stream, dtype=dtypes, chunksize=chunk_rows,
                             float_precision="round_trip", encoding="utf-8-sig")
        for chunk in reader:
            if identity_width == 2 and not np.all(chunk["FID"].to_numpy(dtype=str) == "0"):
                raise ValueError("Numeric-axis caches require FID='0'; family identities cannot be discarded")
            chunk_ids = _ids(chunk[header[identity_width - 1]].to_numpy(dtype=str))
            if len(np.unique(chunk_ids)) != len(chunk_ids) or any(int(v) in seen for v in chunk_ids):
                raise ValueError("CSV sample IDs must be globally unique")
            seen.update(map(int, chunk_ids))
            numeric = chunk.iloc[:, identity_width:].to_numpy(dtype=np.float64)
            if numeric.shape != (len(chunk_ids), target.shape[1]) or np.any(np.isinf(numeric)):
                raise ValueError("Invalid numeric CSV shape or infinite value")
            if infer_families:
                finite = np.isfinite(numeric)
                any_nonmissing |= finite.any(axis=0)
                binary_compatible &= ~(finite & (numeric != 0) & (numeric != 1)).any(axis=0)
            if binary_indices and (binary_only or not rm_nan):
                # A pure disease table already has precisely these columns.
                # Avoid a second chunk-sized float64 panel at large trait counts.
                binary = numeric if len(binary_indices) == numeric.shape[1] else numeric[:, binary_indices]
                if np.any(np.isfinite(binary) & (binary != 0) & (binary != 1)):
                    raise ValueError("Binary phenotype values must be zero, one or missing")
            positions, indices = _lookup(cache_ids, chunk_ids, lookup_index)
            target[indices, :] = numeric[positions, :]
            present[indices] = True
            rows += len(chunk_ids)
            if emit is not None:
                emit({"phase": phase, "source_rows_processed": rows,
                      "cache_rows_present": int(present.sum())})
        _discard(stream.fileno())
    target.flush()
    parsed = {"source_rows": rows, "source_unique_ids": len(seen),
              "cache_rows_present": int(present.sum()),
              "cache_rows_absent": int((~present).sum())}
    if infer_families:
        parsed["inferred_families"] = ["binomial" if has and binary else "gaussian"
                                       for has, binary in zip(any_nonmissing, binary_compatible)]
        parsed["entirely_missing_source_traits"] = int((~any_nonmissing).sum())
    return parsed, present


def _release(array):
    array.flush()
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _missing_policy(rm_nan, binary_only=False):
    return {"schema_version": 2 if binary_only else 1,
            "sample_selection": "phenotype_present_and_selected_covariates_valid" if rm_nan
                                else "finite_phenotype_and_selected_covariates_valid",
            "gaussian": "not_accepted" if binary_only else
                        "eligible_finite_mean_fill_nan" if rm_nan else "finite_available_samples",
            "binomial": ("finite_exact_zero_or_one_and_nan_zero" if binary_only else
                         "one_is_case_all_other_finite_or_nan_are_controls") if rm_nan
                        else "finite_exact_zero_or_one",
            "covariates": "selected_columns_complete_no_imputation",
            "infinite_values": "rejected"}


def apply_phenotype_missing_policy(values, *, family, rm_nan=True, covariate_valid,
                                   binary_only=False):
    """Select one trait's rows and apply its explicit missing-value policy.

    ``covariate_valid`` identifies the eligible analysis population, including
    any caller-selected cohort and source-row alignment. Returned ``values``
    is a new float64 vector in the order of ``sample_indices``. Inputs are never
    changed. The Gaussian mean uses finite phenotypes in that eligible
    population, so filling cannot borrow a different cohort's phenotype mean.
    ``available_n`` always counts eligible finite phenotypes before filling.
    Infinite values are invalid data under both policies, rather than NaNs.
    """
    if type(rm_nan) is not bool:
        raise ValueError("rm_nan must be a boolean")
    if type(binary_only) is not bool:
        raise ValueError("binary_only must be a boolean")
    if family == "binary":
        family = "binomial"
    if family not in ("gaussian", "binomial"):
        raise ValueError("phenotype family must be gaussian or binomial")
    if binary_only and family != "binomial":
        raise ValueError("Binary-only analysis accepts only binomial phenotypes")
    values = np.asarray(values, dtype=np.float64)
    eligible = np.asarray(covariate_valid)
    if values.ndim != 1 or eligible.shape != values.shape or eligible.dtype != np.bool_:
        raise ValueError("phenotype values and boolean eligible-sample mask must be aligned one-dimensional arrays")
    if np.any(np.isinf(values)):
        raise ValueError("Infinite phenotype values are invalid under both missing-value policies")
    finite = np.isfinite(values)
    available = eligible & finite
    available_n, mother_n = int(available.sum()), int(eligible.sum())
    if family == "binomial" and (binary_only or not rm_nan) and np.any(finite & (values != 0) & (values != 1)):
        raise ValueError("Binary phenotype values must be zero, one or missing" +
                         (" in binary-only analysis" if binary_only else " when rm_nan=False"))
    selected = eligible if rm_nan else available
    indices = np.flatnonzero(selected)
    mean = None
    if rm_nan and family == "gaussian" and mother_n:
        if not available_n:
            raise ValueError("Gaussian missing phenotypes cannot be filled without a finite eligible phenotype mean")
        try:
            with np.errstate(over="raise", invalid="raise"):
                mean = float(np.mean(values[available], dtype=np.float64))
        except FloatingPointError as error:
            raise ValueError("The eligible Gaussian phenotype mean is not finite") from error
        if not np.isfinite(mean):
            raise ValueError("The eligible Gaussian phenotype mean is not finite")
    result = np.array(values[indices], dtype=np.float64, copy=True)
    missing = np.isnan(result)
    if rm_nan:
        if family == "binomial":
            if binary_only:
                result[missing] = 0
            else:
                result = (result == 1).astype(np.float64)
        elif mean is not None:
            result[missing] = mean
    cases = int((result == 1).sum()) if family == "binomial" else None
    controls = int((result == 0).sum()) if family == "binomial" else None
    if binary_only and (cases == 0 or controls == 0):
        raise ValueError("Every binary phenotype requires at least one analysis case and one control")
    return {"values": result, "sample_indices": indices, "available_n": available_n,
            "mother_n": mother_n, "imputed_n": int(missing.sum()) if rm_nan else 0,
            "recoded_nonbinary_n": int((finite & eligible & (values != 0) & (values != 1)).sum())
                                   if rm_nan and family == "binomial" else 0,
            "imputation_mean": mean, "rm_nan": rm_nan, "binary_only": binary_only,
            "analysis_cases": cases, "analysis_controls": controls,
            "available_cases": int((available & (values == 1)).sum()) if family == "binomial" else None,
            "available_controls": int((available & (values == 0)).sum()) if family == "binomial" else None,
            "imputed_controls": int(missing.sum()) if family == "binomial" and rm_nan else 0,
            "missing_policy": _missing_policy(rm_nan, binary_only)}


def prepare_phewas_inputs(*, phenotypes, covariates, sample_ids, output, profiles=None,
                         phenotype_dictionary=None, phenotype_manifest=None,
                         covariate_manifest=None, chunk_rows=8192, emit=None,
                         phenotype_families="auto", phenotype_covariates=None, rm_nan=True,
                         binary_only=False, input_contract="mixed"):
    """Create immutable raw float64 arrays in cache sample order.

    ``output`` must not exist.  No intercept, RINT, residualization, ancestry
    filter, null model or association result is computed.  Full-file hashes
    bracket one numeric CSV parse per input; input/stat/source hashes are
    recorded privately. ``trait`` applies each profile's missingness later:
    ``rm_nan=True`` fills Gaussian NaNs with the eligible finite mean and
    binomial values other than one, including NaNs, with zero. ``False`` uses
    finite available samples. Covariates are never filled; each trait only
    checks its selected columns. Raw phenotype arrays remain unmodified.
    With ``profiles=None``, all covariate columns are used for every trait and
    family is inferred from all rows in the single phenotype CSV parse: a
    nonempty column containing only exact 0/1 is binomial, all other columns
    (including entirely missing columns) are Gaussian. This generic profile
    does not reproduce a paper-specific covariate protocol.

    ``binary_only=True`` requires at least two columns, validates exact 0/1
    across every source row, and requires both cases and controls in each
    eligible trait. Only NaNs may become controls; participants absent from
    the phenotype CSV are never created as controls. The legacy generic
    missing-value policy remains available to internal callers.

    ``input_contract="continuous"`` requires at least two inferred Gaussian
    traits and one shared profile containing every covariate. It cannot be
    combined with ``binary_only``. The contract is bound to prepared metadata
    so reopening an input never changes its accepted phenotype types.
    """
    if type(rm_nan) is not bool:
        raise ValueError("rm_nan must be a boolean")
    if type(binary_only) is not bool:
        raise ValueError("binary_only must be a boolean")
    if input_contract not in ("mixed", "continuous"):
        raise ValueError("unknown phenotype input contract")
    if binary_only and input_contract != "mixed":
        raise ValueError("binary_only cannot be combined with a continuous input contract")
    if input_contract == "continuous" and (profiles is not None or phenotype_families != "auto"
                                            or phenotype_covariates is not None):
        raise ValueError("continuous analysis requires inferred families and shared all covariates")
    if not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    started = time.perf_counter()
    output = Path(output)
    if output.exists():
        raise FileExistsError("Prepared input directory already exists")
    axis = _ids(np.load(sample_ids, allow_pickle=False))
    if len(axis) == 0 or len(np.unique(axis)) != len(axis):
        raise ValueError("Cache sample axis is empty or contains duplicates")
    phenotype_header, covariate_header = _header(phenotypes), _header(covariates)
    phenotype_names = phenotype_header[_identity_width(phenotype_header):]
    covariate_names = covariate_header[_identity_width(covariate_header):]
    if not phenotype_names:
        raise ValueError("The phenotype CSV must have at least one numeric column")
    if binary_only and len(phenotype_names) < 2:
        raise ValueError("Binary-only analysis requires at least two phenotype columns")
    if input_contract == "continuous" and len(phenotype_names) < 2:
        raise ValueError("continuous analysis requires at least two phenotype columns")
    traits, profile_indices, dictionary = _metadata(profiles, phenotype_names, covariate_names,
        phenotype_dictionary, phenotype_families=phenotype_families,
        phenotype_covariates=phenotype_covariates)
    if binary_only:
        for trait in traits:
            if trait["family"] not in (None, "binomial"):
                raise ValueError("Binary-only analysis accepts only binomial phenotypes")
            trait["family"] = "binomial"
    pm, expected_p = _source_manifest(phenotype_manifest, phenotypes, "output_sha256")
    cm, expected_c = _source_manifest(covariate_manifest, covariates, "covariate_output_sha256")
    sources = {"phenotypes": _binding(phenotypes, expected_p),
               "covariates": _binding(covariates, expected_c),
               "sample_ids": _binding(sample_ids)}
    if profiles is not None:
        sources["profiles"] = _binding(profiles)
    for name, path in (("phenotype_dictionary", phenotype_dictionary),
                       ("phenotype_manifest", phenotype_manifest),
                       ("covariate_manifest", covariate_manifest)):
        if path is not None:
            sources[name] = _binding(path)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    output.chmod(0o700)
    _json(output / "status.anonymous.json", {"status": "preparing"})
    y = x = None
    try:
        y = np.lib.format.open_memmap(output / "phenotypes.npy", mode="w+", dtype=np.float64,
                                      shape=(len(axis), len(traits)))
        x = np.lib.format.open_memmap(output / "covariates.npy", mode="w+", dtype=np.float64,
                                      shape=(len(axis), len(covariate_names)))
        y[:] = np.nan
        x[:] = np.nan
        infer_families = any(trait["family"] is None for trait in traits)
        p_parse, p_present = _parse(phenotypes, axis, y, families=[r["family"] for r in traits],
                                   infer_families=infer_families,
                                   chunk_rows=chunk_rows, emit=emit, phase="phenotypes", rm_nan=rm_nan,
                                   binary_only=binary_only)
        if infer_families:
            for trait, family in zip(traits, p_parse.pop("inferred_families")):
                if trait["family"] is None:
                    trait["family"] = family
        if input_contract == "continuous":
            from .continuous_run import validate_continuous_values
            validate_continuous_values(phenotype_names, y)
            if any(trait["family"] != "gaussian" for trait in traits):
                raise ValueError("continuous analysis accepts Gaussian phenotypes only")
        if profiles is None and dictionary is not None and any(row.get("model_family") != trait["family"]
                                          for row, trait in zip(dictionary, traits)):
            raise ValueError("Phenotype dictionary family differs from the assigned numeric family")
        c_parse, c_present = _parse(covariates, axis, x, chunk_rows=chunk_rows,
                                   emit=emit, phase="covariates")
        for parsed, original in ((p_parse, pm), (c_parse, cm)):
            if original is not None and parsed["source_rows"] != original["participant_rows"]:
                raise ValueError("Parsed CSV row count differs from source manifest")
        complete = np.stack([np.isfinite(x[:, indices]).all(axis=1) for indices in profile_indices])
        trait_counts = []
        for trait in traits:
            values = y[:, trait["index"]]
            finite = np.isfinite(values)
            eligible = complete[trait["profile_index"]]
            if rm_nan:
                eligible = eligible & p_present
            policy = apply_phenotype_missing_policy(values, family=trait["family"],
                rm_nan=rm_nan, covariate_valid=eligible, binary_only=binary_only)
            analyzed = policy["values"]
            eligible_finite = values[eligible & finite]
            quality = {"index": trait["index"], "family": trait["family"],
                       "phenotype_nonmissing": int(finite.sum()),
                       "phenotype_missing": int((~finite).sum()),
                       "profile_complete_samples": int(complete[trait["profile_index"]].sum()),
                       "analysis_complete_samples": len(analyzed),
                       "available_n": policy["available_n"], "mother_n": policy["mother_n"],
                       "imputed_n": policy["imputed_n"], "imputation_mean": policy["imputation_mean"],
                       "recoded_nonbinary_n": policy["recoded_nonbinary_n"],
                       "minimum": float(values[finite].min()) if finite.any() else None,
                       "maximum": float(values[finite].max()) if finite.any() else None,
                       "eligible_minimum": float(eligible_finite.min()) if eligible_finite.size else None,
                       "eligible_maximum": float(eligible_finite.max()) if eligible_finite.size else None,
                       "analysis_cases": policy["analysis_cases"],
                       "analysis_controls": policy["analysis_controls"],
                       "available_cases": policy["available_cases"],
                       "available_controls": policy["available_controls"],
                       "imputed_controls": policy["imputed_controls"]}
            trait_counts.append(quality)
        np.save(output / "sample_ids.npy", axis, allow_pickle=False)
        np.save(output / "profile_complete.npy", complete, allow_pickle=False)
        np.save(output / "phenotype_source_present.npy", p_present, allow_pickle=False)
        np.save(output / "covariate_source_present.npy", c_present, allow_pickle=False)
        _release(y); y = None
        _release(x); x = None
        for binding in sources.values():
            _unchanged(binding)
        metadata = {"traits": traits, "covariate_names": covariate_names,
                    "rm_nan": rm_nan, "binary_only": binary_only, "input_contract": input_contract,
                    "missing_policy": _missing_policy(rm_nan, binary_only),
                    "profile_covariate_indices": profile_indices,
                    "phenotype_dictionary": dictionary, "quality": trait_counts,
                    "profile_scope": ("explicit_profiles" if profiles is not None else
                        "explicit_trait_covariates" if phenotype_covariates else "all_covariates_generic"),
                    "family_assignment": ("strict_binary_only" if binary_only else
                        "declared_by_profile" if profiles is not None else
                        "auto_with_explicit_overrides" if isinstance(phenotype_families, Mapping) and phenotype_families else
                        "inferred_exact_01_from_all_source_rows")}
        if input_contract == "continuous":
            from .continuous_run import validate_continuous_metadata
            validate_continuous_metadata(metadata)
        _json(output / "traits.private.json", metadata)
        artifacts = {}
        for path in sorted(output.glob("*.npy")):
            path.chmod(0o600)
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        artifacts["traits.private.json"] = {"bytes": (output / "traits.private.json").stat().st_size,
                                            "sha256": sha256(output / "traits.private.json")}
        families = collections.Counter(r["family"] for r in traits)
        family_counts = {}
        for family in sorted(families):
            rows = [q for q in trait_counts if q["family"] == family]
            family_counts[family] = {"traits": families[family],
                "min_analysis_complete_samples": min(q["analysis_complete_samples"] for q in rows),
                "max_analysis_complete_samples": max(q["analysis_complete_samples"] for q in rows),
                "sum_analysis_complete_samples": sum(q["analysis_complete_samples"] for q in rows),
                "min_available_samples": min(q["available_n"] for q in rows),
                "max_available_samples": max(q["available_n"] for q in rows),
                "sum_imputed_samples": sum(q["imputed_n"] for q in rows),
                "empty_traits": sum(q["analysis_complete_samples"] == 0 for q in rows)}
        anonymous = {"status": "complete", "cache_sample_count": len(axis), "phenotype_count": len(traits),
                     "rm_nan": rm_nan, "binary_only": binary_only, "input_contract": input_contract,
                     "missing_policy": _missing_policy(rm_nan, binary_only),
                     "covariate_count": len(covariate_names), "unique_covariate_profiles": len(profile_indices),
                     "families": family_counts, "phenotype_parse": p_parse, "covariate_parse": c_parse,
                     "cache_axis_preserved": True, "ancestry_selection_applied": False,
                     "covariate_profile_scope": metadata["profile_scope"],
                     "family_assignment": metadata["family_assignment"],
                     "paper_specific_covariate_profile_applied": False if profiles is None else None,
                     "joint_all_phenotype_complete_case_selection_applied": False,
                     "intercept_added": False, "phenotype_transformations_applied": False,
                     "null_models_fitted": False, "numeric_csv_parses_per_input": 1,
                     "all_source_bindings_unchanged": True, "all_binary_values_checked": True,
                     "elapsed_seconds": time.perf_counter() - started,
                     "array_bytes": sum(v["bytes"] for k, v in artifacts.items() if k.endswith(".npy"))}
        manifest = {"schema": 1, "status": "complete", "sample_count": len(axis), "trait_count": len(traits),
                    "rm_nan": rm_nan, "binary_only": binary_only, "input_contract": input_contract,
                    "missing_policy": _missing_policy(rm_nan, binary_only),
                    "covariate_count": len(covariate_names), "profile_count": len(profile_indices),
                    "array_dtype": "float64", "array_order": "C", "sample_order": "input_cache_axis",
                    "cohort_scope": "all_supplied_cache_samples; analysis cohort not locked",
                    "sources": sources, "artifacts": artifacts, "anonymous_summary": anonymous}
        if profiles is None:
            manifest["trait_options"] = {
                "rm_nan": rm_nan,
                "binary_only": binary_only,
                "input_contract": input_contract,
                "phenotype_families": dict(phenotype_families) if isinstance(phenotype_families, Mapping) else "auto",
                "phenotype_covariates": {name: list(columns) for name, columns in (phenotype_covariates or {}).items()}}
        _json(output / "manifest.private.json", manifest)
        _json(output / "summary.anonymous.json", anonymous)
        _json(output / "status.anonymous.json", {"status": "complete",
                                                "manifest_sha256": sha256(output / "manifest.private.json")})
        return anonymous
    except BaseException as error:
        _json(output / "status.anonymous.json", {"status": "failed", "error_type": type(error).__name__})
        raise
    finally:
        if y is not None:
            _release(y)
        if x is not None:
            _release(x)


class PhewasInputs:
    """Read immutable raw input mmaps; ``trait`` returns X without intercept.

    The caller fits and transforms each phenotype independently.  Covariate
    columns may be constant in a sex-specific or very small selected sample;
    fitting code should add its intercept then record/drop rank-redundant
    columns with a rank-revealing factorization, rather than alter this input.
    """
    def __init__(self, directory, *, verify_arrays=True, design_cache_limit_bytes=128 << 20):
        if type(design_cache_limit_bytes) is not int or design_cache_limit_bytes < 0:
            raise ValueError("design_cache_limit_bytes must be a nonnegative integer")
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.private.json").read_text())
        status = json.loads((self.directory / "status.anonymous.json").read_text())
        if (self.manifest.get("status") != "complete" or status.get("status") != "complete"
                or status.get("manifest_sha256") != sha256(self.directory / "manifest.private.json")):
            raise ValueError("Prepared input completion binding is invalid")
        for name, binding in self.manifest["artifacts"].items():
            path = self.directory / name
            if path.name != name or path.stat().st_size != binding["bytes"]:
                raise ValueError("Prepared input artifact identity is invalid")
            if verify_arrays and sha256(path) != binding["sha256"]:
                raise ValueError("Prepared input artifact SHA is invalid")
        self.metadata = json.loads((self.directory / "traits.private.json").read_text())
        # Pre-policy manifests retain their original available-sample behavior.
        # A default at the public API must never reinterpret an old fitted input.
        self.rm_nan = self.manifest.get("rm_nan", False)
        self.binary_only = self.manifest.get("binary_only", False)
        trait_options = self.manifest.get("trait_options", {})
        self.input_contract = self.manifest.get("input_contract", trait_options.get("input_contract", "mixed"))
        # The first continuous-only release stored its contract in trait_options
        # alone. New manifests also bind metadata directly; old mixed inputs
        # retain their original default rather than being reinterpreted.
        metadata_default = (trait_options.get("input_contract", "mixed")
                            if "input_contract" not in self.manifest else "mixed")
        if (self.input_contract not in ("mixed", "continuous")
                or self.metadata.get("input_contract", metadata_default) != self.input_contract
                or trait_options.get("input_contract", "mixed") != self.input_contract
                or (self.binary_only and self.input_contract != "mixed")):
            raise ValueError("Prepared phenotype input contract binding is invalid")
        if self.input_contract == "continuous":
            from .continuous_run import validate_continuous_metadata
            validate_continuous_metadata(self.metadata)
        if (type(self.binary_only) is not bool
                or self.metadata.get("binary_only", False) != self.binary_only
                or ("trait_options" in self.manifest and
                    self.manifest["trait_options"].get("binary_only", False) != self.binary_only)
                or (self.binary_only and (self.manifest["trait_count"] < 2 or
                    any(trait.get("family") != "binomial" for trait in self.metadata["traits"])) )):
            raise ValueError("Prepared binary-only input binding is invalid")
        if (type(self.rm_nan) is not bool
                or self.metadata.get("rm_nan", False) != self.rm_nan
                or ("trait_options" in self.manifest and
                    self.manifest["trait_options"].get("rm_nan", False) != self.rm_nan)
                or ((self.rm_nan or "missing_policy" in self.manifest or "missing_policy" in self.metadata)
                    and (self.manifest.get("missing_policy") != _missing_policy(self.rm_nan, self.binary_only)
                         or self.metadata.get("missing_policy") != _missing_policy(self.rm_nan, self.binary_only)))):
            raise ValueError("Prepared phenotype missing-value policy binding is invalid")
        self.sample_ids = np.load(self.directory / "sample_ids.npy", mmap_mode="r", allow_pickle=False)
        self.phenotypes = np.load(self.directory / "phenotypes.npy", mmap_mode="r", allow_pickle=False)
        self.covariates = np.load(self.directory / "covariates.npy", mmap_mode="r", allow_pickle=False)
        self.profile_complete = np.load(self.directory / "profile_complete.npy", mmap_mode="r", allow_pickle=False)
        self.phenotype_source_present = (np.load(self.directory / "phenotype_source_present.npy",
            mmap_mode="r", allow_pickle=False) if self.rm_nan else None)
        n, t, c, p = (self.manifest[key] for key in ("sample_count", "trait_count", "covariate_count", "profile_count"))
        if (self.phenotypes.shape != (n, t) or self.covariates.shape != (n, c)
                or self.profile_complete.shape != (p, n) or self.sample_ids.shape != (n,)
                or self.phenotypes.dtype != np.float64 or self.covariates.dtype != np.float64
                or self.profile_complete.dtype != np.bool_ or len(self.metadata["traits"]) != t
                or (self.rm_nan and (self.phenotype_source_present.shape != (n,)
                                     or self.phenotype_source_present.dtype != np.bool_))):
            raise ValueError("Prepared input array schema is invalid")
        self._design_cache = collections.OrderedDict()
        self._design_cache_bytes = 0
        self._design_cache_limit_bytes = design_cache_limit_bytes
        self._design_cache_hits = self._design_cache_misses = self._design_cache_evictions = 0

    @staticmethod
    def _immutable_array(value):
        """A bytes-backed array cannot be made writable by a downstream fit."""
        value = np.asarray(value)
        return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)

    def _shared_design(self, indices, trait):
        """Share only identical ordered samples and covariate columns, bounded.

        Phenotype values and fitted models remain independent. Different NaN
        patterns reuse the raw covariate mmap but never a mismatched design.
        """
        indices = np.asarray(indices, dtype=np.int64)
        columns = tuple(trait["covariate_indices"])
        key = columns, len(indices), hashlib.sha256(memoryview(indices).cast("B")).digest()
        cached = self._design_cache.get(key)
        if cached is not None and np.array_equal(indices, cached["sample_indices"]):
            self._design_cache.move_to_end(key)
            self._design_cache_hits += 1
            return cached
        self._design_cache_misses += 1
        arrays = {"sample_indices": self._immutable_array(indices),
                  "sample_ids": self._immutable_array(self.sample_ids[indices]),
                  "covariates": self._immutable_array(
                      self.covariates[np.ix_(indices, columns)])}
        size = sum(value.nbytes for value in arrays.values())
        if size <= self._design_cache_limit_bytes:
            while self._design_cache and self._design_cache_bytes + size > self._design_cache_limit_bytes:
                _, evicted = self._design_cache.popitem(last=False)
                self._design_cache_bytes -= sum(value.nbytes for value in evicted.values())
                self._design_cache_evictions += 1
            self._design_cache[key] = arrays
            self._design_cache_bytes += size
        return arrays

    def design_cache_summary(self):
        """Owned CPU input reuse counters; no fitted covariance is shared."""
        return {"entries": len(self._design_cache), "resident_bytes": self._design_cache_bytes,
                "limit_bytes": self._design_cache_limit_bytes, "hits": self._design_cache_hits,
                "misses": self._design_cache_misses, "evictions": self._design_cache_evictions,
                "shared_fitted_models": False}

    def trait(self, index, *, cohort_indices=None):
        """Return phenotype-wise rows under the bound policy, no intercept.

        ``cohort_indices`` may be a unique vector of cache row indices or a
        cache-length boolean mask.  Returned rows retain original cache order.
        ``sample_indices`` map directly to the immutable population cache.
        """
        if not isinstance(index, int) or not 0 <= index < self.manifest["trait_count"]:
            raise IndexError("Phenotype index is outside the prepared matrix")
        trait = self.metadata["traits"][index]
        values = self.phenotypes[:, index]
        keep = np.array(self.profile_complete[trait["profile_index"]], dtype=bool, copy=True)
        if self.rm_nan:
            # Missing trait values in an existing CSV row may be filled. A
            # participant absent from the phenotype CSV is not a new control.
            keep &= self.phenotype_source_present
        if cohort_indices is not None:
            selected = np.asarray(cohort_indices)
            if selected.dtype == np.bool_:
                if selected.shape != keep.shape:
                    raise ValueError("Cohort boolean mask has the wrong shape")
                keep &= selected
            else:
                if selected.ndim != 1 or selected.dtype.kind not in "iu" or np.any(selected < 0) or np.any(selected >= len(keep)) or len(np.unique(selected)) != len(selected):
                    raise ValueError("Cohort indices must be unique cache-axis row indices")
                mask = np.zeros(len(keep), dtype=bool); mask[selected] = True; keep &= mask
        policy = apply_phenotype_missing_policy(values, family=trait["family"],
            rm_nan=self.rm_nan, covariate_valid=keep, binary_only=self.binary_only)
        indices = policy["sample_indices"]
        covariate_indices = trait["covariate_indices"]
        design = self._shared_design(indices, trait)
        return {"index": index, "name": trait["name"], "family": trait["family"],
                "profile": trait["profile"], "sample_indices": design["sample_indices"],
                "sample_ids": design["sample_ids"],
                "y_raw": policy["values"],
                "covariates": design["covariates"],
                "covariate_names": [self.metadata["covariate_names"][i] for i in covariate_indices],
                "intercept_added": False, "rm_nan": self.rm_nan,
                "missing_policy": policy["missing_policy"],
                "available_n": policy["available_n"], "mother_n": policy["mother_n"],
                "imputed_n": policy["imputed_n"], "imputation_mean": policy["imputation_mean"],
                "recoded_nonbinary_n": policy["recoded_nonbinary_n"],
                "binary_only": self.binary_only, "input_contract": self.input_contract,
                **{key: policy[key] for key in ("analysis_cases", "analysis_controls",
                    "available_cases", "available_controls", "imputed_controls")}}

    def close(self):
        self._design_cache.clear()
        self._design_cache_bytes = 0
        for value in (self.sample_ids, self.phenotypes, self.covariates, self.profile_complete,
                      self.phenotype_source_present):
            mapping = getattr(value, "_mmap", None)
            if mapping is not None:
                mapping.close()
