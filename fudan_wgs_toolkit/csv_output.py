"""Atomic association CSVs written directly from computed Python records.

The scalar formatter roundtrips IEEE double precision. Batches retain their
scheduled order, and each batch retains its category and row order.
SPDX-License-Identifier: GPL-3.0-only
"""
from __future__ import annotations

from collections.abc import Mapping
import csv
import hashlib
import math
import os
from operator import itemgetter
from pathlib import Path
import tempfile

import numpy as np

SINGLE_COLUMNS = ("CHR", "POS", "REF", "ALT", "ALT_AF", "MAF", "N", "pvalue",
                  "pvalue_log10", "Score", "Score_se", "Est", "Est_se")
GENE_COLUMNS = ("Gene name", "Chr", "Category", "#SNV", "cMAC") + tuple(
    field for method, combined in (("SKAT", "WGS-S"), ("Burden", "WGS-B"), ("ACAT-V", "WGS-A"))
    for beta in ("1,25", "1,1") for field in (f"{method}({beta})", f"{combined}({beta})")) + ("ACAT-O", "WGS-O")
_KINDS = frozenset(("individual", "singlevariant", "coding", "noncoding", "ncrna"))


def _write_bound_trait_rows(writer, rows, trait_index, source_columns):
    """Synchronously write one checked Single schema without copied mappings.

    The checkpoint caller checks every source row's ordered schema and rejects
    a source ``trait_index`` before yielding it. Values go straight to the
    standard CSV writer: its established scalar/None formatting is unchanged.
    ``writerows`` consumes one tuple at a time and retains no output queue,
    formatted strings or batch-sized temporary container. Only the current
    record and its scalar-reference tuples are held by this helper.
    """
    columns = tuple(source_columns)
    if not columns:
        raise ValueError("bound trait CSV requires nonempty source columns")
    lookup = itemgetter(*columns)
    singleton = len(columns) == 1
    count = selected = failed = 0

    def values():
        nonlocal count, selected, failed
        for source in rows:
            # Production records are plain dictionaries. Keep historical
            # keyword-expansion behavior for subclasses that override item
            # lookup/get; DictWriter previously received this exact plain copy.
            record = source if type(source) is dict else dict(trait_index=trait_index, **source)
            cells = lookup(record)
            yield (trait_index, cells) if singleton else (trait_index, *cells)
            count += 1
            selected += int(bool(record.get('spa_selected', False)))
            failed += int(bool(record.get('spa_failed', False)))

    writer.writerows(values())
    return count, selected, failed


def cell_text(value):
    """Encode one scalar without rounding numeric association statistics."""
    if value is None or np.ma.is_masked(value):
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return format(value, ".17g")
    if isinstance(value, str):
        return value
    raise ValueError("association cells must be scalar strings, numbers, logicals or missing values")


def is_probability_column(name):
    return (name in ("pvalue", "ACAT-O", "WGS-O", "WGS-B") or
            name.startswith(("SKAT(", "Burden(", "ACAT-V(", "WGS-S(", "WGS-B(", "WGS-A(")))


def association_rows(result, *, kind, trait_index=0):
    """Yield one trait's rows in the original scheduled table traversal order."""
    if kind not in _KINDS:
        raise ValueError("unsupported association kind")
    if type(trait_index) is not int or trait_index < 0:
        raise ValueError("trait_index must be a nonnegative integer")
    if isinstance(result, Mapping):
        if kind not in ("coding", "noncoding"):
            raise ValueError("only coding and noncoding results contain a category layer")
        groups = result.values()
    else:
        groups = (result,)
    for traits in groups:
        if not isinstance(traits, (list, tuple)) or len(traits) <= trait_index:
            raise ValueError("association result lacks the requested trait")
        rows = traits[trait_index]
        if not isinstance(rows, (list, tuple)):
            raise ValueError("association trait rows must be a sequence")
        for row in rows:
            if not isinstance(row, Mapping) or not row:
                raise ValueError("association rows must be nonempty mappings")
            yield row


def _normalized_row(row, kind):
    # Historical tables store these two ncRNA/UTR metadata columns as text.
    # Preserve that established representation while leaving all P values intact.
    character_metadata = kind == "ncrna" or (kind == "noncoding" and
        row.get("Category") in ("upstream", "downstream", "UTR", "ncRNA"))
    result = {}
    for name, value in row.items():
        if name.startswith("_"):
            raise ValueError("internal grouping markers must be removed before CSV output")
        if character_metadata and name in ("Chr", "#SNV"):
            value = value if isinstance(value, str) else format(float(value), ".15g")
        result[name] = value
    return result


def _records(results, kind, trait_index):
    for result in results:
        for row in association_rows(result, kind=kind, trait_index=trait_index):
            yield _normalized_row(row, kind)


def _verification(path, results, kind, trait_index, columns, excluded):
    count = numeric = probability = logarithm = 0
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.reader(stream)
        if next(rows, None) != columns:
            raise RuntimeError("CSV column roundtrip verification failed")
        for source in _records(results, kind, trait_index):
            expected = [cell_text(source.get(name)) for name in columns]
            actual = next(rows, None)
            if actual != expected:
                raise RuntimeError("CSV scalar roundtrip verification failed")
            for name, token in zip(columns, actual):
                value = source.get(name)
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, (float, int)) and not isinstance(value, bool):
                    restored = float(token) if isinstance(value, float) else int(token)
                    same = (isinstance(value, float) and math.isnan(value) and math.isnan(restored)) or restored == value
                    if not same or (value == 0 and math.copysign(1, value) != math.copysign(1, restored)):
                        raise RuntimeError("CSV numeric roundtrip verification failed")
                    numeric += 1
                probability += int(is_probability_column(name) and value is not None)
                logarithm += int(name == "pvalue_log10" and value is not None)
            count += 1
        if next(rows, None) is not None:
            raise RuntimeError("CSV has unexpected additional rows")
    if any(name in excluded for name in columns):
        raise RuntimeError("CSV retained an excluded metadata column")
    return dict(roundtrip_passed=True, numeric_roundtrip_passed=True, rows=count,
        checked_cells=count*len(columns), numeric_cells=numeric, p_cells=probability,
        logp_cells=logarithm, maximum_logp_error=0.0, maximum_stored_logp_error=0.0)


def write_association_batch(path, results, *, kind, layout="base", trait_index=0,
                            exclude_columns=(), empty_columns=None):
    """Write one completed scheduled batch to CSV and return its proof summary.

    Coding/noncoding mappings keep their category order within each scheduled
    result. Empty masks contribute no rows. Entirely empty outputs retain a
    full unannotated header unless ``empty_columns`` supplies an explicit one.
    """
    if layout not in ("base", "phewas"):
        raise ValueError("layout must be base or phewas")
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("association output must end in .csv")
    results = list(results)
    if not results:
        raise ValueError("association batch requires at least one scheduled result")
    if kind not in _KINDS:
        raise ValueError("unsupported association kind")
    if kind in ("individual", "singlevariant") and len(results) != 1:
        raise ValueError("Single jobs require separate CSV files")
    excluded = frozenset(exclude_columns)
    columns, positions = [], {}
    row_count = 0
    for row in _records(results, kind, trait_index):
        retained = [name for name in row if name not in excluded]
        for name in retained:
            if name not in positions:
                positions[name] = len(columns)
                columns.append(name)
        if [positions[name] for name in retained] != sorted(positions[name] for name in retained):
            raise ValueError("association tables have incompatible original column orders")
        row_count += 1
    empty_schema = None
    if not columns:
        if row_count:
            raise ValueError("excluded columns remove all association columns")
        selected = empty_columns if empty_columns is not None else (SINGLE_COLUMNS if kind in ("individual", "singlevariant") else GENE_COLUMNS)
        columns = [name for name in selected if name not in excluded]
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("empty CSV requires distinct retained columns")
        empty_schema = "provided" if empty_columns is not None else "default_unannotated"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as temporary:
        staging = Path(temporary.name)
    try:
        with staging.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(columns)
            for row in _records(results, kind, trait_index):
                writer.writerow([cell_text(row.get(name)) for name in columns])
            stream.flush()
            os.fsync(stream.fileno())
        verification = _verification(staging, results, kind, trait_index, columns, excluded)
        digest = hashlib.sha256()
        with staging.open("rb") as stream:
            for block in iter(lambda: stream.read(8*2**20), b""):
                digest.update(block)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)
    return dict(path=str(path), sha256=digest.hexdigest(), columns=columns,
        empty_schema=empty_schema, excluded_columns_absent=True, **verification)
