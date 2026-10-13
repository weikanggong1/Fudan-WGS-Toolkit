"""Association rows and ordered independent-trait result topology.

SPDX-License-Identifier: GPL-3.0-only
Adapted from the work of Xihao Li, Zilin Li and Yuxin Yuan.
The helpers assemble computed statistics; they do not fit null models.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
from typing import Any

from .masks import CODING_CATEGORIES, NONCODING_CATEGORIES


_ELEMENTARY_TESTS = (
    ("SKAT(1,25)", "WGS-S(1,25)"),
    ("SKAT(1,1)", "WGS-S(1,1)"),
    ("Burden(1,25)", "WGS-B(1,25)"),
    ("Burden(1,1)", "WGS-B(1,1)"),
    ("ACAT-V(1,25)", "WGS-A(1,25)"),
    ("ACAT-V(1,1)", "WGS-A(1,1)"),
)


class TraitRows(list):
    """Ordered records retaining original scan labels and allele level order.

    ``row_names`` follows the current row order; ``factor_levels`` maps REF/ALT
    to the original level order before the final position sort. The row values
    remain ordinary mappings so debug JSON and existing Python callers work.
    """
    def __init__(self, rows=(), *, row_names=None, factor_levels=None):
        super().__init__(rows)
        if row_names is not None and len(row_names) != len(self):
            raise ValueError("row_names must have one entry per result row")
        self.row_names = None if row_names is None else list(row_names)
        self.factor_levels = dict(factor_levels or {})


class MatrixRows(list):
    """Result records retaining implicit labels for deterministic table assembly."""
    def __init__(self, rows=(), *, matrix_row_names=None):
        super().__init__(rows)
        if matrix_row_names is not None and len(matrix_row_names) != len(self):
            raise ValueError("matrix_row_names must match the number of rows")
        self.matrix_row_names = matrix_row_names


def coding_record(
    chromosome: object, gene_name: str, category: str,
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a gene-mask row with the original first five column names."""
    row = {
        "Gene name": gene_name, "Chr": chromosome, "Category": category,
        "#SNV": statistics["num_variant"], "cMAC": statistics["cMAC"],
    }
    row.update((key, value) for key, value in statistics.items()
               if key not in ("num_variant", "cMAC"))
    return row


def single_variant_record(
    chromosome: object, position: int, ref: str, alt: str,
    alt_af: float, maf: float, number_samples: int,
    statistics: Mapping[str, Any], *, number_phenotypes: int = 1,
    use_spa: bool = False,
) -> dict[str, Any]:
    """Assemble an individual-variant row from an original-style score result.

    ``pvalue_log`` is the positive natural logarithm of the inverse p-value.
    Ordinary single-trait results require Score, Score_se, Est and Est_se.
    Joint models provide a Score vector, producing Score1, Score2, ... columns.
    The SPA branch requires pvalue and contains no score or estimate columns.
    """
    if number_phenotypes < 1:
        raise ValueError("number_phenotypes must be positive")
    row = {
        "CHR": chromosome, "POS": int(position), "REF": ref, "ALT": alt,
        "ALT_AF": float(alt_af), "MAF": float(maf), "N": int(number_samples),
    }
    if use_spa:
        row["pvalue"] = float(statistics["pvalue"])
        return row
    pvalue_log = float(statistics["pvalue_log"])
    row["pvalue"] = math.exp(-pvalue_log)
    row["pvalue_log10"] = pvalue_log / math.log(10)
    if number_phenotypes == 1:
        for key in ("Score", "Score_se", "Est", "Est_se"):
            row[key] = float(statistics[key])
    else:
        scores = statistics["Score"]
        if len(scores) != number_phenotypes:
            raise ValueError("Score must have one entry per jointly modelled phenotype")
        for index, score in enumerate(scores, 1):
            row[f"Score{index}"] = float(score)
    return row


def _elementary_values(row: Mapping[str, Any], base: str) -> list[float]:
    # Annotation columns share the base-test prefix; combined WGS columns do
    # not. The disruptive column is appended separately and must not repeat.
    return [float(value) for key, value in row.items()
            if key == base or (key.startswith(base + "-") and key != base + "-Disruptive")]


def _combiner(callback: Callable[[Sequence[float]], float] | None):
    if callback is not None:
        return callback
    from .statistics import cct
    return cct


def _summary_key(row: dict[str, Any], public_key: str) -> str:
    """Retain the summary namespace supplied by the computed input row."""
    private_key = "STAAR-" + public_key[4:]
    if public_key in row and private_key in row:
        public_value, private_value = float(row[public_key]), float(row[private_key])
        if public_value != private_value and not (math.isnan(public_value) and math.isnan(private_value)):
            raise ValueError("conflicting public and private association summaries")
        # Identical aliases carry one statistic. Retain the public spelling,
        # rather than leave an alias holding the pre-disruptive combination.
        del row[private_key]
    return private_key if private_key in row else public_key


def add_disruptive_missense(
    missense: Mapping[str, Any], disruptive: Mapping[str, Any] | None,
    *, use_spa: bool = False,
    cauchy_combiner: Callable[[Sequence[float]], float] | None = None,
) -> dict[str, Any]:
    """Append disruptive columns and reproduce the original missense summaries.

    If the disruptive mask has no valid statistical result, its extra columns
    are ones and existing combined p-values remain unchanged. A valid ordinary
    result updates six WGS summaries and WGS-O, retaining ACAT-O. The SPA
    branch updates the two burden summaries and WGS-B, using the upstream
    missing-value/one filtering convention. Each updated summary retains the
    input row's existing namespace, including the private result boundary.
    Conflicting aliases are rejected before producing a combined row.
    """
    row = dict(missense)
    tests = _ELEMENTARY_TESTS[2:4] if use_spa else _ELEMENTARY_TESTS
    summary_keys = {summary: _summary_key(row, summary) for _, summary in tests}
    omnibus_key = _summary_key(row, "WGS-B" if use_spa else "WGS-O")
    for base, _ in tests:
        value = 1.0 if disruptive is None else float(disruptive[base])
        if use_spa and math.isnan(value):
            value = 1.0
        row[base + "-Disruptive"] = value
    if disruptive is None:
        return row
    combine = _combiner(cauchy_combiner)
    all_values = []
    for base, summary in tests:
        values = _elementary_values(missense, base) + [row[base + "-Disruptive"]]
        all_values.extend(values)
        if use_spa:
            selected = [value for value in values if not math.isnan(value) and value < 1]
            # Source tests sum(p[p < 1]) > 0, including its all-zero behavior.
            row[summary_keys[summary]] = float(combine(selected)) if sum(selected) > 0 else 1.0
        else:
            row[summary_keys[summary]] = float(combine(values))
    if use_spa:
        selected = [value for value in all_values if not math.isnan(value) and value < 1]
        row[omnibus_key] = float(combine(selected)) if sum(selected) > 0 else 1.0
    else:
        row[omnibus_key] = float(combine(all_values))
    return row


def assemble_phewas_results(
    records_by_trait: Sequence[Sequence[Mapping[str, Any]]], *, kind: str,
    category: str = "all_categories", include_ptv: bool = False,
    include_ncrna: bool = False, use_spa: bool = False,
    cauchy_combiner: Callable[[Sequence[float]], float] | None = None,
) -> list[list[dict[str, Any]]] | dict[str, list[list[dict[str, Any]]]]:
    """Return the original PheWAS list topology without requiring pandas.

    Input rows are grouped by null-model order. Empty results are empty lists.
    Singlevariant, ncRNA and selected-category results retain that
    outer order. Coding/noncoding all_categories results have categories outside
    the trait lists, including empty entries for masks with no valid result.
    Coding missense rows use a matching disruptive row from the same gene and
    chromosome. Supply both masks even when requesting only missense output.
    """
    if kind not in ("coding", "noncoding", "ncrna", "singlevariant"):
        raise ValueError("kind must be coding, noncoding, ncrna or singlevariant")
    traits = [TraitRows([dict(row) for row in rows],
                       row_names=getattr(rows, "row_names", None),
                       factor_levels=getattr(rows, "factor_levels", None))
              for rows in records_by_trait]
    if kind == "singlevariant":
        # Upstream combines block outputs and finally orders by position.
        ordered = []
        for rows in traits:
            order = sorted(range(len(rows)), key=lambda index: rows[index]["POS"])
            ordered.append(TraitRows([rows[index] for index in order],
                row_names=None if rows.row_names is None else [rows.row_names[index] for index in order],
                factor_levels=rows.factor_levels))
        return ordered
    if kind == "ncrna":
        return traits
    if kind == "coding" and category == "all_categories_incl_ptv":
        category = "all_categories"
        include_ptv = True
    if kind == "coding" and category in ("ptv", "ptv_ds"):
        include_ptv = True
    categories = list(CODING_CATEGORIES if kind == "coding" else NONCODING_CATEGORIES)
    if kind == "coding" and include_ptv:
        categories += ["ptv", "ptv_ds"]
    if kind == "noncoding" and include_ncrna:
        categories += ["ncRNA"]
    if category != "all_categories" and category not in categories:
        raise ValueError(f"unknown {kind} category {category!r}")
    if kind == "coding":
        for rows in traits:
            disruptive_by_gene = {
                (row["Gene name"], str(row["Chr"])): row for row in rows
                if row["Category"] == "disruptive_missense"
            }
            for index, row in enumerate(rows):
                if row["Category"] == "missense":
                    disruptive = disruptive_by_gene.get((row["Gene name"], str(row["Chr"])))
                    rows[index] = add_disruptive_missense(
                        row, disruptive, use_spa=use_spa, cauchy_combiner=cauchy_combiner,
                    )
    if category != "all_categories":
        selected = []
        for rows in traits:
            filtered = [row for row in rows if row["Category"] == category]
            labels = []
            for row in filtered:
                disruptive_exists = any(other["Category"] == "disruptive_missense"
                    and other["Gene name"] == row["Gene name"]
                    and str(other["Chr"]) == str(row["Chr"]) for other in rows)
                labels.append("results_m" if category == "missense" and disruptive_exists
                              else "results_temp")
            selected.append(MatrixRows(filtered, matrix_row_names=labels))
        return selected
    return {
        mask: [[row for row in rows if row["Category"] == mask] for rows in traits]
        for mask in categories
    }
