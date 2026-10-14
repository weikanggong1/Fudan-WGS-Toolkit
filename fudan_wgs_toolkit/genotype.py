"""Diploid dosage algebra and sparse genotype blocks."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np

class PreparedGenotypeReader(Protocol):
    n_samples: int
    n_variants: int
    def sample_ids(self): ...
    def sample_indices(self, identifiers): ...

def _allele_frequency_summary(reference_ac, called_alleles, n_samples: int):
    """Compute reference-allele summaries and initial minor allele counts."""
    called = np.asarray(called_alleles, dtype=np.int64)
    reference = np.asarray(reference_ac, dtype=np.float64).copy()
    reference[called == 0] = np.nan
    ref_af = np.divide(reference, called, out=np.full(reference.shape, np.nan), where=called > 0)
    total_alleles = 2 * n_samples
    missing_alleles = total_alleles - called
    missing_rate = (missing_alleles / total_alleles if total_alleles else
                    np.full(reference.shape, np.nan))
    # Preserve division, complement, multiplication, and ties-to-even round.
    # With an odd called-allele count this differs from round(called / 2).
    alt_ac = 2 * np.rint(n_samples * (1 - missing_rate)) - reference
    initial_mac = np.where(reference >= alt_ac, alt_ac, reference)
    return ref_af, missing_rate, initial_mac, reference, called

def _indices(values, size: int, label: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype.kind not in "iu":
        raise ValueError(f"{label} must be a one-dimensional integer array")
    result = result.astype(np.int64, copy=False)
    if np.any(result < 0) or np.any(result >= size):
        raise IndexError(f"{label} is outside the genotype dimension")
    # Physical variant requests are usually already strictly ordered. Adjacent
    # comparison proves uniqueness without np.unique's copied sort workspace;
    # arbitrary request order retains the original duplicate check and order.
    if (len(result) > 1 and not np.all(result[1:] > result[:-1])
            and len(np.unique(result)) != len(result)):
        raise ValueError(f"{label} contains duplicate indices")
    return result

@dataclass
class SparseMinorBlock:
    """COO genotypes oriented to the minor allele in the union of samples.

    Explicit NaN entries represent missing calls; omitted entries are zero.
    row indices refer to ``sample_indices``; columns to ``variant_indices``.
    The allele orientation remains fixed when selecting individual traits.
    """

    row: np.ndarray
    col: np.ndarray
    value: np.ndarray
    sample_indices: np.ndarray
    variant_indices: np.ndarray
    union_ref_af: np.ndarray
    union_initial_mac: np.ndarray | None = None
    union_missing_rate: np.ndarray | None = None
    union_ref_ac: np.ndarray | None = None
    union_called_alleles: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.sample_indices), len(self.variant_indices)

    def observed_mac(self, trait_rows=None) -> np.ndarray:
        """Sum unfilled called dosages in canonical COO columns, ignoring NaN.

        Reader-produced COO has one entry per sample/site; implicit entries
        are called zero. This integer-valued sum never depends on imputation.
        """
        keep = ~np.isnan(self.value)
        if trait_rows is not None:
            rows = _indices(trait_rows, self.shape[0], "trait_rows")
            members = np.zeros(self.shape[0], dtype=bool)
            members[rows] = True
            keep &= members[self.row]
        return np.bincount(self.col[keep], weights=self.value[keep], minlength=self.shape[1])

    def initial_mac(self) -> np.ndarray:
        """Return source allele-wise MAC, distinct from whole-dosage MAC."""
        if self.union_initial_mac is not None:
            return self.union_initial_mac
        # Backwards-compatible manually constructed complete-diploid COO.
        # A partial allele call requires the explicit source summary fields.
        return self.observed_mac()

    def allele_missing_rate(self) -> np.ndarray:
        """Return source allele missing rates; supplied by every genotype reader."""
        if self.union_missing_rate is not None:
            return self.union_missing_rate
        missing = np.bincount(self.col[np.isnan(self.value)], minlength=self.shape[1])
        return missing / self.shape[0] if self.shape[0] else np.full(self.shape[1], np.nan)

    def select_columns(self, column_indices):
        """Select local variant columns, retaining missing entries and order."""
        columns = _indices(column_indices, self.shape[1], "column_indices")
        mapping = np.full(self.shape[1], -1, dtype=np.int64)
        mapping[columns] = np.arange(len(columns))
        keep = mapping[self.col] >= 0
        summaries = {name: None if getattr(self, name) is None else getattr(self, name)[columns]
                     for name in ("union_initial_mac", "union_missing_rate", "union_ref_ac", "union_called_alleles")}
        return SparseMinorBlock(self.row[keep], mapping[self.col[keep]], self.value[keep],
                                self.sample_indices, self.variant_indices[columns],
                                self.union_ref_af[columns], **summaries)

    def trait_dense(self, trait_rows, imputation: str = "mean", *, frequency_mode: str = "count", dtype=np.float64):
        """Return genotypes, MAF, observed MAC, missing counts, and ALT orientation.

        ``trait_rows`` contains zero-based positions in this block's union
        sample list, in the null model's order.  Mean imputation retains the
        observed-call MAF; minor imputation uses zero and divides MAC by 2N.
        ``dtype`` selects float64 control or direct float32 storage. Frequency
        reductions remain float64; mean fills cast the original double value
        once, matching float64 preparation followed by float32 conversion.
        """
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float64), np.dtype(np.float32)):
            raise ValueError("dosage dtype must be float64 or float32")
        trait_rows = _indices(trait_rows, self.shape[0], "trait_rows")
        if imputation not in ("mean", "minor"):
            raise ValueError("imputation must be 'mean' or 'minor'")
        if frequency_mode not in ("count", "reference"):
            raise ValueError("frequency_mode must be count or reference")
        if frequency_mode == "reference" and len(trait_rows) != self.shape[0]:
            raise ValueError("reference frequency requires the complete union sample set")
        mapping = np.full(self.shape[0], -1, dtype=np.int64)
        mapping[trait_rows] = np.arange(len(trait_rows))
        keep = mapping[self.row] >= 0
        genotype = np.zeros((len(trait_rows), self.shape[1]), dtype=dtype)
        genotype[mapping[self.row[keep]], self.col[keep]] = self.value[keep]
        missing = np.isnan(genotype)
        mac = np.nansum(genotype, axis=0, dtype=np.float64)
        count = len(trait_rows) - missing.sum(axis=0)
        maf = np.divide(mac, 2 * count, out=np.full_like(mac, np.nan), where=count > 0)
        if frequency_mode == "reference":
            # Retain the reference-frequency subtraction order used by the score formula.
            # PheWAS recomputes each trait frequency from its observed MAC.
            alt_af = 1 - self.union_ref_af
            maf = np.where(self.union_ref_af >= alt_af, alt_af, self.union_ref_af)
            # Both original wrappers select ALT at a union AF tie.
            # The block already retains that global orientation.
        if imputation == "mean":
            row, col = np.nonzero(missing)
            genotype[row, col] = 2 * maf[col]
        else:
            genotype[missing] = 0
            if frequency_mode == "reference":
                restored_mac = np.rint(((2 * maf) * (1 - self.allele_missing_rate())) * len(trait_rows))
                maf = restored_mac / (2 * len(trait_rows)) if len(trait_rows) else restored_mac * np.nan
            else:
                maf = mac / (2 * len(trait_rows)) if len(trait_rows) else mac * 0
        is_alt = self.union_ref_af >= 0.5
        return genotype, maf, mac, missing.sum(axis=0), is_alt

    def to_torch_sparse(self, device="cpu"):
        """Return a float64 COO tensor, retaining explicit missing entries."""
        import torch

        coordinates = torch.as_tensor(np.vstack((self.row, self.col)), device=device)
        values = torch.as_tensor(self.value, dtype=torch.float64, device=device)
        return torch.sparse_coo_tensor(coordinates, values, self.shape).coalesce()
