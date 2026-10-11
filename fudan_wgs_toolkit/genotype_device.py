"""Device-resident lossless minor dosage blocks for bounded Single scans.

Only small allele summaries cross to the host. The uint8 matrix uses 3 for
whole-genotype missingness and otherwise contains minor-copy counts 0..2.
Source AF/MAC and trait frequency formulas retain NumPy float64 operation order.
"""
from dataclasses import dataclass, field
import numpy as np
from .genotype import _indices

_COLUMN_DENSE_KERNEL = None


def _column_dense_kernel():
    """Convert selected lossless columns directly into an admitted output."""
    global _COLUMN_DENSE_KERNEL, triton, tl
    if _COLUMN_DENSE_KERNEL is not None:
        return _COLUMN_DENSE_KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(S, ROWS, COLS, MEANS, DEST, O, N, M, SOURCE_ROW_STRIDE,
               SOURCE_COLUMN_STRIDE, OUTPUT_ROW_STRIDE, OUTPUT_COLUMN_STRIDE,
               IDENTITY_ROWS: tl.constexpr, MEAN_IMPUTATION: tl.constexpr,
               BLOCK: tl.constexpr):
        x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = x < N * M
        row, col = x // M, x % M
        source_row = row if IDENTITY_ROWS else tl.load(ROWS + row, valid, other=0)
        source_col = tl.load(COLS + col, valid, other=0)
        state = tl.load(S + source_row.to(tl.int64) * SOURCE_ROW_STRIDE
                        + source_col.to(tl.int64) * SOURCE_COLUMN_STRIDE,
                        valid, other=3)
        if MEAN_IMPUTATION:
            mean = tl.load(MEANS + col, valid, other=0)
            value = tl.where(state == 3, mean, state)
        else:
            value = tl.where(state == 3, 0, state)
        destination = tl.load(DEST + col, valid, other=0)
        tl.store(O + row.to(tl.int64) * OUTPUT_ROW_STRIDE
                 + destination.to(tl.int64) * OUTPUT_COLUMN_STRIDE, value, valid)

    _COLUMN_DENSE_KERNEL = kernel
    return kernel


@dataclass
class DeviceMinorBlock:
    dosage: object
    sample_indices: np.ndarray
    variant_indices: np.ndarray
    union_ref_af: np.ndarray
    union_initial_mac: np.ndarray
    union_missing_rate: np.ndarray
    union_ref_ac: np.ndarray
    union_called_alleles: np.ndarray
    _counts_cache: object = field(default=None, init=False, repr=False)

    @property
    def shape(self):
        return len(self.sample_indices), len(self.variant_indices)

    def initial_mac(self):
        return self.union_initial_mac

    def allele_missing_rate(self):
        return self.union_missing_rate

    def _dosage_stamp(self):
        # Normal Torch in-place writes invalidate cached counts. Inference
        # tensors without version counters conservatively bypass caching.
        try:
            return (id(self.dosage), self.dosage._version, tuple(self.dosage.shape))
        except RuntimeError:
            return None

    def _cached_counts(self, rows):
        stamp = self._dosage_stamp()
        cached = self._counts_cache
        if stamp is not None and cached is not None and stamp == cached[0] and np.array_equal(rows, cached[1]):
            return cached[2], cached[3]
        self._counts_cache = None
        return None

    def _store_counts(self, rows, mac, missing_count):
        stamp = self._dosage_stamp()
        if stamp is not None:
            # Only small host summaries are cached, never selected CUDA
            # dosage/missing matrices. Public return arrays remain independent.
            self._counts_cache = (stamp, rows.copy(), mac.copy(), missing_count.copy())

    def _trait_counts(self, trait_rows):
        import torch
        rows = _indices(trait_rows, self.shape[0], "trait_rows")
        identity = len(rows) == self.shape[0] and np.array_equal(rows, np.arange(self.shape[0]))
        selected = self.dosage if identity else self.dosage.index_select(0, torch.as_tensor(rows, device=self.dosage.device))
        missing = selected == 3
        cached = self._cached_counts(rows)
        if cached is None:
            called = selected.masked_fill(missing, 0)
            # Count whole-genotype missingness from dosage, not allele AF/AC/AN.
            summary = torch.stack((called.sum(0, dtype=torch.int64),
                                   missing.sum(0, dtype=torch.int64))).cpu().numpy()
            mac, missing_count = summary[0].astype(np.float64), summary[1]
            self._store_counts(rows, mac, missing_count)
        else:
            mac, missing_count = cached
        return rows, selected, missing, mac.copy(), missing_count.copy()

    def observed_mac(self, trait_rows=None):
        rows = _indices(np.arange(self.shape[0]) if trait_rows is None else trait_rows,
                        self.shape[0], "trait_rows")
        cached = self._cached_counts(rows)
        return cached[0].copy() if cached is not None else self._trait_counts(rows)[3]

    def select_columns(self, column_indices):
        """Select in caller order; an identity selection shares this block.

        Lossless dosage is read-only during analysis. Normal in-place changes
        invalidate its count cache; returned host summary arrays are copies.
        """
        import torch
        columns = _indices(column_indices, self.shape[1], "column_indices")
        if len(columns) == self.shape[1] and np.array_equal(columns, np.arange(self.shape[1])):
            return self
        dosage = self.dosage.index_select(1, torch.as_tensor(columns, device=self.dosage.device))
        result = DeviceMinorBlock(dosage, self.sample_indices, self.variant_indices[columns],
            *[getattr(self, name)[columns] for name in ("union_ref_af", "union_initial_mac",
               "union_missing_rate", "union_ref_ac", "union_called_alleles")])
        cached = self._counts_cache
        if cached is not None and cached[0] == self._dosage_stamp():
            result._store_counts(cached[1], cached[2][columns], cached[3][columns])
        return result

    def _trait_frequency(self, rows, mac, missing_count, imputation, frequency_mode):
        count = len(rows) - missing_count
        maf = np.divide(mac, 2 * count, out=np.full_like(mac, np.nan), where=count > 0)
        if frequency_mode == "reference":
            alt_af = 1 - self.union_ref_af
            maf = np.where(self.union_ref_af >= alt_af, alt_af, self.union_ref_af)
        if imputation == "minor":
            if frequency_mode == "reference":
                restored_mac = np.rint(((2 * maf) * (1 - self.allele_missing_rate())) * len(rows))
                maf = restored_mac / (2 * len(rows)) if len(rows) else restored_mac * np.nan
            else:
                maf = mac / (2 * len(rows)) if len(rows) else mac * 0
        return maf

    def _trait_rows(self, trait_rows, imputation, frequency_mode):
        if imputation not in ("mean", "minor"):
            raise ValueError("imputation must be 'mean' or 'minor'")
        if frequency_mode not in ("count", "reference"):
            raise ValueError("frequency_mode must be count or reference")
        rows = _indices(trait_rows, self.shape[0], "trait_rows")
        if frequency_mode == "reference" and len(rows) != self.shape[0]:
            raise ValueError("reference frequency requires the complete union sample set")
        return rows

    def trait_summary(self, trait_rows, imputation="mean", *, frequency_mode="count"):
        """Return original host summaries without allocating floating dosage.

        Counts follow the same integer reductions and cache as trait_dense.
        Temporary row selections/missing masks are released on return.
        """
        rows = self._trait_rows(trait_rows, imputation, frequency_mode)
        cached = self._cached_counts(rows)
        if cached is None:
            _, selected, missing, mac, missing_count = self._trait_counts(rows)
            del selected, missing
        else:
            mac, missing_count = [value.copy() for value in cached]
        maf = self._trait_frequency(rows, mac, missing_count, imputation, frequency_mode)
        return maf, mac, missing_count, self.union_ref_af >= 0.5

    def trait_dense(self, trait_rows, imputation="mean", *, frequency_mode="count", dtype=None):
        """Return device dosage and original host frequency/count summaries.

        FP32 fills the original double mean cast once. No sample-by-variant
        payload is sent to CPU. float16/BF16 are rejected.
        """
        import torch
        dtype = torch.float64 if dtype is None else dtype
        if dtype not in (torch.float64, torch.float32):
            raise ValueError("dosage dtype must be torch.float64 or torch.float32")
        rows = self._trait_rows(trait_rows, imputation, frequency_mode)
        rows, selected, missing, mac, missing_count = self._trait_counts(rows)
        maf = self._trait_frequency(rows, mac, missing_count, imputation, frequency_mode)
        genotype = selected.to(dtype=dtype, copy=True)
        if imputation == "mean":
            means = torch.as_tensor(2 * maf, dtype=dtype, device=genotype.device)
            torch.where(missing, means[None, :], genotype, out=genotype)
        else:
            genotype.masked_fill_(missing, 0)
        return genotype, maf, mac, missing_count, self.union_ref_af >= 0.5

    def trait_dense_columns(self, trait_rows, column_indices, imputation="mean", *,
                            frequency_mode="count", dtype=None, out=None,
                            out_columns=None):
        """Write selected columns without a floating host dosage round trip.

        Summaries preserve ``trait_dense``'s integer counts and NumPy FP64
        frequency formulas. CUDA gathers uint8 dosage and fills the original
        double mean cast once directly into ``out``; it creates neither a
        selected uint8 matrix nor a sample-by-variant missing mask. Callers
        must admit the output and the small row/column/mean index workspace
        before invoking this method. ``out_columns`` names unique destination
        columns and allows several blocks to fill one final dosage matrix.
        The returned dosage is ``out`` itself when supplied; summaries always
        follow ``column_indices`` order. CPU mode is the exact small oracle.
        """
        import torch
        dtype = torch.float64 if dtype is None else dtype
        if dtype not in (torch.float64, torch.float32):
            raise ValueError("dosage dtype must be torch.float64 or torch.float32")
        rows = self._trait_rows(trait_rows, imputation, frequency_mode)
        columns = _indices(column_indices, self.shape[1], "column_indices")
        if out is None:
            if out_columns is not None:
                raise ValueError("out_columns requires an output tensor")
            destinations = np.arange(len(columns), dtype=np.int64)
        else:
            if (not isinstance(out, torch.Tensor) or out.ndim != 2
                    or out.shape[0] != len(rows) or out.dtype != dtype
                    or out.device != self.dosage.device or out.layout != torch.strided
                    or out.requires_grad or any(stride <= 0 for stride in out.stride())
                    or (out.numel() and self.dosage.numel()
                        and out.untyped_storage().data_ptr() == self.dosage.untyped_storage().data_ptr())
                    or not (out.stride(0) >= out.shape[1] * out.stride(1)
                            or out.stride(1) >= out.shape[0] * out.stride(0))):
                raise ValueError("output dosage must have matching rows, dtype, device and nonoverlapping row/column-major storage")
            destinations = _indices(np.arange(len(columns), dtype=np.int64)
                if out_columns is None else out_columns, out.shape[1], "out_columns")
            if len(destinations) != len(columns) or len(np.unique(destinations)) != len(destinations):
                raise ValueError("out_columns must provide one unique destination per selected column")
        cached = self._cached_counts(rows)
        if cached is None:
            _, selected, missing, mac, absent = self._trait_counts(rows)
            del selected, missing
        else:
            mac, absent = cached
        all_maf = self._trait_frequency(rows, mac, absent, imputation, frequency_mode)
        maf, mac, absent = all_maf[columns].copy(), mac[columns].copy(), absent[columns].copy()
        is_alt = (self.union_ref_af >= .5)[columns]
        output = out if out is not None else torch.empty(
            (len(rows), len(columns)), dtype=dtype, device=self.dosage.device)
        if not len(rows) or not len(columns):
            return output, maf, mac, absent, is_alt
        identity_rows = len(rows) == self.shape[0] and np.array_equal(rows, np.arange(self.shape[0]))
        if self.dosage.device.type == 'cuda':
            kernel = _column_dense_kernel()
            # Only validated small host axes/summaries move to the device.
            source_rows = None if identity_rows else torch.tensor(rows, device=self.dosage.device)
            source_columns = torch.tensor(columns, device=self.dosage.device)
            destination_columns = torch.tensor(destinations, device=self.dosage.device)
            means = torch.tensor(2 * maf, dtype=dtype, device=self.dosage.device)
            with torch.cuda.device(self.dosage.device):
                kernel[(triton.cdiv(len(rows) * len(columns), 256),)](
                    self.dosage, source_rows, source_columns, means, destination_columns,
                    output, len(rows), len(columns), *self.dosage.stride(),
                    *output.stride(), IDENTITY_ROWS=identity_rows,
                    MEAN_IMPUTATION=imputation == 'mean', BLOCK=256)
        else:
            selected = self.dosage if identity_rows else self.dosage.index_select(
                0, torch.tensor(rows, device=self.dosage.device))
            selected = selected.index_select(1, torch.tensor(columns, device=self.dosage.device))
            dense = selected.to(dtype=dtype, copy=True)
            if imputation == 'mean':
                means = torch.tensor(2 * maf, dtype=dtype, device=self.dosage.device)
                torch.where(selected == 3, means[None, :], dense, out=dense)
            else:
                dense.masked_fill_(selected == 3, 0)
            output.index_copy_(1, torch.tensor(destinations, device=self.dosage.device), dense)
        return output, maf, mac, absent, is_alt
