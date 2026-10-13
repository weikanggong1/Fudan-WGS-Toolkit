"""Bounded independent binary gene tiles with shared exact burden columns."""
from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass, field
import math
import traceback
import numpy as np
import torch

from .binary import (PreparedBinaryBurdens, _BINARY_BURDEN_SEAL, _genotype_stamp,
    _BinaryPreparedTileBinding, _BinaryPanelSource, staar_binary_phewas)
from .binary_null import binary_burden_score_covariance_batch
from .statistics import _double, _chi1_from_score, _cauchy_sf_tensor
from .precision_audit import explicit_binary_spa_fp64


@dataclass(frozen=True)
class BinaryHostPanelSource(_BinaryPanelSource):
    """One open verified cohort/mask recipe; never owns whole-mask dosage.

    Every reread independently verifies the first-pass MAF and variant order.
    Conversion uses the mature CPU imputation helper and original F layout;
    native and exact burden products retain their existing 512-column order.
    """
    reader: object
    binding: object
    samples: np.ndarray
    identity: np.ndarray
    variants: np.ndarray
    expected_maf: np.ndarray
    sample_ids: np.ndarray
    imputation: str
    host_guard: object = field(repr=False)
    host_ready: object = field(repr=False)
    _geometry: object = field(init=False, repr=False)
    _source_owner: object = field(init=False, repr=False)

    def __post_init__(self):
        from .phewas_runtime.shared_state import _immutable, _array_geometry, _readonly_source
        variants = _immutable(self.variants)
        f = np.asarray(self.expected_maf, dtype=np.float64)
        maf = np.frombuffer(f.tobytes(), dtype=f.dtype)
        ids = np.asarray(self.sample_ids, dtype=str)
        ids = np.frombuffer(ids.tobytes(), dtype=ids.dtype)
        if (variants.ndim != 1 or len(variants) != len(maf) or len(ids) != len(self.samples)
                or self.imputation not in ('mean', 'minor')):
            raise ValueError('binary host panel recipe has invalid dimensions or imputation')
        object.__setattr__(self, 'variants', variants)
        object.__setattr__(self, 'expected_maf', maf)
        object.__setattr__(self, 'sample_ids', ids)
        object.__setattr__(self, '_geometry', tuple((_array_geometry(a), _readonly_source(a))
            for a in (variants, maf, ids)))
        broker = self.binding.broker
        object.__setattr__(self, '_source_owner', (broker._reader, broker._container, broker._seal))
        self.validate()

    @property
    def shape(self):
        return len(self.samples), len(self.variants)

    def explicit_storage_owners(self):
        """Direct recipe arrays only; never follow reader or fitted-state owners."""
        self.validate()
        return dict(samples=self.samples, identity=self.identity,
                    variants=self.variants, expected_maf=self.expected_maf,
                    sample_ids=self.sample_ids)

    def validate(self):
        from .phewas_runtime.shared_state import _array_geometry, _readonly_source
        self.binding.validate(self.reader, self.samples, self.identity, self.shape[0])
        broker = self.binding.broker
        if (self._source_owner != (broker._reader, broker._container, broker._seal)
                or self._geometry != tuple((_array_geometry(a), _readonly_source(a))
                    for a in (self.variants, self.expected_maf, self.sample_ids))):
            raise ValueError('binary host panel recipe source or immutable axes changed')
        return (id(self), id(broker), id(broker._container), id(self.binding.axis),
                self._geometry, self.imputation, self.shape)

    def read_columns(self, columns):
        self.validate()
        requested = np.asarray(columns)
        if (requested.ndim != 1 or requested.dtype.kind not in 'iu'
                or len(requested) > self.maximum_panel_width
                or np.any(requested < 0) or np.any(requested >= self.shape[1])):
            raise ValueError('binary host panel columns exceed the verified bounded recipe')
        raw = block = compact = output = tensor = None
        admitted = self.host_guard is None
        try:
            # Admission precedes both compact CPU dosage and floating output.
            if self.host_guard is not None:
                admitted = False
                self.host_guard(len(requested), 0)
                admitted = True
            raw = self.binding.broker.read_states(self.variants[requested])
            block = self.binding.broker.trait_block(raw, self.samples)
            del raw
            raw = None
            if (not np.array_equal(block.variant_indices, self.variants[requested])
                    or not np.array_equal(block.sample_indices, self.samples)):
                raise ValueError('binary host panel reread changed its ordered axes')
            maf = block.trait_summary(self.identity, self.imputation,
                                      frequency_mode='reference')[0]
            if not np.array_equal(maf, self.expected_maf[requested]):
                raise ValueError('binary host panel reread changed first-pass cohort MAF')
            compact = block.to_host()
            block = None
            if self.host_guard is not None:
                admitted = False
                self.host_guard(len(requested), compact.dosage.numel())
                admitted = True
            output = np.empty((self.shape[0], len(requested)), dtype=np.float32, order='F')
            tensor = torch.from_numpy(output)
            for start in range(0, len(requested), 128):
                stop = min(start+128, len(requested))
                compact.trait_dense_columns(self.identity, np.arange(start, stop, dtype=np.int64),
                    self.imputation, frequency_mode='reference', dtype=torch.float32,
                    out=tensor, out_columns=np.arange(start, stop, dtype=np.int64))
            return tensor
        except BaseException as error:
            # Mature conversion/read leaves can otherwise pin panel owners
            # in an exception traceback while admission retries are pending.
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            # Returned panel owns only its initialized FP32 output. The recipe
            # retains no raw/minor states, floating panel or exception locals.
            raw = block = compact = output = tensor = None
            # A rejected host gate can retire its lease. Resetting that closed
            # lease would replace HostMemoryDeferred with a RuntimeError and
            # prevent the caller's established unwind/retry protocol.
            if admitted and self.host_ready is not None:
                self.host_ready()


def _checked_binding(models, prepared, genotype, maf, annotations, names,
                     rare_maf_cutoff, rv_num_cutoff, rv_num_cutoff_max):
    if (not models or not isinstance(prepared, PreparedBinaryBurdens)
            or prepared._seal is not _BINARY_BURDEN_SEAL
            or prepared.source is not genotype
            or prepared.source_stamp != _genotype_stamp(genotype)):
        raise ValueError('binary tile requires the same verified prepared source')
    first = models[0]
    if not np.array_equal(prepared.sample_ids, first.sample_ids):
        raise ValueError('binary tile has a different sample axis')
    for model in models:
        if (model.family != 'binomial' or model.n_pheno != 1
                or model.n != prepared.burdens.shape[0] or model.device != prepared.burdens.device
                or model.matmul_mode != prepared.matmul_mode
                or (model.sample_ids is not first.sample_ids
                    and not np.array_equal(model.sample_ids, first.sample_ids))):
            raise ValueError('binary tile models must retain independent states on one sample axis')
    full_f = _double(maf, device=first.device)
    selected = (full_f > 0) & (full_f < rare_maf_cutoff)
    columns = torch.nonzero(selected, as_tuple=True)[0].cpu().numpy()
    given_a = None if annotations is None else _double(annotations, device=first.device)[selected]
    labels = tuple(f'annotation_{i+1}' for i in range(0 if given_a is None else given_a.shape[1])) if names is None else tuple(names)
    if (not np.array_equal(columns, prepared.columns)
            or not torch.equal(full_f[selected], prepared.maf)
            or (given_a is None) != (prepared.annotations is None)
            or labels != prepared.names
            or (given_a is not None and not torch.equal(given_a, prepared.annotations))):
        raise ValueError('binary tile has different selected MAF/annotations')
    return _BinaryPreparedTileBinding(prepared, models, genotype, maf, annotations, names,
        rare_maf_cutoff, rv_num_cutoff, rv_num_cutoff_max)


def _spa_combination_rows(values):
    """Exported CCT semantics after the binary wrapper's failure/NA filtering."""
    p = values.double()
    accepted = torch.isfinite(p) & (p < 1)
    mass = accepted.sum(1).clamp(min=1)
    safe = torch.where(accepted, p, .5)
    weights = accepted.to(dtype=p.dtype)/mass[:, None]
    small = accepted & (safe < 1e-16)
    statistic = torch.where(small, weights/safe/math.pi, 0.).sum(1)
    statistic += torch.where(accepted & ~small,
        weights*torch.tan((.5-safe)*math.pi), 0.).sum(1)
    tail = _cauchy_sf_tensor(statistic)
    ordinary = torch.where(statistic <= 0, tail, 1-(1-tail))
    result = torch.where(statistic > 1e15, (1/statistic)/math.pi, ordinary)
    # Original wrapper returns one when every accepted probability is zero.
    return torch.where(torch.where(accepted, p, 0.).sum(1) > 0, result, 1.)


def evaluate_binary_burden_tile(models, genotype, maf, annotations, names, *,
        prepared_burdens, spa_acquires, exact_burden_cache,
        rare_maf_cutoff=.01, rv_num_cutoff=2, rv_num_cutoff_max=10**9,
        p_filter_cutoff=.05, tol=2**-13, max_iter=1000,
        normal_allocation_guard=None, normal_workspace_release=None, profiler=None):
    """Return the unchanged per-trait dictionaries with one packed CPU transfer.

    The caller owns one mask cache and the repository leases. Normal products
    use independent fitted states. SPA keeps its original FP64 sidecar and
    solver, taking original FP64 burden columns from the mask cache.
    """
    models, spa_acquires = tuple(models), tuple(spa_acquires)
    if len(models) != len(spa_acquires):
        raise ValueError('every binary model needs its own optional SPA factory')
    binding = _checked_binding(models, prepared_burdens, genotype, maf, annotations, names,
        rare_maf_cutoff, rv_num_cutoff, rv_num_cutoff_max)
    measure = (lambda name: profiler.measure(name, gpu=True)) if profiler is not None else (lambda name: nullcontext())
    try:
        with measure('gene_binary_normal_projection_score'):
            scores, covariances, execution = binary_burden_score_covariance_batch(models,
                prepared_burdens.burdens, allocation_guard=normal_allocation_guard)
    finally:
        if normal_workspace_release is not None:
            normal_workspace_release()
    with measure('gene_binary_shared_exact_burdens'):
        nominal = _chi1_from_score(scores, covariances.diagonal(dim1=-2, dim2=-1))
        # IEEE GEMM and the historical GEMV can differ in their reduction order.
        # Preserve the strict SPA admission boundary with original vector scores
        # only for the rare tile rows near that threshold.
        execution['threshold_score_rechecks'] = 0
        if execution['native_projection_batches']:
            close = ((nominal-p_filter_cutoff).abs() <= 2e-5*p_filter_cutoff).any(1)
            rows = torch.nonzero(close, as_tuple=True)[0].detach().cpu().tolist()
            if rows:
                from .tf32 import matmul
                for i in rows:
                    scores[i].copy_(matmul(prepared_burdens.burdens.T,
                                          models[i].scaled_residuals, mode='tf32'))
                    nominal[i] = _chi1_from_score(scores[i], covariances[i].diagonal())
                execution['threshold_score_rechecks'] = len(rows)
        requires_spa = torch.tensor([factory is not None for factory in spa_acquires],
                                   dtype=torch.bool, device=nominal.device)
        selected = (nominal < p_filter_cutoff) & requires_spa[:, None]
        union = torch.nonzero(selected.any(0), as_tuple=True)[0]
        if len(union) and exact_burden_cache is not None:
            with explicit_binary_spa_fp64():
                exact_burden_cache.get(prepared_burdens, union)
    probabilities, counts = [], []
    for i, (model, acquire) in enumerate(zip(models, spa_acquires)):
        with measure('gene_binary_selective_spa'):
            values, diagnostic = staar_binary_phewas(model, genotype, maf, annotations, names,
                spa_acquire=acquire, prepared_burdens=prepared_burdens,
                rare_maf_cutoff=rare_maf_cutoff, rv_num_cutoff=rv_num_cutoff,
                rv_num_cutoff_max=rv_num_cutoff_max, p_filter_cutoff=p_filter_cutoff,
                tol=tol, max_iter=max_iter, _normal_products=(scores[i], covariances[i]),
                _exact_burden_cache=exact_burden_cache, _device_output=True,
                _prepared_binding=binding)
        probabilities.append(values)
        counts.append(torch.stack((diagnostic['spa_selected'].sum(), diagnostic['spa'].failed.sum())))
    pvalues = torch.stack(probabilities)
    width = len(prepared_burdens.names)+1
    with measure('gene_binary_packed_output'):
        omnibus = torch.stack((_spa_combination_rows(pvalues[:, :width]),
                              _spa_combination_rows(pvalues[:, width:]),
                              _spa_combination_rows(pvalues)), dim=1)
        packed = torch.cat((pvalues, omnibus, torch.stack(counts).double()), dim=1)
        host = packed.detach().cpu().numpy()
    result = []
    for row in host:
        value = dict(num_variant=len(prepared_burdens.maf), cMAC=prepared_burdens.cmac)
        for group, beta in enumerate(('1,25', '1,1')):
            begin = group*width
            value[f'Burden({beta})'] = float(row[begin])
            for j, label in enumerate(prepared_burdens.names, 1):
                value[f'Burden({beta})-{label}'] = float(row[begin+j])
            value[f'STAAR-B({beta})'] = float(row[2*width+group])
        value['STAAR-B'] = float(row[2*width+2])
        result.append((value, int(row[-2]), int(row[-1])))
    return result, execution
