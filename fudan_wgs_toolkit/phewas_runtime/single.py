"""Independent Single cores with shared pointwise tails and result transfer.

Every block is prepared for its own singleton sample axis and source AF. No
union padding, covariance sharing or matrix precision changes take place here.
"""
from __future__ import annotations

import operator
import numpy as np
import torch

from ..genotype_device import DeviceMinorBlock
from ..pipeline import _individual_log_probabilities
from .shared_state import _SingleAxisBinding
from .scores import fp32_score_columns, execution_metadata as score_execution_metadata


_METRICS = dict(calls=0, active_trait_blocks=0, computed_trait_blocks=0,
                pointwise_tail_batches=0, result_transfer_batches=0,
                result_transfer_values=0, maximum_batch_variants=0,
                shared_genotype_batches=0, shared_genotype_traits=0,
                phenotype_score_gemm_calls=0, phenotype_score_batches=0,
                phenotype_score_fp32_vector_calls=0, phenotype_score_fp32_simt_gemm_calls=0,
                phenotype_score_simt_admission_fallbacks=0,
                phenotype_score_extreme_tail_fallback_columns=0,
                phenotype_score_control_d2h_batches=0, phenotype_score_control_d2h_bytes=0,
                original_variance_calls=0,
                projection_trait_batches=0, projection_batched_traits=0,
                projection_budget_fallbacks=0,
                dense_genotype_h2d_bytes=0, dense_genotype_d2h_bytes=0)


def execution_metadata(*, reset=False):
    """Return aggregate counts without model or variant identifiers."""
    result = dict(_METRICS,
                  genotype_core=("exact-cohort shared genotype with FP32 score tiles or original vector fallback and fitted variance"
                                 if _METRICS['shared_genotype_batches'] else "independent singleton row/K order"),
                  tail_dtype="float64 scalar stability",
                  core_storage_dtype="float32",
                  covariance_shared=False, union_sample_padding=False)
    if reset:
        for key in _METRICS:
            _METRICS[key] = 0
    return result


def _integer(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if result < (1 if positive else 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


def process_single_batches(pipelines, blocks, ordinal_offsets, chromosome,
                           mac_cutoff=20, subset_variants_num=5000):
    """Return ``(records_by_trait, new_ordinals)`` for bounded ready blocks.

    ``pipelines`` contains one base-semantics Gaussian or ordinary binary
    model per entry. ``blocks`` contains singleton ``DeviceMinorBlock`` objects
    on that model's row axis; ``None`` marks a trait whose effective-column
    buffer is not ready. ``ordinal_offsets`` is independent for each trait.
    The caller has already selected the chromosome's QC/type variant axes.

    Each original Score/variance core executes separately and releases its
    floating genotype before the next trait. Only O(sum(M)) core vectors are
    concatenated for the unchanged pointwise log tail, SE, and one host copy.
    Records use the same helper as standalone Single, including original
    source AF and extraction/chunk markers. No phenotype values are imputed.
    """
    pipelines, blocks = list(pipelines), list(blocks)
    ordinal_offsets = list(ordinal_offsets)
    if not pipelines or not (len(pipelines) == len(blocks) == len(ordinal_offsets)):
        raise ValueError("pipelines, blocks and ordinal_offsets require equal nonempty lengths")
    subset_variants_num = _integer(subset_variants_num, "subset_variants_num", positive=True)
    ordinals = [_integer(value, "ordinal_offset") for value in ordinal_offsets]
    if not np.isscalar(mac_cutoff) or not np.isfinite(mac_cutoff) or mac_cutoff < 0:
        raise ValueError("mac_cutoff must be finite and nonnegative")
    if chromosome is None:
        raise ValueError("chromosome is required")

    # Admit the complete batch before updating counters or preparing dosage.
    device = None
    for pipeline, block in zip(pipelines, blocks):
        if len(pipeline.models) != 1 or pipeline.options.wrapper_semantics != "base":
            raise ValueError("Single PheWAS requires singleton base-semantics pipelines")
        model = pipeline.models[0]
        if (model.n_pheno != 1 or model.use_spa
                or getattr(model, "family", "gaussian") not in ("gaussian", "binomial")
                or getattr(model, "matmul_mode", "fp64") != "tf32"
                or not hasattr(model, "individual_score_variance")
                or model.x.dtype != torch.float32):
            raise ValueError("Single PheWAS requires a native FP32 ordinary single-trait core without SPA")
        model_device = torch.device(model.device)
        if device is None:
            device = model_device
        elif device != model_device:
            raise ValueError("Single PheWAS cores must use the same device")
        if block is not None:
            if not isinstance(block, DeviceMinorBlock):
                raise TypeError("ready Single blocks must be DeviceMinorBlock objects")
            if block.dosage.device != model_device:
                raise ValueError("Single block and model must use the same device")
            if (not np.array_equal(block.sample_indices, pipeline.union_rows)
                    or len(pipeline.union_rows) != model.n
                    or not np.array_equal(pipeline.trait_rows[0], np.arange(model.n))):
                raise ValueError("Single block must retain its exact singleton model sample order")

    _METRICS["calls"] += 1
    outputs = [[] for _ in pipelines]
    pending = []
    for index, (pipeline, block) in enumerate(zip(pipelines, blocks)):
        if block is None:
            continue
        _METRICS["active_trait_blocks"] += 1
        state, ordinals[index] = pipeline._prepare_individual_block(
            block, ordinals[index], mac_cutoff=mac_cutoff)
        if state is None:
            continue
        prepared = pipeline._prepare_individual_trait(state, 0, mac_cutoff=mac_cutoff)
        if prepared is None:
            continue
        core = pipeline._compute_individual_core(prepared)
        # Preserve the existing per-model core; never retain all traits' G.
        del prepared["genotype"]
        score, variance = core["score"], core["variance"]
        if (score.dtype != torch.float32 or variance.dtype != torch.float32
                or score.ndim != 1 or variance.shape != score.shape
                or score.numel() != len(prepared["selected"])
                or score.device != device or variance.device != device):
            raise RuntimeError("native Single core must return matching FP32 Score/variance vectors")
        _METRICS["computed_trait_blocks"] += 1
        pending.append((index, pipeline, prepared, score, variance))
    if not pending:
        return outputs, ordinals

    profiler = pending[0][1].profiler
    with profiler.measure("phewas_individual_tail", gpu=True):
        score = torch.cat([item[3] for item in pending])
        variance = torch.cat([item[4] for item in pending])
        standard_error = torch.sqrt(variance.to(torch.float64))
        log_probabilities = _individual_log_probabilities(score, variance)
        values = torch.stack((variance, score, standard_error, log_probabilities), dim=1)
    _METRICS["pointwise_tail_batches"] += 1
    _METRICS["maximum_batch_variants"] = max(_METRICS["maximum_batch_variants"], score.numel())
    with profiler.measure("phewas_result_d2h", gpu=True):
        values_cpu = values.detach().cpu().numpy()
    _METRICS["result_transfer_batches"] += 1
    _METRICS["result_transfer_values"] += values.numel()
    offset = 0
    for index, pipeline, prepared, score, _ in pending:
        stop = offset + score.numel()
        outputs[index] = pipeline._individual_records_from_values(
            prepared, values_cpu[offset:stop], subset_variants_num=subset_variants_num)
        offset = stop
    return outputs, ordinals


_UNSUPPLIED_PROJECTION_STATE = object()


def _variance_from_shared_genotype(model, genotype, centered, mean, *,
                                   cross=None, projection_state=_UNSUPPLIED_PROJECTION_STATE):
    """Keep the existing finite-state Single variance, omitting its score GEMV.

    Scores use IEEE FP32 tiles or original vector products. Centering and
    the fitted P1 correction retain the mature per-model diagonal formula;
    neither weights nor covariance are borrowed from another phenotype.
    """
    weighted = projected = rotated = precision = None
    try:
        from ..tf32 import matmul
        if projection_state is _UNSUPPLIED_PROJECTION_STATE:
            state_method = getattr(model, '_centered_single_projection_state', None)
            state = None if state_method is None else state_method()
        else:
            state = projection_state
        if state is not None:
            precision = (model.inverse_variance if model.family == 'gaussian'
                         else model.precision)
            weighted = precision[:, None] * centered
            if cross is None:
                cross = matmul(model.precision_x.T, centered, mode='tf32')
            projected = matmul(cross.T, model.fixed_effect_covariance, mode='tf32')
            variance = ((centered * weighted).sum(dim=0)
                        - (projected * cross.T).sum(dim=1))
            del weighted, cross, projected
            p_one, p_one_sum = state
            variance += (2 * mean * matmul(centered.T, p_one, mode='tf32')
                         + mean.square() * p_one_sum)
            return variance
        if model.family == 'gaussian' and hasattr(model, 'spectrum'):
            rotated = model.spectrum.rotate(genotype, matmul_mode='tf32')
            weighted = model.inverse_variance[:, None] * rotated
            if cross is None:
                cross = matmul(model.precision_x.T, genotype, mode='tf32')
        elif model.family == 'binomial' and hasattr(model, '_fitted_projection'):
            if cross is None:
                weighted, cross = model._fitted_projection(genotype, 'tf32')
            else:
                precision = model.precision
                if precision is None:
                    precision = model.fitted_probability * (1-model.fitted_probability)
                precision = precision.to(dtype=genotype.dtype)
                if precision.ndim == 1:
                    weighted = precision[:, None] * genotype
                elif precision.layout != torch.strided:
                    weighted = torch.sparse.mm(precision, genotype)
                else:
                    weighted = matmul(precision, genotype, mode='tf32')
            rotated = genotype
        else:
            # Third-party models may expose only the validated single-trait API.
            # Count this original computation honestly rather than inventing a
            # projection from incomplete fitted state.
            return model.individual_score_variance(genotype, matmul_mode='tf32')[1]
        projected = matmul(cross.T, model.fixed_effect_covariance, mode='tf32')
        return ((rotated * weighted).sum(dim=0)
                - (projected * cross.T).sum(dim=1))
    finally:
        # A retained error traceback must not pin NEW cohort-sized buffers.
        weighted = projected = rotated = precision = cross = None


def _shared_projection_groups(models, genotype, centered):
    """Group only mature fitted cores with the same native product geometry."""
    from ..binary_null import BinaryNullModel
    from ..null_model import GaussianNullModel
    groups, states = {}, {}
    for index, model in enumerate(models):
        if not isinstance(model, (BinaryNullModel, GaussianNullModel)):
            continue
        state = model._centered_single_projection_state()
        sx = model.precision_x
        if sx is None and model.family == 'binomial':
            sx = model.xw.T
        if (not isinstance(sx, torch.Tensor) or sx.layout != torch.strided
                or sx.dtype != torch.float32 or sx.device != genotype.device):
            continue
        right = centered if state is not None else genotype
        key = (tuple(sx.T.shape), id(right))
        states[index] = state
        groups.setdefault(key, []).append((index, sx.T, right))
    return groups, states


def _batched_variances(models, genotype, centered, mean):
    """Reuse projection launches, preserving each original weighted/diag core.

    Only small trait-by-Q-by-M outputs are retained. Each group is used and
    released before the next group; admission includes the original one-model
    weighted, product and (for Gaussian) rotated genotype peak.
    """
    from .. import tf32
    groups, states = _shared_projection_groups(models, genotype, centered)
    result = [None] * len(models)
    for entries in groups.values():
        for begin in range(0, len(entries), 32):
            tile = entries[begin:begin+32]
            if len(tile) < 2:
                continue
            lefts, right = [entry[1] for entry in tile], tile[0][2]
            estimate = tf32._estimated_batched_workspace(lefts, [right]*len(tile), breakdown=True)
            n, m = genotype.shape
            original_peak = max((12 if models[index].family == 'gaussian' else 8)*n*m
                                for index, _, _ in tile)
            q = lefts[0].shape[0]
            required = dict(estimate, required_bytes=estimate['required_bytes']+original_peak+8*q*m+64*m)
            try:
                tf32._guard_workspace(genotype.device, required,
                    left_shape=(len(tile), *lefts[0].shape), right_shape=tuple(right.shape))
            except MemoryError:
                _METRICS['projection_budget_fallbacks'] += 1
                continue
            crosses = tf32.batched_matmul(lefts, right, mode='tf32')
            try:
                for offset, (index, _, _) in enumerate(tile):
                    result[index] = _variance_from_shared_genotype(models[index], genotype,
                        centered, mean, cross=crosses[offset], projection_state=states[index])
                _METRICS['projection_trait_batches'] += 1
                _METRICS['projection_batched_traits'] += len(tile)
            finally:
                del crosses
    for index, model in enumerate(models):
        if result[index] is None:
            result[index] = _variance_from_shared_genotype(model, genotype, centered, mean)
    return result


def _clear_normal_core_error(error):
    """Retain error/traceback identity without finished-frame tensor owners.

    Called only for the private normal capture mode. The active process and
    wrapper frames clear their aliases explicitly; ended arithmetic frames
    can be cleared without removing traceback entries or altering the error.
    """
    import traceback
    pending, seen = [error], set()
    try:
        while pending:
            item = pending.pop()
            if id(item) in seen:
                continue
            seen.add(id(item))
            traceback.clear_frames(item.__traceback__)
            if item.__cause__ is not None:
                pending.append(item.__cause__)
            if item.__context__ is not None:
                pending.append(item.__context__)
    finally:
        error = pending = item = None


def _shared_single_normal_statistics(pipelines, genotype):
    """Capture the single original complete-tile normal producer, without rows.

    The process below owns the only mathematical implementation. All original
    traits and physical-request columns reach its FP32/TF32 core and extreme
    tail GEMV fallback; this wrapper neither copies nor recomputes results.
    """
    prepared_state = ({'genotype': genotype}, 0)
    try:
        return process_single_shared_genotype(pipelines, None, 0, None,
            prepared_state=prepared_state, _return_normal_core=True)
    finally:
        pipelines = genotype = prepared_state = None


def process_single_shared_genotype(pipelines, block, ordinal_offset, chromosome,
                                  mac_cutoff=20, subset_variants_num=5000,
                                  *, prepared_state=None, binary_correction=None, normal_recheck=None,
                                  _return_normal_core=False):
    """Compute a bounded phenotype tile while retaining one exact-cohort G.

    All pipelines must use precisely the same ordered sample axis and genotype
    preprocessing. Scores use G.T and original fitted FP32 residuals. A
    singleton retains the independent native FP32 GEMV; wider tiles use
    guarded IEEE FP32 SIMT products, with vector fallback under pressure.
    Genotype storage is shared; fitted variances retain their original
    per-trait TF32 projection and finite-state correction.
    Returns rows per supplied phenotype and the shared new eligible ordinal.
    """
    try:
        if type(_return_normal_core) is not bool:
            raise TypeError('_return_normal_core must be a boolean')
        if _return_normal_core and (normal_recheck is not None or binary_correction is not None):
            raise ValueError('normal core capture cannot use recheck or binary correction')
        pipelines = list(pipelines)
        if not pipelines:
            raise ValueError('a nonempty phenotype tile is required')
        ordinal_offset = _integer(ordinal_offset, 'ordinal_offset')
        subset_variants_num = _integer(subset_variants_num, 'subset_variants_num', positive=True)
        if normal_recheck is not None and not callable(normal_recheck):
            raise TypeError('normal_recheck must be callable or None')
        first = pipelines[0]
        first_binding = getattr(first, '_single_axis_binding', None)
        models = []
        for pipeline in pipelines:
            if len(pipeline.models) != 1 or len(pipeline.trait_rows) != 1:
                raise ValueError('shared genotype requires singleton aligned sample axes')
            binding = getattr(pipeline, '_single_axis_binding', None)
            if binding is not None or first_binding is not None:
                if (not isinstance(binding, _SingleAxisBinding)
                        or not isinstance(first_binding, _SingleAxisBinding)
                        or binding.axis is not first_binding.axis
                        or binding.broker is not first_binding.broker):
                    raise ValueError('shared genotype requires the same verified sample-axis binding')
                binding.validate(pipeline.genotype, pipeline.union_rows, pipeline.trait_rows[0], pipeline.models[0].n)
                aligned_rows = True
            else:
                # Independent public callers retain the complete value/order check.
                aligned_rows = np.array_equal(pipeline.trait_rows[0], np.arange(pipeline.models[0].n))
            if (len(pipeline.models) != 1 or pipeline.options.wrapper_semantics != 'base'
                    or (binding is None and not np.array_equal(pipeline.union_rows, first.union_rows))
                    or pipeline.options.imputation != first.options.imputation):
                raise ValueError('shared genotype requires identical ordered cohorts and imputation')
            model = pipeline.models[0]
            if (model.n_pheno != 1 or model.use_spa or model.family not in ('gaussian', 'binomial')
                    or getattr(model, 'matmul_mode', None) != 'tf32'
                    or model.x.dtype != torch.float32 or model.n != len(first.union_rows)
                    or torch.device(model.device) != torch.device(first.models[0].device)
                    or not aligned_rows):
                raise ValueError('shared genotype requires aligned native non-SPA singleton models')
            residual = model.scaled_residuals
            if residual.shape != (model.n,) or residual.dtype != torch.float32 or residual.device != model.x.device:
                raise ValueError('fitted residual must be aligned native FP32')
            models.append(model)
        if prepared_state is None:
            state, new_ordinal = first._prepare_individual_block(block, ordinal_offset, mac_cutoff=mac_cutoff)
            if state is None:
                return [[] for _ in pipelines], new_ordinal
            prepared = first._prepare_individual_trait(state, 0, mac_cutoff=mac_cutoff)
        else:
            prepared, new_ordinal = prepared_state
        if prepared is None:
            return [[] for _ in pipelines], new_ordinal
        genotype = prepared['genotype']
        if genotype.dtype != torch.float32 or genotype.shape[0] != models[0].n:
            raise RuntimeError('shared dosage must retain the native cohort shape and dtype')
        before_score = score_execution_metadata()
        with first.profiler.measure('phewas_phenotype_score_fp32', gpu=True):
            scores = fp32_score_columns(genotype, [model.scaled_residuals for model in models])
        after_score = score_execution_metadata()
        _METRICS['phenotype_score_batches'] += 1
        _METRICS['phenotype_score_fp32_vector_calls'] += (
            after_score['fp32_vector_products']-before_score['fp32_vector_products'])
        _METRICS['phenotype_score_fp32_simt_gemm_calls'] += (
            after_score['fp32_simt_products']-before_score['fp32_simt_products'])
        _METRICS['phenotype_score_simt_admission_fallbacks'] += (
            after_score['simt_admission_fallbacks']-before_score['simt_admission_fallbacks'])
        _METRICS['shared_genotype_batches'] += 1
        _METRICS['shared_genotype_traits'] += len(models)
        mean = genotype.mean(dim=0)
        centered = genotype - mean[None, :]
        variances = []
        if genotype.is_cuda and len(models) > 1:
            with first.profiler.measure('phewas_individual_variance', gpu=True):
                variances = _batched_variances(models, genotype, centered, mean)
            _METRICS['original_variance_calls'] += len(models)
        else:
            for index, model in enumerate(models):
                with first.profiler.measure('phewas_individual_variance', gpu=True):
                    variance = _variance_from_shared_genotype(model, genotype, centered, mean)
                variances.append(variance)
                _METRICS['original_variance_calls'] += 1
        del centered, mean
        # IEEE matrix and vector products have different FP32 reduction orders.
        # Extremely sensitive normal tails must retain the mature entire-column
        # GEMV; selecting only variant rows would change that original reduction
        # shape. This small control copy is separate from final result transfer.
        if after_score['fp32_simt_products'] > before_score['fp32_simt_products']:
            with first.profiler.measure('phewas_score_sensitivity', gpu=True):
                trial_variance = torch.stack(variances)
                trial_logp = _individual_log_probabilities(scores.T, trial_variance)
                sensitive = ((torch.isfinite(trial_variance) & (trial_variance > 0))
                             & (trial_logp > 100.*np.log(10.))).any(dim=1)
                flags = sensitive.detach().cpu().tolist()
            if scores.is_cuda:
                _METRICS['phenotype_score_control_d2h_batches'] += 1
                _METRICS['phenotype_score_control_d2h_bytes'] += sensitive.numel()*sensitive.element_size()
            del trial_variance, trial_logp, sensitive
            for index, flag in enumerate(flags):
                if flag:
                    with first.profiler.measure('phewas_sensitive_score_vector', gpu=True):
                        native = fp32_score_columns(genotype, [models[index].scaled_residuals])
                        scores[:, index].copy_(native[:, 0])
                    del native
                    _METRICS['phenotype_score_batches'] += 1
                    _METRICS['phenotype_score_fp32_vector_calls'] += 1
                    _METRICS['phenotype_score_extreme_tail_fallback_columns'] += 1
            del flags
        if _return_normal_core:
            return scores, variances
        if normal_recheck is not None:
            normal_recheck(pipelines, prepared, scores, variances)
        correction_values, corrected = [], {}
        correction_fields = ('pvalues', 'pvalue_log10', 'pvalue_log', 'normal_pvalues',
                             'normal_pvalue_log10', 'spa_selected', 'spa_zero_log_unverified')
        flag_fields = ('used_bisection', 'failed', 'iterations', 'iteration_limit')
        for index, model in enumerate(models):
            score, variance = scores[:, index], variances[index]
            if model.family == 'binomial' and binary_correction is not None:
                # The callback owns the bounded FP64 sidecar lifetime. Its formal
                # corrected P/logP replace the output; normal scores remain intact.
                correction = binary_correction(index, model, genotype, score, variance)
                if correction.get('normal_only', True):
                    raise RuntimeError('formal binary Single requires the fitted FP64 SPA sidecar')
                fields = [correction[name] for name in correction_fields]
                fields += [getattr(correction['spa'], name) for name in flag_fields]
                if any(value.shape != score.shape or value.device != score.device for value in fields):
                    raise RuntimeError('binary Single correction must retain all original columns/device')
                corrected[index] = len(correction_values)
                # Continuous traits have no SPA columns. Retain diagnostics only
                # for corrected traits instead of allocating eleven zero columns
                # per continuous trait merely to discard them after the host copy.
                correction_values.append(torch.stack(fields, dim=1))
                del correction, fields
        with first.profiler.measure('phewas_individual_tail', gpu=True):
            # These operations are pointwise: widening the tile changes neither
            # per-model reductions nor the mature finite normal-tail expression.
            variance = torch.stack(variances)
            logp = _individual_log_probabilities(scores.T, variance)
            values = torch.stack((variance, scores.T, torch.sqrt(variance.to(torch.float64)), logp), dim=2)
            core_size = values.numel()
            packed = values.reshape(-1)
            if correction_values:
                packed = torch.cat((packed, torch.stack(correction_values).reshape(-1)))
        del variances, correction_values, variance, score, logp, values
        with first.profiler.measure('phewas_result_d2h', gpu=True):
            host = packed.detach().cpu().numpy()
        del packed
        core_host = host[:core_size].reshape(len(models), scores.shape[0], 4)
        correction_host = host[core_size:].reshape(len(corrected), scores.shape[0],
                                                 len(correction_fields)+len(flag_fields))
        _METRICS['result_transfer_batches'] += 1
        _METRICS['result_transfer_values'] += host.size
        _METRICS['pointwise_tail_batches'] += 1
        _METRICS['maximum_batch_variants'] = max(_METRICS['maximum_batch_variants'], scores.numel())
        from .._single_record_metadata import SingleRecordMetadata
        if '_single_record_metadata' not in prepared:
            prepared['_single_record_metadata'] = SingleRecordMetadata.prepare(
                first.genotype, prepared['selected'])
        result = []
        for index, (pipeline, model) in enumerate(zip(pipelines, models)):
            record_state = dict(prepared, model=model)
            record_state.pop('genotype')
            if pipeline.genotype is not first.genotype:
                # Independent callers may use separate readers on equal sample
                # axes. Keep their own labels; only one exact reader shares IO.
                record_state.pop('_single_record_metadata', None)
            rows = pipeline._individual_records_from_values(record_state, core_host[index],
                subset_variants_num=subset_variants_num)
            if index in corrected:
                arrays = {name: correction_host[corrected[index], :, j] for j, name in
                          enumerate((*correction_fields, *flag_fields))}
                for j, row in enumerate(rows):
                    row.update(pvalue=float(arrays['pvalues'][j]),
                        pvalue_log10=float(arrays['pvalue_log10'][j]),
                        pvalue_log=float(arrays['pvalue_log'][j]),
                        normal_pvalue=float(arrays['normal_pvalues'][j]),
                        normal_pvalue_log10=float(arrays['normal_pvalue_log10'][j]),
                        spa_selected=bool(arrays['spa_selected'][j]),
                        spa_failed=bool(arrays['failed'][j]),
                        spa_used_bisection=bool(arrays['used_bisection'][j]),
                        spa_iterations=int(arrays['iterations'][j]),
                        spa_iteration_limit=bool(arrays['iteration_limit'][j]),
                        spa_zero_log_unverified=bool(arrays['spa_zero_log_unverified'][j]))
            result.append(rows)
        return result, new_ordinal
    except BaseException as error:
        if _return_normal_core is True:
            # A held exception must not keep the legacy prepared mapping,
            # raw/centered G, score buffers or model/pipeline aliases alive.
            prepared_state = prepared = genotype = state = block = None
            mean = centered = variance = variances = scores = native = None
            trial_variance = trial_logp = sensitive = flags = None
            pipelines = first = first_binding = pipeline = model = models = residual = None
            _clear_normal_core_error(error)
        raise
