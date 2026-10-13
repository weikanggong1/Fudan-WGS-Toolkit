"""Diagonal formula/admission contracts; synthetic arrays are not benchmarks."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit._binary_burden_diag import diagonal_burden_products
from fudan_wgs_toolkit.binary_null import (
    binary_burden_diag_workspace, binary_burden_score_covariance_batch,
    binary_burden_score_variance_batch, binary_prefitted_state)


def models_and_burdens(n=1031, b=9, t=3, q=4, *, device='cpu', mode='fp64', layout='c'):
    rng = np.random.default_rng(811+n+b+q)
    x = np.column_stack((np.ones(n), rng.normal(size=(n, q-1))))
    z = torch.tensor(rng.normal(size=(n, b)), dtype=torch.float32, device=device)
    if layout == 'f':
        z = z.T.contiguous().T
    models = []
    for trait in range(t):
        precision = rng.uniform(.08, .24, n)
        sx = precision[:, None]*x
        cov = np.linalg.inv(x.T@sx)*(.93+.01*trait)
        probability = rng.uniform(.1, .5, n)
        residual = (rng.binomial(1, probability)-probability)*(.9+.02*trait)
        models.append(binary_prefitted_state(
            sample_ids=np.arange(n).astype(str), covariates=x, residual=residual,
            fitted_probability=probability, xw=sx.T, projection_left=x@cov,
            fixed_effect_covariance=cov, precision=precision,
            precision_covariates=sx, use_spa=False, device=device, matmul_mode=mode))
    return models, z


@pytest.mark.parametrize('chunk', [1, 37, 256, 8192])
@pytest.mark.parametrize('layout', ['c', 'f'])
@pytest.mark.parametrize('precision_none', [False, True])
def test_cpu_oracle_uses_each_supplied_state_and_original_diagonal_formula(chunk, layout, precision_none):
    models, z = models_and_burdens(mode='tf32', layout=layout)
    if precision_none:
        for model in models:
            model.precision = None
            model.precision_x = None
    before = [model.fixed_effect_covariance.clone() for model in models]
    scores, variances, execution = diagonal_burden_products(models, z, sample_chunk_size=chunk)
    assert scores.shape == variances.shape == (len(models), z.shape[1])
    assert scores.dtype == variances.dtype == torch.float32
    assert execution['cuda_diagonal_kernel_launches'] == 0
    assert execution['cpu_diagonal_oracle_batches'] == 1
    assert execution['sample_chunks'] == (len(z)+chunk-1)//chunk
    for index, model in enumerate(models):
        precision = (model.precision if model.precision is not None
                     else model.fitted_probability*(1-model.fitted_probability))
        sx = model.precision_x if model.precision_x is not None else model.xw.T
        cross = sx.T@z
        expected = (z*(precision[:, None]*z)).sum(0)-(
            (cross.T@model.fixed_effect_covariance)*cross.T).sum(1)
        score = z.T@model.scaled_residuals
        torch.testing.assert_close(scores[index], score, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(variances[index], expected, rtol=3e-6, atol=8e-5)
        torch.testing.assert_close(model.fixed_effect_covariance, before[index], rtol=0, atol=0)


@pytest.mark.parametrize('precision_kind', ['diagonal', 'sparse', 'dense', 'none'])
def test_public_cpu_fp64_fallback_is_identical_to_full_covariance(precision_kind):
    models, z = models_and_burdens()
    z = z.double()
    for model in models:
        if precision_kind == 'sparse':
            model.precision = torch.diag(model.precision).to_sparse()
        elif precision_kind == 'dense':
            model.precision = torch.diag(model.precision)
        elif precision_kind == 'none':
            model.precision = None
    score, covariance, _ = binary_burden_score_covariance_batch(models, z)
    actual_score, actual_variance, execution = binary_burden_score_variance_batch(
        models, z, sample_chunk_size=37)
    torch.testing.assert_close(actual_score, score, rtol=0, atol=0)
    torch.testing.assert_close(actual_variance, covariance.diagonal(dim1=-2, dim2=-1), rtol=0, atol=0)
    assert execution['native_diagonal_batches'] == 0
    assert execution['native_projection_batches'] == 0


def test_workspace_is_sample_partial_storage_not_full_trait_sample_weight_slab():
    n, b, q, t, chunk = 343034, 26, 24, 32, 8192
    estimate = binary_burden_diag_workspace(n, b, q, t, sample_chunk_size=chunk)
    parts = (n+chunk-1)//chunk
    actual_parts_bytes = 4*parts*t*b*(2+q)
    assert estimate >= actual_parts_bytes+256*2**20
    assert estimate < 4*n*b*t
    assert binary_burden_diag_workspace(n, b, q, 799) == estimate


@pytest.mark.parametrize('chunk', [0, -1, True, 4.0])
def test_invalid_chunk_is_rejected_before_compute(chunk):
    models, z = models_and_burdens(n=31)
    with pytest.raises(ValueError, match='sample_chunk_size'):
        binary_burden_score_variance_batch(models, z, sample_chunk_size=chunk)
    with pytest.raises(ValueError, match='sample_chunk_size'):
        diagonal_burden_products(models, z.float(), sample_chunk_size=chunk)


@pytest.mark.parametrize('count', [0, 33])
def test_helper_limits_traits_before_any_partial_tensor(count, monkeypatch):
    models, z = models_and_burdens(n=31, t=max(1, count), mode='tf32')
    def forbidden(*args, **kwargs):
        pytest.fail('invalid trait tile allocated partial storage')
    monkeypatch.setattr(torch, 'empty', forbidden)
    with pytest.raises(ValueError, match='between 1 and 32'):
        diagonal_burden_products(models[:count], z)


def test_helper_rejects_precision_matrix_and_gradients_without_interpretation():
    models, z = models_and_burdens(n=31, mode='tf32')
    models[0].precision = torch.diag(models[0].precision)
    with pytest.raises(ValueError, match='aligned independent fitted FP32 diagonal states'):
        diagonal_burden_products(models, z)
    models, z = models_and_burdens(n=31, mode='tf32')
    with pytest.raises(ValueError, match='without gradients'):
        diagonal_burden_products(models, z.requires_grad_())


@pytest.mark.parametrize('exponent', [-4, -12, -25])
def test_tiny_diagonal_precision_is_not_clipped_or_replaced(exponent):
    models, z = models_and_burdens(n=1027, b=3, t=2, q=2, mode='tf32')
    scale = 10.**exponent
    for model in models:
        model.precision.mul_(scale)
        model.precision_x.mul_(scale)
        model.fixed_effect_covariance.div_(scale)
    scores, variances, _ = diagonal_burden_products(models, z, sample_chunk_size=257)
    for index, model in enumerate(models):
        cross = model.precision_x.T@z
        expected = (z*(model.precision[:, None]*z)).sum(0)-(
            (cross.T@model.fixed_effect_covariance)*cross.T).sum(-1)
        assert bool((variances[index] > 0).all())
        torch.testing.assert_close(variances[index], expected, rtol=3e-6, atol=scale*8e-5)
        torch.testing.assert_close(scores[index], z.T@model.scaled_residuals, rtol=2e-5, atol=2e-5)


def test_noncontiguous_fitted_states_keep_actual_strides_and_storage_offsets():
    from fudan_wgs_toolkit._binary_burden_diag import _descriptor_rows
    models, z = models_and_burdens(n=1027, mode='tf32', layout='f')
    for model in models:
        for name in ('precision', 'precision_x', 'scaled_residuals'):
            value = getattr(model, name)
            shape = (2*len(value)+3, *value.shape[1:])
            store = torch.empty(shape, dtype=value.dtype)
            view = store[1:2*len(value)+1:2]
            view.copy_(value)
            setattr(model, name, view)
    descriptors = _descriptor_rows(models)
    for index, model in enumerate(models):
        assert descriptors[0][index] == model.precision.data_ptr()
        assert descriptors[1][index] == 2
        assert descriptors[3][index] == model.scaled_residuals.data_ptr()
        assert descriptors[4][index] == 2
        assert descriptors[5][index] == model.precision_x.data_ptr()
        assert descriptors[6][index] == 2*model.x.shape[1]
    scores, variances, _ = diagonal_burden_products(models, z, sample_chunk_size=257)
    for index, model in enumerate(models):
        cross = model.precision_x.T@z
        expected = (z*(model.precision[:, None]*z)).sum(0)-(
            (cross.T@model.fixed_effect_covariance)*cross.T).sum(-1)
        torch.testing.assert_close(variances[index], expected, rtol=3e-6, atol=8e-5)
        torch.testing.assert_close(scores[index], z.T@model.scaled_residuals, rtol=2e-5, atol=2e-5)


def test_long_sample_axis_uses_bounded_fp32_partials():
    models, z = models_and_burdens(n=343034, b=3, t=2, q=2, mode='tf32')
    scores, variances, execution = diagonal_burden_products(models, z, sample_chunk_size=8192)
    assert execution['sample_chunks'] == 42
    for index, model in enumerate(models):
        cross = model.precision_x.T@z
        expected = (z*(model.precision[:, None]*z)).sum(0)-(
            (cross.T@model.fixed_effect_covariance)*cross.T).sum(-1)
        torch.testing.assert_close(variances[index], expected, rtol=4e-6, atol=.025)
        torch.testing.assert_close(scores[index], z.T@model.scaled_residuals, rtol=4e-5, atol=.006)


@pytest.mark.parametrize('n,b,q', [(0, 9, 4), (31, 0, 4), (31, 9, 0)])
def test_cpu_helper_handles_empty_sample_weight_or_covariate_dimensions(n, b, q):
    model = SimpleNamespace(n=n, x=torch.zeros((n, q)),
        precision=torch.ones(n), precision_x=torch.zeros((n, q)),
        scaled_residuals=torch.ones(n), fixed_effect_covariance=torch.zeros((q, q)))
    z = torch.ones((n, b))
    score, variance, _ = diagonal_burden_products([model], z, sample_chunk_size=17)
    torch.testing.assert_close(score, torch.full((1, b), float(n)))
    torch.testing.assert_close(variance, torch.full((1, b), float(n)))


def sparse_models_and_burdens(n=1031, b=9, t=3, q=4, *, device='cpu',
                             layout='c', sparse_kind='pairs14', strided=False):
    """Independent supplied COO states, with every original edge retained."""
    models, z = models_and_burdens(n=n, b=b, t=t, q=q, device=device,
                                  mode='tf32', layout=layout)
    for index, model in enumerate(models):
        diagonal = torch.arange(n, device=device)
        if sparse_kind == 'pairs14':
            left = torch.arange(7, device=device)*(n//9)
            right = left+1
            edge_values = torch.full((7,), .011+.003*index, device=device)
        elif sparse_kind == 'graph':
            # Traits deliberately have different NNZ counts and signed edges.
            left = torch.arange(max(1, n//(index+2)), device=device)
            right = (left*13+17) % n
            distinct = left != right
            left, right = left[distinct], right[distinct]
            edge_values = (.004+.001*index)*torch.where(
                left % 2 == 0, 1., -1.)
        elif sparse_kind == 'empty':
            left = right = diagonal[:0]
            edge_values = model.precision[:0]
        else:
            raise AssertionError(sparse_kind)
        rows = torch.cat((diagonal, left, right))
        columns = torch.cat((diagonal, right, left))
        values = torch.cat((model.precision, edge_values, edge_values))
        precision = torch.sparse_coo_tensor(torch.stack((rows, columns)), values,
            size=(n, n), device=device).coalesce()
        if strided:
            edges = precision._nnz()
            indices_store = torch.empty((5, 2*edges+3), dtype=torch.int64, device=device)
            indices = indices_store[1:5:2, 1:2*edges+1:2]
            indices.copy_(precision.indices())
            values_store = torch.empty(3*edges+2, dtype=torch.float32, device=device)
            values = values_store[1:3*edges+1:3]
            values.copy_(precision.values())
            precision = torch.sparse_coo_tensor(indices, values, (n, n),
                is_coalesced=True, device=device)
            assert precision.indices().stride() == indices.stride()
            assert precision.values().stride() == values.stride()
        model.precision = precision
        model.precision_x = torch.sparse.mm(precision, model.x)
        model.fixed_effect_covariance = torch.linalg.inv(model.x.T@model.precision_x)*(.93+.01*index)
    return models, z


def sparse_expected(model, z):
    """Independent FP64 formula oracle, including every signed COO entry."""
    host_z = z.cpu().double()
    host_precision = model.precision.cpu().double()
    host_cross = model.precision_x.cpu().double().T@host_z
    variance = (host_z*torch.sparse.mm(host_precision, host_z)).sum(0)-(
        (host_cross.T@model.fixed_effect_covariance.cpu().double())*host_cross.T).sum(-1)
    score = host_z.T@model.scaled_residuals.cpu().double()
    return score, variance


@pytest.mark.parametrize('kind', ['pairs14', 'graph'])
@pytest.mark.parametrize('layout', ['c', 'f'])
@pytest.mark.parametrize('chunk', [37, 257, 8192])
def test_cpu_coo_oracle_matches_full_independent_fp64_formula(kind, layout, chunk):
    models, z = sparse_models_and_burdens(layout=layout, sparse_kind=kind)
    before = [(model.precision.indices().clone(), model.precision.values().clone())
              for model in models]
    scores, variances, execution = diagonal_burden_products(models, z, sample_chunk_size=chunk)
    assert execution['sparse_precision_products'] == len(models)
    assert execution['sparse_nnz_terms'] == sum(model.precision._nnz() for model in models)
    assert execution['sparse_nnz_chunks'] == (max(model.precision._nnz() for model in models)+chunk-1)//chunk
    for index, model in enumerate(models):
        score, variance = sparse_expected(model, z)
        torch.testing.assert_close(scores[index].double(), score, rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(variances[index].double(), variance, rtol=3e-6, atol=8e-5)
        torch.testing.assert_close(model.precision.indices(), before[index][0], rtol=0, atol=0)
        torch.testing.assert_close(model.precision.values(), before[index][1], rtol=0, atol=0)
    if kind == 'pairs14':
        assert all(model.precision._nnz() == len(z)+14 for model in models)


def test_cpu_coo_offdiagonal_is_never_dropped_and_mixed_states_are_supported():
    models, z = sparse_models_and_burdens(n=61, b=3, t=2, q=2)
    # A deterministic weight makes the 14 off-diagonal terms substantial.
    z.fill_(1.)
    models[1].precision = models[1].precision.to_dense().diagonal().clone()
    score, variance, execution = diagonal_burden_products(models, z, sample_chunk_size=17)
    assert execution['sparse_precision_products'] == 1
    model = models[0]
    expected_score, expected_variance = sparse_expected(model, z)
    cross = model.precision_x.double().T@z.double()
    correction = ((cross.T@model.fixed_effect_covariance.double())*cross.T).sum(-1)
    diagonal_only = (z.double().square()*model.precision.to_dense().diagonal().double()[:, None]).sum(0)-correction
    assert bool(((expected_variance-diagonal_only).abs() > .1).all())
    torch.testing.assert_close(variance[0].double(), expected_variance, rtol=3e-6, atol=1e-6)
    torch.testing.assert_close(score[0].double(), expected_score, rtol=3e-6, atol=1e-6)


@pytest.mark.parametrize('exponent', [-4, -12, -25])
def test_cpu_coo_tiny_precision_and_strides_are_preserved(exponent):
    from fudan_wgs_toolkit._binary_burden_diag import _descriptor_rows
    models, z = sparse_models_and_burdens(n=1027, b=3, t=2, q=2, layout='f', strided=True)
    scale = 10.**exponent
    for model in models:
        model.precision.values().mul_(scale)
        model.precision_x.mul_(scale)
        model.fixed_effect_covariance.div_(scale)
    descriptors = _descriptor_rows(models)
    for index, model in enumerate(models):
        assert descriptors[0][index] == model.precision.values().data_ptr()
        assert descriptors[1][index] == 3
        assert descriptors[2][index] == 2
        assert descriptors[8][index] == model.precision.indices().data_ptr()
        assert descriptors[9][index] == model.precision.indices().stride(0)
        assert descriptors[10][index] == 2
        assert descriptors[11][index] == len(z)+14
    score, variance, _ = diagonal_burden_products(models, z, sample_chunk_size=257)
    for index, model in enumerate(models):
        expected_score, expected_variance = sparse_expected(model, z)
        assert bool((variance[index] > 0).all())
        torch.testing.assert_close(variance[index].double(), expected_variance, rtol=3e-6, atol=scale*8e-5)
        torch.testing.assert_close(score[index].double(), expected_score, rtol=3e-5, atol=3e-5)


def test_coo_invalid_geometry_or_bounds_is_rejected_before_partials(monkeypatch):
    models, z = sparse_models_and_burdens(n=31, b=3, t=1, q=2)
    values = models[0].precision.values()
    indices = models[0].precision.indices().clone()
    indices[0, -1] = len(z)
    models[0].precision = torch.sparse_coo_tensor(indices, values, (len(z), len(z)),
        is_coalesced=True, check_invariants=False)
    def forbidden(*args, **kwargs):
        pytest.fail('invalid sparse pointer geometry allocated partial storage')
    monkeypatch.setattr(torch, 'empty', forbidden)
    with pytest.raises(ValueError, match='outside the sample axis'):
        diagonal_burden_products(models, z)


def test_coo_uncoalesced_or_csr_retains_mature_public_cpu_fallback():
    models, z = sparse_models_and_burdens(n=31, b=3, t=1, q=2)
    original = models[0].precision
    models[0].precision = torch.sparse_coo_tensor(original.indices(), original.values(), original.shape)
    with pytest.raises(ValueError, match='coalesced FP32 COO'):
        diagonal_burden_products(models, z)
    models[0].precision = original.to_sparse_csr()
    with pytest.raises(ValueError, match='aligned independent fitted FP32 diagonal states'):
        diagonal_burden_products(models, z)
    models[0].matmul_mode = 'fp64'
    z = z.double()
    score, variance, execution = binary_burden_score_variance_batch(models, z)
    old_score, covariance, _ = binary_burden_score_covariance_batch(models, z)
    assert execution['native_diagonal_batches'] == 0
    torch.testing.assert_close(score, old_score, rtol=0, atol=0)
    torch.testing.assert_close(variance, covariance.diagonal(dim1=-2, dim2=-1), rtol=0, atol=0)


def test_sparse_workspace_accounts_for_nnz_partials_and_pointer_table():
    n, b, q, t, chunk, nnz = 343034, 26, 16, 32, 8192, 343048
    dense = binary_burden_diag_workspace(n, b, q, t, sample_chunk_size=chunk)
    sparse = binary_burden_diag_workspace(n, b, q, t, sample_chunk_size=chunk, maximum_nnz=nnz)
    assert sparse-dense == 4*((nnz+chunk-1)//chunk)*t*b
    assert sparse < 4*n*t*b
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError, match='maximum_nnz'):
            binary_burden_diag_workspace(n, b, q, t, maximum_nnz=invalid)


CUDA_NATIVE = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


@pytest.mark.skipif(not CUDA_NATIVE, reason='requires actual Ampere CUDA TF32 backend')
@pytest.mark.parametrize('layout', ['c', 'f'])
@pytest.mark.parametrize('chunk', [37, 2048, 8192])
def test_cuda_diagonal_kernel_matches_independent_fp32_formula(layout, chunk):
    models, z = models_and_burdens(n=5037, b=26, t=3, q=24,
                                 device='cuda', mode='tf32', layout=layout)
    score, variance, execution = binary_burden_score_variance_batch(models, z,
        sample_chunk_size=chunk)
    assert execution['native_diagonal_batches'] == 1
    assert execution['native_projection_batches'] == 1
    assert execution['cuda_diagonal_kernel_launches'] == 3
    assert execution['ieee_covariance_projection_batches'] == 1
    assert execution['cpu_diagonal_oracle_batches'] == 0
    for index, model in enumerate(models):
        host_z = z.cpu()
        host_cross = model.precision_x.cpu().T@host_z
        expected_variance = (host_z*(model.precision.cpu()[:, None]*host_z)).sum(0)-(
            (host_cross.T@model.fixed_effect_covariance.cpu())*host_cross.T).sum(-1)
        expected_score = host_z.T@model.scaled_residuals.cpu()
        torch.testing.assert_close(score[index].cpu(), expected_score, rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(variance[index].cpu(), expected_variance, rtol=2e-5, atol=1e-4)


@pytest.mark.skipif(not CUDA_NATIVE, reason='requires actual Ampere CUDA TF32 backend')
def test_cuda_diagonal_workspace_rejects_before_any_partial_allocation(monkeypatch):
    from fudan_wgs_toolkit import binary_null
    models, z = models_and_burdens(n=5037, b=26, t=3, device='cuda', mode='tf32')
    observed = []
    def reject(value, required):
        assert value is z
        observed.append(required)
        raise MemoryError('rejected diagonal optional partials')
    monkeypatch.setattr(binary_null, '_guard_binary_burden_diag_workspace', reject)
    def forbidden(*args, **kwargs):
        pytest.fail('rejected partials entered diagonal implementation')
    from fudan_wgs_toolkit import _binary_burden_diag
    monkeypatch.setattr(_binary_burden_diag, 'diagonal_burden_products', forbidden)
    with pytest.raises(MemoryError, match='accurate independent binary diagonal workspace'):
        binary_burden_score_variance_batch(models, z)
    assert len(observed) == 2
    assert observed[1] < observed[0]


@pytest.mark.skipif(not CUDA_NATIVE, reason='requires actual Ampere CUDA TF32 backend')
@pytest.mark.parametrize('kind', ['pairs14', 'graph'])
@pytest.mark.parametrize('layout', ['c', 'f'])
@pytest.mark.parametrize('strided', [False, True])
def test_cuda_coo_native_kernel_matches_independent_fp64_formula(kind, layout, strided):
    models, z = sparse_models_and_burdens(n=5037, b=26, t=3, q=16,
        device='cuda', sparse_kind=kind, layout=layout, strided=strided)
    score, variance, execution = binary_burden_score_variance_batch(models, z, sample_chunk_size=2048)
    assert execution['native_diagonal_batches'] == 1
    assert execution['native_sparse_diagonal_batches'] == 1
    assert execution['native_projection_batches'] == 1
    assert execution['sparse_precision_products'] == len(models)
    assert execution['sparse_nnz_terms'] == sum(model.precision._nnz() for model in models)
    assert execution['cuda_diagonal_kernel_launches'] == 4
    assert execution['ieee_covariance_projection_batches'] == 1
    assert execution['cpu_diagonal_oracle_batches'] == 0
    for index, model in enumerate(models):
        expected_score, expected_variance = sparse_expected(model, z)
        torch.testing.assert_close(score[index].cpu().double(), expected_score, rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(variance[index].cpu().double(), expected_variance, rtol=2e-5, atol=1e-4)


@pytest.mark.skipif(not CUDA_NATIVE, reason='requires actual Ampere CUDA TF32 backend')
def test_cuda_coo_rejected_workspace_does_not_enter_native_helper(monkeypatch):
    from fudan_wgs_toolkit import binary_null, _binary_burden_diag
    models, z = sparse_models_and_burdens(n=5037, b=26, t=3, q=16, device='cuda')
    requests = []
    def reject(value, required):
        requests.append(required)
        raise MemoryError('reject sparse NNZ-split optional workspace')
    monkeypatch.setattr(binary_null, '_guard_binary_burden_diag_workspace', reject)
    monkeypatch.setattr(_binary_burden_diag, 'diagonal_burden_products',
        lambda *args, **kwargs: pytest.fail('rejected sparse path allocated partials'))
    with pytest.raises(MemoryError, match='accurate independent binary diagonal workspace'):
        binary_burden_score_variance_batch(models, z)
    assert len(requests) == 2
    assert requests[0] >= binary_burden_diag_workspace(len(z), z.shape[1], 16, 3,
        maximum_nnz=len(z)+14)
    assert requests[1] < requests[0]


def test_small_ieee_covariance_retains_cancellation_erased_by_tf32_rounding():
    from fudan_wgs_toolkit._binary_burden_diag import _ieee_covariance_projection
    from fudan_wgs_toolkit.statistics import _chi1_from_score
    covariance = torch.tensor([[1., -.9999], [-.9999, 1.]], dtype=torch.float32)
    cross = torch.tensor([[[128., 64.], [128., 64.]]], dtype=torch.float32)
    models = [SimpleNamespace(fixed_effect_covariance=covariance)]
    projected = _ieee_covariance_projection(cross, models)
    correction = (projected*cross.transpose(1, 2)).sum(-1)
    expected = ((cross.double().transpose(1, 2)@covariance.double())
                *cross.double().transpose(1, 2)).sum(-1)
    torch.testing.assert_close(correction.double(), expected, rtol=0, atol=0)
    # These entries round to exactly +/-1 in TF32, erasing this supplied
    # positive quadratic form. The long-H entries are powers of two and
    # representable in either policy, so the fixture isolates only C.
    rounded_covariance = torch.tensor([[1., -1.], [-1., 1.]])
    rounded_correction = ((cross.transpose(1, 2)@rounded_covariance)
                          *cross.transpose(1, 2)).sum(-1)
    assert bool((rounded_correction == 0).all())
    base, score = torch.tensor([[10., 2.5]]), torch.tensor([[3., 1.5]])
    actual_p = _chi1_from_score(score, base-correction)
    expected_p = _chi1_from_score(score.double(), base.double()-expected)
    rounded_p = _chi1_from_score(score, base-rounded_correction)
    torch.testing.assert_close(actual_p.double(), expected_p, rtol=3e-7, atol=0)
    assert float((torch.log10(actual_p)-torch.log10(rounded_p)).abs().max()) > .01


@pytest.mark.parametrize('strided', [False, True])
def test_small_ieee_covariance_uses_bounded_tiles_with_independent_actual_strides(monkeypatch, strided):
    from fudan_wgs_toolkit._binary_burden_diag import _ieee_covariance_projection
    t, q, b = 3, 67, 7
    rng = np.random.default_rng(2873)
    original = torch.tensor(rng.normal(size=(t, q, b)), dtype=torch.float32)
    covariances = [torch.tensor(rng.normal(size=(q, q)), dtype=torch.float32) for _ in range(t)]
    if strided:
        slab = torch.zeros((t*2+1, q*2+1, b*2+1))
        cross = slab[1::2, 1::2, 1::2]
        cross.copy_(original)
        resident = []
        for value in covariances:
            slab_c = torch.zeros((q*2+1, q*2+1))
            c = slab_c[1::2, 1::2].T
            c.copy_(value)
            resident.append(c)
    else:
        cross, resident = original, covariances
    models = [SimpleNamespace(fixed_effect_covariance=value) for value in resident]
    before = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)
    expected = torch.stack([h.double().T@c.double() for h,c in zip(cross, resident)])
    multiply = torch.Tensor.__mul__
    peak_slab = 0
    def checked_multiply(left, right):
        nonlocal peak_slab
        output = multiply(left, right)
        if output.ndim == 3:
            peak_slab = max(peak_slab, output.numel())
            assert output.numel() <= 4*32*16
        return output
    monkeypatch.setattr(torch.Tensor, '__mul__', checked_multiply)
    actual = _ieee_covariance_projection(cross, models)
    assert peak_slab == 4*32*16
    assert actual.shape == (t, b, q) and actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=6e-6)
    assert (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32) == before


@pytest.mark.parametrize('invalid', ['cross_dtype', 'covariance_dtype', 'covariance_shape', 'trait_count'])
def test_small_ieee_covariance_rejects_geometry_before_allocating_output(monkeypatch, invalid):
    from fudan_wgs_toolkit._binary_burden_diag import _ieee_covariance_projection
    cross = torch.ones((2, 5, 7))
    models = [SimpleNamespace(fixed_effect_covariance=torch.eye(5)) for _ in range(2)]
    if invalid == 'cross_dtype':cross = cross.double()
    elif invalid == 'covariance_dtype':models[0].fixed_effect_covariance = torch.eye(5, dtype=torch.float64)
    elif invalid == 'covariance_shape':models[0].fixed_effect_covariance = torch.eye(6)
    else:models = models[:1]
    monkeypatch.setattr(torch, 'empty', lambda *a, **k:pytest.fail('invalid IEEE projection allocated output'))
    with pytest.raises(ValueError, match='IEEE covariance projection'):
        _ieee_covariance_projection(cross, models)


@pytest.mark.skipif(not CUDA_NATIVE, reason='requires actual Ampere CUDA backend')
@pytest.mark.parametrize('layout', ['c', 'f', 'strided'])
def test_cuda_small_ieee_covariance_preserves_supplied_entries_and_layout(layout):
    from fudan_wgs_toolkit._binary_burden_diag import _ieee_covariance_projection
    t, q, b = 3, 37, 7
    torch.manual_seed(2881)
    h = torch.randn((t, q, b), device='cuda')
    covariance = [torch.randn((q, q), device='cuda') for _ in range(t)]
    if layout == 'f':
        h = h.transpose(1, 2).contiguous().transpose(1, 2)
        covariance = [value.T.contiguous().T for value in covariance]
    elif layout == 'strided':
        backing = torch.zeros((2*t+1, 2*q+1, 2*b+1), device='cuda')
        view = backing[1::2, 1::2, 1::2]
        view.copy_(h)
        h = view
        values = []
        for c in covariance:
            backing_c = torch.zeros((2*q+1, 2*q+1), device='cuda')
            value = backing_c[1::2, 1::2].T
            value.copy_(c)
            values.append(value)
        covariance = values
    models = [SimpleNamespace(fixed_effect_covariance=value) for value in covariance]
    before = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)
    expected = torch.stack([value.double().T@c.double() for value,c in zip(h,covariance)])
    actual = _ieee_covariance_projection(h, models)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=6e-6)
    assert (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32) == before
    c = torch.tensor([[1., -.9999], [-.9999, 1.]], device='cuda')
    exact_h = torch.tensor([[[128.,64.], [128.,64.]]], device='cuda')
    projection = _ieee_covariance_projection(exact_h, [SimpleNamespace(fixed_effect_covariance=c)])
    expected_projection = exact_h.double().transpose(1, 2)@c.double()
    torch.testing.assert_close(projection.double(), expected_projection, rtol=0, atol=0)


@pytest.mark.parametrize('rejection', ['false', 'memory_error'])
def test_accurate_native_retry_rejects_all_budgets_without_old_covariance_or_allocation(monkeypatch, rejection):
    """CPU routing contract; native execution is independently GPU gated."""
    from fudan_wgs_toolkit import binary_null, _binary_burden_diag
    models, z = models_and_burdens(n=101, b=7, t=3, q=5, mode='tf32')
    before = [(m.scaled_residuals.clone(),m.fixed_effect_covariance.clone()) for m in models]
    monkeypatch.setattr(binary_null,'_native_burden_diagonal_supported',lambda *a:True)
    monkeypatch.setattr(binary_null,'_normal_fp32_burden_accuracy_required',lambda *a:True)
    monkeypatch.setattr(_binary_burden_diag,'diagonal_burden_products',lambda *a,**k:pytest.fail('denied partials allocated'))
    monkeypatch.setattr(binary_null,'binary_burden_score_covariance_batch',lambda *a,**k:pytest.fail('old covariance fallback'))
    monkeypatch.setattr(torch,'empty',lambda *a,**k:pytest.fail('denied budget allocated output'))
    observed=[]
    def reject(required):
        observed.append(required)
        if rejection=='memory_error':raise MemoryError('test admission')
        return False
    with pytest.raises(MemoryError,match='accurate independent binary diagonal workspace'):
        binary_burden_score_variance_batch(models,z,allocation_guard=reject)
    assert len(observed)==2 and observed[1]<observed[0]
    for m,(r,c) in zip(models,before):
        torch.testing.assert_close(m.scaled_residuals,r,rtol=0,atol=0)
        torch.testing.assert_close(m.fixed_effect_covariance,c,rtol=0,atol=0)


def test_accurate_native_retry_splits_admitted_trait_work_without_refitting(monkeypatch):
    """A CPU formula/routing contract, without a GPU performance claim."""
    from fudan_wgs_toolkit import binary_null
    models,z=models_and_burdens(n=101,b=7,t=3,q=5,mode='tf32')
    singleton=binary_burden_diag_workspace(101,7,5,1)
    observed=[]
    def admit(required):observed.append(required);return required<=singleton
    monkeypatch.setattr(binary_null,'_native_burden_diagonal_supported',lambda *a:True)
    monkeypatch.setattr(binary_null,'_normal_fp32_burden_accuracy_required',lambda *a:True)
    monkeypatch.setattr(binary_null,'_guard_binary_burden_diag_workspace',lambda *a:None)
    monkeypatch.setattr(binary_null,'binary_burden_score_covariance_batch',lambda *a,**k:pytest.fail('old covariance fallback'))
    score,variance,execution=binary_burden_score_variance_batch(models,z,allocation_guard=admit)
    assert len(observed)==4 and observed[0]>singleton and observed[1:]==[singleton]*3
    assert execution['independent_diagonal_retries']==3
    assert execution['diagonal_admission_fallbacks']==1
    assert execution['native_diagonal_batches']==3
    assert execution['cpu_diagonal_oracle_batches']==3
    for i,m in enumerate(models):
        cross=m.precision_x.double().T@z.double()
        expected=(z.double()*(m.precision.double()[:,None]*z.double())).sum(0)-((cross.T@m.fixed_effect_covariance.double())*cross.T).sum(-1)
        torch.testing.assert_close(variance[i].double(),expected,rtol=3e-6,atol=2e-6)
        torch.testing.assert_close(score[i].double(),z.double().T@m.scaled_residuals.double(),rtol=3e-6,atol=2e-6)


def test_mixed_covariate_dimensions_route_to_independent_accurate_native_contract(monkeypatch):
    from fudan_wgs_toolkit import binary_null
    a,z=models_and_burdens(n=101,b=7,t=1,q=3,mode='tf32')
    b,unused=models_and_burdens(n=101,b=7,t=1,q=5,mode='tf32')
    models=a+b
    monkeypatch.setattr(binary_null,'_native_burden_diagonal_supported',lambda models,z:len({m.x.shape[1] for m in models})==1)
    monkeypatch.setattr(binary_null,'_normal_fp32_burden_accuracy_required',lambda *a:True)
    monkeypatch.setattr(binary_null,'_guard_binary_burden_diag_workspace',lambda *a:None)
    monkeypatch.setattr(binary_null,'binary_burden_score_covariance_batch',lambda *a,**k:pytest.fail('old mixed-Q covariance fallback'))
    score,variance,execution=binary_burden_score_variance_batch(models,z)
    assert execution['native_diagonal_batches']==2 and execution['independent_diagonal_retries']==2
    assert execution['diagonal_admission_fallbacks']==0
    for i,m in enumerate(models):
        cross=m.precision_x.double().T@z.double()
        expected=(z.double()*(m.precision.double()[:,None]*z.double())).sum(0)-((cross.T@m.fixed_effect_covariance.double())*cross.T).sum(-1)
        torch.testing.assert_close(variance[i].double(),expected,rtol=3e-6,atol=2e-6)


@pytest.mark.parametrize('kind',['csr','dense','uncoalesced'])
def test_independent_precision_fallback_preserves_adversarial_covariance_and_original_state(monkeypatch,kind):
    """CPU formula contract simulates dispatcher policy, not a CUDA benchmark."""
    from fudan_wgs_toolkit import binary_null
    from fudan_wgs_toolkit.statistics import _chi1_from_score
    models,unused=models_and_burdens(n=128,b=2,t=1,q=2,mode='tf32')
    model=models[0]
    z=torch.ones((128,2));z[:,1]=.5
    dense=torch.diag(torch.full((128,),.25))
    if kind=='csr':model.precision=dense.to_sparse_csr()
    elif kind=='dense':model.precision=dense
    else:
        coo=dense.to_sparse()
        model.precision=torch.sparse_coo_tensor(coo.indices(),coo.values(),coo.shape,is_coalesced=False)
    model.precision_x=torch.ones((128,2))
    model.fixed_effect_covariance=torch.tensor([[1.,-.9999],[-.9999,1.]])
    model.scaled_residuals=torch.full((128,),.0625)
    originals=[v.clone() for v in [model.precision,model.precision_x,model.fixed_effect_covariance,model.scaled_residuals]]
    monkeypatch.setattr(binary_null,'_normal_fp32_burden_accuracy_required',lambda *a:True)
    monkeypatch.setattr(binary_null,'_guard_binary_burden_diag_workspace',lambda *a:None)
    monkeypatch.setattr(binary_null,'matmul',lambda left,right,mode:left.float()@right.float())
    monkeypatch.setattr(binary_null,'binary_burden_score_covariance_batch',lambda *a,**k:pytest.fail('old covariance fallback'))
    requests=[]
    score,variance,execution=binary_burden_score_variance_batch(models,z,allocation_guard=lambda required:requests.append(required))
    assert requests==[binary_null._accurate_independent_burden_workspace(model,z)]
    assert execution['accurate_independent_projection_batches']==1
    assert execution['native_diagonal_batches']==0
    expected_cross=model.precision_x.double().T@z.double()
    base=(z.double()*(dense.double()@z.double())).sum(0)
    correction=((expected_cross.T@model.fixed_effect_covariance.double())*expected_cross.T).sum(-1)
    torch.testing.assert_close(variance[0].double(),base-correction,rtol=0,atol=0)
    torch.testing.assert_close(score[0].double(),z.double().T@model.scaled_residuals.double(),rtol=0,atol=0)
    actualp=_chi1_from_score(score[0],variance[0])
    roundedp=_chi1_from_score(score[0],base.float())
    assert float((torch.log10(actualp)-torch.log10(roundedp)).abs().max())>.01
    for original,current in zip(originals,[model.precision,model.precision_x,model.fixed_effect_covariance,model.scaled_residuals]):
        # Sparse layouts are checked in their original dense value geometry.
        torch.testing.assert_close(original.to_dense(),current.to_dense(),rtol=0,atol=0)


@pytest.mark.parametrize('rejection',['false','memory_error'])
def test_independent_precision_fallback_rejects_before_original_projection(monkeypatch,rejection):
    from fudan_wgs_toolkit import binary_null
    models,z=models_and_burdens(n=101,b=7,t=1,q=5,mode='tf32')
    models[0].precision=torch.diag(models[0].precision).to_sparse_csr()
    monkeypatch.setattr(binary_null,'_normal_fp32_burden_accuracy_required',lambda *a:True)
    monkeypatch.setattr(models[0],'_association_genotype',lambda *a,**k:pytest.fail('denied accurate fallback allocated source'))
    monkeypatch.setattr(binary_null,'binary_burden_score_covariance_batch',lambda *a,**k:pytest.fail('denied accurate fallback tried old C'))
    monkeypatch.setattr(torch,'empty',lambda *a,**k:pytest.fail('denied accurate fallback allocated result'))
    def reject(required):
        if rejection=='memory_error':raise MemoryError('test admission')
        return False
    with pytest.raises(MemoryError):binary_burden_score_variance_batch(models,z,allocation_guard=reject)


@pytest.mark.parametrize('requires_grad', ['burdens','precision','covariance','residual','sx'])
def test_normal_accuracy_dispatch_rejects_gradient_tensors_before_admission(monkeypatch,requires_grad):
    from fudan_wgs_toolkit import binary_null
    models,z=models_and_burdens(n=101,b=1,t=1,q=1,mode='tf32')
    model=models[0]
    target={'burdens':z,'precision':model.precision,'covariance':model.fixed_effect_covariance,
            'residual':model.scaled_residuals,'sx':model.precision_x}[requires_grad]
    target.requires_grad_(True)
    monkeypatch.setattr(binary_null,'_normal_fp32_burden_accuracy_required',lambda *a:True)
    with pytest.raises(ValueError,match='inference tensors without gradients'):
        binary_burden_score_variance_batch(models,z,allocation_guard=lambda *a:pytest.fail('gradient state admitted'))


@pytest.mark.skipif(not CUDA_NATIVE, reason='requires actual Ampere CUDA backend')
def test_cuda_rejected_stack_retries_single_traits_and_mixed_q_retains_accurate_c():
    models,z=models_and_burdens(n=5037,b=26,t=3,q=16,device='cuda',mode='tf32')
    single=binary_burden_diag_workspace(5037,26,16,1)
    requests=[]
    def admit(required):requests.append(required);return required<=single
    score,variance,execution=binary_burden_score_variance_batch(models,z,allocation_guard=admit)
    assert len(requests)==4 and execution['independent_diagonal_retries']==3
    assert execution['native_diagonal_batches']==3
    assert execution['ieee_covariance_projection_batches']==3
    other,unused=models_and_burdens(n=5037,b=26,t=1,q=7,device='cuda',mode='tf32')
    score2,variance2,execution2=binary_burden_score_variance_batch([models[0],other[0]],z)
    assert execution2['native_diagonal_batches']==2
    assert execution2['ieee_covariance_projection_batches']==2
    for i,m in enumerate([models[0],other[0]]):
        cross=m.precision_x.cpu().double().T@z.cpu().double()
        expected=(z.cpu().double()*(m.precision.cpu().double()[:,None]*z.cpu().double())).sum(0)-((cross.T@m.fixed_effect_covariance.cpu().double())*cross.T).sum(-1)
        torch.testing.assert_close(variance2[i].cpu().double(),expected,rtol=2e-5,atol=1e-4)


@pytest.mark.skipif(not CUDA_NATIVE, reason='requires actual Ampere CUDA backend')
@pytest.mark.parametrize('kind',['csr','dense'])
def test_cuda_independent_precision_fallback_preserves_adversarial_small_covariance(kind):
    from fudan_wgs_toolkit import binary_null
    from fudan_wgs_toolkit.statistics import _chi1_from_score
    models,unused=models_and_burdens(n=128,b=2,t=1,q=2,mode='tf32',device='cuda')
    model=models[0]
    z=torch.ones((128,2),device='cuda');z[:,1]=.5
    dense=torch.diag(torch.full((128,),.25,device='cuda'))
    model.precision=dense.to_sparse_csr() if kind=='csr' else dense
    model.precision_x=torch.ones((128,2),device='cuda')
    model.fixed_effect_covariance=torch.tensor([[1.,-.9999],[-.9999,1.]],device='cuda')
    model.scaled_residuals=torch.full((128,),.0625,device='cuda')
    before=(torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32)
    originals=[v.clone() for v in [model.precision,model.precision_x,model.fixed_effect_covariance,model.scaled_residuals]]
    requests=[]
    score,variance,execution=binary_burden_score_variance_batch(models,z,allocation_guard=lambda required:requests.append(required))
    assert requests==[binary_null._accurate_independent_burden_workspace(model,z)]
    assert execution['accurate_independent_projection_batches']==1
    assert execution['native_diagonal_batches']==0
    assert execution['ieee_covariance_projection_batches']==1
    assert execution['cuda_diagonal_kernel_launches']==1
    expected_cross=model.precision_x.double().T@z.double()
    base=(z.double()*(dense.double()@z.double())).sum(0)
    correction=((expected_cross.T@model.fixed_effect_covariance.double())*expected_cross.T).sum(-1)
    torch.testing.assert_close(variance[0].double(),base-correction,rtol=0,atol=0)
    torch.testing.assert_close(score[0].double(),z.double().T@model.scaled_residuals.double(),rtol=0,atol=0)
    assert float((torch.log10(_chi1_from_score(score[0],variance[0]))-torch.log10(_chi1_from_score(score[0],base.float()))).abs().max())>.01
    for original,current in zip(originals,[model.precision,model.precision_x,model.fixed_effect_covariance,model.scaled_residuals]):
        torch.testing.assert_close(original.to_dense(),current.to_dense(),rtol=0,atol=0)
    assert (torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32)==before
