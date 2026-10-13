"""Projection sharing preserves independent Single variance reductions."""
import numpy as np
import pytest
import torch

from fudan_wgs_toolkit import tf32
from fudan_wgs_toolkit import binary_null
from fudan_wgs_toolkit.binary_null import fit_logistic_null
from fudan_wgs_toolkit.phewas_runtime import single


@pytest.fixture(autouse=True)
def independent_cpu_formula_control(monkeypatch):
    # This is a formula/ownership test, not a CUDA performance or precision gate.
    # binary_null imports its mature product alias at module import time.
    product = lambda a,b,**kw:a@b
    monkeypatch.setattr(tf32, 'matmul', product)
    monkeypatch.setattr(binary_null, 'matmul', product)


def _models(count=3, sparse=False):
    n = 40
    x = np.column_stack((np.ones(n), np.linspace(-1, 1, n)))
    models = []
    for index in range(count):
        y = (np.arange(n) % (index+3) == 0).astype(float)
        model = fit_logistic_null(y, x, device='cpu', use_spa=False)
        model.precision = model.fitted_probability*(1-model.fitted_probability)
        model.precision_x = model.xw.T
        if sparse:
            rows = torch.arange(n)
            indices = torch.stack((rows, rows))
            model.precision = torch.sparse_coo_tensor(indices,
                torch.linspace(.2+index*.1, .8, n, dtype=torch.float64), (n, n)).coalesce()
            model.precision_x = torch.sparse.mm(model.precision, model.x)
            model.fixed_effect_covariance = torch.linalg.inv(model.x.T@model.precision_x)
        model.set_matmul_mode('tf32')
        models.append(model)
    return models


def _genotype():
    return (torch.arange(40*7).reshape(40, 7)%3).to(torch.float32)


@pytest.mark.parametrize('sparse', [False, True])
def test_supplied_cross_keeps_the_original_diagonal_formula(monkeypatch, sparse):
    monkeypatch.setattr(tf32, 'matmul', lambda a,b,**kw:a@b)
    g = _genotype();mean = g.mean(0);centered = g-mean
    for model in _models(sparse=sparse):
        original = single._variance_from_shared_genotype(model, g, centered, mean)
        state = model._centered_single_projection_state()
        right = centered if state is not None else g
        cross = model.precision_x.T@right
        actual = single._variance_from_shared_genotype(model, g, centered, mean,
            cross=cross, projection_state=state)
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


def test_trait_batch_avoids_model_stack_and_preserves_all_variances(monkeypatch):
    monkeypatch.setattr(tf32, 'matmul', lambda a,b,**kw:a@b)
    g = _genotype();mean = g.mean(0);centered = g-mean
    models = _models(sparse=True)
    expected = [single._variance_from_shared_genotype(model,g,centered,mean) for model in models]
    calls, guards = [], []
    def batch(left, right, **kwargs):
        calls.append((left,right))
        return torch.stack([a@right for a in left])
    monkeypatch.setattr(tf32,'batched_matmul',batch)
    monkeypatch.setattr(tf32,'_guard_workspace',lambda device,estimate,**kw:guards.append(estimate))
    actual = single._batched_variances(models,g,centered,mean)
    assert len(calls)==1 and calls[0][1] is g
    assert [a.data_ptr() for a in calls[0][0]] == [model.precision_x.data_ptr() for model in models]
    assert guards[0]['required_bytes']>=8*g.numel()+4*len(models)*models[0].x.shape[1]*g.shape[1]
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_budget_rejection_retains_the_mature_variance_path(monkeypatch):
    monkeypatch.setattr(tf32,'matmul',lambda a,b,**kw:a@b)
    g = _genotype();mean = g.mean(0);centered = g-mean
    models = _models(sparse=True)
    expected = [single._variance_from_shared_genotype(model,g,centered,mean) for model in models]
    def reject(*args,**kwargs):
        raise MemoryError('bounded projection workspace')
    monkeypatch.setattr(tf32,'_guard_workspace',reject)
    monkeypatch.setattr(tf32,'batched_matmul',lambda *a,**kw:pytest.fail('not admitted'))
    actual = single._batched_variances(models,g,centered,mean)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
