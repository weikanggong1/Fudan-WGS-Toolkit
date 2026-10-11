"""Covariance-only reuse contracts; small fixtures are not speed benchmarks."""
import importlib
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit import tf32
from fudan_wgs_toolkit._cached_covariance import score_covariance_cached
from fudan_wgs_toolkit.pipeline import PheWASPipeline
from test_cached_covariance import _cuda_model, _cuda_binary_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native CUDA covariance comparison')
@pytest.mark.parametrize('family', ['gaussian', 'binomial'])
@pytest.mark.parametrize('layout', ['C', 'F'])
@pytest.mark.parametrize('panel', [None, 512])
def test_covariance_only_matches_default_bits_and_omits_residual_score_products(
        monkeypatch, family, layout, panel):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip('native TF32 requires Ampere or newer')
    pytest.importorskip('triton')
    tf32.configure_tf32(memory_limit_gib=40, split_k=0)
    n, m = 129, 513  # The one-column tail retains its original GEMV geometry.
    model = _cuda_model(n) if family == 'gaussian' else _cuda_binary_model(n)
    host = np.array(np.random.default_rng(731).integers(0, 3, (n, m)),
        dtype=np.float32, order=layout)
    host[2, -1] = .173
    score, expected, baseline = model.score_covariance_cached(host,
        variant_tile_size=512, panel_variant_size=panel, profile=True)
    assert score.shape == (m,) and baseline['score_product_calls'] == 2
    assert baseline['compute_score'] is baseline['score_retained'] is True
    del score
    module = importlib.import_module(type(model).__module__)
    original_product, original_empty = module.matmul, torch.empty
    residual_products, vector_allocations = [], []
    def observed_product(left, right, *, mode):
        if right is model.scaled_residuals:
            residual_products.append(tuple(left.shape))
        return original_product(left, right, mode=mode)
    def observed_empty(*args, **kwargs):
        if args and args[0] == (m,):
            vector_allocations.append(args[0])
        return original_empty(*args, **kwargs)
    monkeypatch.setattr(module, 'matmul', observed_product)
    monkeypatch.setattr(torch, 'empty', observed_empty)
    omitted, actual, report = model.score_covariance_cached(host,
        variant_tile_size=512, panel_variant_size=panel, profile=True, compute_score=False)
    assert omitted is None and not residual_products and not vector_allocations
    assert torch.equal(actual, expected)
    assert report['score_product_calls'] == 0
    assert report['compute_score'] is report['score_retained'] is False
    assert 'score_prepare' not in report['stages']
    for field in ('preparation_tile_size', 'effective_variant_tile_size',
                  'panel_variant_size', 'prepared_variant_tiles', 'cross_product_calls',
                  'projection_left_product_calls', 'covariance_product_calls',
                  'projection_product_calls', 'score_layout_copy_calls',
                  'score_layout_copy_bytes', 'singleton_product_blocks',
                  'h2d_bytes', 'h2d_calls', 'validation_sync_count'):
        assert report[field] == baseline[field]


@pytest.mark.parametrize('value', [None, 0, 'false'])
def test_covariance_only_flag_requires_explicit_boolean(value):
    with pytest.raises(ValueError, match='compute_score must be Boolean'):
        score_covariance_cached(SimpleNamespace(), np.ones((2, 3)), compute_score=value)


def test_pipeline_forwards_false_without_replacing_native_covariance():
    p = object.__new__(PheWASPipeline)
    p.options = SimpleNamespace(covariance_backend='cached', cached_variant_tile_size=4096,
        memory_limit_gib=80)
    p.covariance_diagnostics = []
    covariance, report, calls = torch.eye(3), {'fixture': True}, []
    def products(host, **kwargs):
        calls.append(kwargs)
        return None if not kwargs['compute_score'] else torch.ones(3), covariance, report
    model = SimpleNamespace(score_covariance_cached=products)
    score, actual = p._long_mask_products(model, np.ones((2, 3)), compute_score=False)
    assert score is None and actual is covariance and p.covariance_diagnostics == [report]
    assert calls == [dict(variant_tile_size=4096, memory_limit_gib=80,
                          profile=True, compute_score=False)]
    score, actual = p._long_mask_products(model, np.ones((2, 3)))
    assert score.shape == (3,) and actual is covariance and calls[-1]['compute_score'] is True


def test_legacy_fallback_does_not_retain_discarded_score():
    p = object.__new__(PheWASPipeline)
    p.options = SimpleNamespace(covariance_backend='legacy', variant_tile_size=512)
    covariance, references = torch.eye(3), []
    def products(host, **kwargs):
        assert kwargs == dict(variant_tile_size=512)
        score = torch.ones(3)
        references.append(weakref.ref(score))
        return score, covariance
    score, actual = p._long_mask_products(SimpleNamespace(score_covariance_tiled=products),
        np.ones((2, 3)), compute_score=False)
    assert score is None and actual is covariance and references[0]() is None
