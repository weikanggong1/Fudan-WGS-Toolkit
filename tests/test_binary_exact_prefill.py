"""Allocation/order/read-count contracts; these fixtures are not a GPU benchmark."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.binary import (ExactBinaryBurdenCache, _BinaryPanelSource,
    prepare_binary_burdens)
from fudan_wgs_toolkit.precision_audit import DenseProductAudit
from fudan_wgs_toolkit.statistics import annotation_weights


class CountedPanel(_BinaryPanelSource):
    def __init__(self, values, sample_ids):
        self.values, self.sample_ids = values, sample_ids
        self.shape, self.reads = values.shape, []

    def validate(self):
        return (id(self), self.values._version)

    def read_columns(self, columns):
        assert len(columns) <= 512
        self.reads.append(tuple(columns))
        return self.values[:, torch.tensor(np.asarray(columns).copy())].contiguous()


def fixture(m, *, native=False, annotations=2):
    n = 32
    model = SimpleNamespace(family='binomial', n_pheno=1, n=n,
        device=torch.device('cpu'), sample_ids=np.arange(n).astype(str),
        matmul_mode='tf32' if native else 'fp64')
    source = torch.zeros((m, n), dtype=torch.float32).T
    source[torch.arange(m) % n, torch.arange(m)] = 1
    maf = torch.linspace(.001, .009, m, dtype=torch.float64)
    annotation = torch.arange(m*annotations).reshape(m, annotations).remainder(13).double()/3
    return model, source, maf, annotation


@pytest.mark.parametrize('m', [2, 31, 511, 512, 513, 4999, 5000, 5001, 15001, 50000, 100000])
def test_prefill_all_lengths_is_one_panel_pass_and_original_exact_values(m):
    model, genotype, maf, annotation = fixture(m)
    native_source = CountedPanel(genotype, model.sample_ids)
    seed_source = CountedPanel(genotype, model.sample_ids)
    original = prepare_binary_burdens(model, native_source, maf, annotation)
    requests = []
    prepared = prepare_binary_burdens(model, seed_source, maf, annotation,
        prefill_exact=True, allocation_guard=requests.append)
    torch.testing.assert_close(prepared.burdens, original.burdens, rtol=0, atol=0)
    assert prepared.cmac == original.cmac
    cache = ExactBinaryBurdenCache(prepared, allocation_guard=requests.append)
    reference = ExactBinaryBurdenCache(original)
    index = torch.tensor([5, 0, 3, 2, 4, 1])
    torch.testing.assert_close(cache.get(prepared, index), reference.get(original, index),
        rtol=0, atol=2e-12)
    assert cache.prefill_used and cache.build_calls == 0 and cache.columns_computed == 6
    assert cache.filled == set(range(6))
    assert sum(map(len, seed_source.reads)) == m
    assert sum(map(len, native_source.reads)) == 2*m
    assert seed_source.reads == native_source.reads[:len(seed_source.reads)]
    assert max(map(len, seed_source.reads)) <= 512
    assert requests[0] >= 8*model.n*min(m, 512) + 16*model.n*6 + 32*m*6
    assert requests[-1] == 8*model.n*6
    assert cache.get(prepared, torch.tensor([], dtype=torch.int64)).shape == (model.n, 0)


def test_prefill_is_declared_spa_only_and_native_outputs_unchanged(monkeypatch):
    from fudan_wgs_toolkit import tf32
    monkeypatch.setattr(tf32, 'matmul', lambda left, right, *, mode: left@right)
    model, genotype, maf, annotation = fixture(513, native=True)
    old = prepare_binary_burdens(model, genotype, maf, annotation)
    with DenseProductAudit(forced=True) as audit:
        actual = prepare_binary_burdens(model, genotype, maf, annotation, prefill_exact=True)
    torch.testing.assert_close(actual.burdens, old.burdens, rtol=0, atol=0)
    assert actual.burdens.dtype == torch.float32
    assert audit.fp64_products == 0
    assert audit.explicit_binary_spa_fp64_products == 2
    assert audit.observed_products == 4


@pytest.mark.parametrize('kind', ['caller', 'caller_false', 'builtin'])
def test_admission_rejection_retains_original_path_before_exact_allocations(monkeypatch, kind):
    from fudan_wgs_toolkit import binary
    model, genotype, maf, annotation = fixture(513)
    expected = prepare_binary_burdens(model, genotype, maf, annotation)
    source = CountedPanel(genotype, model.sample_ids)
    requests = []
    def reject(*args):
        requests.append(args[-1])
        assert not source.reads
        if kind == 'caller_false':
            return False
        raise MemoryError('optional prefill policy')
    if kind == 'builtin':
        monkeypatch.setattr(binary, '_guard_exact_burden_workspace', reject)
    original_weights = binary.annotation_weights
    weight_calls = []
    def weights(*args, **kwargs):
        weight_calls.append(kwargs['dtype'])
        return original_weights(*args, **kwargs)
    monkeypatch.setattr(binary, 'annotation_weights', weights)
    actual = prepare_binary_burdens(model, source, maf, annotation, prefill_exact=True,
        allocation_guard=reject if kind.startswith('caller') else None)
    assert actual.exact_seed is None and len(requests) == 1
    assert weight_calls == [torch.float64]  # Native FP64 fixture; no second exact family.
    torch.testing.assert_close(actual.burdens, expected.burdens, rtol=0, atol=0)
    assert sum(map(len, source.reads)) == genotype.shape[1]


def test_required_exact_cache_false_rejection_precedes_storage_and_panel_reads(monkeypatch):
    from fudan_wgs_toolkit import binary
    model, genotype, maf, annotation = fixture(513)
    source = CountedPanel(genotype, model.sample_ids)
    prepared = prepare_binary_burdens(model, source, maf, annotation)
    source.reads.clear()
    cache = ExactBinaryBurdenCache(prepared, allocation_guard=lambda _: False)
    original_weights = binary.annotation_weights
    calls = []
    def weights(*args, **kwargs):
        calls.append(None)
        return original_weights(*args, **kwargs)
    monkeypatch.setattr(binary,'annotation_weights',weights)
    index = torch.tensor([0, 2])
    with pytest.raises(MemoryError,match='cache allocation'):
        cache.get(prepared,index)
    assert cache.values is cache.weights is None and cache.filled == set()
    assert source.reads == [] and calls == [] and cache.build_calls == 0
    cache.allocation_guard = lambda _: None
    actual = cache.get(prepared,index)
    assert actual.shape == (model.n,2) and cache.build_calls == 1


def test_prefilled_cache_false_output_rejection_retains_verified_seed():
    model, genotype, maf, annotation = fixture(513)
    prepared = prepare_binary_burdens(model,genotype,maf,annotation,prefill_exact=True)
    cache = ExactBinaryBurdenCache(prepared,allocation_guard=lambda _: False)
    before = cache.values
    with pytest.raises(MemoryError,match='output allocation'):
        cache.get(prepared,torch.tensor([1]))
    assert cache.prefill_used and cache.values is before and cache.build_calls == 0
    cache.allocation_guard = lambda _: True
    actual = cache.get(prepared,torch.tensor([1]))
    torch.testing.assert_close(actual,prepared.exact_seed.values[:,[1]],rtol=0,atol=0)


def test_actual_allocation_error_is_not_optional_admission_fallback(monkeypatch):
    from fudan_wgs_toolkit import binary
    model, genotype, maf, annotation = fixture(513)
    original_weights = binary.annotation_weights
    calls = []
    def fail_second_weights(*args, **kwargs):
        calls.append(None)
        if len(calls) == 2:
            raise torch.OutOfMemoryError('actual allocation failure')
        return original_weights(*args, **kwargs)
    monkeypatch.setattr(binary, 'annotation_weights', fail_second_weights)
    with pytest.raises(torch.OutOfMemoryError, match='actual allocation'):
        prepare_binary_burdens(model, genotype, maf, annotation, prefill_exact=True)


def test_different_cache_panel_width_uses_original_lazy_order():
    model, genotype, maf, annotation = fixture(513)
    source = CountedPanel(genotype, model.sample_ids)
    prepared = prepare_binary_burdens(model, source, maf, annotation, prefill_exact=True)
    cache = ExactBinaryBurdenCache(prepared, variant_tile_size=37)
    assert not cache.prefill_used and cache.values is None
    actual = cache.get(prepared, torch.tensor([0, 4]))
    weights = annotation_weights(maf, annotation)[0][:, [0, 4]]
    expected = torch.zeros_like(actual)
    for start in range(0, len(maf), 37):
        expected.add_(genotype[:, start:start+37].double()@weights[start:start+37])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert cache.build_calls == 1 and sum(map(len, source.reads)) == 2*len(maf)


@pytest.mark.parametrize('mutation', ['source', 'maf', 'annotation', 'seed_values', 'seed_weights'])
def test_prefilled_cache_rejects_mutated_source_metadata_and_exact_storage(mutation):
    model, genotype, maf, annotation = fixture(31)
    prepared = prepare_binary_burdens(model, genotype, maf, annotation, prefill_exact=True)
    cache = ExactBinaryBurdenCache(prepared)
    target = {'source': genotype, 'maf': prepared.maf, 'annotation': prepared.annotations,
        'seed_values': prepared.exact_seed.values, 'seed_weights': prepared.exact_seed.weights}[mutation]
    target.flatten()[0] += 1  # All fixture views here are contiguous except source.
    if mutation == 'source':
        genotype[0, 0] += 1
    with pytest.raises(ValueError, match='mutated'):
        cache.get(prepared, torch.tensor([0]))


def test_seed_different_sample_axis_or_seal_is_rejected():
    model, genotype, maf, annotation = fixture(31)
    prepared = prepare_binary_burdens(model, genotype, maf, annotation, prefill_exact=True)
    for replacement in [replace(prepared.exact_seed, sample_ids=('wrong',)),
                        replace(prepared.exact_seed, _seal=object())]:
        with pytest.raises(ValueError, match='seed'):
            ExactBinaryBurdenCache(replace(prepared, exact_seed=replacement))


def test_prefill_keeps_filtered_original_columns_and_unrounded_spa_genotypes(monkeypatch):
    from fudan_wgs_toolkit import tf32
    monkeypatch.setattr(tf32, 'matmul', lambda left, right, *, mode: left@right)
    model, genotype, maf, annotation = fixture(519, native=True)
    genotype = genotype.double() + torch.arange(519).double()[None]/100000019
    maf[torch.arange(len(maf)).remainder(11) == 0] = .2
    source = CountedPanel(genotype, model.sample_ids)
    prepared = prepare_binary_burdens(model, source, maf, annotation, prefill_exact=True)
    cache = ExactBinaryBurdenCache(prepared)
    original = prepare_binary_burdens(model, genotype, maf, annotation)
    index = torch.tensor([0, 5])
    reference = ExactBinaryBurdenCache(original).get(original, index)
    torch.testing.assert_close(cache.get(prepared, index), reference, rtol=0, atol=2e-12)
    torch.testing.assert_close(prepared.burdens, original.burdens, rtol=0, atol=0)
    assert prepared.cmac == original.cmac
    assert tuple(prepared.columns) == tuple(torch.nonzero(maf < .01).flatten().tolist())
    assert sum(map(len, source.reads)) == len(prepared.maf)
    # Exact SPA inputs come from source values, not the rounded native product.
    assert not torch.equal(cache.get(prepared, index), prepared.burdens[:, index].double())


@pytest.mark.parametrize('indices', [[-1], [6], [0, 0]])
def test_prefilled_cache_still_checks_unique_valid_weight_columns(indices):
    model, genotype, maf, annotation = fixture(31)
    source = CountedPanel(genotype, model.sample_ids)
    prepared = prepare_binary_burdens(model, source, maf, annotation, prefill_exact=True)
    cache = ExactBinaryBurdenCache(prepared)
    reads = list(source.reads)
    with pytest.raises(ValueError, match='unique valid'):
        cache.get(prepared, torch.tensor(indices))
    assert source.reads == reads and cache.build_calls == 0


@pytest.mark.parametrize('value', [1, 'yes', None])
def test_prefill_parameter_must_be_boolean(value):
    model, genotype, maf, annotation = fixture(31)
    with pytest.raises(ValueError, match='boolean'):
        prepare_binary_burdens(model, genotype, maf, annotation, prefill_exact=value)
