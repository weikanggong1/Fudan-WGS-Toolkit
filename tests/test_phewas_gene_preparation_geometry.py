"""Fitted-state IO/ownership contracts, not real-data performance benchmarks."""
from contextlib import contextmanager
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.pipeline import AnalysisOptions
from fudan_wgs_toolkit.phewas_runtime import runtime
from fudan_wgs_toolkit.phewas_runtime.metadata import SharedMetadataReader
from fudan_wgs_toolkit.phewas_runtime.shared_state import SharedStateBroker
from test_phewas_batched_runtime import cpu_product_oracle, descriptor, model_for
from test_phewas_shared_state import MetadataReader, make_cache


class _Reader(MetadataReader):
    def read_field(self, name, indices=None):
        values = (np.full(self.n_variants, 'PASS') if name == 'annotation/filter'
                  else np.full(self.n_variants, '21') if name == 'chromosome'
                  else np.arange(self.n_variants))
        return values if indices is None else values[indices]


class _EphemeralRepository:
    """No retained bank entries: every acquire represents a fitted-state load."""
    def __init__(self, descriptors):
        self.items = descriptors
        self.acquired, self.references = [], []
        self.active = 0

    def descriptors(self):
        return self.items

    @contextmanager
    def acquire(self, indices, device):
        assert self.active == 0
        assert all(reference() is None for reference in self.references)
        values = [model_for(self.items[index]['ordered_rows'], .7+index) for index in indices]
        self.references.extend(weakref.ref(value) for value in values)
        self.acquired.append(list(indices))
        self.active += 1
        try:
            yield values
        finally:
            values.clear()
            self.active -= 1


@pytest.mark.parametrize('outcome', ['success', 'ineligible', 'prepare_error', 'no_masks'])
def test_multimask_anchor_dimensions_release_before_preparation_and_keep_trait_math(
        tmp_path, monkeypatch, cpu_product_oracle, outcome):
    n, m = 256, 8
    rows = np.arange(n)
    raw = np.zeros((m, n), dtype=np.uint8)
    raw[np.arange(m), np.arange(m)] = 1
    masks = dict(plof=np.arange(4), missense=np.arange(3, 6), synonymous=np.arange(6, 8))
    if outcome == 'no_masks':
        masks = {}
    repository = _EphemeralRepository([descriptor(i, rows) for i in range(3)])
    _, groups = runtime._descriptor_groups(repository)
    prepared_calls, evaluated, writes, future = [], [], [], []
    native_prepare = runtime._prepare_shared_gene

    def prepare(pipeline, indices, broker):
        # The proxy retains no x/residual/spectrum tensor and no model lease.
        assert repository.active == 0
        assert all(reference() is None for reference in repository.references)
        assert vars(pipeline.models[0]) == dict(n=n, device=torch.device('cpu'), use_spa=False,
            gaussian_rotation_geometry_known=True, gaussian_has_blocks=False,
            gaussian_maximum_family_rows=0)
        prepared_calls.append(indices.copy())
        if outcome == 'prepare_error':
            raise MemoryError('injected preparation rejection')
        return native_prepare(pipeline, indices, broker)

    def evaluate(pipeline, payload, model):
        # This follows real shared Score and per-model covariance kernels.
        evaluated.append((payload['score'].clone(), payload['covariance'].clone()))
        return dict(num_variant=len(payload['maf']), cMAC=payload['cmac'], **{'WGS-O': .25})

    monkeypatch.setattr(runtime, '_prepare_shared_gene', prepare)
    monkeypatch.setattr(runtime.LimitedMaskPipeline, '_select_mask_chunks', lambda *args: masks)
    monkeypatch.setattr(runtime.LimitedMaskPipeline, '_long_mask_products',
        lambda self, model, host, **kwargs: model.score_covariance(torch.from_numpy(host)))
    monkeypatch.setattr(runtime.LimitedMaskPipeline, '_evaluate_prepared', evaluate)
    options = AnalysisOptions(wrapper_semantics='base', memory_limit_gib=80,
                              rv_num_cutoff=4 if outcome == 'ineligible' else 2)
    configuration = {'_host_memory_set_future': lambda value, **kwargs: future.append(value)}
    report = dict(association_rows={'coding': 0}, trait_rows={i: {'coding': 0} for i in range(3)},
                  binary_spa_selected=0, binary_spa_failed=0)
    with SharedStateBroker(SharedMetadataReader(_Reader(n, m)),
                          make_cache(tmp_path/'cache', raw), device='cpu') as broker:
        axis = broker.reader_view(rows)
        structural = runtime._thin_pipeline(axis, SimpleNamespace(n=n, device='cpu', use_spa=False),
                                            axis._axis.samples, options, configuration)
        pins = axis._axis.pins

        def run():
            runtime._batched_gene(groups, repository, broker, broker._reader, options, configuration,
                dict(gene_name='anonymous_gene', start=0, end=m-1),
                SimpleNamespace(write=lambda *args, **kwargs: writes.append(args)), report,
                set(), '21', 0, 2, structural, 'coding')

        if outcome == 'prepare_error':
            with pytest.raises(MemoryError, match='preparation rejection'):
                run()
            assert not writes and not evaluated and repository.acquired == [[0]]
        else:
            run()
            eligible_masks = ([] if outcome == 'no_masks' else
                              [indices for indices in masks.values() if len(indices) >= options.rv_num_cutoff])
            expected_acquires = ([] if outcome == 'no_masks' else [[0]])
            expected_acquires += [[0, 1], [2]]*len(eligible_masks)
            assert repository.acquired == expected_acquires
            assert len(evaluated) == 3*len(eligible_masks)
            for mask_index, indices in enumerate(eligible_masks):
                genotype = torch.from_numpy(np.asfortranarray(raw[indices].T.astype(np.float32)))
                for trait_index in range(3):
                    expected_model = model_for(rows, .7+trait_index)
                    expected_score, expected_covariance = expected_model.score_covariance(genotype)
                    actual_score, actual_covariance = evaluated[3*mask_index+trait_index]
                    torch.testing.assert_close(actual_score, expected_score, rtol=5e-6, atol=5e-6)
                    torch.testing.assert_close(actual_covariance, expected_covariance, rtol=0, atol=0)
            assert len(writes) == 3
            assert report['association_rows']['coding'] == 3*len(eligible_masks)
            for trait in report['trait_rows'].values():
                assert trait['coding'] == len(eligible_masks)
        assert axis._axis.pins == pins
        assert repository.active == 0 and all(reference() is None for reference in repository.references)
        assert not future or future[-1] == 0
        axis.close()


@pytest.mark.parametrize('bad_binding', ['count', 'axis'])
def test_preparation_dimensions_cannot_bypass_model_descriptor_binding(bad_binding):
    rows = np.arange(5)
    item = descriptor(0, rows)
    references = []

    @contextmanager
    def acquire(indices, device):
        values = ([] if bad_binding == 'count' else [model_for(rows[::-1])])
        references.extend(weakref.ref(value) for value in values)
        try:
            yield values
        finally:
            values.clear()

    repository = SimpleNamespace(acquire=acquire)
    message = 'requested model count' if bad_binding == 'count' else 'portable ordered sample binding'
    with pytest.raises(ValueError, match=message) as retained_error:
        runtime._gene_preparation_geometry(repository, item, SimpleNamespace(device='cpu'),
                                          SharedMetadataReader(MetadataReader(5, 1)), set())
    assert retained_error.value is not None
    assert all(reference() is None for reference in references)
