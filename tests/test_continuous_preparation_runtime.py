"""CPU orchestration contracts, including an explicit CUDA-branch marker.

The marker selects runtime branches only. Storage and every numeric operation
remain CPU; these tests do not claim CUDA precision or real-data performance.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit import _gaussian_covariance_shared as panels
from fudan_wgs_toolkit import _gaussian_resident_shared as resident
from fudan_wgs_toolkit._blocked_covariance import blocked_gaussian_covariance
from fudan_wgs_toolkit.pipeline import AnalysisOptions
from fudan_wgs_toolkit.phewas_runtime import runtime
from fudan_wgs_toolkit.phewas_runtime.metadata import SharedMetadataReader
from fudan_wgs_toolkit.phewas_runtime.shared_state import SharedStateBroker
from test_gaussian_resident_shared import model as state_model
from test_phewas_batched_runtime import cpu_product_oracle, descriptor
from test_phewas_gene_preparation_geometry import _Reader
from test_phewas_shared_state import make_cache


class _ResidentBranchMarker(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def fitted(index, rows, incompatible):
    model = state_model(len(rows), shift=.1*index)
    model.sample_ids = np.asarray([f'sample{i}' for i in rows])
    if index in incompatible:
        model.spectrum.blocks[0][1].copy_(torch.tensor([[.6, -.8], [.8, .6]]))
    return model


class EphemeralRepository:
    def __init__(self, rows, incompatible):
        self.items = [descriptor(i, rows) for i in range(3)]
        self.incompatible = incompatible
        self.model_refs, self.state_refs, self.acquired = [], [], []
        self.active = 0

    def descriptors(self):
        return self.items

    @contextmanager
    def acquire(self, indices, device):
        assert self.active == 0
        assert all(reference() is None for reference in self.model_refs+self.state_refs)
        values = [fitted(index, self.items[index]['ordered_rows'], self.incompatible) for index in indices]
        self.model_refs.extend(weakref.ref(value) for value in values)
        self.state_refs.extend(weakref.ref(value.precision_x) for value in values)
        self.acquired.append(list(indices))
        self.active += 1
        try:
            yield values
        finally:
            values.clear()
            self.active -= 1


def install_preparation_observers(monkeypatch, owner_refs, close_receipts):
    for module, cls, name in (
        (resident, resident.SharedGaussianResidentPreparation, 'resident'),
        (panels, panels.SharedGaussianPanelBank, 'host')):
        original = cls.close
        def close(self, original=original, name=name):
            original(self)
            close_receipts.append(dict(kind=name, storage_bytes=self.storage_bytes,
                                       model_retained=(getattr(self, '_cache', None) is not None
                                           and self._cache.model is not None),
                                       genotype_retained=getattr(self, '_genotype', None) is not None))
        monkeypatch.setattr(cls, 'close', close)
    real_resident, real_panels = resident.shared_gaussian_resident, panels.shared_gaussian_panels
    def resident_scope(*args, **kwargs):
        # Real CUDA availability is deliberately not queried by this CPU test.
        kwargs['capacity_bytes'] = 2**20
        owner = real_resident(*args, **kwargs)
        owner_refs.append(weakref.ref(owner))
        return owner
    def panel_scope(*args, **kwargs):
        kwargs['capacity_bytes'] = 2**20
        owner = real_panels(*args, **kwargs)
        owner_refs.append(weakref.ref(owner))
        return owner
    monkeypatch.setattr(resident, 'shared_gaussian_resident', resident_scope)
    monkeypatch.setattr(panels, 'shared_gaussian_panels', panel_scope)


@pytest.mark.parametrize('route', ['resident_marker', 'host'])
@pytest.mark.parametrize('failure', [None, 'score', 'covariance', 'tail', 'writer'])
@pytest.mark.parametrize('incompatible', [{1}, {2}])
def test_mask_scope_closes_models_preparations_and_reader_after_real_runtime_dispatch(
        tmp_path, monkeypatch, cpu_product_oracle, route, failure, incompatible):
    n, m = 256, 8
    rows = np.arange(n)
    raw = np.zeros((m, n), dtype=np.uint8)
    raw[np.arange(m), np.arange(m)] = 1
    masks = dict(plof=np.arange(4), missense=np.arange(3, 6))
    repository = EphemeralRepository(rows, incompatible)
    _, groups = runtime._descriptor_groups(repository)
    owner_refs, tensor_refs, closed, future, observed, writes = [], [], [], [], [], []
    required_events = []
    original_required = runtime._DirectWorkspace.set_required
    def record_required(self, required, *, phase='workspace'):
        required_events.append((required, phase))
        return original_required(self, required, phase=phase)
    monkeypatch.setattr(runtime._DirectWorkspace, 'set_required', record_required)
    install_preparation_observers(monkeypatch, owner_refs, closed)
    real_prepare = runtime._prepare_shared_gene
    def prepare(*args, **kwargs):
        prepared = real_prepare(*args, **kwargs)
        if prepared is not None:
            host = prepared['_genotype_host']
            tensor_refs.append(weakref.ref(host))
            if route == 'resident_marker':
                marker = torch.from_numpy(host).as_subclass(_ResidentBranchMarker)
                prepared['_genotype_device'] = marker
                prepared.pop('_genotype_host')
                tensor_refs.append(weakref.ref(marker))
        return prepared
    monkeypatch.setattr(runtime, '_prepare_shared_gene', prepare)
    monkeypatch.setattr(runtime.LimitedMaskPipeline, '_select_mask_chunks', lambda *args: masks)
    monkeypatch.setattr(runtime.LimitedMaskPipeline, '_supports_long_tf32_products', lambda *args: False)
    # Host CPU runtime otherwise uses its compatibility oracle. Route through
    # the same mature factory-capable blocked helper used by CUDA production.
    def blocked_covariance(model, host, broker, **kwargs):
        kwargs['capacity_bytes'] = 2**20
        output = blocked_gaussian_covariance(model, host, broker, **kwargs)
        tensor_refs.append(weakref.ref(output))
        if failure == 'covariance': raise ValueError('injected covariance failure')
        return output
    monkeypatch.setattr(runtime, '_blocked_gaussian_covariance', blocked_covariance)
    real_covariance = runtime._gaussian_covariance
    def covariance(model, genotype, *, rotated=None, cross=None, **kwargs):
        if rotated is not None:
            tensor_refs.append(weakref.ref(rotated))
        observed.append((model, rotated is not None, cross is not None))
        # Retain only a weak model marker in the external observation.
        observed[-1] = (weakref.ref(model), rotated is not None, cross is not None)
        output = real_covariance(model, genotype, rotated=rotated, cross=cross, **kwargs)
        tensor_refs.append(weakref.ref(output))
        if failure == 'covariance': raise ValueError('injected covariance failure')
        return output
    monkeypatch.setattr(runtime, '_gaussian_covariance', covariance)
    def scores(genotype, models, broker):
        dense = torch.as_tensor(genotype)
        output = torch.stack(tuple(dense.T@model.scaled_residuals for model in models), dim=1)
        tensor_refs.append(weakref.ref(output))
        if failure == 'score': raise ValueError('injected score failure')
        return output
    monkeypatch.setattr(runtime, '_shared_gene_scores', scores)
    def evaluate(pipeline, payload, model):
        tensor_refs.append(weakref.ref(payload['covariance']))
        if failure == 'tail': raise ValueError('injected tail failure')
        return dict(num_variant=len(payload['maf']), cMAC=payload['cmac'], **{'WGS-O': .25})
    monkeypatch.setattr(runtime.LimitedMaskPipeline, '_evaluate_prepared', evaluate)
    def write(*args, **kwargs):
        assert all(reference() is None for reference in owner_refs)
        assert all(reference() is None for reference in tensor_refs)
        writes.append(args)
        if failure == 'writer': raise OSError('injected writer failure')
    options = AnalysisOptions(wrapper_semantics='base', memory_limit_gib=80, rv_num_cutoff=2,
                              variant_tile_size=3)
    config = dict(_continuous_shared_panels=True, _continuous_resident_preparation=True,
                  _host_memory_set_future=lambda value, **kwargs: future.append((value, kwargs['phase'])))
    report = dict(association_rows={'coding': 0}, trait_rows={i: {'coding': 0} for i in range(3)},
                  binary_spa_selected=0, binary_spa_failed=0)
    retained = None
    with SharedStateBroker(SharedMetadataReader(_Reader(n, m)), make_cache(tmp_path/'cache', raw),
                           device='cpu') as broker:
        axis = broker.reader_view(rows)
        structural = runtime._thin_pipeline(axis, SimpleNamespace(n=n, device='cpu', use_spa=False),
                                             axis._axis.samples, options, config)
        initial_pins = axis._axis.pins
        try:
            runtime._batched_gene(groups, repository, broker, broker._reader, options, config,
                dict(gene_name='anonymous_gene', start=0, end=m-1), SimpleNamespace(write=write),
                report, set(), '21', 0, 2, structural, 'coding')
        except (ValueError, OSError) as error:
            retained = error
        if failure is None:
            assert retained is None and len(writes) == 3
            assert len(closed) == 2
            if route == 'resident_marker':
                expected_preparation = ([(True, False), (False, False), (True, True)]
                    if incompatible == {1} else [(True, True), (True, True), (False, False)])
                assert [(r, c) for _, r, c in observed] == expected_preparation*2
                expected_new = []
                for width in (4, 3):
                    base = 32*width*width+256*2**20
                    expected_new.extend(4*n*width*(2 if i in incompatible else 1)+base
                                        for i in range(3))
                # R was materialized and pinned BEFORE each NEW-byte check.
                # It must not be added to the allocator and the future twice.
                assert [value for value, phase in required_events
                        if phase == 'gene_covariance'] == expected_new
        else:
            assert retained is not None and 'injected' in str(retained)
            assert closed
        assert all(reference() is None for reference in repository.model_refs+repository.state_refs)
        assert all(reference() is None for reference in owner_refs+tensor_refs)
        assert all(item['storage_bytes'] == 0 and not item['model_retained']
                   and not item['genotype_retained'] for item in closed)
        assert not broker._optional_device_cache_releases
        assert repository.active == 0 and axis._axis.pins == initial_pins
        assert future[-1] == (0, 'gene_mask_released')
        axis.close()


def test_runtime_covariance_hook_preserves_fallback_and_independent_fitted_operands(cpu_product_oracle):
    models = [state_model(37), state_model(37, shift=.2)]
    genotype = torch.linspace(-1., 1., 37*7).reshape(37, 7)
    broker = SimpleNamespace(_guard=lambda _: None)
    controls = [runtime._gaussian_covariance(model, genotype) for model in models]
    with resident.shared_gaussian_resident(models[0], genotype, broker, capacity_bytes=2**20) as prep:
        crosses = prep.batched_cross(models)
        with prep.pin():
            for model, cross, control in zip(models, crosses, controls):
                actual = runtime._gaussian_covariance(model, genotype,
                    rotated=prep.rotated_for(model), cross=cross)
                torch.testing.assert_close(actual, control, rtol=2e-6, atol=2e-6)
        models[1].spectrum.blocks[0][1].add_(.01)
        assert prep.batched_cross(models) == (None, None)
        assert prep.rotated_for(models[1]) is None
        actual = runtime._gaussian_covariance(models[1], genotype, rotated=None, cross=None)
        torch.testing.assert_close(actual, runtime._gaussian_covariance(models[1], genotype), rtol=0, atol=0)


def test_actual_repository_admission_counts_current_model_and_rotation_only_in_live_counter(monkeypatch):
    from fudan_wgs_toolkit.phewas_storage import ModelRepository
    from test_phewas_workspace import entry, simulated_cuda
    repository = ModelRepository([], workspace_bytes=0)
    key = (0, 'cuda:0')
    repository.models[key] = entry(120)
    repository.pinned[key] = 1
    accounting = simulated_cuda(monkeypatch, repository, total=520)
    # Explicit mocked allocator: borrowed G=100, cached R=80, cross=40.
    # They and the current 120-byte model are already in the live counter.
    accounting['external'] = 220
    with repository.reserve_workspace('cuda:0', 120, reserve_bytes=0) as workspace:
        # Actual mature admission uses max(future, extra)+NEW model exposure,
        # not future+extra, and never adds the resident model or R a second time.
        repository._admit('cuda:0', extra_bytes=60, fitted_bytes=40)
        assert repository.summary()['workspace_required_bytes_highwater'] == 160
        workspace.set_required(80, phase='rotation_is_already_live')
        workspace.ensure_free(20, phase='new_weighted_product')
        assert repository.models[key].bytes == 120 and repository.pinned[key] == 1
        assert accounting['emptied'] == 0
    assert not repository._workspace_reservations
