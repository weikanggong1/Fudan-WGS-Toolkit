"""Runtime ownership under a retained tail exception; no CUDA speed claim."""
from contextlib import contextmanager, nullcontext
import gc
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref
import numpy as np
import torch
from fudan_wgs_toolkit.phewas_runtime import runtime
from fudan_wgs_toolkit.pipeline import AnalysisOptions


class Profiler:
    def measure(self, *args, **kwargs):
        return nullcontext()

class Broker:
    device = torch.device("cpu")
    def __init__(self, available=2**30):
        self.available = available
        self.requests = []
        self.closed_axes = 0

    def _guard(self, required):
        self.requests.append(required)
        if required > self.available:
            raise MemoryError("live broker capacity")

    def _live_available_bytes(self):
        return self.available

    def reader_view(self, rows):
        broker = self
        class Axis:
            _axis = SimpleNamespace(samples=np.asarray(rows))
            def close(self):
                broker.closed_axes += 1
        return Axis()

def gaussian_model(index, n=8):
    x = torch.ones((n, 1), dtype=torch.float32)
    inv = torch.linspace(.5, 1.5, n)*(index+1)
    return SimpleNamespace(index=index, family="gaussian", n=n, x=x, device=torch.device("cpu"),
        inverse_variance=inv, precision_x=x*inv[:, None],
        fixed_effect_covariance=torch.tensor([[1/float(inv.sum())]]),
        spectrum=SimpleNamespace(blocks=[]))

class GeneTailOwnerReleaseTests(unittest.TestCase):
    def test_held_second_tail_exception_releases_loop_model_and_no_partial_write(self):
        """An ephemeral repository isolates evaluate's aliases from its cache."""
        (refs, calls, writes) = ([], [], [])

        class Model:
            pass

        class Repository:
            active = 0

            @contextmanager
            def acquire(self, indices, device):
                models = []
                model = None
                self.active += 1
                try:
                    for index in indices:
                        model = Model()
                        model.__dict__.update(gaussian_model(index).__dict__)
                        models.append(model)
                        if index == 1:
                            refs.extend((weakref.ref(model), weakref.ref(model.x)))
                    model = None
                    yield models
                finally:
                    models.clear()
                    models = model = None
                    self.active -= 1

        class Pipeline:

            def __init__(self, model):
                self.models = [model]
                self.profiler = Profiler()

            def _supports_long_tf32_products(self, model):
                return False

            def ncrna(self, **kwargs):
                return self._run_mask_sets([np.arange(5)])

        def thin(axis, model, rows, options, configuration, structural=None):
            return Pipeline(model)

        def prepare(*args):
            source = np.arange(40, dtype=np.float32).reshape(8, 5)
            refs.append(weakref.ref(source))
            return dict(_genotype_host=source, maf=np.full(5, 0.005), annotations=np.ones((5, 1)), names=['synthetic'], cmac=5)

        def score(genotype, models, broker):
            return torch.zeros((genotype.shape[1], len(models)))

        def covariance(model, host, broker, **kwargs):
            return torch.eye(host.shape[1])

        def evaluate(pipeline, payload, model, genotype, broker, **kwargs):
            calls.append(model.index)
            if model.index == 1:
                # Deliberately retain leaf arguments in the failure traceback.
                raise ArithmeticError("second tail failure")
            return {"WGS-O": .25}
        (repository, broker) = (Repository(), Broker())
        descriptors = [dict(trait_index=i, family='gaussian') for i in range(2)]
        groups = [dict(rows=np.arange(8), descriptors=descriptors)]
        structural = SimpleNamespace(genotype=None, models=[SimpleNamespace(use_spa=False)], union_rows=np.arange(8))
        report = dict(association_rows={'ncrna': 0}, trait_rows={i: {'ncrna': 0} for i in range(2)}, binary_spa_selected=0, binary_spa_failed=0)
        writer = SimpleNamespace(write=lambda *args, **kwargs: writes.append(args))
        held = None
        with patch.object(runtime, '_thin_pipeline', new=thin), patch.object(runtime, '_check_model', new=lambda *args: None), patch.object(runtime, '_prepare_shared_gene', new=prepare), patch.object(runtime, '_shared_gene_scores', new=score), patch.object(runtime, '_blocked_gaussian_covariance', new=covariance), patch.object(runtime, '_evaluate_gaussian_gene_score', new=evaluate):
            try:
                runtime._batched_gene(groups, repository, broker, None, AnalysisOptions(), {}, {}, writer, report, set(), 'synthetic', 0, 2, structural, 'ncrna')
            except ArithmeticError as error:
                held = error
        gc.collect()
        self.assertIsNotNone(held.__traceback__)
        self.assertEqual(calls, [0, 1])
        self.assertFalse(writes)
        self.assertEqual(report['association_rows']['ncrna'], 0)
        self.assertEqual(repository.active, 0)
        self.assertEqual(broker.closed_axes, 1)
        self.assertTrue(all((ref() is None for ref in refs)))

    def test_chained_failure_releases_leaf_storage_without_losing_error_evidence(self):
        refs = []
        def leaf():
            value = torch.ones(32, 32)
            refs.append(weakref.ref(value))
            raise ValueError("leaf arithmetic detail")
        def wrapped():
            value = torch.ones(16, 16)
            refs.append(weakref.ref(value))
            try:
                leaf()
            except ValueError as error:
                raise ArithmeticError("original public failure") from error
        held = None
        try:
            wrapped()
        except ArithmeticError as error:
            held = error
            runtime._clear_inactive_exception_frames(error)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(str(held), "original public failure")
        self.assertEqual(str(held.__cause__), "leaf arithmetic detail")
        self.assertIsNotNone(held.__traceback__)
        self.assertIsNotNone(held.__cause__.__traceback__)
