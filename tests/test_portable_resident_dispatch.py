"""Portable resident-route admission contracts; no GPU benchmark claims."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.cache_runtime.portable import PortableGenotypeReader
from fudan_wgs_toolkit.null_model import GaussianNullModel
from fudan_wgs_toolkit.pipeline import AnalysisOptions, PheWASPipeline
from fudan_wgs_toolkit import pipeline as pipeline_module
from test_cache_portable import portable


def analysis_fixture(reader, monkeypatch, *, free_bytes=2**30,
                     allocated_bytes=0, reserved_bytes=0):
    model = GaussianNullModel.__new__(GaussianNullModel)
    model.sample_ids = np.arange(100).astype(str)
    model.x = SimpleNamespace(device=torch.device("cuda:0"),dtype=torch.float32)
    model.matmul_mode = "tf32"
    model.n_pheno, model.use_spa = 1, False
    analysis = PheWASPipeline.__new__(PheWASPipeline)
    analysis.models = [model]
    analysis.genotype = reader
    analysis.options = AnalysisOptions(memory_limit_gib=.5)
    analysis.resident_genotypes = True
    analysis.union_rows = np.arange(100)
    analysis.local_mask_reuse_counters = {}
    monkeypatch.setattr(pipeline_module,"allocator_bytes",lambda *args:(allocated_bytes,reserved_bytes))
    monkeypatch.setattr(torch.cuda,"mem_get_info",lambda *args:(free_bytes,2**30))
    return analysis


def test_actual_portable_reader_dispatches_gene_resident_without_sdk(portable,monkeypatch):
    _, cache, _, _, _, _ = portable
    with PortableGenotypeReader(cache,device="cuda:0") as reader:
        assert getattr(reader,"_flat_reader",None) is None
        analysis = analysis_fixture(reader,monkeypatch)
        analysis.union_rows = np.arange(reader.n_samples)
        sentinel, calls = object(), []
        def prepare(indices,annotations,options,masks,defer):
            calls.append((indices.copy(),options,masks,defer))
            return sentinel
        analysis._prepare_resident_gene = prepare
        indices = np.asarray([5,1,3],dtype=np.int64)
        masks = [np.asarray([1,3])]
        result = analysis._prepare_test_set(indices,_local_masks=masks,_defer_score=True)
        assert result is sentinel
        np.testing.assert_array_equal(calls[0][0],indices)
        assert calls[0][1] == dict(device=torch.device("cuda:0"),resident=True)
        assert calls[0][2] is masks and calls[0][3] is True
        assert analysis.local_mask_reuse_counters["resident_gene_families"] == 1
        assert reader.reader_metadata["original_genotype_required"] is False


@pytest.mark.parametrize("reader,expected",[
    (SimpleNamespace(_flat_reader=object(),genotype_raw_memory_bytes=0,n_samples=100),True),
    (SimpleNamespace(_flat_reader=None,genotype_raw_memory_bytes=0,n_samples=100),False),
    (SimpleNamespace(genotype_raw_memory_bytes=0,n_samples=100),False),
    (SimpleNamespace(supports_resident_minor_blocks=False,_flat_reader=object(),genotype_raw_memory_bytes=0,n_samples=100),False),
])
def test_legacy_readers_and_explicit_opt_out_remain_compatible(reader,expected,monkeypatch):
    analysis = analysis_fixture(reader,monkeypatch)
    assert bool(analysis._resident_gene_reader_options(np.arange(10))) is expected


@pytest.mark.parametrize("reason",["cap","live","fits"])
def test_portable_capability_preserves_storage_and_live_memory_guards(monkeypatch,reason):
    reader = SimpleNamespace(supports_resident_minor_blocks=True,
        genotype_raw_memory_bytes=1024,n_samples=1000)
    allocated = 400*2**20 if reason=="cap" else 0
    analysis = analysis_fixture(reader,monkeypatch,
        allocated_bytes=allocated,reserved_bytes=allocated,
        free_bytes=1024 if reason=="live" else 2**30)
    result = analysis._resident_gene_reader_options(np.arange(100))
    assert bool(result) is (reason=="fits")
    assert analysis.local_mask_reuse_counters["resident_gene_storage_reserve_bytes_max"] > 256*2**20
    if reason!="fits":
        assert analysis.local_mask_reuse_counters["resident_gene_cpu_route_budget"] == 1


@pytest.mark.parametrize("mode",["disabled","fp64","cpu","spa","multi"])
def test_capability_does_not_bypass_model_compatibility(monkeypatch,mode):
    reader = SimpleNamespace(supports_resident_minor_blocks=True,
        genotype_raw_memory_bytes=0,n_samples=100)
    analysis = analysis_fixture(reader,monkeypatch)
    model = analysis.models[0]
    if mode=="disabled":analysis.resident_genotypes=False
    elif mode=="fp64":model.matmul_mode="fp64"
    elif mode=="cpu":model.x.device=torch.device("cpu")
    elif mode=="spa":model.use_spa=True
    else:model.n_pheno=2
    def fail(*args):pytest.fail("unsupported route inspected CUDA memory")
    monkeypatch.setattr(torch.cuda,"mem_get_info",fail)
    assert analysis._resident_gene_reader_options(np.arange(10)) == {}
