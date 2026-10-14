"""Bounded decoding and numeric contracts; real-mask GPU benchmark is separate."""
import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.binary import (ExactBinaryBurdenCache, prepare_binary_burdens,
    staar_binary_phewas)
from fudan_wgs_toolkit.binary_gene_batch import BinaryHostPanelSource
from fudan_wgs_toolkit.genotype_device import DeviceMinorBlock
from fudan_wgs_toolkit.phewas_models import prepare_phewas_model
from fudan_wgs_toolkit.phewas_runtime import runtime
from test_phewas_device_gene_prepare import fixture_pipeline


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')


@pytest.mark.parametrize('imputation', ['mean', 'minor'])
@pytest.mark.parametrize('rows', [np.arange(512)[::-1].copy(), np.arange(128)[::-1].copy()])
def test_cpu_device_panel_compatibility_preserves_order_and_values(tmp_path, imputation, rows):
    with fixture_pipeline(tmp_path/'cache', device='cpu', rows=rows, imputation=imputation) as (p, broker):
        p._binary_burden_only = True
        source = runtime._prepare_shared_gene(p, np.arange(12), broker)['_genotype_host']
        columns = np.asarray([source.shape[1]-1, 0, 2])
        expected = source.read_columns(columns)
        actual = source.read_device_columns(columns, 'cpu')
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.stride() == expected.stride()


@CUDA
@pytest.mark.parametrize('imputation', ['mean', 'minor'])
@pytest.mark.parametrize('cohort', ['reversed', 'subset'])
def test_cuda_panels_avoid_host_payload_roundtrip_and_preserve_exact_spa_inputs(
        tmp_path, monkeypatch, imputation, cohort):
    rows = np.arange(512)[::-1].copy() if cohort == 'reversed' else np.arange(128)[::-1].copy()
    with fixture_pipeline(tmp_path/'cache', device='cuda:0', rows=rows, imputation=imputation) as (p, broker):
        p._binary_burden_only = True
        p.resident_genotypes = False  # Exercise long-mask recipe on a small exact fixture.
        prepared = runtime._prepare_shared_gene(p, np.arange(12), broker)
        source = prepared['_genotype_host']
        assert isinstance(source, BinaryHostPanelSource)
        columns = np.arange(source.shape[1])[::-1].copy()
        expected = source.read_columns(columns)
        # Build a full host oracle before forbidding further host payloads.
        host = source.read_columns(np.arange(source.shape[1]))
        fitted = prepare_phewas_model((np.arange(len(rows)) % 7 == 0).astype(float),
            source.sample_ids, np.column_stack((np.ones(len(rows)), np.linspace(-1, 1, len(rows)))),
            association_mode='fp64', device='cuda:0')
        monkeypatch.setattr(BinaryHostPanelSource, 'read_columns',
            lambda *args: pytest.fail('CUDA path read a floating CPU panel'))
        monkeypatch.setattr(DeviceMinorBlock, 'to_host',
            lambda *args: pytest.fail('CUDA path copied dosage to CPU'))
        actual = source.read_device_columns(columns, 'cuda:0')
        assert actual.is_cuda and actual.dtype == torch.float32
        assert actual.stride() == (1, len(rows))
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        expected_burdens = prepare_binary_burdens(fitted.model, host, prepared['maf'])
        actual_burdens = prepare_binary_burdens(fitted.model, source, prepared['maf'])
        torch.testing.assert_close(actual_burdens.burdens, expected_burdens.burdens, rtol=0, atol=0)
        assert actual_burdens.cmac == expected_burdens.cmac
        index = torch.tensor([0, 1], device='cuda:0')
        actual_cache = ExactBinaryBurdenCache(actual_burdens)
        expected_cache = ExactBinaryBurdenCache(expected_burdens)
        torch.testing.assert_close(actual_cache.get(actual_burdens, index),
            expected_cache.get(expected_burdens, index), rtol=0, atol=0)
        actual_p = staar_binary_phewas(fitted.model, source, prepared['maf'],
            prepared_burdens=actual_burdens, spa_model=fitted.spa_model,
            _exact_burden_cache=actual_cache, p_filter_cutoff=1)
        expected_p = staar_binary_phewas(fitted.model, host, prepared['maf'],
            prepared_burdens=expected_burdens, spa_model=fitted.spa_model,
            _exact_burden_cache=expected_cache, p_filter_cutoff=1)
        assert actual_p == expected_p


@CUDA
def test_cuda_panel_denial_precedes_reader_and_output(tmp_path, monkeypatch):
    with fixture_pipeline(tmp_path/'cache', device='cuda:0', rows=np.arange(512)) as (p, broker):
        p._binary_burden_only = True
        p.resident_genotypes = False
        source = runtime._prepare_shared_gene(p, np.arange(12), broker)['_genotype_host']
        def denied(*args):
            raise MemoryError('device panel denied')
        object.__setattr__(source, 'host_guard', denied)
        object.__setattr__(source, 'host_ready', lambda: pytest.fail('reset denied lease'))
        monkeypatch.setattr(broker, 'read_states', lambda *args: pytest.fail('read before admission'))
        with pytest.raises(MemoryError, match='denied'):
            source.read_device_columns(np.arange(2), 'cuda:0')
