"""Lossless host preparation contracts; fixtures are not real-data benchmarks."""
import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.genotype_device import DeviceMinorBlock
from fudan_wgs_toolkit.pipeline import _cmac_scalar_sum
from fudan_wgs_toolkit.phewas_runtime import runtime
from test_genotype_device_columns import make_block
from test_phewas_device_gene_prepare import fixture_pipeline


DEVICES = ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable for exact host transfer'))]


@pytest.mark.parametrize('device', DEVICES)
def test_host_transfer_keeps_verified_counts_without_reducing_dosage_again(device, monkeypatch):
    block = make_block(device)
    rows = np.arange(block.shape[0])
    expected = block.trait_dense(rows, frequency_mode='reference', dtype=torch.float32)
    compact = block.select_columns(np.asarray([5, 3, 0]))
    host = compact.to_host()
    assert host.dosage.device.type == 'cpu'
    if device == 'cpu':
        assert host is compact
    np.testing.assert_array_equal(host.dosage.numpy(), compact.dosage.cpu().numpy())
    monkeypatch.setattr(host, '_trait_counts', lambda *args: pytest.fail(
        'host preparation repeated already verified integer reductions'))
    out = torch.empty((3, len(rows)), dtype=torch.float32).T
    actual = host.trait_dense_columns(rows, np.arange(3), frequency_mode='reference',
        dtype=torch.float32, out=out)
    np.testing.assert_array_equal(actual[0].numpy(), expected[0][:, [5, 3, 0]].cpu().numpy())
    for value, reference in zip(actual[1:], expected[1:]):
        np.testing.assert_array_equal(value, reference[[5, 3, 0]])


@pytest.mark.parametrize('device', DEVICES)
def test_host_transfer_does_not_restore_counts_invalidated_by_source_mutation(device):
    block = make_block(device)
    rows = np.arange(block.shape[0])
    block.trait_summary(rows)
    block.dosage[0, 0] = 2
    host = block.to_host()
    expected = host.trait_dense(rows, dtype=torch.float32)
    actual = host.trait_dense_columns(rows, np.asarray([0]), dtype=torch.float32)
    np.testing.assert_array_equal(actual[0].numpy(), expected[0][:, [0]].numpy())
    for value, reference in zip(actual[1:], expected[1:]):
        np.testing.assert_array_equal(value, reference[[0]])
    assert actual[2][0] == 5  # Includes the changed minor allele.


@pytest.mark.parametrize('imputation', ['mean', 'minor'])
@pytest.mark.parametrize('cohort', ['reversed', 'subset'])
def test_host_gene_preparation_writes_final_matrix_with_mature_af_qc_and_order(
        tmp_path, monkeypatch, imputation, cohort):
    rows = np.arange(512)[::-1].copy() if cohort == 'reversed' else np.arange(128)[::-1].copy()
    indices = np.asarray([5, 2, 9, 4, 1, 6, 0, 8, 3, 7, 10, 11], dtype=np.int64)
    with fixture_pipeline(tmp_path/'cache', device='cpu', rows=rows,
                          imputation=imputation) as (pipeline, broker):
        expected = pipeline._prepare_test_set(indices, _defer_score=True)[0]
        monkeypatch.setattr(DeviceMinorBlock, 'trait_dense', lambda *args, **kwargs:
            pytest.fail('host preparation allocated another floating dosage matrix'))
        monkeypatch.setattr(DeviceMinorBlock, '_trait_counts', lambda *args, **kwargs:
            pytest.fail('host preparation repeated broker integer reductions'))
        actual = runtime._prepare_shared_gene(pipeline, indices, broker)
    assert actual['_genotype_host'].flags.f_contiguous
    assert actual['_genotype_host'].dtype == np.float32
    np.testing.assert_array_equal(actual['_genotype_host'], expected['_genotype_host'].numpy())
    for key in ('maf', 'mac', '_variant_indices', '_extraction_groups', 'annotations'):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual['cmac'] == _cmac_scalar_sum(expected['_genotype_host'])
    assert actual['names'] == expected['names']
