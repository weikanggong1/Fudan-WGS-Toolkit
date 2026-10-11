"""Selected device dosage semantics; tiny fixtures are not speed benchmarks."""
import numpy as np
import pytest
import torch

from fudan_wgs_toolkit.genotype_device import DeviceMinorBlock


def make_block(device='cpu'):
    dosage = np.asarray([[0, 3, 1, 2, 3, 0], [1, 3, 3, 0, 2, 2],
                         [3, 3, 2, 1, 0, 1], [2, 3, 0, 3, 1, 3],
                         [0, 3, 1, 1, 2, 0]], dtype=np.uint8)
    n, m = dosage.shape
    ref = np.asarray([.8, np.nan, .5, .03, .92, .71])
    block = DeviceMinorBlock(torch.tensor(dosage, device=device), np.arange(n),
        np.arange(m), ref, np.asarray([2., np.nan, 4., 1., 3., 2.]),
        np.asarray([.1, 1., .1, .1, .2, .1]), np.asarray([8., 0., 5., 1., 9., 7.]),
        np.asarray([9, 0, 9, 9, 8, 9]))
    return block


DEVICES = ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable for selected dosage kernel'))]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('frequency_mode', ['count', 'reference'])
@pytest.mark.parametrize('imputation', ['mean', 'minor'])
def test_selected_columns_equal_existing_dense_including_missing_ties_and_order(
        device, dtype, frequency_mode, imputation):
    block = make_block(device)
    columns = np.asarray([4, 1, 3, 0, 5])
    rows = np.arange(4, -1, -1)
    expected = block.select_columns(columns).trait_dense(rows, imputation,
        frequency_mode=frequency_mode, dtype=dtype)
    actual = block.trait_dense_columns(rows, columns, imputation,
        frequency_mode=frequency_mode, dtype=dtype)
    np.testing.assert_array_equal(actual[0].cpu().numpy(), expected[0].cpu().numpy())
    for candidate, reference in zip(actual[1:], expected[1:]):
        np.testing.assert_array_equal(candidate, reference)


@pytest.mark.parametrize('device', DEVICES)
def test_cached_counts_write_sorted_destination_without_selected_uint8_or_host_payload(
        device, monkeypatch):
    block = make_block(device)
    rows, columns = np.arange(block.shape[0]), np.asarray([5, 3, 0, 2])
    expected = block.select_columns(columns).trait_dense(rows, frequency_mode='reference',
        dtype=torch.float32)
    block.trait_summary(rows, frequency_mode='reference')
    def forbidden(*args, **kwargs):
        raise AssertionError('cached summaries must not gather a full selected/missing matrix')
    monkeypatch.setattr(block, '_trait_counts', forbidden)
    monkeypatch.setattr(block, 'select_columns', forbidden)
    output = torch.full((block.shape[0], 8), -7., dtype=torch.float32, device=device)
    destination = np.asarray([6, 1, 4, 3])
    if device.startswith('cuda'):
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, 'cpu', forbidden)
            result = block.trait_dense_columns(rows, columns, frequency_mode='reference',
                dtype=torch.float32, out=output, out_columns=destination)
    else:
        result = block.trait_dense_columns(rows, columns, frequency_mode='reference',
            dtype=torch.float32, out=output, out_columns=destination)
    assert result[0] is output
    actual = output.cpu().numpy()
    np.testing.assert_array_equal(actual[:, destination], expected[0].cpu().numpy())
    np.testing.assert_array_equal(actual[:, [0, 2, 5, 7]], np.full((5, 4), -7.))
    for candidate, reference in zip(result[1:], expected[1:]):
        np.testing.assert_array_equal(candidate, reference)
    result[1][:] = -1
    np.testing.assert_array_equal(block.trait_summary(rows, frequency_mode='reference')[0],
        np.minimum(block.union_ref_af, 1-block.union_ref_af))


def test_selected_complete_case_rows_and_noncontiguous_row_major_output():
    block = make_block()
    rows, columns = np.asarray([4, 0, 2]), np.asarray([3, 5, 0])
    expected = block.select_columns(columns).trait_dense(rows, dtype=torch.float64)
    owner = torch.full((3, 12), -9., dtype=torch.float64)
    out = owner[:, ::2]
    actual = block.trait_dense_columns(rows, columns, dtype=torch.float64,
        out=out, out_columns=np.asarray([5, 1, 3]))
    assert actual[0] is out
    np.testing.assert_array_equal(out[:, [5, 1, 3]].numpy(), expected[0].numpy())
    np.testing.assert_array_equal(owner[:, 1::2].numpy(), np.full((3, 6), -9.))
    for candidate, reference in zip(actual[1:], expected[1:]):
        np.testing.assert_array_equal(candidate, reference)


@pytest.mark.parametrize('device', DEVICES)
def test_column_major_output_preserves_original_gene_layout_and_untouched_columns(device):
    block = make_block(device)
    rows, columns = np.arange(5), np.asarray([0, 4, 3])
    expected = block.select_columns(columns).trait_dense(rows, frequency_mode='reference',
        dtype=torch.float32)
    output = torch.full((7, 5), -4., dtype=torch.float32, device=device).T
    original_strides = output.stride()
    result = block.trait_dense_columns(rows, columns, frequency_mode='reference',
        dtype=torch.float32, out=output, out_columns=np.asarray([6, 1, 4]))
    assert result[0] is output and output.stride() == original_strides == (1, 5)
    np.testing.assert_array_equal(output[:, [6, 1, 4]].cpu().numpy(), expected[0].cpu().numpy())
    np.testing.assert_array_equal(output[:, [0, 2, 3, 5]].cpu().numpy(), np.full((5, 4), -4.))


def test_mutated_lossless_dosage_invalidates_selected_summary_cache():
    block = make_block()
    rows, columns = np.arange(block.shape[0]), np.asarray([0, 3])
    block.trait_summary(rows)
    block.dosage[0, 0] = 2
    expected = block.select_columns(columns).trait_dense(rows, dtype=torch.float32)
    actual = block.trait_dense_columns(rows, columns, dtype=torch.float32)
    for candidate, reference in zip(actual, expected):
        np.testing.assert_array_equal(candidate.numpy() if isinstance(candidate, torch.Tensor) else candidate,
            reference.numpy() if isinstance(reference, torch.Tensor) else reference)


@pytest.mark.parametrize('kwargs,match', [
    ({'dtype': torch.float16}, 'dosage dtype'),
    ({'out_columns': np.asarray([1, 0])}, 'requires an output'),
    ({'dtype': torch.float32, 'out': torch.empty(4, 3)}, 'output dosage'),
    ({'dtype': torch.float32, 'out': torch.empty(5, 3), 'out_columns': np.asarray([0])}, 'one unique'),
    ({'dtype': torch.float32, 'out': torch.empty(5, 3), 'out_columns': np.asarray([0, 0])}, 'duplicate'),
    ({'dtype': torch.float32, 'out': torch.empty(10).as_strided((5, 3), (1, 1))}, 'output dosage'),
    ({'dtype': torch.float32, 'out': torch.empty(1, 3).expand(5, 3)}, 'output dosage'),
])
def test_invalid_selected_output_rejected_before_counting(monkeypatch, kwargs, match):
    block = make_block()
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid geometry must not enter count/allocation')
    monkeypatch.setattr(block, '_trait_counts', forbidden)
    with pytest.raises(ValueError, match=match):
        block.trait_dense_columns(np.arange(5), np.asarray([0, 3]), **kwargs)


def test_empty_selected_columns_and_rows_preserve_dense_contract():
    block = make_block()
    empty = np.asarray([], dtype=np.int64)
    assert block.trait_dense_columns(np.arange(5), empty)[0].shape == (5, 0)
    result = block.trait_dense_columns(empty, np.asarray([0, 3]))
    assert result[0].shape == (0, 2)
    np.testing.assert_array_equal(result[2], [0., 0.])


def test_output_cannot_overwrite_lossless_input_storage_before_summary(monkeypatch):
    block = make_block()
    # A dtype view can otherwise have valid output geometry while mutating
    # the supposedly read-only source dosage used by every later trait.
    owner = torch.zeros((5, 8), dtype=torch.uint8)
    block.dosage = owner[:, :6]
    output = owner.view(torch.float32)
    def forbidden(*args, **kwargs):
        raise AssertionError('aliased output must be rejected before reading summaries')
    monkeypatch.setattr(block, '_trait_counts', forbidden)
    with pytest.raises(ValueError, match='output dosage'):
        block.trait_dense_columns(np.arange(5), np.asarray([0, 3]),
            dtype=torch.float32, out=output)
    np.testing.assert_array_equal(owner.numpy(), np.zeros((5, 8), dtype=np.uint8))
