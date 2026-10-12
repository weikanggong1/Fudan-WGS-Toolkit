"""Single CSV scalar, schema, bounded-stream and durable-failure contracts.

These small records exercise output contracts; they are not genetic benchmarks.
"""
import csv
import gc
import gzip
import io
import weakref

import numpy as np
import pytest

from fudan_wgs_toolkit.csv_output import _write_bound_trait_rows
from fudan_wgs_toolkit.phewas_trait_csv import PerTraitCheckpointCSVWriter


BINDING = dict(source_implementation_sha256='source', input_manifest_sha256='input',
               configuration_sha256='ordered-schedule', models={'0': 'model0', '1': 'model1'})


class _OriginalSingleWriter(PerTraitCheckpointCSVWriter):
    """Use the original copied mapping and DictWriter for each Single row."""
    def _append_member(self, info, start, rows, *, trait_index=None):
        if trait_index is not None:
            rows = (dict(trait_index=trait_index, **source) for source in rows)
        return super()._append_member(info, start, rows)


def _single_row(position=1, **extra):
    return dict(CHR='21', POS=position, REF='A', ALT='C', ALT_AF=np.float32(.1),
                MAF=np.float64(.1), N=10, pvalue=0., pvalue_log10=-1500.,
                Score=-0., Score_se=np.float64(1.5), Est=1/3, Est_se=2/3,
                _chunk=0, _common=False, _base_group=0, **extra)


def _write(writer, records, *, block=0):
    return writer.write_tile('individual', '21', records, job_index=0,
                            block_index=block, block_identity=f'physical-{block}',
                            ordinal_before=block*11, ordinal_after=(block+1)*11)


def test_original_and_positional_single_members_receipts_and_ordinals_are_identical(tmp_path):
    ordinary = [_single_row(i, note='comma, quote" newline\nα', missing=None,
                            finite=np.float32(1.234567), infinite=float('inf'),
                            not_a_number=float('nan'), flag=np.bool_(True)) for i in range(1400)]
    binary = [_single_row(2, spa_selected=True, spa_failed=False,
                          spa_used_bisection=True, spa_iterations=7,
                          spa_iteration_limit=False, spa_zero_log_unverified=False)]
    reordered = [{name: value for name, value in reversed(tuple(binary[0].items()))}]
    roots = [tmp_path/'original', tmp_path/'positional']
    reports, records = [], []
    for cls, root in zip((_OriginalSingleWriter, PerTraitCheckpointCSVWriter), roots):
        writer = cls(root, binding=BINDING, max_open_streams=1)
        records.append(_write(writer, [(0, ordinary), (1, binary+reordered+binary)]))
        _write(writer, [(0, binary), (1, ordinary[:1])], block=1)
        reports.append(writer.finish())
    assert records[0] == records[1]
    assert reports[0]['receipts'] == reports[1]['receipts']
    assert reports[0]['committed_blocks_binding_sha256'] == reports[1]['committed_blocks_binding_sha256']
    assert records[1]['ordinal_before'] == 0 and records[1]['ordinal_after'] == 11
    assert records[1]['binary_spa_selected'] == 3 and records[1]['binary_spa_failed'] == 0
    for receipt in reports[0]['receipts']:
        relative = receipt['csv_file']
        assert (roots[0]/relative).read_bytes() == (roots[1]/relative).read_bytes()
        with gzip.open(roots[1]/relative, 'rt', newline='') as stream:
            values = list(csv.DictReader(stream))
        assert values[0]['pvalue'] == '0.0' and values[0]['pvalue_log10'] == '-1500.0'
        assert values[0]['Score'] == '-0.0'


def test_positional_writer_consumes_one_row_at_a_time_and_releases_owners():
    state = {'made': 0, 'written': 0}
    refs = []

    class Row(dict):
        pass

    def records():
        for i in range(30):
            assert state['made'] == state['written']
            row = Row(CHR='21', pvalue=i/100, spa_selected=i%2, spa_failed=False)
            refs.append(weakref.ref(row))
            state['made'] += 1
            yield row

    class Writer:
        def writerows(self, values):
            for value in values:
                assert state['made'] == state['written']+1
                assert value[0] == 3 and value[1] == '21'
                state['written'] += 1

    assert _write_bound_trait_rows(Writer(), records(), 3,
        ('CHR', 'pvalue', 'spa_selected', 'spa_failed')) == (30, 15, 0)
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_single_column_uses_the_original_csv_scalar_not_scalar_iteration():
    output = io.StringIO(newline='')
    assert _write_bound_trait_rows(csv.writer(output), [{'pvalue': .12345678901234567}],
                                 7, ['pvalue']) == (1, 0, 0)
    assert output.getvalue() == '7,0.12345678901234566\r\n'


def test_dictionary_subclass_keeps_original_keyword_expansion_and_flag_values():
    class Row(dict):
        def __getitem__(self, name):
            return 'overridden lookup'
        def get(self, name, default=None):
            return True

    source = Row(CHR='21', pvalue=.25, spa_selected=False, spa_failed=False)
    old, new = io.StringIO(newline=''), io.StringIO(newline='')
    columns = ['trait_index', *source]
    original = csv.DictWriter(old, fieldnames=columns, extrasaction='raise')
    original.writerow(dict(trait_index=7, **source))
    assert _write_bound_trait_rows(csv.writer(new), [source], 7, columns[1:]) == (1, 0, 0)
    assert old.getvalue() == new.getvalue()


@pytest.mark.parametrize('bad', [dict(trait_index=99, CHR='21'), {'': 1, 'CHR': '21'},
                                {1: 2, 'CHR': '21'}, object()])
def test_invalid_single_row_mid_tile_never_commits_a_prefix(tmp_path, bad):
    writer = PerTraitCheckpointCSVWriter(tmp_path, binding=BINDING)
    with pytest.raises((ValueError, TypeError)):
        _write(writer, [(0, [_single_row()]), (1, [_single_row(), bad])])
    assert writer.db.execute('SELECT COUNT(*) FROM blocks').fetchone()[0] == 0
    assert writer.db.execute('SELECT COUNT(*) FROM ranges').fetchone()[0] == 0
    assert all(path.stat().st_size == 0 for path in tmp_path.rglob('*.partial'))
    _write(writer, [(0, [_single_row()]), (1, [])])
    assert writer.finish()['rows']['individual'] == 1


def test_single_commit_failure_resume_keeps_only_authenticated_prior_ranges(tmp_path, monkeypatch):
    writer = PerTraitCheckpointCSVWriter(tmp_path, binding=BINDING)
    first = _write(writer, [(0, [_single_row()]), (1, [_single_row(2)])])
    original = writer._commit_tile
    monkeypatch.setattr(writer, '_commit_tile', lambda *args: (_ for _ in ()).throw(
        OSError('injected Single commit failure')))
    with pytest.raises(OSError, match='Single commit failure'):
        _write(writer, [(0, [_single_row(3)]), (1, [_single_row(4)])], block=1)
    monkeypatch.setattr(writer, '_commit_tile', original)
    writer.abort()
    recovered = PerTraitCheckpointCSVWriter(tmp_path, binding=BINDING, resume=True, worker_id='7')
    assert recovered.checkpoint_tile('individual', '21', 0, [0, 1], block_index=0,
        block_identity='physical-0', ordinal_before=0) == first
    assert recovered.checkpoint_tile('individual', '21', 0, [0, 1], block_index=1,
        block_identity='physical-1', ordinal_before=11) is None
    _write(recovered, [(0, [_single_row(5)]), (1, [])], block=1)
    report = recovered.finish()
    assert report['rows']['individual'] == 3
    assert report['reused_blocks'] == report['computed_blocks'] == 1
    actual = []
    for receipt in report['receipts']:
        with gzip.open(tmp_path/receipt['csv_file'], 'rt', newline='') as stream:
            actual.extend(csv.DictReader(stream))
    assert [row['POS'] for row in actual] == ['1', '5', '2']


def test_single_writer_error_is_primary_and_uncommitted_bytes_are_retired(tmp_path, monkeypatch):
    import fudan_wgs_toolkit.phewas_trait_csv as module
    writer = PerTraitCheckpointCSVWriter(tmp_path, binding=BINDING)
    original = module._write_bound_trait_rows

    def fail_after_one(output, rows, trait, columns):
        def fail_rows():
            for i, row in enumerate(rows):
                if i == 1:
                    raise ArithmeticError('injected Single scalar failure')
                yield row
        return original(output, fail_rows(), trait, columns)

    monkeypatch.setattr(module, '_write_bound_trait_rows', fail_after_one)
    with pytest.raises(ArithmeticError, match='Single scalar failure'):
        _write(writer, [(0, [_single_row(1), _single_row(2)]), (1, [])])
    assert writer.db.execute('SELECT COUNT(*) FROM blocks').fetchone()[0] == 0
    assert all(path.stat().st_size == 0 for path in tmp_path.rglob('*.partial'))
    assert list((tmp_path/'.trait_checkpoints/uncommitted').rglob('tail.gz'))
    monkeypatch.setattr(module, '_write_bound_trait_rows', original)
    _write(writer, [(0, [_single_row(3)]), (1, [])])
    assert writer.finish()['rows']['individual'] == 1
