"""Fault-injection contracts for real block resume; not a performance benchmark."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fudan_wgs_toolkit.phewas_checkpoints import CheckpointCSVWriter
from fudan_wgs_toolkit.phewas_runtime import runtime
from fudan_wgs_toolkit.phewas_runtime.shared_state import SharedStateBroker
from fudan_wgs_toolkit.phewas_runtime.metadata import SharedMetadataReader
from fudan_wgs_toolkit.pipeline import AnalysisOptions
from test_phewas_batched_runtime import Repository, descriptor, model_for, cpu_product_oracle
from test_phewas_shared_state import MetadataReader, make_cache, fixture_states


BINDING = dict(source_implementation_sha256='source', input_manifest_sha256='inputs',
               configuration_sha256='schedule-and-fixed-tile-size', models={'0': 'model-0', '1': 'model-1'})


def row(value=.25):
    return {'Chr': '21', 'Gene name': 'fixture', 'WGS-O': value}


def all_rows(directory, report):
    rows = []
    for receipt in report['receipts']:
        with gzip.open(Path(directory)/receipt['csv_file'], 'rt', newline='') as handle:
            rows.extend(csv.DictReader(handle))
    return sorted(rows, key=lambda value: json.dumps(value, sort_keys=True))


def test_checkpoint_requires_complete_scientific_binding(tmp_path):
    with pytest.raises(ValueError, match='source, input, schedule and model hashes'):
        CheckpointCSVWriter(tmp_path, binding={'source_implementation_sha256': 'only-source'}, resume=True)


def test_same_invocation_retry_reuses_committed_tile_without_enabling_old_run_resume(tmp_path):
    writer = CheckpointCSVWriter(tmp_path, binding=BINDING, resume=False)
    first = writer.write_tile('coding', '21', [(0, [row()])], job_index=3)
    assert writer.checkpoint_tile('coding', '21', 3, [0]) == first
    assert writer.write_tile('coding', '21', [(0, [row(.9)])], job_index=3) == first
    report = writer.finish()
    assert report['computed_blocks'] == 1 and report['rows'] == {'coding': 1}
    assert all_rows(tmp_path, report)[0]['WGS-O'] == '0.25'
    with pytest.raises(FileExistsError, match='resume=True'):
        CheckpointCSVWriter(tmp_path, binding=BINDING, resume=False)


def test_interrupted_gene_tiles_resume_on_new_worker_without_duplicates_or_new_source(tmp_path):
    writer = CheckpointCSVWriter(tmp_path, binding=BINDING)
    first = writer.write_tile('coding', '21', [(0, [row()]), (1, [row(.5)])], job_index=7,
                              binary_counts={'selected': 2, 'failed': 0})
    writer.abort()
    recovered = CheckpointCSVWriter(tmp_path, binding=BINDING, resume=True, worker_id='7')
    complete = recovered.checkpoint_tile('coding', '21', 7, [0, 1])
    assert complete == first and complete['binary_spa_selected'] == 2
    recovered.write_tile('coding', '21', [(0, [row(.9)]), (1, [row(.9)])], job_index=7)
    recovered.write_tile('ncrna', '21', [(0, []), (1, None)], job_index=8)
    report = recovered.finish()
    assert report['rows'] == {'coding': 2, 'ncrna': 0}
    assert report['reused_blocks'] == report['computed_blocks'] == 1
    assert len(all_rows(tmp_path, report)) == 2
    assert {item['WGS-O'] for item in all_rows(tmp_path, report)} == {'0.25', '0.5'}
    with pytest.raises(ValueError, match='binding differs'):
        CheckpointCSVWriter(tmp_path, binding=dict(BINDING, models={'0': 'other'}), resume=True)
    with pytest.raises(FileExistsError, match='resume=True'):
        CheckpointCSVWriter(tmp_path, binding=BINDING)


@pytest.mark.parametrize('damage', ['missing_marker', 'bad_csv', 'bad_rows', 'truncated_marker'])
def test_missing_or_corrupt_block_is_recomputed_not_skipped(tmp_path, damage):
    writer = CheckpointCSVWriter(tmp_path, binding=BINDING)
    receipt = writer.write_tile('noncoding', '21', [(0, [row()])], job_index=9)
    marker = next((tmp_path/'.checkpoints'/'blocks').rglob('*.complete.private.json'))
    output = tmp_path/receipt['csv_file']
    if damage == 'missing_marker':
        marker.unlink()
    elif damage == 'bad_csv':
        output.write_bytes(b'corrupt')
    elif damage == 'truncated_marker':
        marker.write_text('{')
    else:
        value = json.loads(marker.read_text())
        value['rows'] = 100
        value.pop('receipt_sha256')
        value['receipt_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        marker.write_text(json.dumps(value))
    writer = CheckpointCSVWriter(tmp_path, binding=BINDING, resume=True)
    assert writer.checkpoint_tile('noncoding', '21', 9, [0]) is None
    writer.write_tile('noncoding', '21', [(0, [row(.5)])], job_index=9)
    report = writer.finish()
    assert report['rows'] == {'noncoding': 1} and report['computed_blocks'] == 1
    assert report['invalid_blocks_recomputed'] == 1
    assert all_rows(tmp_path, report)[0]['WGS-O'] == '0.5'
    assert list((tmp_path/'.checkpoints'/'invalid').iterdir())


def test_final_readback_checks_even_previously_verified_blocks(tmp_path):
    writer = CheckpointCSVWriter(tmp_path, binding=BINDING)
    receipt = writer.write_tile('coding', '21', [(0, [row()])], job_index=0)
    (tmp_path/receipt['csv_file']).write_bytes(b'bad after accepted write')
    with pytest.raises(RuntimeError, match='changed before final CSV audit'):
        writer.finish()


def rewrite_receipt(marker, **changes):
    record = json.loads(marker.read_text())
    record.update(changes)
    record.pop('receipt_sha256')
    record['receipt_sha256'] = hashlib.sha256(json.dumps(record, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    marker.write_text(json.dumps(record))


@pytest.mark.parametrize('changes', [
    {'csv_file': '../unbound.csv.gz'}, {'csv_file': '/unbound.csv.gz'},
    {'kind': 'ncrna'}, {'chromosome': '22'}, {'first_job': 8}, {'last_job': 10},
    {'rows': True}, {'trait_rows': {'0': True}}, {'trait_rows': {'0': 1, '1': 0}},
    {'compressed_bytes': 0}, {'binary_spa_selected': -1}, {'binary_spa_failed': 1},
    {'binary_spa_selected': True}, {'columns': ['trait_index', 'Chr', 'Chr']},
    {'ordinal_after': 0},
])
def test_resigned_receipt_cannot_redirect_output_or_change_scientific_counts(tmp_path, changes):
    writer = CheckpointCSVWriter(tmp_path, binding=BINDING)
    writer.write_tile('coding', '21', [(0, [row()])], job_index=9)
    marker = next((tmp_path/'.checkpoints'/'blocks').rglob('*.complete.private.json'))
    rewrite_receipt(marker, **changes)
    recovered = CheckpointCSVWriter(tmp_path, binding=BINDING, resume=True)
    assert recovered.checkpoint_tile('coding', '21', 9, [0]) is None
    recovered.write_tile('coding', '21', [(0, [row(.5)])], job_index=9)
    final = recovered.finish()
    assert final['invalid_blocks_recomputed'] == final['computed_blocks'] == 1
    assert final['rows'] == {'coding': 1}
    assert all_rows(tmp_path, final)[0]['WGS-O'] == '0.5'


@pytest.mark.parametrize('damage', ['deflate', 'crc', 'truncated', 'csv_limit', 'json_list'])
def test_corrupt_payload_with_matching_digest_is_recomputed(tmp_path, damage):
    writer = CheckpointCSVWriter(tmp_path, binding=BINDING)
    receipt = writer.write_tile('coding', '21', [(0, [row()])], job_index=9)
    marker = next((tmp_path/'.checkpoints'/'blocks').rglob('*.complete.private.json'))
    path = tmp_path/receipt['csv_file']
    limit = csv.field_size_limit()
    try:
        if damage == 'json_list':
            marker.write_text('[]')
        elif damage == 'csv_limit':
            csv.field_size_limit(8)
        else:
            payload = path.read_bytes()
            if damage == 'deflate':
                payload = b'\x1f\x8b\x08\x00'+b'\x00'*6+b'\x07'+b'\x00'*8
            elif damage == 'crc':
                payload = payload[:-8]+bytes([payload[-8] ^ 1])+payload[-7:]
            else:
                payload = payload[:-3]
            path.write_bytes(payload)
            rewrite_receipt(marker, sha256=hashlib.sha256(payload).hexdigest(),
                            compressed_bytes=len(payload))
        recovered = CheckpointCSVWriter(tmp_path, binding=BINDING, resume=True)
        assert recovered.checkpoint_tile('coding', '21', 9, [0]) is None
    finally:
        csv.field_size_limit(limit)
    recovered.write_tile('coding', '21', [(0, [row(.5)])], job_index=9)
    assert recovered.finish()['rows'] == {'coding': 1}


@pytest.mark.parametrize('remove', ['csv', 'marker'])
def test_final_readback_refuses_deleted_completed_artifacts(tmp_path, remove):
    writer = CheckpointCSVWriter(tmp_path, binding=BINDING)
    receipt = writer.write_tile('coding', '21', [(0, [row()])], job_index=0)
    path = (tmp_path/receipt['csv_file'] if remove == 'csv' else
            next((tmp_path/'.checkpoints'/'blocks').rglob('*.complete.private.json')))
    path.unlink()
    with pytest.raises(RuntimeError, match='changed before final CSV audit'):
        writer.finish()
    assert not (tmp_path/'worker_0.complete.private.json').exists()


@pytest.mark.parametrize('trait', [True, .5, '1'])
def test_trait_identity_is_not_silently_coerced(tmp_path, trait):
    writer = CheckpointCSVWriter(tmp_path, binding=BINDING)
    with pytest.raises(ValueError, match='trait indices'):
        writer.write_tile('coding', '21', [(trait, [row()])], job_index=0)
    assert not list((tmp_path/'.checkpoints'/'blocks').rglob('*.csv.gz'))


def test_new_gpu_partition_reuses_exact_tiles_once_and_keeps_category_and_float_order(tmp_path):
    previous = CheckpointCSVWriter(tmp_path, binding=BINDING, worker_id='3')
    tiny = [dict(CHR='21', POS=1, pvalue=0., pvalue_log10=1000.125)]
    categories = {'first': [row(.1), row(.2)], 'second': [row(.3)]}
    previous.write_tile('coding', '21', [(0, categories), (1, [])], job_index=4)
    previous.write_tile('individual', '21', [(0, tiny), (1, [])], job_index=5,
        block_identity='axis-digest', ordinal_before=0, ordinal_after=1)
    previous.finish()
    current = CheckpointCSVWriter(tmp_path, binding=BINDING, resume=True, worker_id='0')
    for _ in range(2):
        assert current.checkpoint_tile('coding', '21', 4, [0, 1]) is not None
        assert current.checkpoint_tile('individual', '21', 5, [0, 1],
            block_identity='axis-digest', ordinal_before=0) is not None
    report = current.finish()
    assert report['rows'] == {'coding': 3, 'individual': 1}
    assert report['files'] == report['reused_blocks'] == 2
    assert report['computed_blocks'] == 0
    assert [r['first_job'] for r in report['receipts']] == [4, 5]
    with gzip.open(tmp_path/report['receipts'][0]['csv_file'], 'rt', newline='') as stream:
        assert [r['WGS-O'] for r in csv.DictReader(stream)] == [format(v, '.17g') for v in (.1, .2, .3)]
    assert next(r for r in all_rows(tmp_path, report) if 'pvalue_log10' in r)['pvalue_log10'] == '1000.125'


def test_completion_receipts_follow_numeric_autosome_and_job_order(tmp_path):
    writer = CheckpointCSVWriter(tmp_path, binding=BINDING)
    for chromosome, job in [('10', 1), ('2', 2), ('1', 3), ('2', 0)]:
        writer.write_tile('coding', chromosome, [(0, [row()])], job_index=job)
    report = writer.finish()
    assert [(r['chromosome'], r['first_job']) for r in report['receipts']] == [
        ('1', 3), ('2', 0), ('2', 2), ('10', 1)]


def test_empty_single_and_ordinal_and_variant_axis_are_bound(tmp_path):
    writer = CheckpointCSVWriter(tmp_path, binding=BINDING)
    writer.write_tile('individual', '21', [(0, []), (1, [])], job_index=0,
        block_index=2, block_identity='variant-axis', ordinal_before=4, ordinal_after=4)
    writer = CheckpointCSVWriter(tmp_path, binding=BINDING, resume=True)
    assert writer.checkpoint_tile('individual', '21', 0, [0, 1], block_index=2,
        block_identity='other-axis', ordinal_before=4) is None
    assert writer.checkpoint_tile('individual', '21', 0, [0, 1], block_index=2,
        block_identity='variant-axis', ordinal_before=3) is None
    assert writer.checkpoint_tile('individual', '21', 0, [0, 1], block_index=2,
        block_identity='variant-axis', ordinal_before=4)['rows'] == 0


def test_single_frame_and_partial_tile_resume_skips_finished_io_and_models(tmp_path, cpu_product_oracle):
    n, m = 24, 10
    raw = fixture_states(m=m, n=n)
    class Reader(MetadataReader):
        def read_field(self, name, indices=None):
            value = np.full(m, 'PASS') if name == 'annotation/filter' else np.arange(m)
            if name == 'chromosome':
                value = np.full(m, '21')
            return value if indices is None else value[indices]
        def read_ref_alt(self, indices=None):
            selected = np.arange(m) if indices is None else indices
            return np.full(len(selected), 'A'), np.full(len(selected), 'C')
    ds = [descriptor(i, np.arange(n)) for i in range(3)]
    models = [model_for(np.arange(n), .7+i) for i in range(3)]
    _, groups = runtime._descriptor_groups(Repository(ds, models))
    config = dict(individual_genotype_block_size=4)
    options = AnalysisOptions(wrapper_semantics='base', memory_limit_gib=80)
    cache = make_cache(tmp_path/'cache', raw)
    output = tmp_path/'resumed'
    class InterruptedWriter(CheckpointCSVWriter):
        calls = 0
        def write_tile(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 4:
                raise OSError('injected interruption after one complete frame and one partial trait tile')
            return super().write_tile(*args, **kwargs)
    def execute(writer):
        repository = Repository(ds, models)
        report = dict(association_rows={'individual': 0},
            trait_rows={i: {'individual': 0} for i in range(3)}, binary_spa_selected=0, binary_spa_failed=0)
        with SharedStateBroker(SharedMetadataReader(Reader(n, m)), cache, device='cpu', own_container=False) as broker:
            axis = broker.reader_view(np.arange(n))
            structural = runtime._thin_pipeline(axis, models[0], axis._axis.samples, options, config)
            structural._base_mask = lambda *args: np.ones(m, dtype=bool)
            try:
                runtime._batched_single(groups, repository, broker, broker._reader, options, config,
                    {'mac_cutoff': 1, 'variant_type': 'variant'}, writer, report, set(), '21', 19, 2, structural)
            finally:
                assert repository.active == 0
            return report, dict(broker.metrics), repository.acquired
    with pytest.raises(OSError, match='injected interruption'):
        execute(InterruptedWriter(output, binding=BINDING))
    recovered = CheckpointCSVWriter(output, binding=BINDING, resume=True, worker_id='4')
    report, metrics, acquired = execute(recovered)
    final = recovered.finish()
    reference = CheckpointCSVWriter(tmp_path/'reference', binding=BINDING)
    expected, _, _ = execute(reference)
    assert report['association_rows'] == expected['association_rows']
    assert report['trait_rows'] == expected['trait_rows']
    assert metrics['state_read_calls'] == 2
    assert report['resumed_single_physical_requests'] == 1
    assert acquired == [[2], [0, 1], [2]]
    assert final['reused_blocks'] == final['computed_blocks'] == 3
    assert all_rows(output, final) == all_rows(tmp_path/'reference', reference.finish())
    cache.close()


def test_gene_recovery_keeps_empty_and_computed_tiles_and_filters_by_fixed_job(tmp_path):
    groups = [{'rows': np.arange(4), 'descriptors': [descriptor(i, np.arange(4)) for i in range(5)]}]
    writer = CheckpointCSVWriter(tmp_path, binding=BINDING)
    writer.write_tile('coding', '21', [(0, []), (1, [row()])], job_index=11)
    writer.write_tile('coding', '21', [(4, [])], job_index=11)
    writer = CheckpointCSVWriter(tmp_path, binding=BINDING, resume=True)
    report = dict(association_rows={'coding': 0}, trait_rows={i: {'coding': 0} for i in range(5)},
                  binary_spa_selected=0, binary_spa_failed=0)
    pending = runtime._pending_gene_groups(groups, writer, report, '21', 11, 'coding', 2)
    assert [[d['trait_index'] for d in group['descriptors']] for group in pending] == [[2, 3]]
    assert report['association_rows'] == {'coding': 1} and report['trait_rows'][1]['coding'] == 1
    assert runtime._pending_gene_groups(groups, writer, report, '21', 12, 'coding', 2) == groups
    staged = runtime._StagedGeneWriter(writer)
    staged.write_tile('coding', '21', [(2, [row(.2)]), (3, [row(.1)])], job_index=11)
    assert writer.computed_blocks == 0
    staged.publish(writer)
    assert writer.finish()['rows'] == {'coding': 3}
