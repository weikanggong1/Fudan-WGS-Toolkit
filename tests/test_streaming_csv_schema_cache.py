"""Schema memo changes costs, not shard order, bytes or ownership."""
import csv
import gc
import gzip
import hashlib
import json
import weakref

import pytest

from fudan_wgs_toolkit.phewas_storage import StreamingCSVWriter


class UnmemoizedWriter(StreamingCSVWriter):
    def _schema(self, row):
        columns = list(row)
        signature = hashlib.sha256(json.dumps(columns).encode()).hexdigest()[:12]
        return columns, signature


def test_repeated_rows_serialize_the_schema_once(tmp_path, monkeypatch):
    import fudan_wgs_toolkit.phewas_storage as storage
    original = storage.json.dumps
    calls = []

    def observed(value, *args, **kwargs):
        if isinstance(value, list) and value[:1] == ['trait_index']:
            calls.append(tuple(value))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(storage.json, 'dumps', observed)
    writer = StreamingCSVWriter(tmp_path, minimum_free_bytes=0)
    writer.write('individual', 3, '1', [dict(CHR=1, POS=i, pvalue=.5) for i in range(20)])
    writer.write('individual', 4, '1', [dict(CHR=1, POS=20, pvalue=.2)])
    report = writer.finish()
    assert calls == [('trait_index', 'CHR', 'POS', 'pvalue')]
    assert report['rows'] == {'individual': 21}
    assert report['receipts'][0]['columns'] == ['trait_index', 'CHR', 'POS', 'pvalue']


def test_changed_fields_and_column_order_keep_independent_schemas(tmp_path):
    writer = StreamingCSVWriter(tmp_path, minimum_free_bytes=0)
    records = [dict(CHR=1, POS=2, pvalue=.5), dict(POS=3, CHR=1, pvalue=.2),
               dict(CHR=1, POS=4, pvalue=.1, logp=-1.)]
    writer.write('individual', 2, '1', records)
    report = writer.finish()
    assert len(report['receipts']) == 3
    assert [r['columns'] for r in report['receipts']] == [
        ['trait_index', 'CHR', 'POS', 'pvalue'],
        ['trait_index', 'POS', 'CHR', 'pvalue'],
        ['trait_index', 'CHR', 'POS', 'pvalue', 'logp']]
    for receipt, source in zip(report['receipts'], records):
        expected = hashlib.sha256(json.dumps(receipt['columns']).encode()).hexdigest()[:12]
        assert expected in receipt['csv_file']
        with gzip.open(tmp_path/receipt['csv_file'], 'rt', newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert rows == [{key: str(value) for key, value in dict(trait_index=2, **source).items()}]


def test_schema_cache_is_bounded_lru_and_recomputes_evicted_schema(tmp_path):
    writer = StreamingCSVWriter(tmp_path, max_open_streams=2, minimum_free_bytes=0)
    writer.write('coding', 0, '1', [dict(CHR=1, old=1)])
    first = ('trait_index', 'CHR', 'old')
    first_signature = writer._schema_cache[first]
    for i in range(StreamingCSVWriter._SCHEMA_CACHE_LIMIT):
        writer.write('coding', 0, '1', [{'CHR': 1, f'column_{i}': i}])
    assert len(writer._schema_cache) == StreamingCSVWriter._SCHEMA_CACHE_LIMIT
    assert first not in writer._schema_cache
    writer.write('coding', 1, '1', [dict(CHR=1, old=2)])
    assert writer._schema_cache[first] == first_signature
    assert len(writer._schema_cache) == StreamingCSVWriter._SCHEMA_CACHE_LIMIT
    assert next(reversed(writer._schema_cache)) == first
    report = writer.finish()
    assert report['rows']['coding'] == StreamingCSVWriter._SCHEMA_CACHE_LIMIT+2


def test_schema_cache_does_not_retain_rows_or_values(tmp_path):
    class Value:
        def __str__(self):
            return 'not retained'

    class Row(dict):
        pass

    writer = StreamingCSVWriter(tmp_path, minimum_free_bytes=0)
    owner = Value()
    row = Row(CHR=1, POS=owner)
    value_ref, row_ref = weakref.ref(owner), weakref.ref(row)
    writer.write('individual', 0, '1', [row])
    del owner, row
    gc.collect()
    assert value_ref() is None and row_ref() is None
    assert writer._schema_cache == {('trait_index', 'CHR', 'POS'): writer._schema_cache[('trait_index', 'CHR', 'POS')]}
    writer.finish()


def test_trait_index_override_preserves_the_existing_error(tmp_path):
    writer = StreamingCSVWriter(tmp_path, minimum_free_bytes=0)
    with pytest.raises(TypeError, match='trait_index'):
        writer.write('individual', 0, '1', [dict(trait_index=99, CHR=1, POS=1)])
    assert not writer._schema_cache and not writer.streams
    writer.abort()


def test_unmemoized_and_memoized_shards_are_byte_identical(tmp_path, monkeypatch):
    monkeypatch.setattr(gzip.time, 'time', lambda: 1700000000.)
    roots = [tmp_path/'original', tmp_path/'memoized']
    reports = []
    for cls, root in zip((UnmemoizedWriter, StreamingCSVWriter), roots):
        writer = cls(root, max_rows_per_shard=3, max_open_streams=2, minimum_free_bytes=0)
        for trait in range(4):
            writer.write('individual', trait, '21', [dict(CHR=21, POS=trait, pvalue=.25),
                dict(POS=trait+1, CHR=21, pvalue=1e-10), dict(CHR=21, POS=trait+2, pvalue=.9)], job_index=trait)
        writer.write('coding', 2, '21', [dict(mask='example', pvalue=.1)], job_index=5)
        reports.append(writer.finish())
    a, b = reports
    assert a['rows'] == b['rows'] and a['files'] == b['files']
    assert a['receipts'] == b['receipts']
    for receipt in a['receipts']:
        relative = receipt['csv_file']
        assert (roots[0]/relative).read_bytes() == (roots[1]/relative).read_bytes()
        assert (roots[0]/(relative+'.receipt.json')).read_bytes() == (roots[1]/(relative+'.receipt.json')).read_bytes()
