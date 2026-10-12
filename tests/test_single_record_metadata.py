"""Bound frame-local labels: semantic/lifecycle tests, not a benchmark."""
import gc
import weakref

import numpy as np
import pytest

from fudan_wgs_toolkit._single_record_metadata import SingleRecordMetadata


class Reader:
    def __init__(self):
        self.calls = 0

    def read_field(self, name, indices):
        assert name == 'chromosome'
        self.calls += 1
        return np.asarray(['21'] * len(indices))

    def read_ref_alt(self, indices):
        self.calls += 1
        return np.asarray(['A'] * len(indices)), np.asarray(['T'] * len(indices))


def test_labels_reused_only_for_same_reader_and_ordered_columns():
    reader = Reader()
    indices = np.asarray([9, 4, 8], dtype=np.int64)
    labels = SingleRecordMetadata.prepare(reader, indices)
    values = labels.values(reader, indices)
    for _ in range(32):
        assert all(left is right for left, right in
                   zip(values, labels.values(reader, indices.copy())))
    assert reader.calls == 2 and not labels.indices.flags.writeable
    with pytest.raises(ValueError, match='reader or ordered columns'):
        labels.values(Reader(), indices)
    with pytest.raises(ValueError, match='reader or ordered columns'):
        labels.values(reader, indices[::-1])
    indices[0] = 5
    with pytest.raises(ValueError, match='reader or ordered columns'):
        labels.values(reader, indices)


def test_label_owner_releases_reader_with_prepared_block():
    reader = Reader()
    reference = weakref.ref(reader)
    prepared = {'_single_record_metadata': SingleRecordMetadata.prepare(reader, np.arange(3))}
    del reader
    assert reference() is not None
    del prepared
    gc.collect()
    assert reference() is None


def test_metadata_requires_complete_column_coverage():
    reader = Reader()
    reader.read_ref_alt = lambda unused: (['A'], ['T'])
    with pytest.raises(ValueError, match='all selected columns'):
        SingleRecordMetadata.prepare(reader, np.arange(3))
