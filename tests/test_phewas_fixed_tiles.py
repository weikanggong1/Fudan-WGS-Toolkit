"""Fixed global tile boundaries survive worker sharding and checkpoint holes."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fudan_wgs_toolkit.phewas_runtime import runtime


def fixture():
    rows = np.arange(7)
    descriptors = [dict(trait_index=i, n=7, family='gaussian', ordered_rows=rows,
                        sample_rows_sha256='a'*64) for i in (0, 1, 4, 5)]
    _, groups = runtime._descriptor_groups(SimpleNamespace(descriptors=lambda: descriptors))
    records = []
    for number, ids in ((0, [0, 1]), (2, [4, 5])):
        record = dict(cohort_index=0, tile_index=number, ordered_sample_rows_sha256='a'*64,
                      n=7, trait_indices=ids, trait_batch_size=2)
        record['tile_binding_sha256'] = hashlib.sha256(json.dumps(record, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        records.append(record)
    return groups, records


def test_resume_hole_retains_original_tile_and_does_not_rechunk():
    groups, records = fixture()
    runtime._bind_trait_tiles(groups, records, 2)
    assert [[d['trait_index'] for d in tile] for tile in runtime._trait_tiles(groups[0], 2)] == [[0, 1], [4, 5]]
    pending = dict(groups[0], descriptors=groups[0]['descriptors'][2:])
    assert [[d['trait_index'] for d in tile] for tile in runtime._trait_tiles(pending, 2)] == [[4, 5]]
    partial = dict(groups[0], descriptors=groups[0]['descriptors'][1:])
    with pytest.raises(ValueError, match='partially resumed'):
        list(runtime._trait_tiles(partial, 2))


@pytest.mark.parametrize('change', ['hash', 'axis', 'coverage', 'duplicate', 'size'])
def test_binding_rejects_changed_tile_or_model_axis(change):
    groups, records = fixture()
    if change == 'hash':
        records[0]['tile_binding_sha256'] = 'b'*64
    elif change == 'axis':
        groups[0]['descriptors'][0]['sample_rows_sha256'] = 'c'*64
    elif change == 'coverage':
        records.pop()
    elif change == 'duplicate':
        records.append(records[0])
    else:
        groups[0]['descriptors'][0]['n'] = 8
    with pytest.raises(ValueError):
        runtime._bind_trait_tiles(groups, records, 2)
