"""Metadata-only fitted-bank estimates for deterministic resource scheduling.

These estimates describe tensor payloads loaded by ModelRepository. They do
not authenticate array contents, predict allocator fragmentation, or replace
live memory admission. NPY files are never read by this module.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


_SHA = re.compile(r'^[0-9a-f]{64}$')
_BINARY_NORMAL = {'x', 'scaled_residuals', 'fitted_probability', 'precision_x',
    'fixed_effect_covariance', 'precision', 'precision_indices', 'precision_values',
    'coefficients', 'phenotype', 'working_phenotype'}
_GAUSSIAN_NORMAL = {'x', 'scaled_residuals', 'coefficients', 'theta', 'precision_theta',
    'fixed_effect_covariance', 'eigenvalues', 'inverse_variance', 'precision_x'}
_SPA_SOURCE = {'x', 'scaled_residuals', 'fitted_probability', 'fixed_effect_covariance',
    'precision', 'precision_indices', 'precision_values', 'coefficients'}


def _digest(value, name):
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f'{name} must be a lowercase SHA-256 digest')
    return value


def _metadata(path, expected, verify):
    expected = _digest(expected, 'metadata checksum')
    payload = path.read_bytes()
    if verify and hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('fitted metadata checksum differs')
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError('fitted metadata must be a JSON object')
    return value


def _array_bytes(spec):
    if not isinstance(spec, Mapping):
        raise ValueError('fitted array specification must be an object')
    shape = spec.get('shape')
    if (not isinstance(shape, list) or any(type(v) is not int or v < 0 for v in shape)
            or not isinstance(spec.get('file'), str)
            or Path(spec['file']).name != spec['file']):
        raise ValueError('fitted array shape/path is invalid')
    dtype = np.dtype(spec['dtype'])
    if dtype.hasobject:
        raise ValueError('fitted arrays cannot contain objects')
    _digest(spec.get('sha256'), 'fitted array checksum')
    return math.prod(shape)*dtype.itemsize


def _payloads(state, fields, trait, namespace):
    arrays = state.get('arrays')
    if not isinstance(arrays, dict):
        raise ValueError('fitted state requires an array dictionary')
    payloads = []
    for name in sorted(fields.intersection(arrays)):
        spec = arrays[name]
        size = _array_bytes(spec)
        # The actual bank deduplicates X only. Identical residual/precision
        # hashes in unrelated states do not imply shared loaded storage.
        identity = (['x', spec['sha256'], spec['shape'], spec['dtype']] if name == 'x'
                    else [namespace, trait, name])
        payloads.append(dict(identity=identity, bytes=size, kind=namespace))
    return payloads


def estimate_fitted_subset_storage(summary, trait_indices):
    """Recompute an assigned subset, deduplicating only actual shared X keys."""
    indices = list(trait_indices)
    if (any(type(i) is not int or i < 0 for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError('trait_indices must contain distinct global nonnegative integers')
    records = {record['trait_index']: record for record in summary['per_trait']}
    if any(i not in records for i in indices):
        raise ValueError('assigned trait index is absent from the footprint inventory')
    selected = [records[i] for i in indices]
    seen, normal, spa, shared = {}, 0, 0, 0
    for record in selected:
        for payload in record['storage_payloads']:
            key = json.dumps(payload['identity'], separators=(',', ':'))
            previous = seen.get(key)
            if previous is not None:
                if previous != payload['bytes']:
                    raise ValueError('shared fitted design geometry disagrees')
                shared += payload['bytes']
                continue
            seen[key] = payload['bytes']
            if payload['kind'] == 'normal':
                normal += payload['bytes']
            else:
                spa += payload['bytes']
    lazy = sum(record['lazy_single_projection_bytes_upper_bound'] for record in selected)
    return dict(trait_count=len(selected), normal_loaded_bytes=normal,
        spa_source_loaded_bytes=spa, persistent_loaded_bytes=normal+spa,
        shared_x_bytes_saved=shared, lazy_single_projection_bytes_upper_bound=lazy,
        persistent_with_lazy_upper_bound_bytes=normal+spa+lazy,
        active_spa_projection_bytes_upper_bound=max((record['active_spa_projection_bytes_upper_bound']
            for record in selected), default=0),
        normal_derived_active_bytes_upper_bound=max((record['normal_derived_active_bytes_upper_bound']
            for record in selected), default=0))


def estimate_fitted_model_storage(entries, *, binary_normal_cache='stored_native',
                                  include_spa=True, include_lazy=False,
                                  verify_metadata=True):
    """Estimate persistent normal/minimal-SPA payloads without loading arrays.

    ``entries`` retains global source-column order. Ordered sample identity is
    the authenticated NPY digest in each root model manifest, never sample N.
    ``include_spa`` estimates the conservative case where every available SPA
    source is retained; actual sparse significance selection may load fewer.
    Lazy Single P1 and active FP64 SX/Xcov are reported separately. Gaussian
    kinship blocks remain in the normal footprint; N-by-N sparse precision is
    counted as its stored indices/values, never as a dense matrix.
    """
    if binary_normal_cache not in ('stored_native', 'derived_fp64'):
        raise ValueError('binary_normal_cache must be stored_native or derived_fp64')
    if any(type(v) is not bool for v in (include_spa, include_lazy, verify_metadata)):
        raise ValueError('footprint flags must be boolean')
    records, seen_indices = [], set()
    for ordinal, entry in enumerate(entries):
        index = entry.get('trait_index', ordinal)
        if type(index) is not int or index < 0 or index in seen_indices:
            raise ValueError('fitted inventory requires distinct global trait indices')
        seen_indices.add(index)
        root = Path(entry['path'])
        model = _metadata(root/'model.private.json', entry['sha256'], verify_metadata)
        family, n = model.get('family'), model.get('n')
        if family not in ('gaussian', 'binomial') or family != entry.get('family') or type(n) is not int or n < 1:
            raise ValueError('fitted family/sample geometry differs from inventory')
        axis_digest = _digest(model.get('sample_rows_sha256'), 'ordered sample-axis checksum')
        normal = _metadata(root/'normal'/'state.private.json', model['state_sha256'], verify_metadata)
        if normal.get('family') != family:
            raise ValueError('normal fitted family differs')
        normal_arrays = normal['arrays']
        shape = normal_arrays.get('x', {}).get('shape')
        if not isinstance(shape, list) or len(shape) != 2 or shape[0] != n or type(shape[1]) is not int or shape[1] < 1:
            raise ValueError('fitted design/sample geometry differs')
        q = shape[1]
        fields = set(_GAUSSIAN_NORMAL if family == 'gaussian' else _BINARY_NORMAL)
        if family == 'gaussian':
            for block in normal.get('blocks', []):
                if type(block) is not int or block < 0:
                    raise ValueError('fitted kinship block index is invalid')
                fields.update((f'block_{block}_rows', f'block_{block}_rotation'))
        direct_normal = _payloads(normal, fields, index, 'normal')
        payloads = direct_normal if family == 'gaussian' or binary_normal_cache == 'stored_native' else []
        spa_payloads, active_spa = [], 0
        source_required = family == 'binomial' and (include_spa or binary_normal_cache == 'derived_fp64')
        if source_required and model.get('spa_state_sha256') is not None:
            source = _metadata(root/'spa'/'state.private.json', model['spa_state_sha256'], verify_metadata)
            if source.get('family') != 'binomial' or source['arrays']['x']['shape'] != shape:
                raise ValueError('SPA fitted family/design geometry differs')
            spa_payloads = _payloads(source, _SPA_SOURCE, index, 'spa')
            active_spa = 16*n*q  # Active FP64 SX and Xcov, never persistent.
            payloads = payloads+spa_payloads
        elif source_required and binary_normal_cache == 'derived_fp64':
            raise ValueError('derived binary normal state requires its original SPA source')
        # P1 uses N FP32 elements plus its scalar sum, not another N-by-Q X.
        # Geometry can disqualify a cache; an intercept/data check remains live.
        precision_shape = normal_arrays.get('inverse_variance' if family == 'gaussian' else 'precision', {}).get('shape')
        lazy = 4*(n+1) if not normal.get('blocks') and precision_shape == [n] else 0
        record = dict(trait_index=index, family=family, n=n, q=q,
            ordered_sample_rows_sha256=axis_digest, model_manifest_sha256=entry['sha256'],
            normal_loaded_bytes=sum(p['bytes'] for p in payloads if p['kind']=='normal'),
            spa_source_loaded_bytes=sum(p['bytes'] for p in spa_payloads),
            lazy_single_projection_bytes_upper_bound=lazy,
            active_spa_projection_bytes_upper_bound=active_spa,
            normal_derived_active_bytes_upper_bound=(sum(p['bytes'] for p in direct_normal)
                if family=='binomial' and binary_normal_cache=='derived_fp64' else 0),
            storage_payloads=payloads)
        records.append(record)
    summary = dict(schema_version=1, per_trait=records, binary_normal_cache=binary_normal_cache,
        include_spa=include_spa, include_lazy=include_lazy, metadata_checksums_verified=verify_metadata,
        array_contents_read=False, array_checksums_verified=False,
        scope='metadata-derived loaded tensor payloads; not allocator/RSS/peak or scientific validation')
    summary.update(estimate_fitted_subset_storage(summary,
        [record['trait_index'] for record in records]))
    summary['estimated_bank_budget_bytes'] = (summary['persistent_with_lazy_upper_bound_bytes']
        if include_lazy else summary['persistent_loaded_bytes'])
    return summary
