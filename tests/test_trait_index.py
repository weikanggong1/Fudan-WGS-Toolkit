"""Derived trait-index publication and recovery, without a CUDA benchmark."""
import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from fudan_wgs_toolkit import phewas_storage as storage
from fudan_wgs_toolkit.phewas_run import prepare_phewas_run
from test_phewas_public_options import api_inputs


NAMES = ['trait_b', 'trait_a']
FAMILIES = {'trait_a': 'binomial', 'trait_b': 'gaussian'}
EXPECTED = (b'trait_index,phenotype,family\n'
            b'0,trait_b,gaussian\n1,trait_a,binomial\n')


def table(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def test_dictionary_is_deterministic_and_keeps_source_column_order(tmp_path):
    expected_sha = hashlib.sha256(EXPECTED).hexdigest()
    assert storage.trait_index_sha256(NAMES, FAMILIES) == expected_sha
    output = tmp_path/'output'
    receipt = storage.write_trait_index(output, NAMES, FAMILIES, expected_sha256=expected_sha)
    path = output/'trait_index.private.csv'
    assert path.read_bytes() == EXPECTED
    assert receipt == dict(file=path.name, sha256=expected_sha, bytes=len(EXPECTED), repaired=False)
    assert path.stat().st_mode & 0o777 == 0o600
    before = path.stat()
    assert storage.write_trait_index(output, NAMES, FAMILIES) == receipt
    after = path.stat()
    assert (before.st_ino, before.st_mtime_ns, before.st_ctime_ns) == (
        after.st_ino, after.st_mtime_ns, after.st_ctime_ns)


@pytest.mark.parametrize('damage', ['missing', 'truncated', 'wrong_order', 'wrong_family'])
def test_dictionary_repairs_only_derived_output_and_preserves_incorrect_version(tmp_path, damage):
    storage.write_trait_index(tmp_path, NAMES, FAMILIES)
    path = tmp_path/'trait_index.private.csv'
    if damage == 'missing':
        path.unlink()
        previous = None
    else:
        previous = {'truncated': EXPECTED[:27],
                    'wrong_order': EXPECTED.replace(b'0,trait_b', b'0,trait_a'),
                    'wrong_family': EXPECTED.replace(b'gaussian', b'binomial')}[damage]
        path.write_bytes(previous)
    unrelated = tmp_path/'untouched-model.bin'
    unrelated.write_bytes(b'private anonymous fitted state')
    receipt = storage.write_trait_index(tmp_path, NAMES, FAMILIES)
    assert receipt['sha256'] == hashlib.sha256(EXPECTED).hexdigest()
    assert path.read_bytes() == EXPECTED
    assert unrelated.read_bytes() == b'private anonymous fitted state'
    retained = list((tmp_path/'trait_index_invalid.private').rglob('trait_index.private.csv'))
    assert len(retained) == (0 if previous is None else 1)
    if previous is not None:
        assert retained[0].read_bytes() == previous


def test_dictionary_interrupted_atomic_publish_retries_same_pid(tmp_path, monkeypatch):
    original_replace, pid = storage.os.replace, os.getpid()
    with monkeypatch.context() as patch:
        def interrupted_publish(source, target):
            if Path(target).name == 'trait_index.private.csv':
                raise OSError('injected dictionary publication interruption')
            return original_replace(source, target)
        patch.setattr(storage.os, 'replace', interrupted_publish)
        with pytest.raises(OSError, match='dictionary publication interruption'):
            storage.write_trait_index(tmp_path, NAMES, FAMILIES)
    assert not (tmp_path/'trait_index.private.csv').exists()
    partials = list(tmp_path.glob(f'trait_index.private.csv.{pid}.*.partial'))
    assert len(partials) == 1 and partials[0].read_bytes() == EXPECTED
    storage.write_trait_index(tmp_path, NAMES, FAMILIES)
    assert os.getpid() == pid
    assert partials[0].read_bytes() == EXPECTED
    assert (tmp_path/'trait_index.private.csv').read_bytes() == EXPECTED


def test_dictionary_quarantines_link_without_overwriting_its_target(tmp_path):
    target = tmp_path/'unrelated.csv'
    target.write_bytes(b'unrelated private file')
    output = tmp_path/'output'
    output.mkdir()
    path = output/'trait_index.private.csv'
    path.symlink_to(target)
    storage.write_trait_index(output, NAMES, FAMILIES)
    assert not path.is_symlink() and path.read_bytes() == EXPECTED
    assert target.read_bytes() == b'unrelated private file'
    retained = list((output/'trait_index_invalid.private').rglob('trait_index.private.csv'))
    assert len(retained) == 1 and retained[0].is_symlink()


def test_expected_dictionary_hash_mismatch_cannot_modify_existing_output(tmp_path):
    storage.write_trait_index(tmp_path, NAMES, FAMILIES)
    path = tmp_path/'trait_index.private.csv'
    with pytest.raises(ValueError, match='scientific plan'):
        storage.write_trait_index(tmp_path, NAMES, FAMILIES, expected_sha256='0'*64)
    assert path.read_bytes() == EXPECTED
    assert not (tmp_path/'trait_index_invalid.private').exists()


def test_legacy_prepare_publishes_and_repairs_the_same_bound_dictionary(api_inputs):
    first = prepare_phewas_run(**api_inputs, analyses=('coding',))
    path = Path(first['output_directory'])/'trait_index.private.csv'
    rows = table(path)
    assert [r['trait_index'] for r in rows] == ['0', '1', '2']
    assert [r['phenotype'] for r in rows] == ['trait_a', 'trait_b', 'trait_c']
    assert [r['family'] for r in rows] == ['binomial', 'gaussian', 'binomial']
    assert storage.digest_file(path) == first['trait_index_sha256']
    original = path.read_bytes()
    path.write_bytes(original[:20])
    resumed = prepare_phewas_run(**api_inputs, analyses=('coding',), resume=True)
    assert path.read_bytes() == original
    assert resumed['trait_index_sha256'] == first['trait_index_sha256']
    retained = list((path.parent/'trait_index_invalid.private').rglob(path.name))
    assert len(retained) == 1 and retained[0].read_bytes() == original[:20]


@pytest.mark.parametrize('damage', ['missing', 'truncated'])
def test_modern_resume_repairs_dictionary_without_refitting_or_recomputing(
        tmp_path, monkeypatch, damage):
    from test_prepared_association import fixture, injected_execution, call
    root, phenotype, covariate, _ = fixture(tmp_path)
    calls = injected_execution(monkeypatch)
    output = tmp_path/'results'
    first = call(root, phenotype, covariate, output)
    path = output/'trait_index.private.csv'
    original = path.read_bytes()
    plan = json.loads((output/'plan.private.json').read_text())
    assert plan['trait_index_sha256'] == storage.digest_file(path)
    if damage == 'missing':
        path.unlink()
    else:
        path.write_bytes(original[:20])
    resumed = call(root, phenotype, covariate, output, resume=True)
    assert path.read_bytes() == original
    assert resumed['association_rows'] == first['association_rows']
    assert len(calls['fits']) == 2 and calls['computed'] == [0, 1, 2]
    assert resumed['workers'][0]['output']['reused_blocks'] == 3
    assert json.loads((output/'plan.private.json').read_text())['trait_index_sha256'] == plan['trait_index_sha256']
