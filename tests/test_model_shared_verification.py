"""Content verification survives process restarts without redundant scans."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from fudan_wgs_toolkit import phewas_storage as storage
from fudan_wgs_toolkit import model_verification_cache as cache


def _source(tmp_path):
    root = tmp_path / 'model'
    state = root / 'normal'
    state.mkdir(parents=True)
    (root / 'model.private.json').write_text('{}')
    path = state / 'residual.npy'
    path.write_bytes(b'authenticated fitted arrays')
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _forget(path):
    storage._VERIFIED_FILES.pop(str(path.resolve()), None)


def test_receipt_reuses_exact_proof_after_process_cache_is_lost(tmp_path, monkeypatch):
    path, expected = _source(tmp_path)
    original, calls = storage.digest_file, []
    monkeypatch.setattr(storage, 'digest_file', lambda p: (calls.append(p), original(p))[1])
    identity = storage.verify_file(path, expected)
    _forget(path)
    assert storage.verify_file(path, expected) == identity
    assert calls == [path]
    receipt = next((tmp_path / '.model_file_verification').glob('*.json'))
    assert receipt.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize('change', ['same_size_edit', 'inode_replacement', 'expected_hash'])
def test_receipt_does_not_authorize_changed_bytes_or_binding(tmp_path, change):
    path, expected = _source(tmp_path)
    storage.verify_file(path, expected)
    _forget(path)
    before = path.stat()
    if change == 'same_size_edit':
        path.write_bytes(b'X' * before.st_size)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif change == 'inode_replacement':
        replacement = path.with_name('replacement')
        replacement.write_bytes(b'X' * before.st_size)
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(path)
    else:
        expected = '0' * 64
    with pytest.raises(ValueError, match='checksum differs'):
        storage.verify_file(path, expected)
    assert str(path.resolve()) not in storage._VERIFIED_FILES


def test_corrupt_receipt_forces_a_full_content_check(tmp_path, monkeypatch):
    path, expected = _source(tmp_path)
    storage.verify_file(path, expected)
    _forget(path)
    receipt = next((tmp_path / '.model_file_verification').glob('*.json'))
    record = json.loads(receipt.read_text())
    record['payload']['files']['normal/residual.npy']['sha256'] = '0' * 64
    receipt.write_text(json.dumps(record))
    original, calls = storage.digest_file, []
    monkeypatch.setattr(storage, 'digest_file', lambda p: (calls.append(p), original(p))[1])
    storage.verify_file(path, expected)
    assert calls == [path]


def test_write_failure_does_not_weaken_or_prevent_content_validation(tmp_path, monkeypatch):
    path, expected = _source(tmp_path)
    def fail(*args):
        raise OSError('readonly filesystem')
    monkeypatch.setattr(cache, '_publish', fail)
    storage.verify_file(path, expected)
    _forget(path)
    path.write_bytes(b'bad source')
    with pytest.raises(ValueError, match='checksum differs'):
        storage.verify_file(path, expected)


def test_foreign_or_public_cache_falls_back_to_content_check(tmp_path, monkeypatch):
    path, expected = _source(tmp_path)
    folder = tmp_path / '.model_file_verification'
    folder.mkdir(mode=0o755)
    original, calls = storage.digest_file, []
    monkeypatch.setattr(storage, 'digest_file', lambda p: (calls.append(p), original(p))[1])
    storage.verify_file(path, expected)
    assert calls == [path]
    assert not list(folder.iterdir())


def test_parallel_workers_hash_once_and_share_a_receipt(tmp_path):
    if cache.fcntl is None:
        pytest.skip('requires POSIX file locks')
    path, expected = _source(tmp_path)
    code = '''import json,sys,time
from fudan_wgs_toolkit import phewas_storage as s
original=s.digest_file
calls=[]
def digest(path):
 calls.append(str(path));time.sleep(.15);return original(path)
s.digest_file=digest
s.verify_file(sys.argv[1],sys.argv[2])
print(json.dumps(dict(hashes=len(calls),hits=s._FILE_VERIFICATION_METRICS['shared_receipt_hits'])))
'''
    workers = [subprocess.Popen([sys.executable, '-c', code, str(path), expected],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    outputs = []
    for worker in workers:
        stdout, stderr = worker.communicate(timeout=30)
        assert worker.returncode == 0, stderr
        outputs.append(json.loads(stdout))
    assert sum(item['hashes'] for item in outputs) == 1
    assert sum(item['hits'] for item in outputs) == 1


def test_hash_must_scan_the_same_open_inode_that_was_identified(tmp_path, monkeypatch):
    path, expected = _source(tmp_path)
    other = tmp_path / 'other.npy'
    other.write_bytes(path.read_bytes())
    original_open = Path.open
    def swapped_open(self, *args, **kwargs):
        return original_open(other if self == path else self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', swapped_open)
    with pytest.raises(ValueError, match='identity changed during verification'):
        storage.verify_file(path, expected)
    assert str(path.resolve()) not in storage._VERIFIED_FILES
    assert not list((tmp_path / '.model_file_verification').glob('*.json'))
