"""Private cross-process receipts for already hashed immutable model files.

Receipts never authorize different bytes or a new file identity. Missing,
invalid or unwritable receipts fall back to an ordinary full content check.
No model arrays are copied, rewritten or retained by this module.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

try:
    import fcntl
except ImportError:
    fcntl = None

FORMAT = 'immutable-model-verification-v1'
MAX_RECEIPT_BYTES = 1 << 20


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def _private(stat_result, *, directory=False):
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    return (kind(stat_result.st_mode) and stat_result.st_uid == os.getuid()
            and stat_result.st_mode & 0o077 == 0)


def _model_root(path):
    parent = path.parent
    if parent.name in ('normal', 'spa'):
        parent = parent.parent
    if not (parent / 'model.private.json').is_file():
        return None
    return parent.resolve()


def _read(receipt, model_root):
    try:
        with receipt.open('rb') as stream:
            info = os.fstat(stream.fileno())
            if not _private(info) or info.st_size > MAX_RECEIPT_BYTES:
                return {}
            record = json.loads(stream.read(MAX_RECEIPT_BYTES + 1))
        payload = record['payload']
        if (payload['format'] != FORMAT or payload['model_root'] != str(model_root)
                or hashlib.sha256(_encoded(payload)).hexdigest() != record['sha256']
                or not isinstance(payload['files'], dict)):
            return {}
        return payload['files']
    except (OSError, ValueError, TypeError, KeyError):
        return {}


def _publish(receipt, model_root, files):
    payload = dict(format=FORMAT, model_root=str(model_root), files=files)
    encoded = _encoded(dict(payload=payload, sha256=hashlib.sha256(_encoded(payload)).hexdigest()))
    if len(encoded) > MAX_RECEIPT_BYTES:
        return False
    fd, name = tempfile.mkstemp(prefix='.verification-', dir=receipt.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, receipt)
        return True
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def shared_verification(path, expected, identity):
    """Yield a cache decision while serializing checks of one model store.

    The caller must perform a full SHA check on a miss and verify file identity
    around that scan. Publication happens only after the caller succeeds.
    The caller also brackets hits with open/fstat and pathname identity checks.
    """
    decision = dict(hit=False, enabled=False, published=False, unavailable=False)
    root = _model_root(Path(path))
    if root is None or fcntl is None:
        yield decision
        return
    try:
        key = str(Path(path).resolve().relative_to(root))
    except ValueError:
        # An authenticated external array link still gets a full content check.
        yield decision
        return
    cache = root.parent / '.model_file_verification'
    lock_fd = None
    try:
        cache.mkdir(mode=0o700, exist_ok=True)
        if not _private(cache.lstat(), directory=True):
            raise PermissionError('verification cache must be a private owned directory')
        stem = hashlib.sha256(str(root).encode('utf-8')).hexdigest()
        flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)
        lock_fd = os.open(cache / (stem + '.lock'), flags, 0o600)
        if not _private(os.fstat(lock_fd)):
            raise PermissionError('verification lock must be a private owned file')
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        receipt = cache / (stem + '.json')
        # Do not follow an externally supplied receipt link.
        if receipt.is_symlink():
            raise PermissionError('verification receipt cannot be a symbolic link')
        files = _read(receipt, root)
        decision['enabled'] = True
        decision['hit'] = files.get(key) == dict(identity=list(identity), sha256=expected)
    except OSError:
        decision['unavailable'] = True
        if lock_fd is not None:
            os.close(lock_fd)
            lock_fd = None
    try:
        yield decision
        if decision['enabled'] and not decision['hit']:
            files[key] = dict(identity=list(identity), sha256=expected)
            try:
                decision['published'] = _publish(receipt, root, files)
            except OSError:
                decision['unavailable'] = True
    finally:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
