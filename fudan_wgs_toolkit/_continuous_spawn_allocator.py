"""Startup allocator policy for continuous Gaussian spawn workers.

Use around the mature multi-GPU Process.start block, never inside its worker
target: CPython chooses PYTHONMALLOC before importing that target. This scope
only changes inherited startup environment, not the running parent's allocator.
Explicit PYTHONMALLOC wins; glibc environment is never read or changed here.
Resource acceptance is measured separately from this startup policy.
"""
from collections.abc import Mapping
from contextlib import contextmanager
import os
import sys
import threading


_DEFAULTS = {'PYTHONMALLOC': 'malloc'}
_SPAWN_ENVIRONMENT_LOCK = threading.RLock()


def _family_values(families):
    values = families.values() if isinstance(families, Mapping) else families
    if not isinstance(families, (Mapping, list, tuple)):
        return ()
    return tuple(values)


@contextmanager
def continuous_spawn_allocator_scope(*, input_contract, families, worker_count,
                                     enabled=True):
    """Supply absent startup defaults and restore parent environment exactly.

    ``families`` must come from the caller's authenticated prepared metadata or
    model inventory, not a guessed family from phenotype values. Only the family
    values are inspected; names and model data are neither retained nor emitted.

    All multi-GPU callers sharing this scope are serialized during their starts,
    including inactive mixed/binary callers. Arbitrary unrelated subprocess
    launchers cannot be protected by this lock and must not overlap this scoped
    environment block. Parent CPython/glibc allocation policy is unchanged.
    """
    if type(worker_count) is not int or worker_count < 1:
        raise ValueError('worker_count must be a positive integer')
    if type(enabled) is not bool:
        raise ValueError('enabled must be a boolean')
    values = _family_values(families)
    eligible = (enabled and input_contract == 'continuous' and worker_count > 1
                and bool(values) and all(family == 'gaussian' for family in values))
    supported = (sys.implementation.name == 'cpython' and sys.platform == 'linux'
                 and not sys.flags.ignore_environment)
    receipt = dict(schema_version=1, policy='continuous_gaussian_spawn_allocator',
        eligible=eligible, active=eligible and supported, worker_count=worker_count,
        family_count=len(values), all_gaussian=bool(values) and all(value == 'gaussian' for value in values),
        parent_allocator_modified=False, single_gpu_inline_policy_modified=False,
        startup_environment_ignored=bool(sys.flags.ignore_environment),
        startup_allocator_runtime_verified=False,
        default_environment_added={}, explicit_environment_preserved={},
        effective_startup_environment={},
        parent_environment_restored=False)
    # Lock even an inactive multi-GPU caller, so it cannot inherit the active
    # candidate of another cooperating dispatcher. Inline never changes env.
    with _SPAWN_ENVIRONMENT_LOCK:
        previous = {name: os.environ.get(name) for name in _DEFAULTS}
        try:
            if receipt['active']:
                for name, value in _DEFAULTS.items():
                    if name in os.environ:
                        receipt['explicit_environment_preserved'][name] = os.environ[name]
                    else:
                        os.environ[name] = value
                        receipt['default_environment_added'][name] = value
            receipt['effective_startup_environment'] = {name: os.environ.get(name) for name in _DEFAULTS}
            yield receipt
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            receipt['parent_environment_restored'] = all(os.environ.get(name) == value for name, value in previous.items())

