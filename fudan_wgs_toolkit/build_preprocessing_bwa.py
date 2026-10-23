"""Rebuild fixed original BWA 0.7.17 or its pair-local mate-profile candidate.

Uses an existing Conda C toolchain/zlib without installation. The original
Makefile, DP/tie order and pairing/batch options are retained. No reference is
indexed and no reads are aligned by this builder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import urllib.request

from . import build_preprocessing_fastp as _common
from ._preprocessing_io import atomic, bound, digest

SOURCE_COMMIT = '9f26bfcc7780753129b60717ecab0ebba6f04b7c'
SOURCE_URL = 'https://codeload.github.com/lh3/bwa/tar.gz/' + SOURCE_COMMIT
SOURCE_BYTES = 232802
SOURCE_SHA256 = '79763dd6c48e8b01b8e27d99e6e5575d8bbca419e3eed9f3e330af072e63b370'
PATCH_SHA256 = '923b124eca08c9d5dd469f33eada27ee7c94ebcbede02704aeba27161e98ce48'
_PACKAGE = Path(__file__).resolve().parent / '_preprocessing_bwa'
_STATUS = 'compiled_API_contract_passed_not_pipeline_validated'
_BUILD_SCHEMA = 'fudan-bwa-build-v1'
_ASSET_SCHEMA = 'fudan-bwa-assets-v1'
_CONTRACT = 'bwa-matesw-workspace-v1'


def _manifest():
    value = json.loads((_PACKAGE / 'source_manifest.json').read_text())
    if (value.get('schema') != 'fudan-bwa-source-v1' or value.get('commit') != SOURCE_COMMIT
            or value.get('source_url') != SOURCE_URL or value.get('archive_bytes') != SOURCE_BYTES
            or value.get('archive_sha256') != SOURCE_SHA256
            or value.get('archive_root') != 'bwa-' + SOURCE_COMMIT
            or value.get('license') != 'GPL-3.0' or value.get('contract_schema') != _CONTRACT):
        raise RuntimeError('Use the fixed official BWA 0.7.17 source manifest')
    for key in ('original_files', 'mate_profile_files'):
        files = value.get(key, {})
        if len(files) != 77:
            raise RuntimeError('Incomplete 77-file BWA source closure')
        for name, sha in files.items():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or str(path) != name or not re.fullmatch('[a-f0-9]{64}', sha):
                raise RuntimeError('Unsafe BWA source manifest entry')
    before, after = value['original_files'], value['mate_profile_files']
    if set(before) != set(after) or {name for name in before if before[name] != after[name]} != {'bwamem_pair.c'}:
        raise RuntimeError('The BWA candidate must change only bwamem_pair.c')
    if set(value.get('packaged_files', {})) != {'mate_profile.patch', 'mate_profile_contract.c', 'BWA_COPYING', 'version_probe.py'}:
        raise RuntimeError('Incomplete BWA patch/license/API contract package')
    for name, sha in value['packaged_files'].items():
        if (_PACKAGE / name).is_symlink() or digest(_PACKAGE / name) != sha:
            raise RuntimeError('Packaged BWA resource changed: ' + name)
    if value['packaged_files']['mate_profile.patch'] != PATCH_SHA256 or value['license_sha256'] != digest(_PACKAGE / 'BWA_COPYING'):
        raise RuntimeError('BWA patch or original GPL license differs')
    return value


def _apply_patch(source, manifest):
    """One exact upstream file, no external patch process or fuzzy matching."""
    lines = (_PACKAGE / 'mate_profile.patch').read_bytes().splitlines(keepends=True)
    if lines[:2] != [b'--- a/bwamem_pair.c\n', b'+++ b/bwamem_pair.c\n']:
        raise RuntimeError('Unexpected BWA patch target')
    original = (source / 'bwamem_pair.c').read_bytes().splitlines(keepends=True)
    result, position, cursor = [], 0, 2
    while cursor < len(lines):
        match = re.fullmatch(rb'@@ -(\d+),(\d+) \+(\d+),(\d+) @@\n', lines[cursor])
        if match is None:
            raise RuntimeError('Invalid BWA patch hunk')
        start, old_count, new_start, new_count = map(int, match.groups()); start -= 1
        if start < position or start > len(original):
            raise RuntimeError('Out-of-order BWA patch hunk')
        result.extend(original[position:start]); position = start
        if len(result) != new_start - 1:
            raise RuntimeError('BWA patch new-line position differs')
        consumed = emitted = 0; cursor += 1
        while cursor < len(lines) and lines[cursor][:1] in (b' ', b'+', b'-'):
            sign, body = lines[cursor][:1], lines[cursor][1:]
            if sign != b'+':
                if position >= len(original) or original[position] != body:
                    raise RuntimeError('BWA patch context differs; no fuzz accepted')
                position += 1; consumed += 1
            if sign != b'-':
                result.append(body); emitted += 1
            cursor += 1
        if (consumed, emitted) != (old_count, new_count):
            raise RuntimeError('BWA patch hunk lengths differ')
    result.extend(original[position:]); content = b''.join(result)
    if hashlib.sha256(content).hexdigest() != manifest['mate_profile_files']['bwamem_pair.c']:
        raise RuntimeError('Patched BWA source SHA differs')
    (source / 'bwamem_pair.c').write_bytes(content)


def build_preprocessing_bwa(conda_prefix, build_dir, *, variant='mate_profile', source_archive=None,
                            cpu_affinity=None, host_memory_limit_gib=2, host_memory_reserve_gib=4):
    """Return one independently built original or mate_profile executable.

    The ABI contract links a patched translation unit against the original
    upstream libbwa.a. Thus a mate_profile build also compiles an original
    oracle library; an original build creates an independent patched contract
    source copy. Neither is used for actual mapping or installed into Conda.
    The actual no-argument BWA version probe intentionally exits 1; its checked
    wrapper records that exact exit instead of accepting arbitrary failures.
    """
    from .single_preprocessing import _Runner, _limits
    from .build_preprocessing_aligner import _admit

    manifest = _manifest()
    if variant not in ('original', 'mate_profile'):
        raise ValueError('variant must be original or mate_profile')
    prefix, output = Path(conda_prefix).resolve(strict=True), Path(build_dir).resolve()
    if not (prefix / 'conda-meta').is_dir():
        raise ValueError('Use an existing Conda environment')
    if (output.exists() or output == prefix or prefix in output.parents or output in prefix.parents
            or output == _PACKAGE or _PACKAGE in output.parents or output in _PACKAGE.parents):
        raise ValueError('build_dir must be new and outside the Conda prefix and package')
    if any(not re.fullmatch(r'[A-Za-z0-9_./+\-]+', str(path)) for path in (prefix, output)):
        raise ValueError('Build paths must use ASCII letters, digits and / . _ + - only')
    limits = _limits()
    cpus = list(cpu_affinity) if cpu_affinity is not None else sorted(os.sched_getaffinity(0))[:1]
    if (not cpus or len(cpus) > 4 or len(cpus) != len(set(cpus))
            or any(type(cpu) is not int for cpu in cpus) or not set(cpus) <= os.sched_getaffinity(0)
            or len(cpus) > limits['cpu_quota_count'] or any(len(cpus) > q for q in limits['finite_cpu_quotas'])):
        raise ValueError('Use 1..4 distinct allowed CPUs within every finite CPU quota')
    for number, low, high in ((host_memory_limit_gib, 1, 16), (host_memory_reserve_gib, 1, 1024)):
        if (not isinstance(number, (int, float)) or isinstance(number, bool)
                or not math.isfinite(number) or not low <= number <= high):
            raise ValueError('Use finite 1..16 GiB task memory and 1..1024 GiB shared reserve')
    tools = {name: prefix / 'bin' / file for name, file in
             (('cc', 'x86_64-conda-linux-gnu-gcc'), ('ar', 'x86_64-conda-linux-gnu-ar'), ('make', 'make'))}
    if any(not path.is_file() or not os.access(path, os.X_OK) for path in tools.values()):
        raise ValueError('Conda must already provide GCC, ar and make')
    zlib = {name: prefix / directory / name for directory, name in
            (('include', 'zlib.h'), ('include', 'zconf.h'), ('lib', 'libz.so.1'))}
    if any(not path.is_file() for path in zlib.values()):
        raise ValueError('Conda must already provide zlib headers and shared library')
    tool_bindings = {key: bound(path) for key, path in tools.items()}
    zlib_bindings = {key: bound(path) for key, path in zlib.items()}
    if any(prefix not in Path(item['path']).parents for item in [*tool_bindings.values(), *zlib_bindings.values()]):
        raise ValueError('Compiler, ar, make and zlib must resolve inside Conda')
    records = {path.name: digest(path) for path in sorted((prefix / 'conda-meta').glob('*.json'))}
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    admission = _admit(cpus, int(host_memory_limit_gib * 2**30), int(host_memory_reserve_gib * 2**30), parent, 512 * 2**20)
    output.mkdir(parents=True, mode=0o700)
    runner = None
    receipt = dict(schema=_BUILD_SCHEMA, status='building', variant=variant,
                   mapping_validation_completed=False, GPU_work_started=False)
    try:
        config = dict(cpu_affinity=cpus, host_memory_limit_gib=host_memory_limit_gib,
                      host_memory_reserve_gib=host_memory_reserve_gib, variant=variant,
                      builder_sha256=digest(__file__), common_helper_sha256=digest(_common.__file__),
                      source_manifest_sha256=digest(_PACKAGE / 'source_manifest.json'),
                      toolchain=tool_bindings, zlib=zlib_bindings, Conda_package_records=records,
                      subprocess_build_overrides_removed=list(_common._BUILD_OVERRIDES))
        runner = _Runner(output, prefix, config, False)
        runner.env = _common._safe_environment(runner.env)
        for key in _common._BUILD_OVERRIDES:
            runner.env.pop(key, None)
        runner.env.update(CONDA_PREFIX=str(prefix), CUDA_VISIBLE_DEVICES='', TMPDIR=str(output / 'tmp'))
        (output / 'tmp').mkdir()
        receipt.update(config=config, resource_admission=admission)
        atomic(output / 'build_receipt.private.json', receipt)
        archive = output / 'bwa.source.tar.gz'
        if source_archive is not None:
            supplied = Path(source_archive).resolve(strict=True)
            if not supplied.is_file() or supplied.stat().st_size != SOURCE_BYTES or digest(supplied) != SOURCE_SHA256:
                raise RuntimeError('Official BWA source archive size or SHA differs')
            shutil.copyfile(supplied, archive)
        else:
            with urllib.request.urlopen(SOURCE_URL, timeout=60) as stream, archive.open('xb') as dest:
                total = 0
                for chunk in iter(lambda: stream.read(64 * 1024), b''):
                    total += len(chunk)
                    if total > SOURCE_BYTES:
                        raise RuntimeError('Official BWA source archive exceeds its fixed size')
                    dest.write(chunk)
        if archive.stat().st_size != SOURCE_BYTES or digest(archive) != SOURCE_SHA256:
            raise RuntimeError('Official BWA source archive size or SHA differs')
        source = output / 'source'
        # The shared extractor verifies the whole fixed bounded archive before
        # writing, rejects links/duplicates/traversal and verifies every SHA.
        _common._extract(archive, source, manifest)
        if variant == 'mate_profile':
            _apply_patch(source, manifest)
        selected_sources = manifest[variant + '_files']
        flags = '-g -Wall -Wno-unused-function -O2 -fcommon'
        include = '-I' + str(prefix / 'include')
        libraries = '-L' + str(prefix / 'lib') + ' -Wl,-rpath,' + str(prefix / 'lib') + ' -lm -lz -lpthread -lrt'
        make = [str(tools['make']), '-j' + str(len(cpus)), 'CC=' + str(tools['cc']),
                'AR=' + str(tools['ar']), 'CFLAGS=' + flags, 'INCLUDES=' + include, 'LIBS=' + libraries]
        runner.stage('compiler_version', [[str(tools['cc']), '--version']], [], 1)
        runner.stage('compile', [make + ['-C', str(source), 'all']], [source / 'bwa', source / 'libbwa.a'], host_memory_limit_gib)
        source_trees = {str(source): selected_sources}
        if variant == 'original':
            oracle, candidate = source, output / 'contract_candidate'
            _common._extract(archive, candidate, manifest); _apply_patch(candidate, manifest)
            source_trees[str(candidate)] = manifest['mate_profile_files']
        else:
            candidate, oracle = source, output / 'contract_original'
            _common._extract(archive, oracle, manifest)
            runner.stage('oracle_compile', [make + ['-C', str(oracle), 'libbwa.a']], [oracle / 'libbwa.a'], host_memory_limit_gib)
            source_trees[str(oracle)] = manifest['original_files']
        contract = output / 'mate_profile_contract'
        argv = [str(tools['cc']), '-g', '-Wall', '-Wno-unused-function', '-O2', '-fcommon',
                '-DHAVE_PTHREAD', '-DUSE_MALLOC_WRAPPERS',
                '-DBWA_PAIR_SOURCE="' + str(candidate / 'bwamem_pair.c') + '"',
                '-I' + str(candidate), include, str(_PACKAGE / 'mate_profile_contract.c'),
                '-L' + str(oracle), '-lbwa', '-L' + str(prefix / 'lib'),
                '-Wl,-rpath,' + str(prefix / 'lib'), '-lm', '-lz', '-lpthread', '-lrt', '-o', str(contract)]
        runner.stage('contract_compile', [argv], [contract], 1)
        runner.stage('API_contract', [[str(contract)]], [], 1)
        if (output / 'stages/API_contract/0.log').read_text().strip() != manifest['contract_expected']:
            raise RuntimeError('Actual BWA API contract output differs')
        version_probe, version_output = output / 'version_probe.private.json', output / 'usage.stdout.txt'
        runner.stage('version', [[sys.executable, '-B', str(_PACKAGE / 'version_probe.py'), str(source / 'bwa'),
                                 str(version_probe), str(version_output)]], [version_probe, version_output], 1)
        probe = json.loads(version_probe.read_text())
        if (probe.get('schema') != 'fudan-bwa-version-probe-v1' or probe.get('status') != 'expected_usage_exit_verified'
                or probe.get('actual_BWA_exit') != 1 or probe.get('expected_version') != '0.7.17-r1188'):
            raise RuntimeError('BWA actual usage exit/version was not verified')
        executable = source / 'bwa'
        loader = _common._elf_interpreter(executable)
        dynamic = _common._dynamic_dependencies(executable, loader, runner.env)
        if dynamic['libz.so.1'] != zlib_bindings['libz.so.1']:
            raise RuntimeError('BWA resolved a different zlib from the Conda binding')
        if records != {path.name: digest(path) for path in sorted((prefix / 'conda-meta').glob('*.json'))}:
            raise RuntimeError('Conda package records changed during BWA build')
        if any(bound(tools[key]) != item for key, item in tool_bindings.items()) or any(bound(zlib[key]) != item for key, item in zlib_bindings.items()):
            raise RuntimeError('Conda build tool or zlib changed during build')
        if any(digest(Path(tree) / name) != sha for tree, files in source_trees.items() for name, sha in files.items()):
            raise RuntimeError('BWA source changed during build')
        result = dict(schema=_ASSET_SCHEMA, status=_STATUS, variant=variant, condaPrefix=str(prefix),
                      executable=bound(executable), loader=bound(loader), dynamic_libraries=dynamic,
                      source_archive=bound(archive), source_trees=source_trees, source_commit=SOURCE_COMMIT,
                      applied_patch_sha256=PATCH_SHA256 if variant == 'mate_profile' else None,
                      license='GPL-3.0', license_sha256=manifest['license_sha256'],
                      builder_sha256=digest(__file__), common_helper_sha256=digest(_common.__file__),
                      source_manifest_sha256=digest(_PACKAGE / 'source_manifest.json'),
                      reported_version='0.7.17-r1188', actual_version_probe_exit=1,
                      version_probe=bound(version_probe), API_contract_schema=_CONTRACT,
                      API_contract_executable=bound(contract), original_oracle_library=bound(oracle / 'libbwa.a'),
                      compile_flags=flags, include_flags=include, link_flags=libraries,
                      mapping_validation_completed=False, GPU_work_started=False)
        receipt.update(status=_STATUS, artifacts=result, steps=runner.events,
                       Conda_package_records_unchanged=True, source_files_unchanged=True, toolchain_zlib_unchanged=True)
        atomic(output / 'build_receipt.private.json', receipt)
        result = dict(result, buildReceipt=str(output / 'build_receipt.private.json'),
                      build_receipt_sha256=digest(output / 'build_receipt.private.json'))
        atomic(output / 'assets.private.json', result)
        return result
    except BaseException as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error), steps=runner.events if runner else [])
        atomic(output / 'build_receipt.private.json', receipt)
        raise


def validate_bwa_assets(assets, *, runtime_env):
    """Check exact source/library provenance before mapping or a resume skip."""
    if isinstance(assets, (str, os.PathLike)):
        assets = json.loads(Path(assets).read_text())
    manifest = _manifest(); env = _common._safe_environment(runtime_env)
    if (not isinstance(assets, dict) or assets.get('schema') != _ASSET_SCHEMA
            or assets.get('status') != _STATUS or assets.get('variant') not in ('original', 'mate_profile')):
        raise ValueError('Use successfully committed BWA build assets')
    receipt_path = Path(assets.get('buildReceipt', ''))
    if not receipt_path.is_file() or digest(receipt_path) != assets.get('build_receipt_sha256'):
        raise RuntimeError('BWA build receipt is missing or SHA changed')
    receipt = json.loads(receipt_path.read_text())
    core = {key: value for key, value in assets.items() if key not in ('buildReceipt', 'build_receipt_sha256')}
    if (receipt.get('schema') != _BUILD_SCHEMA or receipt.get('status') != _STATUS or receipt.get('artifacts') != core
            or any(receipt.get(key) is not True for key in ('Conda_package_records_unchanged', 'source_files_unchanged', 'toolchain_zlib_unchanged'))):
        raise RuntimeError('BWA receipt does not commit these exact assets')
    if (assets.get('builder_sha256') != digest(__file__) or assets.get('common_helper_sha256') != digest(_common.__file__)
            or assets.get('source_manifest_sha256') != digest(_PACKAGE / 'source_manifest.json')
            or assets.get('source_commit') != SOURCE_COMMIT or assets.get('API_contract_schema') != _CONTRACT
            or assets.get('reported_version') != '0.7.17-r1188' or assets.get('actual_version_probe_exit') != 1
            or assets.get('applied_patch_sha256') != (PATCH_SHA256 if assets['variant'] == 'mate_profile' else None)
            or assets.get('license') != 'GPL-3.0' or assets.get('license_sha256') != manifest['license_sha256']):
        raise RuntimeError('BWA builder/shared helper/source/API binding changed; use a new build')
    for key in ('executable', 'loader', 'source_archive', 'version_probe', 'API_contract_executable', 'original_oracle_library'):
        saved = assets[key]
        if bound(saved['path']) != saved:
            raise RuntimeError('BWA artifact identity or SHA changed: ' + key)
    if assets['source_archive']['size'] != SOURCE_BYTES or assets['source_archive']['sha256'] != SOURCE_SHA256:
        raise RuntimeError('BWA source archive binding differs')
    trees = assets.get('source_trees', {})
    if (len(trees) != 2 or list(trees.values()).count(manifest['original_files']) != 1
            or list(trees.values()).count(manifest['mate_profile_files']) != 1):
        raise RuntimeError('BWA assets need original and patched source closures')
    selected = Path(assets['executable']['path']).parent
    if trees.get(str(selected)) != manifest[assets['variant'] + '_files']:
        raise RuntimeError('BWA executable is not bound to its selected source variant')
    for tree, files in trees.items():
        for name, sha in files.items():
            p = Path(tree) / name
            if p.is_symlink() or digest(p) != sha:
                raise RuntimeError('Committed BWA source changed: ' + name)
    if _common._elf_interpreter(assets['executable']['path']) != assets['loader']['path']:
        raise RuntimeError('BWA ELF interpreter changed')
    for name, item in assets['dynamic_libraries'].items():
        if bound(item['path']) != item:
            raise RuntimeError('BWA dynamic library changed: ' + name)
    actual = _common._dynamic_dependencies(assets['executable']['path'], assets['loader']['path'], env)
    if actual != assets['dynamic_libraries']:
        raise RuntimeError('Runtime resolves different BWA libraries; use its build Conda prefix/LD_LIBRARY_PATH or rebuild assets')
    if Path(assets['condaPrefix']) not in Path(actual['libz.so.1']['path']).parents:
        raise RuntimeError('BWA libz escaped its selected Conda prefix')
    return assets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--conda-prefix', required=True); p.add_argument('--build-dir', required=True)
    p.add_argument('--variant', choices=('original', 'mate_profile'), default='mate_profile')
    p.add_argument('--source-archive'); p.add_argument('--cpu-affinity')
    p.add_argument('--host-memory-limit-gib', type=float, default=2); p.add_argument('--host-memory-reserve-gib', type=float, default=4)
    a = p.parse_args(); cpus = [int(x) for x in a.cpu_affinity.split(',')] if a.cpu_affinity else None
    print(json.dumps(build_preprocessing_bwa(a.conda_prefix, a.build_dir, variant=a.variant, source_archive=a.source_archive,
        cpu_affinity=cpus, host_memory_limit_gib=a.host_memory_limit_gib, host_memory_reserve_gib=a.host_memory_reserve_gib)))


if __name__ == '__main__':
    main()
