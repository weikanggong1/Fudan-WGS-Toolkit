"""Build fixed upstream fastp 0.20.0, optionally with the direct-append patch.

Only a new directory is written. Existing Conda compilers and zlib are used;
neither the environment nor installed fastp is replaced. A build/serializer
contract receipt is not evidence of complete FASTQ-to-PLINK validation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import tarfile
import urllib.request

from ._preprocessing_io import atomic, bound, digest

SOURCE_URL = 'https://codeload.github.com/OpenGene/fastp/tar.gz/refs/tags/v0.20.0'
SOURCE_BYTES = 139422
SOURCE_SHA256 = '8d751d2746db11ff233032fc49e3bcc8b53758dd4596fdcf4b4099a4d702ac22'
PATCH_SHA256 = '0e4d8f6a523db4f656b1472070887ad54c2f280c425452b4d573b12f77f39442'
_ASSET_SCHEMA = 'fudan-fastp-assets-v1'
_BUILD_SCHEMA = 'fudan-fastp-build-v1'
_STATUS = 'compiled_contract_passed_not_pipeline_validated'
_PACKAGE = Path(__file__).resolve().parent / '_preprocessing_fastp'
_BUILD_OVERRIDES = ('MAKEFLAGS', 'MFLAGS', 'MAKEFILES', 'CXXFLAGS', 'CFLAGS', 'CPPFLAGS',
                    'LDFLAGS', 'LIBRARY_PATH', 'CPATH', 'C_INCLUDE_PATH',
                    'CPLUS_INCLUDE_PATH', 'GCC_EXEC_PREFIX', 'COMPILER_PATH', 'LD_GOLD')


def _manifest():
    manifest = json.loads((_PACKAGE / 'source_manifest.json').read_text())
    if (manifest.get('schema') != 'fudan-fastp-source-v1'
            or manifest.get('tag') != 'v0.20.0' or manifest.get('source_url') != SOURCE_URL
            or manifest.get('archive_bytes') != SOURCE_BYTES
            or manifest.get('archive_sha256') != SOURCE_SHA256
            or manifest.get('archive_root') != 'fastp-0.20.0'
            or manifest.get('license') != 'MIT'
            or manifest.get('packaged_files', {}).get('fastp-0.20.0-append-direct.patch') != PATCH_SHA256):
        raise RuntimeError('Use the fixed fastp 0.20.0 source manifest')
    for key in ('original_files', 'append_files'):
        files = manifest.get(key, {})
        if len(files) != 73:
            raise RuntimeError('Incomplete upstream fastp source manifest')
        for name, sha in files.items():
            path = PurePosixPath(name)
            if (path.is_absolute() or '..' in path.parts or str(path) != name
                    or not re.fullmatch('[0-9a-f]{64}', sha)):
                raise RuntimeError('Invalid upstream fastp source entry')
    if set(manifest['original_files']) != set(manifest['append_files']):
        raise RuntimeError('The patch must preserve the complete upstream source tree')
    changed = {name for name in manifest['original_files']
               if manifest['original_files'][name] != manifest['append_files'][name]}
    if changed != {'src/read.cpp', 'src/read.h', 'src/peprocessor.cpp'}:
        raise RuntimeError('Unexpected fastp patch scope')
    if set(manifest['packaged_files']) != {'fastp-0.20.0-append-direct.patch', 'append_contract.cpp', 'FASTP_LICENSE'}:
        raise RuntimeError('Incomplete packaged fastp resources')
    for name, sha in manifest['packaged_files'].items():
        source = _PACKAGE / name
        if source.is_symlink() or digest(source) != sha:
            raise RuntimeError('Packaged fastp resource changed: ' + name)
    if manifest['license_sha256'] != digest(_PACKAGE / 'FASTP_LICENSE'):
        raise RuntimeError('The original MIT license changed')
    return manifest


def _extract(archive, destination, manifest):
    """Validate the entire bounded archive before creating any extracted file."""
    with tarfile.open(archive, mode='r:gz') as source:
        entries = source.getmembers()
        if len(entries) > 128:
            raise ValueError('Too many fastp source archive entries')
        files, seen, total = [], set(), 0
        for entry in entries:
            path = PurePosixPath(entry.name)
            if (path.is_absolute() or '..' in path.parts or not path.parts
                    or path.parts[0] != manifest['archive_root']):
                raise ValueError('Unsafe fastp source archive member')
            relative = PurePosixPath(*path.parts[1:])
            if not path.parts[1:]:
                if not entry.isdir():
                    raise ValueError('The fastp archive root must be a directory')
                continue
            name = str(relative)
            if name in seen or str(path) != entry.name.rstrip('/'):
                raise ValueError('Duplicate or noncanonical fastp archive member')
            seen.add(name)
            if entry.isdir():
                continue
            if not entry.isfile() or name not in manifest['original_files'] or entry.size < 0:
                raise ValueError('Unexpected file/link in fastp source archive')
            total += entry.size
            if total > 1024 * 1024:
                raise ValueError('The fastp source archive exceeds its extracted budget')
            files.append((entry, name))
        if {name for _, name in files} != set(manifest['original_files']):
            raise ValueError('Incomplete fastp source archive')
        destination.mkdir()
        for entry, name in files:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(entry) as stream, target.open('xb') as output:
                shutil.copyfileobj(stream, output, 64 * 1024)
            if digest(target) != manifest['original_files'][name]:
                raise RuntimeError('Original fastp source SHA-256 differs: ' + name)


def _apply_patch(source, manifest):
    """Apply exact, non-fuzzy unified hunks; no system patch executable needed."""
    lines = (_PACKAGE / 'fastp-0.20.0-append-direct.patch').read_bytes().splitlines(keepends=True)
    cursor, replacements = 0, {}
    while cursor < len(lines):
        before = lines[cursor].decode().rstrip('\n')
        after = lines[cursor + 1].decode().rstrip('\n') if cursor + 1 < len(lines) else ''
        if not before.startswith('--- a/') or after != '+++ b/' + before[6:]:
            raise RuntimeError('Invalid packaged fastp patch header')
        name = before[6:]
        if name in replacements or name not in manifest['original_files']:
            raise RuntimeError('Duplicate or unknown fastp patch target')
        original = (source / name).read_bytes().splitlines(keepends=True)
        result, position = [], 0
        cursor += 2
        while cursor < len(lines) and not lines[cursor].startswith(b'--- '):
            match = re.fullmatch(rb'@@ -(\d+),(\d+) \+(\d+),(\d+) @@\n', lines[cursor])
            if match is None:
                raise RuntimeError('Invalid packaged fastp patch hunk')
            start, old_count, new_start, new_count = map(int, match.groups())
            start -= 1
            if start < position or start > len(original):
                raise RuntimeError('Out-of-order fastp patch hunk')
            result.extend(original[position:start]); position = start
            if len(result) != new_start - 1:
                raise RuntimeError('Fastp patch new-line position differs')
            consumed = emitted = 0; cursor += 1
            while cursor < len(lines) and lines[cursor][:1] in (b' ', b'+', b'-'):
                if lines[cursor].startswith(b'--- '):
                    break
                sign, body = lines[cursor][:1], lines[cursor][1:]
                if sign != b'+':
                    if position >= len(original) or original[position] != body:
                        raise RuntimeError('Fastp patch context differs; no fuzz is accepted')
                    position += 1; consumed += 1
                if sign != b'-':
                    result.append(body); emitted += 1
                cursor += 1
            if (consumed, emitted) != (old_count, new_count):
                raise RuntimeError('Fastp patch hunk lengths differ')
        result.extend(original[position:])
        content = b''.join(result)
        import hashlib
        if hashlib.sha256(content).hexdigest() != manifest['append_files'][name]:
            raise RuntimeError('Patched fastp source SHA-256 differs: ' + name)
        replacements[name] = content
    if set(replacements) != {'src/read.cpp', 'src/read.h', 'src/peprocessor.cpp'}:
        raise RuntimeError('Incomplete fastp direct-append patch')
    # Validate all hunks before modifying any of the independent source copy.
    for name, content in replacements.items():
        (source / name).write_bytes(content)


def _safe_environment(environment):
    env = dict(environment)
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
        raise ValueError('runtime_env must contain string keys and values')
    # Conda activation's LD_GOLD names a build linker; glibc does not consume it.
    # Keep real loader injection overrides closed, including unknown LD_* names.
    forbidden = [name for name, value in env.items() if value and
                 ((name.startswith('LD_') and name not in ('LD_LIBRARY_PATH', 'LD_GOLD'))
                  or name in ('GLIBC_TUNABLES', 'BASH_ENV', 'ENV'))]
    if forbidden:
        raise ValueError('Unset dynamic-loader overrides before fastp: ' + ', '.join(sorted(forbidden)))
    search = env.get('LD_LIBRARY_PATH', '')
    if search and any(not value or not Path(value).is_absolute() for value in re.split('[:;]', search)):
        raise ValueError('fastp LD_LIBRARY_PATH entries must be absolute and nonempty; cwd-dependent library resolution is refused')
    return env


def _elf_interpreter(executable):
    """Read PT_INTERP, without executing the executable or invoking readelf/ldd."""
    with Path(executable).open('rb') as stream:
        header = stream.read(64)
        if len(header) != 64 or header[:7] != b'\x7fELF\x02\x01\x01':
            raise RuntimeError('This fastp builder requires a Linux x86-64 ELF executable')
        values = struct.unpack('<16sHHIQQQIHHHHHH', header)
        offset, width, count = values[5], values[9], values[10]
        if values[1] not in (2, 3) or values[2] != 62 or width != 56 or not 0 < count <= 128:
            raise RuntimeError('Unsupported fastp ELF program headers')
        interpreters = []
        for number in range(count):
            stream.seek(offset + number * width)
            raw = stream.read(width)
            if len(raw) != width:
                raise RuntimeError('Truncated fastp ELF program headers')
            entry = struct.unpack('<IIQQQQQQ', raw)
            if entry[0] == 3:
                if not 1 < entry[5] <= 4096:
                    raise RuntimeError('Invalid fastp ELF interpreter length')
                stream.seek(entry[2]); value = stream.read(entry[5])
                if len(value) != entry[5] or value[-1:] != b'\0' or b'\0' in value[:-1]:
                    raise RuntimeError('Invalid fastp ELF interpreter')
                path = Path(os.fsdecode(value[:-1]))
                if not path.is_absolute():
                    raise RuntimeError('The fastp ELF interpreter must be absolute')
                interpreters.append(str(path.resolve(strict=True)))
        if len(interpreters) != 1:
            raise RuntimeError('Expected one dynamic fastp ELF interpreter')
        return interpreters[0]


def _dynamic_dependencies(executable, loader, environment):
    """Use the bound ELF loader in trace-only mode under the actual task env."""
    env = _safe_environment(environment)
    result = subprocess.run([str(loader), '--list', str(executable)], env=env,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=20, check=False)
    if result.returncode or len(result.stdout) > 128 * 1024 or len(result.stderr) > 128 * 1024:
        raise RuntimeError('Cannot verify the actual fastp dynamic library resolution')
    libraries = {}
    for line in result.stdout.decode('utf-8', errors='strict').splitlines():
        line = line.strip()
        if not line or line.startswith('linux-vdso.so.'):
            continue
        if '=>' in line:
            name, tail = line.split('=>', 1)
            location = tail.strip().rsplit(' (', 1)[0]
            if not location.startswith('/') or name.strip() in libraries:
                raise RuntimeError('Unresolved or duplicate fastp dynamic dependency')
            libraries[name.strip()] = bound(location)
        elif line.startswith('/'):
            location = line.rsplit(' (', 1)[0]
            if Path(location).resolve(strict=True) != Path(loader).resolve(strict=True):
                raise RuntimeError('Unexpected fastp dynamic-loader trace entry')
        else:
            raise RuntimeError('Unrecognized fastp dynamic-loader trace entry')
    if 'libz.so.1' not in libraries:
        raise RuntimeError('The fastp binary must dynamically link the bound Conda zlib')
    return libraries


def build_preprocessing_fastp(conda_prefix, build_dir, *, variant='append', source_archive=None,
                              cpu_affinity=None, host_memory_limit_gib=2,
                              host_memory_reserve_gib=4):
    """Compile original or direct-append fastp in a new directory, without install.

    Uses 1..4 allowed CPUs, an existing Conda C++/make/zlib toolchain and a
    finite shared host-memory budget. An optional local archive must match the
    same official bytes/SHA as a download. This builder reads no real FASTQ.
    Each variant requires its own new build directory and independent receipt.
    """
    from .single_preprocessing import _Runner, _limits
    from .build_preprocessing_aligner import _admit

    manifest = _manifest()
    if variant not in ('original', 'append'):
        raise ValueError('variant must be original or append')
    prefix, output = Path(conda_prefix).resolve(strict=True), Path(build_dir).resolve()
    if not (prefix / 'conda-meta').is_dir():
        raise ValueError('Use an existing Conda environment')
    if (output.exists() or output == prefix or prefix in output.parents or output in prefix.parents
            or output == _PACKAGE or _PACKAGE in output.parents or output in _PACKAGE.parents):
        raise ValueError('build_dir must be new and outside the Conda prefix and package')
    if any(not re.fullmatch(r'[A-Za-z0-9_./+\-]+', str(path)) for path in (prefix, output)):
        raise ValueError('Build paths must use ASCII letters, digits and / . _ + - only; make/linker metacharacters are refused')
    limits = _limits()
    cpus = list(cpu_affinity) if cpu_affinity is not None else sorted(os.sched_getaffinity(0))[:1]
    if (not cpus or len(cpus) > 4 or len(set(cpus)) != len(cpus)
            or any(type(cpu) is not int for cpu in cpus) or not set(cpus) <= os.sched_getaffinity(0)
            or len(cpus) > limits['cpu_quota_count']
            or any(len(cpus) > quota for quota in limits['finite_cpu_quotas'])):
        raise ValueError('Use one to four distinct allowed CPUs within the actual quota')
    for value, minimum, maximum in ((host_memory_limit_gib, 1, 16), (host_memory_reserve_gib, 1, 1024)):
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or not minimum <= value <= maximum):
            raise ValueError('Use a finite 1..16 GiB build budget and at least 1 GiB reserve')
    tools = {name: prefix / 'bin' / filename for name, filename in
             (('cxx', 'x86_64-conda-linux-gnu-g++'), ('make', 'make'))}
    if any(not path.is_file() or not os.access(path, os.X_OK) for path in tools.values()):
        raise ValueError('The Conda prefix must already provide its C++ compiler and make')
    zlib = {name: prefix / folder / name for folder, name in
            (('include', 'zlib.h'), ('include', 'zconf.h'), ('lib', 'libz.so.1'))}
    if any(not path.is_file() for path in zlib.values()):
        raise ValueError('The Conda prefix must already provide zlib headers and shared library')
    records = {path.name: digest(path) for path in sorted((prefix / 'conda-meta').glob('*.json'))}
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    admission = _admit(cpus, int(host_memory_limit_gib * 2**30),
                       int(host_memory_reserve_gib * 2**30), parent, 512 * 2**20)
    output.mkdir(parents=True, mode=0o700)
    receipt = dict(schema=_BUILD_SCHEMA, status='building', variant=variant,
                   scientific_validation_completed=False, GPU_work_started=False)
    runner = None
    try:
        tool_bindings = {key: bound(path) for key, path in tools.items()}
        zlib_bindings = {key: bound(path) for key, path in zlib.items()}
        if any(prefix not in Path(value['path']).parents
               for value in [*tool_bindings.values(), *zlib_bindings.values()]):
            raise ValueError('Conda compiler/make/zlib must resolve inside the selected prefix')
        config = dict(cpu_affinity=cpus, host_memory_limit_gib=host_memory_limit_gib,
                      host_memory_reserve_gib=host_memory_reserve_gib, variant=variant,
                      builder_sha256=digest(__file__), source_manifest_sha256=digest(_PACKAGE / 'source_manifest.json'),
                      toolchain=tool_bindings, zlib=zlib_bindings, conda_package_records=records,
                      subprocess_build_overrides_removed=list(_BUILD_OVERRIDES))
        runner = _Runner(output, prefix, config, False)
        runner.env = _safe_environment(runner.env)
        for name in _BUILD_OVERRIDES:
            runner.env.pop(name, None)
        runner.env.update(CONDA_PREFIX=str(prefix), CUDA_VISIBLE_DEVICES='',
                          TMPDIR=str(output / 'tmp'))
        (output / 'tmp').mkdir()
        receipt.update(config=config, resource_admission=admission)
        atomic(output / 'build_receipt.private.json', receipt)
        archive = output / 'fastp-v0.20.0.source.tar.gz'
        if source_archive is not None:
            source_input = Path(source_archive).resolve(strict=True)
            if not source_input.is_file() or source_input.stat().st_size != SOURCE_BYTES:
                raise RuntimeError('Official fastp source archive size or SHA-256 differs')
            original_archive = bound(source_input)
            if original_archive['size'] != SOURCE_BYTES or original_archive['sha256'] != SOURCE_SHA256:
                raise RuntimeError('Official fastp source archive size or SHA-256 differs')
            shutil.copyfile(source_input, archive)
        else:
            with urllib.request.urlopen(SOURCE_URL, timeout=60) as stream, archive.open('xb') as target:
                copied = 0
                for chunk in iter(lambda: stream.read(64 * 1024), b''):
                    copied += len(chunk)
                    if copied > SOURCE_BYTES:
                        raise RuntimeError('Official fastp archive exceeds its fixed size')
                    target.write(chunk)
        if archive.stat().st_size != SOURCE_BYTES or digest(archive) != SOURCE_SHA256:
            raise RuntimeError('Official fastp source archive size or SHA-256 differs')
        source = output / 'source'
        _extract(archive, source, manifest)
        if variant == 'append':
            _apply_patch(source, manifest)
        expected_sources = manifest[variant + '_files']
        flags = '-std=c++11 -g -O3 -I./inc -I' + str(prefix / 'include')
        link_flags = '-L' + str(prefix / 'lib') + ' -Wl,-rpath,' + str(prefix / 'lib') + ' -lz -lpthread'
        runner.stage('compiler_version', [[str(tools['cxx']), '--version']], [], 1)
        executable = source / 'fastp'
        runner.stage('compile', [[str(tools['make']), '-C', str(source), '-j' + str(len(cpus)),
                                 'CXX=' + str(tools['cxx']), 'CXXFLAGS=' + flags,
                                 'LD_FLAGS=' + link_flags]], [executable], host_memory_limit_gib)
        contract_source = output / 'append_contract.cpp'
        shutil.copyfile(_PACKAGE / 'append_contract.cpp', contract_source)
        contract = output / 'append_contract'
        runner.stage('contract_compile', [[str(tools['cxx']), '-std=c++11', '-O3', '-Wall', '-Wextra',
                     '-DFASTP_APPEND_PATCHED=' + str(int(variant == 'append')),
                     '-DFASTP_READ_SOURCE="' + str(source / 'src/read.cpp') + '"',
                     '-I' + str(source / 'src'), '-I' + str(prefix / 'include'), str(contract_source),
                     str(source / 'src/sequence.cpp'), '-o', str(contract),
                     '-Wl,-rpath,' + str(prefix / 'lib')]], [contract], 1)
        runner.stage('serializer_contract', [[str(contract)]], [], 1)
        runner.stage('version', [[str(executable), '--version']], [], 1)
        version_log = output / 'stages/version/0.log'
        if version_log.read_text().strip() != 'fastp 0.20.0':
            raise RuntimeError('The built fastp CLI does not report exactly 0.20.0')
        loader = _elf_interpreter(executable)
        libraries = _dynamic_dependencies(executable, loader, runner.env)
        if libraries['libz.so.1'] != zlib_bindings['libz.so.1']:
            raise RuntimeError('The built fastp resolved a different zlib from the Conda binding')
        for name in ('libstdc++.so.6', 'libgcc_s.so.1'):
            if name not in libraries or prefix not in Path(libraries[name]['path']).parents:
                raise RuntimeError('Fastp C++ runtime escaped the selected Conda prefix: ' + name)
        if records != {path.name: digest(path) for path in sorted((prefix / 'conda-meta').glob('*.json'))}:
            raise RuntimeError('Conda package records changed during build')
        if any(bound(tools[key]) != value for key, value in tool_bindings.items()):
            raise RuntimeError('Conda build tool changed during build')
        if any(bound(zlib[key]) != value for key, value in zlib_bindings.items()):
            raise RuntimeError('Conda zlib headers/library changed during build')
        if any(digest(source / name) != sha for name, sha in expected_sources.items()):
            raise RuntimeError('Fastp source changed during build')
        result = dict(schema=_ASSET_SCHEMA, status=_STATUS, variant=variant, condaPrefix=str(prefix),
                      executable=bound(executable), loader=bound(loader), dynamic_libraries=libraries,
                      source_tree=str(source), source_file_sha256=expected_sources,
                      source_archive=bound(archive), source_tag='v0.20.0', source_sha256=SOURCE_SHA256,
                      applied_patch_sha256=PATCH_SHA256 if variant == 'append' else None,
                      license='MIT', license_sha256=manifest['license_sha256'],
                      builder_sha256=digest(__file__), source_manifest_sha256=digest(_PACKAGE / 'source_manifest.json'),
                      compile_flags=flags, link_flags=link_flags,
                      reported_version='fastp 0.20.0', default_fastp_workers=1,
                      scientific_validation_completed=False, GPU_work_started=False)
        receipt.update(status=_STATUS, artifacts=result, steps=runner.events,
                       Conda_package_records_unchanged=True, source_files_unchanged=True,
                       toolchain_unchanged=True, zlib_unchanged=True)
        atomic(output / 'build_receipt.private.json', receipt)
        result = dict(result, buildReceipt=str(output / 'build_receipt.private.json'),
                      build_receipt_sha256=digest(output / 'build_receipt.private.json'))
        atomic(output / 'assets.private.json', result)
        return result
    except BaseException as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error),
                       steps=runner.events if runner is not None else [])
        atomic(output / 'build_receipt.private.json', receipt)
        raise


def validate_fastp_assets(assets, *, runtime_env):
    """Verify a committed build and actual runtime libraries before every use/resume.

    Pass the exact environment that will execute fastp, e.g. ``run.env``. A
    different LD_LIBRARY_PATH is allowed only if every resolved library still
    has the committed identity and SHA. Loader injection overrides are refused.
    This is preflight verification, not a filesystem lock against concurrent
    modification; the pipeline must also freeze these assets in its resume plan.
    """
    env = _safe_environment(runtime_env)
    manifest = _manifest()
    if (assets.get('schema') != _ASSET_SCHEMA or assets.get('status') != _STATUS
            or assets.get('variant') not in ('original', 'append')):
        raise ValueError('Use successfully committed fastp build assets')
    receipt_path = Path(assets.get('buildReceipt', ''))
    if not receipt_path.is_file() or digest(receipt_path) != assets.get('build_receipt_sha256'):
        raise RuntimeError('Fastp build receipt is missing or its SHA-256 differs')
    receipt = json.loads(receipt_path.read_text())
    core = {key: value for key, value in assets.items() if key not in ('buildReceipt', 'build_receipt_sha256')}
    if (receipt.get('schema') != _BUILD_SCHEMA or receipt.get('status') != _STATUS
            or receipt.get('artifacts') != core
            or any(receipt.get(key) is not True for key in ('Conda_package_records_unchanged',
                   'source_files_unchanged', 'toolchain_unchanged', 'zlib_unchanged'))):
        raise RuntimeError('Fastp build receipt does not commit these exact assets')
    expected_patch = PATCH_SHA256 if assets['variant'] == 'append' else None
    if (assets.get('builder_sha256') != digest(__file__)
            or assets.get('source_manifest_sha256') != digest(_PACKAGE / 'source_manifest.json')
            or assets.get('source_tag') != 'v0.20.0' or assets.get('source_sha256') != SOURCE_SHA256
            or assets.get('applied_patch_sha256') != expected_patch
            or assets.get('source_file_sha256') != manifest[assets['variant'] + '_files']
            or assets.get('license') != 'MIT' or assets.get('license_sha256') != manifest['license_sha256']
            or assets.get('default_fastp_workers') != 1):
        raise RuntimeError('Fastp packaged source/patch differs from the committed build')
    for label in ('executable', 'loader', 'source_archive'):
        saved = assets.get(label, {})
        if bound(saved['path']) != saved:
            raise RuntimeError('Fastp artifact identity or SHA-256 changed: ' + label)
    if (assets['source_archive']['size'] != SOURCE_BYTES
            or assets['source_archive']['sha256'] != SOURCE_SHA256):
        raise RuntimeError('Fastp source archive binding differs')
    if _elf_interpreter(assets['executable']['path']) != assets['loader']['path']:
        raise RuntimeError('The fastp ELF interpreter changed')
    source = Path(assets['source_tree'])
    for name, sha in assets['source_file_sha256'].items():
        if (source / name).is_symlink() or digest(source / name) != sha:
            raise RuntimeError('Committed fastp source changed: ' + name)
    for name, saved in assets.get('dynamic_libraries', {}).items():
        if bound(saved['path']) != saved:
            raise RuntimeError('Fastp dynamic library identity or SHA-256 changed: ' + name)
    libraries = _dynamic_dependencies(assets['executable']['path'], assets['loader']['path'], env)
    if libraries != assets.get('dynamic_libraries'):
        expected = assets.get('dynamic_libraries', {})
        changed = sorted(name for name in set(expected) | set(libraries)
                         if expected.get(name) != libraries.get(name))
        raise RuntimeError('Runtime environment resolves different fastp libraries: ' + ', '.join(changed) +
                           '. Use the build Conda prefix and its LD_LIBRARY_PATH, or rebuild fastp assets '
                           'in the selected environment; do not reuse or relabel changed assets.')
    prefix = Path(assets['condaPrefix'])
    if prefix not in Path(libraries['libz.so.1']['path']).parents:
        raise RuntimeError('The runtime zlib escaped the bound Conda prefix')
    return assets


def fastp_argv(assets, *, read1, read2, output1, output2, json_report, html_report,
               runtime_env, workers=1):
    """Return the ordinary paired fastp argv, preserving original filtering defaults.

    No extra filter, compression, adapter, overlap, stdout or reordering options
    are added. ``workers=1`` remains the pipeline default for stable ordering.
    The caller binds inputs/outputs and full assets into its own stage/resume
    plan; this helper validates the actual tool/libraries even for a skipped stage.
    """
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('fastp workers must be an integer in 1..16')
    validate_fastp_assets(assets, runtime_env=runtime_env)
    values = [read1, read2, output1, output2, json_report, html_report]
    if any(not isinstance(value, (str, os.PathLike)) or not str(value) or '\0' in str(value) for value in values):
        raise ValueError('Use six nonempty file paths')
    normalized = [str(Path(value).resolve()) for value in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError('Paired inputs and four outputs must use distinct paths')
    return [assets['executable']['path'], '-i', str(read1), '-I', str(read2),
            '-o', str(output1), '-O', str(output2), '-j', str(json_report),
            '-h', str(html_report), '-w', str(workers)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conda-prefix', required=True)
    parser.add_argument('--build-dir', required=True)
    parser.add_argument('--variant', choices=('original', 'append'), default='append')
    parser.add_argument('--source-archive')
    parser.add_argument('--cpu-affinity')
    parser.add_argument('--host-memory-limit-gib', type=float, default=2)
    parser.add_argument('--host-memory-reserve-gib', type=float, default=4)
    args = parser.parse_args()
    cpus = [int(value) for value in args.cpu_affinity.split(',')] if args.cpu_affinity else None
    print(json.dumps(build_preprocessing_fastp(args.conda_prefix, args.build_dir,
          variant=args.variant, source_archive=args.source_archive, cpu_affinity=cpus,
          host_memory_limit_gib=args.host_memory_limit_gib,
          host_memory_reserve_gib=args.host_memory_reserve_gib)))


if __name__ == '__main__':
    main()
