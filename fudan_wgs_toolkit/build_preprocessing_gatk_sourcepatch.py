"""Build a small overlay from fixed original GATK 4.2.0.0 BQSR sources.

The installed Conda jar is never modified. Only the two patched upstream
classes and a same-JVM class-binding launcher enter the overlay. Build/probe
success is separate from real-data precision, speed and memory validation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import time
import urllib.request
import zipfile

_COMMIT = '1f7fade2afbd706539eb983087907a0cb11ee734'
_VERSION = '4.2.0.0'
_SOURCE_SCHEMA = 'fudan-original-gatk-sourcepatch-source-v1'
_ASSET_SCHEMA = 'fudan-original-gatk-sourcepatch-assets-v1'
_BUILD_SCHEMA = 'fudan-original-gatk-sourcepatch-build-v1'
_CLASS_SCHEMA = 'fudan-original-gatk-sourcepatch-classes-v1'
_STATUS = 'compiled_class_binding_probe_passed_not_scientific_validation'
_MAIN = 'org.fudan.wgs.gatkpatch.SourcePatchedBqsrMain'
_REQUIRED_CLASSES = {
    'org.broadinstitute.hellbender.utils.recalibration.BaseRecalibrationEngine',
    'org.broadinstitute.hellbender.utils.recalibration.ReadRecalibrationInfo', _MAIN}
_PACKAGE_FILES = {'assets/bqsr-allocations.patch', 'assets/SourcePatchedBqsrMain.java',
                  'LICENSE.GATK-4.2.0.0.txt', 'SOURCES.md', 'README.zh.md'}
_BASE = 'src/main/java/org/broadinstitute/hellbender/utils/recalibration/'
_UPSTREAM = {
    _BASE+'BaseRecalibrationEngine.java': 'de5ce3fa392bc2f38f551c1b755cab6029d2cc7916c41b5e8b1c5032b2b443fc',
    _BASE+'ReadRecalibrationInfo.java': '74cea96d4c784b8abac3ff860a39605570e69329a2b890423c4eced428897181',
}
_LICENSE_SHA = '0e53509666a2480048ef7ae7dfdfc24a075fc1d6492295f5fa93aa60eb7d226a'
_DEFAULTS = ('-Dsamjdk.use_async_io_read_samtools=false', '-Dsamjdk.use_async_io_write_samtools=true',
             '-Dsamjdk.use_async_io_write_tribble=false', '-Dsamjdk.compression_level=2')


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''): result.update(data)
    return result.hexdigest()


def _safe_path(path):
    value = Path(path)
    if not value.is_absolute() or ':' in str(value) or '\n' in str(value) or '\r' in str(value):
        raise ValueError('Use absolute Linux paths without classpath separators')
    return value.resolve(strict=True)


def _expected_gatk_version(text):
    """Accept the original version banner, including its literal ``v`` prefix.

    GATK's version and dependency versions share one output stream. Require
    exactly one GATK banner and its complete four-part version, rather than
    accepting a version substring from a dependency or a longer release name.
    """
    versions = re.findall(r'^The Genome Analysis Toolkit \(GATK\) v?([0-9]+(?:\.[0-9]+){3})\r?$',
                          text, flags=re.MULTILINE)
    return versions == [_VERSION]


def _manifest(package=None):
    root = Path(package) if package is not None else Path(__file__).with_name('_preprocessing_gatk_sourcepatch')
    manifest = root/'source_manifest.json'
    value = json.loads(manifest.read_text())
    if (value.get('schema') != _SOURCE_SCHEMA or value.get('upstream_commit') != _COMMIT
            or value.get('upstream_version') != _VERSION
            or value.get('license', {}).get('sha256') != _LICENSE_SHA
            or {x['path']: x['upstream_sha256'] for x in value.get('sources', [])} != _UPSTREAM):
        raise RuntimeError('Use the fixed original GATK 4.2.0.0 source manifest')
    if set(value.get('files', {})) != _PACKAGE_FILES:
        raise RuntimeError('Source manifest must bind the complete public patch and launcher assets')
    for name, expected in value['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise RuntimeError('Unsafe source-patch asset path')
        path = root/relative
        if path.is_symlink() or not path.is_file() or _sha(path) != expected:
            raise RuntimeError('Packaged GATK source-patch asset changed: '+name)
    return root, manifest, value


def _download_sources(package, output, cache=None):
    """Worker invoked under the existing guarded runner; never on import."""
    _, _, manifest = _manifest(package)
    output = Path(output); output.mkdir()
    entries = [dict(path=x['path'], sha256=x['upstream_sha256'], bytes=x['upstream_bytes']) for x in manifest['sources']]
    entries.append(dict(path='LICENSE.TXT', sha256=_LICENSE_SHA, bytes=10966))
    for entry in entries:
        target = output/entry['path']; target.parent.mkdir(parents=True, exist_ok=True)
        if cache is not None:
            source = Path(cache)/entry['path']
            if source.is_symlink() or not source.is_file() or source.stat().st_size != entry['bytes'] or _sha(source) != entry['sha256']:
                raise RuntimeError('Offline original-source cache differs from the fixed upstream bytes')
            shutil.copyfile(source, target)
            if _sha(target) != entry['sha256']:
                raise RuntimeError('Original source changed during cache copy')
            continue
        url = 'https://raw.githubusercontent.com/broadinstitute/gatk/'+_COMMIT+'/'+entry['path']
        deadline = time.monotonic()+120
        temporary = target.with_suffix(target.suffix+'.pending')
        request = urllib.request.Request(url, headers={'User-Agent': 'Fudan-WGS-Toolkit-source-builder'})
        try:
            with urllib.request.urlopen(request, timeout=30) as response, temporary.open('xb') as stream:
                if response.geturl() != url: raise RuntimeError('Unexpected redirect for fixed GATK source')
                count = 0
                while True:
                    if time.monotonic() > deadline: raise TimeoutError('Source download deadline exceeded')
                    data = response.read(65536)
                    if not data: break
                    count += len(data)
                    if count > entry['bytes']: raise RuntimeError('Fixed source exceeds expected byte length')
                    stream.write(data)
                stream.flush(); os.fsync(stream.fileno())
            if count != entry['bytes'] or _sha(temporary) != entry['sha256']:
                raise RuntimeError('Fixed upstream source byte length or SHA256 differs')
            os.replace(temporary, target)
        except BaseException:
            # Failed temporary bytes remain as private diagnostic evidence.
            raise


def _apply_patch(package, upstream, derived):
    """Apply exact unified contexts for the two allowlisted files, no fuzz/offset."""
    root, _, manifest = _manifest(package)
    lines = (root/'assets/bqsr-allocations.patch').read_text().splitlines(True)
    expected = {x['path']: x for x in manifest['sources']}; seen = set(); index = 0
    derived = Path(derived); derived.mkdir()
    while index < len(lines):
        if not lines[index].startswith('--- a/'): raise RuntimeError('Unexpected source patch header')
        name = lines[index][6:].rstrip('\n'); index += 1
        if name not in expected or name in seen or lines[index] != '+++ b/'+name+'\n':
            raise RuntimeError('Unexpected or duplicate production source patch target')
        index += 1; original = Path(upstream)/name
        if _sha(original) != expected[name]['upstream_sha256']: raise RuntimeError('Upstream source changed before patching')
        old = original.read_text().splitlines(True); result = []; cursor = 0; new_cursor = 0
        while index < len(lines) and lines[index].startswith('@@ '):
            match = re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n', lines[index])
            if match is None: raise RuntimeError('Malformed unified source patch hunk')
            old_start, old_count, new_start, new_count = (int(match[1]), int(match[2] or 1), int(match[3]), int(match[4] or 1))
            offset = old_start-1 if old_start else 0
            if offset < cursor: raise RuntimeError('Overlapping source patch hunks')
            result.extend(old[cursor:offset]); new_cursor += offset-cursor; cursor = offset
            if new_cursor != (new_start-1 if new_start else 0): raise RuntimeError('Source patch output coordinate differs')
            index += 1; consumed = produced = 0
            while index < len(lines) and not lines[index].startswith(('@@ ', '--- a/')):
                line = lines[index]; index += 1
                if not line or line[0] not in ' +-': raise RuntimeError('Unsupported source patch operation')
                if line[0] in ' -':
                    if cursor >= len(old) or old[cursor] != line[1:]: raise RuntimeError('Exact source patch context differs')
                    cursor += 1; consumed += 1
                if line[0] in ' +': result.append(line[1:]); produced += 1; new_cursor += 1
            if (consumed, produced) != (old_count, new_count): raise RuntimeError('Source patch hunk extent differs')
        result.extend(old[cursor:]); target = derived/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(''.join(result))
        if _sha(target) != expected[name]['modified_sha256']: raise RuntimeError('Patched source SHA256 differs')
        seen.add(name)
    if seen != set(expected): raise RuntimeError('The source patch omitted a production class')


def _class_name(relative):
    return relative.as_posix()[:-6].replace('/', '.')


def _allowed_class(name):
    return any(name == prefix or name.startswith(prefix+'$') for prefix in (
        'org.broadinstitute.hellbender.utils.recalibration.BaseRecalibrationEngine',
        'org.broadinstitute.hellbender.utils.recalibration.ReadRecalibrationInfo', _MAIN))


def build_preprocessing_gatk_sourcepatch(conda_prefix, build_dir, *, upstream_source_dir=None, cpu_affinity=None,
                                        host_memory_limit_gib=4, host_memory_reserve_gib=4):
    """Compile a new source overlay using the same existing Conda JDK8/GATK.

    Inputs are an existing preprocessing Conda prefix and a new build directory
    outside that prefix/package. The result is a committed assets dictionary,
    also written to ``assets.private.json``. No read data or science runs here.
    """
    from .single_preprocessing import _Runner, _limits
    from ._preprocessing_io import atomic, bound, digest, identity
    prefix = Path(conda_prefix).resolve(strict=True); output = Path(build_dir).resolve()
    package, manifest_path, manifest = _manifest()
    if not (prefix/'conda-meta').is_dir(): raise ValueError('Use an existing Conda environment')
    if (output.exists() or output == prefix or prefix in output.parents or output in prefix.parents
            or output == package or package in output.parents):
        raise ValueError('Source-patch build directory must be new and outside the prefix and package')
    for value in (host_memory_limit_gib, host_memory_reserve_gib):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 1):
            raise ValueError('Build memory budget and reserve must be finite and at least 1 GiB')
    limits = _limits(); allowed = os.sched_getaffinity(0)
    cpus = list(cpu_affinity) if cpu_affinity is not None else sorted(allowed)[:1]
    if (not cpus or len(cpus) > 4 or len(set(cpus)) != len(cpus) or any(type(x) is not int for x in cpus)
            or not set(cpus) <= allowed or len(cpus) > limits['cpu_quota_count']
            or any(len(cpus) > quota for quota in limits['finite_cpu_quotas'])):
        raise ValueError('Use one to four distinct allowed CPUs within all actual quotas')
    required = (host_memory_limit_gib+host_memory_reserve_gib)*2**30
    if min(limits['available_bytes'], limits['admission_available_bytes']) < required:
        raise MemoryError('Source-patch build lacks actual/admission headroom and reserve')
    parent = output.parent
    while not parent.exists(): parent = parent.parent
    if shutil.disk_usage(parent).free < 256 << 20: raise OSError('Source-patch build requires 256 MiB free storage')
    tools = {name: _safe_path(prefix/'bin'/name) for name in ('java', 'javac', 'jar', 'python')}
    if any(not path.is_file() or not os.access(path, os.X_OK) for path in tools.values()): raise ValueError('Conda Java/Javac/Jar/Python are required')
    jars = list((prefix/'share').glob('*/gatk-package-4.2.0.0-local.jar'))
    if len(jars) != 1: raise ValueError('Exactly one installed GATK 4.2.0.0 local jar required')
    gatk = _safe_path(jars[0])
    if prefix not in gatk.parents: raise ValueError('The installed GATK jar must remain inside the Conda prefix')
    records = {p.name: digest(p) for p in sorted((prefix/'conda-meta').glob('*.json'))}
    bindings = {name: bound(path) for name, path in {**tools, 'gatkJar': gatk}.items()}
    cache = _safe_path(upstream_source_dir) if upstream_source_dir is not None else None
    if cache is not None and not cache.is_dir(): raise ValueError('upstream_source_dir must be a fixed original-source directory')
    cached_bindings = {} if cache is None else {x['path']: bound(cache/x['path']) for x in [*manifest['sources'], {'path': 'LICENSE.TXT'}]}
    builder_sha = digest(__file__); manifest_sha = digest(manifest_path)
    output.mkdir(parents=True, mode=0o700); (output/'tmp').mkdir()
    snapshot = output/'source/package'; snapshot.mkdir(parents=True)
    for name in ['source_manifest.json', *manifest['files']]:
        target = snapshot/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(package/name, target)
    worker = output/'source/builder.py'; shutil.copyfile(__file__, worker)
    config = dict(cpu_affinity=cpus, host_memory_limit_gib=host_memory_limit_gib, host_memory_reserve_gib=host_memory_reserve_gib,
                  builder_sha256=builder_sha, source_manifest_sha256=manifest_sha, tool_bindings=bindings, package_records=records, original_source_cache=cached_bindings)
    runner = _Runner(output, prefix, config, False); runner.env.update(CUDA_VISIBLE_DEVICES='', TMPDIR=str(output/'tmp'))
    for key in ('JAVA_TOOL_OPTIONS', '_JAVA_OPTIONS', 'JDK_JAVA_OPTIONS', 'CLASSPATH'): runner.env.pop(key, None)
    receipt = dict(schema=_BUILD_SCHEMA, status='building', scientific_validation_completed=False)
    try:
        runner.stage('java_versions', [[str(tools['java']), '-version'], [str(tools['javac']), '-version']], [], 1)
        for log in (output/'stages/java_versions').glob('*.log'):
            if re.search(r'(?:version\s+"|javac\s+)1\.8\.', log.read_text()) is None: raise ValueError('Source patch requires the fixed Conda JDK 8')
        runner.stage('gatk_version', [[str(tools['java']), '-Xmx512m', '-jar', str(gatk), '--version']], [], 1)
        if not _expected_gatk_version((output/'stages/gatk_version/0.log').read_text()): raise ValueError('Installed GATK version differs from 4.2.0.0')
        upstream = output/'source/upstream'; derived = output/'source/derived'
        download_command = [str(tools['python']), str(worker), '--worker-download', str(snapshot), str(upstream)]
        if cache is not None: download_command += ['--upstream-source-dir', str(cache)]
        runner.stage('download_sources', [download_command], [upstream/x['path'] for x in manifest['sources']]+[upstream/'LICENSE.TXT'], 1)
        runner.stage('patch_sources', [[str(tools['python']), str(worker), '--worker-patch', str(snapshot), str(upstream), str(derived)]], [derived/x['path'] for x in manifest['sources']], 1)
        classes = output/'classes'; classes.mkdir()
        launcher = snapshot/'assets/SourcePatchedBqsrMain.java'
        runner.stage('javac', [[str(tools['javac']), '-J-Xmx512m', '-proc:none', '-encoding', 'UTF-8', '-source', '8', '-target', '8', '-implicit:none', '-cp', str(gatk), '-sourcepath', str(derived), '-d', str(classes), *[str(derived/x['path']) for x in manifest['sources']], str(launcher)]], [], 1)
        class_hashes = {}
        for path in sorted(classes.rglob('*.class')):
            name = _class_name(path.relative_to(classes)); header = path.read_bytes()[:8]
            if not _allowed_class(name) or len(header) != 8 or header[:4] != b'\xca\xfe\xba\xbe' or struct.unpack('>H', header[6:])[0] != 52:
                raise RuntimeError('Unexpected compiler output or Java class version')
            class_hashes[name] = digest(path)
        if not _REQUIRED_CLASSES <= class_hashes.keys(): raise RuntimeError('Compiled source-patch classes are incomplete')
        licenses = classes/'META-INF'; licenses.mkdir()
        shutil.copyfile(upstream/'LICENSE.TXT', licenses/'LICENSE.GATK-4.2.0.0.txt')
        overlay = output/'fudan-gatk-bqsr-sourcepatch.jar'
        runner.stage('jar', [[str(tools['jar']), 'cf', str(overlay), '-C', str(classes), '.']], [overlay], 1)
        with zipfile.ZipFile(gatk) as archive:
            name = 'org/broadinstitute/hellbender/Main.class'
            if archive.getinfo(name).file_size > 1 << 20: raise RuntimeError('Original GATK Main class exceeds its binding allowance')
            main_sha = hashlib.sha256(archive.read(name)).hexdigest()
        class_binding = output/'class_binding.private.json'
        atomic(class_binding, dict(schema=_CLASS_SCHEMA, sourcePatchJar=str(overlay), gatkJar=str(gatk), classes=class_hashes, original_main_class_sha256=main_sha))
        probe = output/'class_probe.private.json'
        runner.stage('class_binding_probe', [[str(tools['java']), '-Xmx512m', '-Dfudan.gatkpatch.binding='+str(class_binding), '-Dfudan.gatkpatch.bindingSha256='+digest(class_binding), '-cp', str(overlay)+':'+str(gatk), _MAIN, 'probe', str(probe)]], [probe], 1)
        if json.loads(probe.read_text()).get('status') != 'classes_verified_before_training': raise RuntimeError('Actual loaded-class probe did not pass')
        if (builder_sha != digest(__file__) or manifest_sha != digest(manifest_path) or _manifest()[2] != manifest
                or records != {p.name: digest(p) for p in sorted((prefix/'conda-meta').glob('*.json'))}
                or any(bound(value['path']) != value for value in bindings.values())
                or any(bound(value['path']) != value for value in cached_bindings.values())):
            raise RuntimeError('Source, installed tools or Conda records changed during build')
        artifacts = dict(sourcePatchJar=str(overlay), gatkJar=str(gatk), java=str(tools['java']), javac=str(tools['javac']), jar=str(tools['jar']), python=str(tools['python']), classBinding=str(class_binding), classProbe=str(probe))
        core = dict(schema=_ASSET_SCHEMA, status=_STATUS, **artifacts, condaPrefix=str(prefix), assetsFile=str(output/'assets.private.json'), mainClass=_MAIN,
                    upstream_commit=_COMMIT, original_GATK_version=_VERSION, builder_sha256=builder_sha, source_manifest_sha256=manifest_sha,
                    environment_records=records, sha256={k: digest(v) for k, v in artifacts.items()}, identities={k: identity(v) for k, v in artifacts.items()},
                    original_source_cache=cached_bindings, scientific_validation_completed=False)
        receipt.update(status=_STATUS, artifacts=core, steps=runner.events, Conda_package_records_unchanged=True)
        atomic(output/'build_receipt.private.json', receipt)
        assets = dict(core, buildReceipt=str(output/'build_receipt.private.json'), build_receipt_sha256=digest(output/'build_receipt.private.json'))
        atomic(output/'assets.private.json', assets)
        return assets
    except BaseException as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error), steps=runner.events)
        atomic(output/'build_receipt.private.json', receipt)
        raise


def verify_gatk_source_patch_assets(assets, *, conda_prefix=None, gatk_jar=None):
    """Verify a committed dictionary or its assets JSON before constructing argv."""
    from ._preprocessing_io import bound, digest
    if isinstance(assets, (str, os.PathLike)):
        assets = json.loads(Path(assets).read_text())
    if not isinstance(assets, dict): raise ValueError('gatk_source_patch_assets must be a committed dictionary or assets JSON path')
    assets = copy.deepcopy(assets)
    if (assets.get('schema') != _ASSET_SCHEMA or assets.get('status') != _STATUS or assets.get('mainClass') != _MAIN
            or assets.get('upstream_commit') != _COMMIT or assets.get('original_GATK_version') != _VERSION):
        raise RuntimeError('Unrecognized or incomplete original GATK source-patch assets')
    receipt_path = _safe_path(assets['buildReceipt'])
    if digest(receipt_path) != assets.get('build_receipt_sha256'): raise RuntimeError('Source-patch build receipt SHA256 differs')
    receipt = json.loads(receipt_path.read_text()); core = {k: v for k, v in assets.items() if k not in ('buildReceipt', 'build_receipt_sha256')}
    if (receipt.get('schema') != _BUILD_SCHEMA or receipt.get('status') != _STATUS or receipt.get('artifacts') != core
            or receipt.get('Conda_package_records_unchanged') is not True):
        raise RuntimeError('Source-patch receipt does not commit these exact assets')
    package, manifest_path, _ = _manifest()
    if assets.get('builder_sha256') != digest(__file__) or assets.get('source_manifest_sha256') != digest(manifest_path):
        raise RuntimeError('Source-patch builder or packaged source binding changed')
    prefix = _safe_path(assets['condaPrefix'])
    if conda_prefix is not None and prefix != _safe_path(conda_prefix): raise RuntimeError('Source patch uses a different Conda prefix')
    if gatk_jar is not None and _safe_path(assets['gatkJar']) != _safe_path(gatk_jar): raise RuntimeError('Source patch uses a different installed GATK jar')
    records = {p.name: digest(p) for p in sorted((prefix/'conda-meta').glob('*.json'))}
    if not (prefix/'conda-meta').is_dir() or records != assets.get('environment_records'): raise RuntimeError('Source-patch Conda records changed')
    required = {'sourcePatchJar', 'gatkJar', 'java', 'javac', 'jar', 'python', 'classBinding', 'classProbe'}
    if set(assets.get('sha256', {})) != required or set(assets.get('identities', {})) != required: raise RuntimeError('Source-patch assets lack complete artifact bindings')
    for key in required:
        actual = bound(_safe_path(assets[key]))
        if actual['sha256'] != assets['sha256'][key] or any(actual.get(name) != value for name, value in assets['identities'][key].items()):
            raise RuntimeError('Source-patch artifact identity or SHA256 changed: '+key)
    binding = json.loads(Path(assets['classBinding']).read_text())
    if (binding.get('schema') != _CLASS_SCHEMA or binding.get('sourcePatchJar') != assets['sourcePatchJar'] or binding.get('gatkJar') != assets['gatkJar']
            or not isinstance(binding.get('classes'), dict) or not _REQUIRED_CLASSES <= binding['classes'].keys()
            or not all(_allowed_class(name) and re.fullmatch('[0-9a-f]{64}', sha) for name, sha in binding['classes'].items())
            or not re.fullmatch('[0-9a-f]{64}', binding.get('original_main_class_sha256', ''))):
        raise RuntimeError('Source-patch loaded-class binding differs')
    _verify_class_audit(assets, Path(assets['classProbe']), 'probe', binding)
    if json.loads(_safe_path(assets['assetsFile']).read_text()) != assets: raise RuntimeError('Source-patch asset file differs from its committed dictionary')
    return assets



def _verify_class_audit(assets, path, entrypoint, binding=None):
    """Audit proves actual loaded classes, never successful scientific completion."""
    if binding is None: binding = json.loads(Path(assets['classBinding']).read_text())
    audit = json.loads(Path(path).read_text())
    expected = {name: dict(code_source=assets['sourcePatchJar'], sha256=sha)
                for name, sha in binding['classes'].items()}
    if (audit.get('schema') != 'fudan-original-gatk-sourcepatch-runtime-v1'
            or audit.get('status') != 'classes_verified_before_training'
            or audit.get('entrypoint') != entrypoint
            or audit.get('binding_sha256') != assets['sha256']['classBinding']
            or audit.get('classes') != expected
            or audit.get('original_main_code_source') != assets['gatkJar']
            or audit.get('original_main_class_sha256') != binding['original_main_class_sha256']):
        raise RuntimeError('Source-patch actual class-load audit is incomplete or differs from its binding')
    return audit


def verify_gatk_source_patch_audit(assets, audit_json, *, entrypoint):
    """Check the scientific JVM audit after a committed training stage or resume."""
    if entrypoint not in ('original', 'parallel'): raise ValueError('Unknown BQSR training entrypoint')
    return _verify_class_audit(verify_gatk_source_patch_assets(assets), audit_json, entrypoint)


def gatk_source_patch_training_argv(assets, arguments, *, audit_json, entrypoint='original', adapter_jar=None,
                                    java_options=(), conda_prefix=None, gatk_jar=None):
    """Only BQSR training, with actual class verification in that same JVM.

    ``arguments`` are either the original ``BaseRecalibrator`` plus its options,
    or the unchanged parallel trainer options. ApplyBQSR, HC and QC are excluded.
    """
    if entrypoint not in ('original', 'parallel'): raise ValueError('Source patch is restricted to original/parallel BQSR training')
    if not isinstance(arguments, (tuple, list)) or any(not isinstance(x, (str, os.PathLike)) for x in arguments): raise ValueError('Explicit BQSR argument list required')
    if not isinstance(java_options, (tuple, list)) or any(not isinstance(x, str) or not x.startswith('-') or x in ('-cp', '-classpath', '--class-path', '-jar') or x.startswith(('--class-path=', '-Djava.class.path', '-Xbootclasspath', '-Dfudan.gatkpatch.', '-javaagent', '-agentlib', '-agentpath', '--module', '-m=')) for x in java_options):
        raise ValueError('Explicit JVM options must not replace the bound source-patch launcher')
    if entrypoint == 'original' and (not arguments or arguments[0] != 'BaseRecalibrator'): raise ValueError('Only BaseRecalibrator may use the original source patch')
    checked = verify_gatk_source_patch_assets(assets, conda_prefix=conda_prefix, gatk_jar=gatk_jar)
    classpath = [checked['sourcePatchJar']]
    if entrypoint == 'parallel':
        if adapter_jar is None: raise ValueError('Parallel BQSR requires its separately committed adapter jar')
        classpath.append(str(_safe_path(adapter_jar)))
    elif adapter_jar is not None: raise ValueError('Original BQSR must not load the parallel adapter')
    classpath.append(checked['gatkJar'])
    return [checked['java'], *_DEFAULTS, *java_options, '-Dfudan.gatkpatch.binding='+checked['classBinding'],
            '-Dfudan.gatkpatch.bindingSha256='+checked['sha256']['classBinding'], '-cp', ':'.join(classpath),
            _MAIN, entrypoint, str(audit_json), *map(str, arguments)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-download', nargs=2, metavar=('PACKAGE', 'OUTPUT'))
    parser.add_argument('--worker-patch', nargs=3, metavar=('PACKAGE', 'UPSTREAM', 'OUTPUT'))
    parser.add_argument('--upstream-source-dir'); parser.add_argument('--conda-prefix'); parser.add_argument('--build-dir'); parser.add_argument('--cpu-affinity')
    args = parser.parse_args(argv)
    if args.worker_download: return _download_sources(*args.worker_download, cache=args.upstream_source_dir)
    if args.worker_patch: return _apply_patch(*args.worker_patch)
    if not args.conda_prefix or not args.build_dir: parser.error('--conda-prefix and --build-dir are required')
    print(json.dumps(build_preprocessing_gatk_sourcepatch(args.conda_prefix, args.build_dir, upstream_source_dir=args.upstream_source_dir,
        cpu_affinity=[int(x) for x in args.cpu_affinity.split(',')] if args.cpu_affinity else None)))


if __name__ == '__main__': main()
