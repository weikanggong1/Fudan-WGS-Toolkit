"""Compile optional BWA-MEM2 with one existing, read-only Conda toolchain.

Official MIT-licensed archives are downloaded to a new build directory. The
Conda environment is never updated. Optional reference indexing is explicit;
this helper does not align reads, call variants or certify downstream parity.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import time
import urllib.request

from ._preprocessing_io import atomic, bound, digest, identity
from .phewas_resources import read_host_memory


SOURCE_COMMIT = '7aa5ff6c3330490e5629ab9b7327683d2dce02d6'
SUBMODULE_COMMIT = '245c4b8cff1d2e7338b7f3a82828fc8e72b29549'
ARTIFACTS = (
    ('bwa_mem2.tar.gz', 'https://codeload.github.com/bwa-mem2/bwa-mem2/tar.gz/'+SOURCE_COMMIT,
     222206, '7dc3f1e0c0ae448e87154000744c36126039a52c54d935412af54dc9f6f0cf58'),
    ('safestringlib.tar.gz', 'https://codeload.github.com/intel/safestringlib/tar.gz/'+SUBMODULE_COMMIT,
     78995, '6dea4f4accdc14eea706c88bb11725e8c2cd69b0d8c5f5ee19a0669d1883a901'),
)


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _extract(archive, destination):
    """Extract only ordinary archive files below one stripped top directory."""
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as source:
        for entry in source:
            path = PurePosixPath(entry.name)
            if path.is_absolute() or '..' in path.parts or not path.parts:
                raise ValueError('Unsafe source archive member')
            relative = Path(*path.parts[1:])
            if not relative.parts:
                continue
            target = destination/relative
            if entry.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif entry.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(entry) as stream, target.open('xb') as output:
                    shutil.copyfileobj(stream, output)
            else:
                raise ValueError('Source links or special files are not accepted')


def _cpu_constraints():
    """Resolve v1/v2 CPU mount namespaces and all visible finite ancestors."""
    memberships = []
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        if not controllers or 'cpu' in controllers.split(','):
            memberships.append((2 if not controllers else 1, relative))
    mounts = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, sep, right = line.partition(' - ')
        if not sep:
            continue
        a, b = left.split(), right.split()
        if len(a) < 6 or len(b) < 3:
            continue
        decode = lambda s: re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), s)
        if b[0] == 'cgroup2' or (b[0] == 'cgroup' and 'cpu' in (b[2]+','+a[5]).split(',')):
            mounts.append((2 if b[0] == 'cgroup2' else 1, Path(decode(a[3])), Path(decode(a[4]))))
    rows, seen = [], set()
    for version, relative in memberships:
        membership = Path(relative)
        for kind, root, mount in mounts:
            if kind != version:
                continue
            candidates = []
            try:
                candidates.append(mount/membership.relative_to(root))
            except ValueError:
                pass
            candidates.append(mount/relative.lstrip('/'))
            if relative == '/':
                candidates.append(mount)
            for leaf in candidates:
                if not leaf.is_dir():
                    continue
                current = leaf
                while current == mount or mount in current.parents:
                    key = (version, str(current))
                    if key not in seen:
                        seen.add(key)
                        quota, usage = None, None
                        if version == 2:
                            path = current/'cpu.max'
                            if path.exists():
                                q, period = path.read_text().split()
                                if q != 'max':
                                    quota = int(q)/int(period)
                            path = current/'cpu.stat'
                            if path.exists():
                                values = dict(x.split() for x in path.read_text().splitlines())
                                if 'usage_usec' in values:
                                    usage = int(values['usage_usec'])*1000
                        else:
                            path = current/'cpu.cfs_quota_us'
                            if path.exists():
                                q = int(path.read_text())
                                if q > 0:
                                    quota = q/int((current/'cpu.cfs_period_us').read_text())
                            path = current/'cpuacct.usage'
                            if path.exists():
                                usage = int(path.read_text())
                        if quota is not None:
                            rows.append(dict(key=key, quota_cores=quota, usage_ns=usage))
                    if current == mount:
                        break
                    current = current.parent
    return rows


def _admit(cpus, prospective_bytes, reserve_bytes, directory, disk_required):
    before = {r['key']: r for r in _cpu_constraints()}
    started = time.monotonic()
    time.sleep(0.25)
    after = _cpu_constraints()
    interval = time.monotonic()-started
    memory = read_host_memory()
    available = min(max(0, r['limit_bytes']-r['current_bytes']) for r in memory['constraints'])
    observed = []
    for row in after:
        old = before.get(row['key'])
        if old and old['usage_ns'] is not None and row['usage_ns'] is not None:
            if row['usage_ns'] >= old['usage_ns']:
                observed.append((row['usage_ns']-old['usage_ns'])/1e9/interval)
    quota = min((r['quota_cores'] for r in after), default=None)
    room_cpu = quota is None or (observed and max(observed)+len(cpus) <= quota)
    free = shutil.disk_usage(directory).free
    result = dict(admitted=bool(available >= prospective_bytes+reserve_bytes and room_cpu and free >= disk_required),
                  live_available_bytes=available, prospective_new_peak_bytes=prospective_bytes,
                  live_reserve_bytes=reserve_bytes, CPU_affinity=cpus,
                  finite_shared_CPU_quota_cores=quota,
                  observed_shared_CPU_cores=max(observed) if observed else None,
                  free_filesystem_bytes=free, prospective_disk_bytes=disk_required,
                  project_disk_quota='unknown', shared_not_exclusive=True,
                  memory_budget_is_admission_not_hard_quota=True)
    if not result['admitted']:
        raise RuntimeError('Build/index resource admission failed: '+json.dumps(result, sort_keys=True))
    return result


def build_preprocessing_aligner(conda_prefix, build_dir, reference=None, *, cpu_affinity=None):
    """Return optional BWA-MEM2 executable/index assets with bound provenance.

    ``conda_prefix`` supplies GCC/G++, binutils, make and zlib. ``build_dir``
    must be new and outside that prefix. ``reference`` is optional uncompressed
    FASTA: supplying it requests a new one-core index in the build directory,
    with 28 times the FASTA bytes plus a 50 GiB live host-memory reserve. Existing
    reference/index files are never replaced. One allowed CPU is used, chosen
    from ``cpu_affinity`` (exactly one ID) or the first currently allowed CPU.

    This creates a component candidate; validation of another binary does not
    automatically validate this build, alignments, VCF or PLINK.
    """
    prefix, output = Path(conda_prefix).resolve(strict=True), Path(build_dir).resolve()
    if not (prefix/'conda-meta').is_dir():
        raise ValueError('conda_prefix must be an existing Conda environment')
    if output.exists() or output == prefix or prefix in output.parents or output in prefix.parents:
        raise ValueError('build_dir must be new and outside the Conda prefix')
    cpus = list(cpu_affinity) if cpu_affinity is not None else sorted(os.sched_getaffinity(0))[:1]
    if (len(cpus) != 1 or type(cpus[0]) is not int or not set(cpus) <= os.sched_getaffinity(0)):
        raise ValueError('cpu_affinity must contain exactly one allowed CPU ID')
    flags = Path('/proc/cpuinfo').read_text()
    if 'avx512bw' in flags:
        arch, isa = 'avx512', '-mavx512bw'
    elif 'avx2' in flags:
        arch, isa = 'avx2', '-mavx2'
    else:
        raise RuntimeError('This BWA-MEM2 builder requires an AVX2 or AVX512BW CPU')
    tools = {name: prefix/'bin'/file for name, file in (
        ('CC','x86_64-conda-linux-gnu-gcc'), ('CXX','x86_64-conda-linux-gnu-g++'),
        ('AR','x86_64-conda-linux-gnu-ar'), ('MAKE','make'))}
    if any(not p.is_file() or not os.access(p, os.X_OK) for p in tools.values()):
        raise RuntimeError('Conda C/C++ compilers, binutils and make must exist in this prefix')
    ref = Path(reference).resolve(strict=True) if reference is not None else None
    if ref is not None:
        with ref.open('rb') as stream:
            if stream.read(1) != b'>':
                raise ValueError('reference must be an uncompressed FASTA; no implicit decompression')
    previous = os.umask(0o077)
    try:
        output.mkdir(parents=True)
    finally:
        os.umask(previous)
    env = dict(os.environ, CONDA_PREFIX=str(prefix), PATH=str(prefix/'bin')+':/usr/bin:/bin',
               LD_LIBRARY_PATH=str(prefix/'lib'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    env.pop('PYTHONPATH', None)
    records = {p.name: digest(p) for p in (prefix/'conda-meta').glob('*.json')}
    receipt = dict(schema='fudan-preprocessing-aligner-build-v1', status='building', started_utc=_utc(),
                   official_release_tag='v2.3', source_commit=SOURCE_COMMIT, submodule_commit=SUBMODULE_COMMIT,
                   license='MIT', prefix=str(prefix), CPU_affinity=cpus, target_architecture=arch,
                   builder_sha256=digest(__file__), Conda_package_records_sha256=records,
                   tool_sha256={k:digest(v) for k,v in tools.items()}, artifacts=[], steps=[],
                   GPU_used=False, mapping_or_variant_validation_assessed=False)
    def save():
        atomic(output/'build_receipt.private.json', receipt)
    def step(name, argv, cwd):
        started = time.monotonic()
        with (output/(name+'.stdout.log')).open('wb') as out, (output/(name+'.stderr.log')).open('wb') as err:
            process = subprocess.Popen(list(map(str,argv)), env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                                       stdout=out, stderr=err, preexec_fn=lambda: os.sched_setaffinity(0,cpus))
            code = process.wait()
        receipt['steps'].append(dict(name=name, exit_code=code, wall_seconds=time.monotonic()-started,
                                     stdout_sha256=digest(output/(name+'.stdout.log')),
                                     stderr_sha256=digest(output/(name+'.stderr.log'))))
        save()
        if code:
            raise RuntimeError('BWA-MEM2 build/index step failed: '+name)
        return (output/(name+'.stdout.log')).read_text()
    save()
    try:
        receipt['compile_admission'] = _admit(cpus,2*2**30,4*2**30,output,2*2**30)
        downloads=output/'downloads';downloads.mkdir()
        for name,url,size,expected in ARTIFACTS:
            path=downloads/name
            with urllib.request.urlopen(url,timeout=60) as source, path.open('xb') as target:
                copied=0
                for chunk in iter(lambda:source.read(1024*1024),b''):
                    copied += len(chunk)
                    if copied > size:
                        raise RuntimeError('Official archive exceeds its fixed size')
                    target.write(chunk)
            if path.stat().st_size != size or digest(path) != expected:
                raise RuntimeError('Official archive size or SHA-256 differs')
            receipt['artifacts'].append(dict(filename=name,url=url,bytes=size,sha256=expected))
        source=output/'source';_extract(downloads/'bwa_mem2.tar.gz',source)
        sub=source/'ext/safestringlib'
        if sub.exists():
            if any(sub.iterdir()):
                raise RuntimeError('Unexpected preexisting source submodule content')
            sub.rmdir()
        _extract(downloads/'safestringlib.tar.gz',sub)
        files={str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()}
        receipt['source_file_sha256']=files;save()
        step('Conda_CXX_version',[tools['CXX'],'--version'],source)
        step('compile',[tools['MAKE'],'-j1','arch='+arch,'EXE=bwa-mem2','CC='+str(tools['CC']),
             'CXX='+str(tools['CXX']),'AR='+str(tools['AR']),
             'CXXFLAGS=-g -O3 -fpermissive '+isa+' -I'+str(prefix/'include'),
             'LDFLAGS=-L'+str(prefix/'lib')+' -Wl,-rpath,'+str(prefix/'lib'),'all'],source)
        binary=source/'bwa-mem2';version=step('version',[binary,'version'],source).strip()
        ldd=shutil.which('ldd',path=env['PATH'])
        if ldd is None:
            raise RuntimeError('ldd is required to verify Conda runtime linkage')
        libraries={}
        for line in step('dynamic_dependencies',[ldd,binary],source).splitlines():
            if '=>' in line:
                name,tail=line.split('=>',1);location=tail.strip().split()[0]
                if location.startswith('/'):
                    libraries[name.strip()]=str(Path(location).resolve())
        for name in ('libz.so.1','libstdc++.so.6','libgcc_s.so.1'):
            if name not in libraries or prefix not in Path(libraries[name]).parents:
                raise RuntimeError('Library escaped the selected Conda prefix: '+name)
        if records != {p.name:digest(p) for p in (prefix/'conda-meta').glob('*.json')}:
            raise RuntimeError('Conda package records changed during build')
        if files != {name:digest(source/name) for name in files}:
            raise RuntimeError('Official source files changed during build')
        receipt.update(executable=str(binary), executable_sha256=digest(binary),
                       reported_cli_version=version, dynamic_dependencies=libraries,
                       Conda_package_records_unchanged=True, source_files_unchanged=True)
        if ref is not None:
            reference_before=bound(ref)
            prospective=28*reference_before['size']
            receipt['index_admission']=_admit(cpus,prospective,50*2**30,output,7*reference_before['size']+5*2**30)
            derived=output/'index';derived.mkdir();index_prefix=derived/'reference'
            step('index',[binary,'index','-p',index_prefix,ref],source)
            artifacts=[Path(str(index_prefix)+suffix) for suffix in ('.0123','.amb','.ann','.bwt.2bit.64','.pac')]
            if any(not p.is_file() or p.stat().st_size==0 for p in artifacts):
                raise RuntimeError('Incomplete BWA-MEM2 index artifacts')
            if identity(ref) != {k:v for k,v in reference_before.items() if k!='sha256'}:
                raise RuntimeError('Reference changed during indexing')
            # The BWA alt-contig sidecar is part of the alignment policy.
            alt=Path(str(ref)+'.alt')
            sidecar=None
            if alt.exists():
                sidecar=bound(alt);shutil.copyfile(alt,str(index_prefix)+'.alt')
                if digest(str(index_prefix)+'.alt') != sidecar['sha256']:
                    raise RuntimeError('ALT-contig sidecar changed during copy')
            receipt.update(index_prefix=str(index_prefix), reference_binding=reference_before,
                           reference_ALT_binding=sidecar,index_artifacts=[identity(p) for p in artifacts],
                           whole_index_SHA_recomputed=False)
        receipt.update(status='compiled_optional_aligner_not_mapping_validated',completed_utc=_utc())
        save()
    except BaseException as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error),failed_utc=_utc())
        save();raise
    return dict(status=receipt['status'],executable=receipt['executable'],executable_sha256=receipt['executable_sha256'],
                index_prefix=receipt.get('index_prefix'),reference_binding=receipt.get('reference_binding'),
                source_commit=SOURCE_COMMIT,reported_cli_version=receipt['reported_cli_version'],
                receipt=str(output/'build_receipt.private.json'),receipt_sha256=digest(output/'build_receipt.private.json'),
                mapping_or_variant_validation_assessed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conda-prefix',required=True);parser.add_argument('--build-dir',required=True)
    parser.add_argument('--reference');parser.add_argument('--cpu',type=int)
    args=parser.parse_args()
    print(json.dumps(build_preprocessing_aligner(args.conda_prefix,args.build_dir,args.reference,
                                               cpu_affinity=[args.cpu] if args.cpu is not None else None)))
