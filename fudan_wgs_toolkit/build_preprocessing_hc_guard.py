"""Build the optional own traversal guard with existing original GATK/JDK8.

This builds no HC optimization or scientific overlay and reads no BAM/FASTA.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, zipfile
from pathlib import Path
_ENTRYPOINT='HCMappedStartGuard'
_ASSET_SCHEMA='fudan-hc-contig-guard-assets-v1'
_BUILD_SCHEMA='fudan-hc-contig-guard-build-v1'
_SOURCE_SCHEMA='fudan-hc-contig-guard-source-v1'

def build_preprocessing_hc_guard(conda_prefix, build_dir, *, cpu_affinity=None,
                              host_memory_limit_gib=4, host_memory_reserve_gib=4):
    """Compile only project adapter classes with the existing Conda Java/GATK.

    ``build_dir`` must be new and outside ``conda_prefix``. Resource admission,
    process-tree PSS, finite CPU quota and session cleanup use the pipeline's
    existing standard-library runner. No input BAM/CRAM is read by this function.
    """
    from .single_preprocessing import _Runner, _limits
    from ._preprocessing_io import atomic, bound, digest, identity

    prefix=Path(conda_prefix).resolve(strict=True)
    output=Path(build_dir).resolve()
    package=Path(__file__).resolve().parent/'_preprocessing_hc_guard'
    if not (prefix/'conda-meta').is_dir():
        raise ValueError('Use an existing Conda environment')
    if (output.exists() or output==prefix or prefix in output.parents
            or output in prefix.parents or output==package or package in output.parents):
        raise ValueError('build_dir must be new and outside the prefix and package')
    limits=_limits()
    cpus=list(cpu_affinity) if cpu_affinity is not None else sorted(os.sched_getaffinity(0))[:min(3,limits['cpu_quota_count'])]
    if (not cpus or len(cpus)>3 or len(set(cpus))!=len(cpus)
            or any(type(cpu) is not int for cpu in cpus) or not set(cpus)<=os.sched_getaffinity(0)
            or len(cpus)>limits['cpu_quota_count'] or any(len(cpus)>q for q in limits['finite_cpu_quotas'])):
        raise ValueError('Request one to three distinct allowed CPUs within the actual quota')
    if (not isinstance(host_memory_limit_gib,(int,float)) or isinstance(host_memory_limit_gib,bool)
            or not math.isfinite(host_memory_limit_gib) or not 1<=host_memory_limit_gib<=32
            or not isinstance(host_memory_reserve_gib,(int,float)) or isinstance(host_memory_reserve_gib,bool)
            or not math.isfinite(host_memory_reserve_gib) or host_memory_reserve_gib<1):
        raise ValueError('Use a finite build budget of 1..32 GiB and at least 1 GiB reserve')
    required=(host_memory_limit_gib+host_memory_reserve_gib)*2**30
    if limits['available_bytes']<required or limits['admission_available_bytes']<required:
        raise MemoryError('HC guard adapter build is not admitted with its memory reserve')
    parent=output.parent
    while not parent.exists():parent=parent.parent
    if shutil.disk_usage(parent).free<256*2**20:
        raise OSError('Adapter build requires 256 MiB free storage')
    tools={name:prefix/'bin'/name for name in ('java','javac','jar')}
    if any(not p.is_file() or not os.access(p,os.X_OK) for p in tools.values()):
        raise ValueError('The selected Conda environment must provide Java tools')
    jars=list((prefix/'share').glob('*/gatk-package-4.2.0.0-local.jar'))
    if len(jars)!=1:raise ValueError('Select the locked GATK 4.2.0.0 Conda environment')
    gatk=jars[0]
    source_manifest=package/'source_manifest.json'
    manifest=json.loads(source_manifest.read_text())
    if manifest.get('schema')!=_SOURCE_SCHEMA:
        raise RuntimeError('Use the packaged parallel HC guard source manifest')
    records={p.name:digest(p) for p in sorted((prefix/'conda-meta').glob('*.json'))}
    output.mkdir(parents=True,mode=0o700)
    snapshot=output/'source';classes=output/'classes'
    classes.mkdir();(output/'tmp').mkdir()
    config=dict(cpu_affinity=cpus,host_memory_limit_gib=host_memory_limit_gib,
                host_memory_reserve_gib=host_memory_reserve_gib,
                builder_sha256=digest(__file__),source_manifest_sha256=digest(source_manifest),
                gatk_jar=bound(gatk),package_records=records,sources=manifest['files'])
    runner=_Runner(output,prefix,config,False)
    runner.env.update(CUDA_VISIBLE_DEVICES='',TMPDIR=str(output/'tmp'))
    receipt=dict(status='building',schema=_BUILD_SCHEMA,real_data_started=False)
    try:
        sources=[]
        for relative,expected in manifest['files'].items():
            path=Path(relative)
            if path.is_absolute() or '..' in path.parts or path.parts[0]!='assets':
                raise ValueError('Invalid HC guard source manifest entry')
            original=package/path
            if original.is_symlink() or digest(original)!=expected:
                raise RuntimeError('Packaged HC guard adapter source differs from manifest')
            target=snapshot/path.relative_to('assets')
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(original,target);sources.append(target)
        runner.stage('javac',[[str(tools['javac']),'-J-Xmx512m','-proc:none','-source','8','-target','8',
                              '-cp',str(gatk),'-d',str(classes),*map(str,sources)]],[],1)
        adapter=output/'fudan-hc-guard.jar'
        runner.stage('jar',[[str(tools['jar']),'cf',str(adapter),'-C',str(classes),'.']],[adapter],1)
        classpath=str(adapter)+':'+str(gatk)
        probe=output/'probe.private.json'
        runner.stage('probe',[[str(tools['java']),'-Xmx512m','-XX:ActiveProcessorCount='+str(len(cpus)),
                             '-cp',classpath,'HcGuardProbe',str(gatk),digest(gatk),str(probe)]],[probe],1)
        probe_value=json.loads(probe.read_text())
        if (probe_value.get('status')!='original_GATK_guard_class_origin_version_probe_complete_not_science'
                or probe_value.get('original_GATK_version')!='4.2.0.0'
                or probe_value.get('original_gatk_jar_sha256')!=digest(gatk)
                or Path(probe_value.get('original_HCE_code_source','')).resolve()!=gatk.resolve()):
            raise RuntimeError('Original GATK guard build probe did not verify its source/version')
        import zipfile
        with zipfile.ZipFile(adapter) as built:
            class_sha={n:hashlib.sha256(built.read(n)).hexdigest() for n in built.namelist() if n.endswith('.class')}
        if set(class_sha)!={'HCMappedStartGuard.class','HCMappedStartGuard$Counts.class','HCMappedStartGuard$CountedSHA.class','HcGuardProbe.class'}:
            raise RuntimeError('Guard build omitted or added Java classes')
        if records!={p.name:digest(p) for p in sorted((prefix/'conda-meta').glob('*.json'))}:
            raise RuntimeError('Conda package records changed during adapter build')
        artifacts=dict(adapterJar=str(adapter),gatkJar=str(gatk),java=str(tools['java']),probe=str(probe))
        result=dict(schema=_ASSET_SCHEMA,status='compiled_guard_probe_passed_not_science',
                    **artifacts,condaPrefix=str(prefix),classpath=classpath,
                    entrypoint=_ENTRYPOINT,class_sha256=class_sha,
                    sha256={key:digest(path) for key,path in artifacts.items()},
                    identities={key:identity(path) for key,path in artifacts.items()},
                    builder_sha256=digest(__file__),source_manifest_sha256=digest(source_manifest),
                    original_GATK_version='4.2.0.0',scientific_validation_completed=False,
                    GPU_work_started=False)
        receipt.update(status=result['status'],artifacts=result,steps=runner.events,Conda_package_records_unchanged=True)
        atomic(output/'build_receipt.private.json',receipt)
        result=dict(result,buildReceipt=str(output/'build_receipt.private.json'),
                    build_receipt_sha256=digest(output/'build_receipt.private.json'))
        # Last, atomic commit marker; no assets are published after a failure.
        atomic(output/'assets.private.json',result)
        return result
    except BaseException as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error),steps=runner.events)
        atomic(output/'build_receipt.private.json',receipt)
        raise


def verify_hc_guard_assets(assets, *, conda_prefix=None, gatk_jar=None):
    """Reject missing/empty bindings; prove this exact packaged own guard build."""
    from ._preprocessing_io import bound, digest, identity
    if isinstance(assets,(str,os.PathLike)):
        assets_path=Path(assets).resolve(strict=True)
        if assets_path.stat().st_size>2**20:raise RuntimeError('Guard asset metadata is too large')
        assets=json.loads(assets_path.read_text())
    if type(assets) is not dict or assets.get('schema')!=_ASSET_SCHEMA or assets.get('status')!='compiled_guard_probe_passed_not_science' or assets.get('entrypoint')!=_ENTRYPOINT:
        raise RuntimeError('Guard assets have no complete committed build')
    manifest=Path(__file__).with_name('_preprocessing_hc_guard')/'source_manifest.json'
    if assets.get('builder_sha256')!=digest(__file__) or assets.get('source_manifest_sha256')!=digest(manifest):
        raise RuntimeError('Guard builder or packaged source manifest changed')
    source=json.loads(manifest.read_text())
    if source.get('schema')!=_SOURCE_SCHEMA or type(source.get('files')) is not dict or len(source['files'])!=2:
        raise RuntimeError('Guard source manifest incomplete')
    for rel,sha in source['files'].items():
        p=Path(rel)
        if p.is_absolute() or '..' in p.parts or p.parts[0]!='assets' or type(sha) is not str or len(sha)!=64 or digest(manifest.parent/p)!=sha:
            raise RuntimeError('Guard source does not match its exact manifest')
    receipt_path=Path(assets.get('buildReceipt',''))
    if not receipt_path.is_file() or digest(receipt_path)!=assets.get('build_receipt_sha256'):
        raise RuntimeError('Guard build receipt missing or SHA changed')
    receipt=json.loads(receipt_path.read_text());core={k:v for k,v in assets.items() if k not in ('buildReceipt','build_receipt_sha256')}
    if receipt.get('schema')!=_BUILD_SCHEMA or receipt.get('status')!=assets['status'] or receipt.get('artifacts')!=core or receipt.get('Conda_package_records_unchanged') is not True:
        raise RuntimeError('Guard receipt does not commit these exact assets')
    steps=receipt.get('steps')
    if type(steps) is not list or len(steps)!=3 or any(x.get('status')!='complete' or x.get('exit_codes')!=[0] for x in steps):
        raise RuntimeError('Guard compile/package/probe actual waits are incomplete')
    keys={'adapterJar','gatkJar','java','probe'}
    if set(assets.get('sha256',{}))!=keys or set(assets.get('identities',{}))!=keys:
        raise RuntimeError('Guard artifact bindings are incomplete')
    for key in keys:
        expected=assets['identities'][key]
        if type(expected) is not dict or set(expected)!={'path','device','inode','size','mtime_ns','ctime_ns'}:
            raise RuntimeError('Guard artifact identity is incomplete')
        actual=bound(assets.get(key,''))
        if actual!={**expected,'sha256':assets['sha256'][key]}:
            raise RuntimeError('Guard artifact identity or SHA changed')
    prefix=Path(assets.get('condaPrefix','')).resolve(strict=True)
    if conda_prefix is not None and prefix!=Path(conda_prefix).resolve(strict=True):raise RuntimeError('Guard Conda prefix differs')
    if Path(assets['java']).resolve()!=(prefix/'bin/java').resolve():raise RuntimeError('Guard Java is outside selected prefix')
    if gatk_jar is not None and Path(assets['gatkJar']).resolve()!=Path(gatk_jar).resolve(strict=True):raise RuntimeError('Guard original GATK jar differs')
    if assets.get('classpath')!=assets['adapterJar']+':'+assets['gatkJar'] or assets.get('original_GATK_version')!='4.2.0.0':raise RuntimeError('Guard classpath/version differs')
    with zipfile.ZipFile(assets['adapterJar']) as archive:
        classes={n:hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist() if n.endswith('.class')}
    if set(classes)!={'HCMappedStartGuard.class','HCMappedStartGuard$Counts.class','HCMappedStartGuard$CountedSHA.class','HcGuardProbe.class'} or classes!=assets.get('class_sha256'):
        raise RuntimeError('Guard own complete class closure differs')
    return assets


def guard_scan_argv(assets, *, bam, fai, hc_command_json, output_json, conda_prefix=None, gatk_jar=None):
    assets=verify_hc_guard_assets(assets,conda_prefix=conda_prefix,gatk_jar=gatk_jar)
    return [assets['java'],'-Xmx1g','-XX:ActiveProcessorCount=1','-XX:ParallelGCThreads=1','-cp',assets['classpath'],_ENTRYPOINT,
            str(bam),str(fai),assets['gatkJar'],assets['sha256']['gatkJar'],str(hc_command_json),str(output_json)]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conda-prefix',required=True);parser.add_argument('--build-dir',required=True)
    parser.add_argument('--cpu-affinity',type=int,nargs='+');parser.add_argument('--host-memory-limit-gib',type=float,default=4)
    parser.add_argument('--host-memory-reserve-gib',type=float,default=4)
    result=build_preprocessing_hc_guard(**vars(parser.parse_args(argv)))
    print(json.dumps({'status':result['status'],'scientific_validation_completed':False}))

if __name__=='__main__':main()
