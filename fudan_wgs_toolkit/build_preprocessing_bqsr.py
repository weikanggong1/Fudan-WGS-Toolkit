"""Build a bounded parallel original-GATK BQSR adapter; no installation.

Workers own the original engine, reference reader and known-site readers.
Unfinalized native tables are combined before one original finalization and
quantization. All decoded records are also saved to a canonical temporary BAM.
The caller runs original ApplyBQSR afterwards and counts all temporary writes.
No GPU kernel, quality approximation, filter shortcut or model reuse is used.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil

_ENTRYPOINT='org.fnit.wgs.bqsr.parallel.ParallelFusedBqsrTrainer'
_ASSET_SCHEMA='fudan-bqsr-parallel-assets-v1'
_BUILD_SCHEMA='fudan-bqsr-parallel-build-v1'
_SOURCE_SCHEMA='fudan-bqsr-parallel-source-v1'
_FEATURE_LOOKAHEAD=1000


def build_preprocessing_bqsr(conda_prefix, build_dir, *, cpu_affinity=None,
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
    package=Path(__file__).resolve().parent/'_preprocessing_bqsr'
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
        raise MemoryError('BQSR adapter build is not admitted with its memory reserve')
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
    if (manifest.get('schema')!=_SOURCE_SCHEMA
            or manifest.get('feature_cache_lookahead_bases')!=_FEATURE_LOOKAHEAD):
        raise RuntimeError('Use the packaged parallel BQSR source manifest')
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
                raise ValueError('Invalid BQSR source manifest entry')
            original=package/path
            if original.is_symlink() or digest(original)!=expected:
                raise RuntimeError('Packaged BQSR adapter source differs from manifest')
            target=snapshot/path.relative_to('assets')
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(original,target);sources.append(target)
        runner.stage('javac',[[str(tools['javac']),'-J-Xmx512m','-proc:none','-source','8','-target','8',
                              '-cp',str(gatk),'-d',str(classes),*map(str,sources)]],[],1)
        adapter=output/'fnit-bqsr-parallel.jar'
        runner.stage('jar',[[str(tools['jar']),'cf',str(adapter),'-C',str(classes),'.']],[adapter],1)
        classpath=str(adapter)+':'+str(gatk)
        runner.stage('help',[[str(tools['java']),'-Xmx512m','-XX:ActiveProcessorCount='+str(len(cpus)),
                             '-cp',classpath,_ENTRYPOINT,'--help']],[],1)
        if records!={p.name:digest(p) for p in sorted((prefix/'conda-meta').glob('*.json'))}:
            raise RuntimeError('Conda package records changed during adapter build')
        artifacts=dict(adapterJar=str(adapter),gatkJar=str(gatk),java=str(tools['java']))
        result=dict(schema=_ASSET_SCHEMA,status='compiled_help_passed_not_science',
                    **artifacts,condaPrefix=str(prefix),classpath=classpath,
                    entrypoint=_ENTRYPOINT,feature_cache_lookahead_bases=_FEATURE_LOOKAHEAD,
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


def bqsr_training_argv(assets, *, input_cram, reference_fasta, known_sites,
                      recalibration_table, canonical_bam, audit_json, temporary_directory,
                      cpu_threads=3, heap_gib=4, replay_compression=0,
                      training_workers=2, queue_capacity=32,
                      gatk_source_patch_assets=None, source_patch_audit_json=None):
    """Return an argv for parallel training/replay; ApplyBQSR is separate.

    The original ordered known-sites list and full original CRAM must be used.
    Optional ``gatk_source_patch_assets`` validates a separate original-GATK
    source overlay and verifies its classes inside this same training JVM.
    ``source_patch_audit_json`` defaults beside ``audit_json``; the replay
    audit and loaded-class audit remain separate. No source patch is applied
    to the subsequent ApplyBQSR.
    No interval/subsampling/filter-changing arguments are accepted by this plan.
    Output paths must be new. The temporary BAM retains every decoded record,
    including records that the original training predicate excludes.
    """
    from ._preprocessing_io import identity, bound, digest
    if (type(cpu_threads) is not int or cpu_threads<1 or type(heap_gib) is not int
            or not 1<=heap_gib<=24 or type(replay_compression) is not int or not 0<=replay_compression<=9):
        raise ValueError('Invalid CPU, heap or temporary compression setting')
    if (type(training_workers) is not int or not 1<=training_workers<=16
            or training_workers>cpu_threads or type(queue_capacity) is not int
            or not 1<=queue_capacity<=64 or training_workers*queue_capacity>256):
        raise ValueError('Use 1..16 workers within cpu_threads, 1..64 queued reads each, and <=256 total queued reads')
    if isinstance(known_sites,(str,bytes)) or not known_sites:
        raise ValueError('Supply the original ordered known-site path sequence')
    if (assets.get('schema')!=_ASSET_SCHEMA
            or assets.get('entrypoint')!=_ENTRYPOINT
            or assets.get('feature_cache_lookahead_bases')!=_FEATURE_LOOKAHEAD):
        raise RuntimeError('BQSR assets do not implement the packaged parallel contract')
    if assets.get('status')!='compiled_help_passed_not_science':
        raise ValueError('Use a successfully committed BQSR adapter build')
    receipt_path=Path(assets.get('buildReceipt',''))
    if (not receipt_path.is_file() or digest(receipt_path)!=assets.get('build_receipt_sha256')):
        raise RuntimeError('BQSR build receipt is missing or its SHA-256 differs')
    receipt=json.loads(receipt_path.read_text())
    core={key:value for key,value in assets.items() if key not in ('buildReceipt','build_receipt_sha256')}
    if (receipt.get('schema')!=_BUILD_SCHEMA or receipt.get('status')!=assets['status'] or receipt.get('artifacts')!=core
            or receipt.get('Conda_package_records_unchanged') is not True):
        raise RuntimeError('BQSR build receipt does not commit these exact assets')
    manifest=Path(__file__).resolve().parent/'_preprocessing_bqsr/source_manifest.json'
    if (assets.get('source_manifest_sha256')!=digest(manifest)
            or assets.get('builder_sha256')!=digest(__file__)):
        raise RuntimeError('BQSR build source manifest differs from the packaged implementation')
    sources=json.loads(manifest.read_text())
    if (sources.get('schema')!=_SOURCE_SCHEMA
            or sources.get('feature_cache_lookahead_bases')!=_FEATURE_LOOKAHEAD):
        raise RuntimeError('BQSR packaged source schema or feature lookahead changed')
    for relative,expected in sources['files'].items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!='assets':
            raise RuntimeError('Invalid packaged BQSR source path')
        source=manifest.parent/path
        if not source.is_file() or source.is_symlink() or digest(source)!=expected:
            raise RuntimeError('Packaged BQSR adapter source differs from manifest')
    if (set(assets.get('sha256',{}))!={'adapterJar','gatkJar','java'}
            or set(assets.get('identities',{}))!={'adapterJar','gatkJar','java'}):
        raise RuntimeError('BQSR assets must bind the Java executable and both jars')
    if assets.get('classpath')!=assets['adapterJar']+':'+assets['gatkJar']:
        raise RuntimeError('BQSR classpath differs from its two bound jars')
    for key,value in assets['identities'].items():
        actual=bound(assets[key])
        if ({name:actual[name] for name in value}!=value or actual['sha256']!=assets['sha256'].get(key)):
            raise RuntimeError('BQSR artifact identity or SHA-256 changed: '+key)
    defaults=['-Dsamjdk.use_async_io_read_samtools=false','-Dsamjdk.use_async_io_write_samtools=true',
              '-Dsamjdk.use_async_io_write_tribble=false','-Dsamjdk.compression_level=2']
    argv=[assets['java'],*defaults,'-Xmx'+str(heap_gib)+'g',
          '-XX:ActiveProcessorCount='+str(cpu_threads),'-XX:ParallelGCThreads=1',
          '-Djava.io.tmpdir='+str(temporary_directory),'-cp',assets['classpath'],
          _ENTRYPOINT,'-R',str(reference_fasta),'-I',str(input_cram),
          '-O',str(recalibration_table),'--canonical-bam',str(canonical_bam),
          '--audit-json',str(audit_json),'--replay-compression',str(replay_compression),
          '--training-workers',str(training_workers),'--queue-capacity',str(queue_capacity)]
    for site in known_sites:argv+=['--known-sites',str(site)]
    if gatk_source_patch_assets is not None:
        from .build_preprocessing_gatk_sourcepatch import gatk_source_patch_training_argv
        cp_index=argv.index('-cp')
        patched_audit=(source_patch_audit_json if source_patch_audit_json is not None
                       else str(audit_json)+'.sourcepatch.json')
        return gatk_source_patch_training_argv(gatk_source_patch_assets,argv[cp_index+3:],
            audit_json=patched_audit,entrypoint='parallel',adapter_jar=assets['adapterJar'],
            java_options=argv[1+len(defaults):cp_index],
            conda_prefix=assets['condaPrefix'],gatk_jar=assets['gatkJar'])
    if source_patch_audit_json is not None:
        raise ValueError('source_patch_audit_json requires gatk_source_patch_assets')
    return argv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conda-prefix',required=True);parser.add_argument('--build-dir',required=True)
    parser.add_argument('--cpu-affinity',required=True)
    args=parser.parse_args()
    assets=build_preprocessing_bqsr(args.conda_prefix,args.build_dir,
        cpu_affinity=[int(item) for item in args.cpu_affinity.split(',')])
    print(json.dumps(assets))


if __name__=='__main__':main()
