"""One subject's paired FASTQ -> GVCF/VCF -> verified autosomal PLINK.

Conda supplies the established open-source callers. The optional project CUDA
PairHMM is experimental; real sparse workloads currently favor original GKL.
All commands and evidence are local and may contain sensitive identifiers.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import time

from ._preprocessing_io import atomic, bound, digest, identity, vcf_oracle, verify_and_split_plink


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _limits():
    """Read physical memory and every finite ancestor cgroup constraint."""
    from .phewas_resources import _memory_directories, _ancestors, _memory_constraint, _unescape
    import posixpath
    info={line.split(':',1)[0]:int(line.split()[1])*1024
          for line in Path('/proc/meminfo').read_text().splitlines()
          if line.startswith(('MemTotal:','MemAvailable:'))}
    constraints=[]; seen=set()
    for version,leaf,mount in _memory_directories():
        for directory in _ancestors(leaf,mount):
            if (version,str(directory)) in seen:continue
            seen.add((version,str(directory)))
            item=_memory_constraint(version,directory)
            if item:
                event=directory/('memory.events' if version==2 else 'memory.failcnt')
                item.update(events=event.read_text().strip() if event.is_file() else None)
                constraints.append(item)
    available=min([info['MemAvailable']]+[max(0,x['limit_bytes']-x['current_bytes']) for x in constraints])
    # Reuse the established host policy for reclaimable inactive file pages.
    # Keep actual usage/headroom separate; no cache is cleared or counted as
    # already released, and no estimate is credited against task-owned PSS.
    admission_available=min([info['MemAvailable']]+[max(0,x['limit_bytes']-x['effective_current_bytes']) for x in constraints])
    cpu_count=len(os.sched_getaffinity(0)); quotas=[]
    memberships=[]
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _,controllers,relative=line.split(':',2)
        if not controllers:memberships.append((2,relative))
        elif 'cpu' in controllers.split(','):memberships.append((1,relative))
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left,sep,right=line.partition(' - ')
        a,b=left.split(),right.split()
        if not sep or len(a)<6 or len(b)<3:continue
        version=2 if b[0]=='cgroup2' else 1 if b[0]=='cgroup' and 'cpu' in ','.join((b[1],b[2],a[5])).split(',') else None
        if version is None:continue
        mount,root=Path(_unescape(a[4])),posixpath.normpath(_unescape(a[3]))
        for kind,group in memberships:
            if kind!=version:continue
            relative=group[len(root):].lstrip('/') if root!='/' and (group==root or group.startswith(root+'/')) else group.lstrip('/')
            leaf=mount/relative
            if not leaf.exists():leaf=mount
            for directory in _ancestors(leaf,mount):
                if version==2 and (directory/'cpu.max').is_file():
                    quota,period=(directory/'cpu.max').read_text().split()
                    if quota=='max':continue
                    quota,period=int(quota),int(period)
                elif version==1 and (directory/'cpu.cfs_quota_us').is_file():
                    quota=int((directory/'cpu.cfs_quota_us').read_text());period=int((directory/'cpu.cfs_period_us').read_text())
                    if quota<0:continue
                else:continue
                if quota<=0 or period<=0:raise RuntimeError('Invalid finite CPU quota')
                quotas.append(quota/period);cpu_count=min(cpu_count,max(1,math.floor(quota/period)))
    return dict(available_bytes=available,admission_available_bytes=admission_available,
                reclaimable_file_cache_is_an_estimate=True,physical_available_bytes=info['MemAvailable'],
                finite_cgroups=constraints,cpu_quota_count=cpu_count,finite_cpu_quotas=quotas)


def _descendants(root):
    from .phewas_resources import _verified_children
    result, pending = set(), [root]
    while pending:
        pid = pending.pop()
        if pid in result:
            continue
        result.add(pid)
        pending.extend(_verified_children(pid,strict=True))
    return result


def _pss(pids, *, diagnostics=None):
    from .phewas_resources import _process_memory_counters
    # A single sample has a bounded settling window. Old kernels may stream
    # inconsistent smaps during exec/mmap; no partial endpoint contributes to
    # the sample, and a persistent unknown still stops the stage.
    began=time.monotonic();deadline=began+2.0
    result=0; verified=0
    for pid in sorted(pids):
        identity_ticks=None;attempt=0
        while True:
            attempt+=1
            try:
                before=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
            except (FileNotFoundError,ProcessLookupError):break
            if before[0] in ('Z','X'):break
            if identity_ticks is None:identity_ticks=before[19]
            if before[19]!=identity_ticks:
                raise RuntimeError('Process identity changed during task PSS measurement')
            measured=_process_memory_counters(pid,include_pss=True)
            try:
                after=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
            except (FileNotFoundError,ProcessLookupError):break
            if after[0] in ('Z','X'):break
            if after[19]!=identity_ticks:
                raise RuntimeError('Process identity changed during task PSS measurement')
            if measured['pss_coverage_complete'] and measured['pss_bytes'] is not None:
                result+=measured['pss_bytes'];verified+=1
                if attempt>1 and diagnostics is not None:
                    diagnostics['recovered_endpoints']=diagnostics.get('recovered_endpoints',0)+1
                break
            # An exec/map change can make an old kernel's streamed smaps
            # temporarily inconsistent. Discard every partial total, retry
            # the entire endpoint with the same identity, and remain closed
            # on persistent unknowns rather than substituting RSS or zero.
            if diagnostics is not None:
                endpoint=dict(pid=pid,start_ticks=identity_ticks,attempt=attempt,
                    state_before=before[0],state_after=after[0],
                    elapsed_seconds=time.monotonic()-began,
                    read_errors=measured.get('pss_read_errors',[]))
                failures=diagnostics.setdefault('endpoint_failures',[])
                if len(failures)<16:
                    failures.append(endpoint)
                diagnostics['last_incomplete_endpoint']=endpoint
                diagnostics['incomplete_endpoint_count']=diagnostics.get('incomplete_endpoint_count',0)+1
            remaining=deadline-time.monotonic()
            if remaining<=0 or attempt>=64:
                if diagnostics is not None:
                    diagnostics['retry_deadline_seconds']=2.0
                    diagnostics['failed_sample_elapsed_seconds']=time.monotonic()-began
                raise RuntimeError('PSS is unavailable for a live task process; memory budget cannot be verified')
            if diagnostics is not None:
                diagnostics['transient_endpoint_retries']=diagnostics.get('transient_endpoint_retries',0)+1
            time.sleep(min(.01*2**min(attempt-1,5),.2,remaining))
    return result if verified else None


def _group_members(group):
    """A stage starts its own session; only that session may be cleaned up."""
    result=[]
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=path.read_text().rsplit(')',1)[1].split()
            if fields[0] not in ('Z','X') and int(fields[3])==group:
                result.append(int(path.parent.name))
        except (OSError,ValueError,IndexError):pass
    return result


def _cleanup_stage(process):
    """Drain the stage's own session, including children of an exited root."""
    deadline=time.monotonic()+5
    while True:
        members=_group_members(process.pid)
        if not members:break
        sig=signal.SIGKILL if time.monotonic()>=deadline else signal.SIGTERM
        for pid in members:
            try:os.kill(pid,sig)
            except ProcessLookupError:pass
        if sig==signal.SIGKILL:
            process.wait(timeout=5)
            until=time.monotonic()+1
            while _group_members(process.pid) and time.monotonic()<until:time.sleep(.05)
            if _group_members(process.pid):
                raise RuntimeError('Stage session did not drain after cleanup')
            return
        time.sleep(.05)
    process.wait(timeout=5)


class _Runner:
    def __init__(self, root, prefix, config, resume):
        self.root, self.prefix, self.config = root, prefix, config
        self.env=dict(os.environ, PATH=str(prefix/'bin')+':/usr/bin:/bin', JAVA_HOME=str(prefix),
                      LD_LIBRARY_PATH=str(prefix/'lib'), OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', LC_ALL='C')
        self.env.pop('PYTHONPATH',None)
        self.cpuset=set(config['cpu_affinity'])
        self.budget=int(config['host_memory_limit_gib']*2**30)
        self.reserve=int(config['host_memory_reserve_gib']*2**30)
        self.events=[]
        plan=root/'plan.private.json'
        if plan.exists():
            previous=json.loads(plan.read_text())
            if not resume or previous != config:
                raise ValueError('Existing output requires resume=True with identical source, tools, inputs and options')
        else:
            if resume:
                raise ValueError('No bound plan exists to resume')
            atomic(plan, config)

    def tool(self,name):
        return str(self.prefix/'bin'/name)

    def gatk(self,name,*args,heap_gib=4):
        cpus=len(self.cpuset)
        return [self.tool('gatk'),'--java-options',
                f'-Xmx{heap_gib}g -XX:ActiveProcessorCount={cpus} -XX:ParallelGCThreads=1 -Djava.io.tmpdir={self.root}/tmp',
                name,*map(str,args)]

    def stage(self,name,commands,outputs,future_gib=1):
        directory=self.root/'stages'/name
        receipt=directory/'complete.private.json'
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            if saved['commands'] != commands:
                raise ValueError('Resume command changed')
            if any(bound(x['path']) != x for x in saved['outputs']):
                raise ValueError('Committed output changed')
            self.events.append(saved)
            return saved
        if directory.exists():
            raise RuntimeError('Incomplete stage preserved; use a new output directory: '+name)
        limits=_limits()
        if future_gib*2**30 > self.budget or limits['admission_available_bytes'] < future_gib*2**30+self.reserve:
            raise MemoryError('Stage not admitted within task budget and actual finite ancestor/physical headroom: '+name)
        directory.mkdir(parents=True)
        began=time.monotonic()
        children_before=resource.getrusage(resource.RUSAGE_CHILDREN)
        controller_before=resource.getrusage(resource.RUSAGE_SELF)
        record=dict(stage=name,started_utc=_utc(),commands=commands,admission=limits,
                    future_peak_gib=future_gib,status='running')
        atomic(directory/'started.private.json',record)
        maximum=None; exits=[]; pss_diagnostics={}
        process=None
        def setup():
            os.sched_setaffinity(0,self.cpuset)
            # RLIMIT_AS is unsuitable for JVM/CUDA; task PSS is monitored instead.
            resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        try:
            for number,command in enumerate(commands):
                with (directory/(str(number)+'.log')).open('wb') as stream:
                    process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                        env=self.env,start_new_session=True,preexec_fn=setup,cwd=self.root)
                    while process.poll() is None:
                        used=_pss(_descendants(process.pid)|{os.getpid()},diagnostics=pss_diagnostics)
                        if used is not None:
                            maximum=max(maximum or 0,used)
                            if used>self.budget:
                                raise MemoryError('Task-owned process PSS exceeded configured shared stage budget')
                        time.sleep(.25)
                    exits.append(process.wait())
                    if exits[-1] != 0:
                        raise RuntimeError('Command exited nonzero; see private stage log: '+name)
                    if _group_members(process.pid):
                        raise RuntimeError('Stage root exited before all session children drained')
                    process=None
            command_wall=time.monotonic()-began
            hashing_started=time.monotonic()
            checked=[bound(p) for p in outputs]
            hashing_wall=time.monotonic()-hashing_started
            if any(x['size']==0 for x in checked):
                raise RuntimeError('Stage produced an empty required output: '+name)
            children_after=resource.getrusage(resource.RUSAGE_CHILDREN)
            controller_after=resource.getrusage(resource.RUSAGE_SELF)
            record.update(status='complete',exit_codes=exits,outputs=checked,
                          wall_seconds=time.monotonic()-began,command_and_resource_observer_seconds=command_wall,
                          artifact_hashing_seconds=hashing_wall,peak_sampled_pss_bytes=maximum,
                          reaped_child_cpu_user_seconds=children_after.ru_utime-children_before.ru_utime,
                          reaped_child_cpu_system_seconds=children_after.ru_stime-children_before.ru_stime,
                          controller_cpu_user_seconds=controller_after.ru_utime-controller_before.ru_utime,
                          controller_cpu_system_seconds=controller_after.ru_stime-controller_before.ru_stime,
                          child_CPU_includes_reaped_descendants_once=True,
                          pss_diagnostics=pss_diagnostics,
                          pss_is_sampled_not_a_cgroup_hard_limit=True,
                          completed_utc=_utc(),resource_snapshot_after=_limits())
            atomic(receipt,record)
            with (self.root/'journal.private.jsonl').open('a') as out:
                out.write(json.dumps(record)+'\n');out.flush();os.fsync(out.fileno())
            self.events.append(record)
            return record
        except BaseException as error:
            if process is not None:
                try:_cleanup_stage(process)
                except BaseException as cleanup_error:
                    record['cleanup_error_type']=type(cleanup_error).__name__
                    record['cleanup_error']=str(cleanup_error)
            record.update(status='failed',error_type=type(error).__name__,error=str(error),
                          exit_codes=exits,wall_seconds=time.monotonic()-began,
                          peak_sampled_pss_bytes=maximum,pss_diagnostics=pss_diagnostics,
                          failed_utc=_utc())
            atomic(directory/'failed.private.json',record)
            raise


def _gpu_evidence(path,uuid,budget):
    from ._preprocessing_gpu.evidence import gpu_evidence
    return gpu_evidence(path,uuid,budget)


def single_subj_preprocessing(read1, read2, reference_fasta, known_sites, output_dir, *,
                             sample_id, conda_prefix=None, cpu_threads=None, cpu_affinity=None,
                             fastp_threads=1,
                             host_memory_limit_gib=None, host_memory_reserve_gib=4,
                             aligner='bwa', aligner_executable=None, index_prefix=None,
                             pairhmm_backend='cpu', gpu_uuid=None, gpu_memory_limit_gib=2, gpu_build_dir=None,
                             intervals=None, qc_mode='shared', bqsr_input='cram',
                             qc_exit_fix=True, runtime_build_dir=None,
                             bqsr_mode='auto', bqsr_workers=None, bqsr_build_dir=None,
                             resume=False):
    """Process one real paired-end subject with established caller semantics.

    ``known_sites`` is a sequence of exactly three reference-matched indexed VCFs.
    Defaults cover the complete reference for calling and autosomes for PLINK.
    ``intervals`` explicitly selects a small calling/QC benchmark; it is never
    described as a complete genome. All paths and command logs remain private.
    CUDA replaces PairHMM only; no GPU speedup is assumed or silently reported.
    """
    started=time.monotonic()
    if type(qc_exit_fix) is not bool:
        raise ValueError('qc_exit_fix must be a boolean')
    if bqsr_mode not in ('auto','original','parallel'):
        raise ValueError('bqsr_mode must be auto, original or parallel')
    if bqsr_workers is not None and (type(bqsr_workers) is not int or not 1<=bqsr_workers<=16):
        raise ValueError('bqsr_workers must be None or an integer of 1..16')
    if not isinstance(sample_id,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',sample_id) or sample_id=='0':
        raise ValueError('sample_id must be a nonzero whitespace-free identifier')
    if not isinstance(known_sites,(list,tuple)) or len(known_sites)!=3:
        raise ValueError('Supply exactly three explicit known-site VCF paths')
    if aligner not in ('bwa','bwa-mem2') or pairhmm_backend not in ('cpu','cuda') or qc_mode not in ('shared','separate') or bqsr_input not in ('cram','bam'):
        raise ValueError('Unsupported aligner, PairHMM, QC or BQSR mode')
    if bqsr_input=='bam':
        raise ValueError('BAM BQSR mode is not yet released; use original CRAM semantics')
    selected_intervals=[] if intervals is None else list(intervals)
    if any(not isinstance(x,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+:[1-9][0-9]*-[1-9][0-9]*',x) for x in selected_intervals):
        raise ValueError('intervals must contain explicit contig:start-end strings')
    root=Path(output_dir).resolve()
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    if root.stat().st_uid!=os.getuid() or root.stat().st_mode & 0o077:
        raise ValueError('output_dir must be owned by the current user and private (mode 0700)')
    saved_affinity=set(os.sched_getaffinity(0))
    lock=(root/'run.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();raise RuntimeError('Another controller owns this output directory') from None
    try:
        prefix=Path(conda_prefix or os.environ.get('CONDA_PREFIX','')).resolve()
        tools=('fastp','bwa','samtools','sambamba','gatk','bcftools','plink2','java','gzip')
        if cpu_threads is not None and (type(cpu_threads) is not int or cpu_threads<2):
            raise ValueError('cpu_threads must be an integer of at least two')
        if any(not (prefix/'bin'/name).is_file() for name in tools):
            raise ValueError('Use the documented preprocessing Conda environment')
        allowed=sorted(os.sched_getaffinity(0))
        cpus=allowed if cpu_affinity is None else list(cpu_affinity)
        if len(set(cpus))!=len(cpus):raise ValueError('CPU affinity must contain distinct IDs')
        limits=_limits()
        threads=min(len(cpus),limits['cpu_quota_count'],cpu_threads or limits['cpu_quota_count'])
        if type(threads) is not int or threads<2 or not cpus or any(type(x) is not int or x not in allowed for x in cpus):
            raise ValueError('At least two actually available CPU cores are required')
        cpus=sorted(set(cpus))[:threads]
        selected_bqsr_mode=('parallel' if threads>=4 else 'original') if bqsr_mode=='auto' else bqsr_mode
        training_workers=(bqsr_workers or min(16,max(1,threads-2))) if selected_bqsr_mode=='parallel' else None
        if bqsr_workers is not None and selected_bqsr_mode!='parallel':
            raise ValueError('bqsr_workers requires the parallel BQSR mode')
        if training_workers is not None and training_workers>max(1,threads-2):
            raise ValueError('BQSR workers must leave the producer and writer within the CPU budget')
        cleaning_threads=min(6,threads) if fastp_threads is None else fastp_threads
        if type(cleaning_threads) is not int or not 1<=cleaning_threads<=min(16,threads):
            raise ValueError('fastp_threads must be 1..min(16, CPU budget); None explicitly enables parallel cleaning')
        os.sched_setaffinity(0,set(cpus))
        budget=host_memory_limit_gib if host_memory_limit_gib is not None else min(64,limits['admission_available_bytes']/2**30-host_memory_reserve_gib)
        if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>0 for x in (budget,host_memory_reserve_gib,gpu_memory_limit_gib)) or budget<12:
            raise ValueError('Host budget must be at least 12 GiB; all budgets/reserves must be finite positive numbers')
        reads=[str(Path(x).resolve(strict=True)) for x in (read1,read2)]
        if reads[0]==reads[1]:raise ValueError('R1 and R2 must be different files')
        reference=Path(reference_fasta).resolve(strict=True)
        if reference.name.endswith(('.gz','.bgz')):
            raise ValueError('This entry requires an uncompressed reference FASTA with matching indices')
        sites=[Path(x).resolve(strict=True) for x in known_sites]
        if len(set(sites))!=3:raise ValueError('Supply three different known-site VCF files')
        reference_index=[Path(str(reference)+'.fai'),reference.with_suffix('.dict')]
        align=Path(aligner_executable).resolve(strict=True) if aligner_executable else prefix/'bin'/aligner
        idx=Path(index_prefix).resolve() if index_prefix else reference
        indices=[Path(str(idx)+s) for s in (('.amb','.ann','.bwt','.pac','.sa') if aligner=='bwa' else ('.0123','.amb','.ann','.bwt.2bit.64','.pac'))]
        inputs={str(p):identity(p) for p in [*map(Path,reads),reference,*reference_index,*sites,*[Path(str(x)+'.tbi') for x in sites],*indices]}
        ref_alt,idx_alt=Path(str(reference)+'.alt'),Path(str(idx)+'.alt')
        if ref_alt.exists()!=idx_alt.exists() or (ref_alt.exists() and digest(ref_alt)!=digest(idx_alt)):
            raise ValueError('ALT sidecar presence/content must match the reference and selected aligner index')
        for sidecar in {ref_alt,idx_alt}:
            if sidecar.exists():inputs[str(sidecar)]=identity(sidecar)
        binaries={name:bound(prefix/'bin'/name) for name in tools}
        binaries['aligner']=bound(align)
        if aligner=='bwa-mem2':
            for binary in sorted(align.parent.glob(align.name+'.*')):
                if binary.is_file() and os.access(binary,os.X_OK):binaries[binary.name]=bound(binary)
        jars=list((prefix/'share').glob('*/gatk-package-4.2.0.0-local.jar'))
        if len(jars)!=1:raise ValueError('The tested GATK 4.2.0.0 local jar is required')
        binaries['gatk_jar']=bound(jars[0])
        source_files=[Path(__file__),Path(__file__).with_name('_preprocessing_io.py'),Path(__file__).with_name('phewas_resources.py')]
        bqsr_assets=None
        if selected_bqsr_mode=='parallel':
            from .build_preprocessing_bqsr import build_preprocessing_bqsr, bqsr_training_argv
            bqsr_helper=Path(__file__).with_name('build_preprocessing_bqsr.py')
            bqsr_package=Path(__file__).with_name('_preprocessing_bqsr')
            bqsr_manifest=bqsr_package/'source_manifest.json'
            build=Path(bqsr_build_dir).resolve() if bqsr_build_dir else root/'bqsr_build'
            asset_file=build/'assets.private.json'
            if resume and not asset_file.is_file():
                raise ValueError('Resume requires the original committed BQSR build')
            bqsr_assets=json.loads(asset_file.read_text()) if asset_file.is_file() else build_preprocessing_bqsr(
                str(prefix),str(build),cpu_affinity=cpus[:1],
                host_memory_limit_gib=4,host_memory_reserve_gib=host_memory_reserve_gib)
            if (bqsr_assets.get('condaPrefix')!=str(prefix)
                    or Path(bqsr_assets.get('java','')).resolve()!=(prefix/'bin'/'java').resolve()
                    or Path(bqsr_assets.get('gatkJar','')).resolve()!=jars[0].resolve()):
                raise ValueError('BQSR assets must use the same Conda prefix')
            # The command builder verifies the committed receipt, packaged
            # sources, executable, jars and exact classpath before any input.
            bqsr_training_argv(bqsr_assets,input_cram=root/'files'/'marked.cram',reference_fasta=reference,
                known_sites=sites,recalibration_table=root/'files'/'recal.table',
                canonical_bam=root/'files'/'canonical_bqsr.bam',audit_json=root/'bqsr_training.private.json',
                temporary_directory=root/'tmp',cpu_threads=threads,heap_gib=4,
                training_workers=training_workers,queue_capacity=min(32,256//training_workers))
            binaries['bqsr_build_receipt']=bound(bqsr_assets['buildReceipt'])
            binaries['bqsr_asset_record']=bound(asset_file)
            for key in bqsr_assets['sha256']:
                binaries['bqsr_'+key]=bound(bqsr_assets[key])
            manifest=json.loads(bqsr_manifest.read_text())
            source_files += [bqsr_helper,bqsr_manifest,*[bqsr_package/path for path in manifest['files']]]
        qc_assets=None
        if qc_exit_fix:
            from ._preprocessing_runtime import build_preprocessing_runtime, qc_argv
            runtime_helper=Path(__file__).with_name('_preprocessing_runtime.py')
            runtime_source=Path(__file__).with_name('_preprocessing_runtime_assets')/'QcMain.java'
            build=Path(runtime_build_dir).resolve() if runtime_build_dir else root/'runtime_build'
            asset_file=build/'assets.private.json'
            if resume and not asset_file.is_file():
                raise ValueError('Resume requires the original committed QC runtime build')
            qc_assets=json.loads(asset_file.read_text()) if asset_file.is_file() else build_preprocessing_runtime(
                str(prefix),str(build),cpu_affinity=cpus[:1],
                host_memory_limit_gib=4,host_memory_reserve_gib=host_memory_reserve_gib)
            if (qc_assets.get('condaPrefix')!=str(prefix)
                    or qc_assets.get('source_sha256')!=digest(runtime_source)
                    or qc_assets.get('builder_sha256')!=digest(runtime_helper)
                    or Path(qc_assets.get('java','')).resolve()!=(prefix/'bin'/'java').resolve()
                    or Path(qc_assets.get('gatkJar','')).resolve()!=jars[0].resolve()):
                raise ValueError('QC runtime build is bound to different Conda or packaged sources')
            # Check committed-build integrity before admitting any scientific stage.
            qc_argv(qc_assets,'CollectMultipleMetrics','--help')
            binaries['qc_build_receipt']=bound(qc_assets['build_receipt'])
            binaries['qc_asset_record']=bound(asset_file)
            for key,expected in qc_assets['sha256'].items():
                if digest(qc_assets[key])!=expected:
                    raise ValueError('QC runtime build artifact SHA changed')
                binaries['qc_'+key]=bound(qc_assets[key])
            source_files += [runtime_helper,runtime_source]
        gpu_assets=None
        if pairhmm_backend=='cuda':
            if not gpu_uuid or not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}',gpu_uuid) or gpu_memory_limit_gib>18:
                raise ValueError('CUDA requires one explicit authorized GPU UUID and at most 18 GiB of owned buffers')
            from .build_preprocessing_gpu import build_preprocessing_gpu
            build=Path(gpu_build_dir).resolve() if gpu_build_dir else root/'gpu_build'
            asset_file=build/'assets.private.json'
            if resume and not asset_file.is_file():raise ValueError('Resume requires the original bound GPU build')
            gpu_assets=json.loads(asset_file.read_text()) if asset_file.exists() else build_preprocessing_gpu(str(prefix),str(build),cpu_affinity=cpus[:4],gpu_selftest_uuid=gpu_uuid)
            build_receipt=build/'build_receipt.private.json'
            if not build_receipt.is_file():raise ValueError('GPU assets have no completed build receipt')
            built=json.loads(build_receipt.read_text())
            manifest=Path(__file__).with_name('_preprocessing_gpu')/'source_manifest.json'
            if (gpu_assets.get('status')!='compiled_components_passed_not_WGS'
                    or built.get('status')!=gpu_assets.get('status') or built.get('artifacts')!=gpu_assets
                    or gpu_assets.get('source_manifest_sha256')!=digest(manifest)):
                raise ValueError('GPU build is incomplete or bound to different packaged sources')
            binaries['gpu_build_receipt']=bound(build_receipt)
            binaries['gpu_asset_record']=bound(asset_file)
            if gpu_assets['condaPrefix']!=str(prefix):raise ValueError('GPU assets must use the same Conda prefix')
            for key,expected in gpu_assets['sha256'].items():
                if digest(gpu_assets[key])!=expected:raise ValueError('GPU build artifact SHA changed')
                binaries['gpu_'+key]=bound(gpu_assets[key])
            source_files += [Path(__file__).with_name('build_preprocessing_gpu.py'),Path(__file__).with_name('_preprocessing_gpu')/'source_manifest.json',Path(__file__).with_name('_preprocessing_gpu')/'evidence.py']
        config=dict(schema=1,sample_id=sample_id,inputs=inputs,tools=binaries,
            source_sha256={p.relative_to(Path(__file__).parent).as_posix():digest(p) for p in source_files},cpu_affinity=cpus,fastp_threads=cleaning_threads,
            host_memory_limit_gib=budget,host_memory_reserve_gib=host_memory_reserve_gib,
            aligner=aligner,index_prefix=str(idx),pairhmm_backend=pairhmm_backend,
            gpu_uuid=gpu_uuid,gpu_memory_limit_gib=gpu_memory_limit_gib,
            gpu_build_dir=str(Path(gpu_build_dir).resolve()) if gpu_build_dir else None,
            intervals=selected_intervals,qc_mode=qc_mode,bqsr_input=bqsr_input,
            qc_exit_fix=qc_exit_fix,
            runtime_build_dir=str(Path(runtime_build_dir).resolve()) if runtime_build_dir else None,
            bqsr_mode=bqsr_mode,selected_bqsr_mode=selected_bqsr_mode,bqsr_workers=training_workers,
            bqsr_build_dir=str(Path(bqsr_build_dir).resolve()) if bqsr_build_dir else None,
            calling_scope='selected_intervals' if selected_intervals else 'complete_reference',
            plink_scope='autosomal_biallelic_hard_calls',input_content_SHA_assessed=False)
        run=_Runner(root,prefix,config,resume)
        run.env['GATK_LOCAL_JAR']=str(jars[0])
        run.env['CUDA_VISIBLE_DEVICES']=''
        (root/'tmp').mkdir(exist_ok=True)
        files=root/'files';files.mkdir(exist_ok=True)
        gatk=run.gatk
        def qc(name,*args):
            original=gatk(name,*args)
            if qc_assets is None:return original
            import shlex
            from ._preprocessing_runtime import qc_argv
            return qc_argv(qc_assets,name,*original[4:],java_options=shlex.split(original[2]))
        clean1,clean2=files/'clean_1.fq.gz',files/'clean_2.fq.gz'
        if shutil.disk_usage(root).free < sum(Path(p).stat().st_size for p in reads)*8+10*2**30:
            raise OSError('Insufficient free storage for conservative FASTQ/intermediate staging reserve')
        dictionary=files/'reference.recomputed.dict'
        run.stage('reference_dictionary',[[run.tool('samtools'),'dict','-o',str(dictionary),str(reference)]],[dictionary],1)
        def sequences(path):
            rows=[]
            for line in Path(path).read_text().splitlines():
                if line.startswith('@SQ\t'):
                    fields=dict(x.split(':',1) for x in line.split('\t')[1:] if ':' in x)
                    rows.append((fields.get('SN'),fields.get('LN'),fields.get('M5')))
            return rows
        if sequences(dictionary)!=sequences(reference.with_suffix('.dict')):
            raise ValueError('FASTA sequence content differs from the input dictionary')
        fai=[tuple(line.split('\t')[:2]) for line in Path(str(reference)+'.fai').read_text().splitlines()]
        if fai != [row[:2] for row in sequences(dictionary)]:raise ValueError('FAI differs from actual reference sequences')
        run.stage('raw_gzip_integrity',[[run.tool('gzip'),'-t','--',*reads]],[],1)
        run.stage('fastp',[[run.tool('fastp'),'-i',reads[0],'-I',reads[1],'-o',str(clean1),'-O',str(clean2),'-j',str(files/'fastp.json'),'-h',str(files/'fastp.html'),'-w',str(cleaning_threads)]],
                  [clean1,clean2,files/'fastp.json',files/'fastp.html'],2)
        import shlex
        sorted_bam,marked,cram,recal,table=[files/name for name in ('sorted.bam','marked.bam','marked.cram','recalibrated.bam','recal.table')]
        rg='@RG\\tID:'+sample_id+'\\tSM:'+sample_id+'\\tLB:'+sample_id+'\\tPL:ILLUMINA'
        mapping=[str(align),'mem','-K','10000000','-M','-Y','-t',str(max(1,threads-1)),'-R',rg,str(idx),str(clean1),str(clean2)]
        # samtools 1.10 derives spill names from -o unless -T is explicit.
        # The indexed-output separator must never enter a temporary BAM name.
        sorting=[run.tool('samtools'),'sort','-@','1','-m','512M','-T',str(root/'tmp'/'sort'),'-O','BAM','--write-index','--reference',str(reference),'-o',str(sorted_bam)+'##idx##'+str(sorted_bam)+'.bai','-']
        run.stage('mapping_sort_index',[['/bin/bash','-c','set -o pipefail\n'+shlex.join(mapping)+' | '+shlex.join(sorting)]],[sorted_bam,Path(str(sorted_bam)+'.bai')],32 if aligner=='bwa-mem2' else 8)
        run.stage('markdup',[[run.tool('sambamba'),'markdup','-t',str(threads),'--sort-buffer-size=6144','--tmpdir='+str(root/'tmp'),str(sorted_bam),str(marked)]],[marked],4)
        run.stage('cram',[[run.tool('samtools'),'view','-@',str(max(0,threads-1)),'-h','-O','CRAM','--write-index','--no-PG','--reference',str(reference),'-o',str(cram)+'##idx##'+str(cram)+'.crai',str(marked)]],[cram,Path(str(cram)+'.crai')],2)
        if bqsr_assets is None:
            cmd=gatk('BaseRecalibrator','-R',reference,'-I',cram,'-O',table)
            for site in sites:cmd+=['--known-sites',str(site)]
            run.stage('BaseRecalibrator',[cmd],[table],8)
            apply_input=cram
        else:
            from .build_preprocessing_bqsr import bqsr_training_argv
            apply_input=files/'canonical_bqsr.bam'
            bqsr_audit=root/'bqsr_training.private.json'
            cmd=bqsr_training_argv(bqsr_assets,input_cram=cram,reference_fasta=reference,known_sites=sites,
                recalibration_table=table,canonical_bam=apply_input,audit_json=bqsr_audit,
                temporary_directory=root/'tmp',cpu_threads=threads,heap_gib=4,
                training_workers=training_workers,queue_capacity=min(32,256//training_workers))
            run.stage('BaseRecalibrator',[cmd],[table,apply_input,bqsr_audit],8)
        run.stage('ApplyBQSR',[gatk('ApplyBQSR','-R',reference,'-I',apply_input,'--bqsr-recal-file',table,'-O',recal,'--create-output-bam-index','false')],[recal],8)
        run.stage('recal_index',[[run.tool('samtools'),'quickcheck','-v',str(recal)],[run.tool('samtools'),'index','-@',str(max(0,threads-1)),str(recal),str(recal)+'.bai']],[Path(str(recal)+'.bai')],1)
        region_args=[item for region in selected_intervals for item in ('-L',region)]
        gvcf=files/'sample.g.vcf.gz'
        heap=min(8,max(4,int(budget/3)))
        hc=gatk('HaplotypeCaller','-R',reference,'-I',recal,'-O',gvcf,'--emit-ref-confidence','GVCF','--native-pair-hmm-threads',str(threads),*region_args,heap_gib=heap)
        hc[2]+=' -Xms'+str(heap)+'g'
        if pairhmm_backend=='cuda':
            if not gpu_uuid or not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}',gpu_uuid) or gpu_memory_limit_gib>18:
                raise ValueError('CUDA requires one explicit authorized GPU UUID and at most 18 GiB of owned buffers')
            from .build_preprocessing_gpu import haplotypecaller_argv, gpu_environment
            assets=gpu_assets
            gpu_audit=root/'pairhmm.private.jsonl'
            hc=haplotypecaller_argv(assets,reference=reference,bam=recal,output=gvcf,
                mode='forceGPU',audit=gpu_audit,native_threads=threads,
                heap_gib=min(8,max(4,int(budget/3))),tmp_dir=root/'tmp',intervals=selected_intervals)
            hc.insert(1,'-XX:ActiveProcessorCount='+str(threads))
            hc=[value.replace('-XX:ParallelGCThreads=2','-XX:ParallelGCThreads=1') for value in hc]
            gpu_env=gpu_environment(assets,gpu_uuid=gpu_uuid,max_device_bytes=int(gpu_memory_limit_gib*2**30))
            run.env.update({key:value for key,value in gpu_env.items() if key.startswith('FNIT_') or key=='CUDA_VISIBLE_DEVICES'})

        run.stage('HaplotypeCaller',[hc],[gvcf,Path(str(gvcf)+'.tbi')],min(12,budget))
        # Preserve WGS coverage semantics; small benchmark intervals restrict only
        # this explicit benchmark. The other two tools share one SAM traversal.
        wgs=files/'wgs_metrics.txt'
        wgsargs=[]
        if selected_intervals:
            interval_list=files/'qc.interval_list'
            interval_list.write_text(reference.with_suffix('.dict').read_text()+''.join(x.split(':')[0]+'\t'+x.split(':')[1].replace('-','\t')+'\t+\twindow'+str(i)+'\n' for i,x in enumerate(selected_intervals)))
            wgsargs=['--INTERVALS',str(interval_list)]
        run.stage('wgs_metrics',[gatk('CollectWgsMetrics','-R',reference,'-I',recal,'-O',wgs,*wgsargs)],[wgs],4)
        if qc_mode=='shared':
            q=files/'qc'
            qcoutputs=[Path(str(q)+x) for x in ('.alignment_summary_metrics','.insert_size_metrics','.insert_size_histogram.pdf','.read_length_histogram.pdf')]
            run.stage('alignment_insert_metrics',[qc('CollectMultipleMetrics','-R',reference,'-I',recal,'-O',q,'--PROGRAM','null','--PROGRAM','CollectAlignmentSummaryMetrics','--PROGRAM','CollectInsertSizeMetrics')],qcoutputs,4)
        else:
            qcoutputs=[files/'alignment_metrics.txt',files/'insert_metrics.txt',files/'insert_metrics.pdf']
            run.stage('alignment_insert_metrics',[qc('CollectAlignmentSummaryMetrics','-R',reference,'-I',recal,'-O',qcoutputs[0]),qc('CollectInsertSizeMetrics','-I',recal,'-O',qcoutputs[1],'-H',qcoutputs[2])],qcoutputs,4)
        raw,vcf=files/'genotyped.vcf.gz',files/'normalized.vcf.gz'
        run.stage('genotype',[gatk('GenotypeGVCFs','-R',reference,'-V',gvcf,'-O',raw,*region_args)],[raw,Path(str(raw)+'.tbi')],8)
        run.stage('normalize',[[run.tool('bcftools'),'norm','-f',str(reference),'-m','-any','--threads',str(max(0,threads-1)),str(raw),'-Oz','-o',str(vcf)],[run.tool('bcftools'),'index','-t',str(vcf)]],[vcf,Path(str(vcf)+'.tbi')],2)
        complete=root/'complete.private.json'
        if complete.exists():
            saved=json.loads(complete.read_text())
            if any(bound(x['path'])!=x for x in saved['final_output_bindings']):raise RuntimeError('Final output changed')
            return saved
        finalization=root/'finalization'
        finalization.mkdir(exist_ok=True)
        attempt=finalization/('attempt_'+str(time.time_ns()))
        attempt.mkdir()
        oracle=attempt/'vcf_oracle'
        conversion_started=time.monotonic()
        scan=vcf_oracle(vcf,sample_id,oracle)
        if scan['variants']:
            all_prefix=files/'autosomes'
            run.stage('plink',[[run.tool('plink2'),'--vcf',str(vcf),'--double-id','--vcf-allow-no-nonvar','--vcf-half-call','error','--set-all-var-ids','@:#:$r:$a','--new-id-max-allele-len',str(scan['maximum_allele_length']),'error','--chr','1-22','--output-chr','26','--threads',str(threads),'--memory',str(min(4096,int(budget*512))),'--make-bed','--out',str(all_prefix)]],[Path(str(all_prefix)+ext) for ext in ('.bed','.bim','.fam')],5)
            verification=verify_and_split_plink(all_prefix,oracle,attempt/'plink',sample_id,scan['chromosome_counts'])
        else:
            (attempt/'plink').mkdir()
            verification=dict(status='no_autosomal_variants',all_hard_calls_verified=None,variants=0,chromosome_counts={},files=[])
        for p,expected in inputs.items():
            if identity(p)!=expected:raise RuntimeError('Bound input changed during the run')
        for p in source_files:
            if digest(p)!=config['source_sha256'][p.relative_to(Path(__file__).parent).as_posix()]:raise RuntimeError('Pipeline source changed during the run')
        for value in binaries.values():
            if bound(value['path'])!=value:raise RuntimeError('Bound tool changed during the run')
        gpu_work=False
        gpu_summary=None
        if pairhmm_backend=='cuda':
            gpu_summary=_gpu_evidence(gpu_audit,gpu_uuid,int(gpu_memory_limit_gib*2**30))
            gpu_work=gpu_summary['gpu_pairs']>0
        result=dict(status='complete_verified',completed_utc=_utc(),scope=config['calling_scope'],
            sample_id=sample_id,gvcf=str(gvcf),vcf=str(vcf),plink_directory=str(attempt/'plink'),
            scan=scan,PLINK=verification,stages=run.events,
            final_output_bindings=[bound(gvcf),bound(vcf),*verification['files']],
            total_wall_seconds=time.monotonic()-started,final_PLINK_readback_seconds=time.monotonic()-conversion_started,
            actual_GPU_work=gpu_work,GPU_evidence=gpu_summary,pairhmm_backend=pairhmm_backend,
            bqsr_mode=selected_bqsr_mode,bqsr_workers=training_workers,
            functional_annotations_created=False,association_performed=False,
            full_reference_calling_completed=not bool(selected_intervals))
        atomic(complete,result)
        return result
    finally:
        os.sched_setaffinity(0,saved_affinity)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=single_subj_preprocessing.__doc__)
    parser.add_argument('--config',required=True,help='JSON object with the Python function parameter names')
    args=parser.parse_args(argv)
    result=single_subj_preprocessing(**json.loads(Path(args.config).read_text()))
    print(json.dumps({'status':result['status'],'scope':result['scope'],'variants':result['scan']['variants'],
                      'wall_seconds':result['total_wall_seconds']}))


if __name__=='__main__':
    main()
