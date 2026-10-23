"""Whole-contig scheduling of the original CPU HaplotypeCaller.

Only fixed-default single-BAM, whole-reference GVCF commands are eligible.
All original JVMs share one task CPU and host-memory budget.
"""
from __future__ import annotations
import contextlib, datetime, functools, gzip, hashlib, heapq, json, math, os, re, resource, shlex, signal, subprocess, time
from dataclasses import dataclass
from pathlib import Path

GIB=2**30
GUARD_STATUS='complete_original_HTSJDK_guard_not_parallel_scientific_equivalence'
GUARD_JAVA_SHA='f3d343db8e7359d191fa2468d116dcee0af19d02be8ef96552344137fe9c37bc'
JOURNAL_MAX_BYTES=64*2**20
READ_COST_BASES=100  # Fixed scheduling heuristic, never a scientific filter or timer.
class BindingError(RuntimeError): pass
class Unsupported(ValueError): pass

def need(value, reason, kind=RuntimeError):
    if not value: raise kind(reason)

def positive_int(value, name):
    need(type(value) is int and value>0,name+' must be a positive integer',ValueError)
    return value

@dataclass(frozen=True)
class Scope:
    intervals: object=None
    backend: str='cpu'
    single_bam: bool=True
    emit_reference_confidence: str='GVCF'
    custom_annotations: bool=False
    custom_read_filters: bool=False
    contamination_fraction: float=0.0
    contamination_file: object=None

@dataclass(frozen=True)
class Api37:
    stage_module: object
    io_module: object
    @classmethod
    def checked(cls, stage_module, io_module):
        for name in ('_limits','_pss','_descendants','_cleanup_stage','_group_members','_utc'):
            need(callable(getattr(stage_module,name,None)),'source37 runtime helper absent: '+name,BindingError)
        for name in ('bound','identity','atomic'):
            need(callable(getattr(io_module,name,None)),'source37 IO helper absent: '+name,BindingError)
        return cls(stage_module,io_module)
    def bound(self,p): return self.io_module.bound(p)
    def identity(self,p): return self.io_module.identity(p)
    def atomic(self,p,value): self.io_module.atomic(Path(p),value)
    def verify(self,record):
        required={'path','device','inode','size','mtime_ns','ctime_ns','sha256'}
        need(type(record) is dict and set(record)==required,'incomplete file binding',BindingError)
        need(type(record['path']) is str and Path(record['path']).is_absolute(),'binding path must be absolute',BindingError)
        need(type(record['sha256']) is str and re.fullmatch('[0-9a-f]{64}',record['sha256']) is not None,'binding SHA256 absent or invalid',BindingError)
        for k in ('device','inode','size','mtime_ns','ctime_ns'):
            need(type(record[k]) is int and record[k]>=0,'invalid file identity integer',BindingError)
        need(record['size']>0 and record['inode']>0,'empty/unidentified bound file',BindingError)
        need(self.bound(record['path'])==record,'bound file changed',BindingError)
        return Path(record['path'])


def budget_plan(cpus, task_gib, heap_gib, workers, nonheap_gib=2, controller_gib=1):
    cpus=tuple(cpus)
    need(cpus and len(set(cpus))==len(cpus) and all(type(x) is int and x>=0 for x in cpus),'invalid CPU set',ValueError)
    positive_int(heap_gib,'heap_gib');positive_int(workers,'workers')
    need(workers<=4 and workers<=len(cpus),'only a few workers within the same CPU set are supported',Unsupported)
    need(type(task_gib) in (int,float) and math.isfinite(task_gib) and task_gib>0,'invalid total task budget',ValueError)
    need(type(nonheap_gib) in (int,float) and nonheap_gib>=1 and type(controller_gib) in (int,float) and controller_gib>=1,'nonheap/controller allowance is too small',ValueError)
    planned=workers*(heap_gib+nonheap_gib)+controller_gib
    need(planned<=task_gib,'multiple original heaps do not fit the shared task budget',Unsupported)
    sets=[tuple(cpus[i::workers]) for i in range(workers)]
    need(set().union(*map(set,sets))==set(cpus) and sum(map(len,sets))==len(cpus),'CPU partition internal error')
    return {'cpu_sets':sets,'workers':workers,'heap_gib_per_worker':heap_gib,'nonheap_gib_per_worker':nonheap_gib,'controller_gib':controller_gib,'prospective_total_gib':planned,'task_gib':task_gib,'OMP_NUM_THREADS':'1','budget_is_shared_PSS_admission_not_independent_cgroup':True}


def select_workers(cpus, task_gib, heap_gib, requested=None):
    """Auto uses at most three original JVMs; their SUM must fit one budget."""
    need(requested is None or (type(requested) is int and 1<=requested<=3),'haplotypecaller_workers must be None or1..3',ValueError)
    if requested==1:return None,'explicit_original_serial'
    candidates=(3,2) if requested is None else (requested,)
    for count in candidates:
        try:return budget_plan(cpus,task_gib,heap_gib,count),'auto_shared_budget' if requested is None else 'explicit_shared_budget'
        except Unsupported:pass
    if requested is not None:raise Unsupported('requested HC JVMs do not fit the shared CPU/heap/nonheap/controller budget')
    return None,'shared_budget_fits_only_original_serial'


def default_command(serial, *, bam, reference, output, threads, scope):
    need(scope==Scope(),'unsupported HC scope: execute the unchanged original serial command',Unsupported)
    need(len(serial)>=4 and serial[1]=='--java-options' and serial[3]=='HaplotypeCaller','nonstandard HC launcher',Unsupported)
    expected=['HaplotypeCaller','-R',str(reference),'-I',str(bam),'-O',str(output),'--emit-ref-confidence','GVCF','--native-pair-hmm-threads',str(threads)]
    need(list(serial[3:])==expected,'only source37 fixed default HC arguments are admitted',Unsupported)
    opts=shlex.split(serial[2]); values={}
    for token in opts:
        if re.fullmatch(r'-Xm[xs][1-9][0-9]*g',token): key=token[:4]; values[key]=int(token[4:-1])
        elif token==f'-XX:ActiveProcessorCount={threads}': key='active'; values[key]=threads
        elif token=='-XX:ParallelGCThreads=1': key='gc'; values[key]=1
        elif token.startswith('-Djava.io.tmpdir=') and len(token)>18: key='tmp'; values[key]=token.split('=',1)[1]
        else: raise Unsupported('nondefault Java option')
    need(len(opts)==len(values) and set(values) in ({'-Xmx','active','gc','tmp'},{'-Xmx','-Xms','active','gc','tmp'}) and values.get('-Xms',values['-Xmx'])<=values['-Xmx'],'nondefault/duplicate Java options',Unsupported)
    return values


def read_dictionary(fai, maximum=100000):
    need(Path(fai).stat().st_size<=8*2**20,'FAI exceeds metadata bound',BindingError)
    rows=[];seen=set()
    with Path(fai).open(encoding='utf-8') as source:
        for line in source:
            f=line.rstrip('\n').split('\t');need(len(f)==5 and f[0] and f[0] not in seen,'invalid/duplicate FAI sequence',BindingError)
            try: length=int(f[1])
            except ValueError as e: raise BindingError('noninteger contig length') from e
            need(0<length<=2**31-1 and len(rows)<maximum,'unsupported dictionary size/length',BindingError)
            seen.add(f[0]);rows.append((f[0],length))
    need(rows,'empty reference dictionary',BindingError)
    return tuple(rows)


def guard_rows(certificate, dictionary, *, expected_bam, expected_fai_sha, expected_command_sha, expected_jar_sha):
    c=certificate
    need(type(c) is dict and c.get('status')==GUARD_STATUS and type(c.get('actual_exit')) is int and c['actual_exit']==0,'guard has no actual successful completion',BindingError)
    for k in ('all_HTSJDK_visible_records_scanned_once','HTSJDK_BGZF_CRC_validation_enabled','BGZF_final_terminator_checked'):
        need(c.get(k) is True,'guard did not prove complete input/CRC/EOF: '+k,BindingError)
    need(c.get('bam_identity')=={k:v for k,v in expected_bam.items() if k!='sha256'} and c.get('full_compressed_bam_sha256')==expected_bam['sha256'] and c.get('full_bam_compressed_bytes_hashed')==expected_bam['size'],'guard did not scan the bound full BAM',BindingError)
    need(c.get('fai_sha256')==expected_fai_sha and c.get('command_sha256')==expected_command_sha,'guard reference/command differs',BindingError)
    proof=c.get('mapped_nonduplicate_bound_proof',{})
    need(proof.get('enabled') is True and proof.get('original_gatk_jar_sha256')==expected_jar_sha,'guard did not load the fixed original GATK jar',BindingError)
    rows=c.get('per_contig');need(type(rows) is list and len(rows)==len(dictionary),'guard dictionary coverage incomplete',BindingError)
    for row,(name,length) in zip(rows,dictionary):
        need(type(row) is dict and row.get('contig')==name and row.get('length')==length,'ordered guard/reference dictionary differs',BindingError)
        for key in ('mapped_all_including_duplicates','max_mapped_at_alignment_start','raw_starts_over_50'):
            need(type(row.get(key)) is int and row[key]>=0,'invalid guard raw count',BindingError)
        need(row['max_mapped_at_alignment_start']<=row['mapped_all_including_duplicates'] and row['raw_starts_over_50']<=row['mapped_all_including_duplicates'],'guard raw maxima/count impossible',BindingError)
        need((row['max_mapped_at_alignment_start']>50)==(row['raw_starts_over_50']>0),'guard raw-risk counters inconsistent',BindingError)
    need(sum(x['mapped_all_including_duplicates'] for x in rows)==c.get('mapped_records_including_duplicates'),'guard totals differ',BindingError)
    return rows


def partition_contigs(rows, workers):
    positive_int(workers,'workers');need(2<=workers<=4,'parallel worker count must be2..4',Unsupported)
    need(rows and len({r['contig'] for r in rows})==len(rows),'contig input has gaps or duplicates',BindingError)
    risky=[r for r in rows if r['max_mapped_at_alignment_start']>50]; safe=[r for r in rows if r['max_mapped_at_alignment_start']<=50]
    need(len(safe)>0,'all contigs require the original common RNG stream',Unsupported)
    groups=[[] for _ in range(workers)];loads=[0]*workers
    def cost(row):
        passed=row.get('exact_original_filter_passing',0)
        need(type(passed) is int and 0<=passed<=row['mapped_all_including_duplicates'],'invalid bound scheduling count',BindingError)
        return row['length']+READ_COST_BASES*passed
    if risky: groups[0]=[r['contig'] for r in risky];loads[0]=sum(cost(r) for r in risky)
    order={r['contig']:i for i,r in enumerate(rows)}
    # A scheduling heuristic, never an elapsed-time completion ratio.
    for row in sorted(safe,key=lambda r:(-cost(r),order[r['contig']])):
        target=min(range(workers),key=lambda i:(loads[i],i));groups[target].append(row['contig']);loads[target]+=cost(row)
    need(all(groups),'not enough complete contigs for the requested workers',Unsupported)
    groups=[tuple(sorted(g,key=order.__getitem__)) for g in groups]
    flat=[x for g in groups for x in g]
    need(len(flat)==len(rows) and len(set(flat))==len(rows) and set(flat)==set(order),'partition omitted/duplicated a contig',BindingError)
    if risky:need(tuple(x for x in groups[0] if x in {r['contig'] for r in risky})==tuple(r['contig'] for r in risky),'risk contigs no longer share dictionary-ordered subsequence in one JVM',BindingError)
    return tuple(groups)


def _commit_owned_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    need(not path.is_symlink(),'run-local scope file must not be a symlink',BindingError)
    if path.exists():
        need(path.is_file() and path.stat().st_nlink==1 and path.read_bytes()==data,'existing run-local scope file differs',BindingError)
    else:
        with path.open('xb') as output:output.write(data);output.flush();os.fsync(output.fileno())


def write_interval_lists(groups,dictionary,reference_dict,root,api):
    """Original complete .dict header + one whole-contig HTSJDK interval row.

    Names are data fields, never a parser expression. This covers valid colon
    and asterisk names without a literal-contig-versus-region ambiguity.
    """
    dictionary=tuple(dictionary);reference_dict=Path(reference_dict)
    need(reference_dict.stat().st_size<=32*2**20,'reference dictionary header exceeds metadata bound',BindingError)
    before=api.bound(reference_dict);header=reference_dict.read_bytes()
    need(header.endswith(b'\n') and b'\r' not in header,'reference dictionary header must have complete LF lines',BindingError)
    parsed=[]
    for line in header.splitlines():
        need(line.startswith(b'@'),'nonheader row in reference dictionary',BindingError)
        if line.startswith(b'@SQ\t'):
            fields={}
            for token in line.split(b'\t')[1:]:
                name,separator,value=token.partition(b':')
                need(separator and name not in fields,'invalid/duplicate SAM dictionary tag',BindingError);fields[name]=value
            need(b'SN' in fields and b'LN' in fields,'SAM sequence dictionary lacks name/length',BindingError)
            try:parsed.append((fields[b'SN'].decode('utf-8'),int(fields[b'LN'])))
            except (ValueError,UnicodeDecodeError) as error:raise BindingError('invalid SAM dictionary name/length') from error
    need(tuple(parsed)==dictionary,'original .dict ordered name/length axis differs from FAI',BindingError)
    all_names=[name for group in groups for name in group]
    need(len(all_names)==len(dictionary) and len(set(all_names))==len(dictionary) and set(all_names)==dict(dictionary).keys(),'interval lists omit or duplicate reference contigs',BindingError)
    lengths=dict(dictionary);rank={name:i for i,(name,length) in enumerate(dictionary)};bindings=[]
    for index,group in enumerate(groups):
        need(tuple(sorted(group,key=rank.__getitem__))==tuple(group),'interval list group order differs from dictionary',BindingError)
        data=header+b''.join((name+'\t1\t'+str(lengths[name])+'\t+\twhole_contig_'+str(rank[name])+'\n').encode('utf-8') for name in group)
        path=Path(root)/('worker_'+str(index))/'whole_contigs.interval_list'
        _commit_owned_bytes(path,data);bindings.append(api.bound(path))
    api.verify(before);return bindings


def worker_commands(serial, groups, budget, root, *, interval_files=None):
    root=Path(root);commands=[];outputs=[]
    for i,contigs in enumerate(groups):
        d=root/('worker_'+str(i));out=d/'sample.g.vcf.gz';tmp=d/'tmp'
        options=shlex.split(serial[2]);options=[('-XX:ActiveProcessorCount='+str(len(budget['cpu_sets'][i]))) if x.startswith('-XX:ActiveProcessorCount=') else '-Djava.io.tmpdir='+str(tmp) if x.startswith('-Djava.io.tmpdir=') else x for x in options]
        command=list(serial);command[2]=shlex.join(options);command[command.index('-O')+1]=str(out)
        if interval_files is None:
            for name in contigs:command+=['-L',name]
        else:
            need(len(interval_files)==len(groups),'interval-file worker count differs',BindingError)
            command+=['-L',str(interval_files[i])]
        commands.append(command);outputs.append((out,Path(str(out)+'.tbi')))
    return commands,outputs


def _proc_snapshot(pid):
    directory=Path('/proc')/str(pid);fields=(directory/'stat').read_text().rsplit(')',1)[1].split()
    return {'uid':directory.stat().st_uid,'start':int(fields[19]),'sid':int(fields[3]),'state':fields[0]}

@dataclass
class Lease:
    process: object
    uid: int
    start: int | None
    sid: int
    drained: bool=False
    def verify(self):
        need(not self.drained,'released PID must never be queried again',BindingError)
        need(self.start is not None,'initial root identity was never captured; refuse signals',BindingError)
        current=_proc_snapshot(self.process.pid)
        need((current['uid'],current['start'],current['sid'])==(self.uid,self.start,self.sid),'owned root PID/UID/start/SID no longer verified',BindingError)
        return current

def capture_lease(process):
    # Do not call poll/wait before capture: unreaped child/zombie PID cannot be
    # reused. Release this lease only after observed empty session + actual wait.
    row=_proc_snapshot(process.pid)
    need(row['uid']==os.getuid() and row['sid']==process.pid,'new original HC session identity differs',BindingError)
    return Lease(process,row['uid'],row['start'],row['sid'])

def _strict_session_members(lease):
    """Same-UID /proc read errors are unknown, never an empty session."""
    lease.verify();members={}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():continue
        try:
            if directory.stat().st_uid!=lease.uid:continue
            row=_proc_snapshot(int(directory.name))
        except FileNotFoundError:continue  # A vanished process cannot be signalled.
        except (PermissionError,OSError,ValueError,IndexError) as error:
            raise BindingError('same-UID process identity enumeration is unknown') from error
        need(row['uid']==lease.uid,'UID changed during process enumeration',BindingError)
        if row['sid']==lease.sid and row['state'] not in ('Z','X'):
            members[int(directory.name)]=row
    lease.verify();return members


def owned_members(lease,api):
    return set(_strict_session_members(lease))


def cleanup_owned(lease,api,grace_seconds=5):
    """Only exact, still-owned PID/start/UID/SID leases may receive a signal.

    Keep the root unreaped throughout cleanup, preventing SID reuse. Every
    member is checked again immediately before each signal. Unknown identity
    refuses signalling and remains an explicit cleanup failure.
    """
    for sig in (signal.SIGTERM,signal.SIGKILL):
        deadline=time.monotonic()+grace_seconds
        while True:
            members=_strict_session_members(lease)
            for pid,before in members.items():
                lease.verify()
                try:now=_proc_snapshot(pid)
                except FileNotFoundError:continue
                need((now['uid'],now['start'],now['sid'])==(before['uid'],before['start'],before['sid']),'owned member PID identity changed; refuse signal',BindingError)
                if now['state'] in ('Z','X'):continue
                try:os.kill(pid,sig)
                except ProcessLookupError:pass
            if lease.verify()['state'] in ('Z','X') and not _strict_session_members(lease):return
            if time.monotonic()>=deadline:break
            time.sleep(.05)
    raise RuntimeError('owned HC session did not drain after bounded cleanup')


def monitor(runner, api, active_leases, prospective_gib):
    m=api.stage_module;limits=m._limits()
    need(limits['admission_available_bytes']>=runner.reserve,'live shared-node reserve is no longer available',MemoryError)
    pids={os.getpid()}
    for lease in active_leases:
        lease.verify();pids.update(owned_members(lease,api))
    used=m._pss(pids)
    need(type(used) in (int,float) and math.isfinite(used) and 0<=used<=runner.budget,'shared task PSS unknown or above budget',MemoryError)
    return used

def successful_waits(waits, count):
    need(type(waits) is list and len(waits)==count,'actual worker wait count is incomplete',BindingError)
    for row in waits:
        need(type(row) is dict and type(row.get('returncode')) is int and row['returncode']==0 and row.get('session_drained_before_lease_release') is True,'worker has no actual successful drained wait',BindingError)
        need(all(type(row.get(k)) is int and row[k]>=0 for k in ('pid','start','sid','uid')) and row['pid']>0 and row['start']>0 and row['sid']==row['pid'] and row['uid']==os.getuid(),'actual worker identity is incomplete',BindingError)
    need(len({(x['pid'],x['start']) for x in waits})==count,'duplicate actual worker identity',BindingError)


def _journal_records(runner):
    path=runner.root/'journal.private.jsonl';records={}
    if not path.exists():return records
    need(path.is_file() and not path.is_symlink(),'HC journal is not an owned regular file',BindingError)
    need(path.stat().st_size<=JOURNAL_MAX_BYTES,'HC journal exceeds total metadata byte bound',BindingError)
    consumed=0
    with path.open('rb') as stream:
        while True:
            line=stream.readline(16*2**20+1)
            if not line:break
            consumed+=len(line)
            need(consumed<=JOURNAL_MAX_BYTES,'HC journal grew beyond total metadata byte bound',BindingError)
            need(len(line)<=16*2**20 and line.endswith(b'\n'),'HC journal has an incomplete/unbounded line',BindingError)
            row=json.loads(line);name=row.get('stage')
            need(type(name) is str and name and '/' not in name and name not in records,'HC journal has invalid/duplicate stage',BindingError)
            records[name]=row
            need(len(records)<=1000,'HC journal stage metadata bound exceeded',BindingError)
    return records


def _append_completed_journal(runner,record):
    """Same completed record as its stage receipt, durable append exactly once.

    A resume never invokes this function or repairs a missing journal entry.
    Fresh completion follows the existing Runner order: complete JSON, journal
    append+flush+fsync, then in-memory event. Unknown/incomplete commits fail.
    """
    name=record.get('stage')
    need(type(name) is str and name and '/' not in name and record.get('status')=='complete','only an actual completed HC stage may commit to journal',BindingError)
    need(name not in _journal_records(runner),'fresh HC completion already has a journal stage',BindingError)
    payload=json.dumps(record,allow_nan=False)+'\n'
    need(len(payload.encode())<=16*2**20,'HC completed journal record exceeds metadata bound',BindingError)
    journal=runner.root/'journal.private.jsonl'
    old_bytes=journal.stat().st_size if journal.exists() else 0
    need(old_bytes+len(payload.encode())<=JOURNAL_MAX_BYTES,'HC journal append exceeds total metadata byte bound',BindingError)
    with journal.open('a') as output:
        output.write(payload);output.flush();os.fsync(output.fileno())


def _verify_completed_journal(runner,record):
    expected=json.loads(json.dumps(record,allow_nan=False))
    need(_journal_records(runner).get(record.get('stage'))==expected,'committed HC stage journal missing or different; never repair on resume',BindingError)


def parallel_stage(runner, api, name, commands, outputs, budget, *, poll_seconds=.25):
    """Thin source37 adapter. Drained roots are removed and never queried again.

    Uses the existing original-HC session cleanup only for a still verified
    UID/start/SID lease. Unknown leases are recorded; no signal is guessed.
    """
    need(len(commands)==len(budget['cpu_sets']) and commands,'worker/CPU cardinality differs')
    directory=runner.root/'stages'/name
    complete=directory/'complete.private.json'
    if complete.is_file():
        saved=json.loads(complete.read_text())
        need(saved.get('status')=='complete' and saved.get('commands')==commands and saved.get('budget')==json.loads(json.dumps(budget)),'parallel resume binding differs',BindingError)
        need(type(saved.get('exit_codes')) is list and len(saved['exit_codes'])==len(commands) and all(type(x) is int and x==0 for x in saved['exit_codes']),'parallel actual exits incomplete',BindingError)
        successful_waits(saved.get('actual_waits'),len(commands))
        for value in saved.get('outputs',[]):api.verify(value)
        need(len(saved.get('outputs',[]))==len(outputs) and [x['path'] for x in saved['outputs']]==[str(Path(p).resolve()) for p in outputs],'parallel output closure/roles incomplete',BindingError)
        _verify_completed_journal(runner,saved)
        runner.events.append(saved);return saved
    need(not directory.exists(),'parallel stage already exists; preserve incomplete output')
    limits=api.stage_module._limits();future=budget['prospective_total_gib']
    need(future*GIB<=runner.budget and limits['admission_available_bytes']>=future*GIB+runner.reserve,'whole cohort is not admitted',MemoryError)
    need(set().union(*map(set,budget['cpu_sets']))<=runner.cpuset,'worker escaped the same task CPU set',ValueError)
    directory.mkdir(parents=True);began=time.monotonic();leases=[];active=[];streams=[];exits=[None]*len(commands);peak=None;record={'stage':name,'commands':commands,'started_utc':api.stage_module._utc(),'status':'running','concurrent_original_JVMs':len(commands),'budget':budget,'actual_waits':[]}
    api.atomic(directory/'started.private.json',record)
    try:
        for i,command in enumerate(commands):
            cpus=set(budget['cpu_sets'][i])
            def setup(cpus=cpus):os.sched_setaffinity(0,cpus);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
            stream=(directory/(str(i)+'.log')).open('xb');streams.append(stream)
            env=dict(runner.env,OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
            for key in ('JAVA_TOOL_OPTIONS','_JAVA_OPTIONS','JDK_JAVA_OPTIONS','CLASSPATH','PYTHONPATH'):env.pop(key,None)
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=runner.root,start_new_session=True,preexec_fn=setup)
            lease=Lease(process,os.getuid(),None,process.pid);leases.append(lease);active.append((i,lease))
            captured=capture_lease(process);lease.start=captured.start;lease.uid=captured.uid;lease.sid=captured.sid
        while active:
            for i,lease in tuple(active):
                state=lease.verify()['state']
                if state in ('Z','X'):
                    need(not owned_members(lease,api),'HC root exited before its owned session drained')
                    exits[i]=lease.process.wait();lease.drained=True;active.remove((i,lease))
                    record['actual_waits'].append({'pid':lease.process.pid,'start':lease.start,'sid':lease.sid,'uid':lease.uid,'returncode':exits[i],'session_drained_before_lease_release':True})
                    need(exits[i]==0,'one original HC worker exited nonzero')
            if active:
                peak=max(peak or 0,monitor(runner,api,[lease for i,lease in active],future));time.sleep(poll_seconds)
        command_wall=time.monotonic()-began;hash_began=time.monotonic();checked=[api.bound(p) for p in outputs]
        need(all(x['size']>0 for x in checked),'empty required shard output')
        record.update(status='complete',exit_codes=exits,outputs=checked,command_and_resource_observer_seconds=command_wall,artifact_hashing_seconds=time.monotonic()-hash_began,wall_seconds=time.monotonic()-began,peak_sampled_pss_bytes=peak,completed_utc=api.stage_module._utc())
        api.atomic(directory/'complete.private.json',record);_append_completed_journal(runner,record);runner.events.append(record);return record
    except BaseException as error:
        cleanup_errors=[]
        for i,lease in tuple(active):
            try:
                lease.verify();owned_members(lease,api)
                cleanup_owned(lease,api)
                exits[i]=lease.process.wait(timeout=5);lease.drained=True;active.remove((i,lease))
                record['actual_waits'].append({'pid':lease.process.pid,'start':lease.start,'sid':lease.sid,'uid':lease.uid,'returncode':exits[i],'strict_owned_cleanup_returned_after_drain':True})
            except BaseException as cleanup:
                cleanup_errors.append({'pid':lease.process.pid,'error_type':type(cleanup).__name__,'error':str(cleanup),'lease_drained':lease.drained,'unknown_identity_never_signaled':isinstance(cleanup,(BindingError,FileNotFoundError,PermissionError))})
                try:
                    # waitpid only for this Popen child; no signal or PID query.
                    # Unknown session drain is STILL unknown even after wait.
                    exits[i]=lease.process.wait(timeout=5)
                    record['actual_waits'].append({'pid':lease.process.pid,'returncode':exits[i],'session_drain_unknown':True,'not_a_successful_cleanup_proof':True})
                except BaseException as waiting:cleanup_errors.append({'pid':lease.process.pid,'wait_error_type':type(waiting).__name__,'wait_error':str(waiting)})
        record.update(status='failed',error_type=type(error).__name__,error=str(error),exit_codes=exits,cleanup_errors=cleanup_errors,unreleased_lease_count=len(active),peak_sampled_pss_bytes=peak,wall_seconds=time.monotonic()-began)
        api.atomic(directory/'failed.private.json',record);raise
    finally:
        for stream in streams:stream.close()


COMMAND_RE=re.compile(rb'^##GATKCommandLine=<ID=HaplotypeCaller,CommandLine="([^"\r\n]*)",Version="4\.2\.0\.0",Date="([^"\r\n]*)">\n$')

def header_command(line, expected_output, expected_contigs):
    match=COMMAND_RE.fullmatch(line);need(match is not None,'unsupported HC command header structure/version',BindingError)
    date=match.group(2).decode('ascii');date_match=re.fullmatch(r'([A-Za-z]+) ([0-9]{1,2}), ([0-9]{4}) ([0-9]{1,2}):([0-9]{2}):([0-9]{2}) (AM|PM) ([A-Z]{2,6})',date)
    need(date_match is not None,'invalid original Date format',BindingError)
    month,day,year,hour,minute,second,ampm,zone=date_match.groups();months=('January','February','March','April','May','June','July','August','September','October','November','December')
    need(month in months and 1<=int(hour)<=12,'invalid original Date value',BindingError)
    try:datetime.datetime(int(year),months.index(month)+1,int(day),int(hour)%12+(12 if ampm=='PM' else 0),int(minute),int(second))
    except ValueError as error:raise BindingError('invalid original Date value') from error
    tokens=shlex.split(match.group(1).decode('utf-8'))
    need(tokens.count(str(expected_output))==1 and tokens[tokens.index(str(expected_output))-1]=='--output','shard output role differs',BindingError)
    intervals=[tokens[i+1] for i,x in enumerate(tokens[:-1]) if x=='--intervals']
    need(intervals==list(expected_contigs),'actual shard HC whole-contig arguments differ',BindingError)
    # Compare every other expanded default scientific flag across shards.
    normalized=[];i=0
    while i<len(tokens):
        if tokens[i]=='--intervals':i+=2;continue
        normalized.append('<OUTPUT>' if tokens[i]==str(expected_output) else tokens[i]);i+=1
    return tuple(normalized)


class Shard:
    def __init__(self,path,contigs,dictionary,*,interval_argument=None,maximum_line=4*2**20,maximum_header=16*2**20):
        self.path=Path(path);self.contigs=tuple(contigs);self.interval_arguments=(str(interval_argument),) if interval_argument is not None else self.contigs;self.allowed=set(contigs);self.dictionary=dictionary;self.rank={x[0]:i for i,x in enumerate(dictionary)};self.length=dict(dictionary);self.maximum_line=maximum_line;self.maximum_header=maximum_header;self.rows=0;self.observed_contigs=set();self.body_sha=hashlib.sha256();self.last=None;self.last_refblock_end=None;self.last_refblock_contig=None
        self.stream=gzip.open(path,'rb');self.header=[];self.command=None;self.normalized_command=None;self.column_line=None;self.first=None
        try:self._header()
        except BaseException:self.close();raise
    def _line(self):
        b=self.stream.readline(self.maximum_line+1)
        need(len(b)<=self.maximum_line and (not b or b.endswith(b'\n')),'truncated/unbounded GVCF line',BindingError);return b
    def _header(self):
        total=0;commands=0
        while True:
            b=self._line();need(b,'missing VCF header/EOF',BindingError);total+=len(b);need(total<=self.maximum_header,'header bound exceeded',BindingError)
            if b.startswith(b'##GATKCommandLine='):
                commands+=1;self.command=b;self.normalized_command=header_command(b,self.path,self.interval_arguments);continue
            self.header.append(b)
            if b.startswith(b'#CHROM\t'):
                self.column_line=b;need(len(b.rstrip(b'\n').split(b'\t'))==10,'default single-sample GVCF requires10 columns',BindingError);break
            need(b.startswith(b'##'),'invalid VCF header',BindingError)
        need(commands==1,'exactly one original HC command header required',BindingError)
    def next(self):
        b=self._line()
        if not b:return None
        need(not b.startswith(b'#'),'header after GVCF records',BindingError);fields=b[:-1].split(b'\t');need(len(fields)==10,'GVCF sample/column count changed',BindingError)
        try:contig=fields[0].decode('utf-8');pos=int(fields[1])
        except (UnicodeDecodeError,ValueError) as error:raise BindingError('invalid GVCF coordinate syntax') from error
        need(contig in self.allowed and contig in self.rank and 1<=pos<=self.length[contig],'shard emitted a foreign/out-of-range contig',BindingError)
        key=(self.rank[contig],pos);need(self.last is None or key>=self.last,'GVCF order decreases',BindingError)
        endings=[v[4:] for v in fields[7].split(b';') if v.startswith(b'END=')];need(len(endings)<=1,'duplicate END',BindingError)
        try:end=int(endings[0]) if endings else pos+len(fields[3])-1
        except ValueError as error:raise BindingError('invalid END integer') from error
        need(pos<=end<=self.length[contig],'END/REF extent outside dictionary',BindingError)
        if self.last_refblock_contig==contig:need(pos>self.last_refblock_end,'overlapping reference blocks',BindingError)
        self.last_refblock_contig=contig if fields[4]==b'<NON_REF>' and endings else None;self.last_refblock_end=end if self.last_refblock_contig else None
        self.observed_contigs.add(contig);self.last=key;self.rows+=1;self.body_sha.update(b);return key,b
    def close(self):self.stream.close()


def merge_shards(shards, groups, dictionary, destination, *, tick=lambda:None, api=None, output_bindings=None, interval_arguments=None):
    """Raw-line k-way merge: no reformatting, genotype parsing, or row tables.

    Preserve EVERY original HC shard GATKCommandLine byte-for-byte.
    All non-command headers must match exactly. No unexecuted serial
    command is created, and no header is presented as the final serial scope.
    """
    need(len(shards)==len(groups) and 1<len(shards)<=4,'missing/extra shard file',BindingError)
    flattened=[x for g in groups for x in g];names=[x[0] for x in dictionary]
    need(len(flattened)==len(names) and len(set(flattened))==len(names) and set(flattened)==set(names),'dictionary partition gap/duplicate',BindingError)
    if output_bindings is not None:
        need(api is not None and len(output_bindings)==len(shards),'closed shard bindings required',BindingError)
        for p,b in zip(shards,output_bindings):need(Path(p).resolve()==api.verify(b).resolve(),'shard binding belongs to another file',BindingError)
    readers=[];began=time.monotonic();last_tick=began
    try:
        need(interval_arguments is None or len(interval_arguments)==len(groups),'actual interval argument closure incomplete',BindingError)
        for index,(p,g) in enumerate(zip(shards,groups)):readers.append(Shard(p,g,dictionary,interval_argument=None if interval_arguments is None else interval_arguments[index]))
        first=readers[0]
        for reader in readers[1:]:need(reader.header==first.header and reader.normalized_command==first.normalized_command,'non-command header or expanded scientific defaults differ',BindingError)
        destination=Path(destination);need(not destination.exists(),'merged destination already exists',BindingError)
        count=0;digest=hashlib.sha256()
        with destination.open('xb') as output:
            def write(b):output.write(b);digest.update(b)
            for b in first.header[:-1]:write(b)
            for reader in readers:write(reader.command)
            write(first.header[-1]);pending=[]
            for i,reader in enumerate(readers):
                item=reader.next()
                if item is not None:heapq.heappush(pending,(item[0],i,item[1]))
            while pending:
                key,i,line=heapq.heappop(pending);write(line);count+=1
                item=readers[i].next()
                if item is not None:heapq.heappush(pending,(item[0],i,item[1]))
                now=time.monotonic()
                if now-last_tick>=.25:tick();last_tick=now
            output.flush();os.fsync(output.fileno())
        tick();need(count==sum(r.rows for r in readers),'merged output dropped records')
        if output_bindings is not None:
            for b in output_bindings:api.verify(b)
        return {'status':'raw_records_merged_complete_EOF_no_scientific_equivalence_claim','records':count,'decoded_sha256':digest.hexdigest(),'shards':[{'records':r.rows,'body_ordered_sha256':r.body_sha.hexdigest()} for r in readers],'all_gzip_sources_read_to_EOF_CRC':True,'wall_seconds':time.monotonic()-began,'every_actual_HC_shard_command_header_preserved_raw':True,'observed_contigs':tuple(name for name,length in dictionary if any(name in r.observed_contigs for r in readers))}
    finally:
        for reader in readers:reader.close()


def compress_index_readback(runner,api,plain,final,expected_decoded_sha256,observed_contigs,*,bgzip_binding,tabix_binding,dictionary):
    bgzip=api.verify(bgzip_binding);tabix=api.verify(tabix_binding);plain=Path(plain);final=Path(final)
    need(str(final)==str(plain)+'.gz','run-local .vcf.gz role required',BindingError)
    if final.exists():need((runner.root/'stages/HaplotypeCaller_bgzip/complete.private.json').is_file(),'uncommitted final GVCF already exists',BindingError)
    runner.stage('HaplotypeCaller_bgzip',[[str(bgzip),'-@','1','-l','2','-k',str(plain)]],[final],1)
    runner.stage('HaplotypeCaller_tabix',[[str(tabix),'-p','vcf',str(final)]],[Path(str(final)+'.tbi')],1)
    digest=hashlib.sha256();total=0
    with gzip.open(final,'rb') as source:
        while True:
            block=source.read(1<<20)
            if not block:break
            digest.update(block);total+=len(block);monitor(runner,api,[],1)
    need(digest.hexdigest()==expected_decoded_sha256,'BGZF full decoded readback differs from merged bytes',BindingError)
    runner.stage('HaplotypeCaller_index_names',[[str(tabix),'-l',str(final)]],[],1)
    names_log=runner.root/'stages/HaplotypeCaller_index_names/0.log'
    need(names_log.stat().st_size<=8*2**20,'index names exceed metadata bound',BindingError)
    names=tuple(names_log.read_text().splitlines())
    need(names==tuple(observed_contigs) and len(set(names))==len(names),'index contig axis differs from observed complete VCF',BindingError)
    # Full-contig BED fields avoid ambiguous colon/asterisk region expressions.
    # Each observed contig occurs once; query output remains an explicit extra
    # private full readback file in this initial candidate.
    lengths=dict(dictionary);need(all(name in lengths for name in names),'index names outside bound dictionary',BindingError)
    bed=runner.root/'stages/HaplotypeCaller_selection/full_index_query.bed'
    _commit_owned_bytes(bed,b''.join((name+'\t0\t'+str(lengths[name])+'\n').encode('utf-8') for name in names))
    query=[str(tabix),'-h','-R',str(bed),str(final)] if names else [str(tabix),'-H',str(final)]
    query_log=runner.root/'stages/HaplotypeCaller_index_full_readback/0.log'
    runner.stage('HaplotypeCaller_index_full_readback',[query],[query_log],1)
    with query_log.open('rb') as source:
        indexed=hashlib.sha256()
        for block in iter(lambda:source.read(1<<20),b''):
            indexed.update(block);monitor(runner,api,[],1)
    need(indexed.hexdigest()==expected_decoded_sha256,'original tabix full-index query omitted/reordered/changed VCF bytes',BindingError)
    return {'status':'BGZF_EOF_and_original_tabix_full_index_readback_match','decoded_bytes':total,'decoded_sha256':digest.hexdigest(),'gvcf':api.bound(final),'tbi':api.bound(str(final)+'.tbi'),'index_names':api.bound(names_log),'full_index_readback':api.bound(query_log),'query_regions':api.bound(bed) if names else None,'all_tool_actual_waits_recorded_by_Runner':True}


def original_serial_fallback(runner,api,serial,outputs,reason):
    directory=runner.root/'stages'/'HaplotypeCaller_selection';directory.mkdir(parents=True,exist_ok=True)
    decision={'selected':'original_serial','reason':str(reason),'scientific_command':serial,'no_scientific_step_skipped':True}
    old=directory/'decision.private.json'
    if old.exists():need(json.loads(old.read_text())==decision,'execution decision changed; new output directory required',BindingError)
    else:api.atomic(old,decision)
    event=runner.stage('HaplotypeCaller',[serial],outputs,min(12,runner.budget/GIB))
    return {'used_parallel':False,'fallback_reason':str(reason),'event':event}


def _record_hc_failure(function):
    @functools.wraps(function)
    def checked(runner,api,*args,**kwargs):
        try:return function(runner,api,*args,**kwargs)
        except BaseException as error:
            # Includes pure-Python merge/CRC/order/index-readback errors. Never
            # fabricate a successful top stage or overwrite a previous commit.
            try:
                directory=runner.root/'stages/HaplotypeCaller_selection'
                directory.mkdir(parents=True,exist_ok=True)
                api.atomic(directory/('failure_attempt_'+str(time.time_ns())+'.private.json'),
                    {'status':'failed_HC_candidate_attempt_not_complete',
                     'error_type':type(error).__name__,'error':str(error),
                     'failed_utc':api.stage_module._utc(),
                     'complete_receipt_existed_not_a_validation_proof':(runner.root/'stages/HaplotypeCaller/complete.private.json').is_file(),
                     'HC_scientific_worker_failure_not_inferred_from_Python_or_binding_error':True})
            except BaseException as recording:
                if hasattr(error,'add_note'):error.add_note('Private failure receipt also failed: '+type(recording).__name__)
            raise
    return checked


@_record_hc_failure

def run_haplotypecaller(runner,api,serial,outputs,*,bam,reference,threads,requested_workers=None,
                        guard_assets=None,scope=Scope(),bgzip_binding=None,tabix_binding=None,fallback_reason=None):
    """Local candidate API glue, not an accepted performance/scientific gate.

    Eligible original HC shards own whole contigs. One original HTSJDK scan
    precedes scheduling; the source50 raw risk upper bound controls scientific
    eligibility. All guard/merge/full-index-readback I/O is part of this stage.
    """
    if fallback_reason is not None:return original_serial_fallback(runner,api,serial,outputs,fallback_reason)
    began=time.monotonic();started_utc=api.stage_module._utc();selection=runner.root/'stages/HaplotypeCaller_selection'
    try:
        options=default_command(serial,bam=bam,reference=reference,output=outputs[0],threads=threads,scope=scope)
        budget,why=select_workers(sorted(runner.cpuset),runner.budget/GIB,options['-Xmx'],requested_workers)
        if budget is None:raise Unsupported(why)
        need(options['-Xmx']==budget['heap_gib_per_worker'],'planned HC heap differs from actual -Xmx',BindingError)
        dictionary=read_dictionary(str(reference)+'.fai')
        need(all(not re.search(r'[\s\x00-\x1f\x7f]',n) for n,length in dictionary),'unsupported whitespace/control in sequence dictionary',Unsupported)
        need(len(dictionary)>=budget['workers'],'not enough complete reference contigs',Unsupported)
    except Unsupported as unsupported:
        return original_serial_fallback(runner,api,serial,outputs,str(unsupported))
    need(guard_assets is not None and bgzip_binding is not None and tabix_binding is not None,'eligible parallel HC requires committed guard and original compression/index tools',BindingError)
    from .build_preprocessing_hc_guard import guard_scan_argv,verify_hc_guard_assets
    guard_assets=verify_hc_guard_assets(guard_assets,conda_prefix=runner.prefix)
    bam_binding=api.bound(bam);fai_binding=api.bound(str(reference)+'.fai');dict_binding=api.bound(Path(reference).with_suffix('.dict'))
    request={'serial_command':serial,'budget':budget,'raw_scientific_risk_threshold':50,
             'scheduling_cost_bases_per_filter_passing_read':READ_COST_BASES,
             'guard_builder_sha256':guard_assets['builder_sha256'],
             'guard_build_receipt_sha256':guard_assets['build_receipt_sha256'],
             'original_gatk_jar_sha256':guard_assets['sha256']['gatkJar'],
             'bam':bam_binding,'fai':fai_binding,'reference_dictionary':dict_binding,'bgzip':bgzip_binding,'tabix':tabix_binding}
    request=json.loads(json.dumps(request))
    top=runner.root/'stages/HaplotypeCaller';complete=top/'complete.private.json'
    if complete.is_file():
        saved=json.loads(complete.read_text());need(saved.get('request')==request and saved.get('status')=='complete','committed HC parallel request differs',BindingError)
        actual_count=saved.get('separate_original_JVMs')
        need(type(actual_count) is int and 2<=actual_count<=budget['workers'] and type(saved.get('exit_codes')) is list and len(saved['exit_codes'])==actual_count and all(type(x) is int and x==0 for x in saved['exit_codes']),'HC worker actual exits incomplete',BindingError)
        successful_waits(saved.get('actual_waits'),actual_count)
        need(len(saved.get('outputs',[]))==2 and [x['path'] for x in saved['outputs']]==[str(Path(p).resolve()) for p in outputs],'committed HC output bindings/roles incomplete',BindingError)
        for bound_output in saved['outputs']:api.verify(bound_output)
        for role in ('guard_receipt','decision','merge'):api.verify(saved.get(role))
        stages=saved.get('aggregates_stage_bindings')
        stage_names=('HaplotypeCaller_guard','HaplotypeCaller_shards','HaplotypeCaller_bgzip','HaplotypeCaller_tabix','HaplotypeCaller_index_names','HaplotypeCaller_index_full_readback')
        need(type(stages) is list and len(stages)==len(stage_names),'HC aggregate constituent stage closure is incomplete',BindingError)
        constituent_events=[]
        for name,binding in zip(stage_names,stages):
            source=api.verify(binding);need(source==runner.root/'stages'/name/'complete.private.json','HC aggregate stage role differs',BindingError)
            row=json.loads(source.read_text());codes=row.get('exit_codes');commands=row.get('commands')
            need(row.get('stage')==name and row.get('status')=='complete' and type(commands) is list and type(codes) is list and len(commands)==len(codes) and codes and all(type(x) is int and x==0 for x in codes),'HC constituent actual wait record is incomplete',BindingError)
            constituent_events.append(row)
        verification=saved.get('verification');need(type(verification) is dict,'HC readback proof missing',BindingError)
        for role in ('gvcf','tbi','index_names','full_index_readback'):api.verify(verification.get(role))
        if verification.get('query_regions') is not None:api.verify(verification['query_regions'])
        old_decision=json.loads(Path(saved['decision']['path']).read_text())
        old_merge=json.loads(Path(saved['merge']['path']).read_text())
        need(old_decision.get('request')==request and old_decision.get('selected')=='original_HC_whole_contig_JVMs','committed HC decision differs',BindingError)
        for binding in old_merge.get('shard_output_bindings',[]):api.verify(binding)
        need(len(old_merge.get('shard_output_bindings',[]))==actual_count,'committed HC shard closure incomplete',BindingError)
        api.verify(old_merge.get('plain_output'))
        interval_bindings=old_decision.get('interval_lists')
        need(type(interval_bindings) is list and len(interval_bindings)==actual_count,'committed whole-contig scope bindings incomplete',BindingError)
        for binding in interval_bindings:api.verify(binding)
        expected_commands,_=worker_commands(serial,old_decision['groups'],old_decision['budget'],runner.root/'files/hc_contig_shards',interval_files=[b['path'] for b in interval_bindings])
        need(saved.get('commands')==expected_commands and len(expected_commands)==actual_count,'committed HC actual command list differs',BindingError)
        need(type(saved.get('started_utc')) is str and type(saved.get('completed_utc')) is str,'committed HC stage clock is incomplete',BindingError)
        # Resume restores the same actual phase ledger as a fresh run. No
        # process is launched; source/output/wait receipts were reverified above.
        for row in (*constituent_events,saved):
            _verify_completed_journal(runner,row)
            previous=[x for x in runner.events if x.get('stage')==row['stage']]
            need(not previous or (len(previous)==1 and previous[0]==row),'existing HC phase ledger differs',BindingError)
            if not previous:runner.events.append(row)
        return {'used_parallel':True,'event':saved,'resumed_committed_stage':True}
    need(not top.exists(),'incomplete parallel HC preserved; use a new output directory',BindingError)
    selection.mkdir(parents=True,exist_ok=True);request_path=selection/'request.private.json'
    if request_path.exists():need(json.loads(request_path.read_text())==request,'HC decision request changed',BindingError)
    else:api.atomic(request_path,request)
    command_path=selection/'original_hc_command.private.json'
    if command_path.exists():need(json.loads(command_path.read_text())==serial,'guard original command changed',BindingError)
    else:api.atomic(command_path,serial)
    certificate=selection/'mapped_start_guard.private.json'
    guard=guard_scan_argv(guard_assets,bam=bam,fai=str(reference)+'.fai',hc_command_json=command_path,
                          output_json=certificate,conda_prefix=runner.prefix)
    guard_stage=runner.stage('HaplotypeCaller_guard',[guard],[certificate],3)
    need(guard_stage.get('status')=='complete' and guard_stage.get('exit_codes')==[0],'guard has no successful actual process wait',BindingError)
    need(certificate.stat().st_size<=64*2**20,'guard certificate exceeds metadata bound',BindingError)
    cert=json.loads(certificate.read_text());command_bound=api.bound(command_path)
    rows=guard_rows(cert,dictionary,expected_bam=bam_binding,expected_fai_sha=fai_binding['sha256'],
                    expected_command_sha=command_bound['sha256'],expected_jar_sha=guard_assets['sha256']['gatkJar'])
    # Revalidate output+source assets after the complete scan, not just its JSON.
    verify_hc_guard_assets(guard_assets,conda_prefix=runner.prefix)
    for item in guard_stage['outputs']:api.verify(item)
    need(api.identity(bam)=={k:v for k,v in bam_binding.items() if k!='sha256'},'scanned BAM identity changed',BindingError)
    # A conservative raw-risk plan can leave fewer than the requested workers.
    # Auto may reduce worker count, explicit parallel requests fail/fallback to
    # the original serial command without skipping any contig.
    groups=None
    for count in range(budget['workers'],1,-1):
        if requested_workers is not None and count!=requested_workers:continue
        try:
            groups=partition_contigs(rows,count)
            budget=budget_plan(sorted(runner.cpuset),runner.budget/GIB,options['-Xmx'],count)
            break
        except Unsupported:pass
    if groups is None:return original_serial_fallback(runner,api,serial,outputs,'whole-contig raw-risk partition cannot populate requested JVMs')
    shard_root=runner.root/'files/hc_contig_shards'
    interval_bindings=write_interval_lists(groups,dictionary,dict_binding['path'],shard_root,api)
    interval_files=[b['path'] for b in interval_bindings]
    commands,pairs=worker_commands(serial,groups,budget,shard_root,interval_files=interval_files)
    decision={'selected':'original_HC_whole_contig_JVMs','groups':groups,'budget':budget,
              'guard':api.bound(certificate),'guard_actual_wait_stage':guard_stage,'interval_lists':interval_bindings,
              'request':request,'cost_heuristic_bases_per_filter_passing_read':READ_COST_BASES,
              'no_scientific_threshold_or_contig_skipped':True,
              'guard_does_not_prove_universal_RNG_independence':True,
              'scientific_performance_gate_status':'runtime_contracts_only_real_data_validation_recorded_separately'}
    decision_path=selection/'decision.private.json'
    # JSON roundtrip makes tuples stable across resume.
    decision=json.loads(json.dumps(decision))
    if decision_path.exists():need(json.loads(decision_path.read_text())==decision,'whole-contig decision differs',BindingError)
    else:api.atomic(decision_path,decision)
    for index in range(len(groups)):(shard_root/('worker_'+str(index))/'tmp').mkdir(parents=True,exist_ok=True)
    flat_outputs=[p for pair in pairs for p in pair]
    parallel=parallel_stage(runner,api,'HaplotypeCaller_shards',commands,flat_outputs,budget)
    # Keep the actual scientific phase for observer joins. The final HC event
    # is explicitly an aggregate, not another launch or additive stage clock.
    closed_by_path={x['path']:x for x in parallel['outputs']}
    shard_bindings=[closed_by_path[str(p[0].resolve())] for p in pairs]
    plain=Path(str(outputs[0])[:-3]);merge_receipt=selection/'merge.private.json'
    if merge_receipt.exists():
        merged=json.loads(merge_receipt.read_text());need(merged.get('groups')==[list(g) for g in groups] and merged.get('shard_output_bindings')==shard_bindings,'merge inputs differ',BindingError)
        api.verify(merged['plain_output'])
    else:
        merged=merge_shards([p[0] for p in pairs],groups,dictionary,plain,
                            tick=lambda:monitor(runner,api,[],1),api=api,output_bindings=shard_bindings,interval_arguments=interval_files)
        merged.update(groups=groups,shard_output_bindings=shard_bindings,plain_output=api.bound(plain))
        api.atomic(merge_receipt,merged)
    verification=compress_index_readback(runner,api,plain,outputs[0],merged['decoded_sha256'],merged['observed_contigs'],
                                          bgzip_binding=bgzip_binding,tabix_binding=tabix_binding,dictionary=dictionary)
    need([Path(x).resolve() for x in outputs]==[Path(verification['gvcf']['path']),Path(verification['tbi']['path'])],'final HC output roles differ',BindingError)
    stage_names=('HaplotypeCaller_guard','HaplotypeCaller_shards','HaplotypeCaller_bgzip','HaplotypeCaller_tabix','HaplotypeCaller_index_names','HaplotypeCaller_index_full_readback')
    aggregate_bindings=[api.bound(runner.root/'stages'/name/'complete.private.json') for name in stage_names]
    top.mkdir();event={'stage':'HaplotypeCaller','status':'complete','commands':commands,'request':request,
        'requested_original_serial_command':serial,'started_utc':started_utc,
        'is_aggregate_stage':True,'is_aggregate_stage_including_guard_merge_readback':True,
        'declared_no_additional_process_launch':True,'aggregates_stage_bindings':aggregate_bindings,
        'aggregate_and_constituent_wall_times_must_not_be_added':True,
        'actual_scientific_commands_executed_once':True,
        'actual_waits':parallel['actual_waits'],'exit_codes':parallel['exit_codes'],'outputs':[verification['gvcf'],verification['tbi']],
        'guard_receipt':api.bound(certificate),'decision':api.bound(decision_path),'merge':api.bound(merge_receipt),
        'verification':verification,'wall_seconds':time.monotonic()-began,
        'peak_sampled_pss_bytes':parallel['peak_sampled_pss_bytes'],
        'separate_original_JVMs':len(groups),'all_guard_merge_index_verification_costs_included':True,
        'extra_IO_plain_merged_file_and_full_index_readback_log_retained':True,
        'whole_contig_parallel_scientific_equivalence_not_proved_by_contracts':True,
        'completed_utc':api.stage_module._utc()}
    api.atomic(complete,event);_append_completed_journal(runner,event);runner.events.append(event)
    return {'used_parallel':True,'event':event,'resumed_committed_stage':False}
