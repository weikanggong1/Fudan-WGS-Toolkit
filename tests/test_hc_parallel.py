"""Small format/planning/lifecycle contracts, not performance or science tests."""
import copy,gzip,hashlib,json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from fudan_wgs_toolkit import _preprocessing_hc_parallel as m
D=(('A',20),('B',40),('C',30),('D',50))
def counts():return [dict(contig=n,length=l,mapped_all_including_duplicates=x,max_mapped_at_alignment_start=y,raw_starts_over_50=int(y>50)) for (n,l),x,y in zip(D,(100,20,200,10),(60,10,80,5))]
def header(path,contigs):
 c='HaplotypeCaller --output '+str(path)+''.join(' --intervals '+n for n in contigs)+' --input /fixture/input.bam --sample-ploidy 2'
 return (b'##fileformat=VCFv4.2\n'+b''.join(('##contig=<ID='+n+',length='+str(l)+'>\n').encode() for n,l in D)+('##GATKCommandLine=<ID=HaplotypeCaller,CommandLine="'+c+'",Version="4.2.0.0",Date="January 1, 2000 12:00:00 AM UTC">\n').encode()+b'#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tfixture\n')
def row(contig,pos=1,end=3):return (contig+'\t'+str(pos)+'\t.\tA\t<NON_REF>\t.\t.\tEND='+str(end)+'\tGT:DP:GQ:PL\t0/0:0:0:0,0,0\n').encode()
def write(path,groups,rows):
 with gzip.open(path,'wb') as out:out.write(header(path,groups));out.write(b''.join(rows))
class Contracts(unittest.TestCase):
 def test_shared_budget_and_CPU_partition(self):
  x=m.budget_plan(tuple(range(8)),32,8,2);self.assertEqual(x['prospective_total_gib'],21);self.assertFalse(set(x['cpu_sets'][0])&set(x['cpu_sets'][1]));self.assertEqual(set().union(*map(set,x['cpu_sets'])),set(range(8)))
  with self.assertRaises(m.Unsupported):m.budget_plan(tuple(range(8)),16,8,2)
 def test_raw_risk_subsequence_safe_can_join_risk_group(self):
  groups=m.partition_contigs(counts(),2);risk={'A','C'};self.assertEqual(tuple(n for n in groups[0] if n in risk),('A','C'));self.assertEqual(set(n for g in groups for n in g),set('ABCD'));self.assertIn('B',groups[0])
 def test_zero_read_long_contig_has_real_cost(self):
  rows=[dict(contig=n,length=l,mapped_all_including_duplicates=0,max_mapped_at_alignment_start=0,raw_starts_over_50=0) for n,l in [('long',1000),('x',10),('y',10)]]
  g=m.partition_contigs(rows,2);self.assertEqual(g[0],('long',));self.assertEqual(g[1],('x','y'))
 def test_unsupported_scope_original_command_not_skipped(self):
  with tempfile.TemporaryDirectory() as d:
   calls=[];r=types.SimpleNamespace(root=Path(d),budget=8*m.GIB,stage=lambda *a:calls.append(a) or {'actual_exit':[0]});api=types.SimpleNamespace(atomic=lambda p,v:p.write_text(json.dumps(v)))
   out=m.original_serial_fallback(r,api,['unchanged','original'],['output'],'unsupported_scope');self.assertFalse(out['used_parallel']);self.assertEqual(calls[0][1],[['unchanged','original']])
 def test_merge_interleaved_groups_keeps_all_raw_headers_body(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a.gz';q=Path(d)/'b.gz';write(p,('A','C'),[row('A'),row('C')]);write(q,('B','D'),[row('B'),row('D')]);out=Path(d)/'merged.vcf';x=m.merge_shards([p,q],[('A','C'),('B','D')],D,out)
   text=out.read_bytes();self.assertIn(header(p,('A','C')).split(b'##GATKCommandLine=')[1].split(b'\n')[0],text);self.assertIn(header(q,('B','D')).split(b'##GATKCommandLine=')[1].split(b'\n')[0],text);self.assertEqual([l.split(b'\t')[0] for l in text.splitlines() if not l.startswith(b'#')],[b'A',b'B',b'C',b'D']);self.assertEqual(x['records'],4)
 def test_partition_omission_and_duplicate_rejected(self):
  with self.assertRaises(m.BindingError):m.merge_shards(['a','b'],[('A','C'),('B',)],D,'never_created')
  with self.assertRaises(m.BindingError):m.merge_shards(['a','b'],[('A','C'),('B','A')],D,'never_created')
 def test_truncated_gzip_and_order_decrease_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a.gz';q=Path(d)/'b.gz';write(p,('A','C'),[row('C'),row('A')]);write(q,('B','D'),[row('B'),row('D')]);out=Path(d)/'out.vcf'
   with self.assertRaises(m.BindingError):m.merge_shards([p,q],[('A','C'),('B','D')],D,out)
   write(p,('A','C'),[row('A'),row('C')]);p.write_bytes(p.read_bytes()[:-6]);out=Path(d)/'out2.vcf'
   with self.assertRaises((EOFError,OSError)):m.merge_shards([p,q],[('A','C'),('B','D')],D,out)
 def test_foreign_contig_and_END_and_header_change_rejected(self):
  for bad in [row('B'),row('A',1,99)]:
   with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'a.gz';q=Path(d)/'b.gz';write(p,('A','C'),[bad]);write(q,('B','D'),[row('B'),row('D')])
    with self.assertRaises(m.BindingError):m.merge_shards([p,q],[('A','C'),('B','D')],D,Path(d)/'out.vcf')
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a.gz';q=Path(d)/'b.gz';write(p,('A','C'),[row('A'),row('C')]);write(q,('B','D'),[row('B'),row('D')])
   raw=gzip.decompress(q.read_bytes()).replace(b'--sample-ploidy 2',b'--sample-ploidy 3');q.write_bytes(gzip.compress(raw))
   with self.assertRaises(m.BindingError):m.merge_shards([p,q],[('A','C'),('B','D')],D,Path(d)/'out.vcf')
 def test_released_PID_is_never_queried(self):
  p=types.SimpleNamespace(pid=123);lease=m.Lease(p,os.getuid(),1,123,True)
  with patch.object(m,'_proc_snapshot',side_effect=AssertionError('released pid queried')):
   with self.assertRaises(m.BindingError):lease.verify()
 def test_unknown_identity_refuses_member_query(self):
  p=types.SimpleNamespace(pid=123);lease=m.Lease(p,os.getuid(),1,123);api=types.SimpleNamespace(stage_module=types.SimpleNamespace(_group_members=lambda _:(_ for _ in ()).throw(AssertionError('must not enumerate unknown session'))))
  with patch.object(m,'_proc_snapshot',return_value=dict(uid=os.getuid(),start=2,sid=123,state='S')):
   with self.assertRaises(m.BindingError):m.owned_members(lease,api)
 def test_batch_releases_finished_PID_and_cleans_every_active_worker(self):
  for failure in (False,True):
   with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
    state={};objects={};cleanup=[];queried=[]
    class Process:
     def __init__(self,*args,**kwargs):
      self.pid=1201+len(objects);self.returncode=None;self.waits=0;objects[self.pid]=self;state[self.pid]='S'
      if self.pid==1202:state[1201]='Z'
     def wait(self,timeout=None):
      self.waits+=1
      if self.returncode is None:
       if state[self.pid]!='Z':raise TimeoutError('live fake root was waited prematurely')
       self.returncode=2 if failure and self.pid==1201 else -15 if self.pid in cleanup else 0
      return self.returncode
    def snapshot(pid):
     if objects[pid].returncode is not None:raise AssertionError('released root PID was queried')
     return dict(uid=os.getuid(),start=pid*10,sid=pid,state=state[pid])
    def members(pid):
     if objects[pid].returncode is not None:raise AssertionError('released root SID was queried')
     queried.append(pid);return [] if state[pid]=='Z' else [pid]
    def pss(pids):state[1202]='Z';return 1024
    def clean(p):cleanup.append(p.pid);state[p.pid]='Z'
    module=types.SimpleNamespace(_limits=lambda:dict(admission_available_bytes=100*m.GIB),_utc=lambda:'fixture',_descendants=lambda p:{p},_group_members=members,_pss=pss,_cleanup_stage=clean)
    from fudan_wgs_toolkit import _preprocessing_io as real_io
    io=types.SimpleNamespace(bound=real_io.bound,atomic=lambda p,v:Path(p).write_text(json.dumps(v)),identity=real_io.identity)
    api=m.Api37.checked(module,io);runner=types.SimpleNamespace(root=Path(directory),reserve=10*m.GIB,budget=32*m.GIB,cpuset=set(range(8)),env={},events=[])
    outputs=[Path(directory)/'one',Path(directory)/'two']
    for output in outputs:output.write_bytes(b'fixture')
    with patch.object(m.subprocess,'Popen',Process),patch.object(m,'_proc_snapshot',snapshot),patch.object(m,'_strict_session_members',lambda lease:{pid:snapshot(pid) for pid in members(lease.sid)}),patch.object(m,'cleanup_owned',lambda lease,api:clean(lease.process)):
     if failure:
      with self.assertRaisesRegex(RuntimeError,'one original HC worker'):m.parallel_stage(runner,api,'HC',[['one'],['two']],outputs,m.budget_plan(range(8),32,8,2),poll_seconds=0)
      result=json.loads((Path(directory)/'stages/HC/failed.private.json').read_text());self.assertEqual(result['exit_codes'],[2,-15]);self.assertEqual(result['unreleased_lease_count'],0);self.assertEqual(cleanup,[1202]);self.assertFalse(result['cleanup_errors'])
     else:
      result=m.parallel_stage(runner,api,'HC',[['one'],['two']],outputs,m.budget_plan(range(8),32,8,2),poll_seconds=0);self.assertEqual(result['exit_codes'],[0,0]);self.assertEqual(cleanup,[])
      journal=Path(directory)/'journal.private.jsonl';before=journal.read_bytes();self.assertEqual(len(before.splitlines()),1)
      with patch.object(m.subprocess,'Popen',side_effect=AssertionError('resume must not launch')):
       resumed=m.parallel_stage(runner,api,'HC',[['one'],['two']],outputs,m.budget_plan(range(8),32,8,2),poll_seconds=0)
      self.assertEqual(resumed,json.loads(json.dumps(result)));self.assertEqual(journal.read_bytes(),before)
    self.assertEqual([objects[x].waits for x in sorted(objects)],[1,1])

class HaplotypeCallerContracts(unittest.TestCase):
 def test_complete_interval_lists_preserve_colon_asterisk_and_bound_scope(self):
  from fudan_wgs_toolkit import _preprocessing_io as io
  api=types.SimpleNamespace(bound=io.bound,verify=lambda b:self.assertEqual(io.bound(b['path']),b))
  dictionary=(('main',100),('ALT*01:01:01',40),('ALT*01:01:01:01',30))
  groups=(('main','ALT*01:01:01:01'),('ALT*01:01:01',))
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);ref=root/'reference.dict';header=b'@HD\tVN:1.6\tSO:unsorted\n'+b''.join((' @SQ\tSN:'+n+'\tLN:'+str(l)+'\n').lstrip().encode() for n,l in dictionary);ref.write_bytes(header)
   bindings=m.write_interval_lists(groups,dictionary,ref,root/'shards',api)
   self.assertEqual(len(bindings),2)
   for group,binding in zip(groups,bindings):
    raw=Path(binding['path']).read_bytes();self.assertTrue(raw.startswith(header));data=raw[len(header):].decode().splitlines()
    self.assertEqual([x.split('\t')[0] for x in data],list(group))
    self.assertEqual([int(x.split('\t')[2]) for x in data],[dict(dictionary)[n] for n in group])
   serial=['gatk','--java-options','-Xmx8g -Xms8g','HaplotypeCaller','-O','out']
   commands,_=m.worker_commands(serial,groups,m.budget_plan(range(8),32,8,2),root/'shards',interval_files=[x['path'] for x in bindings])
   for command,binding in zip(commands,bindings):self.assertEqual(command[-2:],['-L',binding['path']])
   # Original expanded HC headers must show the actual interval file, not a
   # fabricated literal contig request. Group scope is separately file-bound.
   line=('##GATKCommandLine=<ID=HaplotypeCaller,CommandLine="HaplotypeCaller --output out --intervals '+bindings[0]['path']+' --sample-ploidy 2",Version="4.2.0.0",Date="January 1, 2000 12:00:00 AM UTC">\n').encode()
   m.header_command(line,'out',[bindings[0]['path']])
   with self.assertRaises(m.BindingError):m.header_command(line,'out',groups[0])
   Path(bindings[0]['path']).write_bytes(b'changed')
   with self.assertRaises(m.BindingError):m.write_interval_lists(groups,dictionary,ref,root/'shards',api)

 def test_python_failure_is_explicit_and_original_error_preserved(self):
  for recording_fails in (False,True):
   with self.subTest(recording_fails=recording_fails),tempfile.TemporaryDirectory() as d:
    runner=types.SimpleNamespace(root=Path(d))
    def atomic(path,value):
     if recording_fails:raise OSError('fixture metadata disk failure')
     Path(path).write_text(json.dumps(value))
    api=types.SimpleNamespace(atomic=atomic,stage_module=types.SimpleNamespace(_utc=lambda:'fixture'))
    @m._record_hc_failure
    def merge_failure(runner,api):raise EOFError('fixture incomplete gzip merge')
    with self.assertRaisesRegex(EOFError,'fixture incomplete gzip merge'):merge_failure(runner,api)
    self.assertFalse((Path(d)/'stages/HaplotypeCaller/complete.private.json').exists())
    if not recording_fails:
     receipts=list((Path(d)/'stages/HaplotypeCaller_selection').glob('failure_attempt_*.private.json'))
     self.assertEqual(len(receipts),1);saved=json.loads(receipts[0].read_text());self.assertEqual(saved['error_type'],'EOFError');self.assertEqual(saved['status'],'failed_HC_candidate_attempt_not_complete')

 def test_actual_journal_append_flush_fsync_and_resume_verify(self):
  with tempfile.TemporaryDirectory() as d:
   runner=types.SimpleNamespace(root=Path(d));row=dict(stage='HaplotypeCaller',status='complete',commands=[['actual_fixture']],exit_codes=[0],is_aggregate_stage=True)
   with patch.object(m.os,'fsync',wraps=os.fsync) as sync:m._append_completed_journal(runner,row);self.assertEqual(sync.call_count,1)
   before=(Path(d)/'journal.private.jsonl').read_bytes();self.assertEqual(json.loads(before),row)
   m._verify_completed_journal(runner,row);self.assertEqual((Path(d)/'journal.private.jsonl').read_bytes(),before)
   with self.assertRaises(m.BindingError):m._append_completed_journal(runner,row)
 def test_missing_duplicate_or_incomplete_journal_is_never_repaired(self):
  with tempfile.TemporaryDirectory() as d:
   runner=types.SimpleNamespace(root=Path(d));row=dict(stage='HaplotypeCaller',status='complete',commands=[['actual_fixture']],exit_codes=[0]);path=Path(d)/'journal.private.jsonl'
   for bad in (b'',(json.dumps(row)+'\n'+json.dumps(row)+'\n').encode(),json.dumps(row).encode()):
    path.write_bytes(bad)
    with self.assertRaises(m.BindingError):m._verify_completed_journal(runner,row)
    self.assertEqual(path.read_bytes(),bad)

 def test_journal_total_file_bytes_are_bounded_before_read(self):
  with tempfile.TemporaryDirectory() as d:
   runner=types.SimpleNamespace(root=Path(d));path=Path(d)/'journal.private.jsonl'
   with path.open('wb') as stream:stream.truncate(64*2**20+1)
   with self.assertRaisesRegex(m.BindingError,'total metadata byte bound'):m._journal_records(runner)
   self.assertEqual(path.stat().st_size,64*2**20+1)

 def test_append_total_bound_refuses_without_changing_actual_file(self):
  with tempfile.TemporaryDirectory() as d:
   runner=types.SimpleNamespace(root=Path(d));path=Path(d)/'journal.private.jsonl'
   old=dict(stage='earlier',status='complete',padding='x'*120);before=(json.dumps(old)+'\n').encode();path.write_bytes(before)
   row=dict(stage='HaplotypeCaller',status='complete',commands=[['actual_fixture']],exit_codes=[0])
   # Scale only the policy constant to exercise the same actual file writer.
   # The separate sparse-file contract checks the real64MiB threshold.
   with patch.object(m,'JOURNAL_MAX_BYTES',len(before)+1):
    with self.assertRaisesRegex(m.BindingError,'append exceeds total metadata byte bound'):m._append_completed_journal(runner,row)
   self.assertEqual(path.read_bytes(),before)

 def test_auto_sums_heaps_nonheap_controller(self):
  self.assertEqual(m.select_workers(range(8),32,8,None)[0]['workers'],3)
  self.assertEqual(m.select_workers(range(8),28,8,None)[0]['workers'],2)
  self.assertIsNone(m.select_workers(range(8),20,8,None)[0])
  self.assertIsNone(m.select_workers(range(8),32,8,1)[0])
  with self.assertRaises(m.Unsupported):m.select_workers(range(8),20,8,2)
  for bad in (True,0,4,-1):
   with self.assertRaises(ValueError):m.select_workers(range(8),32,8,bad)
 def test_true_runner_command_and_environment_overrides(self):
  from fudan_wgs_toolkit import single_preprocessing as source
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);config=dict(cpu_affinity=list(range(8)),host_memory_limit_gib=32,host_memory_reserve_gib=10)
   with patch.dict(os.environ,{'GATK_LOCAL_JAR':'untrusted','GATK_SPARK_JAR':'untrusted'}):
    runner=source._Runner(root,Path('/fixture/env'),config,False)
   self.assertNotIn('GATK_LOCAL_JAR',runner.env);self.assertNotIn('GATK_SPARK_JAR',runner.env);self.assertEqual(runner.env['OMP_NUM_THREADS'],'1')
   bam='/fixture/recal.bam';ref='/fixture/ref.fa';out='/fixture/sample.g.vcf.gz'
   serial=runner.gatk('HaplotypeCaller','-R',ref,'-I',bam,'-O',out,'--emit-ref-confidence','GVCF','--native-pair-hmm-threads','8',heap_gib=8)
   # Exact production callsite appends Xms after Runner.gatk.
   serial[2]+=' -Xms8g';before=copy.deepcopy(serial)
   opts=m.default_command(serial,bam=bam,reference=ref,output=out,threads=8,scope=m.Scope())
   budget=m.select_workers(range(8),32,opts['-Xmx'])[0]
   commands,_=m.worker_commands(serial,[('A',),('B',),('C',)],budget,root/'shards')
   self.assertEqual(serial,before)
   for command in commands:
    self.assertIn('-Xmx8g',command[2]);self.assertIn('-Xms8g',command[2]);self.assertEqual(command[command.index('--native-pair-hmm-threads')+1],'8')
   self.assertEqual(budget['heap_gib_per_worker'],opts['-Xmx'])
 def test_empty_asset_binding_rejected_before_any_tool(self):
  from fudan_wgs_toolkit.build_preprocessing_hc_guard import verify_hc_guard_assets
  with patch.object(m.subprocess,'Popen',side_effect=AssertionError('no tool allowed')):
   for bad in ({},{'schema':'fudan-hc-contig-guard-assets-v1'}):
    with self.assertRaises(RuntimeError):verify_hc_guard_assets(bad)
 def test_unknown_member_never_signalled(self):
  lease=m.Lease(types.SimpleNamespace(pid=123),os.getuid(),1230,123)
  before=dict(uid=os.getuid(),start=2340,sid=123,state='S')
  def snapshot(pid):
   return dict(uid=os.getuid(),start=(1230 if pid==123 else 999),sid=123,state='S')
  with patch.object(m,'_strict_session_members',return_value={234:before}),patch.object(m,'_proc_snapshot',side_effect=snapshot),patch.object(m.os,'kill',side_effect=AssertionError('unknown PID must not signal')):
   with self.assertRaises(m.BindingError):m.cleanup_owned(lease,None,grace_seconds=0)
 def test_permission_is_unknown_not_drained(self):
  lease=m.Lease(types.SimpleNamespace(pid=123),os.getuid(),1230,123)
  root=types.SimpleNamespace(name='234',stat=lambda:types.SimpleNamespace(st_uid=os.getuid()))
  def snapshot(pid):
   if pid==123:return dict(uid=os.getuid(),start=1230,sid=123,state='S')
   raise PermissionError('fixture')
  with patch.object(m.Path,'iterdir',return_value=iter([root])),patch.object(m,'_proc_snapshot',side_effect=snapshot):
   with self.assertRaises(m.BindingError):m.owned_members(lease,None)

if __name__=='__main__':unittest.main(verbosity=2)
