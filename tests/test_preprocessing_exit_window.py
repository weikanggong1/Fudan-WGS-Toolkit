"""Generated proc-format rejection contracts only; not a scientific benchmark."""
from pathlib import Path
import contextlib,errno,os,tempfile,unittest
from unittest.mock import patch
from fudan_wgs_toolkit import phewas_resources as p
from fudan_wgs_toolkit import single_preprocessing as sp
# Synthetic identifiers written only to a temporary proc-format fixture.
BOOT='00000000-0000-0000-0000-000000000001'
def stat(pid=100,state='D',flags=4,threads=1,start=20):
 f=[state]+['0']*20;f[6]=str(flags);f[17]=str(threads);f[19]=str(start);return f'{pid} (fixture name) '+' '.join(f)+'\n'
class Clock:
 def __init__(self):self.at=0
 def monotonic(self):return self.at
 def sleep(self,s):self.at+=s
class Contracts(unittest.TestCase):
 @contextlib.contextmanager
 def proc(self):
  with tempfile.TemporaryDirectory()as d:
   r=Path(d);q=r/'100';q.mkdir();(q/'task/100').mkdir(parents=True);(r/'sys/kernel/random').mkdir(parents=True);(r/'sys/kernel/random/boot_id').write_text(BOOT+'\n');(q/'stat').write_text(stat());(q/'status').write_text(f'Name:\tfixture\nState:\tD (disk sleep)\nTgid:\t100\nPid:\t100\nUid:\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\nThreads:\t1\n');(q/'statm').write_text('0 0 0 0 0 0 0\n');(q/'smaps').write_bytes(b'');(q/'maps').write_bytes(b'');yield r,q
 def test_complete_conjunction_classifies_without_zero_or_exit_claim(self):
  with self.proc()as(r,q):v=p._exiting_no_attached_mm(100,proc_root=r)
  self.assertEqual(v['classification'],'verified_exiting_no_attached_mm');self.assertEqual(v['task_TIDs'],[100]);self.assertFalse(v['process_exit_confirmed']);self.assertFalse(v['lease_release_authorized']);self.assertFalse(v['PSS_zero_inferred']);self.assertEqual(v['physical_teardown_memory_status'],'unknown_not_measured')
 def test_R_D_S_states_allowed_with_all_other_predicates(self):
  for state in ('R','D','S'):
   with self.subTest(state=state),self.proc()as(r,q):
    (q/'stat').write_text(stat(state=state));self.assertIsNotNone(p._exiting_no_attached_mm(100,proc_root=r))
 def test_pf_exiting_alone_and_kthread_rejected(self):
  for flags in (0,0x200000,0x200004):
   with self.subTest(flags=flags),self.proc()as(r,q):
    (q/'stat').write_text(stat(flags=flags));self.assertIsNone(p._exiting_no_attached_mm(100,proc_root=r))
 def test_present_VmRSS_zero_is_not_no_mm(self):
  for field in ('VmRSS:\t0 kB\n','VmSize:\t0 kB\n','RssAnon:\t0 kB\n','HugetlbPages:\t0 kB\n'):
   with self.subTest(field=field),self.proc()as(r,q):
    with(q/'status').open('a')as f:f.write(field)
    self.assertIsNone(p._exiting_no_attached_mm(100,proc_root=r))
 def test_nonzero_or_malformed_statm_rejected(self):
  for fields in ('1 0 0 0 0 0 0\n','0 1 0 0 0 0 0\n','0 0 0 0 0 0\n','0 0 0 0 0 0 0 0\n','0 0 x 0 0 0 0\n'):
   with self.subTest(fields=fields),self.proc()as(r,q):
    (q/'statm').write_text(fields);self.assertIsNone(p._exiting_no_attached_mm(100,proc_root=r))
 def test_any_smaps_or_maps_bytes_rejected(self):
  for name in ('smaps','maps'):
   for data in (b'\n',b'0',b'\x00'):
    with self.subTest(name=name,data=data),self.proc()as(r,q):
     (q/name).write_bytes(data);self.assertIsNone(p._exiting_no_attached_mm(100,proc_root=r))
 def test_second_TID_or_num_threads_rejected(self):
  with self.proc()as(r,q):
   (q/'task/101').mkdir();self.assertIsNone(p._exiting_no_attached_mm(100,proc_root=r))
  with self.proc()as(r,q):
   (q/'stat').write_text(stat(threads=2));self.assertIsNone(p._exiting_no_attached_mm(100,proc_root=r))
  with self.proc()as(r,q):
   x=(q/'status').read_text();(q/'status').write_text(x.replace('Threads:\t1','Threads:\t2'));self.assertIsNone(p._exiting_no_attached_mm(100,proc_root=r))
 def test_missing_or_truncated_status_not_accepted(self):
  for change in ('missing','no_newline','duplicate','bad_uid'):
   with self.subTest(change=change),self.proc()as(r,q):
    text=(q/'status').read_text()
    if change=='missing':(q/'status').unlink()
    elif change=='no_newline':(q/'status').write_text(text.rstrip('\n'))
    elif change=='duplicate':(q/'status').write_text(text+'Threads:\t1\n')
    else:(q/'status').write_text(text.replace('Uid:', 'OtherUid:'))
    self.assertIsNone(p._exiting_no_attached_mm(100,proc_root=r))
 def test_permission_error_is_unknown(self):
  with self.proc()as(r,q),patch.object(Path,'open',side_effect=PermissionError(errno.EACCES,'denied')):self.assertIsNone(p._exiting_no_attached_mm(100,proc_root=r))
 def test_uid_mismatch_is_unknown(self):
  uid=os.getuid()
  with self.proc()as(r,q),patch.object(p.os,'getuid',return_value=uid+1):self.assertIsNone(p._exiting_no_attached_mm(100,proc_root=r))
 def test_pid_or_start_changed_during_probe_rejected(self):
  original=Path.open
  for change in ('pid','start'):
   with self.subTest(change=change),self.proc()as(r,q):
    calls=[0]
    def opened(path,*a,**kw):
     if path==q/'stat':
      calls[0]+=1
      if calls[0]==2:
       import io
       return io.BytesIO(stat(pid=101 if change=='pid'else 100,start=21 if change=='start'else 20).encode())
     return original(path,*a,**kw)
    with patch.object(Path,'open',opened):self.assertIsNone(p._exiting_no_attached_mm(100,proc_root=r))
 def test_boot_change_during_probe_rejected(self):
  original=Path.open
  with self.proc()as(r,q):
   calls=[0]
   def opened(path,*a,**kw):
    if path==r/'sys/kernel/random/boot_id':
     calls[0]+=1
     if calls[0]==2:
      import io
      return io.BytesIO(b'00000000-0000-0000-0000-000000000000\n')
    return original(path,*a,**kw)
   with patch.object(Path,'open',opened):self.assertIsNone(p._exiting_no_attached_mm(100,proc_root=r))
 def test_terminal_state_not_mislabelled_new_classification(self):
  for state in ('Z','X','x'):
   with self.subTest(state=state),self.proc()as(r,q):
    (q/'stat').write_text(stat(state=state));self.assertIsNone(p._exiting_no_attached_mm(100,proc_root=r))
 def test_counter_keeps_unknown_PSS_and_RSS_when_no_attached_mm(self):
  proof=dict(classification='verified_exiting_no_attached_mm',pid=100)
  with patch.object(p,'_read_rollup_pss',side_effect=FileNotFoundError(errno.ENOENT,'absent')),patch.object(p,'_read_smaps_pss',side_effect=ValueError('empty or incomplete smaps memory counters')),patch.object(p,'_exiting_no_attached_mm',return_value=proof),patch.object(p.Path,'read_text',return_value='Name:\ttest\n'):
   got=p._process_memory_counters(100,include_pss=True)
  self.assertIsNone(got['pss_bytes']);self.assertIsNone(got['rss_bytes']);self.assertFalse(got['pss_coverage_complete']);self.assertEqual(got['exiting_no_attached_mm'],proof)
 def test_permission_or_overlap_does_not_activate_probe(self):
  for error in (PermissionError(errno.EACCES,'denied'),ValueError('invalid or overlapping smaps VMA addresses')):
   with self.subTest(error=error),patch.object(p,'_read_rollup_pss',side_effect=FileNotFoundError()),patch.object(p,'_read_smaps_pss',side_effect=error),patch.object(p,'_exiting_no_attached_mm',side_effect=AssertionError('must not probe')):
    got=p._process_memory_counters(100,include_pss=True);self.assertIsNone(got['exiting_no_attached_mm'])
 def test_sample_only_sums_real_other_PSS_and_keeps_no_mm_scope(self):
  proof=dict(classification='verified_exiting_no_attached_mm',pid=200,start_ticks=20,uid=os.getuid(),physical_teardown_memory_status='unknown_not_measured')
  def counters(pid,include_pss):return dict(pss_coverage_complete=pid==100,pss_bytes=123 if pid==100 else None,exiting_no_attached_mm=None if pid==100 else proof)
  with patch.object(Path,'read_text',return_value=stat()),patch.object(p,'_process_memory_counters',side_effect=counters):
   d={};v=sp._pss({100,200},diagnostics=d)
  self.assertEqual(v,123);self.assertEqual(d['sample_verified_endpoints'],1);self.assertEqual(d['sample_no_attached_mm_count'],1);self.assertEqual(d['sample_physical_teardown_memory_status'],'unknown_for_no_attached_mm_tasks')
 def test_only_no_mm_is_not_fabricated_zero(self):
  proof=dict(classification='verified_exiting_no_attached_mm',pid=100,start_ticks=20,uid=os.getuid())
  with patch.object(Path,'read_text',return_value=stat()),patch.object(p,'_process_memory_counters',return_value=dict(pss_coverage_complete=False,pss_bytes=None,exiting_no_attached_mm=proof)):
   d={};self.assertIsNone(sp._pss({100},diagnostics=d));self.assertFalse(d['sample_coverage_complete'])
 def test_failed_sample_clears_previous_success_fields_preserves_counts(self):
  d=dict(sample_coverage_complete=True,sample_elapsed_seconds=100,sample_verified_endpoints=99,sample_error_type='OLD',recovered_endpoints=4,transient_endpoint_retries=8)
  counter=dict(pss_coverage_complete=False,pss_bytes=None,pss_read_errors=[{'reason':'empty or incomplete smaps memory counters'}],exiting_no_attached_mm=None)
  with patch.object(Path,'read_text',return_value=stat()),patch.object(p,'_process_memory_counters',return_value=counter),patch.object(sp,'time',Clock()):
   with self.assertRaises(RuntimeError)as raised:sp._pss({100},diagnostics=d)
  x=raised.exception.pss_diagnostics;self.assertFalse(x['sample_coverage_complete']);self.assertIsNone(x['sample_elapsed_seconds']);self.assertEqual(x['sample_verified_endpoints'],0);self.assertEqual(x['recovered_endpoints'],4);self.assertGreater(x['transient_endpoint_retries'],8);self.assertEqual(x['sample_error_type'],'RuntimeError')
 def test_normal_success_after_old_failure_has_no_stale_error(self):
  d=dict(sample_error_type='RuntimeError',sample_error_errno=5,last_incomplete_endpoint={'pid':99},sample_no_attached_mm_count=2)
  with patch.object(Path,'read_text',return_value=stat()),patch.object(p,'_process_memory_counters',return_value=dict(pss_coverage_complete=True,pss_bytes=77)):
   self.assertEqual(sp._pss({100},diagnostics=d),77)
  self.assertTrue(d['sample_coverage_complete']);self.assertNotIn('sample_error_type',d);self.assertNotIn('last_incomplete_endpoint',d);self.assertEqual(d['sample_no_attached_mm_count'],0)
 def test_old_failure_history_retained_current_failure_scope_is_fresh(self):
  old=dict(pid=999,attempt=1,reason='prior_sample')
  d=dict(endpoint_failures=[old]*16,sample_endpoint_failures=[old],sample_coverage_complete=True,sample_elapsed_seconds=9,sample_verified_endpoints=99)
  counter=dict(pss_coverage_complete=False,pss_bytes=None,exiting_no_attached_mm=None)
  with patch.object(Path,'read_text',return_value=stat()),patch.object(p,'_process_memory_counters',return_value=counter),patch.object(sp,'time',Clock()):
   with self.assertRaises(RuntimeError):sp._pss({100},diagnostics=d)
  self.assertEqual(d['endpoint_failures'],[old]*16);self.assertLessEqual(len(d['sample_endpoint_failures']),16);self.assertTrue(d['sample_endpoint_failures']);self.assertTrue(all(x['pid']==100 for x in d['sample_endpoint_failures']));self.assertFalse(d['sample_coverage_complete']);self.assertIsNone(d['sample_elapsed_seconds']);self.assertEqual(d['sample_verified_endpoints'],0)
 def test_no_mm_history_retained_bounded_after_later_clean_success(self):
  old=dict(classification='verified_exiting_no_attached_mm',pid=999)
  proof=dict(classification='verified_exiting_no_attached_mm',pid=100,start_ticks=20,uid=os.getuid())
  d=dict(no_attached_mm_endpoints=[old]*16)
  with patch.object(Path,'read_text',return_value=stat()),patch.object(p,'_process_memory_counters',return_value=dict(pss_coverage_complete=False,pss_bytes=None,exiting_no_attached_mm=proof)):
   self.assertIsNone(sp._pss({100},diagnostics=d))
  self.assertEqual(d['no_attached_mm_endpoints'],[old]*16);self.assertEqual(d['last_no_attached_mm_endpoint'],proof);self.assertEqual(d['sample_no_attached_mm_endpoints'],[proof])
  with patch.object(Path,'read_text',return_value=stat()),patch.object(p,'_process_memory_counters',return_value=dict(pss_coverage_complete=True,pss_bytes=88)):
   self.assertEqual(sp._pss({100},diagnostics=d),88)
  self.assertEqual(d['no_attached_mm_endpoints'],[old]*16);self.assertEqual(d['last_no_attached_mm_endpoint'],proof);self.assertEqual(d['sample_no_attached_mm_endpoints'],[]);self.assertEqual(d['sample_no_attached_mm_count'],0);self.assertTrue(d['sample_coverage_complete'])
 def test_short_metadata_reads_continue_to_real_EOF(self):
  original=Path.open
  with self.proc()as(r,q):
   class Short:
    def __init__(self,raw):self.raw=raw
    def __enter__(self):return self
    def __exit__(self,*a):self.raw.close()
    def readinto(self,view):
     sub=view[:1]
     try:return self.raw.readinto(sub)
     finally:sub.release()
   def opened(path,*a,**kw):
    value=original(path,*a,**kw)
    return Short(value)if 'r'in (a[0]if a else'r')else value
   with patch.object(Path,'open',opened):self.assertIsNotNone(p._exiting_no_attached_mm(100,proc_root=r))
 def test_short_valid_status_prefix_does_not_hide_later_Vm_field(self):
  original=Path.open
  with self.proc()as(r,q):
   initial=(q/'status').read_bytes();(q/'status').write_bytes(initial+b'VmRSS:\t0 kB\n')
   class Split:
    def __init__(self,raw):self.raw=raw;self.calls=0
    def __enter__(self):return self
    def __exit__(self,*a):self.raw.close()
    def readinto(self,view):
     self.calls+=1
     sub=view[:len(initial)]if self.calls==1 else view[:1]
     try:return self.raw.readinto(sub)
     finally:sub.release()
   def opened(path,*a,**kw):return Split(original(path,*a,**kw))if path==q/'status'else original(path,*a,**kw)
   with patch.object(Path,'open',opened):self.assertIsNone(p._exiting_no_attached_mm(100,proc_root=r))
 def test_invalid_metadata_readinto_and_overcap_remain_unknown(self):
  original=Path.open
  with self.proc()as(r,q):
   for bad in (None,True,-1,100000):
    class Invalid:
     def __enter__(self):return self
     def __exit__(self,*a):pass
     def readinto(self,view):return bad
    with self.subTest(bad=bad),patch.object(Path,'open',return_value=Invalid()):self.assertIsNone(p._exiting_no_attached_mm(100,proc_root=r))
   (q/'status').write_bytes(b'X'*(65536+1));self.assertIsNone(p._exiting_no_attached_mm(100,proc_root=r))
if __name__=='__main__':unittest.main()
