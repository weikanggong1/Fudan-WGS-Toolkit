"""Generated proc-format/ownership contracts, not scientific/performance data."""
import errno,gc,io,json,os,sys,tempfile,types,unittest,weakref
from pathlib import Path
from unittest.mock import patch
from fudan_wgs_toolkit import phewas_resources as p
from fudan_wgs_toolkit import single_preprocessing as sp

def vma(a=0x1000,b=0x5000,rss=16,pss=8):
 return (f'{a:08x}-{b:08x} rw-p 00000000 00:00 0 [anonymous]\n'
         f'Rss: {rss} kB\nPss: {pss} kB\nSwapPss: 0 kB\nVmFlags: rd wr\n').encode()
def stat(state='R',start='100'):
 fields=[state]+['0']*20;fields[19]=start;return '100 (fixture) '+' '.join(fields)
class Clock:
 def __init__(self):self.at=0
 def monotonic(self):return self.at
 def sleep(self,x):self.at+=x

class Contracts(unittest.TestCase):
 def capture(self,data,*,limit=32,chunk=4,short=None,fail=False):
  events=[];owner=[];refs=[]
  class Buffer(bytearray):pass
  def allocate(n):
   self.assertFalse(owner,'data buffer allocated with proc stream open')
   b=Buffer(n);refs.append(weakref.ref(b));events.append(('allocate',n));return b
  class Stream(io.BytesIO):
   def __init__(inner,value):super().__init__(value);inner.calls=0;inner.buffer=None;owner.append(inner);events.append(('open',len(value)))
   def read(inner,*args):raise AssertionError('new bytes read forbidden')
   def readinto(inner,view):
    inner.calls+=1
    if inner.buffer is None:inner.buffer=id(view.obj)
    self.assertEqual(inner.buffer,id(view.obj))
    if fail and inner.calls>1:raise OSError(errno.EIO,'PRIVATE_CONTENT')
    if short is None:return super().readinto(view)
    n=min(short,len(view));sub=view[:n]
    try:return super().readinto(sub)
    finally:sub.release()
   def close(inner):
    if not inner.closed:events.append(('close',inner.tell()));owner.remove(inner)
    super().close()
  count=[0]
  class FakePath:
   def open(self,*args,**kw):
    i=count[0];count[0]+=1;return Stream(data[min(i,len(data)-1)] if isinstance(data,list) else data)
  with patch.object(p,'bytearray',allocate,create=True),patch.object(p,'_SMAPS_CAPTURE_LIMIT_BYTES',limit),patch.object(p,'_SMAPS_CAPTURE_CHUNK_BYTES',chunk):
   try:value=p._capture_smaps(FakePath())
   except BaseException as error:
    self.assertFalse(owner);self.assertTrue(all(ref() is None for ref in refs));raise
  self.assertFalse(owner);self.assertTrue(all(ref() is None for ref in refs));return value,events
 def test_prealloc_before_open_and_one_buffer_per_open(self):
  result,events=self.capture(b'abc');self.assertEqual(result,b'abc');self.assertEqual(events[0],('allocate',5));self.assertEqual([x[0] for x in events],['allocate','open','close'])
 def test_growth_closes_and_restarts_whole_endpoint(self):
  result,events=self.capture([b'old-prefix-overflow',b'new-end']);self.assertEqual(result,b'new-end');self.assertEqual([e[1] for e in events if e[0]=='allocate'],[5,9])
 def test_short_readinto_reaches_EOF_without_truncation(self):
  result,_=self.capture(b'abcdefghijk',short=2);self.assertEqual(result,b'abcdefghijk')
 def test_exact_cap_requires_EOF_probe(self):
  self.assertEqual(self.capture(b'abcdefgh',limit=8)[0],b'abcdefgh')
 def test_one_byte_over_cap_is_rejected_not_truncated(self):
  with self.assertRaisesRegex(ValueError,'byte limit'):self.capture(b'abcdefghi',limit=8)
 def test_midread_error_releases_buffer_even_when_exception_held(self):
  enabled=gc.isenabled();gc.disable()
  try:
   try:self.capture(b'abcdef',short=2,fail=True)
   except OSError as error:
    held=error;self.assertEqual(held.errno,errno.EIO)
   else:self.fail('EIO was lost')
  finally:
   if enabled:gc.enable()
 def test_invalid_readinto_return_is_rejected(self):
  for value in (None,True,-1,99):
   class Stream(io.BytesIO):
    def readinto(self,view):return value
   with tempfile.TemporaryDirectory() as d,patch.object(Path,'open',return_value=Stream(b'x')):
    with self.assertRaisesRegex(ValueError,'byte stream'):p._capture_smaps(Path(d)/'x')
 def test_full_parser_bytes_and_counters_unchanged(self):
  with tempfile.TemporaryDirectory() as d:
   q=Path(d)/'smaps';q.write_bytes(vma()+vma(0x9000,0xd000,12,6));total,count=p._read_smaps_pss(q)
  self.assertEqual((count,total['Rss'],total['Pss']),(2,28*1024,14*1024))
 def test_strict_parser_still_rejects_overlap_missing_unit_and_empty(self):
  for content in (vma()+vma(0x2000,0x6000),vma().replace(b'Pss: 8 kB',b'Pss: -1 kB'),vma().replace(b'Pss: 8 kB',b'Pss: 8 bytes'),vma().replace(b'Pss: 8 kB\n',b''),b''):
   with tempfile.TemporaryDirectory() as d:
    q=Path(d)/'smaps';q.write_bytes(content)
    with self.assertRaises(ValueError):p._read_smaps_pss(q)
 def test_rollup_absent_smaps_complete_no_RSS_substitution(self):
  with patch.object(p,'_read_rollup_pss',side_effect=FileNotFoundError(errno.ENOENT,'PRIVATE')),patch.object(p,'_read_smaps_pss',return_value=({'Rss':1000,'Pss':123,'SwapPss':0},2)):
   got=p._process_memory_counters(100,include_pss=True)
  self.assertEqual(got['pss_bytes'],123);self.assertEqual(got['pss_source'],'smaps');self.assertEqual(got['pss_read_errors'][0]['reason'],'endpoint_unavailable')
 def test_permission_read_unknown_is_categorized_not_zero(self):
  with patch.object(p,'_read_rollup_pss',side_effect=FileNotFoundError(errno.ENOENT,'PRIVATE')),patch.object(p,'_read_smaps_pss',side_effect=PermissionError(errno.EACCES,'PRIVATE')):
   got=p._process_memory_counters(100,include_pss=True)
  self.assertIsNone(got['pss_bytes']);self.assertFalse(got['pss_coverage_complete']);self.assertEqual(got['pss_read_errors'][-1]['reason'],'endpoint_permission_denied');self.assertNotIn('PRIVATE',json.dumps(got))
 def sample(self,counts,reads=None,diagnostics=None):
  clock=Clock();diag={} if diagnostics is None else diagnostics
  with patch.object(sp,'time',clock),patch.object(Path,'read_text',side_effect=reads if reads is not None else lambda *a,**kw:stat()),patch.object(p,'_process_memory_counters',side_effect=counts):return sp._pss({100},diagnostics=diag),diag
 def test_incomplete_then_complete_preserves_retry_reason(self):
  unknown={'pss_bytes':None,'pss_coverage_complete':False,'pss_read_errors':[{'source':'smaps','reason':'invalid or overlapping smaps VMA addresses'}]}
  value,diag=self.sample([unknown,{'pss_bytes':42,'pss_coverage_complete':True}]);self.assertEqual(value,42);self.assertEqual(diag['recovered_endpoints'],1);self.assertEqual(diag['endpoint_failures'][0]['read_errors'],unknown['pss_read_errors'])
 def test_persistent_unknown_has_bounded_diagnostic_exception(self):
  unknown={'pss_bytes':None,'rss_bytes':999,'pss_coverage_complete':False,'pss_read_errors':[{'source':'smaps','reason':'endpoint_permission_denied','errno':errno.EACCES}]};diag={}
  with self.assertRaisesRegex(RuntimeError,'live task process') as cm:self.sample(lambda *a,**kw:unknown,diagnostics=diag)
  self.assertIsNone(cm.exception.pss_diagnostics['last_incomplete_endpoint']['pss_bytes']);self.assertEqual(cm.exception.pss_diagnostics['last_incomplete_endpoint']['rss_bytes'],999);self.assertLessEqual(len(diag['endpoint_failures']),16)
 def test_diagnostics_attached_even_when_caller_omits_parameter(self):
  with patch.object(Path,'read_text',side_effect=PermissionError(errno.EACCES,'PRIVATE')):
   with self.assertRaises(PermissionError) as cm:sp._pss({100})
  self.assertEqual(cm.exception.pss_diagnostics['sample_error_errno'],errno.EACCES);self.assertEqual(cm.exception.pss_diagnostics['current_endpoint']['phase'],'before_stat')
 def test_identity_change_is_fatal_despite_complete_PSS(self):
  with self.assertRaisesRegex(RuntimeError,'identity changed') as cm:self.sample([{'pss_bytes':42,'pss_coverage_complete':True}],[stat(start='1'),stat(start='2')])
  self.assertEqual(cm.exception.pss_diagnostics['sample_error_type'],'RuntimeError')
 def test_confirmed_exit_during_read_is_excluded_not_fictitious_zero(self):
  value,diag=self.sample([{'pss_bytes':None,'pss_coverage_complete':False}],[stat(),stat(state='Z')]);self.assertIsNone(value);self.assertEqual(diag['excluded_exited_endpoints'],1)
 def test_confirmed_absence_is_distinct_from_permission_unknown(self):
  value,diag=self.sample([],reads=[FileNotFoundError(errno.ENOENT,'PRIVATE')]);self.assertIsNone(value);self.assertEqual(diag['current_endpoint']['endpoint_status'],'confirmed_absent')

if __name__=='__main__':unittest.main(verbosity=2)
