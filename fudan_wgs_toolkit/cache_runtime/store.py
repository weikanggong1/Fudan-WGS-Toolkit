"""Resumable compressed six-state CSR genotype container, prepared on CPU.

Frames cover original physical columns consecutively, including all-zero and
zero-layer sites. Counts apply only to the bound complete ordered sample axis.
CSR and count frames are independently checksummed Zstd. Ordinary conversion
retains its original genotype source; explicitly synthetic sources may use a
direct byte budget. New sources can use an append-only journal without
chromosome-sized lists and lossless row-delta index encoding. Legacy absolute
frames and JSON journals remain readable and resumable with their binding.
A completed container is immutable. Incomplete recovery requires exact binding.
"""
import hashlib, json, mmap, os, shutil, stat
from pathlib import Path
import numpy as np
from . import sparse_codec_fast as csr
from . import codec
FORMAT='private-six-state-csr-container-v1'
CHUNK=1024
INDEX=np.dtype([('start','<u8'),('m','<u4'),('offset','<u8'),('size','<u8'),('header_offset','<u8'),('header_size','<u4'),('counts_offset','<u8'),('counts_size','<u8'),('payload_sha','S64'),('counts_sha','S64')])
def sha(x):return hashlib.sha256(x).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,b):
 p=Path(p);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def usage(p):
 logical=allocated=0
 for entry in Path(p).iterdir():
  value=entry.stat()
  if stat.S_ISREG(value.st_mode):logical+=value.st_size;allocated+=getattr(value,'st_blocks',0)*512
 return dict(logical_bytes=logical,allocated_bytes=allocated)
def _append(p,b):
 with p.open('ab') as f:
  pos=f.tell();f.write(b);f.flush();os.fsync(f.fileno())
 return pos
class Writer:
 def __init__(self,path,binding,samples,m,*,source_bytes=None,storage_fraction=.10,minimum_storage_bytes=0,chunk=CHUNK,storage_limit_bytes=None,streaming=False,sample_encoding='absolute',free_disk_reserve_bytes=64*2**20):
  self.path=Path(path);self.samples=np.asarray(samples)
  if type(streaming) is not bool:raise ValueError('streaming must be a boolean')
  if streaming and not isinstance(binding,dict):raise ValueError('Streaming journals require an explicit source binding')
  if sample_encoding not in ('absolute','row-delta'):raise ValueError('Unknown CSR sample encoding')
  if type(free_disk_reserve_bytes) is not int or free_disk_reserve_bytes<0:raise ValueError('free_disk_reserve_bytes must be a nonnegative integer')
  self.free_disk_reserve_bytes=free_disk_reserve_bytes
  self.streaming=streaming
  if self.samples.dtype!=np.int64 or self.samples.ndim!=1 or len(np.unique(self.samples))!=len(self.samples):raise ValueError('Physical sample axis must be unique int64')
  if type(m)is not int or m<0 or not 0<storage_fraction<=.10:raise ValueError('Invalid dimension/storage budget')
  if storage_limit_bytes is not None:
   if not isinstance(binding,dict) or binding.get('synthetic') is not True:raise ValueError('Explicit storage limits require a synthetic source binding')
   if type(storage_limit_bytes) is not int or storage_limit_bytes<=0:raise ValueError('storage_limit_bytes must be a positive integer')
   if source_bytes is not None and (type(source_bytes) is not int or source_bytes<=0):raise ValueError('Invalid source byte count')
   storage_cap_bytes=storage_limit_bytes
  else:
   if source_bytes is None or source_bytes<=0:raise ValueError('Invalid dimension/storage budget')
   storage_cap_bytes=max(int(source_bytes*storage_fraction),minimum_storage_bytes)
  if type(chunk) is not int or not 1<=chunk<=CHUNK:raise ValueError('chunk must be an integer in 1..1024')
  if type(minimum_storage_bytes) is not int or not 0<=minimum_storage_bytes<=2**20:raise ValueError('minimum_storage_bytes must be 0..1048576')
  self.spec=dict(format=FORMAT,state_encoding=codec.STATE_ENCODING,m=m,n=len(samples),chunk=chunk,binding=binding,sample_sha256=sha(self.samples.astype('<i8').tobytes()),source_bytes=source_bytes,storage_cap_bytes=storage_cap_bytes)
  if storage_limit_bytes is not None:self.spec['storage_budget_kind']='explicit_synthetic_bytes'
  if streaming:self.spec['journal_format']='jsonl-v1'
  if sample_encoding!='absolute':self.spec['sample_encoding']=sample_encoding
  self.journal_path=self.path/('journal.jsonl' if streaming else 'journal.json')
  if self.path.exists():
   if (self.path/'COMPLETE').exists():raise FileExistsError('Completed cache cannot be overwritten')
   if json.loads((self.path/'manifest.pending.json').read_text())!=self.spec:raise ValueError('Resume source/sample binding mismatch')
   actual=np.load(self.path/'samples.npy',allow_pickle=False)
   if not np.array_equal(actual,self.samples) or actual.dtype!=self.samples.dtype:raise ValueError('Resume sample axis mismatch')
   self.rows=None if streaming else json.loads(self.journal_path.read_text())
  else:
   self._disk_budget(self.samples.nbytes+65536)
   self.path.mkdir(parents=True,mode=0o700)
   atomic(self.path/'manifest.pending.json',canonical(self.spec));np.save(self.path/'samples.npy',self.samples,allow_pickle=False)
   for name in ('data.bin','headers.bin','counts.bin'):(self.path/name).touch()
   self.rows=None if streaming else [];atomic(self.journal_path,b'' if streaming else canonical(self.rows))
  self._recover()
  self.peak_storage=usage(self.path)
 def _journal_rows(self,repair_tail=False):
  """Yield one bounded durable record at a time, rejecting committed damage.

  A newline is the commit boundary after all three frame streams are fsynced.
  Only an unterminated final record is an uncommitted torn append; recovery
  removes that tail and later truncates the streams to the last valid record.
  """
  if not self.streaming:
   yield from self.rows
   return
  valid_end=0;torn=False
  with self.journal_path.open('rb') as stream:
   while True:
    line=stream.readline(16385)
    if not line:break
    if len(line)>16384:raise ValueError('Streaming journal record exceeds its byte bound')
    if not line.endswith(b'\n'):
     if not repair_tail:raise ValueError('Uncommitted streaming journal tail')
     torn=True;break
    try:row=json.loads(line)
    except (ValueError,UnicodeError) as error:raise ValueError('Invalid committed streaming journal record') from error
    if not isinstance(row,dict) or set(row)!=set(INDEX.names):raise ValueError('Streaming journal row keys differ')
    for key in INDEX.names:
     value=row[key]
     if key in ('payload_sha','counts_sha'):
      if not isinstance(value,str) or len(value)!=64 or any(character not in '0123456789abcdef' for character in value):raise ValueError('Invalid streaming journal digest')
     elif type(value) is not int or value<0:raise ValueError('Invalid streaming journal integer')
    valid_end=stream.tell()
    yield row
  if torn:
   with self.journal_path.open('r+b') as stream:
    stream.truncate(valid_end);stream.flush();os.fsync(stream.fileno())
 def _recover(self):
  ends=dict(data=0,header=0,counts=0);start=0;frames=0
  for row in self._journal_rows(repair_tail=True):
   if row['start']!=start or row['m']!=min(self.spec['chunk'],self.spec['m']-start):raise ValueError('Nonconsecutive committed frame')
   if row['m']<=0:raise ValueError('Committed frame is empty')
   for kind,key,size in [('data','offset','size'),('header','header_offset','header_size'),('counts','counts_offset','counts_size')]:
    if row[key]!=ends[kind]:raise ValueError('Frame stream gap')
    ends[kind]+=row[size]
   _read_frame(self.path,row,self.spec);start+=row['m'];frames+=1
  for kind,name in [('data','data.bin'),('header','headers.bin'),('counts','counts.bin')]:
   p=self.path/name
   if p.stat().st_size<ends[kind]:raise ValueError('Committed stream truncated')
   with p.open('r+b') as f:f.truncate(ends[kind]);f.flush();os.fsync(f.fileno())
  if self.streaming:
   # An interrupted finalization can leave an incomplete or already renamed
   # index. Without COMPLETE neither is published; rebuild from verified ACKs
   # rather than charging the stale index twice against the byte budget.
   for name in ('index.npy.tmp','index.npy'):
    p=self.path/name
    if p.exists():p.unlink()
  self.next_start=start;self.frame_count=frames
 def _budget(self,extra):
  u=usage(self.path);cap=self.spec['storage_cap_bytes']
  # Logical projection plus per-file alignment/headroom; no filesystem quota guarantee.
  if max(u.values())+extra+65536>cap:raise MemoryError('Additional cache storage exceeds configured genotype-container byte budget')
  self._disk_budget(extra+65536)
 def _disk_budget(self,extra):
  ancestor=self.path
  while not ancestor.exists():ancestor=ancestor.parent
  if shutil.disk_usage(ancestor).free<extra+self.free_disk_reserve_bytes:raise OSError('Insufficient free disk space for cache write and reserved free space')
 def storage_report(self):
  """Measure this container's retained files, excluding portable metadata.

  Peak counters cover observed write boundaries in this Writer process. They
  are disk-byte measurements, not process RAM or a filesystem hard quota.
  """
  measured=self._observe_storage()
  return dict(scope='genotype container files only; excludes metadata subdirectories',
   current=measured,peak_observed=dict(self.peak_storage),free_disk_reserve_bytes=self.free_disk_reserve_bytes)
 def _observe_storage(self):
  measured=usage(self.path)
  self.peak_storage={key:max(self.peak_storage[key],value) for key,value in measured.items()}
  return measured
 def append(self,states,original_counts):
  s=codec._states(states);m,n=map(int,s.shape);start=self.next_start
  if n!=self.spec['n'] or m!=min(self.spec['chunk'],self.spec['m']-start) or m<=0:raise ValueError('Frame shape/order mismatch')
  payload,meta=csr.encode(s,level=3,sample_encoding=self.spec.get('sample_encoding','absolute'));arrays=csr.decode_compact(payload,meta)
  direct=csr.compact(s);counts=csr.integer_counts(*arrays,m,n)
  if not all(np.array_equal(a,b) for a,b in zip(arrays,direct)):raise ValueError('CSR roundtrip mismatch')
  self._commit_frame(payload,meta,counts,original_counts,m,n)
 def append_compact(self,offsets,samples,values,original_counts):
  """Append one full ordered CSR frame with the normal durable validations.

  All six states, exact integer counts, compression roundtrip, storage
  admission and journal/fsync checks are shared with the dense writer.
  """
  if not isinstance(offsets,np.ndarray) or offsets.ndim!=1:raise ValueError('CSR offsets must be a one-dimensional NumPy array')
  m,n=len(offsets)-1,self.spec['n']
  if m!=min(self.spec['chunk'],self.spec['m']-self.next_start) or m<=0:raise ValueError('Frame shape/order mismatch')
  payload,meta=csr.encode_compact(offsets,samples,values,m,n,level=3,sample_encoding=self.spec.get('sample_encoding','absolute'))
  arrays=csr.decode_compact(payload,meta)
  if not all(np.array_equal(a,b) for a,b in zip(arrays,(offsets,samples,values))):raise ValueError('CSR roundtrip mismatch')
  counts=csr.integer_counts(*arrays,m,n)
  self._commit_frame(payload,meta,counts,original_counts,m,n)
 def _commit_frame(self,payload,meta,counts,original_counts,m,n):
  start=self.next_start
  keys=('reference_alleles','called_alleles','half_missing_samples')
  if not all(np.asarray(original_counts[k]).dtype==np.int64 and np.array_equal(counts[k],original_counts[k]) for k in keys):raise ValueError('Original integer summary differs before writing')
  raw=np.column_stack([counts[k] for k in keys]).astype('<i8').tobytes();cp=codec._zstd().ZstdCompressor(level=3,write_content_size=True).compress(raw)
  header=canonical(dict(csr_meta=meta,counts_raw_sha256=sha(raw),counts_raw_bytes=len(raw),counts_compressed_sha256=sha(cp),counts_compressed_bytes=len(cp)))
  row=dict(start=start,m=m,offset=(self.path/'data.bin').stat().st_size,size=len(payload),header_offset=(self.path/'headers.bin').stat().st_size,header_size=len(header),counts_offset=(self.path/'counts.bin').stat().st_size,counts_size=len(cp),payload_sha=sha(payload),counts_sha=sha(cp))
  self._budget(len(payload)+len(cp)+len(header)+len(canonical(row))*2)
  for name,b in [('data.bin',payload),('counts.bin',cp),('headers.bin',header)]:_append(self.path/name,b)
  if self.streaming:_append(self.journal_path,canonical(row)+b'\n')
  else:self.rows.append(row);atomic(self.journal_path,canonical(self.rows))
  self.next_start+=m;self.frame_count+=1
  measured=self._observe_storage()
  if max(measured.values())>self.spec['storage_cap_bytes']:raise MemoryError('Measured cache storage cap exceeded')
 def finish(self):
  if self.next_start!=self.spec['m']:raise ValueError('Incomplete physical axis')
  tmp=self.path/'index.npy.tmp'
  if self.streaming:
   # The index is the ordinary reader-compatible NPY structured array. Build
   # it through a mapping and one small block, never a chromosome-sized list.
   self._budget(self.frame_count*INDEX.itemsize+512)
   a=np.lib.format.open_memmap(tmp,mode='w+',dtype=INDEX,shape=(self.frame_count,))
   block=np.empty(min(4096,max(1,self.frame_count)),dtype=INDEX);first=0;count=0
   for row in self._journal_rows():
    if first+count>=self.frame_count:raise ValueError('Streaming journal gained unexpected frames')
    block[count]=tuple(row[k] for k in INDEX.names);count+=1
    if count==len(block):
     a[first:first+count]=block[:count];a.flush();first+=count;count=0
     if hasattr(a._mmap,'madvise'):
      try:a._mmap.madvise(mmap.MADV_DONTNEED)
      except (AttributeError,OSError,ValueError):pass
   if count:a[first:first+count]=block[:count];first+=count
   if first!=self.frame_count:raise ValueError('Streaming journal lost committed frames')
   a.flush();del a,block
   with tmp.open('rb') as f:os.fsync(f.fileno())
  else:
   a=np.empty(len(self.rows),dtype=INDEX)
   for i,row in enumerate(self.rows):a[i]=tuple(row[k] for k in INDEX.names)
   with tmp.open('wb') as f:np.save(f,a,allow_pickle=False);f.flush();os.fsync(f.fileno())
  os.replace(tmp,self.path/'index.npy')
  hashes={name:file_sha(self.path/name) for name in ('data.bin','headers.bin','counts.bin','samples.npy','index.npy')}
  manifest=dict(self.spec,frames=self.frame_count,files_sha256=hashes,storage=usage(self.path),association_precision_accepted=False)
  b=canonical(manifest);self._budget(len(b)+128)
  atomic(self.path/'manifest.json',b);atomic(self.path/'COMPLETE',sha(b).encode())
  measured=self._observe_storage()
  if max(measured.values())>self.spec['storage_cap_bytes']:(self.path/'COMPLETE').unlink();raise MemoryError('Final measured storage cap exceeded')
  return manifest

def _range(path,offset,size):
 if offset<0 or size<0 or offset+size>path.stat().st_size:raise ValueError('Frame out of bounds')
 with path.open('rb') as f:f.seek(offset);b=f.read(size)
 if len(b)!=size:raise ValueError('Short frame read')
 return b
def _read_frame(path,row,spec):
 header=json.loads(_range(path/'headers.bin',int(row['header_offset']),int(row['header_size'])))
 payload=_range(path/'data.bin',int(row['offset']),int(row['size']));cp=_range(path/'counts.bin',int(row['counts_offset']),int(row['counts_size']))
 for key,b in [('payload_sha',payload),('counts_sha',cp)]:
  expected=row[key];expected=expected.decode() if isinstance(expected,bytes) else str(expected)
  if sha(b)!=expected:raise ValueError('Frame SHA mismatch')
 meta=header['csr_meta'];m,n=meta['m'],meta['n']
 if spec.get('sample_encoding','absolute')!=meta.get('sample_encoding','absolute'):raise ValueError('Frame sample encoding differs from container binding')
 if m!=int(row['m']) or n!=spec['n'] or header['counts_raw_bytes']!=m*24 or header['counts_compressed_bytes']!=len(cp) or sha(cp)!=header['counts_compressed_sha256']:raise ValueError('Frame dimensions/count metadata mismatch')
 arrays=csr.decode_compact(payload,meta)
 if codec._zstd().frame_content_size(cp)!=m*24:raise ValueError('Counts frame content size mismatch')
 raw=codec._zstd().ZstdDecompressor().decompress(cp,max_output_size=max(1,m*24),allow_extra_data=False)
 if len(raw)!=m*24 or sha(raw)!=header['counts_raw_sha256']:raise ValueError('Counts raw SHA mismatch')
 c=np.frombuffer(raw,dtype='<i8').reshape(m,3)
 # Counts generated and proven at conversion, checksummed thereafter; no repeated CSR count reduction.
 if np.any(c<0) or np.any(c[:,0]>c[:,1]) or np.any(c[:,1]>2*n) or np.any(c[:,2]>n):raise ValueError('Impossible cached integer summary')
 return dict(offsets=arrays[0],sample_index=arrays[1],state=arrays[2],reference_alleles=c[:,0],called_alleles=c[:,1],half_missing_samples=c[:,2],m=m,n=n,start=int(row['start']))
class Container:
 def __init__(self,path,expected_source_binding=None,expected_samples=None):
  self.path=Path(path);b=(self.path/'manifest.json').read_bytes()
  if (self.path/'COMPLETE').read_text()!=sha(b):raise ValueError('Missing/invalid completed marker')
  self.manifest=json.loads(b);s=self.manifest
  if s['format']!=FORMAT or s['state_encoding']!=codec.STATE_ENCODING or type(s['chunk']) is not int or not 1<=s['chunk']<=CHUNK:raise ValueError('Container semantic mismatch')
  if s.get('sample_encoding','absolute') not in ('absolute','row-delta'):raise ValueError('Container sample encoding mismatch')
  chunk=s['chunk']  # Readers accept committed smaller frames; Writer stays at CHUNK.
  if expected_source_binding is not None and s['binding']!=expected_source_binding:raise ValueError('Source binding mismatch')
  self.samples=np.load(self.path/'samples.npy',allow_pickle=False)
  index_mode='r' if s.get('journal_format')=='jsonl-v1' else None
  self.index=np.load(self.path/'index.npy',allow_pickle=False,mmap_mode=index_mode)
  if self.samples.dtype!=np.int64 or self.samples.shape!=(s['n'],) or sha(self.samples.astype('<i8').tobytes())!=s['sample_sha256']:raise ValueError('Sample binding mismatch')
  if expected_samples is not None and not np.array_equal(self.samples,expected_samples):raise ValueError('Requested sample axis differs')
  if self.index.dtype!=INDEX or len(self.index)!=s['frames'] or len(self.index)!=(s['m']+chunk-1)//chunk:raise ValueError('Invalid frame index')
  for name in ('samples.npy','index.npy'):
   if file_sha(self.path/name)!=s['files_sha256'][name]:raise ValueError('Index/sample file checksum mismatch')
  start=0;ends=[0,0,0]
  for row in self.index:
   if int(row['start'])!=start or int(row['m'])!=min(chunk,s['m']-start):raise ValueError('Index physical order mismatch')
   for j,(offset,size) in enumerate([('offset','size'),('header_offset','header_size'),('counts_offset','counts_size')]):
    if int(row[offset])!=ends[j]:raise ValueError('Index stream gap')
    ends[j]+=int(row[size])
   start+=int(row['m'])
  for end,name in zip(ends,('data.bin','headers.bin','counts.bin')):
   if (self.path/name).stat().st_size!=end:raise ValueError('Stream length mismatch')
  self.samples.flags.writeable=False;self.index.flags.writeable=False;self.complete=True
 def read_frame(self,frame_id):
  if type(frame_id)is not int or not 0<=frame_id<len(self.index):raise IndexError('Frame ID out of bounds')
  return _read_frame(self.path,{k:self.index[frame_id][k] for k in INDEX.names},self.manifest)
 def verify_streams(self):
  for name,h in self.manifest['files_sha256'].items():
   if file_sha(self.path/name)!=h:raise ValueError('Container file checksum mismatch')
  return True
open_container=Container
