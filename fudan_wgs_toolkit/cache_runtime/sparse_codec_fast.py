"""Private vector-validation variant-CSR six-state exceptions; default state0 REF homozygote.

No dense decode, genotype, GPU, or allele-code reconstruction. Root mapping v1:
0=(ref2,called2),1=(1,2),2=(0,2),3=(0,0),4=(1,1),5=(0,1).
Output compact arrays are offsets[m+1], sample indices[nnz], states[nnz].
Validation stays in both decode/count APIs; vector checks replace Python row
loops, so no caller can claim prevalidated mutable arrays. Integer prefix
counts and compressed bytes/header semantics are unchanged.
Uncompressed-byte limit excludes input, masks/nonzero temporaries, compressor
scratch and decompressed arrays' validation scratch; it is not a total RAM cap.
"""
import hashlib,json
import numpy as np
from . import codec
FORMAT='private-reference-six-state-csr'
DEFAULT_MAX_BYTES=256*2**20
REF=np.array([2,1,0,0,1,0],dtype=np.int64)
CALLED=np.array([2,2,2,0,1,1],dtype=np.int64)
def _sha(data):return hashlib.sha256(data).hexdigest()
def _header_sha(meta):return _sha(json.dumps({k:v for k,v in meta.items() if k!='header_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def _sample_dtype(n):return np.dtype('<u2' if n<65536 else '<u4')
def _size(m,n,nnz):return 4*(m+1)+nnz*(_sample_dtype(n).itemsize+1)
def _budget(size,limit):
 if type(limit) is not int or limit<0:raise ValueError('Byte budget must be nonnegative integer')
 if size>limit:raise MemoryError('Compact CSR serialized output exceeds byte budget')
def compact(states,*,max_raw_bytes=DEFAULT_MAX_BYTES):
 s=codec._states(states);m,n=map(int,s.shape)
 if n>2**32:raise ValueError('Sample dimension exceeds uint32 indices')
 counts=np.count_nonzero(s,axis=1)
 nnz=int(counts.sum(dtype=np.int64))
 if nnz>2**32-1:raise ValueError('CSR offset exceeds uint32 capacity')
 _budget(_size(m,n,nnz),max_raw_bytes)
 offsets=np.empty(m+1,dtype='<u4');offsets[0]=0;offsets[1:]=np.cumsum(counts,dtype=np.uint64)
 # numpy nonzero is in C order, guaranteeing sorted sample indices per variant.
 rows,indices=np.nonzero(s)
 samples=indices.astype(_sample_dtype(n));values=s[rows,indices].astype(np.uint8)
 return offsets,samples,values

def validate(offsets,samples,values,m,n):
 if type(m) is not int or type(n) is not int or min(m,n)<0 or n>2**32:raise ValueError('Invalid CSR dimensions')
 if offsets.dtype!=np.dtype('<u4') or offsets.shape!=(m+1,) or samples.dtype!=_sample_dtype(n) or samples.ndim!=1 or values.dtype!=np.uint8 or values.ndim!=1:raise ValueError('CSR array dtype/shape mismatch')
 if offsets[0]!=0 or int(offsets[-1])!=len(samples) or len(values)!=len(samples) or np.any(offsets[1:]<offsets[:-1]):raise ValueError('Invalid CSR offsets')
 if values.size and (values.min()<1 or values.max()>5):raise ValueError('Exceptions must be states1..5')
 if samples.size and int(samples.max())>=n:raise ValueError('Sample index out of bounds')
 if len(samples)>1:
  # Adjacent samples must increase except at a real row start. Duplicate
  # offsets from empty rows are harmless; exclude 0/end before subtracting.
  bad=samples[1:]<=samples[:-1]
  boundaries=offsets[1:-1]
  boundaries=boundaries[(boundaries>0)&(boundaries<len(samples))]
  bad[boundaries-1]=False
  if np.any(bad):raise ValueError('Sample indices must be strictly sorted per variant')

def integer_counts(offsets,samples,values,m,n):
 """Whole selected-cache refAC/called totals; not threshold eligibility.

 A new subset must recompute exceptions in that selected subset, retaining its
 denominator/order. InitialMAC rounding/orientation remains the old caller's
 job; these integer counts never authorize a different AC/AN filter.
 """
 validate(offsets,samples,values,m,n)
 # Prefix deltas support empty rows without reduceat's empty-row ambiguity.
 ref_prefix=np.empty(len(values)+1,dtype=np.int64);called_prefix=np.empty_like(ref_prefix)
 ref_prefix[0]=0;called_prefix[0]=0
 np.cumsum(REF[values]-2,out=ref_prefix[1:]);np.cumsum(CALLED[values]-2,out=called_prefix[1:])
 ref=2*n+ref_prefix[offsets[1:]]-ref_prefix[offsets[:-1]]
 called=2*n+called_prefix[offsets[1:]]-called_prefix[offsets[:-1]]
 half_prefix=np.empty(len(values)+1,dtype=np.int64);half_prefix[0]=0;np.cumsum(values>=4,out=half_prefix[1:])
 half=half_prefix[offsets[1:]]-half_prefix[offsets[:-1]]
 return dict(reference_alleles=ref,called_alleles=called,half_missing_samples=half)

def encode(states,level=1,*,max_raw_bytes=DEFAULT_MAX_BYTES):
 if type(level) is not int or level not in (1,3):raise ValueError('Sparse probe levels are1/3')
 s=codec._states(states);m,n=map(int,s.shape);offsets,samples,values=compact(s,max_raw_bytes=max_raw_bytes)
 return encode_compact(offsets,samples,values,m,n,level,max_raw_bytes=max_raw_bytes)

def encode_compact(offsets,samples,values,m,n,level=1,*,max_raw_bytes=DEFAULT_MAX_BYTES):
 """Encode validated sparse exceptions without allocating an m-by-n matrix.

 The payload and header are identical to ``encode`` for the same states.
 Caller-owned arrays are never trusted as prevalidated or modified in place.
 """
 if type(level) is not int or level not in (1,3):raise ValueError('Sparse probe levels are1/3')
 if any(not isinstance(value,np.ndarray) for value in (offsets,samples,values)):raise ValueError('CSR inputs must be NumPy arrays')
 validate(offsets,samples,values,m,n)
 if len(values)>2**32-1:raise ValueError('CSR offset exceeds uint32 capacity')
 _budget(_size(m,n,len(values)),max_raw_bytes)
 raw=offsets.tobytes()+samples.tobytes()+values.tobytes()
 payload=codec._zstd().ZstdCompressor(level=level,write_content_size=True).compress(raw)
 meta=dict(format=FORMAT,version=1,state_encoding=codec.STATE_ENCODING,state_mapping=[v.copy() for v in codec.MAPPING],default_state=0,m=m,n=n,nnz=len(values),offset_dtype='<u4',sample_dtype=_sample_dtype(n).str,value_dtype='|u1',array_order=['offsets','samples','states'],codec='zstd',level=level,raw_bytes=len(raw),compressed_bytes=len(payload),raw_sha256=_sha(raw),compressed_sha256=_sha(payload))
 meta['header_sha256']=_header_sha(meta)
 _header(meta,max_raw_bytes)
 return payload,meta

def _header(meta,limit):
 keys={'format','version','state_encoding','state_mapping','default_state','m','n','nnz','offset_dtype','sample_dtype','value_dtype','array_order','codec','level','raw_bytes','compressed_bytes','raw_sha256','compressed_sha256','header_sha256'}
 if not isinstance(meta,dict) or set(meta)!=keys:raise ValueError('CSR header keys mismatch')
 if meta['header_sha256']!=_header_sha(meta):raise ValueError('Header checksum mismatch')
 if meta['format']!=FORMAT or type(meta['version']) is not int or meta['version']!=1 or meta['state_encoding']!=codec.STATE_ENCODING or meta['state_mapping']!=codec.MAPPING or type(meta['default_state']) is not int or meta['default_state']!=0 or meta['array_order']!=['offsets','samples','states'] or meta['codec']!='zstd' or type(meta['level']) is not int or meta['level'] not in (1,3):raise ValueError('CSR semantic header mismatch')
 m=codec._integer(meta['m'],'m');n=codec._integer(meta['n'],'n');nnz=codec._integer(meta['nnz'],'nnz')
 if n>2**32 or nnz>2**32-1 or nnz>m*n:raise ValueError('CSR dimension/index capacity exceeded')
 if meta['offset_dtype']!='<u4' or meta['sample_dtype']!=_sample_dtype(n).str or meta['value_dtype']!='|u1':raise ValueError('CSR header dtype mismatch')
 size=_size(m,n,nnz);_budget(size,limit)
 if codec._integer(meta['raw_bytes'],'raw_bytes')!=size:raise ValueError('CSR raw byte formula mismatch')
 if codec._integer(meta['compressed_bytes'],'compressed_bytes')>limit+2**20:raise MemoryError('Compressed payload exceeds budget plus1MiB')
 for key in ('raw_sha256','compressed_sha256','header_sha256'):
  if not isinstance(meta[key],str) or len(meta[key])!=64 or any(c not in '0123456789abcdef' for c in meta[key]):raise ValueError('Invalid SHA256 field')
 return m,n,nnz,size

def decode_compact(payload,meta,*,max_raw_bytes=DEFAULT_MAX_BYTES):
 m,n,nnz,size=_header(meta,max_raw_bytes);compressed=bytes(payload)
 if len(compressed)!=meta['compressed_bytes'] or _sha(compressed)!=meta['compressed_sha256']:raise ValueError('Compressed size/checksum mismatch')
 zstd=codec._zstd()
 if zstd.frame_content_size(compressed)!=size:raise ValueError('Zstd content-size mismatch')
 raw=zstd.ZstdDecompressor().decompress(compressed,max_output_size=max(1,size),allow_extra_data=False)
 if len(raw)!=size or _sha(raw)!=meta['raw_sha256']:raise ValueError('Raw size/checksum mismatch')
 start=4*(m+1);end=start+nnz*_sample_dtype(n).itemsize
 offsets=np.frombuffer(raw,dtype='<u4',count=m+1);samples=np.frombuffer(raw,dtype=_sample_dtype(n),count=nnz,offset=start);values=np.frombuffer(raw,dtype=np.uint8,count=nnz,offset=end)
 # Views keep the decompressed bytes alive and are intentionally read-only.
 validate(offsets,samples,values,m,n)
 return offsets,samples,values
