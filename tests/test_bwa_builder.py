"""Fixed BWA builder/asset/version contracts; no mapping or speed benchmark."""
import copy,hashlib,json,os,struct,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from fudan_wgs_toolkit import build_preprocessing_bwa as module
from fudan_wgs_toolkit._preprocessing_io import atomic,bound,digest


def elf(interpreter):
 value=os.fsencode(interpreter)+b'\0'
 return struct.pack('<16sHHIQQQIHHHHHH',b'\x7fELF\x02\x01\x01'+bytes(9),3,62,1,0,64,0,0,64,56,1,0,0,0)+struct.pack('<IIQQQQQQ',3,4,120,0,0,len(value),len(value),1)+value


class PackageTests(unittest.TestCase):
 def test_official_source_and_one_file_patch_are_fixed(self):
  m=module._manifest();self.assertEqual(len(m['original_files']),77);self.assertEqual(m['archive_bytes'],232802)
  self.assertEqual(m['archive_sha256'],'79763dd6c48e8b01b8e27d99e6e5575d8bbca419e3eed9f3e330af072e63b370')
  self.assertEqual({n for n in m['original_files'] if m['original_files'][n]!=m['mate_profile_files'][n]},{'bwamem_pair.c'})
  self.assertIn('GNU GENERAL PUBLIC LICENSE',(module._PACKAGE/'BWA_COPYING').read_text())
  self.assertEqual(digest(module._PACKAGE/'mate_profile.patch'),module.PATCH_SHA256)
 def test_modified_contract_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   for f in module._PACKAGE.iterdir():
    if f.is_file():(p/f.name).write_bytes(f.read_bytes())
   (p/'mate_profile_contract.c').write_text('unbound C contract')
   with patch.object(module,'_PACKAGE',p),self.assertRaisesRegex(RuntimeError,'resource changed'):module._manifest()
 def test_invalid_prefix_budget_and_cpus_do_not_create_output(self):
  from fudan_wgs_toolkit import single_preprocessing
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);e=r/'env';(e/'conda-meta').mkdir(parents=True);cpu=next(iter(os.sched_getaffinity(0)))
   with patch.object(single_preprocessing,'_limits',return_value=dict(cpu_quota_count=1,finite_cpu_quotas=[1])):
    with self.assertRaisesRegex(ValueError,'already provide GCC'):module.build_preprocessing_bwa(e,r/'no-tools',cpu_affinity=[cpu])
    with self.assertRaisesRegex(ValueError,'finite'):module.build_preprocessing_bwa(e,r/'nan',cpu_affinity=[cpu],host_memory_limit_gib=float('nan'))
    with self.assertRaisesRegex(ValueError,'distinct'):module.build_preprocessing_bwa(e,r/'dup',cpu_affinity=[cpu,cpu])
    with self.assertRaisesRegex(ValueError,'outside'):module.build_preprocessing_bwa(e,e/'new')
   self.assertFalse((r/'no-tools').exists());self.assertFalse((r/'nan').exists())
 def test_bad_archive_preserves_failure_not_assets(self):
  from fudan_wgs_toolkit import single_preprocessing,build_preprocessing_aligner
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);e=r/'env'
   for n in ('conda-meta','bin','include','lib'):(e/n).mkdir(parents=True,exist_ok=True)
   for n in ('x86_64-conda-linux-gnu-gcc','x86_64-conda-linux-gnu-ar','make'):
    p=e/'bin'/n;p.write_text('not executed');p.chmod(0o700)
   for n in ('zlib.h','zconf.h'):(e/'include'/n).write_text('header')
   (e/'lib/libz.so.1').write_text('library');a=r/'wrong.tar.gz';a.write_bytes(b'wrong')
   cpu=next(iter(os.sched_getaffinity(0)));out=r/'build'
   with patch.object(single_preprocessing,'_limits',return_value=dict(cpu_quota_count=1,finite_cpu_quotas=[1])),patch.object(build_preprocessing_aligner,'_admit',return_value=dict(admitted=True)):
    with self.assertRaisesRegex(RuntimeError,'archive size or SHA'):module.build_preprocessing_bwa(e,out,source_archive=a,cpu_affinity=[cpu])
   self.assertEqual(json.loads((out/'build_receipt.private.json').read_text())['status'],'failed')
   self.assertFalse((out/'assets.private.json').exists());self.assertFalse((out/'source').exists())
 def test_version_exit_one_is_explicitly_checked_not_silenced(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);tool=r/'bwa';p=module._PACKAGE/'version_probe.py'
   data='Program: bwa (alignment via Burrows-Wheeler transformation)\\nVersion: 0.7.17-r1188\\nUsage:   bwa <command> [options]\\n'
   for code,version,expected in ((1,data,0),(0,data,2),(1,'wrong version\\n',2),(2,data,2)):
    tool.write_text("#!/bin/sh\nprintf '"+version+"'\nexit "+str(code)+'\n');tool.chmod(0o700)
    result=subprocess.run([sys.executable,'-B',str(p),str(tool),str(r/'probe.json'),str(r/'usage.txt')])
    self.assertEqual(result.returncode,expected);row=json.loads((r/'probe.json').read_text())
    self.assertEqual(row['actual_BWA_exit'],code);self.assertFalse(row['mapping_performed'])
    self.assertEqual(row['status']=='expected_usage_exit_verified',expected==0)
 def test_exact_patch_refuses_context_without_partial_write(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);source=root/'source';source.mkdir();target=source/'bwamem_pair.c';target.write_bytes(b'first\nold\nlast\n')
   (root/'mate_profile.patch').write_bytes(b'--- a/bwamem_pair.c\n+++ b/bwamem_pair.c\n@@ -1,3 +1,3 @@\n first\n-old\n+new\n last\n')
   m=dict(mate_profile_files={'bwamem_pair.c':hashlib.sha256(b'first\nnew\nlast\n').hexdigest()})
   with patch.object(module,'_PACKAGE',root):
    module._apply_patch(source,m);self.assertEqual(target.read_bytes(),b'first\nnew\nlast\n')
    target.write_bytes(b'bad context\n');old=target.read_bytes()
    with self.assertRaisesRegex(RuntimeError,'context differs'):module._apply_patch(source,m)
    self.assertEqual(target.read_bytes(),old)


class AssetTests(unittest.TestCase):
 """Interface-only fake artifacts bind identity; none is a mapping program."""
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.r=Path(self.temp.name);self.e=self.r/'env';(self.e/'lib').mkdir(parents=True)
  self.source=self.r/'source';self.oracle=self.r/'original';self.source.mkdir();self.oracle.mkdir()
  (self.source/'bwamem_pair.c').write_bytes(b'patched source');(self.oracle/'bwamem_pair.c').write_bytes(b'original source')
  self.loader=self.r/'loader';self.loader.write_bytes(b'loader');self.exe=self.source/'bwa';self.exe.write_bytes(elf(self.loader))
  self.z=self.e/'lib/libz.so.1';self.z.write_bytes(b'zlib');self.a=self.r/'source.tar.gz';self.a.write_bytes(b'archive')
  files={}
  for key,name in (('version_probe','version.json'),('API_contract_executable','contract'),('original_oracle_library','libbwa.a')):
   p=self.r/name;p.write_bytes(key.encode());files[key]=bound(p)
  self.m=dict(original_files={'bwamem_pair.c':digest(self.oracle/'bwamem_pair.c')},mate_profile_files={'bwamem_pair.c':digest(self.source/'bwamem_pair.c')},license_sha256='license')
  self.libs={'libz.so.1':bound(self.z)}
  self.assets=dict(schema=module._ASSET_SCHEMA,status=module._STATUS,variant='mate_profile',condaPrefix=str(self.e),executable=bound(self.exe),loader=bound(self.loader),dynamic_libraries=copy.deepcopy(self.libs),source_archive=bound(self.a),source_trees={str(self.source):self.m['mate_profile_files'],str(self.oracle):self.m['original_files']},source_commit=module.SOURCE_COMMIT,applied_patch_sha256=module.PATCH_SHA256,license='GPL-3.0',license_sha256='license',builder_sha256=digest(module.__file__),common_helper_sha256=digest(module._common.__file__),source_manifest_sha256=digest(module._PACKAGE/'source_manifest.json'),reported_version='0.7.17-r1188',actual_version_probe_exit=1,API_contract_schema=module._CONTRACT,**files)
  self.receipt=self.r/'receipt.json';self.commit();self.env={'LD_LIBRARY_PATH':str(self.e/'lib')}
  for name,value in (('_manifest',lambda:self.m),('SOURCE_BYTES',self.a.stat().st_size),('SOURCE_SHA256',digest(self.a))):
   p=patch.object(module,name,value);p.start();self.addCleanup(p.stop)
  p=patch.object(module._common,'_dynamic_dependencies',return_value=self.libs);self.deps=p.start();self.addCleanup(p.stop)
 def commit(self):
  core={k:v for k,v in self.assets.items() if k not in ('buildReceipt','build_receipt_sha256')}
  atomic(self.receipt,dict(schema=module._BUILD_SCHEMA,status=module._STATUS,artifacts=core,Conda_package_records_unchanged=True,source_files_unchanged=True,toolchain_zlib_unchanged=True))
  self.assets.update(buildReceipt=str(self.receipt),build_receipt_sha256=digest(self.receipt))
 def check(self):return module.validate_bwa_assets(self.assets,runtime_env=self.env)
 def test_exact_assets_and_json_file_verify_without_mapping(self):
  self.assertEqual(self.check(),self.assets);p=self.r/'assets.json';p.write_text(json.dumps(self.assets))
  self.assertEqual(module.validate_bwa_assets(p,runtime_env=self.env),self.assets)
  self.deps.assert_called_with(str(self.exe),str(self.loader),self.env)
 def test_shared_elf_helper_revision_requires_new_build(self):
  self.assets['common_helper_sha256']='old helper';self.commit()
  with self.assertRaisesRegex(RuntimeError,'helper/source/API'):self.check()
 def test_tool_replacement_with_identical_bytes_is_refused(self):
  b=self.exe.read_bytes();self.exe.unlink();self.exe.write_bytes(b)
  with self.assertRaisesRegex(RuntimeError,'artifact identity'):self.check()
 def test_runtime_libz_change_is_rejected_before_trace(self):
  self.z.write_bytes(b'changed zlib')
  with self.assertRaisesRegex(RuntimeError,'dynamic library changed'):self.check()
  self.deps.assert_not_called()
 def test_environment_must_resolve_exact_same_libraries(self):
  other=self.r/'other';other.write_bytes(b'another zlib');self.deps.return_value={'libz.so.1':bound(other)}
  with self.assertRaisesRegex(RuntimeError,'different BWA libraries'):self.check()
 def test_ld_injection_rejected_but_conda_gold_allowed(self):
  self.env['LD_PRELOAD']='unbound'
  with self.assertRaises(ValueError):self.check()
  self.env.pop('LD_PRELOAD');self.env['LD_GOLD']='/existing/conda/bin/ld.gold';self.check()
 def test_receipt_status_tampering_cannot_commit_artifacts(self):
  row=json.loads(self.receipt.read_text());row['status']='building';atomic(self.receipt,row)
  with self.assertRaisesRegex(RuntimeError,'SHA changed'):self.check()
  self.assets['build_receipt_sha256']=digest(self.receipt)
  with self.assertRaisesRegex(RuntimeError,'exact assets'):self.check()
 def test_variant_and_api_contract_are_not_relabelable(self):
  self.assets['variant']='original'
  with self.assertRaisesRegex(RuntimeError,'exact assets'):self.check()
  self.assets['variant']='mate_profile';self.assets['API_contract_schema']='unknown ABI';self.commit()
  with self.assertRaisesRegex(RuntimeError,'helper/source/API'):self.check()
 def test_changed_original_or_candidate_sources_are_rejected(self):
  (self.oracle/'bwamem_pair.c').write_bytes(b'changed oracle')
  with self.assertRaisesRegex(RuntimeError,'source changed'):self.check()
 def test_bad_source_closure_is_refused_even_if_receipt_recommitted(self):
  self.assets['source_trees'].pop(str(self.oracle));self.commit()
  with self.assertRaisesRegex(RuntimeError,'source closures'):self.check()
 def test_resume_cannot_reuse_changed_assets_plan(self):
  from fudan_wgs_toolkit.single_preprocessing import _Runner
  r=self.r/'run';r.mkdir();cpu=next(iter(os.sched_getaffinity(0)));cfg=dict(cpu_affinity=[cpu],host_memory_limit_gib=2,host_memory_reserve_gib=4,bwa_assets=self.assets)
  atomic(r/'plan.private.json',cfg);changed=copy.deepcopy(cfg);changed['bwa_assets']['source_commit']='other'
  with self.assertRaisesRegex(ValueError,'identical source, tools'):_Runner(r,self.e,changed,True)


if __name__=='__main__':unittest.main()
