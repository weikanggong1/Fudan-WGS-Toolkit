"""Public preprocessing source/cache integration contracts, not WGS benchmarks.

Fake executable bytes exercise binding checks. The only commands executed here
are a test-owned dictionary shim and the project's small dictionary copy; real
FASTQ processing and original-software scientific accuracy are assessed apart.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from fudan_wgs_toolkit import single_preprocessing as pipeline
from fudan_wgs_toolkit import preprocessing_reference as reference_module
from fudan_wgs_toolkit import build_preprocessing_fastp as fastp
from fudan_wgs_toolkit import build_preprocessing_bwa as bwa
from fudan_wgs_toolkit._preprocessing_io import atomic, bound, digest
from test_fastp_builder import elf_bytes
import test_gatk_sourcepatch as _fixtures


class _ReachedTraining(Exception):pass


class PublicSourceIntegrationContracts(unittest.TestCase):
    setUp = _fixtures.SourcePatchContracts.setUp

    def prepare_inputs(self):
        options=_fixtures.PipelineDispatchContracts.prepare_inputs(self)
        self.options=options
        self.marker=self.root/'dictionary_calls.txt'
        samtools=self.prefix/'bin/samtools'
        samtools.write_text('#!'+sys.executable+'\nimport pathlib,sys\n'
            'assert sys.argv[1]=="dict"\n'
            'pathlib.Path(sys.argv[3]).write_text(pathlib.Path(sys.argv[4]).with_suffix(".dict").read_text())\n'
            'with pathlib.Path('+repr(str(self.marker))+').open("a") as out:out.write("dict\\n")\n')
        samtools.chmod(0o700)
        return options

    def make_fastp(self):
        (self.prefix/'lib').mkdir(exist_ok=True)
        loader=self.root/'loader';loader.write_bytes(b'contract loader')
        executable=self.root/'patched_fastp';executable.write_bytes(elf_bytes(loader))
        zlib=self.prefix/'lib/libz.so.1';zlib.write_bytes(b'contract zlib')
        archive=self.root/'fastp_source.tar.gz';archive.write_bytes(b'contract archive')
        source=self.root/'fastp_source';source.mkdir();(source/'read.cpp').write_bytes(b'contract source')
        self.fastp_manifest=dict(original_files={'read.cpp':digest(source/'read.cpp')},
            append_files={'read.cpp':digest(source/'read.cpp')},license_sha256='contract license')
        self.dynamic={'libz.so.1':bound(zlib)}
        core=dict(schema=fastp._ASSET_SCHEMA,status=fastp._STATUS,variant='append',condaPrefix=str(self.prefix),
            executable=bound(executable),loader=bound(loader),dynamic_libraries=copy.deepcopy(self.dynamic),
            source_tree=str(source),source_file_sha256=self.fastp_manifest['append_files'],source_archive=bound(archive),
            source_tag='v0.20.0',source_sha256=digest(archive),applied_patch_sha256=fastp.PATCH_SHA256,
            license='MIT',license_sha256='contract license',builder_sha256=digest(fastp.__file__),
            source_manifest_sha256=digest(fastp._PACKAGE/'source_manifest.json'),default_fastp_workers=1)
        receipt=self.root/'fastp_build_receipt.private.json'
        atomic(receipt,dict(schema=fastp._BUILD_SCHEMA,status=fastp._STATUS,artifacts=core,
            Conda_package_records_unchanged=True,source_files_unchanged=True,toolchain_unchanged=True,zlib_unchanged=True))
        self.fastp_assets=dict(core,buildReceipt=str(receipt),build_receipt_sha256=digest(receipt))
        self.fastp_asset_file=self.root/'fastp_assets.private.json';atomic(self.fastp_asset_file,self.fastp_assets)
        self.traces=[]
        def dynamic_trace(executable, loader, env):
            self.traces.append((str(executable),str(loader),dict(env)))
            return copy.deepcopy(self.dynamic)
        for name,value in (('_manifest',lambda:self.fastp_manifest),('SOURCE_BYTES',archive.stat().st_size),
                           ('SOURCE_SHA256',digest(archive)),('_dynamic_dependencies',dynamic_trace)):
            patcher=patch.object(fastp,name,value);patcher.start();self.addCleanup(patcher.stop)
        return self.fastp_assets

    def make_bwa(self, conda_prefix=None):
        environment=Path(conda_prefix or self.prefix)
        (environment/'lib').mkdir(parents=True,exist_ok=True)
        loader=self.root/'bwa_loader';loader.write_bytes(b'contract loader')
        source=self.root/'bwa_candidate';source.mkdir();(source/'bwamem_pair.c').write_bytes(b'candidate source')
        oracle=self.root/'bwa_original';oracle.mkdir();(oracle/'bwamem_pair.c').write_bytes(b'original source')
        executable=source/'bwa';executable.write_bytes(elf_bytes(loader))
        zlib=environment/'lib/libz.so.1';zlib.write_bytes(b'contract zlib')
        archive=self.root/'bwa_source.tar.gz';archive.write_bytes(b'contract source archive')
        manifest=dict(original_files={'bwamem_pair.c':digest(oracle/'bwamem_pair.c')},
            mate_profile_files={'bwamem_pair.c':digest(source/'bwamem_pair.c')},license_sha256='contract license')
        self.bwa_dynamic={'libz.so.1':bound(zlib)};self.bwa_traces=[]
        artifacts={}
        for key,name in (('version_probe','version.json'),('API_contract_executable','contract'),('original_oracle_library','libbwa.a')):
            path=self.root/('bwa_'+name);path.write_bytes(('contract '+key).encode());artifacts[key]=bound(path)
        core=dict(schema=bwa._ASSET_SCHEMA,status=bwa._STATUS,variant='mate_profile',condaPrefix=str(environment),
            executable=bound(executable),loader=bound(loader),dynamic_libraries=copy.deepcopy(self.bwa_dynamic),
            source_archive=bound(archive),source_trees={str(source):manifest['mate_profile_files'],str(oracle):manifest['original_files']},
            source_commit=bwa.SOURCE_COMMIT,applied_patch_sha256=bwa.PATCH_SHA256,license='GPL-3.0',license_sha256='contract license',
            builder_sha256=digest(bwa.__file__),common_helper_sha256=digest(fastp.__file__),
            source_manifest_sha256=digest(bwa._PACKAGE/'source_manifest.json'),reported_version='0.7.17-r1188',
            actual_version_probe_exit=1,API_contract_schema=bwa._CONTRACT,**artifacts)
        receipt=self.root/'bwa_build_receipt.private.json'
        atomic(receipt,dict(schema=bwa._BUILD_SCHEMA,status=bwa._STATUS,artifacts=core,
            Conda_package_records_unchanged=True,source_files_unchanged=True,toolchain_zlib_unchanged=True))
        self.bwa_assets=dict(core,buildReceipt=str(receipt),build_receipt_sha256=digest(receipt))
        self.bwa_asset_file=self.root/'bwa_assets.private.json';atomic(self.bwa_asset_file,self.bwa_assets)
        def trace(executable,loader,env):
            self.bwa_traces.append((str(executable),str(loader),dict(env)))
            return copy.deepcopy(self.bwa_dynamic)
        for name,value in (('_manifest',lambda:manifest),('SOURCE_BYTES',archive.stat().st_size),('SOURCE_SHA256',digest(archive))):
            patcher=patch.object(bwa,name,value);patcher.start();self.addCleanup(patcher.stop)
        patcher=patch.object(fastp,'_dynamic_dependencies',trace);patcher.start();self.addCleanup(patcher.stop)
        return self.bwa_assets

    def make_reference_cache(self):
        # This known test-owned shim stands in for samtools only for the
        # integration contract. The fixture sequence's MD5 and FAI are exact.
        return reference_module.prepare_preprocessing_reference(self.options['reference_fasta'],self.prefix,
            self.root/'reference_cache',host_memory_limit_gib=2,host_memory_reserve_gib=.01)

    def invoke(self, output, *, fastp_assets=None, bwa_assets=None, reference_assets=None, resume=False, corrupt_copied_md5=False,
               stop_at_training=True, after_stage=None, **options_override):
        owner=self;runner_class=pipeline._Runner;calls=[];runners=[]
        class Capture(runner_class):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs);runners.append(self)
            def stage(self,name,commands,outputs,future_gib=1):
                record=dict(stage=name,commands=commands,outputs=list(outputs),runtime_env=dict(self.env))
                calls.append(record)
                if name=='BaseRecalibrator' and stop_at_training:raise _ReachedTraining()
                if name in ('reference_dictionary','reference_validation_reuse'):
                    if name=='reference_dictionary' and (fastp_assets is not None or bwa_assets is not None):
                        # A fake libz fixture cannot load a real Python ELF.
                        # This branch only tests fastp dispatch; dictionary
                        # execution/reuse is independently covered below.
                        Path(outputs[0]).write_bytes(Path(owner.options['reference_fasta']).with_suffix('.dict').read_bytes())
                    else:
                        for command in commands:
                            subprocess.run(command,check=True,env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    if corrupt_copied_md5:
                        path=Path(outputs[0]);path.write_text(path.read_text().replace('M5:f1f8f4bf413b16ad135722aa4591043e','M5:00000000000000000000000000000000'))
                else:
                    for output_file in outputs:
                        path=Path(output_file);path.parent.mkdir(parents=True,exist_ok=True);path.write_text('contract output')
                record['output_bindings']=[bound(path) for path in outputs]
                if after_stage is not None:after_stage(name, self.env)
                return dict(status='complete',exit_codes=[0],outputs=record['output_bindings'])
        self.last_calls=calls;self.last_runners=runners
        effective_options=dict(self.options,**options_override)
        limits=dict(cpu_quota_count=4,finite_cpu_quotas=[4],available_bytes=64*2**30,admission_available_bytes=64*2**30)
        with patch.object(pipeline,'_Runner',Capture),patch.object(pipeline,'_limits',return_value=limits),\
                patch.object(pipeline.os,'sched_getaffinity',return_value={0,1,2,3}),patch.object(pipeline.os,'sched_setaffinity'),\
                patch.object(pipeline.shutil,'disk_usage',return_value=type('Disk',(),{'free':100*2**30})()),\
                patch.dict(os.environ,{'PIPELINE_CONTRACT_MARKER':'retained','PYTHONPATH':'must_be_removed',
                    'JAVA_TOOL_OPTIONS':'-javaagent:/contract/injected.jar',
                    '_JAVA_OPTIONS':'-Xmx999g','JDK_JAVA_OPTIONS':'--class-path=/contract/classes',
                    'CLASSPATH':'/contract/classes'},clear=True):
            try:
                pipeline.single_subj_preprocessing(**effective_options,output_dir=output,bqsr_mode='original',
                    fastp_source_patch_assets=fastp_assets,bwa_source_patch_assets=bwa_assets,
                    reference_validation_assets=reference_assets,resume=resume)
            except _ReachedTraining:pass
        self.last_calls=calls;self.last_runners=runners
        return calls

    def test_public_fastp_dictionary_and_json_assets_bind_actual_argv_and_runtime_env(self):
        self.prepare_inputs();assets=self.make_fastp()
        # With an explicit source build the original installed fastp is not a
        # dependency. This catches accidental fall-through to run.tool('fastp').
        (self.prefix/'bin/fastp').unlink()
        for label,value in [('dictionary',assets),('json_path',self.fastp_asset_file)]:
            with self.subTest(kind=label):
                self.traces.clear();out=self.root/('fastp_'+label)
                calls=self.invoke(out,fastp_assets=value)
                record=next(row for row in calls if row['stage']=='fastp');argv=record['commands'][0]
                self.assertEqual(argv,[assets['executable']['path'],'-i',str(self.options['read1']),'-I',str(self.options['read2']),
                    '-o',str(out/'files/clean_1.fq.gz'),'-O',str(out/'files/clean_2.fq.gz'),
                    '-j',str(out/'files/fastp.json'),'-h',str(out/'files/fastp.html'),'-w','1'])
                plan=json.loads((out/'plan.private.json').read_text())
                self.assertEqual(plan['fastp_source_patch_assets'],assets)
                self.assertEqual(plan['tools']['fastp'],assets['executable'])
                self.assertEqual(plan['tools']['fastp_build_receipt'],bound(assets['buildReceipt']))
                self.assertIn('build_preprocessing_fastp.py',plan['source_sha256'])
                if label=='json_path':self.assertEqual(plan['tools']['fastp_asset_record'],bound(value))
                self.assertEqual(len(self.traces),2)
                preflight=self.traces[0][2];actual=self.traces[1][2]
                self.assertEqual(actual,record['runtime_env'])
                self.assertEqual(actual,self.last_runners[0].env)
                self.assertEqual(preflight['PATH'],str(self.prefix/'bin')+':/usr/bin:/bin')
                self.assertEqual(preflight['LD_LIBRARY_PATH'],str(self.prefix/'lib'))
                for name in ('PYTHONPATH','JAVA_TOOL_OPTIONS','_JAVA_OPTIONS','JDK_JAVA_OPTIONS','CLASSPATH'):
                    self.assertNotIn(name,preflight)
                    self.assertNotIn(name,actual)
                self.assertEqual(actual['PIPELINE_CONTRACT_MARKER'],'retained')
                self.assertEqual(actual['CUDA_VISIBLE_DEVICES'],'')
                self.assertEqual(actual['GATK_LOCAL_JAR'],str(self.prefix/'share/gatk/gatk-package-4.2.0.0-local.jar'))
                self.assertEqual({k:v for k,v in actual.items() if k not in ('GATK_LOCAL_JAR','CUDA_VISIBLE_DEVICES')},preflight)

    def test_java_environment_overrides_do_not_change_resume_plan_or_actual_environment(self):
        self.prepare_inputs();out=self.root/'java_override_resume'
        self.invoke(out)
        plan=(out/'plan.private.json').read_bytes()
        self.invoke(out,resume=True)
        self.assertEqual((out/'plan.private.json').read_bytes(),plan)
        for runner in self.last_runners:
            for name in ('JAVA_TOOL_OPTIONS','_JAVA_OPTIONS','JDK_JAVA_OPTIONS','CLASSPATH'):
                self.assertNotIn(name,runner.env)
        training=next(row for row in self.last_calls if row['stage']=='BaseRecalibrator')
        self.assertIn('-Xmx4g',training['commands'][0][2])

    def test_public_fastp_resume_rechecks_modified_binary_before_any_stage(self):
        self.prepare_inputs();assets=self.make_fastp();out=self.root/'resume_fastp'
        self.invoke(out,fastp_assets=assets)
        before=(out/'plan.private.json').read_bytes();self.traces.clear()
        Path(assets['executable']['path']).write_bytes(b'modified after a completed contract stage')
        with self.assertRaisesRegex(RuntimeError,'artifact identity'):
            self.invoke(out,fastp_assets=assets,resume=True)
        self.assertEqual((out/'plan.private.json').read_bytes(),before)
        self.assertEqual(self.traces,[])

    def test_public_fastp_resume_rejects_a_relabelled_assets_record(self):
        self.prepare_inputs();assets=self.make_fastp();out=self.root/'resume_label'
        self.invoke(out,fastp_assets=assets)
        changed=copy.deepcopy(assets);changed['variant']='original'
        with self.assertRaisesRegex(RuntimeError,'exact assets'):
            self.invoke(out,fastp_assets=changed,resume=True)

    def test_public_bwa_assets_and_json_path_dispatch_bound_mapper_and_freeze_provenance(self):
        import shlex
        self.prepare_inputs();assets=self.make_bwa();(self.prefix/'bin/bwa').unlink()
        for label,value in [('dictionary',assets),('json_path',self.bwa_asset_file)]:
            with self.subTest(kind=label):
                self.bwa_traces.clear();out=self.root/('bwa_'+label);calls=self.invoke(out,bwa_assets=value)
                mapping=next(row for row in calls if row['stage']=='mapping_sort_index')
                script=mapping['commands'][0][2];argv=shlex.split(script.splitlines()[1].split(' | ',1)[0])
                self.assertEqual(argv[:2],[assets['executable']['path'],'mem'])
                self.assertEqual(argv[argv.index('-K')+1],'10000000')
                # The established pipeline reserves one of four CPUs for sort.
                self.assertEqual(argv[argv.index('-t')+1],'3')
                plan=json.loads((out/'plan.private.json').read_text())
                self.assertEqual(plan['bwa_source_patch_assets'],assets)
                self.assertEqual(plan['tools']['bwa'],assets['executable'])
                self.assertEqual(plan['tools']['bwa_build_receipt'],bound(assets['buildReceipt']))
                if label=='json_path':self.assertEqual(plan['tools']['bwa_asset_record'],bound(value))
                self.assertIn('build_preprocessing_bwa.py',plan['source_sha256'])
                self.assertIn('build_preprocessing_fastp.py',plan['source_sha256'])
                self.assertIn('_preprocessing_bwa/mate_profile_contract.c',plan['source_sha256'])
                self.assertEqual(len(self.bwa_traces),2)
                self.assertEqual(self.bwa_traces[-1][2],mapping['runtime_env'])
                self.assertEqual(self.bwa_traces[-1][2]['LD_LIBRARY_PATH'],str(self.prefix/'lib'))

    def test_cached_complete_rechecks_source_assets_after_resumed_stages(self):
        self.prepare_inputs();assets=self.make_bwa();out=self.root/'bwa_cached_complete'
        self.invoke(out,bwa_assets=assets)
        # A previously committed completion can have no final output bindings
        # in this isolated contract. Runtime source assets must still be
        # checked after the final resumed stage, before returning its receipt.
        atomic(out/'complete.private.json',{'final_output_bindings':[]})
        def change_after_normalize(name,env):
            if name=='normalize':
                Path(assets['dynamic_libraries']['libz.so.1']['path']).write_bytes(b'changed after preflight')
        with self.assertRaisesRegex(RuntimeError,'dynamic library changed'):
            self.invoke(out,bwa_assets=assets,resume=True,stop_at_training=False,
                        after_stage=change_after_normalize)

    def test_public_bwa_refuses_wrong_aligner_and_conflicting_explicit_binary(self):
        self.prepare_inputs();assets=self.make_bwa()
        with self.assertRaisesRegex(ValueError,'require aligner=bwa'):
            self.invoke(self.root/'wrong_mapper',bwa_assets=assets,aligner='bwa-mem2')
        with self.assertRaisesRegex(ValueError,'differs from the bound BWA'):
            self.invoke(self.root/'wrong_binary',bwa_assets=assets,aligner_executable=self.prefix/'bin/bwa')
        self.assertFalse((self.root/'wrong_binary/plan.private.json').exists())

    def test_public_bwa_requires_same_conda_prefix(self):
        self.prepare_inputs();assets=self.make_bwa(self.root/'another_conda')
        with self.assertRaisesRegex(ValueError,'same Conda prefix'):
            self.invoke(self.root/'wrong_bwa_conda',bwa_assets=assets)
        self.assertFalse((self.root/'wrong_bwa_conda/plan.private.json').exists())

    def test_public_bwa_resume_rechecks_artifacts_and_rejects_relabelled_plan(self):
        self.prepare_inputs();assets=self.make_bwa();out=self.root/'bwa_resume'
        self.invoke(out,bwa_assets=assets);before=(out/'plan.private.json').read_bytes();self.bwa_traces.clear()
        changed=copy.deepcopy(assets);changed['variant']='original'
        with self.assertRaisesRegex(RuntimeError,'exact assets'):
            self.invoke(out,bwa_assets=changed,resume=True)
        Path(assets['executable']['path']).write_bytes(b'changed mapper after committed stages')
        with self.assertRaisesRegex(RuntimeError,'artifact identity'):
            self.invoke(out,bwa_assets=assets,resume=True)
        self.assertEqual((out/'plan.private.json').read_bytes(),before);self.assertEqual(self.bwa_traces,[])

    def test_public_bwa_rechecks_libraries_before_mapping_and_before_final_commit(self):
        self.prepare_inputs();assets=self.make_bwa();library=Path(assets['dynamic_libraries']['libz.so.1']['path'])
        original=library.read_bytes()
        def after_cleaning(name,env):
            if name=='fastp':library.write_bytes(b'changed before mapping')
        with self.assertRaisesRegex(RuntimeError,'dynamic library changed'):
            self.invoke(self.root/'bwa_changed_before_mapping',bwa_assets=assets,after_stage=after_cleaning)
        self.assertNotIn('mapping_sort_index',[row['stage'] for row in self.last_calls])
        # A new physical binding is required after any write, even restored bytes.
        library.write_bytes(original)
        self.bwa_dynamic['libz.so.1']=bound(library)
        assets['dynamic_libraries']=copy.deepcopy(self.bwa_dynamic)
        core={key:value for key,value in assets.items() if key not in ('buildReceipt','build_receipt_sha256')}
        atomic(assets['buildReceipt'],dict(schema=bwa._BUILD_SCHEMA,status=bwa._STATUS,artifacts=core,
            Conda_package_records_unchanged=True,source_files_unchanged=True,toolchain_zlib_unchanged=True))
        assets['build_receipt_sha256']=digest(assets['buildReceipt'])
        def after_genotyping(name,env):
            if name=='genotype':library.write_bytes(b'changed after mapping before final commit')
        out=self.root/'bwa_changed_at_final'
        with patch.object(pipeline,'vcf_oracle',return_value=dict(variants=0,chromosome_counts={})),                self.assertRaisesRegex(RuntimeError,'dynamic library changed'):
            self.invoke(out,bwa_assets=assets,stop_at_training=False,after_stage=after_genotyping)
        self.assertIn('mapping_sort_index',[row['stage'] for row in self.last_calls])
        self.assertIn('genotype',[row['stage'] for row in self.last_calls])
        self.assertFalse((out/'complete.private.json').exists())

    def test_public_shared_reference_skips_rescan_and_copies_bound_dictionary(self):
        self.prepare_inputs();assets=self.make_reference_cache()
        self.assertEqual(self.marker.read_text(),'dict\n')
        expected_fasta=assets['inputs'][str(self.options['reference_fasta'])]
        self.assertEqual(expected_fasta['sha256'],digest(self.options['reference_fasta']))
        original_bound=reference_module.bound
        def no_fasta_rescan(path):
            self.assertNotEqual(Path(path).resolve(),Path(self.options['reference_fasta']).resolve())
            return original_bound(path)
        out=self.root/'cached_reference'
        with patch.object(reference_module,'bound',side_effect=no_fasta_rescan):
            calls=self.invoke(out,reference_assets=assets)
        self.assertNotIn('reference_dictionary',[row['stage'] for row in calls])
        copy_row=next(row for row in calls if row['stage']=='reference_validation_reuse')
        command=copy_row['commands'][0]
        self.assertEqual(command[0],sys.executable)
        self.assertEqual(command[-2:], [assets['computed_dictionary']['path'],str(out/'files/reference.recomputed.dict')])
        self.assertEqual(self.marker.read_text(),'dict\n')
        self.assertEqual(Path(command[-1]).read_bytes(),Path(assets['computed_dictionary']['path']).read_bytes())
        self.assertEqual(copy_row['output_bindings'],[bound(command[-1])])
        self.assertEqual(reference_module.dictionary_sequences(command[-1]),reference_module.dictionary_sequences(Path(self.options['reference_fasta']).with_suffix('.dict')))
        plan=json.loads((out/'plan.private.json').read_text())
        self.assertEqual(plan['reference_validation_assets'],assets)
        self.assertEqual(plan['tools']['reference_validation_receipt'],assets['receipt'])
        self.assertEqual(plan['tools']['reference_computed_dictionary'],assets['computed_dictionary'])
        self.assertEqual(plan['tools']['reference_copy_python'],bound(sys.executable))
        self.assertIn('preprocessing_reference.py',plan['source_sha256'])

    def test_public_reference_default_still_scans_dictionary(self):
        self.prepare_inputs();calls=self.invoke(self.root/'uncached')
        self.assertIn('reference_dictionary',[row['stage'] for row in calls])
        self.assertNotIn('reference_validation_reuse',[row['stage'] for row in calls])
        self.assertEqual(self.marker.read_text(),'dict\n')

    def test_public_reference_change_rejected_before_copy_or_fastp(self):
        self.prepare_inputs();assets=self.make_reference_cache();out=self.root/'cached_resume'
        self.invoke(out,reference_assets=assets)
        reference=Path(self.options['reference_fasta']);reference.write_bytes(reference.read_bytes().replace(b'ACGT',b'ACGA'))
        with self.assertRaisesRegex(ValueError,'identity changed'):
            self.invoke(out,reference_assets=assets,resume=True)
        self.assertEqual(self.marker.read_text(),'dict\n')

    def test_public_reference_fai_geometry_change_rejected_even_if_axis_unchanged(self):
        self.prepare_inputs();assets=self.make_reference_cache()
        fai=Path(str(self.options['reference_fasta'])+'.fai');fai.write_text('chr1\t4\t7\t4\t5\n')
        with self.assertRaisesRegex(ValueError,'identity changed'):
            self.invoke(self.root/'bad_fai',reference_assets=assets)
        self.assertEqual(self.marker.read_text(),'dict\n')

    def test_public_cached_copy_still_checks_md5_against_supplied_dictionary(self):
        self.prepare_inputs();assets=self.make_reference_cache()
        with self.assertRaisesRegex(ValueError,'sequence content differs'):
            self.invoke(self.root/'bad_copied_md5',reference_assets=assets,corrupt_copied_md5=True)
        self.assertEqual(self.marker.read_text(),'dict\n')


if __name__=='__main__':unittest.main()
