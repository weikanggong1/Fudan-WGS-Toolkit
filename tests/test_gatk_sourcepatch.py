"""Build/dispatch/resume contracts; fake jars are not scientific benchmarks."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from fudan_wgs_toolkit._preprocessing_io import bound, digest, identity
from fudan_wgs_toolkit import build_preprocessing_gatk_sourcepatch as module
from fudan_wgs_toolkit import build_preprocessing_bqsr as parallel
from fudan_wgs_toolkit import single_preprocessing as pipeline


def write_json(path, value):
    Path(path).write_text(json.dumps(value))


def fake_patch_assets(root, prefix):
    """Satisfy format/hash contracts without claiming actual compiled classes."""
    root.mkdir(parents=True)
    jar=prefix/'share/gatk/gatk-package-4.2.0.0-local.jar'
    jar.parent.mkdir(parents=True,exist_ok=True)
    if not jar.exists():jar.write_bytes(b'contract original jar')
    files={}
    for name in ('java','javac','jar','python'):
        tool=prefix/'bin'/name;tool.parent.mkdir(parents=True,exist_ok=True)
        if not tool.exists():tool.write_bytes(('contract '+name).encode())
        files[name]=str(tool)
    overlay=root/'overlay.jar';overlay.write_bytes(b'contract compiled overlay')
    files.update(sourcePatchJar=str(overlay),gatkJar=str(jar))
    binding=dict(schema=module._CLASS_SCHEMA,sourcePatchJar=str(overlay),gatkJar=str(jar),
        classes={name:hashlib.sha256(name.encode()).hexdigest() for name in sorted(module._REQUIRED_CLASSES)},
        original_main_class_sha256='a'*64)
    binding_file=root/'class_binding.private.json';write_json(binding_file,binding)
    probe=dict(schema='fudan-original-gatk-sourcepatch-runtime-v1',status='classes_verified_before_training',
        entrypoint='probe',binding_sha256=digest(binding_file),
        classes={name:dict(code_source=str(overlay),sha256=sha) for name,sha in binding['classes'].items()},
        original_main_code_source=str(jar),original_main_class_sha256='a'*64)
    probe_file=root/'class_probe.private.json';write_json(probe_file,probe)
    files.update(classBinding=str(binding_file),classProbe=str(probe_file))
    package=Path(module.__file__).with_name('_preprocessing_gatk_sourcepatch')
    core=dict(schema=module._ASSET_SCHEMA,status=module._STATUS,**files,condaPrefix=str(prefix),
        assetsFile=str(root/'assets.private.json'),mainClass=module._MAIN,upstream_commit=module._COMMIT,
        original_GATK_version=module._VERSION,builder_sha256=digest(module.__file__),
        source_manifest_sha256=digest(package/'source_manifest.json'),
        environment_records={p.name:digest(p) for p in (prefix/'conda-meta').glob('*.json')},
        sha256={key:digest(path) for key,path in files.items()},identities={key:identity(path) for key,path in files.items()},
        original_source_cache={},scientific_validation_completed=False)
    receipt=root/'build_receipt.private.json'
    write_json(receipt,dict(schema=module._BUILD_SCHEMA,status=module._STATUS,
        artifacts=core,Conda_package_records_unchanged=True))
    assets=dict(core,buildReceipt=str(receipt),build_receipt_sha256=digest(receipt))
    write_json(root/'assets.private.json',assets)
    return assets


def fake_parallel_assets(root, prefix, patch_assets):
    root.mkdir(parents=True)
    jar=root/'adapter.jar';jar.write_bytes(b'contract adapter')
    files=dict(adapterJar=str(jar),gatkJar=patch_assets['gatkJar'],java=patch_assets['java'])
    manifest=Path(parallel.__file__).with_name('_preprocessing_bqsr')/'source_manifest.json'
    core=dict(schema=parallel._ASSET_SCHEMA,status='compiled_help_passed_not_science',**files,
        condaPrefix=str(prefix),classpath=str(jar)+':'+files['gatkJar'],entrypoint=parallel._ENTRYPOINT,
        feature_cache_lookahead_bases=1000,sha256={k:digest(v) for k,v in files.items()},
        identities={k:identity(v) for k,v in files.items()},builder_sha256=digest(parallel.__file__),
        source_manifest_sha256=digest(manifest))
    receipt=root/'build_receipt.private.json'
    write_json(receipt,dict(schema=parallel._BUILD_SCHEMA,status=core['status'],
        artifacts=core,Conda_package_records_unchanged=True))
    assets=dict(core,buildReceipt=str(receipt),build_receipt_sha256=digest(receipt))
    write_json(root/'assets.private.json',assets)
    return assets


class SourcePatchContracts(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.prefix=self.root/'env';(self.prefix/'conda-meta').mkdir(parents=True)
        write_json(self.prefix/'conda-meta/jdk.json',dict(name='openjdk',version='8'))
        self.assets=fake_patch_assets(self.root/'build',self.prefix)

    def test_source_package_complete_and_fixed(self):
        _,_,manifest=module._manifest()
        self.assertEqual(len(manifest['sources']),2)
        self.assertEqual(set(manifest['files']),module._PACKAGE_FILES)
        for entry in manifest['sources']:
            self.assertNotEqual(entry['upstream_sha256'],entry['modified_sha256'])

    def test_actual_gatk_version_banner_accepts_v_but_rejects_other_versions(self):
        actual = ('The Genome Analysis Toolkit (GATK) v4.2.0.0\n'
                  'HTSJDK Version: 2.24.0\nPicard Version: 2.25.0\n')
        self.assertTrue(module._expected_gatk_version(actual))
        self.assertTrue(module._expected_gatk_version(actual.replace(' v4.', ' 4.')))
        self.assertTrue(module._expected_gatk_version(actual.replace('\n', '\r\n')))
        for version in ('4.2.0.1', '14.2.0.0', '4.2.0.01', '4.2.0.0-extra', '4.2.0.0.1'):
            with self.subTest(version=version):
                self.assertFalse(module._expected_gatk_version(actual.replace('4.2.0.0', version)))
        self.assertFalse(module._expected_gatk_version('Dependency 4.2.0.0\n'))
        self.assertFalse(module._expected_gatk_version(actual+actual))

    def test_committed_assets_are_verified_without_executing_a_java_process(self):
        self.assertEqual(module.verify_gatk_source_patch_assets(self.assets),self.assets)
        self.assertEqual(module.verify_gatk_source_patch_assets(self.assets['assetsFile']),self.assets)
        other=self.root/'other';other.mkdir()
        with self.assertRaisesRegex(RuntimeError,'different Conda'):
            module.verify_gatk_source_patch_assets(self.assets,conda_prefix=other)
        with self.assertRaisesRegex(RuntimeError,'different installed'):
            module.verify_gatk_source_patch_assets(self.assets,gatk_jar=self.assets['sourcePatchJar'])

    def test_build_receipt_source_environment_and_artifact_tampering_rejected(self):
        original=copy.deepcopy(self.assets)
        for name in ('receipt','environment','artifact','missing_asset'):
            with self.subTest(name=name):
                if name=='receipt':
                    target=Path(original['buildReceipt']);old=target.read_bytes();target.write_bytes(old+b'\n')
                elif name=='environment':
                    target=self.prefix/'conda-meta/jdk.json';old=target.read_bytes();target.write_bytes(old+b'\n')
                elif name=='artifact':
                    target=Path(original['sourcePatchJar']);old=target.read_bytes();target.write_bytes(old+b'changed')
                else:
                    target=Path(original['assetsFile']);old=target.read_bytes();target.write_text('{}')
                try:
                    with self.assertRaisesRegex(RuntimeError,'receipt|records changed|artifact identity|asset file differs'):
                        module.verify_gatk_source_patch_assets(original)
                finally:target.write_bytes(old)
                # Restoring bytes changes inode timestamps. Rebind a NEW fake
                # build for the next independent rejection case.
                self.assets=fake_patch_assets(self.root/('next_'+name),self.prefix)
                original=copy.deepcopy(self.assets)
        package=self.root/'package';package.mkdir()
        builder=package/Path(module.__file__).name;shutil.copyfile(module.__file__,builder)
        shutil.copytree(Path(module.__file__).with_name('_preprocessing_gatk_sourcepatch'),package/'_preprocessing_gatk_sourcepatch')
        source=package/'_preprocessing_gatk_sourcepatch/assets/SourcePatchedBqsrMain.java'
        source.write_bytes(source.read_bytes()+b'\n// changed\n')
        with patch.object(module,'__file__',str(builder)):
            with self.assertRaisesRegex(RuntimeError,'asset changed'):
                module.verify_gatk_source_patch_assets(self.assets)

    def test_original_and_parallel_select_exact_classpaths_keep_known_site_order(self):
        argv=module.gatk_source_patch_training_argv(self.assets,
            ['BaseRecalibrator','-R','reference.fa','--known-sites','one.vcf.gz','--known-sites','two.vcf.gz'],
            audit_json=self.root/'runtime.json',java_options=['-Xmx4g'])
        self.assertEqual(argv[argv.index('-cp')+1],self.assets['sourcePatchJar']+':'+self.assets['gatkJar'])
        self.assertEqual(argv[argv.index(module._MAIN)+1],'original')
        adapter=fake_parallel_assets(self.root/'parallel',self.prefix,self.assets)
        options=dict(input_cram='reads.cram',reference_fasta='ref.fa',known_sites=['b.vcf.gz','a.vcf.gz'],
            recalibration_table='recal.table',canonical_bam='canonical.bam',audit_json='adapter.json',temporary_directory='tmp')
        before=copy.deepcopy(adapter)
        old=parallel.bqsr_training_argv(adapter,**options)
        new=parallel.bqsr_training_argv(adapter,**options,gatk_source_patch_assets=self.assets,
            source_patch_audit_json=self.root/'parallel_class.json')
        self.assertEqual(adapter,before)
        self.assertEqual(old[old.index('-cp')+1],adapter['classpath'])
        self.assertEqual(new[new.index('-cp')+1],self.assets['sourcePatchJar']+':'+adapter['classpath'])
        self.assertEqual(new[new.index(module._MAIN)+1],'parallel')
        self.assertNotIn(parallel._ENTRYPOINT,new)  # launcher invokes the same adapter main.
        self.assertEqual([new[i+1] for i,x in enumerate(new) if x=='--known-sites'],options['known_sites'])
        self.assertEqual(old[old.index(parallel._ENTRYPOINT)+1:],new[new.index(module._MAIN)+3:])
        self.assertNotIn('-L',new)

    def test_reject_non_training_or_classpath_replacement_before_launch(self):
        for tool in ('ApplyBQSR','HaplotypeCaller','CollectWgsMetrics'):
            with self.subTest(tool=tool),self.assertRaisesRegex(ValueError,'BaseRecalibrator'):
                module.gatk_source_patch_training_argv(self.assets,[tool],audit_json='never.json')
        for option in ('-jar','--class-path=x','-Djava.class.path=x','-javaagent:x','-Xbootclasspath/p:x',
                       '-Dfudan.gatkpatch.binding=x'):
            with self.subTest(option=option),self.assertRaisesRegex(ValueError,'bound source-patch launcher'):
                module.gatk_source_patch_training_argv(self.assets,['BaseRecalibrator'],audit_json='never.json',java_options=[option])
        with self.assertRaisesRegex(ValueError,'separately committed'):
            module.gatk_source_patch_training_argv(self.assets,[],audit_json='x',entrypoint='parallel')
        self.assertFalse((self.root/'never.json').exists())

    def test_actual_runtime_audit_all_classes_not_a_status_label(self):
        audit=json.loads(Path(self.assets['classProbe']).read_text());audit['entrypoint']='original'
        target=self.root/'runtime.json';write_json(target,audit)
        module.verify_gatk_source_patch_audit(self.assets,target,entrypoint='original')
        for kind in ('entrypoint','class','binding','original'):
            changed=copy.deepcopy(audit)
            if kind=='entrypoint':changed['entrypoint']='parallel'
            elif kind=='class':changed['classes'].pop(module._MAIN)
            elif kind=='binding':changed['binding_sha256']='0'*64
            else:changed['original_main_code_source']=self.assets['sourcePatchJar']
            write_json(target,changed)
            with self.subTest(kind=kind),self.assertRaisesRegex(RuntimeError,'class-load audit'):
                module.verify_gatk_source_patch_audit(self.assets,target,entrypoint='original')

    def test_exact_patch_contexts_and_complete_file_list(self):
        # Tiny text is only a patch-parser contract; real fixed-source hashes
        # are checked separately, and this is not BQSR science data.
        package=self.root/'synthetic_package';(package/'assets').mkdir(parents=True)
        upstream=self.root/'upstream';upstream.mkdir();sources=[];chunks=[]
        for name in ('first.java','second.java'):
            old='class Original {}\n';new='// modified\n'+old
            (upstream/name).write_text(old)
            sources.append(dict(path=name,upstream_sha256=hashlib.sha256(old.encode()).hexdigest(),modified_sha256=hashlib.sha256(new.encode()).hexdigest()))
            chunks.append('--- a/'+name+'\n+++ b/'+name+'\n@@ -1 +1,2 @@\n+// modified\n '+old)
        patch_file=package/'assets/bqsr-allocations.patch';patch_file.write_text(''.join(chunks))
        manifest=dict(sources=sources)
        with patch.object(module,'_manifest',return_value=(package,None,manifest)):
            module._apply_patch(package,upstream,self.root/'derived')
            patch_file.write_text(''.join(chunks).replace(' class Original',' class Different',1))
            with self.assertRaisesRegex(RuntimeError,'context differs'):
                module._apply_patch(package,upstream,self.root/'rejected')
            patch_file.write_text(chunks[0])
            with self.assertRaisesRegex(RuntimeError,'omitted'):
                module._apply_patch(package,upstream,self.root/'missing')

    def test_nonfinite_budget_rejected_before_creating_build_or_child(self):
        for value in (float('nan'),float('inf'),True,0):
            output=self.root/'not_created'
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'finite'):
                module.build_preprocessing_gatk_sourcepatch(self.prefix,output,host_memory_limit_gib=value)
            self.assertFalse(output.exists())


class _CapturedThroughApply(Exception):pass


class PipelineDispatchContracts(unittest.TestCase):
    setUp = SourcePatchContracts.setUp
    def prepare_inputs(self):
        for name in ('fastp','bwa','samtools','sambamba','gatk','bcftools','plink2','gzip'):
            (self.prefix/'bin'/name).write_text('contract '+name)
        reference=self.root/'ref.fa';reference.write_text('>chr1\nACGT\n')
        (self.root/'ref.fa.fai').write_text('chr1\t4\t6\t4\t5\n')
        reference.with_suffix('.dict').write_text('@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:4\tM5:f1f8f4bf413b16ad135722aa4591043e\n')
        for ext in ('.amb','.ann','.bwt','.pac','.sa'):Path(str(reference)+ext).write_text('contract index')
        sites=[]
        for i in range(3):
            path=self.root/('site'+str(i)+'.vcf.gz');path.write_text('contract known sites');sites.append(path)
            Path(str(path)+'.tbi').write_text('contract index')
        reads=[]
        for i in range(2):
            path=self.root/('read'+str(i)+'.fq.gz');path.write_text('contract input');reads.append(path)
        return dict(read1=reads[0],read2=reads[1],reference_fasta=reference,known_sites=sites,
            sample_id='subject_a',conda_prefix=self.prefix,cpu_threads=4,cpu_affinity=[0,1,2,3],
            host_memory_limit_gib=16,host_memory_reserve_gib=1,qc_exit_fix=False)

    def invoke(self, options, output, *, mode, assets, resume=False, bqsr_build=None):
        real_runner=pipeline._Runner;captured=[];owner=self
        class Capture(real_runner):
            def stage(self, name, commands, outputs, future_gib=1):
                captured.append((name,commands,list(outputs)))
                if name=='ApplyBQSR':raise _CapturedThroughApply()
                for output_file in outputs:
                    path=Path(output_file);path.parent.mkdir(parents=True,exist_ok=True)
                    if name=='reference_dictionary':path.write_text(Path(options['reference_fasta']).with_suffix('.dict').read_text())
                    elif path.name=='bqsr_sourcepatch.private.json':
                        audit=json.loads(Path(assets['classProbe']).read_text())
                        audit['entrypoint']='parallel' if mode in ('parallel','auto') else 'original'
                        write_json(path,audit)
                    else:path.write_text('contract stage output')
                return dict(status='complete',outputs=[])
        limits=dict(cpu_quota_count=4,finite_cpu_quotas=[4],available_bytes=64*2**30,admission_available_bytes=64*2**30)
        with patch.object(pipeline,'_Runner',Capture),patch.object(pipeline,'_limits',return_value=limits),\
                patch.object(pipeline.os,'sched_getaffinity',return_value={0,1,2,3}),patch.object(pipeline.os,'sched_setaffinity'),\
                patch.object(pipeline.shutil,'disk_usage',return_value=type('Disk',(),{'free':100*2**30})()):
            try:
                pipeline.single_subj_preprocessing(**options,output_dir=output,bqsr_mode=mode,
                    bqsr_build_dir=bqsr_build,gatk_source_patch_assets=assets,resume=resume)
            except _CapturedThroughApply:pass
        return captured

    def test_public_original_parallel_auto_and_none_only_patch_training(self):
        options=self.prepare_inputs()
        adapter=fake_parallel_assets(self.root/'adapter_build',self.prefix,self.assets)
        for mode,enabled in [('original',False),('original',True),('parallel',True),('auto',True)]:
            with self.subTest(mode=mode,enabled=enabled):
                out=self.root/('run_'+mode+str(enabled))
                calls=self.invoke(options,out,mode=mode,assets=self.assets if enabled else None,bqsr_build=self.root/'adapter_build')
                by_name={name:(commands,outputs) for name,commands,outputs in calls}
                train=by_name['BaseRecalibrator'][0][0];apply=by_name['ApplyBQSR'][0][0]
                self.assertEqual(apply[0],str(self.prefix/'bin/gatk'))
                self.assertNotIn(module._MAIN,apply)
                if enabled:
                    expected=self.assets['sourcePatchJar']+':'+(adapter['classpath'] if mode!='original' else self.assets['gatkJar'])
                    self.assertEqual(train[train.index('-cp')+1],expected)
                    self.assertIn(out/'bqsr_sourcepatch.private.json',by_name['BaseRecalibrator'][1])
                    plan=json.loads((out/'plan.private.json').read_text())
                    self.assertEqual(plan['gatk_source_patch_assets']['class_binding_sha256'],self.assets['sha256']['classBinding'])
                    self.assertIn('gatk_source_patch_receipt',plan['tools'])
                else:
                    self.assertEqual(train[0],str(self.prefix/'bin/gatk'))
                    self.assertNotIn(module._MAIN,train)

    def test_public_resume_rejects_alternate_valid_source_patch_build(self):
        options=self.prepare_inputs();out=self.root/'resume_run'
        self.invoke(options,out,mode='original',assets=self.assets)
        other=fake_patch_assets(self.root/'other_build',self.prefix)
        with self.assertRaisesRegex(ValueError,'identical source, tools, inputs and options'):
            self.invoke(options,out,mode='original',assets=other,resume=True)


if __name__=='__main__':unittest.main()
