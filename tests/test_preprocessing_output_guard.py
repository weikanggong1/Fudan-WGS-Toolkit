"""Output namespace contracts, using test-owned files and command shims only."""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from fudan_wgs_toolkit import single_preprocessing as pipeline
import test_preprocessing_sources_integration as _source_contracts


class OutputNamespaceContracts(unittest.TestCase):
    setUp=_source_contracts.PublicSourceIntegrationContracts.setUp
    prepare_inputs=_source_contracts.PublicSourceIntegrationContracts.prepare_inputs
    invoke=_source_contracts.PublicSourceIntegrationContracts.invoke
    make_fastp=_source_contracts.PublicSourceIntegrationContracts.make_fastp

    def assert_rejected_without_writes(self,output,**options):
        names=('run.lock','plan.private.json','stages','runtime_build','bqsr_build')
        before={name:(output/name).exists() for name in names}
        with self.assertRaisesRegex(ValueError,'protected|symlink|aliases'):
            self.invoke(output,**options)
        self.assertEqual(self.last_calls,[])
        self.assertEqual({name:(output/name).exists() for name in names},before)

    def test_main_rejects_raw_output_overlap_with_installed_or_source_fastp(self):
        self.prepare_inputs();assets=self.make_fastp()
        for label,source in (('installed',None),('source',assets)):
            with self.subTest(label=label):
                output=self.root/label;(output/'files').mkdir(parents=True,mode=0o700);output.chmod(0o700)
                raw=output/'files/clean_1.fq.gz';raw.write_bytes(b'test-owned raw sentinel')
                self.assert_rejected_without_writes(output,fastp_assets=source,read1=raw)
                self.assertEqual(raw.read_bytes(),b'test-owned raw sentinel')

    def test_main_rejects_each_readonly_input_namespace_before_plan(self):
        self.prepare_inputs()
        for kind in ('reference','fai','dict','known_site','site_index','bwa_index','tool','asset_json','asset_source'):
            with self.subTest(kind=kind):
                output=self.root/('overlap_'+kind);output.mkdir(mode=0o700)
                item=output/'protected';item.write_bytes(b'unchanged input')
                options={}
                if kind=='reference':options['reference_fasta']=item
                elif kind in ('fai','dict'):
                    # A protected reference's sidecar can be a symlink into the
                    # output even when the reference itself remains external.
                    reference=Path(self.options['reference_fasta'])
                    sidecar=Path(str(reference)+'.fai') if kind=='fai' else reference.with_suffix('.dict')
                    old=sidecar.read_bytes();sidecar.unlink();sidecar.symlink_to(item)
                elif kind in ('known_site','site_index'):
                    if kind=='known_site':options['known_sites']=[item,*self.options['known_sites'][1:]]
                    else:
                        sidecar=Path(str(self.options['known_sites'][0])+'.tbi')
                        old=sidecar.read_bytes();sidecar.unlink();sidecar.symlink_to(item)
                elif kind=='bwa_index':
                    sidecar=Path(str(self.options['reference_fasta'])+'.bwt')
                    old=sidecar.read_bytes();sidecar.unlink();sidecar.symlink_to(item)
                elif kind=='tool':options['aligner_executable']=item
                elif kind=='asset_json':
                    item.write_text('{}');options['fastp_assets']=item
                else:options['fastp_assets']={'source_tree':str(item)}
                try:
                    before=item.read_bytes();self.assert_rejected_without_writes(output,**options)
                    self.assertEqual(item.read_bytes(),before)
                finally:
                    if kind in ('fai','dict','site_index','bwa_index'):
                        sidecar.unlink();sidecar.write_bytes(old)

    def test_original_tool_prefix_and_project_sources_are_protected(self):
        self.prepare_inputs()
        for root in (self.prefix,self.root,Path(pipeline.__file__).parent):
            with self.subTest(root=root),self.assertRaisesRegex(ValueError,'protected'):
                pipeline._guard_preprocessing_output(root,[self.prefix,Path(pipeline.__file__).parent])

    def test_fixed_output_hardlink_cannot_alias_external_raw_input(self):
        self.prepare_inputs();output=self.root/'hardlink';(output/'files').mkdir(parents=True,mode=0o700);output.chmod(0o700)
        raw=Path(self.options['read1']);before=raw.read_bytes()
        os.link(raw,output/'files/clean_1.fq.gz')
        self.assert_rejected_without_writes(output)
        self.assertEqual(raw.read_bytes(),before)

    def test_hardlink_to_unenumerated_source_or_library_is_also_refused(self):
        self.prepare_inputs();output=self.root/'other_hardlink';(output/'files').mkdir(parents=True,mode=0o700)
        source=self.root/'not_enumerated_source.cpp';source.write_bytes(b'preserve source')
        os.link(source,output/'files/fastp.json')
        with self.assertRaisesRegex(ValueError,'unverifiable hardlinks'):
            pipeline._guard_preprocessing_output(output,[self.options['read1'],self.prefix])
        self.assertEqual(source.read_bytes(),b'preserve source')

    def test_shared_environment_removes_external_java_and_python_injection(self):
        names=('PYTHONPATH','JAVA_TOOL_OPTIONS','_JAVA_OPTIONS','JDK_JAVA_OPTIONS','CLASSPATH')
        with patch.dict(os.environ,{**{name:'external override' for name in names},'UNRELATED_MARKER':'preserved'}):
            before={name:os.environ[name] for name in names}
            environment=pipeline._tool_environment(self.prefix)
            for name in names:self.assertNotIn(name,environment)
            self.assertEqual(environment['UNRELATED_MARKER'],'preserved')
            self.assertEqual({name:os.environ[name] for name in names},before)

    def test_output_directory_and_file_symlinks_do_not_escape_namespace(self):
        self.prepare_inputs();external=self.root/'external';external.mkdir();sentinel=external/'sentinel';sentinel.write_bytes(b'preserve')
        for label in ('files_directory','stages_directory','output_file','custom_build'):
            with self.subTest(label=label):
                output=self.root/label;output.mkdir(mode=0o700);options={}
                if label=='output_file':
                    (output/'files').mkdir();(output/'files/fastp.json').symlink_to(sentinel)
                elif label=='custom_build':
                    (output/'custom').symlink_to(external,target_is_directory=True)
                    options['runtime_build_dir']=output/'custom'
                else:(output/label.split('_')[0]).symlink_to(external,target_is_directory=True)
                self.assert_rejected_without_writes(output,**options)
                self.assertEqual(sentinel.read_bytes(),b'preserve')

    def test_default_generated_adapter_builds_remain_legal_for_resume(self):
        self.prepare_inputs();output=self.root/'normal';output.mkdir(mode=0o700)
        for name in ('runtime_build','bqsr_build','gpu_build'):
            (output/name).mkdir();(output/name/'assets.private.json').write_text('{}')
        pipeline._guard_preprocessing_output(output,[self.options['read1'],self.prefix,Path(pipeline.__file__).parent])
        calls=self.invoke(output)
        mapping=next(v for v in calls if v['stage']=='mapping_sort_index')['commands'][0][2]
        self.assertIn('mem -K 10000000 -M -Y -t 3',mapping)
        self.assertNotIn('--dont_overwrite',next(v for v in calls if v['stage']=='fastp')['commands'][0])

    def test_unknown_input_inode_does_not_assert_hardlink_independence(self):
        self.prepare_inputs();output=self.root/'unknown_inode';(output/'files').mkdir(parents=True,mode=0o700)
        raw=Path(self.options['read1']);os.link(raw,output/'files/clean_1.fq.gz')
        real_identity=pipeline.identity
        def unavailable_inode(path):return dict(real_identity(path),inode=0)
        with patch.object(pipeline,'identity',side_effect=unavailable_inode):
            with self.assertRaisesRegex(ValueError,'unverifiable hardlinks'):
                pipeline._guard_preprocessing_output(output,[raw])


if __name__=='__main__':unittest.main()
