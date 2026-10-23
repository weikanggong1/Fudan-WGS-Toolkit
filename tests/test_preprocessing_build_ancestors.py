"""Run-local explicit build ancestors, test-owned paths and command shims only."""
import os
from pathlib import Path
import unittest
from fudan_wgs_toolkit import single_preprocessing as pipeline
import test_preprocessing_output_guard as guards


class GeneratedBuildAncestorContracts(unittest.TestCase):
    setUp=guards.OutputNamespaceContracts.setUp
    prepare_inputs=guards.OutputNamespaceContracts.prepare_inputs
    invoke=guards.OutputNamespaceContracts.invoke
    make_fastp=guards.OutputNamespaceContracts.make_fastp
    assert_rejected_without_writes=guards.OutputNamespaceContracts.assert_rejected_without_writes

    def test_each_public_explicit_build_rejects_symlink_ancestor_before_writes(self):
        self.prepare_inputs();external=self.root/'external';external.mkdir()
        sentinel=external/'sentinel';sentinel.write_bytes(b'preserved external')
        for option in ('runtime_build_dir','bqsr_build_dir','gpu_build_dir'):
            with self.subTest(option=option):
                output=self.root/option;output.mkdir(mode=0o700)
                (output/'alias').symlink_to(external,target_is_directory=True)
                self.assert_rejected_without_writes(output,**{option:output/'alias/missing'})
                self.assertEqual(sentinel.read_bytes(),b'preserved external')
                self.assertFalse((external/'missing').exists())

    def test_deep_missing_and_existing_leaves_reject_ancestor_without_external_scan(self):
        self.prepare_inputs();external=self.root/'deep_external';external.mkdir()
        (external/'present').mkdir();(external/'present/sentinel').write_bytes(b'original')
        for leaf in ('missing/deeper','present'):
            with self.subTest(leaf=leaf):
                output=self.root/leaf.replace('/','_');output.mkdir(mode=0o700)
                (output/'nested').mkdir();(output/'nested/alias').symlink_to(external,target_is_directory=True)
                with self.assertRaisesRegex(ValueError,'symlink'):
                    pipeline._guard_preprocessing_output(output,[self.options['read1']],
                        [output/'nested/alias'/leaf])
                self.assertEqual((external/'present/sentinel').read_bytes(),b'original')

    def test_parent_components_do_not_hide_an_existing_run_symlink(self):
        self.prepare_inputs();output=self.root/'dotdot';output.mkdir(mode=0o700)
        external=self.root/'dotdot_external';external.mkdir()
        (output/'alias').symlink_to(external,target_is_directory=True)
        # resolve() later removes the nonexistent component and reaches alias;
        # the guard must check the normalized ancestors too, before any mkdir.
        for path in (output/'absent/../alias/new',output/'alias/../new'):
            with self.subTest(path=path),self.assertRaisesRegex(ValueError,'symlink'):
                pipeline._guard_preprocessing_output(output,[self.options['read1']],[path])
        self.assertFalse((external/'new').exists())

    def test_clean_nested_run_build_and_existing_private_resume_files_are_legal(self):
        self.prepare_inputs();output=self.root/'valid_nested';output.mkdir(mode=0o700)
        build=output/'nested/native/build';build.mkdir(parents=True)
        artifact=build/'assets.private.json';artifact.write_text('{}')
        before=artifact.read_bytes()
        for path in (build,build/'future',output/'missing/../nested/native/build'):
            pipeline._guard_preprocessing_output(output,[self.options['read1'],self.prefix],[path])
        self.assertEqual(artifact.read_bytes(),before)
        self.assertFalse((build/'future').exists())

    def test_external_explicit_build_and_authenticated_alias_remain_outside_run_guard(self):
        self.prepare_inputs();output=self.root/'external_allowed';output.mkdir(mode=0o700)
        external=self.root/'trusted_external';external.mkdir();(external/'keep').write_bytes(b'private')
        alias=self.root/'trusted_external_alias';alias.symlink_to(external,target_is_directory=True)
        for path in (external,external/'new',alias/'new',output/'../trusted_external_alias/new'):
            pipeline._guard_preprocessing_output(output,[self.options['read1'],self.prefix],[path])
        self.assertEqual((external/'keep').read_bytes(),b'private')
        self.assertFalse((external/'new').exists())

    def test_normalized_run_build_still_checks_existing_output_hardlinks(self):
        self.prepare_inputs();output=self.root/'hardlink_dotdot';output.mkdir(mode=0o700)
        build=output/'nested/build';build.mkdir(parents=True)
        os.link(self.options['read1'],build/'existing.asset')
        with self.assertRaisesRegex(ValueError,'hardlinks|aliases'):
            pipeline._guard_preprocessing_output(output,[self.options['read1']],
                [output/'absent/../nested/build'])


if __name__=='__main__':unittest.main()
