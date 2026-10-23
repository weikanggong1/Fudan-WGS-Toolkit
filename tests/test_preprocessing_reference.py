"""Reference reuse contracts, not biological speed or precision benchmarks."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fudan_wgs_toolkit.preprocessing_reference import (
    prepare_preprocessing_reference, verify_preprocessing_reference,
    dictionary_sequences)


class ReferenceReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prefix = self.root/'environment'
        (self.prefix/'bin').mkdir(parents=True)
        (self.prefix/'conda-meta').mkdir()
        self.tool = self.prefix/'bin/samtools'
        self.reference = self.root/'reference.fasta'
        self.reference.write_bytes(b'>contig_a\nACGT\n')
        self.reference.with_suffix('.dict').write_text('@SQ\tSN:contig_a\tLN:4\tM5:'+hashlib.md5(b'ACGT').hexdigest()+'\n')
        Path(str(self.reference)+'.fai').write_text('contig_a\t4\t10\t4\t5\n')
        self.tool.write_text('#!/usr/bin/python3\nimport pathlib,sys\nassert sys.argv[1]=="dict"\npathlib.Path(sys.argv[3]).write_text(pathlib.Path(sys.argv[4]).with_suffix(".dict").read_text())\n')
        self.tool.chmod(0o700)
        self.cache = self.root/'shared_validation'

    def build(self):
        return prepare_preprocessing_reference(self.reference, self.prefix, self.cache,
            host_memory_limit_gib=2,host_memory_reserve_gib=.01)

    def verify(self, assets):
        return verify_preprocessing_reference(assets,reference_fasta=self.reference,conda_prefix=self.prefix)

    def test_reuse_does_not_scan_reference_or_execute_dictionary_again(self):
        assets = self.build()
        from fudan_wgs_toolkit import preprocessing_reference as module
        real_bound = module.bound
        def no_full_reference_hash(path):
            self.assertNotEqual(Path(path).resolve(), self.reference.resolve())
            return real_bound(path)
        with patch.object(module,'bound',side_effect=no_full_reference_hash),patch('fudan_wgs_toolkit.single_preprocessing._Runner.stage',side_effect=AssertionError('Unexpected reference rescan')):
            self.assertEqual(self.verify(assets),assets)
            reused = prepare_preprocessing_reference(self.reference,self.prefix,self.cache,resume=True)
            self.assertEqual(reused,assets)
        self.assertEqual(assets['contigs'],1)

    def test_reference_replacement_even_with_equal_bytes_invalidates_cache(self):
        assets = self.build()
        replacement=self.reference.with_suffix('.new')
        replacement.write_bytes(self.reference.read_bytes())
        os.replace(replacement,self.reference)
        with self.assertRaisesRegex(ValueError,'identity changed'):
            self.verify(assets)

    def test_mutation_of_any_bound_small_input_or_tool_invalidates_cache(self):
        assets = self.build()
        for path in (Path(str(self.reference)+'.fai'),self.reference.with_suffix('.dict'),self.tool):
            with self.subTest(path=path):
                old=path.read_bytes();path.write_bytes(old+b'\n')
                with self.assertRaises(ValueError):self.verify(assets)
                path.write_bytes(old)

    def test_receipt_or_asset_tampering_rejected(self):
        assets=self.build()
        altered=json.loads(json.dumps(assets));altered['contigs']=2
        with self.assertRaisesRegex(ValueError,'disagree'):self.verify(altered)
        Path(assets['receipt']['path']).write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'receipt changed'):self.verify(assets)

    def test_axis_mismatch_does_not_commit_assets(self):
        Path(str(self.reference)+'.fai').write_text('contig_b\t4\t10\t4\t5\n')
        with self.assertRaisesRegex(ValueError,'FAI offsets'):self.build()
        self.assertFalse((self.cache/'assets.private.json').exists())

    def test_duplicate_dictionary_contigs_are_rejected(self):
        path=self.reference.with_suffix('.dict')
        path.write_text(path.read_text()*2)
        with self.assertRaisesRegex(ValueError,'unique'):dictionary_sequences(path)

    def test_wrong_fai_offset_with_correct_names_and_lengths_is_rejected(self):
        Path(str(self.reference)+'.fai').write_text('contig_a\t4\t11\t4\t5\n')
        with self.assertRaisesRegex(ValueError,'geometry'):self.build()

    def test_crlf_short_final_line_and_two_contigs_geometry(self):
        self.reference.write_bytes(b'>a\r\nACGT\r\nAC\r\n>b\r\nT\r\n')
        self.reference.with_suffix('.dict').write_text('@SQ\tSN:a\tLN:6\tM5:'+hashlib.md5(b'ACGTAC').hexdigest()+'\n@SQ\tSN:b\tLN:1\tM5:'+hashlib.md5(b'T').hexdigest()+'\n')
        Path(str(self.reference)+'.fai').write_text('a\t6\t4\t4\t6\nb\t1\t18\t1\t3\n')
        self.assertEqual(self.build()['contigs'],2)

    def test_cannot_create_cache_inside_the_conda_environment(self):
        with self.assertRaisesRegex(ValueError,'outside'):
            prepare_preprocessing_reference(self.reference,self.prefix,self.prefix/'cache')
        self.assertFalse((self.prefix/'cache').exists())

    def test_output_containing_reference_cannot_overwrite_any_input(self):
        original=self.reference.read_bytes()
        with self.assertRaisesRegex(ValueError,'must not contain'):
            prepare_preprocessing_reference(self.reference,self.prefix,self.root)
        self.assertEqual(self.reference.read_bytes(),original)
        self.assertFalse((self.root/'plan.private.json').exists())

    def test_empty_existing_output_requires_explicit_resume_and_plan(self):
        self.cache.mkdir(mode=0o700)
        with self.assertRaisesRegex(ValueError,'No bound plan'):
            prepare_preprocessing_reference(self.reference,self.prefix,self.cache,resume=True)


if __name__=='__main__':
    unittest.main()
