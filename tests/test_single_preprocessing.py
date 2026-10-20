"""Format/guard regressions. Real-data benchmark evidence is kept separately."""
import gzip
from pathlib import Path

import unittest
import tempfile
import os
import sys
from unittest.mock import patch

from fudan_wgs_toolkit import single_subj_preprocessing
from fudan_wgs_toolkit._preprocessing_io import vcf_oracle, verify_and_split_plink


def fixture(tmp_path, calls=('0/0','0/1','1/1','./.')):
    vcf=tmp_path/'calls.vcf.gz'
    with gzip.open(vcf,'wt') as out:
        out.write('##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsubject_a\n')
        for pos,gt in enumerate(calls,1):
            out.write(f'chr1\t{pos}\t.\tA\tC\t50\tPASS\t.\tGT\t{gt}\n')
    return vcf


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.tmp_path=Path(self.directory.name)

    def test_public_name_and_early_sample_guard(self):
        self.assertEqual(single_subj_preprocessing.__name__,'single_subj_preprocessing')
        with self.assertRaisesRegex(ValueError,'sample_id'):
            single_subj_preprocessing('a','b','ref',[],self.tmp_path/'output',sample_id='bad\tID')
        self.assertFalse((self.tmp_path/'output').exists())

    def test_no_silent_hard_call_coercion(self):
        for number,gt in enumerate(('0/.','1','0/2')):
            with self.subTest(gt=gt),self.assertRaisesRegex(ValueError,'Partial, haploid'):
                vcf_oracle(fixture(self.tmp_path,(gt,)),'subject_a',self.tmp_path/('oracle'+str(number)))

    def test_true_sample_axis(self):
        with self.assertRaisesRegex(ValueError,'requested single sample'):
            vcf_oracle(fixture(self.tmp_path),'wrong_subject',self.tmp_path/'oracle')

    def test_bed_complete_readback_and_partition(self):
        tmp=self.tmp_path;oracle=tmp/'oracle'
        scan=vcf_oracle(fixture(tmp),'subject_a',oracle)
        prefix=tmp/'plink'
        Path(str(prefix)+'.bed').write_bytes(b'\x6c\x1b\x01\x03\x02\x00\x01')
        Path(str(prefix)+'.bim').write_text(''.join(f'1\t1:{i}:A:C\t0\t{i}\tC\tA\n' for i in range(1,5)))
        Path(str(prefix)+'.fam').write_text('subject_a subject_a 0 0 0 -9\n')
        result=verify_and_split_plink(prefix,oracle,tmp/'split','subject_a',scan['chromosome_counts'])
        self.assertEqual(result['variants'],4)
        self.assertTrue(result['all_hard_calls_verified'])
        self.assertEqual((tmp/'split/chr1.bed').read_bytes(),Path(str(prefix)+'.bed').read_bytes())
        Path(str(prefix)+'.bed').write_bytes(b'\x6c\x1b\x01\x03\x03\x00\x01')
        with self.assertRaisesRegex(ValueError,'hard call'):
            verify_and_split_plink(prefix,oracle,tmp/'corrupt','subject_a',scan['chromosome_counts'])

    def test_duplicate_variant_rejected(self):
        vcf=fixture(self.tmp_path,('0/1',))
        with gzip.open(vcf,'rt') as src:text=src.read()
        with gzip.open(vcf,'wt') as dst:dst.write(text+text.splitlines()[-1]+'\n')
        with self.assertRaisesRegex(ValueError,'Duplicate variant'):
            vcf_oracle(vcf,'subject_a',self.tmp_path/'oracle')

    def test_fresh_stage_creates_parents_and_commits(self):
        from fudan_wgs_toolkit.single_preprocessing import _Runner
        config={'cpu_affinity':[min(os.sched_getaffinity(0))],
                'host_memory_limit_gib':2,'host_memory_reserve_gib':.01}
        runner=_Runner(self.tmp_path,Path(sys.executable).parent.parent,config,False)
        receipt=runner.stage('first',[[sys.executable,'-c','print("ok")']],[],.01)
        self.assertEqual(receipt['exit_codes'],[0])
        self.assertTrue((self.tmp_path/'stages/first/complete.private.json').is_file())

    def test_inactive_file_estimate_separate_from_actual_headroom(self):
        from fudan_wgs_toolkit.single_preprocessing import _limits
        cgroup=self.tmp_path/'cgroup';cgroup.mkdir()
        (cgroup/'memory.limit_in_bytes').write_text(str(2**30))
        (cgroup/'memory.usage_in_bytes').write_text(str(int(.95*2**30)))
        (cgroup/'memory.stat').write_text('total_inactive_file '+str(int(.9*2**30))+'\ntotal_rss '+str(int(.05*2**30))+'\n')
        (cgroup/'memory.failcnt').write_text('0')
        with patch('fudan_wgs_toolkit.phewas_resources._memory_directories',return_value=[(1,cgroup,cgroup)]):
            measured=_limits()
        self.assertLess(measured['available_bytes'],.06*2**30)
        self.assertGreater(measured['admission_available_bytes'],.9*2**30)
        self.assertTrue(measured['reclaimable_file_cache_is_an_estimate'])

    def test_pss_falls_back_on_old_kernel_and_rejects_live_unknown(self):
        from fudan_wgs_toolkit.single_preprocessing import _pss
        with patch('fudan_wgs_toolkit.phewas_resources._read_rollup_pss',side_effect=FileNotFoundError):
            measured=_pss({os.getpid()})
        self.assertGreater(measured,0)
        with patch('fudan_wgs_toolkit.phewas_resources._process_memory_counters',
                   return_value={'pss_bytes':None,'pss_coverage_complete':False}):
            with self.assertRaisesRegex(RuntimeError,'live task process'):_pss({os.getpid()})
        diagnostics={}
        with patch('fudan_wgs_toolkit.phewas_resources._process_memory_counters',side_effect=[
                   {'pss_bytes':None,'pss_coverage_complete':False},
                   {'pss_bytes':42,'pss_coverage_complete':True}]):
            self.assertEqual(_pss({os.getpid()},diagnostics=diagnostics),42)
        self.assertEqual(diagnostics['transient_endpoint_retries'],1)

    def test_failed_stage_drains_child_after_root_exit(self):
        from fudan_wgs_toolkit.single_preprocessing import _Runner
        config={'cpu_affinity':[min(os.sched_getaffinity(0))],
                'host_memory_limit_gib':2,'host_memory_reserve_gib':.01}
        runner=_Runner(self.tmp_path,Path(sys.executable).parent.parent,config,False)
        child_pid=self.tmp_path/'child.pid'
        code=('import subprocess,sys,pathlib,time;'
              'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);'
              'pathlib.Path(sys.argv[1]).write_text(str(p.pid));time.sleep(.2);sys.exit(1)')
        with self.assertRaisesRegex(RuntimeError,'nonzero'):
            runner.stage('fails',[[sys.executable,'-c',code,str(child_pid)]],[],.01)
        pid=int(child_pid.read_text())
        stat=Path('/proc',str(pid),'stat')
        self.assertTrue(not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0] in ('Z','X'))
        self.assertTrue((self.tmp_path/'stages/fails/failed.private.json').is_file())


if __name__=='__main__':
    unittest.main()
