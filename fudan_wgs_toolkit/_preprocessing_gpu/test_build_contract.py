"""Interface admission tests; no CUDA, DNA fixture, or scientific benchmark."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

FILE = Path(__file__).resolve().parents[1] / 'build_preprocessing_gpu.py'
spec = importlib.util.spec_from_file_location('public_gpu_builder', FILE)
builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)

class BuildContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / 'bound.file'; path.write_bytes(b'interface-only')
        self.assets = dict(java='/absolute/conda/bin/java', originalClasspath='/absolute/conda/gatk.jar',
            classpath='/absolute/build/adapter.jar:/absolute/conda/gatk.jar', nativeLib='/absolute/build/libjni.so',
            gatkJar=str(path), sha256={'gatkJar':hashlib.sha256(path.read_bytes()).hexdigest()},
            identities={'gatkJar':builder._identity(path)})
        self.options = dict(reference='/absolute/ref.fa', bam='/absolute/input.bam', output='/absolute/new.g.vcf.gz')
    def tearDown(self):
        self.temp.cleanup()
    def test_default_CPU_uses_only_original_classpath(self):
        argv = builder.haplotypecaller_argv(self.assets, **self.options)
        self.assertEqual(argv[argv.index('-cp')+1], self.assets['originalClasspath'])
        self.assertFalse(any('fnit.pairhmm' in x for x in argv))
        self.assertEqual(argv[argv.index('--native-pair-hmm-threads')+1], '16')
    def test_auto_requires_explicit_positive_thresholds(self):
        with self.assertRaises(ValueError):
            builder.haplotypecaller_argv(self.assets, **self.options, mode='auto', audit='/absolute/new.jsonl')
    def test_forceGPU_has_explicit_library_and_audit(self):
        argv = builder.haplotypecaller_argv(self.assets, **self.options, mode='forceGPU', audit='/absolute/new.jsonl')
        self.assertIn('-Dfnit.pairhmm.library='+self.assets['nativeLib'], argv)
        self.assertEqual(argv[argv.index('-cp')+1], self.assets['classpath'])
    def test_changed_identity_is_rejected(self):
        Path(self.assets['gatkJar']).write_bytes(b'changed interface fixture')
        with self.assertRaises(RuntimeError):builder.haplotypecaller_argv(self.assets, **self.options)
    def test_every_bundled_compile_input_matches_manifest(self):
        bundle=Path(__file__).resolve().parent; manifest=json.loads((bundle/'source_manifest.json').read_text())
        self.assertEqual(manifest['provider'], builder.PROVIDER)
        for name, expected in manifest['files'].items():
            self.assertEqual(hashlib.sha256((bundle/name).read_bytes()).hexdigest(), expected, name)

    def _assert_resource_rejected_before_output(self, limits, error_type, message, *, free_bytes=16*2**30):
        prefix=Path(self.temp.name)/'candidate_prefix'
        (prefix/'conda-meta').mkdir(parents=True)
        output=Path(self.temp.name)/'new_build'
        cpu=min(os.sched_getaffinity(0))
        with patch('fudan_wgs_toolkit.single_preprocessing._limits',return_value=limits), \
                patch.object(builder.shutil,'disk_usage',return_value=SimpleNamespace(free=free_bytes)), \
                patch.object(builder.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(error_type,message):
                builder.build_preprocessing_gpu(prefix,output,cpu_affinity=[cpu])
            self.assertFalse(output.exists(),'Resource rejection must precede output creation')
            launch.assert_not_called()

    def test_fractional_ancestor_quota_rejects_one_build_CPU(self):
        self._assert_resource_rejected_before_output(
            dict(cpu_quota_count=1,finite_cpu_quotas=[.5],
                 available_bytes=16*2**30,admission_available_bytes=16*2**30),
            ValueError,'actual ancestor CPU quota')

    def test_insufficient_live_headroom_rejects_build(self):
        self._assert_resource_rejected_before_output(
            dict(cpu_quota_count=1,finite_cpu_quotas=[1.0],
                 available_bytes=8*2**30-1,admission_available_bytes=16*2**30),
            MemoryError,'4 GiB prospective PSS plus 4 GiB live headroom')

    def test_insufficient_admission_headroom_rejects_build(self):
        self._assert_resource_rejected_before_output(
            dict(cpu_quota_count=1,finite_cpu_quotas=[1.0],
                 available_bytes=16*2**30,admission_available_bytes=8*2**30-1),
            MemoryError,'4 GiB prospective PSS plus 4 GiB live headroom')

    def test_insufficient_build_storage_rejects_before_output(self):
        self._assert_resource_rejected_before_output(
            dict(cpu_quota_count=1,finite_cpu_quotas=[1.0],
                 available_bytes=16*2**30,admission_available_bytes=16*2**30),
            OSError,'4 GiB free storage',free_bytes=4*2**30-1)

if __name__=='__main__':unittest.main()
