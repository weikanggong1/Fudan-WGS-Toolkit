"""Commit/resource contracts only; no simulated scientific benchmark."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from fudan_wgs_toolkit._preprocessing_io import bound,digest,identity
from fudan_wgs_toolkit import build_preprocessing_bqsr as module


class BqsrCommitContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name);self.root=root
        artifacts={}
        for key in ['adapterJar','gatkJar','java']:
            path=root/key;path.write_text('interface-'+key);artifacts[key]=str(path)
        self.core=dict(schema='fudan-bqsr-parallel-assets-v1',status='compiled_help_passed_not_science',
            **artifacts,classpath=artifacts['adapterJar']+':'+artifacts['gatkJar'],
            entrypoint='org.fnit.wgs.bqsr.parallel.ParallelFusedBqsrTrainer',feature_cache_lookahead_bases=1000,
            sha256={k:digest(v) for k,v in artifacts.items()},identities={k:identity(v) for k,v in artifacts.items()},
            builder_sha256=digest(module.__file__),
            source_manifest_sha256=digest(Path(module.__file__).parent/'_preprocessing_bqsr/source_manifest.json'))
        self.receipt=root/'build_receipt.private.json'
        self.receipt.write_text(json.dumps(dict(schema='fudan-bqsr-parallel-build-v1',status=self.core['status'],
            artifacts=self.core,Conda_package_records_unchanged=True)))
        self.assets=dict(self.core,buildReceipt=str(self.receipt),build_receipt_sha256=digest(self.receipt))
        self.options=dict(input_cram=root/'original.cram',reference_fasta=root/'reference.fa',
            known_sites=['known_one.vcf.gz','known_two.vcf.gz','known_three.vcf.gz'],
            recalibration_table=root/'new.table',canonical_bam=root/'new.bam',
            audit_json=root/'new.json',temporary_directory=root/'tmp')

    def test_committed_plan_preserves_known_site_order_without_creating_outputs(self):
        argv=module.bqsr_training_argv(self.assets,**self.options)
        self.assertIn('org.fnit.wgs.bqsr.parallel.ParallelFusedBqsrTrainer',argv)
        self.assertEqual(argv.count('--training-workers'),1)
        self.assertEqual(argv[argv.index('--training-workers')+1],'2')
        self.assertEqual(argv.count('--queue-capacity'),1)
        self.assertEqual(argv[argv.index('--queue-capacity')+1],'32')
        self.assertEqual([argv[n+1] for n,v in enumerate(argv) if v=='--known-sites'],self.options['known_sites'])
        self.assertNotIn('-L',argv);self.assertNotIn('--read-filter',argv)
        self.assertNotIn('--feature-lookahead',argv)
        self.assertFalse(Path(self.options['canonical_bam']).exists())
        for cpus,workers,capacity in [(1,1,1),(16,16,16),(4,4,64)]:
            with self.subTest(valid=(cpus,workers,capacity)):
                command=module.bqsr_training_argv(self.assets,**self.options,cpu_threads=cpus,
                    training_workers=workers,queue_capacity=capacity)
                self.assertEqual(command[command.index('--training-workers')+1],str(workers))
                self.assertEqual(command[command.index('--queue-capacity')+1],str(capacity))
        for cpus,workers,capacity in [(17,17,1),(1,2,32),(16,16,17),(4,4,65),
                                     (3,0,32),(3,2,0),(3,True,32),(3,2.0,32),(3,2,False),(3,2,32.0)]:
            with self.subTest(invalid=(cpus,workers,capacity)):
                # A missing receipt proves resource bounds reject the request
                # before filesystem/hash work or a subprocess could occur.
                invalid=dict(self.assets,buildReceipt=str(self.root/'not_present'))
                with self.assertRaisesRegex(ValueError,'total queued reads'):
                    module.bqsr_training_argv(invalid,**self.options,cpu_threads=cpus,
                        training_workers=workers,queue_capacity=capacity)

    def test_assets_status_alone_cannot_replace_missing_build_receipt(self):
        self.receipt.unlink()
        with self.assertRaisesRegex(RuntimeError,'receipt is missing'):
            module.bqsr_training_argv(self.assets,**self.options)

    def test_old_schema_or_mismatched_receipt_with_correct_hash_is_rejected(self):
        original=json.loads(self.receipt.read_text())
        for kind in ['failed','old_receipt_schema','different_artifacts','old_asset_schema']:
            with self.subTest(kind=kind):
                value=copy.deepcopy(original);assets=copy.deepcopy(self.assets)
                if kind=='failed':value['status']='failed'
                elif kind=='old_receipt_schema':value['schema']='fudan-bqsr-adapter-build-v1'
                elif kind=='different_artifacts':value['artifacts']['classpath']+=':uncommitted.jar'
                else:
                    assets['schema']='fudan-bqsr-replay-assets-v1'
                    value['artifacts']['schema']=assets['schema']
                self.receipt.write_text(json.dumps(value));assets['build_receipt_sha256']=digest(self.receipt)
                with self.assertRaisesRegex(RuntimeError,'does not commit|parallel contract'):
                    module.bqsr_training_argv(assets,**self.options)
        # Mutate a private copy of the package, keeping its manifest and
        # builder byte-identical, so no real worktree source is touched.
        packaged=self.root/'packaged';packaged.mkdir()
        builder=packaged/Path(module.__file__).name
        shutil.copyfile(module.__file__,builder)
        shutil.copytree(Path(module.__file__).parent/'_preprocessing_bqsr',packaged/'_preprocessing_bqsr')
        changed=next((packaged/'_preprocessing_bqsr/assets').rglob('ParallelFusedBqsrTrainer.java'))
        changed.write_bytes(changed.read_bytes()+b'\n// changed after build\n')
        self.receipt.write_text(json.dumps(original))
        assets=dict(self.assets,build_receipt_sha256=digest(self.receipt))
        with patch.object(module,'__file__',str(builder)):
            with self.assertRaisesRegex(RuntimeError,'source differs from manifest'):
                module.bqsr_training_argv(assets,**self.options)

    def test_each_actual_artifact_hash_is_rechecked_independently_of_identity(self):
        for key in ['java','adapterJar','gatkJar']:
            with self.subTest(key=key):
                path=Path(self.assets[key]);original=path.read_bytes()
                try:
                    path.write_bytes(original+b'-modified')
                    assets=copy.deepcopy(self.assets)
                    # Refresh identity and commit a matching receipt but keep
                    # the original expected content hash: only SHA catches it.
                    assets['identities']={name:identity(assets[name]) for name in assets['sha256']}
                    core={k:v for k,v in assets.items() if k not in ('buildReceipt','build_receipt_sha256')}
                    receipt=dict(schema='fudan-bqsr-parallel-build-v1',status=assets['status'],
                        artifacts=core,Conda_package_records_unchanged=True)
                    self.receipt.write_text(json.dumps(receipt));assets['build_receipt_sha256']=digest(self.receipt)
                    with self.assertRaisesRegex(RuntimeError,'identity or SHA-256 changed: '+key):
                        module.bqsr_training_argv(assets,**self.options)
                finally:path.write_bytes(original)

    def test_nan_reserve_rejected_before_build_output_or_child(self):
        prefix=self.root/'prefix';(prefix/'conda-meta').mkdir(parents=True)
        output=self.root/'new_build';cpu=min(os.sched_getaffinity(0))
        limits=dict(cpu_quota_count=1,finite_cpu_quotas=[1],available_bytes=16*2**30,admission_available_bytes=16*2**30)
        with patch('fudan_wgs_toolkit.single_preprocessing._limits',return_value=limits):
            with self.assertRaisesRegex(ValueError,'finite build budget'):
                module.build_preprocessing_bqsr(prefix,output,cpu_affinity=[cpu],host_memory_reserve_gib=float('nan'))
        self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
