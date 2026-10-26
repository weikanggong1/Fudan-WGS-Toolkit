"""QC CLI lifecycle contracts; no scientific fixture or external tools."""
import copy
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
import tempfile
import unittest

from fudan_wgs_toolkit._preprocessing_io import atomic, digest, identity
from fudan_wgs_toolkit._preprocessing_runtime import QC_MAIN, qc_argv


class QcRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        artifacts = {name: root / name for name in ("runtimeJar", "gatkJar", "java")}
        for name, path in artifacts.items():
            path.write_text("interface-only-" + name)
        self.assets = dict(schema="fudan-QC-runtime-assets-v1", mainClass=QC_MAIN,
                           condaPrefix=str(root),status="compiled_not_scientific_validation",
                           **{key: str(value) for key, value in artifacts.items()},
                           identities={key: identity(value) for key, value in artifacts.items()},
                           sha256={key: digest(value) for key, value in artifacts.items()},
                           source_sha256="interface-source", builder_sha256="interface-builder")
        receipt = root / "build_receipt.private.json"
        atomic(receipt, dict(status="compiled_not_scientific_validation", artifacts=copy.deepcopy(self.assets)))
        self.assets.update(build_receipt=str(receipt), build_receipt_sha256=digest(receipt))

    def test_host_resource_guards_do_not_import_association_dependencies(self):
        script = textwrap.dedent("""
            import importlib.abc, os, sys
            forbidden = {"torch", "triton", "numpy", "scipy", "pandas"}
            class NoAssociationDependencies(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split(".")[0] in forbidden:
                        raise AssertionError("host-only preprocessing imported " + fullname)
            sys.meta_path.insert(0, NoAssociationDependencies())
            from fudan_wgs_toolkit.single_preprocessing import _limits, _pss
            assert _limits()["available_bytes"] >= 0
            assert _pss({os.getpid()}) > 0
            assert not forbidden.intersection(sys.modules)
        """)
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                   PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", "-c", script], env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_original_defaults_and_explicit_options_are_preserved(self):
        argv = qc_argv(self.assets, "CollectMultipleMetrics", "-I", "input.bam",
                       java_options=["-Xmx3g", "-XX:ParallelGCThreads=1"])
        self.assertEqual(argv[0], self.assets["java"])
        self.assertIn("-Dsamjdk.use_async_io_write_samtools=true", argv)
        self.assertLess(argv.index("-Xmx4g"), argv.index("-Xmx3g"))
        self.assertEqual(argv[argv.index("-cp") + 1], self.assets["runtimeJar"] + ":" + self.assets["gatkJar"])
        self.assertEqual(argv[-4:], [QC_MAIN, "CollectMultipleMetrics", "-I", "input.bam"])

    def test_non_QC_commands_and_launcher_changes_are_rejected(self):
        with self.assertRaises(ValueError):
            qc_argv(self.assets, "HaplotypeCaller")
        for options in (["-jar"], "-Xmx4g", (x for x in ["-Xmx4g"])):
            with self.subTest(options_type=type(options).__name__), self.assertRaises(ValueError):
                qc_argv(self.assets, "CollectInsertSizeMetrics", java_options=options)

    def test_uncommitted_or_changed_build_receipt_is_rejected(self):
        path = Path(self.assets["build_receipt"])
        receipt = json.loads(path.read_text()); receipt["status"] = "building"
        atomic(path, receipt)
        with self.assertRaisesRegex(RuntimeError, "receipt SHA"):
            qc_argv(self.assets, "CollectMultipleMetrics")
        self.assets["build_receipt_sha256"] = digest(path)
        with self.assertRaisesRegex(RuntimeError, "did not commit"):
            qc_argv(self.assets, "CollectMultipleMetrics")

    def test_changed_artifact_or_inconsistent_assets_are_rejected(self):
        changed = copy.deepcopy(self.assets); changed["source_sha256"] = "different"
        with self.assertRaisesRegex(RuntimeError, "committed build"):
            qc_argv(changed, "CollectMultipleMetrics")
        Path(self.assets["runtimeJar"]).write_text("changed interface artifact")
        with self.assertRaisesRegex(RuntimeError, "artifact identity"):
            qc_argv(self.assets, "CollectMultipleMetrics")

    def test_changed_conda_label_cannot_rebind_an_old_committed_build(self):
        changed=copy.deepcopy(self.assets)
        changed['condaPrefix']='/another/environment'
        with self.assertRaisesRegex(RuntimeError,'committed build'):
            qc_argv(changed,'CollectMultipleMetrics')


if __name__ == "__main__":
    unittest.main()
