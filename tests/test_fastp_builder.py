"""Original fastp source/build/runtime contracts, without FASTQ or network jobs."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

from fudan_wgs_toolkit import build_preprocessing_fastp as module
from fudan_wgs_toolkit._preprocessing_io import atomic, bound, digest


def elf_bytes(interpreter):
    value = os.fsencode(interpreter) + b'\0'
    header = struct.pack('<16sHHIQQQIHHHHHH', b'\x7fELF\x02\x01\x01' + bytes(9),
                         3, 62, 1, 0, 64, 0, 0, 64, 56, 1, 0, 0, 0)
    return header + struct.pack('<IIQQQQQQ', 3, 4, 120, 0, 0, len(value), len(value), 1) + value


class FastpPackageTests(unittest.TestCase):
    def test_fixed_upstream_and_exact_three_file_patch(self):
        manifest = module._manifest()
        self.assertEqual(manifest['archive_bytes'], 139422)
        self.assertEqual(manifest['archive_sha256'], module.SOURCE_SHA256)
        self.assertEqual(len(manifest['original_files']), 73)
        self.assertEqual(digest(module._PACKAGE / 'fastp-0.20.0-append-direct.patch'), module.PATCH_SHA256)
        self.assertIn('MIT License', (module._PACKAGE / 'FASTP_LICENSE').read_text())
        changed = {name for name in manifest['original_files']
                   if manifest['original_files'][name] != manifest['append_files'][name]}
        self.assertEqual(changed, {'src/read.cpp', 'src/read.h', 'src/peprocessor.cpp'})

    def test_packaged_patch_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for path in module._PACKAGE.iterdir():
                if path.is_file():
                    (target / path.name).write_bytes(path.read_bytes())
            (target / 'fastp-0.20.0-append-direct.patch').write_text('unbound patch')
            with patch.object(module, '_PACKAGE', target), self.assertRaisesRegex(RuntimeError, 'resource changed'):
                module._manifest()

    def test_archive_validated_before_any_extracted_file(self):
        for bad_name, kind in (('../escape', 'file'), ('fastp-0.20.0/read', 'link'),
                               ('wrong/read', 'file'), ('fastp-0.20.0/extra', 'file')):
            with self.subTest(member=bad_name, kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); archive = root / 'source.tar.gz'
                with tarfile.open(archive, 'w:gz') as stream:
                    item = tarfile.TarInfo(bad_name)
                    if kind == 'link':
                        item.type = tarfile.SYMTYPE; item.linkname = '/etc/passwd'
                        stream.addfile(item)
                    else:
                        item.size = 1; stream.addfile(item, io.BytesIO(b'x'))
                manifest = dict(archive_root='fastp-0.20.0', original_files={'read': hashlib.sha256(b'x').hexdigest()})
                with self.assertRaises(ValueError):
                    module._extract(archive, root / 'extracted', manifest)
                self.assertFalse((root / 'extracted').exists())
                self.assertFalse((root / 'escape').exists())

    def test_archive_duplicate_and_missing_files_are_rejected(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); archive = root / 'source.tar.gz'
                with tarfile.open(archive, 'w:gz') as stream:
                    for _ in range(2 if duplicate else 1):
                        item = tarfile.TarInfo('fastp-0.20.0/read')
                        item.size = 1; stream.addfile(item, io.BytesIO(b'x'))
                manifest = dict(archive_root='fastp-0.20.0', original_files={'read': 'a', 'missing': 'b'})
                with self.assertRaises(ValueError):
                    module._extract(archive, root / 'extracted', manifest)
                self.assertFalse((root / 'extracted').exists())

    def test_exact_patch_applies_all_hunks_before_any_write(self):
        names = ('src/read.cpp', 'src/read.h', 'src/peprocessor.cpp')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; (source / 'src').mkdir(parents=True)
            original = b'first\nold\nlast\n'; expected = b'first\nnew\nlast\n'
            patch_bytes = b''.join(('--- a/' + name + '\n+++ b/' + name +
                 '\n@@ -1,3 +1,3 @@\n first\n-old\n+new\n last\n').encode() for name in names)
            for name in names:
                (source / name).write_bytes(original)
            (root / 'fastp-0.20.0-append-direct.patch').write_bytes(patch_bytes)
            manifest = dict(original_files={name: hashlib.sha256(original).hexdigest() for name in names},
                            append_files={name: hashlib.sha256(expected).hexdigest() for name in names})
            with patch.object(module, '_PACKAGE', root):
                module._apply_patch(source, manifest)
                self.assertTrue(all((source / name).read_bytes() == expected for name in names))
                for name in names:
                    (source / name).write_bytes(original)
                # A late hunk error must not leave the earlier two files patched.
                (source / names[-1]).write_bytes(b'late context mismatch\n')
                with self.assertRaisesRegex(RuntimeError, 'context differs'):
                    module._apply_patch(source, manifest)
                self.assertTrue(all((source / name).read_bytes() == original for name in names[:-1]))

    def test_missing_conda_tools_and_invalid_resources_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); prefix = root / 'env'; (prefix / 'conda-meta').mkdir(parents=True)
            limits = dict(cpu_quota_count=1, finite_cpu_quotas=[1])
            from fudan_wgs_toolkit import single_preprocessing
            cpu = next(iter(os.sched_getaffinity(0)))
            with patch.object(single_preprocessing, '_limits', return_value=limits):
                with self.assertRaisesRegex(ValueError, 'compiler and make'):
                    module.build_preprocessing_fastp(prefix, root / 'absent', cpu_affinity=[cpu])
                self.assertFalse((root / 'absent').exists())
                with self.assertRaisesRegex(ValueError, 'finite'):
                    module.build_preprocessing_fastp(prefix, root / 'nan', cpu_affinity=[cpu], host_memory_limit_gib=float('nan'))
                self.assertFalse((root / 'nan').exists())
                with self.assertRaisesRegex(ValueError, 'distinct allowed'):
                    module.build_preprocessing_fastp(prefix, root / 'duplicate', cpu_affinity=[cpu, cpu])
                with self.assertRaisesRegex(ValueError, 'outside'):
                    module.build_preprocessing_fastp(prefix, prefix / 'must_not_install', cpu_affinity=[cpu])

    def test_bad_archive_preserves_failed_receipt_and_never_publishes_assets(self):
        from fudan_wgs_toolkit import single_preprocessing, build_preprocessing_aligner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); prefix = root / 'env'
            for name in ('conda-meta', 'bin', 'lib', 'include'):
                (prefix / name).mkdir(parents=True, exist_ok=True)
            for name in ('x86_64-conda-linux-gnu-g++', 'make'):
                tool = prefix / 'bin' / name; tool.write_text('not executed'); tool.chmod(0o700)
            for name in ('zlib.h', 'zconf.h'):
                (prefix / 'include' / name).write_text('header fixture')
            (prefix / 'lib/libz.so.1').write_text('library fixture')
            archive = root / 'wrong.tar.gz'; archive.write_bytes(b'not the official archive')
            output = root / 'build'; cpu = next(iter(os.sched_getaffinity(0)))
            limits = dict(cpu_quota_count=1, finite_cpu_quotas=[1])
            with patch.object(single_preprocessing, '_limits', return_value=limits), \
                    patch.object(build_preprocessing_aligner, '_admit', return_value={'admitted': True}):
                with self.assertRaisesRegex(RuntimeError, 'archive size or SHA'):
                    module.build_preprocessing_fastp(prefix, output, cpu_affinity=[cpu], source_archive=archive)
            failed = json.loads((output / 'build_receipt.private.json').read_text())
            self.assertEqual(failed['status'], 'failed')
            self.assertEqual(failed['steps'], [])
            self.assertFalse((output / 'assets.private.json').exists())
            self.assertFalse((output / 'source').exists())
            self.assertEqual((prefix / 'lib/libz.so.1').read_text(), 'library fixture')
            compiler = prefix / 'bin/x86_64-conda-linux-gnu-g++'
            compiler.unlink(); compiler.symlink_to('/bin/true')
            with patch.object(single_preprocessing, '_limits', return_value=limits), \
                    patch.object(build_preprocessing_aligner, '_admit', return_value={'admitted': True}):
                with self.assertRaisesRegex(ValueError, 'resolve inside'):
                    module.build_preprocessing_fastp(prefix, root / 'escaped_tool_build',
                                                    cpu_affinity=[cpu], source_archive=archive)
            self.assertFalse((root / 'escaped_tool_build/assets.private.json').exists())

    def test_build_path_shell_metacharacters_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); prefix = root / 'env'; (prefix / 'conda-meta').mkdir(parents=True)
            for name in ('bad`command`', 'bad;command', 'bad:library', 'bad space', 'bad$(command)'):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'ASCII'):
                    module.build_preprocessing_fastp(prefix, root / name)
                self.assertFalse((root / name).exists())


class FastpAssetTests(unittest.TestCase):
    """Fake bytes test ownership/bindings only; no claims about fastp accuracy."""
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name); self.prefix = self.root / 'env'
        (self.prefix / 'lib').mkdir(parents=True)
        self.loader = self.root / 'loader'; self.loader.write_bytes(b'interface-only-loader')
        self.executable = self.root / 'fastp'; self.executable.write_bytes(elf_bytes(self.loader))
        self.library = self.prefix / 'lib/libz.so.1'; self.library.write_bytes(b'interface-only-zlib')
        self.archive = self.root / 'source.tar.gz'; self.archive.write_bytes(b'interface-only-archive')
        self.source = self.root / 'source'; self.source.mkdir(); (self.source / 'read.cpp').write_bytes(b'source fixture')
        self.manifest = dict(original_files={'read.cpp': digest(self.source / 'read.cpp')},
                             append_files={'read.cpp': digest(self.source / 'read.cpp')}, license_sha256='license-fixture')
        self.dynamic = {'libz.so.1': bound(self.library)}
        self.assets = dict(schema=module._ASSET_SCHEMA, status=module._STATUS, variant='append', condaPrefix=str(self.prefix),
            executable=bound(self.executable), loader=bound(self.loader), dynamic_libraries=copy.deepcopy(self.dynamic),
            source_tree=str(self.source), source_file_sha256=self.manifest['append_files'], source_archive=bound(self.archive),
            source_tag='v0.20.0', source_sha256=digest(self.archive), applied_patch_sha256=module.PATCH_SHA256,
            license='MIT', license_sha256='license-fixture', builder_sha256=digest(module.__file__),
            source_manifest_sha256=digest(module._PACKAGE / 'source_manifest.json'), default_fastp_workers=1)
        self.receipt = self.root / 'build_receipt.private.json'
        self.commit()
        for name, value in (('_manifest', lambda: self.manifest), ('SOURCE_BYTES', self.archive.stat().st_size),
                            ('SOURCE_SHA256', digest(self.archive))):
            patcher = patch.object(module, name, value); patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch.object(module, '_dynamic_dependencies', return_value=self.dynamic)
        self.dependencies = patcher.start(); self.addCleanup(patcher.stop)
        self.env = {'PATH': '/usr/bin:/bin', 'LD_LIBRARY_PATH': str(self.prefix / 'lib')}

    def commit(self):
        core = {key: value for key, value in self.assets.items() if key not in ('buildReceipt', 'build_receipt_sha256')}
        atomic(self.receipt, dict(schema=module._BUILD_SCHEMA, status=module._STATUS, artifacts=core,
            Conda_package_records_unchanged=True, source_files_unchanged=True, toolchain_unchanged=True, zlib_unchanged=True))
        self.assets.update(buildReceipt=str(self.receipt), build_receipt_sha256=digest(self.receipt))

    def argv(self, **kwargs):
        return module.fastp_argv(self.assets, read1='r1.fq.gz', read2='r2.fq.gz', output1='c1.fq.gz',
                 output2='c2.fq.gz', json_report='fastp.json', html_report='fastp.html', runtime_env=self.env, **kwargs)

    def test_default_normal_paired_argv_does_not_change_filter_or_compression(self):
        argv = self.argv()
        self.assertEqual(argv, [str(self.executable), '-i', 'r1.fq.gz', '-I', 'r2.fq.gz', '-o', 'c1.fq.gz',
                              '-O', 'c2.fq.gz', '-j', 'fastp.json', '-h', 'fastp.html', '-w', '1'])
        self.dependencies.assert_called_once_with(str(self.executable), str(self.loader), self.env)
        self.assertEqual(self.argv(workers=4)[-2:], ['-w', '4'])

    def test_worker_boundaries_and_path_aliases_are_rejected(self):
        for value in (True, 0, 17, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.argv(workers=value)
        with self.assertRaisesRegex(ValueError, 'distinct'):
            module.fastp_argv(self.assets, read1='a', read2='b', output1='./a', output2='c',
                json_report='d', html_report='e', runtime_env=self.env)

    def test_loader_injection_is_refused_before_any_trace(self):
        for key in ('LD_PRELOAD', 'LD_AUDIT', 'LD_TRACE_LOADED_OBJECTS', 'GLIBC_TUNABLES', 'BASH_ENV'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'overrides'):
                module.validate_fastp_assets(self.assets, runtime_env=dict(self.env, **{key: 'unbound'}))
        self.dependencies.assert_not_called()

    def test_conda_gold_linker_variable_is_not_a_runtime_loader_override(self):
        env=dict(self.env, LD_GOLD='/existing/conda/bin/x86_64-conda-linux-gnu-ld.gold')
        self.assertEqual(module.validate_fastp_assets(self.assets, runtime_env=env), self.assets)
        self.dependencies.assert_called_once_with(str(self.executable), str(self.loader), env)

    def test_cwd_dependent_library_search_is_refused(self):
        for value in ('relative/lib', '/safe/lib:', ':/safe/lib', '/safe/lib::/other/lib', '$ORIGIN/lib'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'absolute and nonempty'):
                module.validate_fastp_assets(self.assets, runtime_env=dict(self.env, LD_LIBRARY_PATH=value))
        self.dependencies.assert_not_called()

    def test_different_runtime_libz_is_refused_even_when_path_label_unchanged(self):
        alternative = self.root / 'other-libz'; alternative.write_bytes(b'other zlib')
        self.dependencies.return_value = {'libz.so.1': bound(alternative)}
        with self.assertRaisesRegex(RuntimeError, 'resolves different'):
            self.argv()

    def test_changed_runtime_library_is_refused_before_trace(self):
        self.library.write_bytes(b'changed zlib')
        with self.assertRaisesRegex(RuntimeError, 'dynamic library identity'):
            self.argv()
        self.dependencies.assert_not_called()

    def test_changed_executable_or_loader_is_refused_before_trace(self):
        self.executable.write_bytes(b'changed binary')
        with self.assertRaisesRegex(RuntimeError, 'artifact identity'):
            self.argv()
        self.dependencies.assert_not_called()

    def test_replaced_same_bytes_executable_is_refused(self):
        content = self.executable.read_bytes(); self.executable.unlink(); self.executable.write_bytes(content)
        with self.assertRaisesRegex(RuntimeError, 'artifact identity'):
            self.argv()

    def test_changed_source_cannot_reuse_compiled_receipt(self):
        (self.source / 'read.cpp').write_bytes(b'changed source')
        with self.assertRaisesRegex(RuntimeError, 'source changed'):
            self.argv()

    def test_uncommitted_and_rewritten_receipts_are_rejected(self):
        row = json.loads(self.receipt.read_text()); row['status'] = 'building'; atomic(self.receipt, row)
        with self.assertRaisesRegex(RuntimeError, 'receipt is missing or its SHA'):
            self.argv()
        self.assets['build_receipt_sha256'] = digest(self.receipt)
        with self.assertRaisesRegex(RuntimeError, 'does not commit'):
            self.argv()

    def test_variant_or_conda_label_cannot_rebind_existing_build(self):
        for key, value in (('variant', 'original'), ('condaPrefix', '/another/prefix'),
                           ('applied_patch_sha256', 'other-patch')):
            original = self.assets[key]; self.assets[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'exact assets'):
                self.argv()
            self.assets[key] = original

    def test_source_or_builder_revision_requires_new_build(self):
        self.assets['builder_sha256'] = 'older-builder'; self.commit()
        with self.assertRaisesRegex(RuntimeError, 'packaged source/patch differs'):
            self.argv()

    def test_validation_precedes_successful_resume_skip(self):
        from fudan_wgs_toolkit.single_preprocessing import _Runner
        root = self.root / 'pipeline'; (root / 'stages/fastp').mkdir(parents=True)
        cpu = next(iter(os.sched_getaffinity(0)))
        config = dict(cpu_affinity=[cpu], host_memory_limit_gib=2, host_memory_reserve_gib=4, fastp_assets=self.assets)
        atomic(root / 'plan.private.json', config)
        runner = _Runner(root, self.prefix, config, True)
        output = root / 'cleaned'; output.write_bytes(b'committed output')
        command = self.argv()
        atomic(root / 'stages/fastp/complete.private.json', dict(commands=[command], outputs=[bound(output)], status='complete'))
        # A good resume performs no fastp execution, but still verifies the tool.
        self.assertEqual(runner.stage('fastp', [self.argv()], [output])['status'], 'complete')
        self.library.write_bytes(b'library changed after the completed stage')
        with self.assertRaisesRegex(RuntimeError, 'dynamic library identity'):
            runner.stage('fastp', [self.argv()], [output])
        self.assertEqual(output.read_bytes(), b'committed output')

    def test_resume_plan_with_changed_assets_is_rejected(self):
        from fudan_wgs_toolkit.single_preprocessing import _Runner
        root = self.root / 'pipeline'; root.mkdir()
        cpu = next(iter(os.sched_getaffinity(0)))
        config = dict(cpu_affinity=[cpu], host_memory_limit_gib=2, host_memory_reserve_gib=4, fastp_assets=self.assets)
        atomic(root / 'plan.private.json', config)
        changed = copy.deepcopy(config); changed['fastp_assets']['applied_patch_sha256'] = 'new candidate'
        with self.assertRaisesRegex(ValueError, 'identical source, tools'):
            _Runner(root, self.prefix, changed, True)


class FastpElfTests(unittest.TestCase):
    def test_only_bound_elf_interpreter_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); loader = root / 'loader'; loader.write_bytes(b'loader')
            executable = root / 'fastp'; executable.write_bytes(elf_bytes(loader))
            self.assertEqual(module._elf_interpreter(executable), str(loader))
            executable.write_bytes(b'#! /bin/sh\n')
            with self.assertRaisesRegex(RuntimeError, 'ELF executable'):
                module._elf_interpreter(executable)

    def test_truncated_program_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fastp'; path.write_bytes(elf_bytes('/bin/sh')[:80])
            with self.assertRaisesRegex(RuntimeError, 'Truncated'):
                module._elf_interpreter(path)

    def test_dependency_trace_binds_resolved_real_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); library = root / 'libz.so.1'; library.write_bytes(b'library')
            loader = root / 'loader'; loader.write_bytes(b'loader')
            result = types.SimpleNamespace(returncode=0, stdout=(f'linux-vdso.so.1 (0x000)\n'
                f' libz.so.1 => {library} (0x123)\n {loader} (0x456)\n').encode(), stderr=b'')
            with patch.object(module.subprocess, 'run', return_value=result) as run:
                resolved = module._dynamic_dependencies('fastp', loader, {'LD_LIBRARY_PATH': str(root)})
            self.assertEqual(resolved, {'libz.so.1': bound(library)})
            self.assertEqual(run.call_args.args[0], [str(loader), '--list', 'fastp'])

    def test_unresolved_or_missing_libz_refuses_runtime(self):
        for stdout in (b'libz.so.1 => not found\n', b'linux-vdso.so.1 (0x1)\n', b'unknown loader line\n'):
            with self.subTest(stdout=stdout):
                result = types.SimpleNamespace(returncode=0, stdout=stdout, stderr=b'')
                with patch.object(module.subprocess, 'run', return_value=result), self.assertRaises(RuntimeError):
                    module._dynamic_dependencies('fastp', '/loader', {})


if __name__ == '__main__':
    unittest.main()
