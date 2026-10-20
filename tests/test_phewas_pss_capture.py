"""Bounded proc-format interface fixtures; not scientific benchmark evidence."""
import errno
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fudan_wgs_toolkit import phewas_resources as r


def mapping(start, end, rss, pss, path="[anon:worker arena]", permissions="rw-p"):
    return (f"{start:08x}-{end:08x} {permissions} 00000000 00:00 0 {path}\n"
            "Size:                 16 kB\nKernelPageSize:         4 kB\n"
            f"Rss:                  {rss} kB\nPss:                  {pss} kB\n"
            "Anonymous:             0 kB\nSwapPss:               0 kB\n"
            "ProtectionKey:         0\nVmFlags: rd wr mr mw me ac\n")


class PssCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.process = self.root / "123"
        self.process.mkdir()
        (self.process / "status").write_text("VmRSS: 900 kB\nRssAnon: 400 kB\nRssFile: 500 kB\n")
        self.path = self.process / "smaps"
        self.data = (mapping(0x1000, 0x5000, 16, 8, "/reference/example library:1.so", "r-xp")
                     + mapping(0x9000, 0xd000, 12, 6)
                     + mapping(0xffffffffff600000, 0xffffffffff601000, 0, 0,
                               "[vsyscall]", "--xp")).encode()
        self.path.write_bytes(self.data)

    def counters(self, include_pss=True):
        real_path = Path
        with patch.object(r, "Path", lambda value: real_path(str(value).replace("/proc", str(self.root), 1))):
            return r._process_memory_counters(123, include_pss=include_pss)

    def test_complete_capture_is_closed_before_strict_parsing(self):
        real_open, original_counter = Path.open, r._memory_kib
        state = {"eof": False, "closed": False}

        class Capture(io.BytesIO):
            def read(inner, size=-1):
                value = super().read(size)
                if value == b"": state["eof"] = True
                return value

            def close(inner):
                state["closed"] = True
                # The parser must consume its captured bytes, not reopen this changed file.
                with real_open(self.path, "wb") as changed:
                    changed.write(b"changed after completed capture\n")
                super().close()

        def opened(path, *args, **kwargs):
            return Capture(self.data) if path == self.path else real_open(path, *args, **kwargs)

        def parse_counter(value):
            self.assertTrue(state["eof"] and state["closed"])
            return original_counter(value)

        with patch.object(Path, "open", opened), patch.object(r, "_memory_kib", parse_counter):
            totals, count = r._read_smaps_pss(self.path)
        self.assertEqual((count, totals["Rss"], totals["Pss"]), (3, 28 * 1024, 14 * 1024))

    def test_short_binary_reads_continue_until_EOF(self):
        real_open = Path.open

        class ShortReads(io.BytesIO):
            def read(inner, size=-1):
                return super().read(min(size, 7))

        def opened(path, *args, **kwargs):
            return ShortReads(self.data) if path == self.path else real_open(path, *args, **kwargs)

        with patch.object(Path, "open", opened):
            totals, count = r._read_smaps_pss(self.path)
        self.assertEqual(count, 3)
        self.assertEqual(totals["Pss"], 14 * 1024)

    def test_exact_limit_requires_EOF_and_one_more_byte_is_unknown(self):
        with patch.object(r, "_SMAPS_CAPTURE_LIMIT_BYTES", len(self.data)):
            self.assertEqual(self.counters()["pss_bytes"], 14 * 1024)
            with self.path.open("ab") as stream: stream.write(b"\n")
            value = self.counters()
        self.assertIsNone(value["pss_bytes"])
        self.assertFalse(value["pss_coverage_complete"])
        self.assertEqual(value["rss_bytes"], 900 * 1024)
        self.assertEqual(value["pss_read_errors"][-1]["reason"], "smaps capture byte limit exceeded")

    def test_complete_prefix_does_not_hide_partial_or_overlapping_tail(self):
        first = mapping(0x1000, 0x5000, 16, 8)
        for tail in ("00009000-0000d000 rw-p 00000000 00:00 0 [anonymous]\n",
                     mapping(0x2000, 0x6000, 4, 2),
                     mapping(0x9000, 0xd000, 12, 6).replace("Pss:                  6 kB\n", "")):
            with self.subTest(tail_kind=tail.splitlines()[0]):
                self.path.write_text(first + tail)
                value = self.counters()
                self.assertIsNone(value["pss_bytes"])
                self.assertIsNone(value["swap_pss_bytes"])
                self.assertFalse(value["pss_coverage_complete"])
                self.assertEqual(value["rss_bytes"], 900 * 1024)

    def test_empty_and_read_error_are_unknown_with_content_free_diagnostics(self):
        self.path.write_bytes(b"")
        value = self.counters()
        self.assertEqual(value["pss_read_errors"][-1]["reason"], "empty or incomplete smaps memory counters")
        self.assertIsNone(value["pss_bytes"])
        real_open = Path.open

        class Failure(io.BytesIO):
            def read(inner, size=-1):
                if inner.tell(): raise OSError(errno.EIO, "PRIVATE_ADDRESS_AND_CONTENT")
                return super().read(size)

        def opened(path, *args, **kwargs):
            return Failure(self.data) if path == self.path else real_open(path, *args, **kwargs)

        with patch.object(Path, "open", opened): value = self.counters()
        self.assertIsNone(value["pss_bytes"])
        self.assertEqual(value["pss_read_errors"][-1], dict(source="smaps", error_type="OSError",
                         errno=errno.EIO, reason="endpoint_read_failed"))
        self.assertNotIn("PRIVATE", json.dumps(value))
        self.assertNotIn(str(self.root), json.dumps(value))

    def test_rollup_remains_preferred_and_disabled_PSS_has_no_errors(self):
        (self.process / "smaps_rollup").write_text("Rss: 28 kB\nPss: 14 kB\nPss_Anon: 10 kB\n")
        with patch.object(r, "_capture_smaps", side_effect=AssertionError("fallback must not run")):
            value = self.counters()
        self.assertEqual(value["pss_bytes"], 14 * 1024)
        self.assertEqual(value["pss_read_errors"], [])
        self.assertEqual(value["pss_source"], "smaps_rollup")
        value = self.counters(include_pss=False)
        self.assertIsNone(value["pss_bytes"])
        self.assertFalse(value["pss_coverage_complete"])
        self.assertEqual(value["pss_read_errors"], [])

    def test_missing_or_denied_rollup_keeps_exact_error_class_and_errno(self):
        real_open = Path.open

        def opened(path, *args, **kwargs):
            if path.name == "smaps_rollup": raise PermissionError(errno.EACCES, "PRIVATE_PATH")
            return real_open(path, *args, **kwargs)

        value = self.counters()
        self.assertEqual(value["pss_bytes"], 14 * 1024)
        self.assertEqual(value["pss_read_errors"][0]["errno"], errno.ENOENT)
        with patch.object(Path, "open", opened): value = self.counters()
        self.assertEqual(value["pss_bytes"], 14 * 1024)
        self.assertEqual(value["pss_read_errors"][0], dict(source="smaps_rollup", error_type="PermissionError",
                         errno=errno.EACCES, reason="endpoint_permission_denied"))
        self.assertNotIn("PRIVATE", json.dumps(value))


if __name__ == "__main__":
    unittest.main()
