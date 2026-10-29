"""Old/new proc-parser byte equivalence; no scientific benchmark or proc scan."""
import io
import itertools
import random
import re
import unittest
from unittest.mock import patch

from fudan_wgs_toolkit import phewas_resources as r


# Independent copy of the pre-dispatch grammar/algorithm. Do not call the new
# parser's regexes or numeric helper: these are the compatibility oracle.
_OLD_HEADER = re.compile(r"^([0-9a-fA-F]+)-([0-9a-fA-F]+) [r-][w-][x-][ps] "
    r"[0-9a-fA-F]+ [0-9a-fA-F]+:[0-9a-fA-F]+ [0-9]+(?:\s+.*)?$")
_OLD_COUNTERS = frozenset(("Rss", "Pss", "SwapPss"))
_OLD_REQUIRED = frozenset(("Rss", "Pss"))


def old_memory_kib(tail):
    if re.fullmatch(r"\s*[0-9]+\s+kB\s*", tail) is None:
        raise ValueError("memory counter must be nonnegative integer kB")
    return int(tail.split()[0])*1024


def old_smaps(captured):
    totals = dict.fromkeys(_OLD_COUNTERS, 0)
    swap_complete = True
    current, previous_end, count = None, 0, 0
    with io.StringIO(captured.decode("utf-8", errors="replace")) as stream:
        for line in stream:
            header = _OLD_HEADER.fullmatch(line.rstrip("\n"))
            if header is not None:
                if current is not None:
                    if not _OLD_REQUIRED <= current.keys():
                        raise ValueError("incomplete smaps VMA memory counters")
                    for key in totals:
                        if key in current:
                            totals[key] += current[key]
                        else:
                            swap_complete = False
                start, end = (int(value, 16) for value in header.groups())
                if start < previous_end or start >= end:
                    raise ValueError("invalid or overlapping smaps VMA addresses")
                current, previous_end = {}, end
                count += 1
                continue
            key, separator, tail = line.partition(":")
            if (current is None or not separator
                    or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key) is None):
                raise ValueError("invalid smaps VMA or field format")
            if key in _OLD_COUNTERS:
                if key in current:
                    raise ValueError("duplicate smaps VMA memory counter")
                current[key] = old_memory_kib(tail)
    if current is None or not _OLD_REQUIRED <= current.keys():
        raise ValueError("empty or incomplete smaps memory counters")
    for key in totals:
        if key in current:
            totals[key] += current[key]
        else:
            swap_complete = False
    if not swap_complete:
        totals["SwapPss"] = None
    return totals, count


def old_rollup(path):
    values = {}
    fields = _OLD_COUNTERS | {"Pss_Anon", "Pss_File", "Pss_Shmem"}
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if _OLD_HEADER.fullmatch(line.rstrip("\n")) is not None:
                continue
            key, separator, tail = line.partition(":")
            if not separator or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key) is None:
                raise ValueError("invalid rollup field format")
            if separator and key in fields:
                if key in values:
                    raise ValueError("duplicate rollup memory counter")
                values[key] = old_memory_kib(tail)
    if not _OLD_REQUIRED <= values.keys():
        raise ValueError("incomplete rollup memory counters")
    return values


def vma(start=0x1000, end=0x5000, *, permission="rw-p", path="[anonymous]",
        rss="16", pss="8", swap="0", newline="\n"):
    lines = [f"{start:08x}-{end:08x} {permission} 00000000 00:00 0 {path}",
             "Size: 16 kB", f"Rss: {rss} kB", f"Pss: {pss} kB",
             "VmFlags: rd wr mr mw me ac"]
    if swap is not None:
        lines.append(f"SwapPss: {swap} kB")
    return (newline.join(lines)+newline).encode("utf-8")


def outcome(function, value):
    try:
        return "success", function(value)
    except Exception as error:
        return "error", type(error), str(error)


class TextPath:
    def __init__(self, raw):
        self.raw = raw

    def open(self, **kwargs):
        # Match Path.open's universal newline decoding without touching a file.
        return io.StringIO(self.raw.decode(**kwargs), newline=None)


class PssDispatchDifferentialTests(unittest.TestCase):
    compared_smaps = 0
    compared_rollup = 0

    @classmethod
    def tearDownClass(cls):
        print("differential_cases smaps=%d rollup=%d total=%d" %
              (cls.compared_smaps, cls.compared_rollup,
               cls.compared_smaps+cls.compared_rollup))

    def check_smaps(self, raw):
        with patch.object(r, "_capture_smaps", return_value=raw) as capture:
            actual = outcome(r._read_smaps_pss, object())
        capture.assert_called_once()
        expected = outcome(old_smaps, raw)
        self.assertEqual(actual, expected, repr(raw[:180]))
        type(self).compared_smaps += 1
        return expected

    def check_rollup(self, raw):
        self.assertEqual(outcome(r._read_rollup_pss, TextPath(raw)),
                         outcome(old_rollup, TextPath(raw)), repr(raw[:180]))
        type(self).compared_rollup += 1

    def test_valid_totals_and_optional_swap_coverage(self):
        for first_swap, second_swap in itertools.product((None, "0", "3"), repeat=2):
            self.check_smaps(vma(swap=first_swap)+vma(0x9000, 0xd000, rss="12", pss="6", swap=second_swap))
        expected = self.check_smaps(vma()+vma(0x5000, 0x9000, rss="12", pss="6"))
        self.assertEqual(expected, ("success", ({"Rss": 28*1024, "Pss": 14*1024, "SwapPss": 0}, 2)))

    def test_counter_duplicate_missing_and_error_priority(self):
        base = vma()
        for key in (b"Rss", b"Pss", b"SwapPss"):
            self.check_smaps(base+key+b": 1 kB\n")
            self.check_smaps(base.replace(key+b": ", key+b": -", 1))
            self.check_smaps(base.replace(key+b": "+{b"Rss": b"16", b"Pss": b"8", b"SwapPss": b"0"}[key]+b" kB\n", b""))
        for raw in (b"Pss: -1 kB\n", b"Pss: 1 kB\nPss: 2 kB\n", b"Future: ignored\n"):
            expected = self.check_smaps(raw)
            self.assertEqual(expected[2], "invalid smaps VMA or field format")
        incomplete = base.replace(b"Pss: 8 kB\n", b"")
        expected = self.check_smaps(incomplete+vma(0x2000, 0x6000))
        self.assertEqual(expected[2], "incomplete smaps VMA memory counters")

    def test_unknown_fields_and_case_sensitive_counter_names(self):
        for key in (b"VmFlags", b"ProtectionKey", b"Pss_Anon", b"Future123", b"A_0"):
            for tail in (b"anything : nested", b"", b"-99 bytes", b"\xff\x00\t:opaque"):
                self.check_smaps(vma()+key+b":"+tail+b"\n"+key+b": repeated unknown value\n")
        for key in (b"Pss ", b" Pss", b"Pss-Other", b"_Pss", b"1Pss", b"Ps\xffs", b"Ps.s"):
            self.check_smaps(vma()+key+b": 1 kB\n")
        for key in (b"pss", b"PSS", b"pSs"):
            self.check_smaps(vma().replace(b"Pss:", key+b":"))

    def test_all_permissions_address_bounds_and_header_paths(self):
        for permission in map("".join, itertools.product("r-", "w-", "x-", "ps")):
            self.check_smaps(vma(permission=permission))
        for start, end in ((0, 1), (1, 2), (0xABCDEF, 0xABCDFF),
                           (0xffffffffff600000, 0xffffffffff601000), (7, 7), (8, 7)):
            self.check_smaps(vma(start, end))
        for path in ("/some library:device.so (deleted)", "[anon:worker arena]", "汉字:路径", ""):
            self.check_smaps(vma(path=path))
        header = vma().split(b"\n", 1)[0]
        for changed in (header.upper(), header[:17].upper()+header[17:], header.replace(b"rw-p", b"rwxq"),
                        header.replace(b"00:00", b"00-00"), header.replace(b" 0 ", b" -1 "),
                        header.replace(b"00001000-00005000", b"00001000+00005000"),
                        header.replace(b" rw-p ", b"  rw-p ")):
            self.check_smaps(changed+b"\nRss: 16 kB\nPss: 8 kB\n")

    def test_memory_grammar_preserves_unicode_whitespace_and_ascii_digits(self):
        for tail in ("0 kB", "0009 kB", "\t9\tkB\r", "\u00a09\u2003kB\u00a0", "9\vkB\f", "+9 kB", "-1 kB",
                     "9 KB", "9 kb", "9 bytes", "9 kB extra", "9.0 kB", "９ kB", "٩ kB", "9\x00 kB", ""):
            self.check_smaps(vma().replace(b"Pss: 8 kB", b"Pss: "+tail.encode("utf-8")))
        # Preserve int()'s own large-decimal limit/exception when enabled.
        self.check_smaps(vma().replace(b"Pss: 8 kB", b"Pss: "+b"9"*5000+b" kB"))

    def test_utf8_replacement_and_newline_boundaries(self):
        base = vma()
        for raw in (base.replace(b"[anonymous]", b"[\xff\xfe anonymous]"),
                    base+b"Future: opaque\xff\xfe\n", base.replace(b"Pss:", b"Ps\xffs:"),
                    base.replace(b"Pss: 8 kB", b"Pss: \xff kB"),
                    base.replace(b"rw-p", b"r\xff-p"),
                    base.replace(b"\n", b"\r\n"), base.rstrip(b"\n"), base+b"\n", base+b"\r",
                    base.replace(b"\nPss", b"\rPss"), base+"Future: line\u2028rest\n".encode("utf-8")):
            self.check_smaps(raw)

    def test_complete_prefix_never_hides_bad_or_partial_tail(self):
        base = vma()
        for tail in (b"", b"unparseable partial tail", b"Rss", b": no key\n", b"Future without separator\n",
                     b"00009000-0000d000 rw-p 00000000 00:00 0\n", vma(0x2000, 0x6000),
                     vma(0x9000, 0xd000).replace(b"Rss: 16 kB\n", b""),
                     vma(0x9000, 0xd000).replace(b"Pss: 8 kB\n", b"")):
            self.check_smaps(base+tail)
        for raw in (b"", b"\n", b"\x00", b"Future: opaque\n"):
            self.check_smaps(raw)

    def test_shared_rollup_helpers_keep_old_results_and_errors(self):
        base = b"Rss: 28 kB\nPss: 14 kB\nPss_Anon: 10 kB\nPss_File: 4 kB\nPss_Shmem: 0 kB\nSwapPss: 3 kB\n"
        for raw in (base, b"00001000-0000d000 ---p 00000000 00:00 0 [rollup]\n"+base,
                    base+b"Future: ignored\xff\n", base+b"Pss: 1 kB\n", base.replace(b"Pss: 14", b"Pss: -14"),
                    base.replace(b"Pss: 14 kB\n", b""), base.replace(b"\n", b"\r\n"),
                    base+b"Invalid-Key: no\n", base+b"Future: any\nFuture: repeated\n", b"", b"Rss: 1 kB\n"):
            self.check_rollup(raw)
        for tail in ("\u00a01\u2003kB", "１ kB", "1 bytes", "1 kB trailing"):
            self.check_rollup(base.replace(b"Pss: 14 kB", b"Pss: "+tail.encode("utf-8")))

    def test_seeded_small_byte_mutations(self):
        rng = random.Random(20261028)
        seeds = [vma(), vma()+vma(0x9000, 0xd000, swap=None),
                 vma(path="/library:version.so (deleted)"), vma()+b"Future_0: opaque: value\n"]
        alphabet = b"0123456789abcdefABCDEF-_: \t\n\r\x00\xffkBpsrwxyz"
        for ordinal in range(600):
            raw = bytearray(seeds[ordinal % len(seeds)])
            for _ in range(rng.randint(1, 4)):
                position = rng.randrange(len(raw)+1)
                operation = rng.randrange(3)
                if operation == 0:
                    raw[position:position] = bytes([rng.choice(alphabet)])
                elif operation == 1 and position < len(raw):
                    del raw[position]
                elif position < len(raw):
                    raw[position] = rng.choice(alphabet)
            self.check_smaps(bytes(raw))


if __name__ == "__main__":
    unittest.main(verbosity=2)
