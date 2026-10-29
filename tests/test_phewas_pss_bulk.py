"""Bulk/fallback proc-format contracts; no scientific data or live proc scan."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fudan_wgs_toolkit import phewas_resources as r


def vma(start=0x1000, *, swap=True, newline='\n', path='[anonymous]'):
    rows=[f'{start:08x}-{start+0x1000:08x} rw-p 00000000 00:00 0 {path}',
          'Size: 4 kB','Rss: 4 kB','Pss: 2 kB','Future_Field: arbitrary:\ufffd value','VmFlags: rd wr']
    if swap:rows.append('SwapPss: 0 kB')
    return (newline.join(rows)+newline).encode()


def outcome(function, value):
    try:return 'success',function(value)
    except Exception as error:return 'error',type(error),str(error)


class BulkPssTests(unittest.TestCase):
    def compare(self, data):
        expected=outcome(r._parse_smaps_lines,data.decode('utf-8',errors='replace'))
        with patch.object(r,'_capture_smaps',return_value=data) as capture:
            actual=outcome(r._read_smaps_pss,object())
        capture.assert_called_once();self.assertEqual(actual,expected)
        return actual

    def test_valid_complete_maps_use_fast_path(self):
        for data in (vma(),vma()+vma(0x9000,swap=False),vma(newline='\r\n'),vma().rstrip(b'\n'),
                     vma(path='/anonymous/unicode_\u2603:library'),vma().replace(b'2 kB',b'\xc2\xa0 2\xe2\x80\x83kB\r')):
            expected=outcome(r._parse_smaps_lines,data.decode('utf-8',errors='replace'))
            with self.subTest(ending=data[-24:]),patch.object(r,'_parse_smaps_lines',side_effect=AssertionError('valid fast path used fallback')):
                with patch.object(r,'_capture_smaps',return_value=data) as capture:
                    self.assertEqual(outcome(r._read_smaps_pss,object()),expected)
                capture.assert_called_once()

    def test_corrupt_block_and_late_tail_use_same_capture_fallback(self):
        good=vma()+vma(0x9000)
        for data in (b'bad-prefix\n'+good,good+b'Bad-Key: ignored\n',good+b'garbage',
                     good.replace(b'Pss: 2 kB',b'Pss: -2 kB',1),good+b'Pss: 8 kB\n',
                     vma().replace(b'Pss: 2 kB\n',b'')+vma(0x1000)):
            with self.subTest(ending=data[-32:]),patch.object(r,'_parse_smaps_lines',wraps=r._parse_smaps_lines) as fallback:
                with patch.object(r,'_capture_smaps',return_value=data) as capture:
                    actual=outcome(r._read_smaps_pss,object())
                capture.assert_called_once();fallback.assert_called_once_with(data.decode('utf-8',errors='replace'))
            self.assertEqual(actual[0],'error')
            self.assertEqual(actual,outcome(r._parse_smaps_lines,data.decode('utf-8',errors='replace')))

    def test_64_field_fast_bound_preserves_longer_legal_blocks(self):
        header=vma().split(b'\n',1)[0]+b'\n';required=b'Rss: 4 kB\nPss: 2 kB\n'
        within=header+required+b'Future: ignored\n'*62
        expected=outcome(r._parse_smaps_lines,within.decode())
        with patch.object(r,'_parse_smaps_lines',side_effect=AssertionError('64 fields should be fast')):
            with patch.object(r,'_capture_smaps',return_value=within):
                self.assertEqual(outcome(r._read_smaps_pss,object()),expected)
        for count in (63,100000):
            data=header+required+b'Future: ignored\n'*count
            with patch.object(r,'_parse_smaps_lines',wraps=r._parse_smaps_lines) as fallback:
                with patch.object(r,'_capture_smaps',return_value=data) as capture:
                    actual=outcome(r._read_smaps_pss,object())
                capture.assert_called_once();fallback.assert_called_once_with(data.decode())
            self.assertEqual(actual,outcome(r._parse_smaps_lines,data.decode()))
            self.assertEqual(actual,('success',({'Rss':4096,'Pss':2048,'SwapPss':None},1)))

    def test_capture_reaches_eof_and_closes_before_fast_or_fallback_parse(self):
        for data in (vma(),vma()+b'Bad-Key: value\n'):
            events=[]
            class Endpoint(io.BytesIO):
                def readinto(self, view):
                    count=super().readinto(view)
                    if count==0:events.append('eof')
                    return count
                def close(self):events.append('closed');return super().close()
            class ProcPath:
                def open(self,*args,**kwargs):events.append('open');return Endpoint(data)
            original=r._parse_smaps_bulk
            def bulk(text):
                self.assertEqual(events,['open','eof','closed']);return original(text)
            with patch.object(r,'_capture_smaps',wraps=r._capture_smaps) as capture,patch.object(r,'_parse_smaps_bulk',side_effect=bulk):
                outcome(r._read_smaps_pss,ProcPath())
            capture.assert_called_once()

    def test_block_match_advances_only_from_current_position(self):
        real=r._SMAPS_BULK_BLOCK;positions=[]
        class Matcher:
            def match(self,text,position):positions.append(position);return real.match(text,position)
        data=vma()+vma(0x9000)+vma(0x11000)
        with patch.object(r,'_SMAPS_BULK_BLOCK',Matcher()):self.assertEqual(self.compare(data)[0],'success')
        self.assertEqual(positions,[0,len(vma().decode()),len(vma().decode())+len(vma(0x9000).decode())])

    def test_long_bad_keys_headers_and_unknown_values_keep_old_results(self):
        for size in (4096,65536,262144):
            for tail in (b'A'*size+b'!: bad\n',b'0'*size+b'-not-a-header\n',
                         b'Future: '+b'A'*size+b'\n',b'Future: '+b'\xff'*size):
                with self.subTest(bytes=size,kind=tail[:16]):self.compare(vma()+tail)

    def test_late_corruption_never_admits_complete_prefix_totals(self):
        for tail in (b'Bad-Key: value\n',b'garbage',vma(0x9000).replace(b'Pss: 2 kB\n',b'')):
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);process=root/'123';process.mkdir()
                (process/'status').write_text('VmRSS: 900 kB\n')
                (process/'smaps').write_bytes(vma()+tail)
                real_path=Path
                with patch.object(r,'Path',lambda value:real_path(str(value).replace('/proc',str(root),1))):
                    counters=r._process_memory_counters(123,include_pss=True)
            self.assertIsNone(counters['pss_bytes']);self.assertIsNone(counters['swap_pss_bytes'])
            self.assertFalse(counters['pss_coverage_complete']);self.assertEqual(counters['rss_bytes'],900*1024)
            self.assertEqual(counters['pss_read_errors'][-1]['source'],'smaps')
            self.assertEqual(counters['pss_read_errors'][-1]['error_type'],'ValueError')


if __name__=='__main__':unittest.main()
