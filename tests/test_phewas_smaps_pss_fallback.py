"""Older Linux full-smaps PSS fallback; anonymous process/file fixtures."""
from pathlib import Path
import io
import errno

import pytest

from fudan_wgs_toolkit import phewas_resources as r


def vma(start,end,rss,pss,swap=0,*,with_swap=True):
    return (f'{start:08x}-{end:08x} rw-p 00000000 00:00 0 [anonymous]\n'
        'Size:                 16 kB\nKernelPageSize:        4 kB\nMMUPageSize:           4 kB\n'
        f'Rss:                  {rss} kB\nPss:                  {pss} kB\n'
        'Shared_Clean:         0 kB\nPrivate_Dirty:        0 kB\nAnonymous:            0 kB\n'
        'Swap:                 0 kB\n'
        +(f'SwapPss:              {swap} kB\n' if with_swap else '')
        +'Locked:               0 kB\nVmFlags: rd wr mr mw me ac\n')


@pytest.fixture
def files(tmp_path,monkeypatch):
    folder=tmp_path/'123';folder.mkdir()
    (folder/'status').write_text('VmRSS: 900 kB\nRssAnon: 400 kB\nRssFile: 500 kB\nVmSwap: 1 kB\n')
    real_path=Path
    monkeypatch.setattr(r,'Path',lambda value:real_path(str(value).replace('/proc',str(tmp_path),1)))
    return folder


def test_missing_rollup_streams_every_vma_and_never_uses_rss_as_pss(files):
    (files/'smaps').write_text(vma(0x1000,0x5000,16,8,1)+vma(0x9000,0xd000,12,6,2))
    result=r._process_memory_counters(123,include_pss=True)
    assert result['pss_bytes']==14*1024 and result['rss_bytes']==28*1024
    assert result['swap_pss_bytes']==3*1024
    assert result['pss_source']=='smaps' and result['smaps_vma_count']==2
    assert result['pss_coverage_complete'] is True
    assert result['anonymous_pss_bytes'] is None
    assert result['mapped_file_pss_bytes'] is None and result['shared_pss_bytes'] is None


def test_old_kernel_without_optional_swap_pss_has_complete_resident_pss(files):
    (files/'smaps').write_text(vma(0x1000,0x5000,16,8,with_swap=False)
        +vma(0x9000,0xd000,12,6,with_swap=False))
    result=r._process_memory_counters(123,include_pss=True)
    assert result['pss_bytes']==14*1024 and result['pss_coverage_complete'] is True
    assert result['swap_pss_bytes'] is None


def test_optional_swap_pss_requires_full_counter_coverage_to_report_a_total(files):
    (files/'smaps').write_text(vma(0x1000,0x5000,16,8,3)+vma(0x9000,0xd000,12,6,with_swap=False))
    result=r._process_memory_counters(123,include_pss=True)
    assert result['pss_bytes']==14*1024 and result['swap_pss_bytes'] is None


def test_rollup_is_preferred_and_keeps_reliable_pss_components(files,monkeypatch):
    (files/'smaps_rollup').write_text('00001000-0000d000 ---p 00000000 00:00 0 [rollup]\n'
        'Rss: 28 kB\nPss: 14 kB\nPss_Anon: 10 kB\nPss_File: 4 kB\nPss_Shmem: 0 kB\nSwapPss: 3 kB\n')
    monkeypatch.setattr(r,'_read_smaps_pss',lambda path:pytest.fail('valid rollup must not read full smaps'))
    result=r._process_memory_counters(123,include_pss=True)
    assert result['pss_source']=='smaps_rollup' and result['pss_bytes']==14*1024
    assert result['anonymous_pss_bytes']==10*1024 and result['smaps_vma_count'] is None


@pytest.mark.parametrize('bad',[
    '', 'Rss: 16 kB\nPss: 8 kB\n',
    vma(0x1000,0x5000,16,8)+vma(0x9000,0xd000,12,6).replace('Pss:                  6 kB\n',''),
    vma(0x1000,0x5000,16,8)+vma(0x9000,0xd000,12,6).replace('Rss:                  12 kB\n',''),
    vma(0x1000,0x5000,16,8)+vma(0x9000,0xd000,12,6).replace('Pss:                  6 kB','Pss:                  wrong kB'),
    vma(0x1000,0x5000,16,8).replace('Pss:                  8 kB','Pss:                  -1 kB'),
    vma(0x1000,0x5000,16,8).replace('Pss:                  8 kB','Pss:                  8 bytes'),
    vma(0x1000,0x5000,16,8)+'Pss: 2 kB\n',
    vma(0x1000,0x5000,16,8)+vma(0x2000,0x6000,12,6),
    vma(0x1000,0x5000,16,8)+'00009000-0000d000 rw-p 00000000 00:00 0\n',
    vma(0x1000,0x5000,16,8)+'unparseable partial tail\n'])
def test_partial_empty_or_bad_smaps_never_produces_partial_pss(files,bad):
    (files/'smaps').write_text(bad)
    result=r._process_memory_counters(123,include_pss=True)
    assert result['pss_bytes'] is None and result['swap_pss_bytes'] is None
    assert result['pss_source'] is None and result['pss_coverage_complete'] is False
    assert result['rss_bytes']==900*1024  # Status remains diagnostic, never PSS.


def test_midstream_oserror_discards_complete_prefix_instead_of_admitting_partial_pss(files,monkeypatch):
    class FailingStream(io.BytesIO):
        def read(self,size=-1):
            if self.tell():
                raise OSError(errno.EIO,'anonymous midstream read error')
            return super().read(size)
    real_open=Path.open
    def opened(path,*args,**kwargs):
        if path==files/'smaps':return FailingStream(vma(0x1000,0x5000,16,8).encode())
        return real_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',opened)
    result=r._process_memory_counters(123,include_pss=True)
    assert result['pss_bytes'] is None and result['rss_bytes']==900*1024
    assert result['pss_read_errors'][-1]==dict(source='smaps',error_type='OSError',
        errno=errno.EIO,reason='endpoint_read_failed')


def test_unreadable_smaps_keeps_pss_unknown_even_when_status_has_rss(files,monkeypatch):
    real_open=Path.open
    def opened(path,*args,**kwargs):
        if path==files/'smaps':raise PermissionError('anonymous denied PSS')
        return real_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',opened)
    result=r._process_memory_counters(123,include_pss=True)
    assert result['pss_bytes'] is None and result['rss_bytes']==900*1024


def test_rollup_permission_error_can_use_complete_readable_smaps(files,monkeypatch):
    (files/'smaps').write_text(vma(0x1000,0x5000,16,8))
    real_open=Path.open
    def opened(path,*args,**kwargs):
        if path==files/'smaps_rollup':raise PermissionError('anonymous missing rollup permission')
        return real_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',opened)
    assert r._process_memory_counters(123,include_pss=True)['pss_bytes']==8*1024


def test_malformed_rollup_does_not_keep_an_earlier_successful_pss_field(files):
    (files/'smaps_rollup').write_text('Rss: 16 kB\nPss: 8 kB\nSwapPss: bad kB\n')
    result=r._process_memory_counters(123,include_pss=True)
    assert result['pss_bytes'] is None and result['pss_source'] is None
    (files/'smaps').write_text(vma(0x1000,0x5000,16,4))
    result=r._process_memory_counters(123,include_pss=True)
    assert result['pss_bytes']==4*1024 and result['pss_source']=='smaps'


def test_fallback_uses_bounded_binary_capture_not_unbounded_read_text(files,monkeypatch):
    (files/'smaps').write_text(vma(0x1000,0x5000,16,8))
    real_read=Path.read_text
    def read(path,*args,**kwargs):
        if path.name in ('smaps','smaps_rollup'):pytest.fail('PSS paths must not use unbounded read_text')
        return real_read(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    assert r._process_memory_counters(123,include_pss=True)['pss_bytes']==8*1024


def test_task_snapshot_reports_fallback_provenance_and_brackets_identity(files,monkeypatch):
    (files/'smaps').write_text(vma(0x1000,0x5000,16,8,with_swap=False))
    identity=dict(pid=123,start_ticks=1,boot_id='anonymous-boot')
    identities={123:identity}
    monkeypatch.setattr(r,'process_identity',lambda pid:identities.get(pid))
    result=r._task_memory_snapshot([identity])
    assert result['pss_bytes']==8*1024 and result['pss_sources']=={'smaps':1}
    original=r._process_memory_counters
    def changed(pid,**kwargs):
        value=original(pid,**kwargs)
        identities[pid]=dict(identity,start_ticks=2)
        return value
    monkeypatch.setattr(r,'_process_memory_counters',changed)
    with pytest.raises(RuntimeError,match='identity changed during'):
        r._task_memory_snapshot([identity])
