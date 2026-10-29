"""Real spawn reader/ordering/budget/error contracts, not performance tests."""
import os
import multiprocessing
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from test_cache_portable import portable
from fudan_wgs_toolkit.cache_runtime.portable import PortableGenotypeReader
from fudan_wgs_toolkit.cache_runtime.prepared_process_pool import PreparedProcessPool


def assert_exact(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if key == "summaries":
            for a, b in zip(left[key], right[key]):
                assert a.dtype == b.dtype
                np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
        elif isinstance(left[key], np.ndarray):
            assert left[key].dtype == right[key].dtype
            np.testing.assert_array_equal(left[key].view(np.uint8), right[key].view(np.uint8))
        else:
            assert left[key] == right[key]


def requests():
    return [np.array([5, 0], dtype=np.int64), np.array([1, 3]),
            np.array([], dtype=np.int64), np.array([2, 4])]


def make_reader(cache, derived, budget=2**20, processes=2):
    return PortableGenotypeReader(cache, device="cpu", prepared_cache_directory=derived,
        prepared_cache_max_bytes=budget, prefetch_depth=2, prefetch_processes=processes)


def test_spawn_cold_then_warm_exact_order_no_cuda_or_warm_recompute(portable, tmp_path):
    _, cache, *_ = portable
    samples = np.array([2, 0, 1])
    with PortableGenotypeReader(cache, device="cpu") as control:
        expected = [control._prepare(v, samples, 2) for v in requests()]
    derived = tmp_path / "derived"
    with make_reader(cache, derived) as reader:
        actual = list(reader._prepared_requests(requests(), samples, 2))
        for got, want in zip(actual, expected):
            assert_exact(got[0], want[0])
            np.testing.assert_array_equal(got[1], want[1])
            np.testing.assert_array_equal(got[2], want[2])
        metrics = reader._metrics
        assert metrics["prefetch_process_requests_submitted"] == 4
        assert metrics["prefetch_process_cold_prepared"] == 4
        assert metrics["cohort_compact_writes"] == 4
        assert metrics.get("cohort_compact_hits", 0) == 0
        assert metrics["cohort_compact_misses"] == 4
        assert metrics["prefetch_process_child_frame_loads"] > 0
        info = reader.reader_metadata["analysis_cache"]["process_pool"]
        assert info["active_processes"] == 2
        assert info["workers"] and all(value["cuda_initialized"] is False for value in info["workers"])
        assert {value["pid"] for value in info["workers"]} == set(reader._prepare_pool._executor._processes)
        assert all(len(value["source_binding_sha256"]) == 64 for value in info["workers"])
        assert info["pending_estimated_bytes_current"] == 0
        pids = list(reader._prepare_pool._executor._processes)
        frame_loads = metrics["frame_loads"]
        writes = metrics["cohort_compact_writes"]
        # The spawning loader thread has finished. Parent-process protection
        # must keep a persistent pool alive past a watchdog tick.
        time.sleep(1.25)
        warm = list(reader._prepared_requests(requests(), samples, 2))
        assert metrics["prefetch_process_warm_hits"] == 4
        assert metrics["prefetch_process_requests_submitted"] == 4
        assert metrics["frame_loads"] == frame_loads
        assert metrics["cohort_compact_writes"] == writes
        for got, want in zip(warm, expected):
            assert_exact(got[0], want[0])
        temporary = reader._prepare_pool._temporary
    assert not temporary.exists()
    assert all(not Path(f"/proc/{pid}").exists() for pid in pids)


def test_budget_exhaustion_handoff_never_reprepares_same_request(portable, tmp_path):
    _, cache, *_ = portable
    samples = np.arange(4, dtype=np.int64)
    # Exactly one small entry can fit. Concurrent misses after it fill the
    # cache use transient handoffs, while later requests run the original path.
    with make_reader(cache, tmp_path / "small", budget=8192) as reader:
        calls = []
        original = reader._prepare
        def tracked(v, s, mac):
            calls.append(tuple(v))
            return original(v, s, mac)
        reader._prepare = tracked
        got = list(reader._prepared_requests(requests()*2, samples, None))
        assert len(got) == 8
        assert reader._metrics.get("prefetch_process_budget_skips", 0) > 0
        assert reader._metrics["prefetch_process_handoff_requests"] == reader._metrics["prefetch_process_budget_skips"]
        assert reader._metrics["prefetch_process_transient_current_bytes"] == 0
        assert reader._metrics["prefetch_process_transient_peak_bytes"] > 0
        # Every initial dispatched request is prepared in a child only.
        submitted = reader._metrics["prefetch_process_requests_submitted"]
        warm = reader._metrics.get("prefetch_process_warm_hits", 0)
        assert len(calls) == 8-submitted-warm
        assert reader._metrics["prefetch_process_fallback_requests"] == len(calls)
        assert reader._prepare_pool._disabled


def test_obviously_full_budget_and_disabled_cache_keep_original_thread(portable, tmp_path):
    _, cache, *_ = portable
    for derived, budget in ((None, 2**20), (tmp_path / "full", 1)):
        with PortableGenotypeReader(cache, device="cpu", prepared_cache_directory=derived,
                prepared_cache_max_bytes=budget, prefetch_depth=2, prefetch_processes=2) as reader:
            assert len(list(reader._prepared_requests(requests(), np.arange(4), None))) == 4
            assert reader._metrics.get("prefetch_process_requests_submitted", 0) == 0
            if derived is not None:
                assert reader._metrics["prefetch_process_fallback_requests"] == 4


def test_early_close_drains_before_reader_close_and_keeps_mapped_result(portable, tmp_path):
    _, cache, *_ = portable
    reader = make_reader(cache, tmp_path / "derived")
    iterator = reader._prepared_requests(requests()*3, np.arange(4), None)
    first = next(iterator)
    preserved = first[0]["exception_state"].copy()
    iterator.close()
    assert not any(t.name == "fudan_wgs_toolkit-compact-prefetch" for t in threading.enumerate())
    assert reader._metrics["prefetch_queued_items"] == 0
    pids = list(reader._prepare_pool._executor._processes)
    reader.close()
    assert all(not Path(f"/proc/{pid}").exists() for pid in pids)
    np.testing.assert_array_equal(first[0]["exception_state"], preserved)


def test_corrupt_cache_and_child_source_binding_errors_propagate(portable, tmp_path):
    _, cache, *_ = portable
    with make_reader(cache, tmp_path / "derived") as reader:
        list(reader._prepared_requests(requests()[:1], np.arange(4), None))
        blob = next((tmp_path / "derived").glob("*.cpc"))
        with blob.open("r+b") as stream:
            stream.seek(-1, 2)
            byte = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([byte[0]^1]))
        with pytest.raises(ValueError, match="checksum"):
            list(reader._prepared_requests(requests()[:1], np.arange(4), None))
    with make_reader(cache, tmp_path / "badbinding") as reader:
        reader._process_descriptor = dict(reader._process_descriptor,
            source_binding={"wrong": "identity"})
        from concurrent.futures.process import BrokenProcessPool
        with pytest.raises(BrokenProcessPool):
            list(reader._prepared_requests(requests()[:1], np.arange(4), None))


def test_zero_requests_does_not_spawn(portable, tmp_path):
    _, cache, *_ = portable
    with make_reader(cache, tmp_path / "empty") as reader:
        assert list(reader._prepared_requests([], np.arange(4), None)) == []
        assert reader._prepare_pool._executor is None


def test_estimate_rejects_invalid_raw_header_before_spawn(portable, tmp_path, monkeypatch):
    _, cache, *_ = portable
    from fudan_wgs_toolkit.cache_runtime import prepared_process_pool
    original = prepared_process_pool.json.loads
    def damaged_header(text, *args, **kwargs):
        value = original(text, *args, **kwargs)
        if isinstance(value, dict) and "csr_meta" in value:
            value["csr_meta"]["raw_bytes"] = -1
        return value
    with make_reader(cache, tmp_path / "invalid") as reader:
        monkeypatch.setattr(prepared_process_pool.json, "loads", damaged_header)
        with pytest.raises(ValueError, match="checksum"):
            list(reader._prepared_requests(requests(), np.arange(4), None))
        assert reader._prepare_pool._executor is None


def test_oversized_preparation_is_alone_and_dead_child_does_not_retry(portable, tmp_path):
    _, cache, *_ = portable
    samples = np.arange(4)
    reader = make_reader(cache, tmp_path / "bounded")
    pool = PreparedProcessPool(reader, samples, 2, memory_bytes=1)
    reader._prepare_pool = pool
    try:
        values = list(pool.iter_prepared(requests(), None))
        assert len(values) == 4
        assert reader._metrics["prefetch_process_oversized_requests"] == 4
        assert reader._metrics["prefetch_process_pending_estimated_bytes_max"] == max(pool._estimate(v) for v in requests())
        # An unexpected CPU child exit breaks its executor visibly, including
        # future dispatch; it must never fall back and conceal the failure.
        pid, child = next(iter(pool._executor._processes.items()))
        manager = pool._executor._executor_manager_thread
        identities = list(pool.metadata()["workers"])
        import signal
        from multiprocessing.connection import wait
        os.kill(pid, signal.SIGKILL)
        # The sentinel establishes OS exit without reaping a Process owned by
        # the executor. Concurrent join()/poll() can observe ECHILD before the
        # manager has assigned returncode, transiently making is_alive() true.
        assert wait([child.sentinel], timeout=5) == [child.sentinel]
        # Let the broken executor own all child joins, with a separate bounded
        # cleanup check before observing its Process objects or testing reuse.
        manager.join(timeout=5)
        assert not manager.is_alive(), "broken executor did not finish child cleanup"
        assert child.exitcode == -signal.SIGKILL
        assert not child.is_alive()
        assert not any(_same_live_identity(value) for value in identities)
        from concurrent.futures.process import BrokenProcessPool
        for request in (requests()[0], np.array([0])):
            with pytest.raises(BrokenProcessPool):
                list(reader._prepared_requests([request], samples, None))
        pool._disabled = True
        with pytest.raises(BrokenProcessPool):
            list(reader._prepared_requests([np.array([0])], samples, None))
        assert reader._metrics.get("prefetch_process_fallback_requests", 0) == 0
    finally:
        reader.close()


def _leave_reserved_write(variant_bytes, marker):
    """Disposable child fixture holding an unfinished atomic write lock."""
    from fudan_wgs_toolkit.cache_runtime import prepared_process_pool
    reader = prepared_process_pool._CHILD
    cache = reader._prepared_cache
    variants = np.frombuffer(variant_bytes, dtype=np.int64)
    key, _, _, _ = cache._identity(variants, prepared_process_pool._SAMPLES, None)
    temporary = key + ".interrupted.tmp"
    with cache._key_lock(key):
        with cache._database() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("INSERT INTO entries VALUES(?,?,?,?,?,?)", (key,"writing",12,8192,None,temporary))
            connection.execute("UPDATE config SET used_bytes=used_bytes+8192 WHERE id=1")
            connection.commit()
        (cache.directory / temporary).write_bytes(b"unfinished")
        Path(marker).write_text(key)
        time.sleep(60)


def _disposable_owner(cache_directory, derived_directory, connection):
    reader = make_reader(cache_directory, derived_directory)
    try:
        list(reader._prepared_requests([np.array([5, 0])], np.arange(4), None))
        pool = reader._prepare_pool
        marker = str(pool._temporary / "unfinished-test.ready")
        pool._executor.submit(_leave_reserved_write, np.array([2]).tobytes(), marker)
        deadline = time.monotonic()+20
        while not Path(marker).exists() and time.monotonic() < deadline:
            time.sleep(.01)
        if not Path(marker).exists():
            raise RuntimeError("disposable child write fixture did not become ready")
        connection.send((pool.metadata()["workers"], reader._prepared_cache.source_binding,
            Path(marker).read_text(), str(pool._temporary)))
        time.sleep(60)
    except BaseException as error:
        connection.send(dict(error=repr(error)))
        raise
    finally:
        reader.close()
        connection.close()


def _same_live_identity(value):
    try:
        fields = Path("/proc/%d/stat" % value["pid"]).read_text().split(") ",1)[1].split()
        return int(fields[19]) == value["start_ticks"] and fields[0] not in ("Z", "X")
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="Linux process identity contract")
def test_owner_sigkill_exits_all_children_and_unfinished_write_recovers(portable, tmp_path):
    _, cache, *_ = portable
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    derived = tmp_path / "owner-death"
    owner = context.Process(target=_disposable_owner, args=(str(cache), str(derived), sender))
    children = []
    owner.start()
    sender.close()
    try:
        assert receiver.poll(30), "disposable owner did not report initialized children"
        report = receiver.recv()
        assert not isinstance(report, dict), report
        children, binding, key, transient = report
        assert len(children) == 2 and all(_same_live_identity(value) for value in children)
        import signal
        os.kill(owner.pid, signal.SIGKILL)
        owner.join(5)
        assert not owner.is_alive()
        deadline = time.monotonic()+5
        while any(_same_live_identity(value) for value in children) and time.monotonic() < deadline:
            time.sleep(.05)
        assert not any(_same_live_identity(value) for value in children), "CPU children survived their owning process"
        from fudan_wgs_toolkit.cache_runtime.cohort_compact_cache import CohortCompactCache
        cache_store = CohortCompactCache(derived, binding, 2**20)
        # The exact key lock is now released: recovery removes the reservation
        # and unfinished bytes; a later exact request can publish normally.
        assert cache_store.load(np.array([2]), np.arange(4), None) is None
        assert cache_store.metrics["cohort_compact_recovered_writes"] == 1
        assert not (derived / (key+".interrupted.tmp")).exists()
        with make_reader(cache, derived) as reader:
            assert len(list(reader._prepared_requests([np.array([2])], np.arange(4), None))) == 1
        # Fatal owner death preserves its private scratch directory for audit.
        assert Path(transient).exists()
    finally:
        if owner.is_alive():
            owner.terminate()
            owner.join(5)
        import signal
        for value in children:
            if _same_live_identity(value):
                os.kill(value["pid"], signal.SIGKILL)
        receiver.close()
