"""Host-local, cross-process admission for shared PheWAS preparation.

Only the standard library is used. Ledgers and process/node identities belong
in the private task directory. Memory is admitted, never waited for while a
partial genotype is retained; callers unwind preparation before retrying.
"""
from __future__ import annotations
from ._cuda_memory import allocator_bytes

from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import posixpath
import re
import socket
import time
import uuid


GiB = 2**30


class HostMemoryDeferred(MemoryError):
    """Temporary pressure: release prepared data before requesting a new lease."""
    def __init__(self, required_bytes, *, total_required_bytes=None,
                 available_bytes=0, reason="host_memory_pressure"):
        self.required_bytes = int(required_bytes)
        self.required = self.required_bytes
        self.total_required_bytes = int(total_required_bytes or required_bytes)
        self.available_bytes = max(0, int(available_bytes))
        self.reason = str(reason)
        super().__init__(f"host preparation deferred: {self.required_bytes} workspace bytes; {self.reason}")


class HostMemoryRequestTooLarge(MemoryError):
    """An immutable capacity cannot admit this job, even without other leases."""
    def __init__(self, required_bytes, capacity_bytes):
        self.required_bytes, self.capacity_bytes = int(required_bytes), int(capacity_bytes)
        super().__init__("host preparation request exceeds capacity after its memory reserve")


def _positive_bytes(value, name, *, allow_zero=True):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a nonnegative integer byte count")
    return value


def _gib_bytes(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite GiB value")
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} GiB value")
    return int(value*GiB)


def process_identity(pid):
    """Bind a Linux process to PID, start ticks and the running boot."""
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
        fields = stat[stat.rindex(")")+2:].split()
        if fields[0] in ("Z", "X"):
            return None
        return dict(pid=int(pid), start_ticks=int(fields[19]),
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    except (OSError, ValueError, IndexError):
        return None


def _process_rss(pid):
    try:
        for line in Path(f"/proc/{int(pid)}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])*1024
    except (OSError, ValueError):
        pass
    return 0


_SMAPS_HEADER = re.compile(r"^([0-9a-fA-F]+)-([0-9a-fA-F]+) [r-][w-][x-][ps] "
    r"[0-9a-fA-F]+ [0-9a-fA-F]+:[0-9a-fA-F]+ [0-9]+(?:\s+.*)?$")
_PSS_COUNTERS = frozenset(("Rss", "Pss", "SwapPss"))
_PSS_REQUIRED = frozenset(("Rss", "Pss"))


def _memory_kib(tail):
    if re.fullmatch(r"\s*[0-9]+\s+kB\s*", tail) is None:
        raise ValueError("memory counter must be nonnegative integer kB")
    return int(tail.split()[0])*1024


def _read_rollup_pss(path):
    values = {}
    fields = _PSS_COUNTERS | {"Pss_Anon", "Pss_File", "Pss_Shmem"}
    with path.open(encoding="utf-8",errors="replace") as stream:
        for line in stream:
            if _SMAPS_HEADER.fullmatch(line.rstrip("\n")) is not None:
                continue
            key, separator, tail = line.partition(":")
            if not separator or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*",key) is None:
                raise ValueError("invalid rollup field format")
            if separator and key in fields:
                if key in values:
                    raise ValueError("duplicate rollup memory counter")
                values[key] = _memory_kib(tail)
    if not _PSS_REQUIRED <= values.keys():
        raise ValueError("incomplete rollup memory counters")
    return values


def _read_smaps_pss(path):
    """Stream all VMAs; one invalid/incomplete block invalidates every total.

    Older kernels expose PSS only here. Each complete mapping contributes
    exactly once. VMA path/type cannot reliably split proportional anonymous,
    file and shared memory, so this fallback reports only Rss/Pss/SwapPss.
    """
    totals = dict.fromkeys(_PSS_COUNTERS,0)
    swap_complete = True
    current, previous_end, count = None,0,0
    with path.open(encoding="utf-8",errors="replace") as stream:
        for line in stream:
            header = _SMAPS_HEADER.fullmatch(line.rstrip("\n"))
            if header is not None:
                if current is not None:
                    if not _PSS_REQUIRED <= current.keys():
                        raise ValueError("incomplete smaps VMA memory counters")
                    for key in totals:
                        if key in current:
                            totals[key] += current[key]
                        else:
                            swap_complete = False
                start,end = (int(value,16) for value in header.groups())
                if start<previous_end or start>=end:
                    raise ValueError("invalid or overlapping smaps VMA addresses")
                current,previous_end = {},end
                count += 1
                continue
            key, separator, tail = line.partition(":")
            if (current is None or not separator
                    or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*",key) is None):
                raise ValueError("invalid smaps VMA or field format")
            if key in _PSS_COUNTERS:
                if key in current:
                    raise ValueError("duplicate smaps VMA memory counter")
                current[key] = _memory_kib(tail)
    if current is None or not _PSS_REQUIRED <= current.keys():
        raise ValueError("empty or incomplete smaps memory counters")
    for key in totals:
        if key in current:
            totals[key] += current[key]
        else:
            swap_complete = False
    if not swap_complete:
        totals["SwapPss"] = None
    return totals,count


def _process_memory_counters(pid, *, include_pss):
    """Read one endpoint; unavailable PSS remains unknown rather than zero."""
    status = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, separator, tail = line.partition(":")
            if separator and key in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap"):
                status[key] = int(tail.split()[0])*1024
    except (OSError, ValueError, IndexError):
        pass
    rollup = {}
    pss_source,smaps_vma_count = None,None
    if include_pss:
        try:
            rollup = _read_rollup_pss(Path(f"/proc/{pid}/smaps_rollup"))
            pss_source = "smaps_rollup"
        except (OSError, ValueError, IndexError):
            try:
                rollup,smaps_vma_count = _read_smaps_pss(Path(f"/proc/{pid}/smaps"))
                pss_source = "smaps"
            except (OSError, ValueError, IndexError):
                # Never keep counters accumulated before a partial read or
                # malformed later VMA; RSS may still be diagnostic status data.
                rollup = {}
    return dict(rss_bytes=rollup.get("Rss",status.get("VmRSS")),
        pss_bytes=rollup.get("Pss"), anonymous_rss_bytes=status.get("RssAnon"),
        mapped_file_rss_bytes=status.get("RssFile"), shared_rss_bytes=status.get("RssShmem"),
        anonymous_pss_bytes=rollup.get("Pss_Anon"), mapped_file_pss_bytes=rollup.get("Pss_File"),
        shared_pss_bytes=rollup.get("Pss_Shmem"), swap_bytes=status.get("VmSwap"),
        swap_pss_bytes=rollup.get("SwapPss"),pss_source=pss_source,
        pss_coverage_complete=pss_source is not None,smaps_vma_count=smaps_vma_count)


def _verified_children(pid, *, strict=False):
    """Loader-spawned children are attached to their TID, not just the leader."""
    children = set()
    try:
        for task in Path(f"/proc/{pid}/task").iterdir():
            try:
                children.update(int(value) for value in (task/"children").read_text().split())
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (OSError, ValueError):
                if strict:
                    raise RuntimeError("task descendant membership cannot be read safely")
    except (FileNotFoundError, ProcessLookupError):
        pass
    except OSError:
        if strict:
            raise RuntimeError("task thread membership cannot be read safely")
    result = []
    for child in children:
        try:
            fields = Path(f"/proc/{child}/stat").read_text().rsplit(")",1)[1].split()
            if int(fields[1]) == pid:
                result.append(child)
        except (FileNotFoundError, ProcessLookupError):
            pass
        except (OSError, ValueError, IndexError):
            if strict:
                raise RuntimeError("task child identity cannot be read safely")
    return result


def _valid_identity(identity):
    return (isinstance(identity,dict) and set(identity)=={"pid","start_ticks","boot_id"}
        and type(identity["pid"]) is int and identity["pid"]>0
        and type(identity["start_ticks"]) is int and identity["start_ticks"]>=0
        and isinstance(identity["boot_id"],str) and bool(identity["boot_id"]))


def _task_processes(owners):
    """Refresh registered roots and their descendants, deduplicating identities.

    Remember children already observed so an orphan cannot escape the task
    budget merely because its registered parent exits or closes its gate.
    """
    identities = {}
    for owner in owners.values():
        root = owner["identity"]
        roots = ([] if owner.get("closed",False) else [root])+owner["descendants"]
        pending, descendants, seen = list(roots), [], set()
        while pending:
            identity = pending.pop()
            key = (identity["pid"],identity["start_ticks"],identity["boot_id"])
            if key in seen or process_identity(identity["pid"]) != identity:
                continue
            seen.add(key);identities[key] = identity
            if identity != root:
                descendants.append(identity)
            for child in _verified_children(identity["pid"],strict=True):
                current = process_identity(child)
                if current is not None:
                    pending.append(current)
        owner["descendants"] = descendants
    return list(identities.values())


def _task_memory_snapshot(identities):
    """Fresh proportional resident bytes; summed RSS is diagnostic only."""
    started = time.perf_counter()
    pss = rss = 0
    sources = {}
    rss_complete = True
    for identity in identities:
        if process_identity(identity["pid"]) != identity:
            raise RuntimeError("task PSS process identity changed before sampling")
        counters = _process_memory_counters(identity["pid"],include_pss=True)
        if process_identity(identity["pid"]) != identity:
            raise RuntimeError("task PSS process identity changed during sampling")
        value = counters["pss_bytes"]
        if type(value) is not int or value<0:
            raise RuntimeError("task PSS coverage cannot be verified")
        pss += value
        source = counters.get("pss_source","unspecified")
        sources[source] = sources.get(source,0)+1
        value = counters["rss_bytes"]
        if type(value) is int and value>=0:
            rss += value
        else:
            rss_complete = False
    if not identities:
        raise RuntimeError("task PSS has no verified registered owners")
    return dict(pss_bytes=pss,rss_bytes=rss if rss_complete else None,
        pss_sources=sources,
        observed_process_count=len(identities),pss_coverage_complete=True,
        sample_host_seconds=time.perf_counter()-started,rss_sum_is_unique_memory=False,
        scope="registered task roots and descendants; proportional resident endpoint, not an independent cgroup cap")


def process_memory_snapshot(identity=None, *, include_children=True, include_pss=True):
    """Anonymous, identity-bracketed process-tree RSS/PSS and mapping counters.

    This explicit diagnostic call may read ``smaps_rollup``; it is not called
    by admission. RSS sums include shared mappings. PSS is an endpoint sharing
    estimate, not project-exclusive physical use or a peak. Missing counters
    stay ``None`` while their observed subtotal and coverage remain available.
    """
    identity = process_identity(os.getpid()) if identity is None else identity
    names = ("rss_bytes", "pss_bytes", "anonymous_rss_bytes", "mapped_file_rss_bytes",
        "shared_rss_bytes", "anonymous_pss_bytes", "mapped_file_pss_bytes",
        "shared_pss_bytes", "swap_bytes", "swap_pss_bytes")
    records, pending, seen = [], ([] if identity is None else [identity["pid"]]), set()
    root_valid = identity is not None and process_identity(identity["pid"]) == identity
    if root_valid:
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            current = process_identity(pid)
            if current is None:
                continue
            seen.add(pid)
            children = _verified_children(pid) if include_children else []
            value = _process_memory_counters(pid,include_pss=include_pss)
            if process_identity(pid) != current:
                continue
            records.append(value)
            pending.extend(children)
        root_valid = process_identity(identity["pid"]) == identity
    if not root_valid:
        records = []
    count = len(records)
    coverage = {name:sum(value.get(name) is not None for value in records) for name in names}
    result = {name:(sum(value[name] for value in records) if count and coverage[name] == count else None)
        for name in names}
    result.update(observed_subtotals={name:sum(value.get(name) or 0 for value in records) for name in names},
        counter_process_coverage=coverage, observed_process_count=count,
        root_identity_verified=root_valid, descendant_traversal_requested=bool(include_children),
        pss_requested=bool(include_pss), rss_sum_is_unique_memory=False,
        pss_is_endpoint_estimate=True, scope="identity-verified owned process tree endpoint")
    return result


def tensor_storage_footprint(tensors):
    """Actual backing bytes held by explicit tensor/array owners, with aliases deduplicated.

    No tensor is moved, synchronized or retained by this function. Intersecting
    address ranges on each device count once, including a small view keeping
    a larger allocation alive. This is ownership coverage supplied by callers;
    it does not claim to enumerate all allocator or native CUDA allocations.
    """
    pending = [tensors]
    intervals, observed, mapped_host = {}, 0, 0
    while pending:
        value = pending.pop()
        if value is None:
            continue
        if isinstance(value,dict):
            pending.extend(value.values());continue
        if isinstance(value,(tuple,list)):
            pending.extend(value);continue
        if hasattr(value,"untyped_storage"):
            if str(value.device).startswith("meta"):
                raise ValueError("meta tensors do not own materialized memory")
            storage = value.untyped_storage()
            device, start, size = str(value.device),int(storage.data_ptr()),int(storage.nbytes())
        elif hasattr(value,"__array_interface__"):
            owner = value
            while hasattr(getattr(owner,"base",None),"__array_interface__"):
                owner = owner.base
            device, start, size = "cpu",int(owner.__array_interface__["data"][0]),int(owner.nbytes)
            if hasattr(owner,"filename"):
                mapped_host += size
        else:
            raise TypeError("storage telemetry accepts explicit tensors, arrays and nested containers")
        if size < 0 or start < 0:
            raise ValueError("storage telemetry received invalid backing geometry")
        observed += 1
        if size:
            intervals.setdefault(device,[]).append((start,start+size))
    totals = {}
    for device,ranges in intervals.items():
        total, begin, end = 0,None,None
        for left,right in sorted(ranges):
            if begin is None:
                begin,end=left,right
            elif left<=end:
                end=max(end,right)
            else:
                total+=end-begin;begin,end=left,right
        totals[device]=total+(0 if begin is None else end-begin)
    return dict(retained_storage_bytes_by_device=totals,retained_storage_bytes=sum(totals.values()),
        unique_cuda_storage_bytes=sum(size for device,size in totals.items() if device.startswith("cuda")),
        observed_tensor_array_count=observed,mapped_host_bytes_before_alias_deduplication=mapped_host,
        logical_view_bytes_are_not_backing_bytes=True,ownership_coverage_is_explicit=True)


def sample_gpu_memory(device, *, limit_bytes=None, reserve_bytes=256*2**20,
                      owned_tensors=(), required_new_bytes=0):
    """One shared CUDA NEW-byte admission arithmetic and backing-storage endpoint.

    Live allocator bytes already include materialized tensors. Driver free
    bytes constrain shared-card pressure; unused allocator reservations can be
    reused. Neither explicit owned-storage bytes nor reserved bytes are added
    a second time. Sampling does not empty caches or change allocator policy.
    """
    _positive_bytes(reserve_bytes,"reserve_bytes")
    _positive_bytes(required_new_bytes,"required_new_bytes")
    if limit_bytes is not None:
        _positive_bytes(limit_bytes,"limit_bytes",allow_zero=False)
    import torch
    target = torch.device(device)
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda",torch.cuda.current_device())
    footprint = tensor_storage_footprint(owned_tensors)
    if target.type != "cuda":
        return dict(device_type=target.type,**footprint,cuda_memory_observed=False)
    allocated, reserved = allocator_bytes(target)
    free,total = map(int,torch.cuda.mem_get_info(target))
    unused = max(0,reserved-allocated)
    capacity = total if limit_bytes is None else min(total,limit_bytes)
    available = max(0,min(capacity-allocated,free+unused)-reserve_bytes)
    return dict(device_type="cuda",cuda_memory_observed=True,**footprint,
        allocated_bytes=allocated,reserved_bytes=reserved,unused_reserved_bytes=unused,
        driver_free_bytes=free,physical_total_bytes=total,effective_limit_bytes=capacity,
        explicit_limit_bytes=limit_bytes,reserve_bytes=reserve_bytes,available_new_bytes=available,
        owned_current_device_storage_bytes=footprint["retained_storage_bytes_by_device"].get(str(target),0),
        required_new_bytes=required_new_bytes,new_allocation_admitted=required_new_bytes<=available,
        owned_storage_already_in_allocated=True,driver_free_is_shared_card_constraint=True,
        scope="owned allocator and explicit backing storage endpoint with live shared-device free memory")


class ResourceTelemetry:
    """Optional rate-limited endpoints; admission never depends on cached samples."""
    def __init__(self,minimum_interval_seconds=5.):
        if (isinstance(minimum_interval_seconds,bool) or not isinstance(minimum_interval_seconds,(int,float))
                or not math.isfinite(minimum_interval_seconds) or minimum_interval_seconds<0):
            raise ValueError("telemetry interval must be finite and nonnegative")
        self.interval = float(minimum_interval_seconds)
        self._last_time,self._last,self._last_key = None,None,None

    def sample(self, *, device=None, limit_bytes=None, reserve_bytes=256*2**20,
               owned_tensors=(), queue_summary=None, include_pss=True, force=False):
        now = time.monotonic()
        key = (str(device),limit_bytes,reserve_bytes,bool(include_pss))
        if not force and self._last is not None and key==self._last_key and now-self._last_time<self.interval:
            return copy.deepcopy(dict(self._last,reused_sample=True,sample_age_seconds=max(0,now-self._last_time)))
        result = dict(host=read_host_memory(),processes=process_memory_snapshot(include_pss=include_pss),
            sampled_at=time.time(),reused_sample=False,sample_age_seconds=0.)
        if device is not None:
            result["gpu"] = sample_gpu_memory(device,limit_bytes=limit_bytes,reserve_bytes=reserve_bytes,
                                             owned_tensors=owned_tensors)
        if queue_summary is not None:
            # Copy numerical observations, never queue/model/tensor owners.
            result["queues"] = json.loads(json.dumps(queue_summary,allow_nan=False))
        self._last_time,self._last,self._last_key = now,result,key
        return copy.deepcopy(result)


def process_tree_rss(identity):
    """Read the identity-verified owner and all verified process descendants.

    Linux records children per TID, including children spawned by a loader
    thread. Shared mmap pages can inflate summed RSS. This is a diagnostic
    observation; host admission uses authoritative live pressure and NEW bytes.
    """
    if process_identity(identity["pid"]) != identity:
        return 0
    total, pending, seen = 0, [identity["pid"]], set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        current = process_identity(pid)
        if current is None:
            continue
        seen.add(pid)
        children = set()
        try:
            for task in Path(f"/proc/{pid}/task").iterdir():
                try:
                    children.update(int(v) for v in (task/"children").read_text().split())
                except (OSError, ValueError):
                    continue
        except OSError:
            pass
        rss = _process_rss(pid)
        if process_identity(pid) != current:
            continue
        total += rss
        for child in children:
            try:
                fields = Path(f"/proc/{child}/stat").read_text().rsplit(")",1)[1].split()
                if int(fields[1]) == pid:
                    pending.append(child)
            except (OSError, ValueError, IndexError):
                continue
    if process_identity(identity["pid"]) != identity:
        return 0
    return total


def _unescape(value):
    return re.sub(r"\\(040|011|012|134)", lambda m: chr(int(m.group(1),8)), value)


def _memory_directories():
    """Resolve memory memberships against Linux mount and cgroup namespaces."""
    memberships, mounts = [], []
    try:
        raw = Path("/proc/self/cgroup").read_text()
        for line in raw.splitlines():
            parts = line.split(":",2)
            if len(parts) != 3:
                continue
            if parts[0] == "0" and not parts[1]:
                memberships.append((2,parts[2]))
            elif "memory" in parts[1].split(","):
                memberships.append((1,parts[2]))
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, separator, right = line.partition(" - ")
            a,b = left.split(),right.split()
            if not separator or len(a)<6 or len(b)<3:
                continue
            if b[0] == "cgroup2":
                version = 2
            elif b[0] == "cgroup" and "memory" in ",".join([b[1],b[2],a[5]]).split(","):
                version = 1
            else:
                continue
            mounts.append((version,_unescape(a[3]),Path(_unescape(a[4]))))
    except OSError:
        pass
    result = []
    for version,membership in memberships:
        group = posixpath.normpath("/"+membership.lstrip("/"))
        for candidate,root,mount in mounts:
            if candidate != version:
                continue
            root = posixpath.normpath("/"+root.lstrip("/"))
            if root == "/":
                relative = group.lstrip("/")
            elif group == root:
                relative = ""
            elif group.startswith(root+"/"):
                relative = group[len(root)+1:]
            else:
                relative = group.lstrip("/")
            leaf = mount/relative
            # A cgroup namespace may name its own root '/', while a host
            # mount exposes that root directly rather than another subdir.
            if not leaf.exists() and mount.exists():
                leaf = mount
            if leaf.exists():
                result.append((version,leaf,mount))
    if not result:
        for version,path in ((1,Path("/sys/fs/cgroup/memory")),(2,Path("/sys/fs/cgroup"))):
            if (path/("memory.limit_in_bytes" if version == 1 else "memory.max")).exists():
                result.append((version,path,path))
    return result


def _ancestors(leaf, mount):
    current = leaf
    while True:
        yield current
        if current == mount or mount not in current.parents:
            return
        current = current.parent


def _memory_constraint(version, directory):
    try:
        limit_text = (directory/("memory.limit_in_bytes" if version == 1 else "memory.max")).read_text().strip()
    except FileNotFoundError:
        # The cgroup v2 hierarchy root can expose counters without a max
        # controller file. No visible limit exists at that level.
        return None
    except OSError as error:
        raise RuntimeError("visible cgroup memory limit cannot be read safely") from error
    try:
        if limit_text == "max":
            return None
        maximum = int(limit_text)
        if maximum < 0:
            raise ValueError("negative memory limit")
        if maximum >= 2**60:
            return None
        current = int((directory/("memory.usage_in_bytes" if version == 1 else "memory.current")).read_text())
        stats = dict(line.split() for line in (directory/"memory.stat").read_text().splitlines())
        inactive = int(stats.get("total_inactive_file",stats.get("inactive_file",0))) if version == 1 else int(stats.get("inactive_file",0))
        if current < 0 or inactive < 0:
            raise ValueError("negative memory pressure counter")
        prefix = "total_" if version == 1 else ""
        def counter(*names):
            for name in names:
                if name in stats:
                    value = int(stats[name])
                    if value<0:raise ValueError("negative memory telemetry counter")
                    return value
            return None
        return dict(current_bytes=current,limit_bytes=maximum,inactive_file_bytes=inactive,
            effective_current_bytes=max(0,current-inactive),scope=f"cgroup_v{version}",
            anonymous_bytes=counter(prefix+"rss","anon","rss"),
            file_cache_bytes=counter(prefix+"cache","file","cache"),
            mapped_file_bytes=counter(prefix+"mapped_file","file_mapped","mapped_file"),
            shared_memory_bytes=counter(prefix+"shmem","shmem"),
            swap_bytes=counter(prefix+"swap","swap"),kernel_bytes=counter("kernel"),
            slab_bytes=counter("slab"),inactive_file_is_reclaimable_estimate=True,
            current_bytes_include_shared_background=True)
    except (OSError,ValueError) as error:
        # Never silently discard a finite ancestor because its usage/stat
        # read failed: the wider physical limit cannot substitute for it.
        raise RuntimeError("finite cgroup memory pressure cannot be read safely") from error


def read_host_memory():
    """Read all finite memory ancestors plus physical available memory.

    As in the established runner, inactive file-cache pages are reclaimable and
    excluded from current pressure. Every finite ancestor is checked; a parent
    with other jobs may constrain admission more than its child limit alone.
    Reports contain anonymous numerical constraints, not membership paths.
    """
    constraints, seen = [], set()
    for version,leaf,mount in _memory_directories():
        for directory in _ancestors(leaf,mount):
            key = (version,str(directory))
            if key not in seen:
                seen.add(key)
                constraint = _memory_constraint(version,directory)
                if constraint:
                    constraint["visible_ancestor_index"] = len(constraints)
                    constraints.append(constraint)
    try:
        values = {line.split(":",1)[0]:int(line.split()[1])*1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(("MemTotal:","MemAvailable:"))}
        if "MemTotal" in values and "MemAvailable" in values:
            effective = max(0,values["MemTotal"]-values["MemAvailable"])
            constraints.append(dict(current_bytes=effective,limit_bytes=values["MemTotal"],
                inactive_file_bytes=0,effective_current_bytes=effective,scope="system_available_memory"))
    except (OSError,ValueError,IndexError) as error:
        raise RuntimeError("physical available memory cannot be read safely") from error
    if "MemTotal" not in values or "MemAvailable" not in values:
        raise RuntimeError("physical available memory counters are incomplete")
    if not constraints:
        raise RuntimeError("live host memory constraints cannot be read safely")
    chosen = min(constraints,key=lambda c:c["limit_bytes"]-c["effective_current_bytes"])
    return dict(chosen,constraints=constraints)


def host_scope():
    """Private host/cgroup key: CPFS peers on another node never share leases."""
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    groups = []
    for version,leaf,_ in _memory_directories():
        stat = leaf.stat()
        groups.append(dict(version=version,path=str(leaf),device=stat.st_dev,inode=stat.st_ino))
    if not groups:
        groups = [dict(version=0,path="system")]
    try:
        machine = Path("/etc/machine-id").read_text().strip()
    except OSError:
        machine = ""
    identity = dict(node=socket.gethostname(),machine_id=machine,boot_id=boot,
                    cgroups=sorted(groups,key=lambda g:(g["version"],g["path"])))
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return key,identity


def _local_ledger_filesystem(directory):
    """Verify an explicit ephemeral ledger really uses a node-local mount.

    Mount names alone (for example ``/tmp``) do not prove locality. Resolve the
    longest matching mount from the current process's own mount namespace and
    fail closed for distributed, FUSE, unknown or unreadable mount types.
    """
    directory = Path(directory).resolve()
    matches = []
    try:
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            left, separator, right = line.partition(' - ')
            fields, filesystem = left.split(), right.split()
            if not separator or len(fields) < 6 or len(filesystem) < 3:
                continue
            mount = Path(_unescape(fields[4]))
            if directory == mount or mount in directory.parents:
                matches.append((len(mount.parts), int(fields[0]), filesystem[0]))
    except OSError as error:
        raise RuntimeError('ephemeral host ledger mount cannot be verified') from error
    kind = max(matches)[2] if matches else None
    if kind not in ('tmpfs', 'ramfs', 'ext4', 'xfs', 'btrfs', 'f2fs', 'overlay', 'aufs'):
        raise ValueError('ephemeral host ledger requires a verified node-local filesystem')
    return kind


def shared_gene_workspace_bytes(n,m,*,variant_tile_size=128,scratch_bytes=256*2**20):
    """Peak uint8 parts + FP32 host G + a bounded conversion tile + metadata.

    The tile includes FP32 dosage, selected uint8 dosage, a missing-value mask
    and one temporary byte array. Persistent readers/model state is already
    measured by live physical/cgroup pressure; it is not reserved again here.
    Callers subtract only their exact initialized owned-array footprint.
    """
    _positive_bytes(n,"n",allow_zero=False)
    _positive_bytes(m,"m")
    _positive_bytes(variant_tile_size,"variant_tile_size",allow_zero=False)
    _positive_bytes(scratch_bytes,"scratch_bytes")
    return 5*n*m + 7*n*min(m,variant_tile_size) + 512*m + scratch_bytes


def single_output_workspace_bytes(rows,traits,*,bytes_per_row=2048,
                                  packed_bytes_per_row=256,scratch_bytes=64*1024):
    """Bound NEW Single row dictionaries and packed D2H values for one tile.

    ``rows`` is the physical-request upper bound per trait, before filtering.
    This deliberately excludes the genotype already held on the GPU. The
    estimate covers Python rows/scalars, list slots and packed host fields;
    it is a conservative allocation forecast, not a measured RSS footprint.
    Zero output rows require no future host allocation.
    """
    _positive_bytes(rows,"rows")
    _positive_bytes(traits,"traits",allow_zero=False)
    _positive_bytes(bytes_per_row,"bytes_per_row",allow_zero=False)
    _positive_bytes(packed_bytes_per_row,"packed_bytes_per_row")
    _positive_bytes(scratch_bytes,"scratch_bytes")
    return rows*traits*(bytes_per_row+packed_bytes_per_row)+scratch_bytes if rows else 0


class HostOutputReservation:
    """A future-only output reservation; materialization grants no RSS credit."""
    def __init__(self,lease,required_bytes):
        self.lease,self.required_bytes = lease,required_bytes
        self.rows_materialized = False

    def materialized(self):
        """Clear the forecast immediately after the complete tile exists.

        Its initialized rows/packed arrays are now represented by live host
        usage. Waiting or retrying calculation after this point is forbidden;
        write the tile once, release its owners, then leave this scope.
        """
        if not self.rows_materialized:
            if self.lease is not None:
                self.lease.set_future(0,phase="single_output_materialized")
            self.rows_materialized = True


@contextmanager
def host_output_lease_scope(factory,rows,traits,*,chromosome_index,job_index,
                            retry_seconds=5.,maximum_wait_seconds=None,on_wait=None,
                            bytes_per_row=2048,packed_bytes_per_row=256,
                            scratch_bytes=64*1024):
    """Admit one Single tile before output allocation or any writer call.

    Only entry admission is retried. Code inside the scope executes exactly
    once, so a writer or calculation error cannot repeat an already published
    prefix. Each deferral releases its exact lease before waiting. Delays are
    bounded to 60 seconds, and ``on_wait`` may publish progress diagnostics.
    Call ``reservation.materialized()`` as soon as output construction ends.
    All exits release the lease; cleanup cannot replace the original error.
    A missing factory retains compatible behavior without a host gate.
    """
    required = single_output_workspace_bytes(rows,traits,bytes_per_row=bytes_per_row,
        packed_bytes_per_row=packed_bytes_per_row,scratch_bytes=scratch_bytes)
    for name,value in (("chromosome_index",chromosome_index),("job_index",job_index)):
        _positive_bytes(value,name)
    if (isinstance(retry_seconds,bool) or not isinstance(retry_seconds,(int,float))
            or not math.isfinite(retry_seconds) or not 0<retry_seconds<=60):
        raise ValueError("Single output retry delay must be positive, finite and at most 60 seconds")
    if maximum_wait_seconds is not None and (isinstance(maximum_wait_seconds,bool)
            or not isinstance(maximum_wait_seconds,(int,float))
            or not math.isfinite(maximum_wait_seconds) or maximum_wait_seconds<0):
        raise ValueError("Single output maximum wait must be finite and nonnegative or None")
    if factory is not None and not callable(factory):
        raise TypeError("Single host lease factory must be callable or None")
    if on_wait is not None and not callable(on_wait):
        raise TypeError("Single output wait callback must be callable or None")
    lease,started,attempts = None,time.monotonic(),0
    if factory is not None:
        while True:
            lease = factory("individual",chromosome_index,job_index)
            try:
                lease.__enter__()
                lease.set_future(required,phase="single_output_rows_and_transfer")
                break
            except (HostMemoryDeferred,HostMemoryRequestTooLarge) as error:
                try:
                    lease.release()
                except BaseException:
                    pass  # The admission error and its identity-bound retirement win.
                if isinstance(error,HostMemoryRequestTooLarge):
                    error.reason = "single_output_tile_exceeds_host_capacity"
                    # Python 3.10 has no exception notes. Keep the capacity
                    # error intact rather than masking it with AttributeError.
                    if hasattr(error, "add_note"):
                        error.add_note(f"Single output tile: at most {rows} rows per trait, {traits} traits; "
                                       "reduce the output tile or increase the host budget")
                    raise
                attempts += 1
                elapsed = time.monotonic()-started
                if maximum_wait_seconds is not None and elapsed>=maximum_wait_seconds:
                    raise HostMemoryDeferred(required,available_bytes=error.available_bytes,
                        reason="single_output_admission_timeout") from error
                delay = (retry_seconds if maximum_wait_seconds is None else
                         min(retry_seconds,maximum_wait_seconds-elapsed))
                if on_wait is not None:
                    on_wait(dict(event="single_host_output_wait",attempt=attempts,
                        chromosome_index=chromosome_index,job_index=job_index,
                        required_new_bytes=required,rows_per_trait_bound=rows,traits=traits,
                        available_bytes=error.available_bytes,reason=error.reason,
                        elapsed_wait_seconds=elapsed,wait_seconds=delay))
                time.sleep(delay)
            except BaseException:
                try:
                    lease.release()
                except BaseException:
                    pass
                raise
    failed = True
    try:
        yield HostOutputReservation(lease,required)
        failed = False
    finally:
        if lease is not None:
            try:
                lease.release()
            except BaseException:
                if not failed:
                    raise


class HostMemoryGate:
    """Reserve only future host allocations under private host-local leases.

    Persistent, identity-bound registrations cover cached/model phases without
    a lease. Task PSS plus future bytes obey the task limit independently of
    every actual cgroup ancestor and physical available memory. Summed RSS is
    diagnostic only. This proportional task budget is not an isolated cgroup.
    Only increasing future budgets sample pressure; zero or decreasing budgets
    retain identity-bound ledger publication without claiming live headroom.
    """
    def __init__(self,directory,*,host_limit_gib=None,reserve_gib=None,lock_timeout_seconds=5,
                 ledger_directory=None,ledger_durable=True,task_root_identity=None,
                 host_memory_sampler=None):
        """Create one task's host-memory coordinator.

        ``directory`` retains the task's logical ownership. An explicit
        ``ledger_directory`` relocates IPC to node-local storage, with a key
        bound to both that logical directory and this node/boot/cgroup.
        ``ledger_durable=False`` is allowed only on a verified local mount;
        atomic replacement and flock remain mandatory. It affects ephemeral
        budget IPC only, never scientific output or final report durability.

        ``task_root_identity`` includes a controller and all its descendants
        alongside registered peers, deduplicated before one fresh PSS scan.
        ``host_memory_sampler(task_snapshot)`` may transform real node/cgroup
        constraints using that same fresh scan. It must not cache endpoints or
        perform another task scan. The ordinary default calls read_host_memory.
        """
        if type(ledger_durable) is not bool:
            raise ValueError('ledger_durable must be boolean')
        if not ledger_durable and ledger_directory is None:
            raise ValueError('nondurable host IPC requires an explicit local ledger_directory')
        if host_memory_sampler is not None and not callable(host_memory_sampler):
            raise TypeError('host_memory_sampler must be callable or None')
        if task_root_identity is not None and not _valid_identity(task_root_identity):
            raise ValueError('task_root_identity requires exact PID/start/boot identity')
        self.task_root_identity = copy.deepcopy(task_root_identity)
        self.host_memory_sampler = host_memory_sampler
        self.ledger_durable = ledger_durable
        self._ledger_writes = self._ledger_fsyncs = self._ledger_noops = 0
        self._ledger_write_seconds = self._lock_wait_seconds = 0.
        self._lock_acquisitions = self._lock_wait_attempts = 0
        self._loaded_payload = None
        self.automatic_limit = host_limit_gib is None
        self.automatic_reserve = reserve_gib is None
        requested_limit = None if self.automatic_limit else _gib_bytes(host_limit_gib,"host_limit_gib")
        detected_limit = None
        if self.automatic_limit or self.automatic_reserve:
            memory = read_host_memory()
            detected_limit = min(int(c["limit_bytes"]) for c in memory.get("constraints",[memory]))
            if detected_limit<0:
                raise ValueError("detected host memory capacity is negative")
        self.limit_bytes = detected_limit if self.automatic_limit else requested_limit
        # A small allocation receives proportional headroom; larger nodes keep
        # the established 20 GiB reserve. Peers bind the resolved byte policy.
        self.reserve_bytes = (min(20*GiB,min(self.limit_bytes,detected_limit)//10) if self.automatic_reserve
                              else _gib_bytes(reserve_gib,"reserve_gib",allow_zero=True))
        if self.reserve_bytes > self.limit_bytes or (self.limit_bytes and self.reserve_bytes == self.limit_bytes):
            raise ValueError("host memory reserve must be smaller than its limit")
        if not isinstance(lock_timeout_seconds,(int,float)) or not math.isfinite(lock_timeout_seconds) or lock_timeout_seconds<=0:
            raise ValueError("lock timeout must be positive and finite")
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        logical_directory = Path(directory).resolve()
        self.directory = Path(directory if ledger_directory is None else ledger_directory)/"host_memory"
        self.directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.key,self.scope = host_scope()
        self.ledger_filesystem = None
        if ledger_directory is not None:
            self.ledger_filesystem = _local_ledger_filesystem(self.directory)
            task_key = hashlib.sha256(os.fsencode(str(logical_directory))).hexdigest()
            self.key = hashlib.sha256((self.key+'\0'+task_key).encode()).hexdigest()
            self.scope = dict(self.scope, logical_task_directory_sha256=task_key,
                              ipc_ledger_durable=self.ledger_durable)
        self.path = self.directory/(self.key+".private.json")
        self.lock_path = self.directory/(self.key+".lock")
        self._retired_tokens = {}
        self.registration_token = uuid.uuid4().hex
        self.identity = process_identity(os.getpid())
        if self.identity is None:
            raise RuntimeError("host memory task owner identity cannot be verified")
        if self.task_root_identity is not None:
            if process_identity(self.task_root_identity['pid']) != self.task_root_identity:
                raise RuntimeError('task root identity cannot be verified')
            root = dict(identity=self.task_root_identity, descendants=[], closed=False)
            if self.identity not in _task_processes({'root': root}):
                raise RuntimeError('host gate caller is not owned by the supplied task root')
        self.closed = False
        self._pss_samples = self._pss_unverified = 0
        self._pss_host_seconds = 0.
        self._last_task_memory = None
        with self._locked(0):
            state = self._load()
            self._live_leases(state)
            self._live_owners(state)
            root = state.get('task_root_owner')
            if (root is None) != (self.task_root_identity is None) or (root is not None
                    and root['identity'] != self.task_root_identity):
                old_scope_alive = bool(root is not None and _task_processes({'root':root}))
                if state['owners'] or state['leases'] or old_scope_alive:
                    raise ValueError('host memory peers must share one task root identity')
                # A crashed controller can leave its last root binding behind.
                # Rebind only after exact PID/start/boot reaping proves that no
                # registered peer or remembered orphan from that scope lives.
                if self.task_root_identity is None:
                    state.pop('task_root_owner',None)
                else:
                    state['task_root_owner'] = dict(identity=self.task_root_identity,
                                                   descendants=[],closed=False)
            state["owners"][self.registration_token] = dict(identity=self.identity,
                descendants=[],closed=False)
            self._write(state)

    @contextmanager
    def _locked(self,required_bytes):
        fd = os.open(self.lock_path,os.O_CREAT|os.O_RDWR,0o600)
        started = time.monotonic()
        wait_recorded = False
        try:
            while True:
                try:
                    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic()-started >= self.lock_timeout_seconds:
                        raise HostMemoryDeferred(required_bytes,reason="host_ledger_lock_busy")
                    self._lock_wait_attempts += 1
                    time.sleep(.005)
            self._lock_acquisitions += 1
            self._lock_wait_seconds += time.monotonic()-started
            wait_recorded = True
            yield
        finally:
            if not wait_recorded:
                self._lock_wait_seconds += time.monotonic()-started
            fcntl.flock(fd,fcntl.LOCK_UN)
            os.close(fd)

    def _load(self):
        if not self.path.exists():
            self._loaded_payload = None
            return dict(schema_version=2,scope_key=self.key,scope=self.scope,
                policy=dict(limit_bytes=self.limit_bytes,reserve_bytes=self.reserve_bytes),leases={},owners={},
                **({} if self.task_root_identity is None else {'task_root_owner':dict(
                    identity=self.task_root_identity, descendants=[], closed=False)}))
        self._loaded_payload = self.path.read_text()
        state = json.loads(self._loaded_payload)
        if state.get("schema_version") != 2 or state.get("scope_key") != self.key or state.get("scope") != self.scope:
            raise ValueError("private host memory ledger scope binding differs")
        policy = state.get("policy")
        if (not isinstance(policy,dict) or set(policy)!={"limit_bytes","reserve_bytes"}
                or any(type(value) is not int or value<0 for value in policy.values())
                or policy["reserve_bytes"]>policy["limit_bytes"]
                or (policy["limit_bytes"] and policy["reserve_bytes"]==policy["limit_bytes"] )):
            raise ValueError("private host memory ledger has an invalid limit/reserve policy")
        if not isinstance(state.get("leases"),dict):
            raise ValueError("private host memory ledger has invalid leases")
        root = state.get('task_root_owner')
        if root is not None and (not isinstance(root,dict) or not _valid_identity(root.get('identity'))
                or not isinstance(root.get('descendants'),list)
                or any(not _valid_identity(identity) for identity in root['descendants'])
                or root.get('closed') is not False):
            raise ValueError('private host memory task root has invalid ownership')
        for token,record in state["leases"].items():
            if not isinstance(token,str) or not token or not isinstance(record,dict):
                raise ValueError("private host memory ledger has an invalid lease record")
            numbers = [record.get(name) for name in ("future_bytes","peak_future_bytes")]
            if any(type(value) is not int or value<0 for value in numbers):
                raise ValueError("private host memory ledger budgets must be nonnegative integers")
            if numbers[1] < numbers[0]:
                raise ValueError("private host memory ledger peak is below its future budget")
            if not isinstance(record.get("phase"),str) or not record["phase"]:
                raise ValueError("private host memory ledger has an invalid phase")
            identity = record.get("identity")
            if not _valid_identity(identity):
                raise ValueError("private host memory ledger has invalid process identity")
        if "owners" not in state:
            # Upgrade future-only schema2 ledgers without discarding live
            # leases; their materialized owners now receive task accounting.
            state["owners"] = {"legacy:"+token:dict(identity=record["identity"],
                descendants=[],closed=False) for token,record in state["leases"].items()}
        if not isinstance(state["owners"],dict):
            raise ValueError("private host memory ledger has invalid task owners")
        for token,owner in state["owners"].items():
            if (not isinstance(token,str) or not token or not isinstance(owner,dict)
                    or not _valid_identity(owner.get("identity"))
                    or type(owner.get("closed")) is not bool
                    or not isinstance(owner.get("descendants"),list)
                    or any(not _valid_identity(value) for value in owner["descendants"])):
                raise ValueError("private host memory ledger has invalid task owner identity")
        expected_policy = dict(limit_bytes=self.limit_bytes,reserve_bytes=self.reserve_bytes)
        if policy != expected_policy:
            self._live_leases(state)
            self._live_owners(state)
            if state["leases"] or state["owners"]:
                raise ValueError("host memory peers must use the same limit/reserve policy")
            # A later resume may choose another resource budget. Only a ledger
            # with no identity-verified live owners can change its byte policy;
            # concurrent workers must continue to share one exact policy.
            state["policy"] = expected_policy
        return state

    def _write(self,state):
        payload = json.dumps(state,sort_keys=True,allow_nan=False)+'\n'
        if payload == self._loaded_payload:
            self._ledger_noops += 1
            return
        started = time.perf_counter()
        temporary = self.path.with_name(self.path.name+f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        fd = os.open(temporary,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        try:
            with os.fdopen(fd,"w") as stream:
                stream.write(payload)
                stream.flush()
                if self.ledger_durable:
                    os.fsync(stream.fileno())
                    self._ledger_fsyncs += 1
            os.replace(temporary,self.path)
            self._loaded_payload = payload
            self._ledger_writes += 1
        finally:
            temporary.unlink(missing_ok=True)
            self._ledger_write_seconds += time.perf_counter()-started

    def _live_leases(self,state):
        for token,lease in list(state["leases"].items()):
            identity = lease.get("identity")
            if not isinstance(identity,dict) or process_identity(identity.get("pid",-1)) != identity:
                del state["leases"][token]
                continue

    def _live_owners(self,state):
        for token,owner in list(state["owners"].items()):
            owner["descendants"] = [identity for identity in owner["descendants"]
                if process_identity(identity["pid"]) == identity]
            root_live = (not owner["closed"] and
                process_identity(owner["identity"]["pid"]) == owner["identity"])
            if not root_live and not owner["descendants"]:
                del state["owners"][token]

    def close(self):
        """Retire only this gate registration after its owned resources close."""
        if self.closed:
            return
        with self._locked(0):
            state = self._load()
            for token,identity in self._retired_tokens.items():
                if state["leases"].get(token,{}).get("identity") == identity:
                    state["leases"].pop(token,None)
            self._live_leases(state)
            if any(record.get("registration_token") == self.registration_token
                   for record in state["leases"].values()):
                raise RuntimeError("cannot close a host gate with active leases")
            self._live_owners(state)
            _task_processes(state["owners"])
            owner = state["owners"].get(self.registration_token)
            if owner is not None:
                owner["closed"] = True
            self._live_owners(state)
            self._write(state)
        self.closed = True
        self._retired_tokens.clear()

    def lease(self,initial_bytes=0,*,label=None):
        if self.closed:
            raise RuntimeError("host memory task gate is closed")
        return HostMemoryLease(self,_positive_bytes(initial_bytes,"initial_bytes"),label=label)

    def metadata(self):
        """Bounded local sampling costs and the last fresh admission endpoint."""
        return dict(pss_sampling_count=self._pss_samples,pss_sampling_host_seconds=self._pss_host_seconds,
            pss_unverified_count=self._pss_unverified,last_task_memory=copy.deepcopy(self._last_task_memory),
            task_limit_bytes=self.limit_bytes,reserve_bytes=self.reserve_bytes,
            pss_sampling_scope="NEW-byte growth only; zero/shrink/release do not sample",
            task_root_scope_enabled=self.task_root_identity is not None,
            fresh_task_snapshot_shared_with_host_sampler=self.host_memory_sampler is not None,
            ledger_durable=self.ledger_durable,ledger_relocated=self.ledger_filesystem is not None,
            ledger_filesystem=self.ledger_filesystem,ledger_writes=self._ledger_writes,
            ledger_fsyncs=self._ledger_fsyncs,ledger_noop_publications=self._ledger_noops,
            ledger_write_host_seconds=self._ledger_write_seconds,
            ledger_lock_acquisitions=self._lock_acquisitions,
            ledger_lock_wait_seconds=self._lock_wait_seconds,ledger_lock_wait_attempts=self._lock_wait_attempts,
            task_budget_is_cgroup_hard_limit=False,closed=self.closed)

    def _fresh_task_memory(self,state):
        """Scan the union once while all peer future budgets remain locked."""
        owners = dict(state['owners'])
        root = state.get('task_root_owner')
        if self.task_root_identity is not None:
            if (root is None or root['identity'] != self.task_root_identity
                    or process_identity(self.task_root_identity['pid']) != self.task_root_identity):
                raise RuntimeError('complete task root identity cannot be verified')
            # The root record remembers observed children so reparenting cannot
            # erase their ownership. _task_processes deduplicates registered
            # children against the root traversal before any PSS reads occur.
            owners['scope_root'] = root
        elif root is not None:
            raise RuntimeError('task root binding changed during admission')
        result = _task_memory_snapshot(_task_processes(owners))
        if (self.task_root_identity is not None
                and process_identity(self.task_root_identity['pid']) != self.task_root_identity):
            raise RuntimeError('task root identity changed during complete PSS sampling')
        return result

    def lease_factory(self,kind,chromosome_index,job_index):
        return self.lease(label=dict(kind=str(kind),chromosome_index=int(chromosome_index),job_index=int(job_index)))

    def _admit(self,lease,required_bytes,*,phase):
        with self._locked(required_bytes):
            state = self._load()
            # A bounded lock timeout can prevent immediate removal during
            # unwind. Only this gate's own retired, identity-bound tokens may
            # be removed on its next successful transaction.
            for token,identity in self._retired_tokens.items():
                if state["leases"].get(token,{}).get("identity") == identity:
                    state["leases"].pop(token,None)
            self._live_leases(state)
            self._live_owners(state)
            if process_identity(lease.identity["pid"]) != lease.identity:
                raise RuntimeError("host memory lease owner identity changed")
            if self.closed or self.registration_token not in state["owners"]:
                raise RuntimeError("host memory task registration is not active")
            for token,record in state["leases"].items():
                if token != lease.token and record["identity"] == lease.identity:
                    raise RuntimeError("one owner cannot hold nested host memory leases")
            previous = state["leases"].get(lease.token)
            if previous is not None and previous["identity"] != lease.identity:
                raise RuntimeError("host memory lease token belongs to another owner")
            others_future = sum(int(record["future_bytes"])
                for token,record in state["leases"].items() if token != lease.token)
            future = others_future+required_bytes
            shrinking = previous is not None and required_bytes <= previous["future_bytes"]
            # Zero or decreasing budgets cannot admit a NEW allocation. Keep
            # the identity checks, peer ledger, lock and durable publication,
            # but do not scan live cgroups or an attached task's complete PSS
            # tree while buffers are being released. Every later increase
            # takes the fresh-pressure path below, including growth from zero.
            if not required_bytes or shrinking:
                coalesced = (previous is not None
                             and required_bytes == previous['future_bytes']
                             and previous.get('registration_token') == self.registration_token
                             and previous.get('label') == lease.label)
                # The shared record keeps the phase of its last budget or
                # ownership/label change; current phase remains on the lease.
                # Locks, identity checks and reaping above always run. _write
                # still durably publishes any real state change, including
                # peer cleanup; only a phase-only publication is coalesced.
                if not coalesced:
                    state["leases"][lease.token] = dict(identity=lease.identity,
                        future_bytes=required_bytes,peak_future_bytes=max(required_bytes,
                            previous["peak_future_bytes"] if previous else 0),phase=phase,label=lease.label,
                        registration_token=self.registration_token)
                self._write(state)
                self._retired_tokens.clear()
                lease.required_bytes,lease.total_bytes = required_bytes,required_bytes
                lease.phase = phase
                # These are not observed current capacity/pressure values.
                # Explicit nulls prevent diagnostics from mistaking a prior
                # sample or the configured limit for newly measured headroom.
                lease.last_admission = dict(workspace_bytes=required_bytes,total_bytes=required_bytes,
                    capacity_bytes=None,future_bytes=future,live_effective_bytes=None,
                    active_leases=len(state["leases"]),reserve_bytes=self.reserve_bytes,
                    resolved_limit_bytes=self.limit_bytes,automatic_limit=self.automatic_limit,
                    automatic_reserve=self.automatic_reserve,
                    materialized_credit_bytes=lease.materialized_credit_bytes,
                    future_new_allocations=True,shared_rss_admission=False,phase=phase,
                    task_memory=None,task_pss_admission=False,task_budget_is_cgroup_hard_limit=False,
                    pressure_constraints=[],shrinking_or_zero=True,
                    ledger_phase_coalesced=coalesced,
                    pressure_check_performed=False,
                    pressure_check_skipped_reason="zero_or_nonincreasing_future_budget")
                return
            for token,record in state["leases"].items():
                if not any(owner["identity"]==record["identity"] and not owner["closed"]
                           for owner in state["owners"].values()):
                    state["owners"]["legacy:"+token] = dict(identity=record["identity"],
                        descendants=[],closed=False)
            try:
                pss_started = time.perf_counter()
                self._pss_samples += 1
                task = self._fresh_task_memory(state)
            except RuntimeError as error:
                self._pss_unverified += 1
                state["leases"].pop(lease.token,None)
                self._write(state);lease.closed = True
                raise HostMemoryDeferred(required_bytes,total_required_bytes=future,
                    reason="task_pss_coverage_not_verified") from error
            finally:
                self._pss_host_seconds += time.perf_counter()-pss_started
            self._last_task_memory = copy.deepcopy(task)
            # A private resource policy consumes this exact fresh endpoint;
            # there is no second controller-tree scan and no cached admission.
            live = (read_host_memory() if self.host_memory_sampler is None else
                    self.host_memory_sampler(copy.deepcopy(task)))
            if (self.task_root_identity is not None
                    and process_identity(self.task_root_identity['pid']) != self.task_root_identity):
                raise RuntimeError('task root identity changed during host-pressure sampling')
            if not isinstance(live,dict):
                raise RuntimeError('host memory sampler must return live constraints')
            constraints = live.get("constraints",[live])
            if not isinstance(constraints,(list,tuple)) or not constraints:
                raise RuntimeError('live host memory constraints are unavailable')
            capacity = max(0,min(self.limit_bytes,*[int(c["limit_bytes"]) for c in constraints])-self.reserve_bytes)
            # A future allocation larger than the immutable total capacity
            # cannot ever fit. Shared/file-backed RSS is not a job-size bound.
            if required_bytes > capacity:
                state["leases"].pop(lease.token,None)
                self._write(state)
                lease.closed = True
                raise HostMemoryRequestTooLarge(required_bytes,capacity)
            pressures = []
            for constraint in constraints:
                maximum = int(constraint["limit_bytes"])
                # Every shared ancestor is compared with its own actual max,
                # never with this task's smaller proportional-memory budget.
                pressures.append(dict(constraint,capacity_bytes=maximum-self.reserve_bytes,
                    projected_bytes=int(constraint["effective_current_bytes"])+future))
            available = min([capacity-others_future]+[
                c["capacity_bytes"]-c["effective_current_bytes"]-others_future for c in pressures])
            # All increases check freshly observed task PSS, every actual
            # node constraint and every other owner's future budget. No
            # cached endpoint can admit new bytes.
            task_available = self.limit_bytes-self.reserve_bytes-task["pss_bytes"]-others_future
            available = min(available,task_available)
            if (task["pss_bytes"]+future>self.limit_bytes-self.reserve_bytes
                    or any(c["projected_bytes"]>c["capacity_bytes"] for c in pressures)):
                state["leases"].pop(lease.token,None)
                self._write(state)
                lease.closed = True
                raise HostMemoryDeferred(required_bytes,total_required_bytes=future,
                    available_bytes=available,reason="live_usage_and_future_leases")
            state["leases"][lease.token] = dict(identity=lease.identity,
                future_bytes=required_bytes,peak_future_bytes=max(required_bytes,
                    previous["peak_future_bytes"] if previous else 0),phase=phase,label=lease.label,
                registration_token=self.registration_token)
            self._write(state)
            self._retired_tokens.clear()
            lease.required_bytes,lease.total_bytes = required_bytes,required_bytes
            lease.phase = phase
            lease.last_admission = dict(workspace_bytes=required_bytes,total_bytes=required_bytes,
                capacity_bytes=capacity,future_bytes=future,
                live_effective_bytes=int(live["effective_current_bytes"]),
                active_leases=len(state["leases"]),reserve_bytes=self.reserve_bytes,
                resolved_limit_bytes=self.limit_bytes,automatic_limit=self.automatic_limit,
                automatic_reserve=self.automatic_reserve,
                materialized_credit_bytes=lease.materialized_credit_bytes,
                future_new_allocations=True,shared_rss_admission=False,phase=phase,
                task_memory=task,task_pss_admission=True,task_budget_is_cgroup_hard_limit=False,
                pressure_constraints=pressures,shrinking_or_zero=False,
                pressure_check_performed=True,pressure_check_skipped_reason=None)

    def _release(self,lease):
        # A tiny ledger lock may be held briefly by another publisher; no memory
        # budget is awaited here. Failed cleanup remains conservative and is
        # reaped automatically when its identity no longer exists.
        with self._locked(lease.required_bytes):
            state = self._load()
            state["leases"].pop(lease.token,None)
            self._write(state)


class HostMemoryLease:
    def __init__(self,gate,initial_bytes,*,label=None):
        self.gate,self.initial_bytes,self.label = gate,initial_bytes,label
        self.token = uuid.uuid4().hex
        self.identity = process_identity(os.getpid())
        if self.identity is None:
            raise RuntimeError("host memory owner identity cannot be verified")
        self.required_bytes = self.total_bytes = 0
        self.materialized_credit_bytes = 0
        self.phase = "initial"
        self.last_admission = None
        self.closed,self.entered = False,False

    def __enter__(self):
        if self.entered or self.closed:
            raise RuntimeError("host memory lease cannot be reused")
        self.entered = True
        self.set_future(self.initial_bytes,phase="initial")
        return self

    def grow(self,required_bytes):
        """Compatibility alias for replacing this lease's absolute NEW bytes."""
        return self.set_future(required_bytes,phase="workspace")

    def set_future(self,required_bytes,*,phase="workspace"):
        """Replace unmaterialized NEW bytes; shrinking and zeroing never wait.

        This must precede the corresponding allocation. Already initialized
        owned buffers are measured by live cgroup usage, not retained here as
        a duplicate future reservation. Callers reset the budget when a cohort
        or mask ends, including exceptional exits.
        """
        return self._set_future(required_bytes,phase=phase,materialized_credit_bytes=0)

    def _set_future(self,required_bytes,*,phase,materialized_credit_bytes):
        if not self.entered or self.closed:
            raise RuntimeError("host memory lease is not active")
        _positive_bytes(required_bytes,"required_bytes")
        if not isinstance(phase,str) or not phase:
            raise ValueError("host memory phase must be a nonempty string")
        self.materialized_credit_bytes = materialized_credit_bytes
        try:
            self.gate._admit(self,required_bytes,phase=phase)
        except HostMemoryDeferred:
            if not self.closed:
                # Do not wait for a busy publisher again while genotype data
                # is retained. The caller unwinds immediately; a later retry
                # retires this exact token before admitting another workspace.
                self.gate._retired_tokens[self.token] = self.identity
                self.closed = True
            raise
        return 0.

    def guard(self,n,m,phase="prepare",*,variant_tile_size=128,materialized_bytes=0):
        """Reserve the preparation peak less initialized buffers owned here.

        ``materialized_bytes`` is an explicit caller protocol, never an RSS
        delta. It counts only retained, initialized compact/dense allocations
        represented in live memory usage. Uninitialized ``np.empty`` capacity,
        reader caches and shared file mappings grant no materialization credit.
        """
        if phase not in ("prepare","prepare_nonresident"):
            raise ValueError("unknown PheWAS host preparation phase")
        peak = shared_gene_workspace_bytes(int(n),int(m),variant_tile_size=variant_tile_size)
        credit = _positive_bytes(materialized_bytes,"materialized_bytes")
        if credit > peak:
            raise ValueError("materialized host bytes exceed the owned preparation peak")
        return self._set_future(peak-credit,phase=phase,materialized_credit_bytes=credit)

    def release(self):
        if self.entered and not self.closed:
            try:
                self.gate._release(self)
            except HostMemoryDeferred:
                self.gate._retired_tokens[self.token] = self.identity
                raise
            finally:
                self.closed = True

    def __exit__(self,exc_type,exc,tb):
        self.release()
        return False
