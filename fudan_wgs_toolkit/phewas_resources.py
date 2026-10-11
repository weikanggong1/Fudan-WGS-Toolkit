"""Host-local, cross-process admission for shared PheWAS preparation.

Only the standard library is used. Ledgers and process/node identities belong
in the private task directory. Memory is admitted, never waited for while a
partial genotype is retained; callers unwind preparation before retrying.
"""
from __future__ import annotations

from contextlib import contextmanager
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
        if maximum <= 0 or maximum >= 2**60:
            return None
        current = int((directory/("memory.usage_in_bytes" if version == 1 else "memory.current")).read_text())
        stats = dict(line.split() for line in (directory/"memory.stat").read_text().splitlines())
        inactive = int(stats.get("total_inactive_file",stats.get("inactive_file",0))) if version == 1 else int(stats.get("inactive_file",0))
        if current < 0 or inactive < 0:
            raise ValueError("negative memory pressure counter")
        return dict(current_bytes=current,limit_bytes=maximum,inactive_file_bytes=inactive,
            effective_current_bytes=max(0,current-inactive),scope=f"cgroup_v{version}")
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
                    constraints.append(constraint)
    try:
        values = {line.split(":",1)[0]:int(line.split()[1])*1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(("MemTotal:","MemAvailable:"))}
        if "MemTotal" in values and "MemAvailable" in values:
            effective = max(0,values["MemTotal"]-values["MemAvailable"])
            constraints.append(dict(current_bytes=effective,limit_bytes=values["MemTotal"],
                inactive_file_bytes=0,effective_current_bytes=effective,scope="system_available_memory"))
    except (OSError,ValueError):
        pass
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


class HostMemoryGate:
    """Reserve only future host allocations under private host-local leases.

    Cgroup usage already contains materialized arrays and counts shared pages
    once. Summed process RSS therefore never supplies a second admission gate.
    Callers explicitly replace the future budget as their owned buffers become
    initialized or are released; an unknown RSS increase grants no credit.
    """
    def __init__(self,directory,*,host_limit_gib=200,reserve_gib=20,lock_timeout_seconds=5):
        self.limit_bytes = _gib_bytes(host_limit_gib,"host_limit_gib")
        self.reserve_bytes = _gib_bytes(reserve_gib,"reserve_gib",allow_zero=True)
        if self.reserve_bytes >= self.limit_bytes:
            raise ValueError("host memory reserve must be smaller than its limit")
        if not isinstance(lock_timeout_seconds,(int,float)) or not math.isfinite(lock_timeout_seconds) or lock_timeout_seconds<=0:
            raise ValueError("lock timeout must be positive and finite")
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.directory = Path(directory)/"host_memory"
        self.directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.key,self.scope = host_scope()
        self.path = self.directory/(self.key+".private.json")
        self.lock_path = self.directory/(self.key+".lock")
        self._retired_tokens = {}

    @contextmanager
    def _locked(self,required_bytes):
        fd = os.open(self.lock_path,os.O_CREAT|os.O_RDWR,0o600)
        started = time.monotonic()
        try:
            while True:
                try:
                    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic()-started >= self.lock_timeout_seconds:
                        raise HostMemoryDeferred(required_bytes,reason="host_ledger_lock_busy")
                    time.sleep(.025)
            yield
        finally:
            fcntl.flock(fd,fcntl.LOCK_UN)
            os.close(fd)

    def _load(self):
        if not self.path.exists():
            return dict(schema_version=2,scope_key=self.key,scope=self.scope,
                policy=dict(limit_bytes=self.limit_bytes,reserve_bytes=self.reserve_bytes),leases={})
        state = json.loads(self.path.read_text())
        if state.get("schema_version") != 2 or state.get("scope_key") != self.key or state.get("scope") != self.scope:
            raise ValueError("private host memory ledger scope binding differs")
        if state.get("policy") != dict(limit_bytes=self.limit_bytes,reserve_bytes=self.reserve_bytes):
            raise ValueError("host memory peers must use the same limit/reserve policy")
        if not isinstance(state.get("leases"),dict):
            raise ValueError("private host memory ledger has invalid leases")
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
            if (not isinstance(identity,dict) or set(identity)!={"pid","start_ticks","boot_id"}
                    or type(identity["pid"]) is not int or identity["pid"]<=0
                    or type(identity["start_ticks"]) is not int or identity["start_ticks"]<0
                    or not isinstance(identity["boot_id"],str) or not identity["boot_id"]):
                raise ValueError("private host memory ledger has invalid process identity")
        return state

    def _write(self,state):
        temporary = self.path.with_name(self.path.name+f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        fd = os.open(temporary,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        try:
            with os.fdopen(fd,"w") as stream:
                json.dump(state,stream,sort_keys=True,allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary,self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _live_leases(self,state):
        for token,lease in list(state["leases"].items()):
            identity = lease.get("identity")
            if not isinstance(identity,dict) or process_identity(identity.get("pid",-1)) != identity:
                del state["leases"][token]
                continue

    def lease(self,initial_bytes=0,*,label=None):
        return HostMemoryLease(self,_positive_bytes(initial_bytes,"initial_bytes"),label=label)

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
            if process_identity(lease.identity["pid"]) != lease.identity:
                raise RuntimeError("host memory lease owner identity changed")
            for token,record in state["leases"].items():
                if token != lease.token and record["identity"] == lease.identity:
                    raise RuntimeError("one owner cannot hold nested host memory leases")
            live = read_host_memory()
            constraints = live.get("constraints",[live])
            capacity = min(self.limit_bytes,*[int(c["limit_bytes"]) for c in constraints])-self.reserve_bytes
            # A future allocation larger than the immutable total capacity
            # cannot ever fit. Shared/file-backed RSS is not a job-size bound.
            if required_bytes > capacity:
                state["leases"].pop(lease.token,None)
                self._write(state)
                lease.closed = True
                raise HostMemoryRequestTooLarge(required_bytes,capacity)
            others_future = sum(int(record["future_bytes"])
                for token,record in state["leases"].items() if token != lease.token)
            future = others_future+required_bytes
            pressures = []
            project_scope_seen = False
            for constraint in constraints:
                maximum = int(constraint["limit_bytes"])
                # The first finite memory membership contains these workers.
                # Wider ancestors/system RAM also contain unrelated cgroups;
                # their current usage must be compared with their own limits,
                # not the project's smaller 200/250 GiB budget.
                if constraint["scope"].startswith("cgroup") and not project_scope_seen:
                    maximum = min(self.limit_bytes,maximum)
                    project_scope_seen = True
                pressures.append(dict(constraint,capacity_bytes=maximum-self.reserve_bytes,
                    projected_bytes=int(constraint["effective_current_bytes"])+future))
            available = min(capacity-others_future,*[
                c["capacity_bytes"]-c["effective_current_bytes"]-others_future for c in pressures])
            previous = state["leases"].get(lease.token)
            # A zero budget allocates nothing. Decreasing an existing admitted
            # future budget is also safe under pressure and must never force a
            # caller to wait while releasing its buffers. An increase checks
            # all live constraints plus every other owner's future budget.
            shrinking = previous is not None and required_bytes <= previous["future_bytes"]
            if (required_bytes and not shrinking and
                    (future > capacity or any(c["projected_bytes"]>c["capacity_bytes"] for c in pressures))):
                state["leases"].pop(lease.token,None)
                self._write(state)
                lease.closed = True
                raise HostMemoryDeferred(required_bytes,total_required_bytes=future,
                    available_bytes=available,reason="live_usage_and_future_leases")
            state["leases"][lease.token] = dict(identity=lease.identity,
                future_bytes=required_bytes,peak_future_bytes=max(required_bytes,
                    previous["peak_future_bytes"] if previous else 0),phase=phase,label=lease.label)
            self._write(state)
            self._retired_tokens.clear()
            lease.required_bytes,lease.total_bytes = required_bytes,required_bytes
            lease.phase = phase
            lease.last_admission = dict(workspace_bytes=required_bytes,total_bytes=required_bytes,
                capacity_bytes=capacity,future_bytes=future,
                live_effective_bytes=int(live["effective_current_bytes"]),
                active_leases=len(state["leases"]),reserve_bytes=self.reserve_bytes,
                materialized_credit_bytes=lease.materialized_credit_bytes,
                future_new_allocations=True,shared_rss_admission=False,phase=phase,
                pressure_constraints=pressures,shrinking_or_zero=shrinking or not required_bytes)

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
