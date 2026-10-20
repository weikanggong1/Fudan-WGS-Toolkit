"""Validate the actual adaptive Java/JNI event schema, without sequence data."""
import hashlib
import json
import math
from pathlib import Path


NATIVE_PROVIDER = "head-native-cuda-pairhmm-fp32-wavefront-packed-H2D-initfix-v1b"


def gpu_evidence(path, uuid, budget):
    """Require completed forceGPU contexts, correct device and actual kernels.

    ``initialized`` has native=null; ``gpu_initialized`` precedes the first
    compute. Allocation peaks are maxima across sequential contexts, not sums.
    This checks execution evidence, not long-run stability or ground truth.
    """
    path = Path(path)
    before = path.stat()
    expected_uuid = uuid.removeprefix("GPU-").replace("-", "").lower()
    contexts = {}; finals = []; active = None; digest = hashlib.sha256()
    work_fields = ("cpu_calls", "gpu_calls", "cpu_pairs", "gpu_pairs", "cpu_dp_cells", "gpu_dp_cells")
    integer_fields = ("compute_calls", "input_pairs", "gpu_pairs", "gpu_kernel_launches", "gpu_dp_cells",
                      "cpu_double_fallback_pairs", "invalid_gpu_result_pairs", "h2d_bytes", "d2h_bytes",
                      "owned_peak_allocated_bytes")
    floating_fields = ("gpu_kernel_milliseconds", "host_wall_milliseconds")

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def integer(value):
        return type(value) is int and value >= 0

    with path.open("rb") as source:
        for raw in source:
            digest.update(raw)
            if not raw.strip():
                continue
            event = json.loads(raw)
            require(event.get("provider") == "FNIT_ADAPTIVE_PAIRHMM_CANDIDATE"
                    and event.get("mode") == "forceGPU", "Unexpected adaptive GPU provider or mode")
            key = event.get("context_id"); kind = event.get("event"); native = event.get("native")
            require(integer(key) and key > 0, "Invalid provider context identity")
            require(all(integer(event.get(name)) for name in work_fields), "Invalid Java provider counters")
            require(event.get("original_GKL_child_loader") is False
                    and event["cpu_calls"] == event["cpu_pairs"] == event["cpu_dp_cells"] == 0,
                    "forceGPU unexpectedly selected a CPU batch provider")
            if kind == "initialized":
                require(key not in contexts and active is None and native is None
                        and event.get("gpu_initialized") is False
                        and event["gpu_calls"] == event["gpu_pairs"] == event["gpu_dp_cells"] == 0,
                        "Invalid initialized event or overlapping project contexts")
                contexts[key] = {"state": "initialized", "previous": {}}
                active = key
                continue
            require(key in contexts and active == key and contexts[key]["state"] != "closed",
                    "Missing initialized event, duplicate completion or event after completion")
            state = contexts[key]
            if kind == "gpu_initialized":
                require(state["state"] == "initialized" and isinstance(native, dict)
                        and native.get("compute_calls") == 0 and event["gpu_calls"] == 0,
                        "Invalid or duplicate gpu_initialized event")
                state["state"] = "GPU_active"
            elif kind in ("computed", "completed"):
                require(state["state"] == "GPU_active" and isinstance(native, dict),
                        "Missing actual CUDA initialization or unfinished provider lifecycle")
            else:
                raise RuntimeError("Failed or unsupported GPU provider event")
            require(event.get("gpu_initialized") is True, "GPU context was not active")
            require(native.get("provider") == NATIVE_PROVIDER
                    and native.get("device_uuid_hex") == expected_uuid and native.get("device_ordinal") == 0,
                    "Native provider or authorized GPU identity changed")
            require(native.get("max_device_bytes") == budget, "Native allocation budget differs")
            for name in integer_fields:
                value = native.get(name)
                require(integer(value) and value >= state["previous"].get(name, 0),
                        "Invalid or decreasing native counter: " + name)
            for name in floating_fields:
                value = native.get(name)
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0
                        and value >= state["previous"].get(name, 0),
                        "Nonfinite or decreasing native timing: " + name)
            allocated = native.get("owned_allocated_bytes")
            require(integer(allocated) and allocated <= native["owned_peak_allocated_bytes"] <= budget
                    and native["invalid_gpu_result_pairs"] == 0
                    and native["cpu_double_fallback_pairs"] <= native["gpu_pairs"],
                    "GPU memory or numerical-result gate failed")
            require(native["compute_calls"] == event["gpu_calls"]
                    and native["input_pairs"] == native["gpu_pairs"] == event["gpu_pairs"]
                    and native["gpu_dp_cells"] == event["gpu_dp_cells"], "Java/native work accounting differs")
            for name in ("device_free_bytes_before", "device_free_bytes_after", "device_total_bytes"):
                value = native.get(name)
                require(value is None or integer(value), "Invalid optional device-memory snapshot")
            state["previous"] = native
            if kind in ("computed", "completed"):
                require(native["compute_calls"] > 0 and native["gpu_pairs"] > 0
                        and native["gpu_kernel_launches"] >= native["compute_calls"],
                        "No verified CUDA kernel work")
            if kind == "completed":
                state["state"] = "closed"; active = None; finals.append(native)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "Provider audit changed during validation")
    require(bool(finals) and active is None and all(x["state"] == "closed" for x in contexts.values()),
            "Missing actual GPU work or normally completed context")
    return dict(gpu_pairs=sum(x["gpu_pairs"] for x in finals),
                gpu_kernel_launches=sum(x["gpu_kernel_launches"] for x in finals),
                cpu_double_fallback_pairs=sum(x["cpu_double_fallback_pairs"] for x in finals),
                invalid_gpu_result_pairs=0, completed_contexts=len(finals),
                owned_peak_bytes=max(x["owned_peak_allocated_bytes"] for x in finals),
                actual_gpu_work_verified=True, native_provider=NATIVE_PROVIDER,
                driver_and_other_users_memory_included=False, audit_sha256=digest.hexdigest())
