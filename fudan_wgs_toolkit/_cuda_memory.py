"""Fresh native allocator counters, loaded only when CUDA is queried.

Host-only resource guards remain usable without Torch. The cached object is
only the native query function; allocator counter values are never cached.
"""
memory_stats_as_nested_dict = None


def allocator_bytes(device=None):
    global memory_stats_as_nested_dict
    if memory_stats_as_nested_dict is None:
        from torch.cuda.memory import memory_stats_as_nested_dict as native_snapshot
        memory_stats_as_nested_dict = native_snapshot
    stats = memory_stats_as_nested_dict(device)
    return (int(stats.get("allocated_bytes", {}).get("all", {}).get("current", 0)),
            int(stats.get("reserved_bytes", {}).get("all", {}).get("current", 0)))
