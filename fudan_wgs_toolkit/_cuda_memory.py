"""Fresh native allocator snapshot; no cache or change to admission policy."""
from torch.cuda.memory import memory_stats_as_nested_dict


def allocator_bytes(device=None):
    stats = memory_stats_as_nested_dict(device)
    return (int(stats.get("allocated_bytes", {}).get("all", {}).get("current", 0)),
            int(stats.get("reserved_bytes", {}).get("all", {}).get("current", 0)))
