"""In-process GPU memory cap for GB10 unified memory (cgroups do not see CUDA allocations).

Called at the start of every GPU workload. The cap is the lease's declared
gpu_memory_bytes; the allocator raises OOM inside our process instead of pushing the
shared machine into reclaim. The system watchdog (MemAvailable) remains the backstop.
"""
from __future__ import annotations

import os


def apply_cap(torch_module=None) -> dict:
    import torch  # noqa: WPS433
    t = torch_module or torch
    declared = int(os.environ.get("RRP_GPU_MEMORY_BYTES", "0") or 0)
    if not t.cuda.is_available():
        return {"cuda": False}
    if declared <= 0:
        raise RuntimeError("GPU job without a declared RRP_GPU_MEMORY_BYTES lease allowance")
    total = t.cuda.get_device_properties(0).total_memory
    frac = min(1.0, declared / total)
    t.cuda.set_per_process_memory_fraction(frac, 0)
    return {"cuda": True, "declared_bytes": declared, "device_total_bytes": total, "fraction": frac}
