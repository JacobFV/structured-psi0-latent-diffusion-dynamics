"""Workload side of the orchestrator <-> job contract (W4: from rrp.ops.gpu and rrp.ops.jobs, unchanged).

- `apply_cap`: in-process GPU memory cap for GB10 unified memory (cgroups do not see CUDA allocations).
- `CheckpointSignal`: SIGUSR1/SIGTERM -> "checkpoint and exit" flag for training loops.

apply_cap:

Called at the start of every GPU workload. The cap is the lease's declared
gpu_memory_bytes; the allocator raises OOM inside our process instead of pushing the
shared machine into reclaim. The system watchdog (MemAvailable) remains the backstop.
"""
from __future__ import annotations

import os
import signal


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


class CheckpointSignal:
    """Workload helper: set flag on SIGUSR1/SIGTERM so training loops checkpoint and exit."""

    def __init__(self):
        self.requested = False
        self.reason = None
        signal.signal(signal.SIGUSR1, self._h)
        signal.signal(signal.SIGTERM, self._h)

    def _h(self, signum, frame):
        self.requested = True
        self.reason = signal.Signals(signum).name


def select_device(*, on_cap_error: str = "raise"):
    """THE device choice of a workload (W5 dedup of the `_dev` helpers): cuda when available, with the in-process
    cap applied first, else cpu. on_cap_error="raise" (arm trainers, former latent_train._dev) propagates an
    apply_cap failure; "ignore" (legged, former controllers.bundles._dev) runs uncapped. `training.adapt._device`
    stays separate: it also returns the cap info and disables TF32 matmuls (likelihood ratios)."""
    import torch
    if on_cap_error not in ("raise", "ignore"):
        raise ValueError(on_cap_error)
    if torch.cuda.is_available():
        try:
            apply_cap()
        except Exception:
            if on_cap_error == "raise":
                raise
        return torch.device("cuda")
    return torch.device("cpu")
