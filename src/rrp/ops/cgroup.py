"""Real aggregate enforcement via delegated systemd --user slices (no sudo).

Hierarchy: rrp.slice (parent = whole project allocation) -> rrp-lease-<id>.slice
(per-lease reservation, caps <= parent) -> rrp-job-<id>.service (workload).
systemd maps dashes to nesting, so every lease is a descendant of rrp.slice and the
parent quota bounds the aggregate regardless of child settings (tested).

A `runner` prefix (e.g. ["ssh", "gb10-direct"]) applies the same backend on the peer.
"""
from __future__ import annotations

import math
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

PARENT_SLICE = "rrp.slice"
_SAFE = re.compile(r"^[a-z0-9][a-z0-9_]{0,40}$")


class EnforcementError(RuntimeError):
    code = "enforcement_failed"


def _quota_percent(cores: float) -> str:
    # systemd CPUQuota granularity is 1%; floor so we never round capacity up.
    pct = int(math.floor(cores * 100.0))
    if pct < 1:
        raise EnforcementError(f"cpu allocation {cores} cores is below 1% granularity")
    return f"{pct}%"


def lease_slice(lease_id: str) -> str:
    if not _SAFE.match(lease_id):
        raise EnforcementError(f"unsafe lease id {lease_id!r}")
    return f"rrp-lease-{lease_id}.slice"


def job_unit(lease_id: str) -> str:
    if not _SAFE.match(lease_id):
        raise EnforcementError(f"unsafe lease id {lease_id!r}")
    return f"rrp-job-{lease_id}.service"


@dataclass
class SystemdUserBackend:
    runner: tuple[str, ...] = ()
    dry_run: bool = False

    def _run(self, args: list[str], check: bool = True, timeout: float = 30) -> subprocess.CompletedProcess:
        cmd = list(args)
        if self.runner:
            cmd = [*self.runner, " ".join(shlex.quote(a) for a in args)]
        if self.dry_run:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if check and r.returncode != 0:
            raise EnforcementError(f"{' '.join(args[:4])}... failed: {r.stderr.strip()[-500:]}")
        return r

    def set_slice(self, slice_name: str, *, cpu_cores: float, memory_bytes: int,
                  tasks_max: int = 4096, memory_high_fraction: float = 0.8) -> None:
        if memory_bytes <= 0:
            raise EnforcementError("memory allocation must be positive")
        high = int(memory_bytes * memory_high_fraction)
        self._run(["systemctl", "--user", "set-property", "--runtime", slice_name,
                   f"CPUQuota={_quota_percent(cpu_cores)}", f"MemoryMax={int(memory_bytes)}",
                   f"MemoryHigh={high}", "MemorySwapMax=0", f"TasksMax={int(tasks_max)}"])

    def ensure_parent(self, *, cpu_cores: float, memory_bytes: int, tasks_max: int = 4096) -> None:
        # a slice unit must be loaded before set-property; starting it is harmless
        self._run(["systemctl", "--user", "start", PARENT_SLICE])
        self.set_slice(PARENT_SLICE, cpu_cores=cpu_cores, memory_bytes=memory_bytes, tasks_max=tasks_max)

    def create_lease(self, lease_id: str, *, cpu_cores: float, memory_bytes: int, tasks_max: int = 1024) -> str:
        s = lease_slice(lease_id)
        self._run(["systemctl", "--user", "start", s])
        self.set_slice(s, cpu_cores=cpu_cores, memory_bytes=memory_bytes, tasks_max=tasks_max)
        return s

    def start_job(self, lease_id: str, argv: list[str], *, cwd: str, env: dict[str, str] | None = None,
                  runtime_max_s: int | None = None, nice: int = 10, private_devices: bool = False) -> str:
        unit = job_unit(lease_id)
        args = ["systemd-run", "--user", "--quiet", f"--unit={unit}", f"--slice={lease_slice(lease_id)}",
                f"--working-directory={cwd}", "-p", "KillMode=control-group", "-p", "OOMPolicy=kill",
                "-p", f"Nice={nice}", "-p", "IOSchedulingClass=idle", "-p", "TimeoutStopSec=45"]
        if runtime_max_s:
            args += ["-p", f"RuntimeMaxSec={int(runtime_max_s)}"]
        if private_devices:
            # hard host-GPU exclusion: /dev/nvidia* and /dev/dri are absent inside the unit
            args += ["-p", "PrivateDevices=yes"]
        for k, v in (env or {}).items():
            args.append(f"--setenv={k}={v}")
        args += ["--", *argv]
        self._run(args)
        return unit

    def signal(self, unit: str, sig: str = "SIGTERM") -> None:
        self._run(["systemctl", "--user", "kill", f"--signal={sig}", unit], check=False)

    def stop_unit(self, unit: str) -> None:
        self._run(["systemctl", "--user", "stop", unit], check=False, timeout=90)

    def unit_state(self, unit: str) -> dict:
        r = self._run(["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "SubState",
                       "-p", "MainPID", "-p", "ExecMainStartTimestampMonotonic", "-p", "ControlGroup",
                       "-p", "Result", "-p", "ExecMainStatus"], check=False)
        out = {}
        for line in r.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k] = v
        return out

    def list_owned_units(self) -> list[str]:
        r = self._run(["systemctl", "--user", "list-units", "--all", "--no-legend", "--plain",
                       "rrp-*"], check=False)
        return [ln.split()[0] for ln in r.stdout.splitlines() if ln.strip()]

    def remove_lease(self, lease_id: str) -> None:
        self.stop_unit(job_unit(lease_id))
        self.stop_unit(lease_slice(lease_id))


class FakeEnforcementBackend:
    """Arithmetic/state-machine tests ONLY. Never used for real workloads."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.parent = None
        self.leases: dict[str, dict] = {}

    def ensure_parent(self, *, cpu_cores, memory_bytes, tasks_max=4096):
        self.parent = dict(cpu_cores=cpu_cores, memory_bytes=memory_bytes)
        self.calls.append(("parent", cpu_cores, memory_bytes))

    def create_lease(self, lease_id, *, cpu_cores, memory_bytes, tasks_max=1024):
        self.leases[lease_id] = dict(cpu_cores=cpu_cores, memory_bytes=memory_bytes)
        self.calls.append(("lease", lease_id))
        return lease_slice(lease_id)

    def remove_lease(self, lease_id):
        self.leases.pop(lease_id, None)
        self.calls.append(("remove", lease_id))

    def signal(self, unit, sig="SIGTERM"):
        self.calls.append(("signal", unit, sig))

    def stop_unit(self, unit):
        self.calls.append(("stop", unit))
