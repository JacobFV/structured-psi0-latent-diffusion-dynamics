"""Truthful local telemetry (stdlib only). Unavailable readings are None, never zero."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

GIB = 1024 ** 3
CGROUP_ROOT = Path("/sys/fs/cgroup")


def read_meminfo(text: str | None = None) -> dict[str, int]:
    text = text if text is not None else Path("/proc/meminfo").read_text()
    out: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r"^(\w+):\s+(\d+)(?:\s+kB)?", line)
        if m:
            out[m.group(1)] = int(m.group(2)) * (1024 if "kB" in line else 1)
    return out


def _cpu_times() -> tuple[list[int], list[int]]:
    """Per-cpu (idle+iowait, total) jiffies."""
    idle, total = [], []
    for line in Path("/proc/stat").read_text().splitlines():
        if re.match(r"^cpu\d+ ", line):
            vals = [int(x) for x in line.split()[1:]]
            idle.append(vals[3] + (vals[4] if len(vals) > 4 else 0))
            total.append(sum(vals[:8]))
    return idle, total


def allowed_cpus() -> list[int]:
    return sorted(os.sched_getaffinity(0))


def sample_idle_cores(window_s: float = 10.0, sample_s: float = 1.0) -> dict:
    """Idle cores on allowed CPUs, measured per sample; returns min (conservative)."""
    cpus = allowed_cpus()
    samples = []
    i0, t0 = _cpu_times()
    end = time.monotonic() + window_s
    while time.monotonic() < end:
        time.sleep(sample_s)
        i1, t1 = _cpu_times()
        idle = 0.0
        for c in cpus:
            if c < len(i1):
                dt = t1[c] - t0[c]
                if dt > 0:
                    idle += (i1[c] - i0[c]) / dt
        samples.append(idle)
        i0, t0 = i1, t1
    return {"allowed_cpu_count": len(cpus), "samples": samples,
            "idle_cores_min": min(samples) if samples else None,
            "idle_cores_mean": sum(samples) / len(samples) if samples else None}


def read_psi(path: Path) -> dict | None:
    try:
        text = path.read_text()
    except OSError:
        return None
    out = {}
    for line in text.splitlines():
        parts = line.split()
        kind = parts[0]
        out[kind] = {k: float(v) for k, v in (p.split("=") for p in parts[1:])}
    return out


def disk_usage(path: str | Path) -> dict:
    u = shutil.disk_usage(str(path))
    return {"path": str(path), "total_bytes": u.total, "free_bytes": u.free, "used_bytes": u.used}


def dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def user_service_cgroup() -> Path:
    uid = os.getuid()
    return CGROUP_ROOT / f"user.slice/user-{uid}.slice/user@{uid}.service"


def slice_cgroup_path(slice_name: str) -> Path:
    """systemd maps 'rrp-lease-x.slice' to rrp.slice/rrp-lease.slice/rrp-lease-x.slice."""
    assert slice_name.endswith(".slice")
    stem = slice_name[:-len(".slice")]
    parts = stem.split("-")
    path = user_service_cgroup()
    for i in range(1, len(parts) + 1):
        path = path / ("-".join(parts[:i]) + ".slice")
    return path


def read_cgroup(path: Path) -> dict | None:
    if not path.exists():
        return None
    def rd(name):
        try:
            return (path / name).read_text().strip()
        except OSError:
            return None
    stat = {}
    cs = rd("cpu.stat")
    if cs:
        for line in cs.splitlines():
            k, v = line.split()
            stat[k] = int(v)
    mcur = rd("memory.current")
    return {
        "path": str(path),
        "memory_current": int(mcur) if mcur and mcur.isdigit() else None,
        "memory_max": rd("memory.max"), "memory_high": rd("memory.high"),
        "memory_swap_max": rd("memory.swap.max"), "cpu_max": rd("cpu.max"),
        "pids_max": rd("pids.max"), "pids_current": rd("pids.current"),
        "cpu_usage_usec": stat.get("usage_usec"),
        "memory_pressure": read_psi(path / "memory.pressure"),
        "memory_events": rd("memory.events"),
    }


def gpu_status() -> dict:
    """nvidia-smi diagnostic. On GB10 memory fields report N/A (unified memory)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"status": "unavailable"}
    try:
        r = subprocess.run([exe, "--query-gpu=name,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        apps = subprocess.run([exe, "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"status": "failed", "error": str(e)}
    if r.returncode != 0:
        return {"status": "failed", "error": r.stderr[-400:]}
    row = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
    def num(x):
        try:
            return float(x)
        except ValueError:
            return None  # N/A is unknown, not zero
    return {"status": "reported", "name": row[0], "temperature_c": num(row[1]),
            "utilization_pct": num(row[2]), "throttle_reasons": row[3] if len(row) > 3 else None,
            "compute_apps": [a.strip() for a in apps.stdout.strip().splitlines() if a.strip()]}


def cpu_thermal_max_c() -> float | None:
    temps = []
    for z in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temps.append(int(z.read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            pass
    return max(temps) if temps else None


@dataclass
class HostSnapshot:
    utc: float
    memory_total: int
    memory_available: int
    swap_total: int
    swap_free: int
    system_memory_pressure: dict | None
    disk: dict
    thermal_max_c: float | None
    project_cgroup: dict | None
    errors: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def snapshot(disk_path: str | Path, project_slice: str = "rrp.slice") -> HostSnapshot:
    errors = []
    try:
        mi = read_meminfo()
    except OSError as e:
        mi = {}
        errors.append(f"meminfo:{e}")
    return HostSnapshot(
        utc=time.time(),
        memory_total=mi.get("MemTotal", -1), memory_available=mi.get("MemAvailable", -1),
        swap_total=mi.get("SwapTotal", -1), swap_free=mi.get("SwapFree", -1),
        system_memory_pressure=read_psi(Path("/proc/pressure/memory")),
        disk=disk_usage(disk_path), thermal_max_c=cpu_thermal_max_c(),
        project_cgroup=read_cgroup(slice_cgroup_path(project_slice)), errors=errors)


def cpu_freq_ratio() -> float | None:
    """min over cpus of current/max frequency (None if unavailable)."""
    ratios = []
    for c in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq"):
        try:
            cur = int((c / "scaling_cur_freq").read_text())
            mx = int((c / "cpuinfo_max_freq").read_text())
            ratios.append(cur / mx)
        except (OSError, ValueError, ZeroDivisionError):
            pass
    return min(ratios) if ratios else None
