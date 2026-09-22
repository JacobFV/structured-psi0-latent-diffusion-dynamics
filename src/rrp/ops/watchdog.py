"""Live watchdog: 2 s samples; lowers limits or stops ONLY owned leases on pressure.

`evaluate` is a pure decision function over a telemetry sample (unit tested). `run_loop`
samples real telemetry, heartbeats the broker, recomputes live limits and escalates:
stop admission -> ask owned jobs to checkpoint -> terminate their units after grace.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import telemetry
from .budget import live_memory_limit_bytes, live_cpu_limit

GIB = 1024 ** 3


@dataclass
class WatchdogConfig:
    memory_reserve_bytes: int
    disk_reserve_bytes: int
    startup_memory_bytes: int
    startup_cpu_cores: float
    disk_path: str
    fraction: float = 0.5
    swap_growth_stop_bytes: int = 256 * 1024 ** 2
    swap_growth_shed_bytes: int = 1 * GIB
    psi_full_avg10_shed: float = 25.0
    psi_sustain_samples: int = 3
    project_psi_full_avg10_shed: float = 60.0
    thermal_shed_c: float = 100.0          # hard ceiling (no firmware trip points exposed on GB10)
    thermal_stop_admission_c: float = 97.0
    gpu_thermal_shed_c: float = 95.0
    cpu_freq_throttle_ratio: float = 0.7   # shed if hot AND clocks dropped below this fraction of max
    stable_window_samples: int = 15
    project_disk_limit_bytes: int | None = None


@dataclass
class WatchdogState:
    baseline_swap_free: int | None = None
    psi_high_count: int = 0
    project_psi_high_count: int = 0
    stable_count: int = 0
    last_cpu_usage_usec: int | None = None
    last_t: float | None = None


@dataclass
class Verdict:
    level: str                    # ok | stop_admission | shed | emergency
    reasons: list = field(default_factory=list)
    live_memory_bytes: int | None = None
    live_cpu_cores: float | None = None


def evaluate(sample: dict, cfg: WatchdogConfig, st: WatchdogState) -> Verdict:
    """sample keys: memory_available, swap_free, psi_full_avg10, project_memory,
    project_psi_full_avg10, disk_free, thermal_c, gpu_temp_c, idle_cores, project_cpu_cores,
    telemetry_errors, project_disk_bytes."""
    reasons, level = [], "ok"

    def bump(lv, why):
        nonlocal level
        order = ["ok", "stop_admission", "shed", "emergency"]
        if order.index(lv) > order.index(level):
            level = lv
        reasons.append(why)

    if sample.get("telemetry_errors"):
        bump("stop_admission", f"telemetry_failed:{sample['telemetry_errors']}")
    avail = sample.get("memory_available")
    proj = sample.get("project_memory")
    live_mem = None
    if avail is None or avail < 0:
        bump("stop_admission", "memory_telemetry_unavailable")
    else:
        live_mem = live_memory_limit_bytes(
            startup_limit=cfg.startup_memory_bytes, available_now=avail,
            project_resident=proj or 0, reserve=cfg.memory_reserve_bytes, fraction=cfg.fraction,
            project_attribution_accurate=proj is not None)
        if avail < cfg.memory_reserve_bytes:
            bump("emergency", f"available_memory_below_reserve:{avail}")
        elif proj is not None and proj > live_mem:
            bump("shed", f"project_memory_{proj}_exceeds_live_limit_{live_mem}")
    sf = sample.get("swap_free")
    if sf is not None and sf >= 0:
        if st.baseline_swap_free is None:
            st.baseline_swap_free = sf
        growth = st.baseline_swap_free - sf
        if growth > cfg.swap_growth_shed_bytes:
            bump("shed", f"swap_growth:{growth}")
        elif growth > cfg.swap_growth_stop_bytes:
            bump("stop_admission", f"swap_growth:{growth}")
    psi = sample.get("psi_full_avg10")
    if psi is not None and psi > cfg.psi_full_avg10_shed:
        st.psi_high_count += 1
        if st.psi_high_count >= cfg.psi_sustain_samples:
            bump("shed", f"sustained_system_memory_psi:{psi}")
        else:
            bump("stop_admission", f"system_memory_psi:{psi}")
    else:
        st.psi_high_count = 0
    ppsi = sample.get("project_psi_full_avg10")
    if ppsi is not None and ppsi > cfg.project_psi_full_avg10_shed:
        st.project_psi_high_count += 1
        if st.project_psi_high_count >= cfg.psi_sustain_samples:
            bump("shed", f"sustained_project_memory_psi:{ppsi}")
    else:
        st.project_psi_high_count = 0
    df = sample.get("disk_free")
    if df is None:
        bump("stop_admission", "disk_telemetry_unavailable")
    elif df < cfg.disk_reserve_bytes:
        bump("shed", f"disk_below_reserve:{df}")
    pdb = sample.get("project_disk_bytes")
    if cfg.project_disk_limit_bytes is not None and pdb is not None and pdb > cfg.project_disk_limit_bytes:
        bump("stop_admission", f"project_disk_over_budget:{pdb}")
    t = sample.get("thermal_c")
    fr = sample.get("cpu_freq_ratio")
    if t is not None and t >= cfg.thermal_shed_c:
        bump("shed", f"cpu_thermal:{t}")
    elif t is not None and t >= 90.0 and fr is not None and fr < cfg.cpu_freq_throttle_ratio:
        bump("shed", f"cpu_thermal_throttle:{t}C@{fr:.2f}")
    elif t is not None and t >= cfg.thermal_stop_admission_c:
        bump("stop_admission", f"cpu_hot:{t}")
    g = sample.get("gpu_temp_c")
    if sample.get("gpu_thermal_throttle"):
        bump("shed", "gpu_thermal_slowdown_active")
    elif g is not None and g > cfg.gpu_thermal_shed_c:
        bump("shed", f"gpu_thermal:{g}")
    live_cpu = None
    if sample.get("idle_cores") is not None:
        live_cpu = live_cpu_limit(startup_limit=cfg.startup_cpu_cores, idle_now=sample["idle_cores"],
                                  project_usage=sample.get("project_cpu_cores") or 0.0,
                                  fraction=cfg.fraction)
    st.stable_count = st.stable_count + 1 if level == "ok" else 0
    return Verdict(level, reasons, live_mem, live_cpu)


def collect_sample(cfg: WatchdogConfig, st: WatchdogState, project_slice="rrp.slice",
                   idle_window_s: float = 1.0, project_dir: Path | None = None,
                   gpu: bool = False) -> dict:
    errors = []
    try:
        mi = telemetry.read_meminfo()
    except OSError as e:
        mi, _ = {}, errors.append(str(e))
    psi = telemetry.read_psi(Path("/proc/pressure/memory"))
    cg = telemetry.read_cgroup(telemetry.slice_cgroup_path(project_slice))
    try:
        idle = telemetry.sample_idle_cores(window_s=idle_window_s, sample_s=idle_window_s)["idle_cores_min"]
    except OSError as e:
        idle = None
        errors.append(f"cpu:{e}")
    now = time.monotonic()
    proj_cpu = None
    if cg and cg.get("cpu_usage_usec") is not None:
        if st.last_cpu_usage_usec is not None and st.last_t:
            proj_cpu = (cg["cpu_usage_usec"] - st.last_cpu_usage_usec) / 1e6 / max(now - st.last_t, 1e-3)
        st.last_cpu_usage_usec, st.last_t = cg["cpu_usage_usec"], now
    try:
        disk_free = telemetry.disk_usage(cfg.disk_path)["free_bytes"]
    except OSError:
        disk_free = None
    gtemp, gthrot = None, None
    if gpu:
        g = telemetry.gpu_status()
        gtemp = g.get("temperature_c")
        tr = g.get("throttle_reasons") or ""
        try:
            gthrot = bool(int(tr, 16) & 0x60) if tr.startswith("0x") else None   # HW/SW thermal slowdown bits
        except ValueError:
            gthrot = None
    return dict(
        memory_available=mi.get("MemAvailable"), swap_free=mi.get("SwapFree"),
        psi_full_avg10=(psi or {}).get("full", {}).get("avg10"),
        project_memory=(cg or {}).get("memory_current"),
        project_psi_full_avg10=((cg or {}).get("memory_pressure") or {}).get("full", {}).get("avg10"),
        disk_free=disk_free, thermal_c=telemetry.cpu_thermal_max_c(), gpu_temp_c=gtemp,
        idle_cores=idle, project_cpu_cores=proj_cpu, telemetry_errors=errors,
        cpu_freq_ratio=telemetry.cpu_freq_ratio(), gpu_thermal_throttle=gthrot,
        project_disk_bytes=None,
    )


def run_loop(broker, backend, cfg: WatchdogConfig, *, interval_s: float = 2.0, log_path: Path,
             max_iterations: int | None = None, checkpoint_grace_s: float = 30.0, gpu: bool = False,
             sample_fn=None, emergency_grace_s: float = 5.0, max_log_bytes: int = 20 * 1024 ** 2):
    st = WatchdogState()
    shed_started: dict[str, float] = {}
    i = 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    while max_iterations is None or i < max_iterations:
        i += 1
        t0 = time.monotonic()
        sample = sample_fn(cfg, st) if sample_fn else collect_sample(
            cfg, st, idle_window_s=min(1.0, interval_s / 2), gpu=gpu)
        v = evaluate(sample, cfg, st)
        broker.watchdog_beat(dict(level=v.level, reasons=v.reasons, t=time.time(),
                                  live_memory_bytes=v.live_memory_bytes, live_cpu_cores=v.live_cpu_cores))
        if v.live_memory_bytes is not None and v.live_cpu_cores is not None:
            broker.update_limits(cpu_cores=max(v.live_cpu_cores, 0.05), memory_bytes=v.live_memory_bytes)
        if v.level == "ok":
            if st.stable_count >= cfg.stable_window_samples and broker.totals()["admission_stopped"]:
                broker.resume_admission()
        else:
            broker.stop_admission(";".join(v.reasons))
        if v.level in ("shed", "emergency"):
            leases = broker.leases()
            active = sorted(((k, l) for k, l in leases.items() if l["state"] in ("active", "revoke_requested")),
                            key=lambda kv: -kv[1]["created"])
            victims = active if v.level == "emergency" else active[:1]
            for lid, l in victims:
                broker.revoke(lid, ";".join(v.reasons))
                shed_started.setdefault(lid, time.monotonic())
        # hard fallback: owned units that ignored checkpoint requests past grace
        for lid, t_start in list(shed_started.items()):
            grace = emergency_grace_s if v.level == "emergency" else checkpoint_grace_s + 20
            if time.monotonic() - t_start > grace:
                try:
                    backend.remove_lease(lid)
                    broker.release(lid)
                except Exception:  # noqa: BLE001
                    pass
                shed_started.pop(lid, None)
        if i % 15 == 0 and hasattr(backend, "unit_state"):
            try:
                from .cgroup import job_unit
                broker.gc(is_unit_active=lambda lid: backend.unit_state(job_unit(lid)).get("ActiveState") == "active")
            except Exception:  # noqa: BLE001
                pass
        if log_path.exists() and log_path.stat().st_size > max_log_bytes:
            os.replace(log_path, log_path.with_suffix(".1.jsonl"))
        with open(log_path, "a") as f:
            f.write(json.dumps(dict(t=time.time(), level=v.level, reasons=v.reasons, sample=sample,
                                    live_mem=v.live_memory_bytes, live_cpu=v.live_cpu_cores)) + "\n")
        dt = time.monotonic() - t0
        if dt < interval_s:
            time.sleep(interval_s - dt)
