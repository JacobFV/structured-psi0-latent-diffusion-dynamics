"""Live watchdog: 2 s samples; lowers limits or stops ONLY owned leases on pressure.

`evaluate` is a pure decision function over a telemetry sample (unit tested). `run_loop`
samples real telemetry, heartbeats the broker, recomputes live limits and escalates:
stop admission -> ask owned jobs to checkpoint -> terminate their units after grace.
"""
from __future__ import annotations

import collections
import math

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from rrp.orchestration import telemetry
from rrp.orchestration.budget import live_memory_limit_bytes, live_cpu_limit

GIB = 1024 ** 3


@dataclass
class WatchdogConfig:
    memory_reserve_bytes: int
    disk_reserve_bytes: int
    startup_memory_bytes: int
    startup_cpu_cores: float
    disk_path: str
    fraction: float = 0.5          # memory fraction of free
    cpu_fraction: float = 0.5
    swap_growth_stop_bytes: int = 256 * 1024 ** 2
    swap_growth_shed_bytes: int = 1 * GIB
    psi_full_avg10_shed: float = 25.0
    psi_sustain_samples: int = 3
    project_psi_full_avg10_shed: float = 60.0
    # D-127 addendum: when the sustained project PSI coincides with per-lease memory.high throttling, the throttled lease(s)
    # are the likely cause (a stalled lease drives the slice-wide PSI). Shed ONLY those culprits first (at most
    # max_culprits, ranked by throttle events over the PSI window); if the project PSI is still above the threshold
    # culprit_grace_s after that, fall back to the default shed policy (newest lease).
    culprit_shed: bool = True
    max_culprits: int = 3
    culprit_grace_s: float = 30.0
    # OPT-IN, default off (D-117 lesson: live cgroup changes of a running lease are not allowed without the lead): before
    # shedding a culprit, raise its lease slice memory.high to its memory.max (never touches memory.max) and wait
    # culprit_grace_s; see research/tracks/robust.md "D-127 addendum" for the safety analysis.
    raise_high_first: bool = False
    thermal_shed_c: float = 100.0          # hard ceiling (no firmware trip points exposed on GB10)
    thermal_stop_admission_c: float = 97.0
    gpu_thermal_shed_c: float = 95.0
    cpu_freq_throttle_ratio: float = 0.7   # shed if hot AND clocks dropped below this fraction of max
    stable_window_samples: int = 15
    project_disk_limit_bytes: int | None = None
    # D-111 follow-up: the peer's artifact store lives in RAM (tmpfs /dev/shm). Its pages (meminfo Shmem) are not
    # reclaimable and not part of any lease, but the static admission cap (startup_memory_bytes, D-106) was sized
    # on an empty store. With this on, the startup term of the live limit becomes startup - Shmem. (The dynamic
    # terms already exclude Shmem: MemAvailable does not count shared-memory pages as available.)
    subtract_shmem: bool = False
    # Hysteresis for the RAM-store term (lead, 2026-09-27 14:35: the first restart with subtract_shmem shed two jobs): when the
    # project exceeds the live limit ONLY because of the Shmem subtraction, stop admission at once but shed only after the
    # excess has persisted together with real memory pressure (PSI full avg10 >= shmem_shed_psi, or MemAvailable below
    # 2x the reserve) for shmem_shed_grace_s (consecutive samples at sample_interval_s).
    shmem_shed_grace_s: float = 30.0
    shmem_shed_psi: float = 10.0
    sample_interval_s: float = 2.0


@dataclass
class WatchdogState:
    baseline_swap_free: int | None = None
    psi_high_count: int = 0
    project_psi_high_count: int = 0
    stable_count: int = 0
    last_cpu_usage_usec: int | None = None
    last_t: float | None = None
    shmem_excess_count: int = 0           # consecutive samples with a RAM-store-caused excess under memory pressure
    lease_high: dict = field(default_factory=dict)   # lease id -> last seen memory.events `high` count (D-117)
    recent_throttle: list = field(default_factory=list)   # per-sample {lease: high delta} over the PSI sustain window
    culprit_phase: str | None = None      # None | "raised" | "shed" (D-127 addendum)
    culprit_since: int = 0                # samples since the current culprit phase started
    culprits: list = field(default_factory=list)


@dataclass
class Verdict:
    level: str                    # ok | stop_admission | shed | emergency
    reasons: list = field(default_factory=list)
    live_memory_bytes: int | None = None
    live_cpu_cores: float | None = None
    victims: list | None = None       # shed: these lease ids only (culprits); None = the default policy (newest lease)
    raise_high: list | None = None    # opt-in: raise memory.high to memory.max for these lease ids (no shed this sample)


def evaluate(sample: dict, cfg: WatchdogConfig, st: WatchdogState) -> Verdict:
    """sample keys: memory_available, swap_free, shmem, psi_full_avg10, project_memory,
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
        startup = cfg.startup_memory_bytes
        shm = sample.get("shmem")
        if cfg.subtract_shmem and shm is not None and shm >= 0:
            startup = max(0, startup - int(shm))
        kw = dict(available_now=avail, project_resident=proj or 0, reserve=cfg.memory_reserve_bytes, fraction=cfg.fraction,
                  project_attribution_accurate=proj is not None)
        live_mem = live_memory_limit_bytes(startup_limit=startup, **kw)
        live_mem_no_store = live_memory_limit_bytes(startup_limit=cfg.startup_memory_bytes, **kw) \
            if startup != cfg.startup_memory_bytes else live_mem
        store_caused = proj is not None and live_mem < proj <= live_mem_no_store
        psi_now = sample.get("psi_full_avg10")
        pressure = (psi_now is not None and psi_now >= cfg.shmem_shed_psi) or avail < 2 * cfg.memory_reserve_bytes
        st.shmem_excess_count = st.shmem_excess_count + 1 if (store_caused and pressure) else 0
        need = max(1, int(math.ceil(cfg.shmem_shed_grace_s / cfg.sample_interval_s)))
        if avail < cfg.memory_reserve_bytes:
            bump("emergency", f"available_memory_below_reserve:{avail}")
        elif store_caused and st.shmem_excess_count < need:
            bump("stop_admission", f"project_memory_{proj}_exceeds_live_limit_{live_mem}_(ram_store_term; "
                                   f"pressure {st.shmem_excess_count}/{need} samples)")
        elif proj is not None and proj > live_mem:
            bump("shed", f"project_memory_{proj}_exceeds_live_limit_{live_mem}" + ("_(ram_store_term, sustained)"
                                                                                  if store_caused else ""))
    sf = sample.get("swap_free")
    if sf is not None and sf >= 0:
        # swap GROWTH over a recent window (~60 s at 2 s sampling); already-swapped pages that
        # never come back must not latch the guard forever (D-024)
        st.swap_hist = (getattr(st, "swap_hist", None) or [])[-30:] + [sf]
        growth = max(st.swap_hist) - sf
        tight = avail is not None and avail >= 0 and avail < cfg.memory_reserve_bytes * 3
        if growth > cfg.swap_growth_shed_bytes and tight:
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
    deltas = {lid: e.get("delta", 0) for lid, e in (sample.get("lease_memory_high") or {}).items() if e.get("delta", 0) > 0}
    st.recent_throttle = (st.recent_throttle + [deltas])[-max(cfg.psi_sustain_samples, 1):]
    victims = raise_high = None
    if ppsi is not None and ppsi > cfg.project_psi_full_avg10_shed:
        st.project_psi_high_count += 1
        if st.project_psi_high_count >= cfg.psi_sustain_samples:
            victims, raise_high, why = _culprit_step(cfg, st)
            if why:
                bump("shed" if victims else "stop_admission", f"sustained_project_memory_psi:{ppsi}:{why}")
            else:
                bump("shed", f"sustained_project_memory_psi:{ppsi}")
    else:
        st.project_psi_high_count = 0
        st.culprit_phase, st.culprit_since, st.culprits = None, 0, []
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
                                  fraction=cfg.cpu_fraction)
    st.stable_count = st.stable_count + 1 if level == "ok" else 0
    v = Verdict(level, reasons, live_mem, live_cpu)
    if level == "shed" and victims is not None:
        only_psi = all(r.startswith("sustained_project_memory_psi") for r in reasons)
        v.victims = victims if only_psi else None          # another shed reason present: default policy
    if raise_high and level != "emergency":
        v.raise_high = raise_high
    return v


def _culprit_step(cfg: WatchdogConfig, st: WatchdogState):
    """Sustained project PSI: returns (victims | None, raise_high | None, reason | None).
    Phase None: rank throttled leases in the PSI window; none -> (None, None, None) = default shed. Otherwise either raise
    their memory.high (opt-in) or shed them. Phases wait culprit_grace_s; if PSI persists: raised -> shed culprits,
    shed -> default policy."""
    need = max(1, int(math.ceil(cfg.culprit_grace_s / cfg.sample_interval_s)))
    if not cfg.culprit_shed:
        return None, None, None
    if st.culprit_phase is None:
        tot = collections.Counter()
        for d in st.recent_throttle:
            tot.update(d)
        if not tot:
            return None, None, None
        st.culprits = [lid for lid, _ in tot.most_common(cfg.max_culprits)]
        st.culprit_since = 0
        if cfg.raise_high_first:
            st.culprit_phase = "raised"
            return None, list(st.culprits), f"raise_memory_high:{','.join(st.culprits)}"
        st.culprit_phase = "shed"
        return list(st.culprits), None, f"throttled_culprits:{','.join(st.culprits)}"
    st.culprit_since += 1
    if st.culprit_since < need:
        return None, None, f"awaiting_{st.culprit_phase}_culprits:{','.join(st.culprits)}:{st.culprit_since}/{need}"
    if st.culprit_phase == "raised":
        st.culprit_phase, st.culprit_since = "shed", 0
        return list(st.culprits), None, f"throttled_culprits_after_raise:{','.join(st.culprits)}"
    return None, None, None                                  # culprits already shed; pressure persists: default policy


def project_memory_fields(cg: dict | None, gpu_bytes: int | None) -> dict:
    """Project memory for the live limit (D-116, lead-approved): NON-RECLAIMABLE usage of rrp.slice = anon + shmem + kernel
    from memory.stat (the RAM store counted once; reclaimable page cache excluded, since MemAvailable already counts it),
    falling back to memory.current when memory.stat is unreadable; plus the project's CUDA/unified GPU bytes (gpu nodes;
    invisible to memcg on GB10). The source used is recorded in the sample."""
    cg = cg or {}
    nr = telemetry.nonreclaimable_bytes(cg.get("memory_stat"))
    cur = cg.get("memory_current")
    if nr is not None:
        base, src = nr, "memory.stat:anon+shmem+kernel"
    elif cur is not None:
        base, src = cur, "memory.current (fallback: memory.stat unreadable)"
    else:
        return dict(project_memory=None, project_memory_source=None, project_memory_current=None, project_gpu_bytes=gpu_bytes)
    return dict(project_memory=base + (gpu_bytes or 0), project_memory_source=src, project_memory_current=cur,
                project_gpu_bytes=gpu_bytes)


def lease_high_fields(st: "WatchdogState", events: dict) -> dict:
    """D-117: per-lease memory.high throttling since the previous sample. events = {lease: parsed memory.events}.
    Returns sample fields: lease_memory_high = {lease: {"high": total, "delta": new events}} for leases with any high events,
    throttled_leases = [leases with delta > 0]. A lease seen for the first time counts its whole total as the delta."""
    now = {lid: int((ev or {}).get("high", 0)) for lid, ev in events.items()}
    out, throttled = {}, []
    for lid, n in now.items():
        d = n - st.lease_high.get(lid, 0)
        if n > 0:
            out[lid] = dict(high=n, delta=max(d, 0))
        if d > 0:
            throttled.append(lid)
    st.lease_high = now                      # leases that ended drop out
    return dict(lease_memory_high=out, throttled_leases=sorted(throttled))


def throttle_warnings(sample: dict, leases: dict | None = None) -> list[str]:
    """Human-readable warnings for leases throttled at memory.high in this sample (label and declared memory if known)."""
    out = []
    for lid in sample.get("throttled_leases") or []:
        e = (sample.get("lease_memory_high") or {}).get(lid, {})
        req = ((leases or {}).get(lid) or {}).get("request", {})
        mem = req.get("memory_bytes")
        out.append(f"lease {lid} ({req.get('label', '?')}) throttled at memory.high: +{e.get('delta')} events "
                   f"(total {e.get('high')})" + (f"; declared {mem / 2**30:.1f}G -> memory.high {0.8 * mem / 2**30:.1f}G; "
                                                  "declare >= 1.3 x measured peak" if mem else ""))
    return out


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
        memory_available=mi.get("MemAvailable"), swap_free=mi.get("SwapFree"), shmem=mi.get("Shmem"),
        psi_full_avg10=(psi or {}).get("full", {}).get("avg10"),
        **project_memory_fields(cg, telemetry.project_gpu_bytes() if gpu else None),
        project_psi_full_avg10=((cg or {}).get("memory_pressure") or {}).get("full", {}).get("avg10"),
        disk_free=disk_free, thermal_c=telemetry.cpu_thermal_max_c(), gpu_temp_c=gtemp,
        idle_cores=idle, project_cpu_cores=proj_cpu, telemetry_errors=errors,
        cpu_freq_ratio=telemetry.cpu_freq_ratio(), gpu_thermal_throttle=gthrot,
        project_disk_bytes=None,
        **lease_high_fields(st, telemetry.lease_memory_events()),
    )


def run_loop(broker, backend, cfg: WatchdogConfig, *, interval_s: float = 2.0, log_path: Path,
             max_iterations: int | None = None, checkpoint_grace_s: float = 30.0, gpu: bool = False,
             sample_fn=None, emergency_grace_s: float = 5.0, max_log_bytes: int = 20 * 1024 ** 2,
             adjust_limits: bool = True):
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
        if adjust_limits and v.live_memory_bytes is not None and v.live_cpu_cores is not None:
            broker.update_limits(cpu_cores=max(v.live_cpu_cores, 0.05), memory_bytes=v.live_memory_bytes)
        if v.level == "ok":
            if st.stable_count >= cfg.stable_window_samples and broker.totals()["admission_stopped"]:
                broker.resume_admission()
        else:
            broker.stop_admission(";".join(v.reasons))
        if v.raise_high:                                          # opt-in (cfg.raise_high_first), default off
            for lid in v.raise_high:
                try:
                    r = backend.raise_memory_high(lid)
                    print(f"[watchdog] raised memory.high of lease {lid} to its memory.max: {r}", flush=True)
                except Exception as e:  # noqa: BLE001 - a failed raise falls through to the culprit shed after the grace
                    print(f"[watchdog] raise memory.high of lease {lid} failed: {e}", flush=True)
        if v.level in ("shed", "emergency"):
            leases = broker.leases()
            active = sorted(((k, l) for k, l in leases.items() if l["state"] in ("active", "revoke_requested")),
                            key=lambda kv: -kv[1]["created"])
            if v.level == "shed" and v.victims is not None:        # D-127 addendum: the throttled culprits only
                victims = [(k, l) for k, l in active if k in set(v.victims)]
            else:
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
                from rrp.orchestration.cgroup import job_unit
                broker.gc(is_unit_active=lambda lid: backend.unit_state(job_unit(lid)).get("ActiveState") == "active")
            except Exception:  # noqa: BLE001
                pass
        if log_path.exists() and log_path.stat().st_size > max_log_bytes:
            os.replace(log_path, log_path.with_suffix(".1.jsonl"))
        warns = []
        if sample.get("throttled_leases"):
            try:
                warns = throttle_warnings(sample, broker.leases())
            except Exception:  # noqa: BLE001 - a warning must never break the loop
                warns = throttle_warnings(sample)
            for w in warns:
                print(f"[watchdog] WARNING {w}", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(dict(t=time.time(), level=v.level, reasons=v.reasons, sample=sample,
                                    live_mem=v.live_memory_bytes, live_cpu=v.live_cpu_cores,
                                    **({"warnings": warns} if warns else {}))) + "\n")
        dt = time.monotonic() - t0
        if dt < interval_s:
            time.sleep(interval_s - dt)
