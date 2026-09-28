"""/api/live: ONE short ssh read of the peer per refresh (bounded 5 s), cached in viz/data/live.json; host basics.

Only with --live. Without it the exporter never touches the peer: it re-serves the cached peer snapshot and marks it
stale when older than 60 s. Read-only on the peer (cat/glob/nvidia-smi query); nothing is started or changed.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

from .common import Config, envelope, iso, read_json, write_json

# Runs on the peer under its system python3 (stdin). Prints one JSON object. Time-boxed internally to ~3.5 s.
REMOTE = r'''
import glob, json, os, subprocess, time
T0 = time.time(); P = %(root)r; R = P + "/repo"; out = {"t": time.time(), "errors": []}
def rd(p, n=None):
    try:
        with open(p) as fh: return fh.read() if n is None else fh.read(n)
    except Exception as e: return None
try:
    st = json.load(open(R + "/ops/broker/peer/state.json"))
    act = {k: v for k, v in st.get("leases", {}).items() if v.get("state") == "active"}
    cg = "/sys/fs/cgroup/user.slice/user-%%d.slice/user@%%d.service/rrp.slice/rrp-lease.slice" %% (os.getuid(), os.getuid())
    for lid, l in act.items():
        d = cg + "/rrp-lease-" + lid + ".slice"
        m = {}
        for k in ("memory.current", "memory.peak", "memory.high", "memory.max"):
            v = rd(d + "/" + k)
            m[k] = v.strip() if v else None
        ev = rd(d + "/memory.events")
        m["memory.events"] = dict(x.split() for x in ev.splitlines()) if ev else None
        l["cgroup"] = m
    ev = st.get("events", [])
    out["broker"] = {k: st.get(k) for k in ("admission_stopped", "admission_reason", "limits", "startup_limits",
                                              "watchdog_heartbeat", "watchdog_last")}
    out["broker"]["n_leases_total"] = len(st.get("leases", {}))
    out["broker"]["recent_events"] = ev[-40:]
    out["leases"] = act
except Exception as e:
    out["errors"].append("broker: %%r" %% (e,))
try:
    lines = []
    for f in (R + "/ops/watchdog/peer.jsonl",):
        with open(f, "rb") as fh:
            fh.seek(0, 2); sz = fh.tell(); fh.seek(max(0, sz - 400000))
            lines = fh.read().decode(errors="replace").splitlines()[-200:]
    wd = []
    for ln in lines:
        try:
            r = json.loads(ln); s = r.get("sample") or {}
            wd.append({"t": r.get("t"), "level": r.get("level"), "reasons": r.get("reasons"),
                       "memory_available": s.get("memory_available"), "psi_full_avg10": s.get("psi_full_avg10"),
                       "project_memory": s.get("project_memory"), "project_memory_current": s.get("project_memory_current"),
                       "project_gpu_bytes": s.get("project_gpu_bytes"), "disk_free": s.get("disk_free"),
                       "thermal_c": s.get("thermal_c"), "gpu_temp_c": s.get("gpu_temp_c"), "idle_cores": s.get("idle_cores"),
                       "project_cpu_cores": s.get("project_cpu_cores"), "gpu_thermal_throttle": s.get("gpu_thermal_throttle"),
                       "throttled_leases": s.get("throttled_leases"), "lease_memory_high": s.get("lease_memory_high"),
                       "swap_free": s.get("swap_free")})
        except Exception:
            pass
    out["watchdog"] = wd
except Exception as e:
    out["errors"].append("watchdog: %%r" %% (e,))
try:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=1.5).stdout.strip()
    out["nvidia_smi"] = [dict(zip(["name", "util_pct", "temp_c", "mem_used_mib", "mem_total_mib", "power_w"],
                                  [c.strip() for c in ln.split(",")])) for ln in q.splitlines()]
except Exception as e:
    out["errors"].append("nvidia-smi: %%r" %% (e,))
mi = {}
for ln in (rd("/proc/meminfo") or "").splitlines()[:30]:
    k, _, v = ln.partition(":"); mi[k] = v.strip()
out["meminfo"] = {k: mi.get(k) for k in ("MemTotal", "MemAvailable", "SwapFree", "Shmem")}
out["psi_memory"] = rd("/proc/pressure/memory"); out["psi_cpu"] = rd("/proc/pressure/cpu")
out["loadavg"] = rd("/proc/loadavg")
try:
    s = os.statvfs(P); out["shm_free_bytes"] = s.f_bavail * s.f_frsize
    s = os.statvfs(os.path.expanduser("~")); out["home_free_bytes"] = s.f_bavail * s.f_frsize
except Exception as e:
    out["errors"].append("statvfs: %%r" %% (e,))
temps = []
for z in glob.glob("/sys/class/thermal/thermal_zone*/temp")[:12]:
    v = rd(z)
    if v and v.strip().lstrip("-").isdigit(): temps.append(int(v) / 1000.0)
out["cpu_temp_c_max"] = max(temps) if temps else None
led = {}
for pat in (P + "/wt/*/artifacts/runs/*/_dags/*/ledger.json", R + "/artifacts/runs/*/_dags/*/ledger.json"):
    for f in glob.glob(pat)[:80]:
        if time.time() - T0 > 3.0: break
        try:
            if os.path.getsize(f) < 400000: led[f] = {"mtime": os.path.getmtime(f), "doc": json.load(open(f))}
        except Exception: pass
out["ledgers"] = led
small = []
for f in glob.glob(R + "/artifacts/runs/*/*.summary.json") + glob.glob(R + "/artifacts/runs/*/*/*.summary.json") + \
         glob.glob(R + "/artifacts/runs/*/*compare*.json") + glob.glob(R + "/artifacts/runs/*/*/summary.json"):
    if time.time() - T0 > 3.5: break
    try:
        m = os.path.getmtime(f)
        if time.time() - m < 86400 and os.path.getsize(f) < 60000:
            small.append({"path": f[len(R) + 1:], "mtime": m, "doc": json.load(open(f))})
    except Exception: pass
out["recent_summaries"] = small[:150]
out["read_s"] = round(time.time() - T0, 3)
print(json.dumps(out))
'''


def read_peer(cfg: Config) -> dict:
    """One ssh call; returns {"ok": bool, "t": …, "data" | "error"}."""
    t0 = time.time()
    code = REMOTE % {"root": cfg.peer_root}
    try:
        cp = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", cfg.peer, "python3", "-"],
                            input=code, capture_output=True, text=True, timeout=cfg.peer_timeout_s)
        if cp.returncode != 0:
            return {"ok": False, "t": time.time(), "error": f"ssh rc={cp.returncode}: {cp.stderr.strip()[-300:]}",
                    "elapsed_s": round(time.time() - t0, 3)}
        return {"ok": True, "t": time.time(), "data": json.loads(cp.stdout), "elapsed_s": round(time.time() - t0, 3)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "t": time.time(), "error": f"timeout after {cfg.peer_timeout_s} s",
                "elapsed_s": round(time.time() - t0, 3)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "t": time.time(), "error": repr(e), "elapsed_s": round(time.time() - t0, 3)}


def host_basics() -> dict:
    mi = {}
    try:
        for ln in open("/proc/meminfo").read().splitlines()[:30]:
            k, _, v = ln.partition(":")
            mi[k] = v.strip()
    except OSError:
        pass
    try:
        load = open("/proc/loadavg").read().split()[:3]
    except OSError:
        load = None
    try:
        psi = open("/proc/pressure/memory").read().strip()
    except OSError:
        psi = None
    du = shutil.disk_usage(os.path.expanduser("~"))
    return {"loadavg": load, "cpus": os.cpu_count(), "mem_total": mi.get("MemTotal"),
            "mem_available": mi.get("MemAvailable"), "swap_free": mi.get("SwapFree"), "psi_memory": psi,
            "home_free_bytes": du.free, "gpu": "host GPU work is off (D-115)"}


def _kib(s):
    try:
        return int(str(s).split()[0]) * 1024
    except Exception:
        return None


def _int(s):
    try:
        return int(s)
    except Exception:
        return None


def lease_rows(data: dict, lease_ws: dict, now: float) -> list[dict]:
    out = []
    for lid, l in (data.get("leases") or {}).items():
        rq = l.get("request") or {}
        cg = l.get("cgroup") or {}
        ev = cg.get("memory.events") or {}
        ws = lease_ws.get(lid)
        out.append({"id": lid, "label": rq.get("label"), "workstream": ws.get("track") if ws else None,
                    "dag": ws.get("dag") if ws else None, "dag_node": ws.get("node") if ws else None,
                    "unit": l.get("unit"), "declared": {"memory_bytes": rq.get("memory_bytes"),
                                                        "cpu_cores": rq.get("cpu_cores"), "gpu": rq.get("gpu"),
                                                        "gpu_memory_bytes": rq.get("gpu_memory_bytes"),
                                                        "max_seconds": rq.get("max_seconds")},
                    "measured": {"memory_current": _int(cg.get("memory.current")),
                                 "memory_peak": _int(cg.get("memory.peak")),
                                 "memory_high": cg.get("memory.high"), "memory_max": cg.get("memory.max"),
                                 "memory_high_events": _int(ev.get("high")), "oom_events": _int(ev.get("oom")),
                                 "oom_kill_events": _int(ev.get("oom_kill"))},
                    "created": l.get("created"), "age_s": round(now - l["created"], 1) if l.get("created") else None,
                    "heartbeat_at": l.get("heartbeat_at"), "expires_at": l.get("expires_at")})
    out.sort(key=lambda r: r["created"] or 0)
    return out


def peer_state(cfg: Config) -> dict:
    """The last good peer read (fresh with --live, else the cached one) and this refresh's attempt, if any."""
    cache_p = cfg.cache_dir / "peer_read.json"
    prev = read_json(cache_p)
    read = None
    if cfg.live:
        read = read_peer(cfg)
        if read["ok"]:
            prev = read
            write_json(cache_p, read)
    return {"prev": prev, "read": read}


def build_live(cfg: Config, state: dict, lease_ws: dict | None = None) -> dict:
    prev, read = state["prev"], state["read"]
    now = time.time()
    peer_ok = prev is not None and prev.get("ok")
    data = prev.get("data") if peer_ok else {}
    age = round(now - prev["t"], 1) if peer_ok else None
    wd = data.get("watchdog") or []
    last = wd[-1] if wd else {}
    br = data.get("broker") or {}
    node = {"gpu": data.get("nvidia_smi"), "gpu_temp_c": last.get("gpu_temp_c"),
            "cpu_temp_c": last.get("thermal_c") if last else data.get("cpu_temp_c_max"),
            "cpu_temp_c_max_zones": data.get("cpu_temp_c_max"),
            "memory_available": _kib((data.get("meminfo") or {}).get("MemAvailable")),
            "mem_total": _kib((data.get("meminfo") or {}).get("MemTotal")), "psi_memory": data.get("psi_memory"),
            "psi_cpu": data.get("psi_cpu"), "loadavg": data.get("loadavg"),
            "project_memory": last.get("project_memory"), "project_gpu_bytes": last.get("project_gpu_bytes"),
            "disk_free": last.get("disk_free"), "shm_free_bytes": data.get("shm_free_bytes"),
            "home_free_bytes": data.get("home_free_bytes"),
            "admission": {"stopped": br.get("admission_stopped"), "reason": br.get("admission_reason")},
            "limits": br.get("limits"), "watchdog_last": br.get("watchdog_last"),
            "watchdog_heartbeat": br.get("watchdog_heartbeat"),
            "watchdog_heartbeat_age_s": round(prev["t"] - br["watchdog_heartbeat"], 1)
            if peer_ok and br.get("watchdog_heartbeat") else None}
    doc = envelope("live", cfg, [f"{cfg.peer}:{cfg.peer_root}/repo/ops/broker/peer/state.json",
                                 f"{cfg.peer}:{cfg.peer_root}/repo/ops/watchdog/peer.jsonl", "/proc (host)"],
                   peer=cfg.peer, peer_read_at=iso(prev["t"]) if peer_ok else None,
                   peer_read_at_unix=prev["t"] if peer_ok else None, age_s=age,
                   stale=(not peer_ok) or age > cfg.stale_after_s,
                   stale_after_s=cfg.stale_after_s, refreshed_now=bool(read and read["ok"]),
                   last_error=(read or {}).get("error") if read and not read["ok"] else None,
                   read_elapsed_s=(read or {}).get("elapsed_s"), peer_errors=data.get("errors"),
                   node=node, leases=lease_rows(data, lease_ws or {}, prev["t"] if peer_ok else now),
                   n_leases_total=br.get("n_leases_total"), broker_events=br.get("recent_events"),
                   watchdog=wd, host=host_basics(),
                   notes=["Peer values come from one bounded ssh read; 'stale' is true when that read is older than "
                          f"{int(cfg.stale_after_s)} s or missing. Nothing here is estimated."])
    return doc
