"""Single aggregate resource broker. All project work acquires leases here.

State is a JSON file guarded by an fcntl lock so the CLI, workbench, workers and the
watchdog share one budget. Leases are reservations under the parent allocation; the
sum of active leases can never exceed the current live limit. Lease slices are created
under the enforced parent slice, so even a mis-sized child cannot exceed the parent.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import secrets
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


class CapacityError(RuntimeError):
    code = "capacity_exceeded"


class AdmissionStopped(RuntimeError):
    code = "admission_stopped"


class LeaseError(RuntimeError):
    code = "lease_invalid"


@dataclass
class ResourceRequest:
    cpu_cores: float
    memory_bytes: int
    gpu: bool = False
    label: str = "unlabeled"
    node: str = "host"
    max_seconds: int = 21600
    disk_bytes: int = 0
    # GB10 unified memory: CUDA allocations are NOT charged to memcg (measured, see
    # research/decisions.md D-004). GPU memory is declared, counted in the aggregate
    # memory sum, and enforced in-process by rrp.ops.gpu.apply_cap + the system watchdog.
    gpu_memory_bytes: int = 0

    def validate(self):
        if not (self.cpu_cores > 0) or self.cpu_cores != self.cpu_cores:
            raise ValueError("cpu_cores must be positive")
        if self.memory_bytes <= 0:
            raise ValueError("memory_bytes must be positive")
        if self.max_seconds <= 0 or self.max_seconds > 21600:
            raise ValueError("max_seconds must be in (0, 21600] (6h cap; use resumable segments)")
        if self.disk_bytes < 0:
            raise ValueError("disk_bytes must be nonnegative")
        if self.gpu_memory_bytes < 0 or (self.gpu_memory_bytes and not self.gpu):
            raise ValueError("gpu_memory_bytes requires gpu=True and must be nonnegative")


@dataclass
class Lease:
    lease_id: str
    request: dict
    created: float
    expires_at: float
    heartbeat_at: float
    state: str = "active"          # active | revoke_requested | released | expired
    revoke_reason: str | None = None
    usage: dict = field(default_factory=dict)
    unit: str | None = None
    pid: int | None = None
    pid_start_ticks: int | None = None


@dataclass
class LeaseDecision:
    action: str   # continue | checkpoint_and_stop | stop_now
    reason: str | None = None
    expires_at: float | None = None


def _now() -> float:
    return time.time()


class ResourceBroker:
    def __init__(self, cpu_limit: float, memory_limit_bytes: int, backend: Any,
                 state_dir: str | Path | None = None, *, lease_expiry_s: float = 20.0,
                 gpu_slots: int = 0, disk_limit_bytes: int | None = None,
                 require_watchdog: bool = False, watchdog_max_age_s: float = 10.0,
                 clock=_now):
        self.backend = backend
        self.lease_expiry_s = lease_expiry_s
        self.clock = clock
        self.require_watchdog = require_watchdog
        self.watchdog_max_age_s = watchdog_max_age_s
        self._mem_state: dict | None = None
        self.state_dir = Path(state_dir) if state_dir else None
        if self.state_dir:
            self.state_dir.mkdir(parents=True, exist_ok=True)
        startup = dict(cpu_cores=float(cpu_limit), memory_bytes=int(memory_limit_bytes),
                       gpu_slots=int(gpu_slots),
                       disk_bytes=int(disk_limit_bytes) if disk_limit_bytes is not None else None)
        fresh = False
        with self._locked() as st:
            if not st.get("initialized"):
                st.update(initialized=True, leases={}, admission_stopped=False, admission_reason=None,
                          events=[])
            if st.get("startup_limits") != startup:
                # new admission baseline (ops init); live limits restart from it
                st["startup_limits"] = dict(startup)
                st["limits"] = dict(startup)
                fresh = True
        if fresh:
            backend.ensure_parent(cpu_cores=float(cpu_limit), memory_bytes=int(memory_limit_bytes))

    # ---- state persistence -------------------------------------------------
    @contextlib.contextmanager
    def _locked(self):
        if self.state_dir is None:
            if self._mem_state is None:
                self._mem_state = {}
            yield self._mem_state
            return
        lock = open(self.state_dir / "lock", "a+")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            p = self.state_dir / "state.json"
            st = json.loads(p.read_text()) if p.exists() and p.stat().st_size else {}
            yield st
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(st, indent=1, sort_keys=True))
            os.replace(tmp, p)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

    def _log(self, st, kind, **kw):
        st.setdefault("events", []).append(dict(t=self.clock(), kind=kind, **kw))
        st["events"] = st["events"][-5000:]

    def _expire(self, st):
        now = self.clock()
        for lid, l in list(st["leases"].items()):
            if l["state"] in ("active", "revoke_requested") and l["expires_at"] < now:
                l["state"] = "expired"
                self._log(st, "lease_expired", lease_id=lid)
                try:
                    self.backend.remove_lease(lid)
                except Exception as e:  # noqa: BLE001 - record, do not mask
                    self._log(st, "lease_cleanup_error", lease_id=lid, error=str(e))

    @staticmethod
    def _active(st):
        return {k: v for k, v in st["leases"].items() if v["state"] in ("active", "revoke_requested")}

    def totals(self) -> dict:
        with self._locked() as st:
            self._expire(st)
            act = self._active(st)
            return dict(cpu_cores=sum(l["request"]["cpu_cores"] for l in act.values()),
                        memory_bytes=sum(l["request"]["memory_bytes"] + l["request"].get("gpu_memory_bytes", 0)
                                         for l in act.values()),
                        gpu=sum(1 for l in act.values() if l["request"]["gpu"]),
                        disk_bytes=sum(l["request"].get("disk_bytes", 0) for l in act.values()),
                        leases=len(act), limits=dict(st["limits"]),
                        admission_stopped=st["admission_stopped"])

    # ---- lease API ----------------------------------------------------------
    def acquire(self, request: ResourceRequest) -> Lease:
        request.validate()
        with self._locked() as st:
            self._expire(st)
            if st["admission_stopped"]:
                raise AdmissionStopped(st.get("admission_reason") or "admission stopped")
            if self.require_watchdog:
                hb = st.get("watchdog_heartbeat", 0)
                if self.clock() - hb > self.watchdog_max_age_s:
                    raise AdmissionStopped("watchdog heartbeat missing or stale")
            lim = st["limits"]
            act = self._active(st)
            cpu = sum(l["request"]["cpu_cores"] for l in act.values()) + request.cpu_cores
            mem = sum(l["request"]["memory_bytes"] + l["request"].get("gpu_memory_bytes", 0)
                      for l in act.values()) + request.memory_bytes + request.gpu_memory_bytes
            gpu = sum(1 for l in act.values() if l["request"]["gpu"]) + (1 if request.gpu else 0)
            disk = sum(l["request"].get("disk_bytes", 0) for l in act.values()) + request.disk_bytes
            if cpu > lim["cpu_cores"] + 1e-9:
                raise CapacityError(f"cpu {cpu:.2f} > aggregate limit {lim['cpu_cores']:.2f}")
            if mem > lim["memory_bytes"]:
                raise CapacityError(f"memory {mem} > aggregate limit {lim['memory_bytes']}")
            if gpu > lim["gpu_slots"]:
                raise CapacityError(f"gpu owners {gpu} > slots {lim['gpu_slots']}")
            if lim.get("disk_bytes") is not None and disk > lim["disk_bytes"]:
                raise CapacityError(f"disk {disk} > limit {lim['disk_bytes']}")
            lid = f"{int(self.clock())}_{secrets.token_hex(3)}"
            now = self.clock()
            lease = Lease(lease_id=lid, request=asdict(request), created=now,
                          expires_at=now + self.lease_expiry_s, heartbeat_at=now)
            self.backend.create_lease(lid, cpu_cores=request.cpu_cores, memory_bytes=request.memory_bytes)
            st["leases"][lid] = asdict(lease)
            self._log(st, "lease_acquired", lease_id=lid, label=request.label,
                      cpu=request.cpu_cores, mem=request.memory_bytes, gpu=request.gpu)
            return lease

    def heartbeat(self, lease_id: str, usage: dict | None = None) -> LeaseDecision:
        with self._locked() as st:
            self._expire(st)
            l = st["leases"].get(lease_id)
            if l is None:
                raise LeaseError(f"unknown lease {lease_id}")
            if l["state"] == "expired":
                return LeaseDecision("stop_now", "lease_expired")
            if l["state"] == "released":
                return LeaseDecision("stop_now", "lease_released")
            now = self.clock()
            if now - l["created"] > l["request"]["max_seconds"]:
                l["state"] = "revoke_requested"
                l["revoke_reason"] = "max_seconds_exceeded"
            l["heartbeat_at"] = now
            l["expires_at"] = now + self.lease_expiry_s
            if usage:
                l["usage"] = usage
            if l["state"] == "revoke_requested":
                return LeaseDecision("checkpoint_and_stop", l["revoke_reason"], l["expires_at"])
            return LeaseDecision("continue", None, l["expires_at"])

    def shrink(self, lease_id: str, *, memory_bytes: int | None = None, gpu_memory_bytes: int | None = None,
               cpu_cores: float | None = None) -> dict:
        """Reduce a live lease's reservation (never grow it). The enforced slice caps are lowered too."""
        with self._locked() as st:
            l = st["leases"].get(lease_id)
            if l is None or l["state"] not in ("active", "revoke_requested"):
                raise LeaseError(f"lease {lease_id} not active")
            r = l["request"]
            new = dict(memory_bytes=memory_bytes if memory_bytes is not None else r["memory_bytes"],
                       gpu_memory_bytes=gpu_memory_bytes if gpu_memory_bytes is not None else r.get("gpu_memory_bytes", 0),
                       cpu_cores=cpu_cores if cpu_cores is not None else r["cpu_cores"])
            if new["memory_bytes"] > r["memory_bytes"] or new["gpu_memory_bytes"] > r.get("gpu_memory_bytes", 0) \
                    or new["cpu_cores"] > r["cpu_cores"]:
                raise CapacityError("shrink cannot grow a lease")
            r.update(new)
            self._log(st, "lease_shrunk", lease_id=lease_id, **new)
        if hasattr(self.backend, "set_slice"):
            from .cgroup import lease_slice
            self.backend.set_slice(lease_slice(lease_id), cpu_cores=new["cpu_cores"], memory_bytes=new["memory_bytes"])
        return new

    def attach_process(self, lease_id: str, unit: str | None, pid: int | None, start_ticks: int | None):
        with self._locked() as st:
            l = st["leases"][lease_id]
            l.update(unit=unit, pid=pid, pid_start_ticks=start_ticks)

    def release(self, lease_id: str, cleanup_backend: bool = True) -> None:
        """cleanup_backend=False is used by the in-unit child so it does not stop its own
        unit before recording results; gc() removes the slice after the unit exits."""
        with self._locked() as st:
            l = st["leases"].get(lease_id)
            if l is None:
                raise LeaseError(f"unknown lease {lease_id}")
            if l["state"] in ("active", "revoke_requested"):
                l["state"] = "released"
                self._log(st, "lease_released", lease_id=lease_id)
            if cleanup_backend:
                self.backend.remove_lease(lease_id)
                l["backend_removed"] = True

    def gc(self, is_unit_active=None) -> list[str]:
        """Remove enforcement objects of released/expired leases whose units have exited."""
        removed = []
        with self._locked() as st:
            for lid, l in st["leases"].items():
                if l["state"] in ("released", "expired") and not l.get("backend_removed"):
                    if is_unit_active is not None and is_unit_active(lid):
                        continue
                    self.backend.remove_lease(lid)
                    l["backend_removed"] = True
                    removed.append(lid)
            # bound state size: forget old finished leases (events keep the audit trail)
            done = [k for k, v in st["leases"].items() if v.get("backend_removed")]
            for k in sorted(done, key=lambda k: st["leases"][k]["created"])[:-200]:
                st["leases"].pop(k)
        return removed

    def revoke(self, lease_id: str, reason: str) -> None:
        with self._locked() as st:
            l = st["leases"].get(lease_id)
            if l and l["state"] == "active":
                l["state"] = "revoke_requested"
                l["revoke_reason"] = reason
                self._log(st, "lease_revoke_requested", lease_id=lease_id, reason=reason)

    # ---- admission / pressure ----------------------------------------------
    def stop_admission(self, reason: str) -> None:
        with self._locked() as st:
            if not st["admission_stopped"]:
                self._log(st, "admission_stopped", reason=reason)
            st["admission_stopped"] = True
            st["admission_reason"] = reason

    def resume_admission(self) -> None:
        with self._locked() as st:
            if st["admission_stopped"]:
                self._log(st, "admission_resumed")
            st["admission_stopped"] = False
            st["admission_reason"] = None

    def watchdog_beat(self, info: dict | None = None) -> None:
        with self._locked() as st:
            st["watchdog_heartbeat"] = self.clock()
            if info:
                st["watchdog_last"] = info

    def update_limits(self, *, cpu_cores: float, memory_bytes: int) -> list[str]:
        """Atomically lower/raise the live limit (never above startup). Returns leases
        asked to checkpoint because the aggregate no longer fits (newest first)."""
        revoked = []
        with self._locked() as st:
            start = st["startup_limits"]
            cpu = min(cpu_cores, start["cpu_cores"])
            mem = min(memory_bytes, start["memory_bytes"])
            st["limits"]["cpu_cores"] = cpu
            st["limits"]["memory_bytes"] = mem
            act = sorted(self._active(st).items(), key=lambda kv: -kv[1]["created"])
            total_mem = sum(l["request"]["memory_bytes"] + l["request"].get("gpu_memory_bytes", 0) for _, l in act)
            for lid, l in act:
                if total_mem <= mem:
                    break
                if l["state"] == "active":
                    l["state"] = "revoke_requested"
                    l["revoke_reason"] = "live_limit_reduced"
                    revoked.append(lid)
                    self._log(st, "lease_revoke_requested", lease_id=lid, reason="live_limit_reduced")
                total_mem -= l["request"]["memory_bytes"] + l["request"].get("gpu_memory_bytes", 0)
            if abs(st.get("_last_logged_mem", 0) - mem) > 2 * 1024 ** 3:   # don't flood the audit log
                self._log(st, "limits_updated", cpu=cpu, mem=mem)
                st["_last_logged_mem"] = mem
        # CPU throttling is safe to apply immediately; memory max stays until checkpoint.
        try:
            self.backend.ensure_parent(cpu_cores=max(cpu, 0.01), memory_bytes=start["memory_bytes"])
        except Exception:  # noqa: BLE001
            pass
        return revoked

    def leases(self) -> dict:
        with self._locked() as st:
            self._expire(st)
            return json.loads(json.dumps(st["leases"]))

    def events(self) -> list:
        with self._locked() as st:
            return list(st.get("events", []))
