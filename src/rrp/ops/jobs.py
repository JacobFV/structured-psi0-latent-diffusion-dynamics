"""Owned job execution: leased child with heartbeat, checkpoint-before-stop, identity checks.

`run_leased_child` is executed INSIDE the lease's systemd unit (rrp-job-<id>.service). It
spawns the workload in its own process group, heartbeats the broker, and on a
checkpoint_and_stop decision sends SIGUSR1 (checkpoint request) then SIGTERM then SIGKILL
after bounded grace periods. If the broker/lease disappears, the worker stops at expiry.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .broker import ResourceBroker, LeaseError


def pid_start_ticks(pid: int) -> int | None:
    """Process start time in clock ticks since boot (field 22 of /proc/pid/stat)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    rest = stat.rsplit(")", 1)[1].split()
    return int(rest[19])


def same_process(pid: int, start_ticks: int | None) -> bool:
    """Guards against PID reuse: the pid must exist with the recorded start time."""
    if start_ticks is None:
        return False
    return pid_start_ticks(pid) == start_ticks


def proc_cgroup(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/cgroup").read_text().strip().split("::", 1)[-1]
    except OSError:
        return None


def is_owned(pid: int, start_ticks: int | None, required_cgroup_prefix: str = "rrp") -> bool:
    cg = proc_cgroup(pid)
    return same_process(pid, start_ticks) and cg is not None and "/rrp.slice/" in cg


@dataclass
class ChildResult:
    returncode: int | None
    stopped_by: str | None
    heartbeats: int


def run_leased_child(broker: ResourceBroker, lease_id: str, argv: list[str], *,
                     heartbeat_s: float = 5.0, checkpoint_grace_s: float = 30.0,
                     term_grace_s: float = 15.0, usage_fn=None, cwd: str | None = None,
                     env: dict | None = None) -> ChildResult:
    proc = subprocess.Popen(argv, cwd=cwd, env=env, start_new_session=True)
    broker.attach_process(lease_id, os.environ.get("RRP_UNIT"), proc.pid, pid_start_ticks(proc.pid))
    stopped_by = None
    beats = 0
    deadline_stop = None
    phase = "run"
    while True:
        try:
            rc = proc.wait(timeout=heartbeat_s)
            break
        except subprocess.TimeoutExpired:
            pass
        try:
            d = broker.heartbeat(lease_id, usage_fn() if usage_fn else None)
            beats += 1
            action, reason = d.action, d.reason
        except (LeaseError, OSError, ValueError) as e:  # broker gone/corrupt: stop at expiry
            action, reason = "stop_now", f"broker_unavailable:{e}"
        now = time.monotonic()
        if action == "checkpoint_and_stop" and phase == "run":
            stopped_by = reason
            phase = "checkpoint"
            _kill_group(proc, signal.SIGUSR1)
            deadline_stop = now + checkpoint_grace_s
        elif action == "stop_now" and phase in ("run", "checkpoint"):
            stopped_by = stopped_by or reason
            phase = "term"
            _kill_group(proc, signal.SIGTERM)
            deadline_stop = now + term_grace_s
        elif phase == "checkpoint" and deadline_stop and now > deadline_stop:
            phase = "term"
            _kill_group(proc, signal.SIGTERM)
            deadline_stop = now + term_grace_s
        elif phase == "term" and deadline_stop and now > deadline_stop:
            _kill_group(proc, signal.SIGKILL)
    return ChildResult(rc, stopped_by, beats)


def _kill_group(proc: subprocess.Popen, sig):
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


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
