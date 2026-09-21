"""Real cgroup enforcement with bounded child fixtures (tiny memory/CPU, seconds long).

These run under the project's rrp.slice, never pressure the whole machine, and only
signal processes they created.
"""
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rrp.ops.cgroup import SystemdUserBackend
from rrp.ops.broker import ResourceBroker, ResourceRequest
from rrp.ops.jobs import run_leased_child, pid_start_ticks, same_process, is_owned
from rrp.ops import telemetry
from rrp.ops.watchdog import WatchdogConfig, WatchdogState, evaluate, run_loop

pytestmark = pytest.mark.cgroup
M = 1024 ** 2

ALLOC = "import time\nb=bytearray()\nfor i in range({n}):\n    b+=bytearray(1<<20)\nprint('ALLOCATED', len(b)>>20, flush=True)\ntime.sleep({hold})\n"
SPIN = "import time\nt=time.time()\nwhile time.time()-t<{sec}:\n    pass\n"


def _systemctl(*a):
    return subprocess.run(["systemctl", "--user", *a], capture_output=True, text=True)


def _run_in_slice(slice_name, code, unit, props=(), wait=True, timeout=60):
    args = ["systemd-run", "--user", "--quiet", f"--unit={unit}", f"--slice={slice_name}", "-p", "Nice=19",
            *sum((["-p", p] for p in props), []), "--collect"]
    if wait:
        args += ["--wait", "--pipe"]
    args += ["--", sys.executable, "-c", code]
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


@pytest.fixture
def tslice():
    tag = "t" + secrets.token_hex(3)
    parent = f"rrp-{tag}.slice"
    yield tag, parent
    for u in _systemctl("list-units", "--all", "--no-legend", "--plain", f"rrp-{tag}*").stdout.splitlines():
        _systemctl("stop", u.split()[0])
    _systemctl("stop", parent)


def test_parent_bounds_aggregate_of_children_that_each_ask_for_more(tslice):
    tag, parent = tslice
    be = SystemdUserBackend()
    _systemctl("start", parent)
    be.set_slice(parent, cpu_cores=1.0, memory_bytes=160 * M, memory_high_fraction=1.0)
    for c in ("a", "b"):
        _systemctl("start", f"rrp-{tag}-{c}.slice")
        be.set_slice(f"rrp-{tag}-{c}.slice", cpu_cores=1.0, memory_bytes=1024 * M, memory_high_fraction=1.0)
    # each child alone fits its 1 GiB lease, but 2 x 110 MiB exceeds the 160 MiB parent
    _run_in_slice(f"rrp-{tag}-a.slice", ALLOC.format(n=110, hold=20), f"rrp-{tag}-ja", wait=False)
    time.sleep(3)
    r = _run_in_slice(f"rrp-{tag}-b.slice", ALLOC.format(n=110, hold=0), f"rrp-{tag}-jb", timeout=60)
    ev = telemetry.read_cgroup(telemetry.slice_cgroup_path(parent))["memory_events"]
    ooms = int([l for l in ev.splitlines() if l.startswith("oom_kill")][0].split()[1])
    assert ooms >= 1, (r.stdout, r.stderr, ev)


def test_cpu_quota_is_enforced(tslice):
    tag, parent = tslice
    be = SystemdUserBackend()
    _systemctl("start", parent)
    be.set_slice(parent, cpu_cores=0.3, memory_bytes=256 * M)
    cg = telemetry.slice_cgroup_path(parent)
    for i in range(2):
        _run_in_slice(parent, SPIN.format(sec=6), f"rrp-{tag}-spin{i}", wait=False)
    time.sleep(1.0)
    u0 = telemetry.read_cgroup(cg)["cpu_usage_usec"]
    t0 = time.monotonic()
    time.sleep(3.0)
    u1 = telemetry.read_cgroup(cg)["cpu_usage_usec"]
    cores = (u1 - u0) / 1e6 / (time.monotonic() - t0)
    assert cores < 0.3 * 1.25, cores   # two spinning processes share a 0.3-core quota
    assert cores > 0.1


def test_pids_limit(tslice):
    tag, parent = tslice
    _systemctl("start", parent)
    _systemctl("set-property", "--runtime", parent, "TasksMax=8", "MemoryMax=256M", "CPUQuota=50%")
    code = ("import subprocess,sys\nps=[]\nerr=0\nfor i in range(20):\n    try:\n"
            "        ps.append(subprocess.Popen([sys.executable,'-c','import time;time.sleep(3)']))\n"
            "    except Exception as e:\n        err+=1\nprint('SPAWNED',len(ps),'ERR',err,flush=True)\n"
            "[p.wait() for p in ps]\n")
    r = _run_in_slice(parent, code, f"rrp-{tag}-pids", timeout=60)
    spawned = int(r.stdout.split("SPAWNED")[1].split()[0])
    assert spawned < 20, r.stdout


def test_memory_high_stall_is_detected_and_owned_lease_is_shed(tmp_path, tslice):
    """A child that exceeds MemoryHigh stalls in reclaim (no swap). The watchdog must see the
    sustained project pressure and stop that owned lease."""
    tag, parent = tslice
    be = SystemdUserBackend()
    _systemctl("start", parent)
    be.set_slice(parent, cpu_cores=0.5, memory_bytes=96 * M, memory_high_fraction=0.8)
    unit = f"rrp-{tag}-stall.service"
    _run_in_slice(parent, ALLOC.format(n=300, hold=0), unit[:-8], wait=False)
    cg = telemetry.slice_cgroup_path(parent)

    class LeaseBackend:
        removed = []

        def ensure_parent(self, **kw):
            pass

        def create_lease(self, lid, **kw):
            return parent

        def remove_lease(self, lid):
            self.removed.append(lid)
            _systemctl("stop", unit)

    lb = LeaseBackend()
    br = ResourceBroker(cpu_limit=1, memory_limit_bytes=96 * M, backend=lb, state_dir=tmp_path)
    lease = br.acquire(ResourceRequest(cpu_cores=0.5, memory_bytes=96 * M, label="stall"))
    cfg = WatchdogConfig(memory_reserve_bytes=0, disk_reserve_bytes=0, startup_memory_bytes=96 * M,
                         startup_cpu_cores=1, disk_path="/", project_psi_full_avg10_shed=30.0)

    def sample(c, s):
        info = telemetry.read_cgroup(cg) or {}
        psi = (info.get("memory_pressure") or {}).get("full", {}).get("avg10")
        return dict(memory_available=10 ** 12, swap_free=None, psi_full_avg10=None,
                    project_memory=info.get("memory_current"), project_psi_full_avg10=psi,
                    disk_free=10 ** 12, thermal_c=None, gpu_temp_c=None, idle_cores=None,
                    project_cpu_cores=None, telemetry_errors=[])

    t0 = time.monotonic()
    while time.monotonic() - t0 < 45 and not lb.removed:
        run_loop(br, lb, cfg, interval_s=1.0, log_path=tmp_path / "wd.jsonl", max_iterations=1,
                 checkpoint_grace_s=0.0, sample_fn=sample)
        # emulate grace expiry in a single-iteration harness
        l = br.leases()[lease.lease_id]
        if l["state"] == "revoke_requested":
            lb.remove_lease(lease.lease_id)
    assert lb.removed, "watchdog never shed the stalled lease"
    time.sleep(1)
    assert _systemctl("is-active", unit).stdout.strip() != "active"


def test_worker_stops_when_lease_is_released_or_expires(tmp_path):
    from tests.support import fake_enforcement_backend
    br = ResourceBroker(cpu_limit=1, memory_limit_bytes=10 ** 9, backend=fake_enforcement_backend(),
                        state_dir=tmp_path, lease_expiry_s=2)
    lease = br.acquire(ResourceRequest(cpu_cores=0.5, memory_bytes=10 ** 8))
    import threading
    threading.Timer(1.5, lambda: br.revoke(lease.lease_id, "test_pressure")).start()
    t0 = time.monotonic()
    res = run_leased_child(br, lease.lease_id, [sys.executable, "-c", "import time; time.sleep(60)"],
                           heartbeat_s=0.5, checkpoint_grace_s=1.0, term_grace_s=1.0)
    assert time.monotonic() - t0 < 15
    assert res.stopped_by == "test_pressure"
    assert res.returncode != 0


def test_checkpoint_signal_reaches_worker_before_termination(tmp_path):
    from tests.support import fake_enforcement_backend
    br = ResourceBroker(cpu_limit=1, memory_limit_bytes=10 ** 9, backend=fake_enforcement_backend(),
                        state_dir=tmp_path)
    lease = br.acquire(ResourceRequest(cpu_cores=0.5, memory_bytes=10 ** 8))
    marker = tmp_path / "ckpt"
    code = (f"import signal,time,sys\n"
            f"def h(s,f):\n    open({str(marker)!r},'w').write('saved'); sys.exit(0)\n"
            f"signal.signal(signal.SIGUSR1,h)\ntime.sleep(60)\n")
    import threading
    threading.Timer(1.0, lambda: br.revoke(lease.lease_id, "pressure")).start()
    res = run_leased_child(br, lease.lease_id, [sys.executable, "-c", code], heartbeat_s=0.5,
                           checkpoint_grace_s=10)
    assert marker.read_text() == "saved"
    assert res.returncode == 0


def test_pid_reuse_guard():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    ticks = pid_start_ticks(p.pid)
    assert same_process(p.pid, ticks)
    assert not same_process(p.pid, ticks + 1)
    assert not is_owned(p.pid, ticks)  # not inside rrp.slice => not owned
    p.kill()
    p.wait()
    assert not same_process(p.pid, ticks)


def test_owned_only_shutdown_does_not_touch_foreign_processes(tslice):
    tag, parent = tslice
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    _systemctl("start", parent)
    _systemctl("set-property", "--runtime", parent, "MemoryMax=64M", "CPUQuota=10%")
    _run_in_slice(parent, "import time; time.sleep(30)", f"rrp-{tag}-owned", wait=False)
    time.sleep(1)
    assert _systemctl("is-active", f"rrp-{tag}-owned.service").stdout.strip() == "active"
    from rrp.ops.cgroup import SystemdUserBackend as B
    b = B()
    for u in b.list_owned_units():
        if u.startswith(f"rrp-{tag}"):
            b.stop_unit(u)
    assert _systemctl("is-active", f"rrp-{tag}-owned.service").stdout.strip() != "active"
    assert foreign.poll() is None
    foreign.kill()
    foreign.wait()
