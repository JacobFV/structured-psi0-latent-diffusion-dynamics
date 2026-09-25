"""Project-level wiring of budget -> broker -> enforcement for a node (host or peer)."""
from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .broker import ResourceBroker, ResourceRequest
from .budget import compute_budget, GIB
from .cgroup import SystemdUserBackend, job_unit, lease_slice
from . import telemetry

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "configs" / "resources.local.json"


def repo_root() -> Path:
    return Path(os.environ.get("RRP_REPO", REPO))


def ops_root() -> Path:
    """Single shared ops state (broker, config, logs, ledger) — one aggregate budget per node even when several
    worktrees/branches run code. Defaults to RRP_OPS_ROOT, else the main checkout, else this repo."""
    env = os.environ.get("RRP_OPS_ROOT")
    if env:
        return Path(env)
    main = Path.home() / "work" / "relational-robot-policy"
    if node_role() == "host" and (main / "configs" / "resources.local.json").exists():
        return main
    return repo_root()


def config_path() -> Path:
    return ops_root() / "configs" / "resources.local.json"


def load_config() -> dict:
    return json.loads(config_path().read_text())


def node_role() -> str:
    return os.environ.get("RRP_NODE", "host")


def state_dir(role: str | None = None) -> Path:
    return ops_root() / "ops" / "broker" / (role or node_role())


def measure_and_budget(role: str, disk_path: Path, window_s: float = 10.0) -> dict:
    mi = telemetry.read_meminfo()
    idle = telemetry.sample_idle_cores(window_s=window_s, sample_s=1.0)
    du = telemetry.disk_usage(disk_path)
    b = compute_budget(role=role, total_ram_gib=mi["MemTotal"] / GIB, available_ram_gib=mi["MemAvailable"] / GIB,
                       free_cpu_cores=idle["idle_cores_min"], free_disk_gib=du["free_bytes"] / GIB,
                       total_disk_gib=du["total_bytes"] / GIB)
    return {"measured_utc": time.time(), "meminfo": {k: mi.get(k) for k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")},
            "idle_cpu": idle, "disk": du, "budget": b.to_dict()}


def make_broker(role: str | None = None, *, require_watchdog: bool = True) -> tuple[ResourceBroker, SystemdUserBackend]:
    role = role or node_role()
    cfg = load_config()[role]
    unrestricted = bool(cfg.get("unrestricted"))
    be = SystemdUserBackend(unrestricted=unrestricted)
    lim = dict(cfg["enforced"])
    if unrestricted:   # registry/scheduler only: nothing is refused for capacity
        lim.update(cpu_cores=10000.0, memory_bytes=10 ** 15, gpu_slots=64, disk_bytes=None)
    br = ResourceBroker(cpu_limit=lim["cpu_cores"], memory_limit_bytes=lim["memory_bytes"], backend=be,
                        state_dir=state_dir(role), lease_expiry_s=cfg.get("lease_expiry_s", 20),
                        gpu_slots=lim.get("gpu_slots", 0), disk_limit_bytes=lim.get("disk_bytes"),
                        require_watchdog=require_watchdog)
    return br, be


SOFTWARE_RENDER_ENV = {
    "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "EGL_PLATFORM": "surfaceless",
    "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
    "LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe",
}


def thread_env(cpu_cores: float) -> dict:
    n = str(max(1, int(math.floor(cpu_cores))))
    return {k: n for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "UV_CONCURRENT_BUILDS",
                           "UV_CONCURRENT_DOWNLOADS", "MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL")}


def run_leased(argv: list[str], *, cpu: float, memory_bytes: int, label: str, gpu: bool = False,
               gpu_memory_bytes: int = 0,
               max_seconds: int = 21600, cwd: Path | None = None, wait: bool = True,
               extra_env: dict | None = None, log_dir: Path | None = None, disk_bytes: int = 0) -> dict:
    br, be = make_broker()
    lease = br.acquire(ResourceRequest(cpu_cores=cpu, memory_bytes=memory_bytes, gpu=gpu, label=label,
                                       node=node_role(), max_seconds=max_seconds,
                                       gpu_memory_bytes=gpu_memory_bytes, disk_bytes=disk_bytes))
    log_dir = log_dir or (ops_root() / "ops" / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{lease.lease_id}_{label}.log"
    env = {**thread_env(cpu), **(extra_env or {}), "RRP_LEASE": lease.lease_id, "RRP_NODE": node_role(),
           "RRP_REPO": str(repo_root()), "RRP_OPS_ROOT": str(ops_root()), "RRP_UNIT": job_unit(lease.lease_id),
           "PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}
    for k in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "XDG_CACHE_HOME",
              "TMPDIR", "PLAYWRIGHT_BROWSERS_PATH", "MUJOCO_GL", "PYOPENGL_PLATFORM", "npm_config_cache"):
        if k in os.environ and k not in env:
            env[k] = os.environ[k]
    host = node_role() == "host"
    if host and gpu and not load_config()["host"].get("gpu_authorized"):
        raise RuntimeError("host GPU work is disabled unless authorized (D-004/D-007/D-027)")
    if not gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env.update(SOFTWARE_RENDER_ENV)
    else:
        env["RRP_GPU_MEMORY_BYTES"] = str(int(gpu_memory_bytes))
    child = [sys.executable, "-m", "rrp.ops.child", "--lease", lease.lease_id, "--log", str(log),
             "--max-seconds", str(max_seconds), "--", *argv]
    env["PYTHONPATH"] = str(repo_root() / "src") + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    unit = be.start_job(lease.lease_id, child, cwd=str(cwd or repo_root()), env=env,
                        runtime_max_s=max_seconds + 120, private_devices=(host and not gpu))
    out = {"lease_id": lease.lease_id, "unit": unit, "log": str(log)}
    if wait:
        out.update(wait_unit(be, unit, log))
        br.gc(is_unit_active=lambda lid: be.unit_state(job_unit(lid)).get("ActiveState") == "active")
    return out


def wait_unit(be: SystemdUserBackend, unit: str, log: Path, poll_s: float = 2.0, echo: bool = True) -> dict:
    pos = 0
    state = {}
    while True:
        state = be.unit_state(unit)
        if log.exists() and echo:
            with open(log, "rb") as f:
                f.seek(pos)
                chunk = f.read()
                pos += len(chunk)
                if chunk:
                    sys.stdout.write(chunk.decode(errors="replace"))
                    sys.stdout.flush()
        if state.get("ActiveState") in ("inactive", "failed", "") and state.get("SubState") != "running":
            break
        time.sleep(poll_s)
    if log.exists() and echo:
        with open(log, "rb") as f:
            f.seek(pos)
            sys.stdout.write(f.read().decode(errors="replace"))
    rc_file = log.with_suffix(".rc")
    rc = int(rc_file.read_text()) if rc_file.exists() else None
    return {"returncode": rc, "unit_result": state.get("Result"), "exec_status": state.get("ExecMainStatus")}


def stop_owned(role: str | None = None) -> list[str]:
    """Stop ONLY project-owned units (rrp-* in the user manager, inside rrp.slice)."""
    be = SystemdUserBackend()
    stopped = []
    for unit in be.list_owned_units():
        if unit.startswith("rrp-") and (unit.endswith(".service") or unit.endswith(".scope")):
            st = be.unit_state(unit)
            if "/rrp.slice/" in st.get("ControlGroup", "") or st.get("ActiveState") in ("inactive", "failed"):
                be.stop_unit(unit)
                stopped.append(unit)
    for unit in be.list_owned_units():
        if unit.startswith("rrp-lease-") and unit.endswith(".slice"):
            be.stop_unit(unit)
            stopped.append(unit)
    return stopped
