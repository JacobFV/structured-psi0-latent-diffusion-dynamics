"""`rrp` command-line interface. Subcommands are registered by modules lazily so that
the ops layer works with only the standard library (bootstrap before the venv exists)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _parse_bytes(s: str) -> int:
    s = s.strip().upper()
    mult = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}
    if s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(s)


# ---------------------------------------------------------------- ops
def cmd_ops_init(a):
    from rrp.ops.runtime import measure_and_budget, config_path, repo_root
    from rrp.ops.cgroup import SystemdUserBackend
    role = a.role
    m = measure_and_budget(role, repo_root(), window_s=a.window)
    b = m["budget"]
    cfgp = config_path()
    cfg = json.loads(cfgp.read_text()) if cfgp.exists() else {"schema_version": "1.0", "policy_id": "shared-host-half-free-v1"}
    enforced = {"cpu_cores": round(b["cpu_cores"] - 0.005, 2), "memory_bytes": b["memory_bytes"],
                "gpu_slots": 2 if role == "host" else 3, "disk_bytes": int(b["new_disk_gib"] * 1024 ** 3)}
    if a.cpu_cap is not None:
        enforced["cpu_cores"] = min(enforced["cpu_cores"], a.cpu_cap)
    if a.memory_cap:
        enforced["memory_bytes"] = min(enforced["memory_bytes"], _parse_bytes(a.memory_cap))
    if role == "host":
        enforced["host_gpu"] = "authorized_by_user_D027_in_process_cap_plus_watchdog"
    cfg[role] = {"measurement": m, "enforced": enforced, "lease_expiry_s": 120, "gpu_authorized": role == "host",
                 "memory_reserve_bytes": int(b["memory_reserve_gib"] * 1024 ** 3),
                 "disk_reserve_bytes": int(b["disk_reserve_gib"] * 1024 ** 3),
                 "startup_new_disk_bytes": int(b["new_disk_gib"] * 1024 ** 3)}
    cfgp.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfgp.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=1))
    os.replace(tmp, cfgp)
    be = SystemdUserBackend()
    be.ensure_parent(cpu_cores=enforced["cpu_cores"], memory_bytes=enforced["memory_bytes"])
    print(json.dumps({"role": role, "enforced": enforced, "budget": b}, indent=1))


def cmd_ops_watchdog(a):
    from rrp.ops.runtime import make_broker, load_config, repo_root, node_role
    from rrp.ops.watchdog import WatchdogConfig, run_loop
    role = node_role()
    cfg = load_config()[role]
    br, be = make_broker(require_watchdog=False)
    if cfg.get("unrestricted"):
        # emergency-only guard (D-026): act only when the machine itself is about to fail
        cfg = dict(cfg, memory_reserve_bytes=3 * 1024 ** 3, disk_reserve_bytes=2 * 1024 ** 3)
    wc = WatchdogConfig(memory_reserve_bytes=cfg["memory_reserve_bytes"], disk_reserve_bytes=cfg["disk_reserve_bytes"],
                        startup_memory_bytes=cfg["enforced"]["memory_bytes"],
                        startup_cpu_cores=cfg["enforced"]["cpu_cores"], disk_path=str(a.disk_path or repo_root()),
                        fraction=0.8 if role == "host" else 1.0, psi_full_avg10_shed=25.0 if role == "host" else 101.0)
    if cfg.get("unrestricted"):
        wc.startup_memory_bytes = 10 ** 15
        wc.startup_cpu_cores = 10000.0
        wc.project_psi_full_avg10_shed = 101.0
        wc.swap_growth_stop_bytes = wc.swap_growth_shed_bytes = 10 ** 15
        wc.thermal_stop_admission_c = 100.0
    run_loop(br, be, wc, interval_s=a.interval, log_path=repo_root() / "ops" / "watchdog" / f"{role}.jsonl",
             max_iterations=a.iterations, gpu=True, adjust_limits=not cfg.get("unrestricted"))


def cmd_ops_start_watchdog(a):
    """Start the watchdog as an owned user service inside rrp.slice (small, 0.1 CPU)."""
    import subprocess
    from rrp.ops.runtime import repo_root, node_role
    unit = f"rrp-watchdog-{node_role()}.service"
    r = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True)
    if r.stdout.strip() == "active":
        print(f"{unit} already active")
        return
    subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True)
    args = ["systemd-run", "--user", "--quiet", f"--unit={unit}", "--slice=rrp-control.slice",
            f"--working-directory={repo_root()}", "-p", "CPUQuota=15%", "-p", "MemoryMax=256M",
            "-p", "Restart=on-failure", "-p", "RestartSec=3", "-p", "Nice=5",
            f"--setenv=RRP_NODE={node_role()}", f"--setenv=RRP_REPO={repo_root()}",
            f"--setenv=PYTHONPATH={repo_root() / 'src'}", f"--setenv=PATH={os.environ.get('PATH', '')}",
            "--", sys.executable, "-m", "rrp.cli", "ops", "watchdog", "--interval", "2"]
    subprocess.run(args, check=True)
    print(f"started {unit}")


def cmd_ops_run(a):
    from rrp.ops.runtime import run_leased
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        raise SystemExit("no command")
    res = run_leased(cmd, cpu=a.cpu, memory_bytes=_parse_bytes(a.mem), label=a.label, gpu=a.gpu,
                     gpu_memory_bytes=_parse_bytes(a.gpu_mem) if a.gpu_mem else 0,
                     max_seconds=a.max_seconds, wait=not a.detach,
                     extra_env=dict(kv.split("=", 1) for kv in (a.env or [])))
    print(json.dumps(res), file=sys.stderr)
    if not a.detach and res.get("returncode") not in (0,):
        raise SystemExit(1)


def cmd_ops_status(a):
    from rrp.ops.runtime import make_broker, node_role
    from rrp.ops.cgroup import SystemdUserBackend
    from rrp.ops import telemetry
    br, be = make_broker(require_watchdog=False)
    t = br.totals()
    leases = {k: v for k, v in br.leases().items() if v["state"] in ("active", "revoke_requested")}
    out = {"node": node_role(), "totals": t, "active_leases": leases,
           "owned_units": be.list_owned_units(),
           "parent_cgroup": telemetry.read_cgroup(telemetry.slice_cgroup_path("rrp.slice"))}
    import json as _j
    st = _j.loads((br.state_dir / "state.json").read_text())
    out["watchdog_age_s"] = time.time() - st.get("watchdog_heartbeat", 0)
    out["watchdog_last"] = st.get("watchdog_last")
    print(json.dumps(out, indent=1, default=str))


def cmd_ops_shrink(a):
    from rrp.ops.runtime import make_broker
    br, _ = make_broker(require_watchdog=False)
    print(json.dumps(br.shrink(a.lease, memory_bytes=_parse_bytes(a.mem) if a.mem else None,
                               gpu_memory_bytes=_parse_bytes(a.gpu_mem) if a.gpu_mem else None, cpu_cores=a.cpu)))


def cmd_ops_stop(a):
    from rrp.ops.runtime import stop_owned
    from rrp.ops.cgroup import SystemdUserBackend, job_unit
    if not a.owned_only:
        raise SystemExit("refusing: pass --owned-only (this tool never stops unrelated processes)")
    if a.lease:
        be = SystemdUserBackend()
        for lid in a.lease:
            be.remove_lease(lid)
        print(json.dumps({"stopped_leases": a.lease}))
        return
    if not a.all_project_jobs:
        raise SystemExit("refusing: several engineers share this project; pass --lease ID (your own jobs) or "
                         "--all-project-jobs for the final shutdown only")
    print(json.dumps({"stopped": stop_owned()}, indent=1))


def cmd_ops_discover(a):
    from rrp.ops.discovery import discover
    res = discover(a.peer)
    txt = json.dumps(res, indent=1)
    if a.write:
        Path(a.write).parent.mkdir(parents=True, exist_ok=True)
        Path(a.write).write_text(txt)
    print(txt)


def cmd_doctor(a):
    from rrp.ops.runtime import measure_and_budget, repo_root
    from rrp.ops import telemetry
    import platform
    m = measure_and_budget(a.role, repo_root(), window_s=a.window)
    m["platform"] = platform.platform()
    m["python"] = sys.version
    m["gpu"] = telemetry.gpu_status()
    m["project_cgroup"] = telemetry.read_cgroup(telemetry.slice_cgroup_path("rrp.slice"))
    txt = json.dumps(m, indent=1, default=str)
    if a.write:
        Path(a.write).parent.mkdir(parents=True, exist_ok=True)
        Path(a.write).write_text(txt)
    print(txt)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rrp", description="relational robot policy research system")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="measure this node (read-only)")
    d.add_argument("--role", default="host", choices=["host", "peer"])
    d.add_argument("--window", type=float, default=10.0)
    d.add_argument("--write")
    d.set_defaults(fn=cmd_doctor)

    ops = sub.add_parser("ops", help="resource broker, enforcement, watchdog").add_subparsers(dest="ops_cmd", required=True)
    o = ops.add_parser("init", help="measure free capacity and enforce the parent quota")
    o.add_argument("--role", default="host", choices=["host", "peer"])
    o.add_argument("--window", type=float, default=10.0)
    o.add_argument("--cpu-cap", type=float)
    o.add_argument("--memory-cap")
    o.set_defaults(fn=cmd_ops_init)
    o = ops.add_parser("watchdog", help="run the watchdog loop in the foreground")
    o.add_argument("--interval", type=float, default=2.0)
    o.add_argument("--iterations", type=int)
    o.add_argument("--disk-path")
    o.set_defaults(fn=cmd_ops_watchdog)
    o = ops.add_parser("start-watchdog", help="start watchdog as an owned user service")
    o.set_defaults(fn=cmd_ops_start_watchdog)
    o = ops.add_parser("run", help="run a command under a broker lease inside rrp.slice")
    o.add_argument("--cpu", type=float, required=True)
    o.add_argument("--mem", required=True)
    o.add_argument("--label", default="job")
    o.add_argument("--gpu", action="store_true")
    o.add_argument("--gpu-mem", help="declared GPU (unified) memory, counted in the aggregate")
    o.add_argument("--max-seconds", type=int, default=3600)
    o.add_argument("--detach", action="store_true")
    o.add_argument("--env", action="append", help="KEY=VALUE passed to the job")
    o.add_argument("cmd", nargs=argparse.REMAINDER)
    o.set_defaults(fn=cmd_ops_run)
    o = ops.add_parser("shrink", help="reduce a live lease reservation")
    o.add_argument("--lease", required=True)
    o.add_argument("--mem")
    o.add_argument("--gpu-mem")
    o.add_argument("--cpu", type=float)
    o.set_defaults(fn=cmd_ops_shrink)
    o = ops.add_parser("status")
    o.set_defaults(fn=cmd_ops_status)
    o = ops.add_parser("stop")
    o.add_argument("--owned-only", action="store_true")
    o.add_argument("--lease", action="append", help="stop only these leases (repeatable)")
    o.add_argument("--all-project-jobs", action="store_true", help="final shutdown of every project job")
    o.set_defaults(fn=cmd_ops_stop)
    o = ops.add_parser("discover", help="discover/verify the SSH peer among configured aliases")
    o.add_argument("--peer")
    o.add_argument("--write")
    o.set_defaults(fn=cmd_ops_discover)

    try:
        from rrp import cli_ext
        cli_ext.register(sub)
    except ImportError as e:  # heavy deps (torch/mujoco) may be absent on the bootstrap python
        if "cli_ext" not in str(e) and os.environ.get("RRP_DEBUG_CLI"):
            print(f"[rrp] extended commands unavailable: {e}", file=sys.stderr)
    return p


def main(argv=None):
    p = build_parser()
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
