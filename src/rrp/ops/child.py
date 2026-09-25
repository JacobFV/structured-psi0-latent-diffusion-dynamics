"""Runs inside rrp-job-<lease>.service: executes the workload under lease heartbeats."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import json
import time

from .jobs import run_leased_child
from .runtime import make_broker, ops_root, node_role
from . import telemetry


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lease", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--max-seconds", type=int, default=21600)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    log = Path(a.log)
    br, _ = make_broker(require_watchdog=False)
    cg = telemetry.slice_cgroup_path(f"rrp-lease-{a.lease}.slice")
    t0 = time.time()
    with open(log, "ab", buffering=0) as f:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        res = run_leased_child(br, a.lease, cmd)
    info = telemetry.read_cgroup(cg) or {}
    lease = br.leases().get(a.lease, {})
    req = lease.get("request", {})
    rec = dict(t_start=t0, t_end=time.time(), wall_s=time.time() - t0, node=node_role(), lease=a.lease,
               label=req.get("label"), cpu_quota=req.get("cpu_cores"), memory_bytes=req.get("memory_bytes"),
               gpu=req.get("gpu"), gpu_memory_bytes=req.get("gpu_memory_bytes"),
               cpu_core_s=(info.get("cpu_usage_usec") or 0) / 1e6,
               gpu_device_s=(time.time() - t0) if req.get("gpu") else 0.0,
               returncode=res.returncode, stopped_by=res.stopped_by, cmd=" ".join(cmd)[:300])
    ledger = ops_root() / "ops" / "resource-ledger.jsonl"
    with open(ledger, "a") as lf:
        lf.write(json.dumps(rec) + "\n")
    log.with_suffix(".rc").write_text(str(res.returncode if res.returncode is not None else -1))
    print(f"[rrp.child] lease={a.lease} rc={res.returncode} stopped_by={res.stopped_by} heartbeats={res.heartbeats}",
          flush=True)
    try:
        br.release(a.lease, cleanup_backend=False)
    except Exception as e:  # noqa: BLE001
        print(f"[rrp.child] release failed: {e}", flush=True)
    return 0 if res.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
