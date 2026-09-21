"""Peer discovery restricted to literal, already-configured SSH aliases.

No network scanning, no host-key changes (StrictHostKeyChecking=yes, BatchMode=yes),
no private-key inspection. Identity is verified via machine-id/hostname and must differ
from the local machine. Unique identifiers are redacted in written manifests.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
from dataclasses import dataclass, asdict, field
from pathlib import Path

RELEVANT = re.compile(r"(gb10|spark|dgx|peer)", re.I)

PROBE = r"""set -e
echo "hostname=$(hostname)"
echo "machine_id=$(cat /etc/machine-id)"
echo "arch=$(uname -m)"
echo "kernel=$(uname -r)"
echo "nproc=$(nproc)"
awk '/MemTotal|MemAvailable|SwapTotal|SwapFree/{print tolower($1)"="$2*1024}' /proc/meminfo | tr -d ':'
df -B1 --output=size,avail "$HOME" | tail -1 | awk '{print "disk_total="$1"\ndisk_free="$2}'
df -B1 --output=size,avail /dev/shm | tail -1 | awk '{print "shm_total="$1"\nshm_free="$2}'
echo "load=$(cut -d' ' -f1-3 /proc/loadavg)"
echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
echo "gpu_apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')"
echo "cuda=$(ls -d /usr/local/cuda-* 2>/dev/null | tr '\n' ' ')"
echo "cgroup_controllers=$(cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/cgroup.controllers 2>/dev/null)"
echo "python=$(python3 --version 2>&1)"
"""


def ssh_aliases(config: Path = Path.home() / ".ssh/config") -> list[dict]:
    out, cur = [], None
    if not config.exists():
        return out
    for line in config.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        key, _, val = s.partition(" ")
        key = key.lower()
        if key == "host":
            cur = {"aliases": val.split(), "hostname": None}
            out.append(cur)
        elif cur is not None and key == "hostname":
            cur["hostname"] = val.strip()
    return out


def candidate_aliases(explicit: str | None = None, config: Path | None = None) -> list[str]:
    if explicit:
        return [explicit]
    cands = []
    for entry in ssh_aliases(config or Path.home() / ".ssh/config"):
        for a in entry["aliases"]:
            if "*" not in a and RELEVANT.search(a):
                cands.append(a)
    return cands


def parse_probe(text: str) -> dict:
    d = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    for k in ("nproc", "memtotal", "memavailable", "swaptotal", "swapfree", "disk_total",
              "disk_free", "shm_total", "shm_free"):
        if k in d:
            try:
                d[k] = int(float(d[k]))
            except ValueError:
                d[k] = None
    return d


def local_identity() -> dict:
    return {"hostname": socket.gethostname(), "machine_id": Path("/etc/machine-id").read_text().strip()}


def redact(v: str) -> str:
    return "sha256:" + hashlib.sha256(v.encode()).hexdigest()[:12]


@dataclass
class PeerProbe:
    alias: str
    reachable: bool
    error: str | None = None
    facts: dict = field(default_factory=dict)
    distinct_from_local: bool | None = None
    route: str | None = None


def probe_alias(alias: str, timeout_s: int = 12) -> PeerProbe:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={min(timeout_s, 10)}",
           "-o", "StrictHostKeyChecking=yes", alias, "bash", "-s"]
    try:
        r = subprocess.run(cmd, input=PROBE, capture_output=True, text=True, timeout=timeout_s + 10)
    except subprocess.TimeoutExpired:
        return PeerProbe(alias, False, "timeout")
    if r.returncode != 0:
        return PeerProbe(alias, False, r.stderr.strip()[-200:])
    facts = parse_probe(r.stdout[:20000])
    loc = local_identity()
    distinct = facts.get("machine_id") not in (None, "", loc["machine_id"])
    return PeerProbe(alias, True, None, facts, distinct)


def route_for(alias: str) -> str | None:
    for e in ssh_aliases():
        if alias in e["aliases"] and e["hostname"]:
            try:
                r = subprocess.run(["ip", "route", "get", e["hostname"]], capture_output=True, text=True, timeout=5)
                return r.stdout.strip().splitlines()[0] if r.stdout else None
            except (OSError, subprocess.TimeoutExpired):
                return None
    return None


def discover(explicit: str | None = None) -> dict:
    probes = []
    for a in candidate_aliases(explicit or os.environ.get("ROBOT_PEER")):
        p = probe_alias(a)
        p.route = route_for(a)
        probes.append(p)
    # prefer: reachable, distinct, no foreign GPU compute app, physically direct route
    def score(p: PeerProbe):
        if not (p.reachable and p.distinct_from_local):
            return -1
        s = 1
        if not p.facts.get("gpu_apps"):
            s += 2
        if p.route and not re.search(r"tailscale|zt", p.route):
            s += 1
        return s
    ranked = sorted(probes, key=score, reverse=True)
    chosen = ranked[0] if ranked and score(ranked[0]) > 0 else None
    # machines seen via several aliases are the same machine
    by_id = {}
    for p in probes:
        if p.reachable:
            by_id.setdefault(p.facts.get("machine_id"), []).append(p.alias)
    return {"local": {k: (redact(v) if k == "machine_id" else v) for k, v in local_identity().items()},
            "chosen_peer_alias": chosen.alias if chosen else None,
            "probes": [_redacted(p) for p in probes],
            "aliases_by_machine": {redact(k or "none"): v for k, v in by_id.items()}}


def _redacted(p: PeerProbe) -> dict:
    d = asdict(p)
    f = dict(d["facts"])
    if "machine_id" in f:
        f["machine_id"] = redact(f["machine_id"])
    d["facts"] = f
    return d
