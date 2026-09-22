"""Sum CPU/GPU cost of the legged track from resource ledgers (peer + host copies).
usage: python3 scripts/legged_costs.py host_ledger.jsonl peer_ledger.jsonl"""
import json, sys
from collections import defaultdict
PREFIX = ("tracker_", "val_", "cpg_val", "legged_", "hex_val", "wp_smoke", "poolprof")
tot = defaultdict(lambda: dict(runs=0, wall_s=0.0, cpu_core_s=0.0, gpu_device_s=0.0, stopped=[]))
for path in sys.argv[1:]:
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        lab = r.get("label", "")
        if not lab.startswith(PREFIX):
            continue
        key = f"{r.get('node')}:{lab}"
        t = tot[key]
        t["runs"] += 1
        t["wall_s"] += r.get("wall_s") or 0
        t["cpu_core_s"] += r.get("cpu_core_s") or 0
        t["gpu_device_s"] += r.get("gpu_device_s") or 0
        if r.get("stopped_by"):
            t["stopped"].append(r["stopped_by"])
out = {k: dict(v, cpu_core_h=round(v["cpu_core_s"] / 3600, 3)) for k, v in sorted(tot.items())}
print(json.dumps(out, indent=1))
print("TOTAL cpu_core_h", round(sum(v["cpu_core_s"] for v in tot.values()) / 3600, 2),
      "gpu_device_h", round(sum(v["gpu_device_s"] for v in tot.values()) / 3600, 3))
