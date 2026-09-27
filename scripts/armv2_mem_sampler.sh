#!/usr/bin/env bash
# Sample cgroup memory.peak of every active armv2 (a2*) lease slice on the peer; keep the max per label in a JSON file.
OUT=$HOME/work/rrp-wt/ladder/artifacts/runs/armv2/_dags/arm_lineage_v2/memory_peaks.json
for i in $(seq 1 2000); do
  ssh gb10-direct 'cd /dev/shm/rrp-brandonin/repo && python3 - <<PY
import json, pathlib
s = json.load(open("ops/broker/peer/state.json"))
base = pathlib.Path("/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/rrp.slice/rrp-lease.slice")
out = {}
for lid, v in s.get("leases", {}).items():
    lab = v["request"]["label"]
    if v.get("state") == "active" and lab.startswith("a2"):
        p = base / f"rrp-lease-{lid}.slice" / "memory.peak"
        try: out[lab] = int(p.read_text())
        except OSError: pass
print(json.dumps(out))
PY' 2>/dev/null | python3 -c "
import json,sys,os
new=json.loads(sys.stdin.read() or '{}'); p='$OUT'
old=json.load(open(p)) if os.path.exists(p) else {}
for k,v in new.items(): old[k]=max(old.get(k,0),v)
json.dump(old,open(p,'w'),indent=1,sort_keys=True)"
  systemctl --user is-active -q rrp-armv2-dag4 || systemctl --user is-active -q rrp-armv2-dag5 || break
  sleep 60
done
