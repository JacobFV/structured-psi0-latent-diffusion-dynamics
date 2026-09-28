#!/usr/bin/env bash
# Sample, every 60 s, for every active armv6 (a6*) lease on the peer: the lease slice's cgroup memory.peak (RAM incl. page
# cache) and the CUDA memory of its processes (nvidia-smi compute apps mapped to the lease via /proc/<pid>/cgroup; on the
# GB10 unified memory this is NOT in the cgroup figure). Keeps the max per label in memory_peaks.json (bytes / MiB).
OUT=$HOME/work/rrp-wt/ladder/artifacts/runs/armv6/_dags/arm_lineage_v6/memory_peaks.json
for i in $(seq 1 2000); do
  ssh gb10-direct 'cd /dev/shm/rrp-brandonin/repo && python3 - <<PY
import json, pathlib, subprocess, re
s = json.load(open("ops/broker/peer/state.json"))
base = pathlib.Path("/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/rrp.slice/rrp-lease.slice")
lab = {lid: v["request"]["label"] for lid, v in s.get("leases", {}).items() if v.get("state") == "active" and v["request"]["label"].startswith("a6")}
out = {}
for lid, l in lab.items():
    try: out[l] = {"ram_peak_bytes": int((base / f"rrp-lease-{lid}.slice" / "memory.peak").read_text())}
    except OSError: pass
q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
for line in q.splitlines():
    try: pid, mib = [x.strip() for x in line.split(",")]
    except ValueError: continue
    try: cg = open(f"/proc/{pid}/cgroup").read()
    except OSError: continue
    m = re.search(r"rrp-lease-([0-9]+_[0-9a-f]+)\.slice", cg)
    if m and m.group(1) in lab:
        d = out.setdefault(lab[m.group(1)], {}); d["cuda_mib"] = d.get("cuda_mib", 0) + int(mib)
print(json.dumps(out))
PY' 2>/dev/null | python3 -c "
import json,sys,os
new=json.loads(sys.stdin.read() or '{}'); p='$OUT'
old=json.load(open(p)) if os.path.exists(p) else {}
for k,v in new.items():
    o=old.get(k); o=o if isinstance(o,dict) else ({'ram_peak_bytes':o} if o else {})
    for f,x in v.items(): o[f]=max(o.get(f,0),x)
    old[k]=o
json.dump(old,open(p,'w'),indent=1,sort_keys=True)"
  systemctl --user is-active -q rrp-armv6-dag || systemctl --user is-active -q rrp-armv6-dag2 || break
  sleep 60
done
