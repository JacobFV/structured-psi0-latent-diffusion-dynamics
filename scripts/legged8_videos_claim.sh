#!/usr/bin/env bash
# Render W8 ctx-halt clips inside the track GPU budget: claim one GPU slot in the shared-budget dir (a ledger the run-dag
# coordinators count), wait until the rest of W8 holds <= MAXGPU-1 GPU leases, render each clip (short GPU lease), release.
# usage: [MAXGPU=3] legged8_videos_claim.sh NAME "BODY TRAINSEED EVALSEED KIND [LINEAGE_BODY]" ...   (env CAVEAT passed through)
cd ~/work/rrp-wt/legged8
NAME=$1; shift; MAXGPU=${MAXGPU:-3}
D=artifacts/runs/legged8/_dags/_manual_videos_$NAME; mkdir -p $D; L=$D/ledger.json
claim() { printf '{"schema": "dag-ledger-1", "nodes": {"videos": {"state": "%s", "resources": {"cpu": 2, "mem": "8G", "gpu": true, "gpu_mem": "2G"}}}}\n' "$1" > $L.tmp && mv $L.tmp $L; }
claim running
others() { python3 - "$NAME" <<'PY'
import json, glob, sys
n = 0
for f in glob.glob("artifacts/runs/legged8/_dags/*/ledger.json"):
    if f"_manual_videos_{sys.argv[1]}/" in f: continue
    try: d = json.load(open(f))
    except Exception: continue
    n += sum(1 for e in d.get("nodes", {}).values() if e.get("state") == "running" and (e.get("resources") or {}).get("gpu"))
print(n)
PY
}
for i in $(seq 1 720); do [ "$(others)" -le $((MAXGPU - 1)) ] && break; sleep 30; done
set -o pipefail
rc=0
for job in "$@"; do set -- $job
  ok=0
  for try in $(seq 1 30); do     # admission refusals (capacity / AdmissionStopped) are waited for, bounded (30 x 60 s)
    o=$(RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/legged8 scripts/peer_run.sh --gpu --gpu-mem 2G --cpu 2 --mem 8G --label l8_video_$1_$4 --max-seconds 1800 -- \
      env PY=/dev/shm/rrp-brandonin/venv/bin/python LB=${5:-$1} CAVEAT="${CAVEAT:-}" bash scripts/legged8_videos.sh $1 $2 $3 $4 2>&1); r=$?
    if grep -qE "AdmissionStopped|CapacityError" <<<"$o" && ! grep -q '"lease_id"' <<<"$o"; then echo "admission refused ($1 $4), waiting"; sleep 60; continue; fi
    echo "$o" | tail -4; [ $r = 0 ] && grep -q '"returncode": 0' <<<"$o" && ok=1; break
  done
  [ $ok = 1 ] || { echo "FAIL video $1 $3 $4"; rc=1; }
done
claim completed
echo "videos $NAME rc=$rc"
