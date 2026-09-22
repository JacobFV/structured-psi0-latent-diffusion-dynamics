#!/usr/bin/env bash
# Host-side supervisor for a resumable legged tracker training job on the peer (via the broker).
# Relaunches with --resume after thermal/pressure sheds, only when the peer CPU is below a
# temperature threshold. Never touches jobs it did not start (matches its own label).
# usage: scripts/legged_supervise.sh BODY CPU MEM WORKERS ITERS [extra trainer args...]
set -uo pipefail
BODY=$1; CPU=$2; MEM=$3; WORKERS=$4; ITERS=$5; shift 5; EXTRA="$*"
LABEL="tracker_${BODY}"
OUT=/dev/shm/rrp-brandonin/legged_runs/$BODY
PEER=gb10-direct
PY=/dev/shm/rrp-brandonin/venv/bin/python
while true; do
  last=$(ssh $PEER "tail -n1 $OUT/train_log.jsonl 2>/dev/null | python3 -c 'import json,sys; print(json.loads(sys.stdin.read() or \"{}\").get(\"iter\",-1))' 2>/dev/null || echo -1")
  if [ "${last:--1}" -ge $((ITERS-1)) ]; then echo "$(date +%T) $BODY done at iter $last"; exit 0; fi
  active=$(ssh $PEER "cd /dev/shm/rrp-brandonin/repo && PATH=/dev/shm/rrp-brandonin/bin:\$PATH PYTHONPATH=src RRP_NODE=peer RRP_REPO=\$PWD python3 -m rrp.cli ops status 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(sum(1 for v in d[\"active_leases\"].values() if v[\"request\"][\"label\"]==\"$LABEL\"))'")
  if [ "${active:-0}" = "0" ]; then
    temp=$(ssh $PEER "cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1")
    if [ "${temp:-99999}" -lt 88000 ]; then
      echo "$(date +%T) launching $BODY (iter $last, temp $temp)"
      ssh $PEER "mkdir -p $OUT && cd /dev/shm/rrp-brandonin/repo && export PATH=/dev/shm/rrp-brandonin/bin:\$PATH PYTHONPATH=src RRP_NODE=peer RRP_REPO=\$PWD && python3 -m rrp.cli ops run --cpu $CPU --mem $MEM --label $LABEL --max-seconds 21000 --detach -- $PY -m rrp.control.tracker_training --body $BODY --out $OUT --iters $ITERS --workers $WORKERS --resume $EXTRA" 2>&1 | tail -1
    else
      echo "$(date +%T) $BODY waiting: peer temp $temp"
    fi
  fi
  sleep 90
done
