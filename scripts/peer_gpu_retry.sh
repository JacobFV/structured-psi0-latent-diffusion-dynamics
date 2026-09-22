#!/usr/bin/env bash
# Retry a detached peer GPU lease on CapacityError/AdmissionStopped (slowly). Usage:
#   scripts/peer_gpu_retry.sh LABEL MAX_SECONDS GPU_MEM CPU MEM -- python-args...
set -u
label=$1; maxs=$2; gmem=$3; cpu=$4; mem=$5; shift 6
for i in $(seq 1 40); do
  out=$(ssh gb10-direct "cd /dev/shm/rrp-brandonin/repo && export PATH=/dev/shm/rrp-brandonin/bin:\$PATH PYTHONPATH=src RRP_NODE=peer RRP_REPO=\$PWD && python3 -m rrp.cli ops run --gpu --gpu-mem $gmem --cpu $cpu --mem $mem --label $label --max-seconds $maxs --detach -- /dev/shm/rrp-brandonin/venv/bin/python $*" 2>&1)
  if echo "$out" | grep -q '"lease_id"'; then echo "$out" | tail -3; exit 0; fi
  echo "try $i: $(echo "$out" | tail -1)"; sleep 240
done
exit 1
