#!/usr/bin/env bash
# Host-side supervisor for one baseline slot: re-launches the (resumable) slot lease on the peer in <=6 h segments
# (broker cap) until the slot writes its done marker. Lightweight; run detached:
#   setsid nohup scripts/baselines_supervise.sh <method> <seeds...> > ops-local/<name>.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export RRP_PEER_REPO=${RRP_PEER_REPO:-/dev/shm/rrp-brandonin/wt/baselines}
method=$1; shift
ROOT=${ROOT:-artifacts/runs/latent_slice1}
marker="$RRP_PEER_REPO/$ROOT/.slot_done_${method}_$(echo "$@" | tr ' ' _)"
for seg in $(seq 1 20); do
  if ssh gb10-direct "test -e $marker"; then echo "$(date -Is) slot done"; exit 0; fi
  echo "$(date -Is) segment $seg"
  scripts/peer_run.sh --gpu --gpu-mem 16G --cpu 4 --mem 24G --label "baselines_${method#baseline_}" --max-seconds 21000 \
    -- env PY=/dev/shm/rrp-brandonin/venv/bin/python ROOT="$ROOT" bash scripts/baselines_slot.sh "$method" "$@" 2>&1 \
    | grep -E '^\[slot\]|lease_id|FAILED|Error' | tail -40
  sleep 60
done
