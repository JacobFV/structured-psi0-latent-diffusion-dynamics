#!/usr/bin/env bash
# W6 robustness sweep launcher — PEER ONLY (D-115: no training or heavy compute on the host).
# One peer `rrp ops run` lease per attempt, bounded retries; shards resume (finished shards/episodes are skipped).
# usage: scripts/robust_peer_run.sh LABEL CPU MEM MAX_ATTEMPTS -- <rrp.evaluation.robustness run args...>
# Outputs land in the shared peer store (artifacts/ of the peer code dir); fetch them with rsync afterwards.
set -u
LABEL=$1 CPU=$2 MEM=$3 MAXA=$4; shift 5
cd "$(dirname "$0")/.."
: "${RRP_PEER_REPO:?export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/robust (and scripts/peer_sync.sh push) first}"
for a in $(seq 1 "$MAXA"); do
  scripts/peer_run.sh --cpu "$CPU" --mem "$MEM" --label "$LABEL" --max-seconds 21600 -- \
    env RRP_CONTACT_MODEL=v2 CUDA_VISIBLE_DEVICES= PY -m rrp.evaluation.robustness run "$@"
  rc=$?
  echo "[robust_peer_run] $LABEL attempt $a rc=$rc $(date +%T)"
  [ $rc -eq 0 ] && exit 0
  sleep 120
done
echo "[robust_peer_run] $LABEL FAILED after $MAXA attempts"; exit 1
