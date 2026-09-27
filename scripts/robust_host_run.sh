#!/usr/bin/env bash
# W6 robustness sweep launcher (host): one `rrp ops run` lease per call, bounded retries (the host memory-PSI
# watchdog sheds jobs under external swap pressure; shards resume: finished shards/episodes are skipped).
# usage: scripts/robust_host_run.sh LABEL CPU MEM MAX_ATTEMPTS -- <robustness run args...>
set -u
LABEL=$1 CPU=$2 MEM=$3 MAXA=$4; shift 5
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
for a in $(seq 1 "$MAXA"); do
  PYTHONPATH=src "$PY" -m rrp.cli ops run --cpu "$CPU" --mem "$MEM" --label "$LABEL" --max-seconds 21600 -- \
    env RRP_CONTACT_MODEL=v2 CUDA_VISIBLE_DEVICES= PYTHONPATH=src "$PY" -m rrp.evaluation.robustness run "$@"
  rc=$?
  echo "[robust_host_run] $LABEL attempt $a rc=$rc $(date +%T)"
  [ $rc -eq 0 ] && exit 0
  sleep 120
done
echo "[robust_host_run] $LABEL FAILED after $MAXA attempts"; exit 1
