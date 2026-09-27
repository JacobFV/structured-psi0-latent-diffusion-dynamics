#!/usr/bin/env bash
# Bounded driver for dags/armexpert_v4dart.yaml on the host: waits for broker admission (host memory-PSI stops),
# reruns failed nodes (collect/generate and pack resume or restart safely) at most MAX times.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/work/relational-robot-policy/.venv/bin/python; MAX=${MAX:-8}
admit() { for _ in $(seq 1 360); do PYTHONPATH=src $PY -m rrp.cli ops status 2>/dev/null | grep -q '"admission_stopped": false' && return 0; sleep 10; done; return 1; }
for i in $(seq 1 $MAX); do
  admit || { echo "admission stopped for 60 min"; exit 2; }
  extra=(); [ $i = 1 ] && extra=("$@")
  PYTHONPATH=src $PY -m rrp.cli run-dag dags/armexpert_v4dart.yaml --retry-failed --poll 20 "${extra[@]}"
  PYTHONPATH=src $PY -m rrp.cli run-dag dags/armexpert_v4dart.yaml --dry-run | grep -q "\[completed" && \
    ! PYTHONPATH=src $PY -m rrp.cli run-dag dags/armexpert_v4dart.yaml --dry-run | grep -q "\[failed\|\[planned\|\[blocked" \
    && { echo "DAG complete (attempt $i)"; exit 0; }
  echo "attempt $i incomplete"
done
exit 1
