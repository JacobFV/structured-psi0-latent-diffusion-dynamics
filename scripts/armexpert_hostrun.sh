#!/usr/bin/env bash
# Run one leased host job after waiting (<= 40 min) for broker admission (the host memory-PSI watchdog stops
# admission under external memory pressure). Usage: scripts/armexpert_hostrun.sh <ops-run args> -- cmd...
set -uo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/work/relational-robot-policy/.venv/bin/python
for _ in $(seq 1 240); do
  PYTHONPATH=src $PY -m rrp.cli ops status 2>/dev/null | grep -q '"admission_stopped": false' && break
  sleep 10
done
PYTHONPATH=src exec $PY -m rrp.cli ops run "$@"
