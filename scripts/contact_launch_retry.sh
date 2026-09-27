#!/usr/bin/env bash
# Bounded admission retry for host jobs (D-061): try every 120 s for at most $TRIES attempts; exit 0 once admitted, 3 if never.
# usage: TRIES=45 scripts/contact_launch_retry.sh <ops-run args...> -- cmd...
set -u
TRIES=${TRIES:-45}
PY=$HOME/work/relational-robot-policy/.venv/bin/python
for i in $(seq 1 "$TRIES"); do
  out=$(PYTHONPATH=src "$PY" -m rrp.cli ops run --detach "$@" 2>&1 | tail -1)
  if echo "$out" | grep -q lease_id; then echo "admitted after $i: $out"; exit 0; fi
  echo "try $i refused: $(echo "$out" | cut -c1-120)"; sleep 120
done
exit 3
