#!/usr/bin/env bash
# LEGGED FIXED-SEM RERUN: Stage A (bounded semantic NLL, probe_lv_min -4) then flow, per body. One GPU lease.
# go2 Stage A runs alone; then go2 flow and hexapod6 Stage A run concurrently; then hexapod6 flow. rc checked.
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
rep() { R=artifacts/runs/legged_fixsem_rep_sem_$1_lv4
  [ -f $R/representation.pt ] || PYTHONPATH=src $PY -m rrp.learning.legged_latent_train rep --config configs/legged_fixsem/rep_sem_$1_lv4.json --out $R > $R.out 2>&1; }
flow() { F=artifacts/runs/legged_fixsem_flow_sem_$1_lv4
  [ -f $F/policy.pt ] || PYTHONPATH=src $PY -m rrp.learning.legged_latent_train flow --config configs/legged_fixsem/flow_sem_$1_lv4.json --out $F > $F.out 2>&1; }
BODIES=${BODIES:-go2 hexapod6}
if [ "$BODIES" = "go2 hexapod6" ]; then
  rep go2 || exit 1
  flow go2 & p1=$!; rep hexapod6 & p2=$!
  wait $p1 || exit 2; wait $p2 || exit 3
  flow hexapod6 || exit 4
else
  for b in $BODIES; do rep $b || exit 1; flow $b || exit 2; done
fi
