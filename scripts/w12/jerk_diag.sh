#!/usr/bin/env bash
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
export RRP_GRASP_CONTACT=v2.1
O=artifacts/runs/w12_smoke/jerk_diag.jsonl
for spec in "handover panda_pg2__ur5e_pg2 0" "handover parm5_pg2__parm5_pg2 0" "support_insert panda_pg2__ur5e_pg2 0"; do
  for o in '{}' '{"limit_aware": false}' '{"limit_aware": false, "confirm_grasp": false}'; do
    $PY scripts/w12/jerk_diag.py $spec "$o" v3 900 >> $O
  done
  $PY scripts/w12/jerk_diag.py $spec '{}' v2 900 >> $O
done
