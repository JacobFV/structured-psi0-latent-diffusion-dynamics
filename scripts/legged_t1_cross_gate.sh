#!/usr/bin/env bash
# pre-refit system-0 error of sem / nosem on states visited by EACH R1 route (cross), labels = stateless BC
set -uo pipefail
PY=${PY:-python}
for v in sem nosem; do for s in sem nosem; do
  OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY -m rrp.learning.legged_dagger gate --rep artifacts/runs/legged_rep_${v}_t1_v2/representation.pt --buf artifacts/runs/legged_buf/dag1_${s}_t1 --body t1 --out artifacts/runs/legged_gate/t1_${v}_on_r1${s}_states.json > /dev/null 2>&1 &
done; done; wait
