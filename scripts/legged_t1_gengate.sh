#!/usr/bin/env bash
set -uo pipefail
PY=${PY:-python}
for v in sem nosem; do for s in bc_t1 dag1_${v}_t1; do
  OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY -m rrp.learning.legged_dagger gate --rep artifacts/runs/legged_rep_${v}_t1_v2/representation.pt --realizer artifacts/runs/legged_rz_${v}_t1_dag1/realizer.pt --flow artifacts/runs/legged_flow_${v}_t1_v2/policy.pt --buf artifacts/runs/legged_buf/$s --body t1 --out artifacts/runs/legged_gate/t1_${v}_dag1_gen_on_${s}.json > /dev/null 2>&1 &
done; done; wait
