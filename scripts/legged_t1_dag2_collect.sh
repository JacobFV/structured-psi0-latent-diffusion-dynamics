#!/usr/bin/env bash
set -uo pipefail
B=artifacts/runs/legged_bc_t1_v1/policy.pt
for v in sem nosem; do
  bash scripts/legged_collect.sh artifacts/runs/legged_buf/dag2_${v}_t1 3 22000 22023 oracle_bc t1 $B artifacts/runs/legged_rep_${v}_t1_v2/representation.pt artifacts/runs/legged_rz_${v}_t1_dag1/realizer.pt &
done
wait
