#!/usr/bin/env bash
set -uo pipefail
B=artifacts/runs/legged_bc_t1_v1/policy.pt
for v in sem nosem; do
  bash scripts/legged_collect.sh artifacts/runs/legged_buf/dag1_${v}_t1s1 4 21000 21031 oracle_bc t1 $B artifacts/runs/legged_rep_${v}_t1_v2s1/representation.pt &
done
wait
