#!/usr/bin/env bash
# seed-1 t1: R1 (DAgger-1 system 0) and R2 (final flow; original and DAgger-1 system 0), identical to seed 0
set -uo pipefail
B=artifacts/runs/legged_bc_t1_v1/policy.pt
for v in sem nosem; do
  bash scripts/legged_ladder.sh t1 ${v}_t1s1_dag1 2 r1 $B artifacts/runs/legged_rep_${v}_t1_v2s1/representation.pt artifacts/runs/legged_rz_${v}_t1s1_dag1/realizer.pt &
  bash scripts/legged_ladder.sh t1 ${v}_t1s1_dag1_policy 2 r2 - - artifacts/runs/legged_rz_${v}_t1s1_dag1/realizer.pt artifacts/runs/legged_flow_${v}_t1_v2s1/policy.pt &
  bash scripts/legged_ladder.sh t1 ${v}_t1s1_orig_policy 2 r2 - - - artifacts/runs/legged_flow_${v}_t1_v2s1/policy.pt &
done
wait
