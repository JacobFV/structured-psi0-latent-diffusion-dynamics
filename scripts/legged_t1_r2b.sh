#!/usr/bin/env bash
# t1 R2: final flows with the DAgger-1 system 0s, and snap8000 flows with the ORIGINAL (pre-DAgger) system 0s
set -uo pipefail
for v in sem nosem; do
  bash scripts/legged_ladder.sh t1 ${v}_t1_dag1_policy 2 r2 - - artifacts/runs/legged_rz_${v}_t1_dag1/realizer.pt artifacts/runs/legged_flow_${v}_t1_v2/policy.pt &
  bash scripts/legged_ladder.sh t1 ${v}_t1_v2_snap_s8000 2 r2 - - - artifacts/runs/legged_flow_${v}_t1_v2/snap_s8000.pt &
done
wait
