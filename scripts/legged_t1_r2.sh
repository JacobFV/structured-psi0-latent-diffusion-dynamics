#!/usr/bin/env bash
set -uo pipefail
CK=${1:-snap_s8000.pt}
for v in sem nosem; do
  bash scripts/legged_ladder.sh t1 ${v}_t1_dag1_${CK%.pt} 3 r2 - - artifacts/runs/legged_rz_${v}_t1_dag1/realizer.pt artifacts/runs/legged_flow_${v}_t1_v2/$CK &
done
wait
