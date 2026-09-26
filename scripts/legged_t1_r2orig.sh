#!/usr/bin/env bash
# R2 with the ORIGINAL (no DAgger) system 0 and the final flow, sem + nosem, for a t1 tag (t1_v2, t1_v2s1, ...)
set -uo pipefail
T=$1
for v in sem nosem; do
  bash scripts/legged_ladder.sh t1 ${v}_${T}_orig_policy 3 r2 - - - artifacts/runs/legged_flow_${v}_${T}/policy.pt &
done
wait
