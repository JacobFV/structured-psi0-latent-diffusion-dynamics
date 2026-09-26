#!/usr/bin/env bash
set -uo pipefail
for v in sem nosem; do bash scripts/legged_ladder.sh hexapod6 ${v}_hexapod6_v2_policy 3 r2 - - - artifacts/runs/legged_flow_${v}_hexapod6_v2/policy.pt & done; wait
