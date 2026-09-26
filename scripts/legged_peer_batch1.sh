#!/usr/bin/env bash
set -uo pipefail
bash scripts/legged_collect.sh artifacts/runs/legged_buf/bc_hexapod6 3 20000 20023 bc hexapod6 artifacts/runs/legged_bc_hexapod6_v1/policy.pt &
bash scripts/legged_collect.sh artifacts/runs/legged_buf/bc_t1 3 20000 20023 bc t1 artifacts/runs/legged_bc_t1_v1/policy.pt &
wait
bash scripts/legged_r2_suite.sh go2 go2_v2 snap_s8000.pt 3
