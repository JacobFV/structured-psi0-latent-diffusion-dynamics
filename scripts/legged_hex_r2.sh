#!/usr/bin/env bash
# hexapod6 deployable route: R2 ladder + gen gate, z edits, task-context edits (one CPU lease). usage: CK
set -uo pipefail
CK=${1:-snap_s4000.pt}
bash scripts/legged_r2_suite.sh hexapod6 hexapod6_v2 $CK 3
bash scripts/legged_r2_edits.sh hexapod6 hexapod6_v2 $CK &
bash scripts/legged_r2_ctx_edits.sh hexapod6 hexapod6_v2 $CK &
wait
