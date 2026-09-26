#!/usr/bin/env bash
# Demo sprint: plain-BC positive-control clips through the ladder's learned route (same scenes/tracker/evaluator).
# Usage: render_bc.sh <ckpt.pt> <label> <robot> <seeds>
set -uo pipefail
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
$PY scripts/ladder.py --route learned --policy $1 --policy-label $2 --prev-action zero --robot $3 --tag $2 \
    --render-seeds $4 --video-out artifacts/runs/demo_video --out artifacts/runs/demo_video/ladder_tmp
