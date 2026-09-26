#!/usr/bin/env bash
# Demo sprint: stateless R1 clips (ORACLE DIAGNOSTIC: packet = E(BC chunk at current state) -> system 0).
# Usage: render_orcbc.sh <rep.pt> <tag> <robot> <seeds> [bc ckpt] [bc label]
set -uo pipefail
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
BC=${5:-artifacts/runs/baselines_bc_ckpts/direct1701_u12000.pt}; BL=${6:-direct1701_u12000}
$PY scripts/ladder.py --route oracle --oracle-expert bc --policy $BC --policy-label $BL --prev-action zero --robot $3 \
    --rep $1 --tag ${2}_orcbc --render-seeds $4 --video-out artifacts/runs/demo_video --out artifacts/runs/demo_video/ladder_tmp
