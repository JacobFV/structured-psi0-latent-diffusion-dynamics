#!/usr/bin/env bash
# Demo sprint: ladder R1 (ORACLE DIAGNOSTIC) clips, re-anchored every replan (8 ticks; valid per D-049), prev-action 0.
# Usage: render_ladder.sh <rep.pt> <tag> <robot> <seeds>
set -uo pipefail
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
REP=$1; TAG=$2; ROBOT=$3; SEEDS=$4
$PY scripts/ladder.py --route oracle --reanchor --prev-action zero --robot $ROBOT --rep $REP --tag $TAG \
    --render-seeds $SEEDS --video-out artifacts/runs/demo_video --out artifacts/runs/demo_video/ladder_tmp
