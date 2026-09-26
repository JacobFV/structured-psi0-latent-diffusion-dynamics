#!/usr/bin/env bash
# Same seed: R0 scripted teacher | plain BC | stateless R1 ORACLE (E(BC chunk) -> system 0 <tag>). Compose.
# Usage: render_orcbc_triptych.sh <seed> <robot> <rep.pt> <tag>
set -uo pipefail
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
SD=$1; R=$2; REP=$3; T=$4; BC=artifacts/runs/baselines_bc_ckpts/direct1701_u12000.pt; BT=direct1701_u12000
O=artifacts/runs/demo_video/orcbc_${R}_$SD; mkdir -p $O
$PY scripts/ladder.py --route teacher --robot $R --tag teacher --render-seeds $SD --video-out $O --out $O/tmp
$PY scripts/ladder.py --route learned --policy $BC --policy-label $BT --prev-action zero --robot $R --tag $BT --render-seeds $SD --video-out $O --out $O/tmp
$PY scripts/ladder.py --route oracle --oracle-expert bc --policy $BC --policy-label $BT --prev-action zero --robot $R --rep $REP --tag ${T}_orcbc --render-seeds $SD --video-out $O --out $O/tmp
$PY scripts/demo/side_by_side.py artifacts/runs/demo_video/2026-09-25_orcbctriptych_${R}_s${SD}_teacher_bc_orcbc-${T}.mp4 \
    $(ls $O/*_teacher_*.mp4) $(ls $O/*_learned_*_${BT}_*.mp4) $(ls $O/*_${T}_orcbc_*.mp4)
cat $O/INDEX.md
