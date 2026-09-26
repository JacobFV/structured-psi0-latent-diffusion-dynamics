#!/usr/bin/env bash
# Same seed, three controllers: R0 scripted teacher | plain BC (learned) | R1 oracle diagnostic (latent). Then compose.
# Usage: render_triptych.sh <seed> <oracle rep> <oracle tag> <bc ckpt> <bc tag>
set -uo pipefail
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
SD=$1; REP=$2; OT=$3; BC=$4; BT=$5; R=panda_pg2
O=artifacts/runs/demo_video/tri_$SD; mkdir -p $O
$PY scripts/ladder.py --route teacher --robot $R --tag teacher --render-seeds $SD --video-out $O --out $O/tmp
$PY scripts/ladder.py --route learned --policy $BC --policy-label $BT --prev-action zero --robot $R --tag $BT --render-seeds $SD --video-out $O --out $O/tmp
$PY scripts/ladder.py --route oracle --reanchor --prev-action zero --robot $R --rep $REP --tag $OT --render-seeds $SD --video-out $O --out $O/tmp
$PY scripts/demo/side_by_side.py artifacts/runs/demo_video/2026-09-25_triptych_panda_pg2_s${SD}_teacher_bc-${BT}_oracle-${OT}.mp4 \
    $(ls $O/*_teacher_*.mp4) $(ls $O/*_${BT}_*.mp4) $(ls $O/*_${OT}_*.mp4)
cat $O/INDEX.md
