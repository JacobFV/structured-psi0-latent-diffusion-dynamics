#!/usr/bin/env bash
# Same seed: R0 scripted teacher | plain BC (learned) | R2 GENERATED latent route (learned sys-i flow -> sys-0). Compose.
# Usage: render_r2_triptych.sh <seed> <robot> <rep.pt> <flow.pt> <flow tag> <bc ckpt> <bc tag>
set -uo pipefail
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
SD=$1; R=$2; REP=$3; FLOW=$4; FT=$5; BC=$6; BT=$7
O=artifacts/runs/demo_video/r2_${R}_$SD; mkdir -p $O
[ -n "$(ls $O/*_teacher_*.mp4 2>/dev/null)" ] || $PY scripts/ladder.py --route teacher --robot $R --tag teacher --render-seeds $SD --video-out $O --out $O/tmp
[ -n "$(ls $O/*_${BT}_*.mp4 2>/dev/null)" ] || $PY scripts/ladder.py --route learned --policy $BC --policy-label $BT --prev-action zero --robot $R --tag $BT --render-seeds $SD --video-out $O --out $O/tmp
$PY scripts/ladder.py --route generated --rep $REP --flow $FLOW --prev-action zero --robot $R --tag $FT --render-seeds $SD --video-out $O --out $O/tmp
$PY scripts/demo/side_by_side.py artifacts/runs/demo_video/2026-09-25_r2triptych_${R}_s${SD}_teacher_bc-${BT}_generated-${FT}.mp4 \
    $(ls $O/*_teacher_*.mp4) $(ls $O/*_${BT}_*.mp4) $(ls $O/*_${FT}_*.mp4)
cat $O/INDEX.md
