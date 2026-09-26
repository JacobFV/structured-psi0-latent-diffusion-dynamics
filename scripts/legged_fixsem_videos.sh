#!/usr/bin/env bash
# LEGGED FIXED-SEM clips (EGL; GPU lease). usage: legged_fixsem_videos.sh BODY
# (1) fixed-sem R2 full episode on the seed where the original sem snap_s4000 route fell (go2 10017);
# (2) task-view `halt` context edit from t=2 s, fixed sem vs nosem, same seed (10000), 8 s.
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
B=$1; V=artifacts/runs/legged_fixsem_video/$B; mkdir -p $V; rc=0
export MUJOCO_GL=egl PYTHONPATH=src
E="$PY -m rrp.evaluation.legged_latent_eval --bodies $B --video-dir $V --video-n 1"
FX=artifacts/runs/legged_fixsem_flow_sem_${B}_lv4/snap_s4000.pt; NS=artifacts/runs/legged_flow_nosem_${B}_v2/snap_s4000.pt
$E --seeds ${SEED_FULL:-10017} --max-s 40 --flow $FX --out $V/r2_fixsem_full.jsonl > $V/log1 2>&1 || { echo FAIL full; rc=1; }
$E --seeds 10000 --max-s 8 --edit halt --t-edit 2.0 --flow $FX --out $V/ctxhalt_fixsem.jsonl > $V/log2 2>&1 || { echo FAIL h1; rc=1; }
$E --seeds 10000 --max-s 8 --edit halt --t-edit 2.0 --flow $NS --out $V/ctxhalt_nosem.jsonl > $V/log3 2>&1 || { echo FAIL h2; rc=1; }
$E --seeds 10000 --max-s 8 --edit none --t-edit 2.0 --flow $FX --out $V/none_fixsem.jsonl > $V/log4 2>&1 || { echo FAIL h3; rc=1; }
ls $V; exit $rc
