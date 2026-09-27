#!/usr/bin/env bash
# Context-halt clips (EGL; GPU lease): for each (BODY, training seed, eval seed) render the fixed-sem and nosem R2 routes (snap_s4000),
# unedited and with the task-view `halt` context edit from t=2 s (8 s cap), then tile them 2x2 (scripts/t1_edits_tile.py).
# usage: t1_edits_videos.sh BODY:TRAINSEED:EVALSEED:KIND ...   (KIND is a filename word, e.g. effect / noeffect)
# Models: artifacts/runs/legged_fixrep_flow_{fixsem,nosem}_<body>_s<k> (k=1,2); t1 uses the t1_edits_eval.sh names.
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
export MUJOCO_GL=egl PYTHONPATH=src; rc=0
for spec in "$@"; do IFS=: read B K S KIND <<<"$spec"
  V=artifacts/runs/t1_edits_video/${B}_s${K}_${S}; mkdir -p $V
  if [ $B = t1 ]; then
    case $K in 0) FX=t1diag_flow_sem_lv4; NS=legged_flow_nosem_t1_v2;; *) FX=t1diag_flow_sem_lv4_s$K; NS=legged_flow_nosem_t1_v2s$K;; esac
    CK=${T1CK:-snap_s4000}
  else FX=legged_fixrep_flow_fixsem_${B}_s$K; NS=legged_fixrep_flow_nosem_${B}_s$K; CK=snap_s4000; fi
  for v in fixsem nosem; do F=$FX; [ $v = nosem ] && F=$NS
    for ed in none halt; do
      $PY -m rrp.evaluation.legged_latent_eval --bodies $B --video-dir $V/$v-$ed --video-n 1 --seeds $S --max-s 8 --edit $ed --t-edit 2.0 \
        --flow artifacts/runs/$F/$CK.pt --out $V/$v-$ed.jsonl > $V/$v-$ed.log 2>&1 || { echo "FAIL $spec $v $ed"; rc=1; }
    done; done
  $PY scripts/t1_edits_tile.py $V $B $K $S $KIND $V || rc=1
done
exit $rc
