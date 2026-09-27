#!/usr/bin/env bash
# W8 context-halt clips (EGL; small GPU lease on the peer): fixed sem vs nosem, unedited vs task-view `halt` from t=2 s, 8 s cap,
# for one (body, training seed, eval seed), then tiled by scripts/t1_edits_tile.py.
# usage: legged8_videos.sh BODY TRAINSEED EVALSEED KIND(effect|noeffect) [LINEAGE_PREFIX=legged8]
set -uo pipefail
PY=${PY:-python}; B=$1; K=$2; S=$3; KIND=$4; PRE=${5:-legged8}
V=artifacts/runs/legged8/video/${B}_s${K}_${S}; mkdir -p $V; rc=0
export MUJOCO_GL=egl PYTHONPATH=src RRP_CONTACT_MODEL=contact_v2
for v in semfix nosem; do
  F=artifacts/runs/legged8/$PRE-$B-$v/train_flow_s$K/snap_s4000.pt
  PP=""; [ $v = nosem ] && PP="--posthoc-probe artifacts/runs/legged8/$PRE-$B-nosem/probes_s$K/probe_posthoc.pt"
  lv=$([ $v = semfix ] && echo fixsem || echo nosem)
  for ed in none halt; do
    rm -rf $V/$lv-$ed; mkdir -p $V/$lv-$ed
    $PY -m rrp.evaluation.legged_latent_eval --flow $F $PP --bodies $B --seeds $S --edit $ed --t-edit 2.0 --max-s 8 \
      --video-dir $V/$lv-$ed --video-n 1 --out $V/$lv-$ed.jsonl > $V/$lv-$ed.log 2>&1 || { echo "FAIL $v $ed"; rc=1; }
  done
done
[ $rc = 0 ] && TILE_TRACK="W8 legged8, contact_v2" TILE_NOTE="Physics contact_v2 (ideal PD actuators); gait source $(python3 -c "import json;print(json.loads(open('$V/fixsem-none.jsonl').readline())['tracker'])") ; the packet route replaces the tracker at run time (system 0)." \
  $PY scripts/t1_edits_tile.py $V $B $K $S $KIND artifacts/runs/legged8/video || rc=1
exit $rc
