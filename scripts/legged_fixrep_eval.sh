#!/usr/bin/env bash
# LEGGED FIXED-SEM REPLICATION evals (CPU, one lease). Same suites/seeds as legged_fixsem_eval.sh, plus random |dz| 16/25.
# usage: legged_fixrep_eval.sh BODY PAR TAG...   (TAG = {fixsem,nosem}_<body>_s{1,2})
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
BODY=$1; PAR=$2; shift 2; rc=0
ZE="none probe_yaw:0.6 probe_yaw:-0.6 probe_goal_mirror probe_halt contact:0:1 contact:0:0 rand_norm:1 rand_norm:2 rand_norm:4 rand_norm:8 rand_norm:12 rand_norm:16 rand_norm:25 zero"
CE="none mirror_goal mirror_active mirror_inactive halt"
run() { o=$("$@" 2>&1); echo "$o"; grep -q FAIL <<<"$o" && rc=1; }
for t in "$@"; do
  F=artifacts/runs/legged_fixrep_flow_$t; R=artifacts/runs/legged_fixrep_rep_$t
  [ -f $F/snap_s4000.pt ] && [ -f $F/policy.pt ] || { echo "missing $F"; rc=1; continue; }
  PP=""; [[ $t == nosem_* ]] && PP="--posthoc-probe $R/probe_posthoc.pt"
  for ck in snap_s4000 policy; do run bash scripts/legged_ladder.sh $BODY fixrep_${t}_$ck $PAR r2 - - - $F/$ck.pt; done
  run env EDITS="$ZE" bash scripts/legged_edit_suite.sh $BODY r2_fixrep_${t}_snap_s4000 $PAR "--flow $F/snap_s4000.pt $PP"
  run env EDITS="$CE" bash scripts/legged_edit_suite.sh $BODY r2ctx_fixrep_${t}_snap_s4000 $PAR "--flow $F/snap_s4000.pt $PP"
  E=artifacts/runs/legged_edits/$BODY
  PYTHONPATH=src $PY scripts/legged_mirror_effect.py $E/r2_fixrep_${t}_snap_s4000 probe_goal_mirror rand_norm_8 || rc=1
  PYTHONPATH=src $PY scripts/legged_mirror_effect.py $E/r2ctx_fixrep_${t}_snap_s4000 mirror_goal mirror_active mirror_inactive halt || rc=1
  for ck in snap_s4000 policy; do [ "$(wc -l < artifacts/runs/legged_ladder/$BODY/r2_fixrep_${t}_$ck.jsonl)" = 30 ] || { echo "short $t $ck"; rc=1; }; done
done
exit $rc
