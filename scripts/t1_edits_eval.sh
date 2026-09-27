#!/usr/bin/env bash
# T1 CONTEXT-HALT SUITE: the legged_fixrep eval protocol (scripts/legged_fixrep_eval.sh: same z-edit list incl. random |dz| 16/25,
# same task-context list, same windows t_edit=2 s / max 5 s, dev seeds 10000-10019 for edits and 10000-10029 for R2) applied to the
# EXISTING t1 models of D-087 (no training): fixed sem (bounded NLL, t1diag_*_sem_lv4*) and nosem (legged_*_nosem_t1_v2*),
# training seeds 0/1/3. nosem z edits go through its post-hoc probe (seed 0 existing; seeds 1/3 fitted with the same 4000-step recipe).
# usage: t1_edits_eval.sh PAR CKPT TAG...   (CKPT = snap_s4000 | policy; TAG = {fixsem,nosem}_s{0,1,3})
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
PAR=$1; CK=$2; shift 2; rc=0; BODY=t1
ZE="none probe_yaw:0.6 probe_yaw:-0.6 probe_goal_mirror probe_halt contact:0:1 contact:0:0 rand_norm:1 rand_norm:2 rand_norm:4 rand_norm:8 rand_norm:12 rand_norm:16 rand_norm:25 zero"
CE="none mirror_goal mirror_active mirror_inactive halt"
run() { o=$("$@" 2>&1); echo "$o"; grep -q FAIL <<<"$o" && rc=1; }
paths() { case $1 in
  fixsem_s0) F=t1diag_flow_sem_lv4; R=t1diag_rep_sem_lv4;;
  fixsem_s1) F=t1diag_flow_sem_lv4_s1; R=t1diag_rep_sem_lv4_s1;;
  fixsem_s3) F=t1diag_flow_sem_lv4_s3; R=t1diag_rep_sem_lv4_s3;;
  nosem_s0) F=legged_flow_nosem_t1_v2; R=legged_rep_nosem_t1_v2;;
  nosem_s1) F=legged_flow_nosem_t1_v2s1; R=legged_rep_nosem_t1_v2s1;;
  nosem_s3) F=legged_flow_nosem_t1_v2s3; R=legged_rep_nosem_t1_v2s3;;
  *) return 1;; esac; F=artifacts/runs/$F; R=artifacts/runs/$R; }
for t in "$@"; do
  paths $t || { echo "bad tag $t"; rc=1; continue; }
  [ -f $F/$CK.pt ] || { echo "missing $F/$CK.pt"; rc=1; continue; }
  PP=""; if [[ $t == nosem_* ]]; then [ -f $R/probe_posthoc.pt ] || { echo "missing probe $R"; rc=1; continue; }; PP="--posthoc-probe $R/probe_posthoc.pt"; fi
  T=t1edits_${t}_$CK
  LJ=artifacts/runs/legged_ladder/$BODY/r2_$T.jsonl   # resume: keep a finished R2 file
  [ -f $LJ ] && [ "$(wc -l < $LJ)" = 30 ] || run bash scripts/legged_ladder.sh $BODY $T $PAR r2 - - - $F/$CK.pt
  run env EDITS="$ZE" bash scripts/legged_edit_suite.sh $BODY r2_$T $PAR "--flow $F/$CK.pt $PP"
  run env EDITS="$CE" bash scripts/legged_edit_suite.sh $BODY r2ctx_$T $PAR "--flow $F/$CK.pt $PP"
  E=artifacts/runs/legged_edits/$BODY
  PYTHONPATH=src $PY scripts/legged_mirror_effect.py $E/r2_$T probe_goal_mirror rand_norm_8 || rc=1
  PYTHONPATH=src $PY scripts/legged_mirror_effect.py $E/r2ctx_$T mirror_goal mirror_active mirror_inactive halt || rc=1
  [ "$(wc -l < artifacts/runs/legged_ladder/$BODY/r2_$T.jsonl)" = 30 ] || { echo "short $T"; rc=1; }
done
exit $rc
