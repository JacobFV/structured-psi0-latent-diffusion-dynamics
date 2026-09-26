#!/usr/bin/env bash
# LEGGED FIXED-SEM RERUN evals (CPU, one lease; peer). Same scripts/seeds as the original sem runs:
# R2 ladder dev 10000-10029 (snap_s4000 + final), generator gate, probe z-edit suite and task-context edit suite on
# R2 snap_s4000 (20 dev seeds 10000-10019, edit from t=2 s, window 2-5 s), sign-normalized mirror effects.
# usage: legged_fixsem_eval.sh BODY [PAR]
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
BODY=$1; PAR=${2:-6}; rc=0
F=artifacts/runs/legged_fixsem_flow_sem_${BODY}_lv4; R=artifacts/runs/legged_fixsem_rep_sem_${BODY}_lv4
for f in $F/snap_s4000.pt $F/policy.pt $R/representation.pt; do [ -f $f ] || { echo "missing $f"; exit 3; }; done
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY -m rrp.learning.legged_dagger gate --rep $R/representation.pt \
  --flow $F/snap_s4000.pt --buf artifacts/runs/legged_buf/bc_$BODY --body $BODY --out artifacts/runs/legged_gate/${BODY}_fixsem_lv4_gen_snap_s4000.json > /dev/null 2>&1 || { echo FAIL gate; rc=1; }
for ck in snap_s4000 policy; do
  o=$(bash scripts/legged_ladder.sh $BODY fixsem_${BODY}_lv4_$ck $PAR r2 - - - $F/$ck.pt 2>&1); echo "$o"; grep -q FAIL <<<"$o" && rc=1
done
o=$(bash scripts/legged_edit_suite.sh $BODY r2_fixsem_snap_s4000 $PAR "--flow $F/snap_s4000.pt" 2>&1); echo "$o"; grep -q FAIL <<<"$o" && rc=1
o=$(EDITS="none mirror_goal mirror_active mirror_inactive halt" bash scripts/legged_edit_suite.sh $BODY r2ctx_fixsem_snap_s4000 $PAR "--flow $F/snap_s4000.pt" 2>&1); echo "$o"; grep -q FAIL <<<"$o" && rc=1
E=artifacts/runs/legged_edits/$BODY
PYTHONPATH=src $PY scripts/legged_mirror_effect.py $E/r2_fixsem_snap_s4000 probe_goal_mirror rand_norm_8 || rc=1
PYTHONPATH=src $PY scripts/legged_mirror_effect.py $E/r2ctx_fixsem_snap_s4000 mirror_goal mirror_active mirror_inactive halt || rc=1
for n in r2_fixsem_${BODY}_lv4_snap_s4000 r2_fixsem_${BODY}_lv4_policy; do
  [ "$(wc -l < artifacts/runs/legged_ladder/$BODY/$n.jsonl)" = 30 ] || { echo "short $n"; rc=1; }; done
exit $rc
