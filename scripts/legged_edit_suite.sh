#!/usr/bin/env bash
# Causal packet edits with irrelevant-edit controls (CPU; one lease). Every condition runs on the same seeds; edits start at
# T_EDIT and apply to every later packet; episodes stop at MAXS (effects are measured in the window).
# usage: legged_edit_suite.sh BODY TAG PAR ROUTEARGS(quoted) [SEEDS] [T_EDIT] [MAXS]
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
BODY=$1; TAG=$2; PAR=$3; RA=$4; SEEDS=${5:-10000-10019}; TE=${6:-2.0}; MX=${7:-5.0}
OUT=artifacts/runs/legged_edits/$BODY/$TAG; mkdir -p $OUT
cmds=()
EDITS=${EDITS:-none probe_yaw:0.6 probe_yaw:-0.6 probe_goal_mirror probe_halt contact:0:1 contact:0:0 rand_norm:1 rand_norm:2 rand_norm:4 rand_norm:8 rand_norm:12 zero}
for ed in $EDITS; do
  cmds+=("$PY -m rrp.evaluation.legged_latent_eval $RA --bodies $BODY --seeds $SEEDS --edit $ed --t-edit $TE --max-s $MX --out $OUT/${ed//:/_}.jsonl")
done
for ed in $EDITS; do rm -f $OUT/${ed//:/_}.jsonl; done
printf '%s\n' "${cmds[@]}" | xargs -P "$PAR" -I{} bash -c 'OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src {} > /dev/null 2>&1 || echo FAIL: {}'
PYTHONPATH=src $PY scripts/legged_edit_effects.py $OUT
