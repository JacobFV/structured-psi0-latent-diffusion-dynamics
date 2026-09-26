#!/usr/bin/env bash
# offline gate for sem + nosem reps. usage: legged_gate_pair.sh TAG CKPTNAME BODY DATA [BUF] [RZ_SUFFIX]
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
TAG=$1; CK=$2; BODY=$3; DATA=$4; BUF=${5:-}
for v in sem nosem; do
  A="--rep artifacts/runs/legged_rep_${v}_$TAG/$CK --teacher-data $DATA --body $BODY"
  [ -n "$BUF" ] && A="$A --buf $BUF"
  CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY -m rrp.learning.legged_dagger gate $A --out artifacts/runs/legged_gate/${BODY}_${v}_${TAG}_${CK%.pt}$( [ -n "$BUF" ] && echo _$(basename $(dirname $BUF/x)) ).json &
done
wait
