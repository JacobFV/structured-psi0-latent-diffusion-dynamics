#!/usr/bin/env bash
# R2 (deployable: flow -> system 0) ladder for sem + nosem, plus the generator-gap gate. usage: BODY TAG CK PAR
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
BODY=$1; T=$2; CK=$3; PAR=$4
for v in sem nosem; do
  F=artifacts/runs/legged_flow_${v}_$T/$CK
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY -m rrp.learning.legged_dagger gate --rep artifacts/runs/legged_rep_${v}_$T/representation.pt --flow $F --buf artifacts/runs/legged_buf/bc_$BODY --body $BODY --out artifacts/runs/legged_gate/${BODY}_${v}_${T}_gen_${CK%.pt}.json > /dev/null 2>&1 &
  bash scripts/legged_ladder.sh $BODY ${v}_${T}_${CK%.pt} $PAR r2 - - - $F
done
wait
