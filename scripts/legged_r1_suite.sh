#!/usr/bin/env bash
# Offline gate (teacher held-out + BC-visited states) and R1 stateless-oracle ladder (normal + qd zeroed) for sem and nosem.
# usage: legged_r1_suite.sh BODY TAG(rep tag) DATA BUF BC PAR [RZ_SEM|-] [RZ_NOSEM|-] [LADDER_TAG]
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
BODY=$1; T=$2; DATA=$3; BUF=$4; BC=$5; PAR=$6; RZS=${7:--}; RZN=${8:--}; LT=${9:-$T}
for v in sem nosem; do
  RZ=$RZS; [ $v = nosem ] && RZ=$RZN; RA=""; [ "$RZ" != "-" ] && RA="--realizer $RZ"
  OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY -m rrp.learning.legged_dagger gate --rep artifacts/runs/legged_rep_${v}_$T/representation.pt $RA \
     --teacher-data $DATA --buf $BUF --body $BODY --out artifacts/runs/legged_gate/${BODY}_${v}_${LT}.json > /dev/null 2>&1 &
done
wait
for v in sem nosem; do
  RZ=$RZS; [ $v = nosem ] && RZ=$RZN
  bash scripts/legged_ladder.sh $BODY ${v}_$LT $PAR r1,r1qd0 $BC artifacts/runs/legged_rep_${v}_$T/representation.pt $RZ
done
