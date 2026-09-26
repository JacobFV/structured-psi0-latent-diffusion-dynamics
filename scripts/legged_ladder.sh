#!/usr/bin/env bash
# Legged ladder on matched dev seeds (CPU; run inside ONE lease). Routes: teacher | bc | r1 (stateless oracle E(BC chunk)
# -> system 0) | r1qd0 (same, qd zeroed at system 0) | r2 (flow -> system 0).
# usage: legged_ladder.sh BODY TAG PAR ROUTES [BC] [REP] [REALIZER|-] [FLOW|-] [SEEDS]
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
BODY=$1; TAG=$2; PAR=$3; ROUTES=$4; BC=${5:-}; REP=${6:-}; RZ=${7:--}; FLOW=${8:--}; SEEDS=${9:-10000-10029}
OUT=artifacts/runs/legged_ladder/$BODY; mkdir -p $OUT
a=${SEEDS%-*}; b=${SEEDS#*-}; n=$(( (b - a + 1 + PAR - 1) / PAR ))
RZA=""; [ "$RZ" != "-" ] && RZA="--realizer $RZ"
cmds=()
for r in ${ROUTES//,/ }; do
  case $r in
    teacher) A=""; [ "$BODY" = g1 ] && A="--arc-only g1";;
    bc) A="--bc $BC";;
    r1) A="--rep $REP --bc $BC --oracle-bc $RZA";;
    r1qd0) A="--rep $REP --bc $BC --oracle-bc $RZA --zero-qd";;
    r2) A="--flow $FLOW $RZA";;
  esac
  for ((s=a; s<=b; s+=n)); do e=$((s+n-1)); [ $e -gt $b ] && e=$b
    cmds+=("$PY -m rrp.evaluation.legged_latent_eval $A --bodies $BODY --seeds $s-$e --out $OUT/${r}_${TAG}.part$s.jsonl"); done
done
printf '%s\n' "${cmds[@]}" | xargs -P "$PAR" -I{} bash -c 'OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src {} > /dev/null 2>&1 || echo FAIL: {}'
for r in ${ROUTES//,/ }; do cat $OUT/${r}_${TAG}.part*.jsonl > $OUT/${r}_${TAG}.jsonl && rm -f $OUT/${r}_${TAG}.part*; done
PYTHONPATH=src $PY scripts/legged_ladder_summary.py $OUT/*_${TAG}.jsonl
