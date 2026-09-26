#!/usr/bin/env bash
# Parallel DAgger/gate buffer collection (CPU; one lease). usage: legged_collect.sh OUT PAR FIRST LAST ROUTE BODY BC [REP] [RZ|-] [FLOW|-]
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
OUT=$1; PAR=$2; FIRST=$3; LAST=$4; ROUTE=$5; BODY=$6; BC=$7; REP=${8:-}; RZ=${9:--}; FLOW=${10:--}
A="--route $ROUTE --body $BODY --bc $BC"; [ -n "$REP" ] && A="$A --rep $REP"; [ "$RZ" != "-" ] && A="$A --realizer $RZ"; [ "$FLOW" != "-" ] && A="$A --flow $FLOW"
n=$(( (LAST - FIRST + 1 + PAR - 1) / PAR )); cmds=()
for ((s=FIRST; s<=LAST; s+=n)); do e=$((s+n-1)); [ $e -gt $LAST ] && e=$LAST; cmds+=("$PY -m rrp.learning.legged_dagger collect $A --seeds $s-$e --out $OUT"); done
printf '%s\n' "${cmds[@]}" | xargs -P "$PAR" -I{} bash -c 'OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src {} 2>&1 | tail -1'
