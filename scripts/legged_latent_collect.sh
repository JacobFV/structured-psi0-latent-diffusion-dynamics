#!/usr/bin/env bash
# Parallel tick-level legged teacher collection for the latent path (run inside ONE leased job).
# usage: scripts/legged_latent_collect.sh OUT PAR "body:tracker ..." FIRST LAST SHARD
set -euo pipefail
OUT=$1; PAR=$2; BODIES=$3; FIRST=$4; LAST=$5; SHARD=${6:-100}
PY=${PY:-python}
jobs=()
for bt in $BODIES; do b=${bt%%:*}; t=${bt##*:}
  for ((s=FIRST; s<=LAST; s+=SHARD)); do e=$((s+SHARD-1)); [ $e -gt $LAST ] && e=$LAST
    jobs+=("$b $t $s $e"); done; done
printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -L 1 bash -c 'OMP_NUM_THREADS=1 '"$PY"' -m rrp.data.legged_latent_collect --body $0 --tracker $1 --seeds $2-$3 --out '"$OUT"' > '"$OUT"'/log_$0_$2.txt 2>&1 && tail -1 '"$OUT"'/log_$0_$2.txt'
