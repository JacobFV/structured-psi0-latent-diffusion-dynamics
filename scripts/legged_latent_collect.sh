#!/usr/bin/env bash
# Parallel tick-level legged teacher collection for the latent path (run inside ONE leased job).
# usage: [SIGMAS=0,0.1,0.2,0.3] scripts/legged_latent_collect.sh OUT PAR "body:tracker[:arc] ..." FIRST LAST SHARD
set -euo pipefail
OUT=$1; PAR=$2; BODIES=$3; FIRST=$4; LAST=$5; SHARD=${6:-100}
PY=${PY:-python}; SIGMAS=${SIGMAS:-0,0.1,0.2,0.3}
mkdir -p "$OUT"
jobs=()
for bt in $BODIES; do IFS=: read -r b t x <<< "$bt"; ex=""; [ "${x:-}" = arc ] && ex="--arc-only"
  for ((s=FIRST; s<=LAST; s+=SHARD)); do e=$((s+SHARD-1)); [ $e -gt $LAST ] && e=$LAST
    jobs+=("$b $t $s $e${ex:+ $ex}"); done; done
printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -L 1 bash -c 'OMP_NUM_THREADS=1 '"$PY"' -m rrp.data.legged_latent_collect --body $0 --tracker $1 --seeds $2-$3 --sigmas '"$SIGMAS"' $4 --out '"$OUT"' > '"$OUT"'/log_$0_$2.txt 2>&1 && tail -1 '"$OUT"'/log_$0_$2.txt'
