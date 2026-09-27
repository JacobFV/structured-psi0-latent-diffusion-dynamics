#!/usr/bin/env bash
# D-112 gate for a dataset collected before per-episode motion metrics existed: re-collect the same seeds (deterministic
# per seed) with the current collector (motion recorder attached, read-only) into a scratch dir, in parallel shards, then
# scripts/legged8_dataset_gate.py compares every array with the stored shards and gates the replay's slip metrics.
# usage: legged8_gate_replay.sh BODY STORED_DATA_DIR SIGMAS FIRST LAST SHARD PAR
set -uo pipefail
PY=${PY:-python}; B=$1; REF=$2; SIG=$3; A=$4; Z=$5; SH=$6; PAR=$7
OUT=artifacts/runs/legged8/diag/replay_$B; rm -rf $OUT; mkdir -p $OUT/$B
export RRP_CONTACT_MODEL=contact_v2 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src
for ((s=A; s<=Z; s+=SH)); do e=$((s+SH-1)); echo "$s $e"; done | xargs -P $PAR -L 1 bash -c \
  "$PY -m rrp.data.legged_latent_collect --body $B --seeds \$0-\$1 --sigmas $SIG --out $OUT > $OUT/$B/log_\$0.txt 2>&1 || echo FAIL \$0"
$PY scripts/legged8_dataset_gate.py artifacts/runs/legged8/diag/gate_$B.json $OUT $B $REF
