#!/usr/bin/env bash
# System-0 DAgger collection: R1 (re-anchored oracle) rollouts of the CURRENT realizer on source-train bodies (SEED env: start),
# seeds 3,200,000+ (disjoint from dev eval 3,000,000+ and GRPO train 3,100,000+). Usage (peer dir):
#   ladder_dagger_collect.sh <rep.pt> <out_dir> <n_eps> robot1 robot2 ...
set -uo pipefail
REP=$1; OUT=$2; N=$3; shift 3
PY=/dev/shm/rrp-brandonin/venv/bin/python
export CUDA_VISIBLE_DEVICES=
mkdir -p $OUT
for r in "$@"; do
  [ -f $OUT/$r.npz ] && continue
  $PY scripts/ladder.py --route oracle --reanchor --prev-action zero --robot $r --n $N --seed-start ${SEED:-3200000} \
    --rep $REP --out $OUT --tag $r --no-compare --collect-dagger $OUT/$r.npz || echo "FAILED $r"
done
