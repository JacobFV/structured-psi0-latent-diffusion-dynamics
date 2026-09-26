#!/usr/bin/env bash
# R1 with a STATELESS expert (ORACLE DIAGNOSTIC): packet = E(chunk the learned BC policy emits at the current state),
# system 0 executes; matched dev seeds, prev-action input zero. Usage (peer dir): ladder_eval_orcbc.sh <rep.pt> <tag> [robots...]
set -uo pipefail
REP=$1; TAG=$2; shift 2
ROBOTS=${*:-panda_pg2 parm6_tf3}
BC=${BC:-artifacts/runs/baselines_bc_ckpts/direct1701_u12000.pt}; BCL=${BCL:-direct1701_u12000}
PY=/dev/shm/rrp-brandonin/venv/bin/python
export CUDA_VISIBLE_DEVICES=
sha256sum $REP
for r in $ROBOTS; do
  $PY scripts/ladder.py --route oracle --oracle-expert bc --policy $BC --policy-label $BCL --tag zero_${TAG}_orcbc \
    --prev-action zero --robot $r --n 30 --rep $REP --out artifacts/runs/ladder_v1/$r
done
