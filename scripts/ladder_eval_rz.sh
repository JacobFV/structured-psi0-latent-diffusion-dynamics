#!/usr/bin/env bash
# Evaluate a system-0 candidate on the ladder (CPU): shadow check on the teacher trajectory, R1 (plain and re-anchored
# expert) on panda_pg2 and parm6_tf3, 30 matched dev seeds each, prev-action input = zero (deployment).
# Usage (peer dir): ladder_eval_rz.sh <rep.pt> <tag> [robots...]
set -uo pipefail
REP=$1; TAG=$2; shift 2
ROBOTS=${*:-panda_pg2 parm6_tf3}
PY=/dev/shm/rrp-brandonin/venv/bin/python
export CUDA_VISIBLE_DEVICES=
sha256sum $REP
for r in $ROBOTS; do
  O=artifacts/runs/ladder_v1/$r
  $PY scripts/ladder.py --route teacher --tag shadow_zero_$TAG --prev-action zero --robot $r --n 8 --rep $REP --out $O
  $PY scripts/ladder.py --route oracle --tag zero_${TAG}_reanchor --reanchor --prev-action zero --robot $r --n 30 --rep $REP --out $O
  $PY scripts/ladder.py --route oracle --tag zero_$TAG --prev-action zero --robot $r --n 30 --rep $REP --out $O
done
