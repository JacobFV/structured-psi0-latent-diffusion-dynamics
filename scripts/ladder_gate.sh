#!/usr/bin/env bash
# Offline gate for a system-0 candidate (peer, CPU): stateless localization on BC-visited states, panda_pg2 + parm6_tf3,
# 30 matched dev seeds. Gate metric: sys0 arm error / hold-still arm error (target <= 0.20). Usage: [FLOW=<flow.pt>] ladder_gate.sh <rep.pt> <tag>
# FLOW: also system 0's error from the GENERATED packet at the same states (gen gate, target <= ~0.5).
set -uo pipefail
REP=$1; TAG=$2
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
export CUDA_VISIBLE_DEVICES=
for r in panda_pg2 parm6_tf3; do
  $PY scripts/ladder_localize.py --policy artifacts/runs/baselines_bc_ckpts/direct1701_u12000.pt --policy-label direct1701_u12000 \
    --rep $REP ${FLOW:+--flow $FLOW} --robot $r --n 30 --out artifacts/runs/ladder_localize/$r/bc_direct1701_u12000__$TAG.json
done
