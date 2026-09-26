#!/usr/bin/env bash
# Full ladder evaluation of a system-0 candidate on the jointfix latent space (CPU): offline gates (oracle + generated
# packet at BC states), R1 stateless oracle (BC expert) and R2 generated (flow_jointfix final), panda_pg2 + parm6_tf3,
# 30 matched dev seeds. Usage: [PY=..] [FLOW=..] ladder_eval_all.sh <rep.pt> <tag>
set -uo pipefail
REP=$1; TAG=$2
export PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python} CUDA_VISIBLE_DEVICES=
FLOW=${FLOW:-artifacts/runs/ladder_flow_jointfix/snap_final_s20000.pt}
FT=$(basename $(dirname $FLOW))_$(basename $FLOW .pt)
for r in parm6_tf3 panda_pg2; do
  $PY scripts/ladder.py --route generated --flow $FLOW --rep $REP --prev-action zero --robot $r --n 30 \
    --tag zero_${FT}_rz$TAG --out artifacts/runs/ladder_v1/$r
done
FLOW=$FLOW bash scripts/ladder_gate.sh $REP ${TAG}__$FT
bash scripts/ladder_eval_orcbc.sh $REP $TAG
