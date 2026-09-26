#!/usr/bin/env bash
# System-0 DAgger collection: R1 rollouts of the CURRENT realizer on source-train bodies (SEED env: start),
# seeds 3,200,000+ (disjoint from dev eval 3,000,000+ and GRPO train 3,100,000+). Usage (peer dir):
#   [EXPERT=bc BC=<ckpt> BCL=<label>] ladder_dagger_collect.sh <rep.pt> <out_dir> <n_eps> robot1 robot2 ...
# EXPERT=teacher (default): re-anchored shadow teacher packets + teacher relabels (CONFOUNDED off-trajectory, D-050).
# EXPERT=bc: stateless expert, packet = E(BC chunk at the replan state), label = that packet's plan row j.
# EXPERT=bc FLOW=<flow.pt>: route generated (system i's own packets drive the rollout and are stored as z), label = the
#   BC expert's plan row j from the replan state.
set -uo pipefail
REP=$1; OUT=$2; N=$3; shift 3
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
export CUDA_VISIBLE_DEVICES=
if [ "${EXPERT:-teacher}" = bc ]; then
  X="--oracle-expert bc --policy ${BC:-artifacts/runs/baselines_bc_ckpts/direct1701_u12000.pt} --policy-label ${BCL:-direct1701_u12000}"
else
  X="--reanchor"
fi
mkdir -p $OUT
for r in "$@"; do
  [ -f $OUT/$r.npz ] && continue
  if [ -n "${FLOW:-}" ]; then ROUTE="generated --flow $FLOW"; NC=""; else ROUTE=oracle; NC=--no-compare; fi
  $PY scripts/ladder.py --route $ROUTE $X --prev-action zero --robot $r --n $N --seed-start ${SEED:-3200000} \
    --rep $REP --out $OUT --tag $r $NC ${GENCTX:+--collect-gen-ctx} --collect-dagger $OUT/$r.npz || echo "FAILED $r"
done
