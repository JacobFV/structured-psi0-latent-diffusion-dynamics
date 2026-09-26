#!/usr/bin/env bash
# T1 DIAGNOSIS clips, training seed 0, dev seed 10001: R2 sem (fall) vs R2 nosem (success); teacher-encoded packets
# (PRIVILEGED ORACLE, diagnostic) sem (fall) vs nosem (success).
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
V=artifacts/runs/t1_diag/video; mkdir -p $V
export MUJOCO_GL=egl PYTHONPATH=src
E="$PY -m rrp.evaluation.legged_latent_eval --bodies t1 --seeds ${SEED:-10001} --video-dir $V --video-n 1 --max-s 40"
rc=0
for v in sem nosem; do
  $E --flow artifacts/runs/legged_flow_${v}_t1_v2/policy.pt --out $V/r2_${v}.jsonl > /dev/null 2>&1 || { echo FAIL r2 $v; rc=1; }
  $E --rep artifacts/runs/legged_rep_${v}_t1_v2/representation.pt --oracle --out $V/r1t_${v}.jsonl > /dev/null 2>&1 || { echo FAIL r1t $v; rc=1; }
done
ls $V; exit $rc
