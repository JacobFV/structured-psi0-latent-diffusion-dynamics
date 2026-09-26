#!/usr/bin/env bash
# T1 DIAGNOSIS clip: the bounded-NLL sem route (deployable R2) on the same dev seed as the original sem fall (10001).
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
V=artifacts/runs/t1_diag/video_lv4; mkdir -p $V
MUJOCO_GL=egl PYTHONPATH=src $PY -m rrp.evaluation.legged_latent_eval --bodies t1 --seeds ${SEED:-10001} --video-dir $V --video-n 1 --max-s 40 --flow artifacts/runs/t1diag_flow_sem_lv4/policy.pt --out $V/r2_sem_lv4.jsonl > /dev/null 2>&1
