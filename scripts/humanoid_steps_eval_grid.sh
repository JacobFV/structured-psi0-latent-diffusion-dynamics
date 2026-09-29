#!/usr/bin/env bash
# W13 P2: C-MuJoCo h_steps grid: expert (privileged scan) at h_frac 0.10/0.15/0.20/0.25/0.30 and the blind tracker at 0.10/0.15.
# usage: scripts/humanoid_steps_eval_grid.sh <body> <expert actor> <blind tracker actor> <out dir>
set -uo pipefail
body=$1; ex=$2; bl=$3; out=$4; mkdir -p "$out"
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
export PYTHONPATH=src RRP_CONTACT_MODEL=v2
for hf in 0.10 0.15 0.20 0.25 0.30; do $PY scripts/humanoid_steps_eval.py "$body" "$ex" 7200 20 "$out/expert_h$hf.json" $hf 2>&1 | tail -1; done
for hf in 0.10 0.15; do $PY scripts/humanoid_steps_eval.py "$body" "$bl" 7200 20 "$out/blind_h$hf.json" $hf 2>&1 | tail -1; done
