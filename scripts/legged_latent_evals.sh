#!/usr/bin/env bash
# Closed-loop evaluations for one legged latent flow (CPU lease). usage: legged_latent_evals.sh RUN_NAME [PAR]
set -uo pipefail
PY=/dev/shm/rrp-brandonin/venv/bin/python
RUN=/dev/shm/rrp-brandonin/repo/artifacts/runs/$1
PAR=${2:-6}
F=$RUN/policy.pt
B6=go2,pquad4,hexapod6,sprawl4,sprawl8,hexapod6_long
E="$PY -m rrp.evaluation.legged_latent_eval --flow $F"
cmds=()
for b in ${B6//,/ }; do cmds+=("$E --bodies $b --seeds 10000-10019 --out $RUN/eval_dev_$b.jsonl"); done
for ed in mirror_goal halt probe_yaw:0.6 probe_yaw:-0.6 zero; do
  for b in go2 pquad4; do cmds+=("$E --bodies $b --seeds 10000-10009 --edit $ed --t-edit 1.0 --out $RUN/edit_${ed/:/}_$b.jsonl"); done; done
printf '%s\n' "${cmds[@]}" | xargs -P "$PAR" -I{} bash -c 'OMP_NUM_THREADS=1 {} > /dev/null 2>&1; echo done: {} | cut -c1-200'
