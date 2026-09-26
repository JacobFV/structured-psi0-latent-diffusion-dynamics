#!/usr/bin/env bash
set -uo pipefail
PY=${PY:-python}; O=artifacts/runs/legged_ladder/g1; mkdir -p $O
for rp in 1 2; do for s in 10000 10005; do
  OMP_NUM_THREADS=1 PYTHONPATH=src $PY -m rrp.evaluation.legged_latent_eval --bc artifacts/runs/legged_bc_g1_v1/policy.pt --replan $rp --bodies g1 --seeds $s-$((s+4)) --out $O/bc_replan${rp}.part$s.jsonl > /dev/null 2>&1 &
done; done; wait
for rp in 1 2; do cat $O/bc_replan${rp}.part*.jsonl > $O/bc_replan${rp}.jsonl; rm -f $O/bc_replan${rp}.part*; done
PYTHONPATH=src $PY scripts/legged_ladder_summary.py $O/bc_replan*.jsonl
