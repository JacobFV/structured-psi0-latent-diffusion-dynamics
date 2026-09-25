#!/usr/bin/env bash
# Stage B v2 (standardized flow target, resumable) -> closed-loop eval -> disturbance -> videos.
# SERIAL: each step is a foreground lease, so at most one GPU job of this chain runs at a time.
# Idempotent: re-running resumes training from policy_last.pt and skips finished steps.
# Usage: latent_chain_v2.sh [run names...]   (default: latent_sem_v2 latent_nosem_v2)
set -uo pipefail
cd /dev/shm/rrp-brandonin/repo
export PATH=/dev/shm/rrp-brandonin/bin:$PATH PYTHONPATH=src RRP_NODE=peer RRP_REPO=$PWD
PY=/dev/shm/rrp-brandonin/venv/bin/python
ROB=panda_pg2,parm6_tf3,parm5s_tf3,parm5l_pg2      # source/development bodies only (D-025)
run() { python3 -m rrp.cli ops run "$@"; }
NAMES=${*:-latent_sem_v2 latent_nosem_v2}
for n in $NAMES; do
  R=artifacts/runs/flow_$n
  [ -f $R/policy.pt ] || run --gpu --gpu-mem 20G --cpu 4 --mem 24G --label flow_$n --max-seconds 21600 -- \
    $PY -m rrp.cli latent train-flow --config configs/latent/flow_$n.json
  [ -f $R/policy.pt ] || { echo "flow_$n did not finish; stopping chain"; exit 1; }
done
for n in $NAMES; do
  R=artifacts/runs/flow_$n
  if [ ! -f $R/eval_dev.done ]; then
    rm -f $R/eval_dev.jsonl
    run --gpu --gpu-mem 8G --cpu 6 --mem 16G --label eval_flow_$n --max-seconds 10800 -- \
      $PY -m rrp.cli latent evaluate --checkpoint $R/policy.pt --robots $ROB --episodes 20 \
        --seed-start 3000000 --method flow_$n --batch 20 --out $R/eval_dev.jsonl > $R/eval_dev.summary.txt 2>&1 \
      && touch $R/eval_dev.done
  fi
  if [ ! -f $R/disturbance.done ]; then
    run --gpu --gpu-mem 6G --cpu 2 --mem 12G --label disturb_$n --max-seconds 3600 -- \
      $PY -m rrp.cli latent disturbance --checkpoint $R/policy.pt --robots panda_pg2 --episodes 10 \
        --seed-start 3000100 --out $R/disturbance.jsonl > $R/disturbance.summary.txt 2>&1 && touch $R/disturbance.done
  fi
  if [ ! -f $R/video.done ]; then
    run --gpu --gpu-mem 4G --cpu 2 --mem 8G --label video_$n --max-seconds 1800 -- \
      $PY scripts/render_episode.py --robot panda_pg2 --seeds 3000001,3000002,3000004 --source learned_latent \
        --checkpoint $R/policy.pt --out artifacts/video && touch $R/video.done
  fi
done
echo chain-v2-done
