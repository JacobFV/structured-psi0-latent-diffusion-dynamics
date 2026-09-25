#!/usr/bin/env bash
set -uo pipefail
cd /dev/shm/rrp-brandonin/repo
export PATH=/dev/shm/rrp-brandonin/bin:$PATH PYTHONPATH=src RRP_NODE=peer RRP_REPO=$PWD
PY=/dev/shm/rrp-brandonin/venv/bin/python
ROB=panda_pg2,parm6_tf3,parm5s_tf3,parm5l_pg2
for n in latent_sem_v1 latent_nosem_v1; do
  until test -f artifacts/runs/flow_$n/policy.pt; do sleep 60; done
  python3 -m rrp.cli ops run --gpu --gpu-mem 8G --cpu 6 --mem 16G --label eval_flow_$n --max-seconds 10800 --detach -- \
    $PY -m rrp.cli latent evaluate --checkpoint artifacts/runs/flow_$n/policy.pt --robots $ROB --episodes 20 \
      --seed-start 3000000 --method $n --batch 20 --out artifacts/runs/flow_$n/eval_dev.jsonl
  python3 -m rrp.cli ops run --gpu --gpu-mem 6G --cpu 2 --mem 12G --label disturb_$n --max-seconds 3600 --detach -- \
    $PY -m rrp.cli latent disturbance --checkpoint artifacts/runs/flow_$n/policy.pt --robots panda_pg2 --episodes 10 \
      --seed-start 3000100 --out artifacts/runs/flow_$n/disturbance.jsonl
done
echo eval-chain-launched
