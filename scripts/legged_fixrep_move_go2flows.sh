#!/usr/bin/env bash
# One-off: after the 3 peer go2 chains finish Stage A (+ nosem probe), stop peer lease 1790440106_9ac1a9, discard the just-started
# peer flows, copy the Stage-A dirs to the host and train the 3 go2 flows from scratch on the host GPU (one lease).
set -uo pipefail
T="fixsem_go2_s1 fixsem_go2_s2 nosem_go2_s2"; PA=/dev/shm/rrp-brandonin/repo/artifacts/runs
until ssh gb10-direct "cd $PA && for t in $T; do test -f legged_fixrep_flow_\$t/train_log.jsonl -o -d legged_fixrep_flow_\$t || exit 1; done; test -f legged_fixrep_rep_nosem_go2_s2/probe_posthoc.pt"; do sleep 20; done
ssh gb10-direct "cd /dev/shm/rrp-brandonin/repo && PATH=/dev/shm/rrp-brandonin/bin:\$PATH PYTHONPATH=src RRP_NODE=peer RRP_OPS_ROOT=/dev/shm/rrp-brandonin/repo python3 -m rrp.cli ops stop --owned-only --lease 1790440106_9ac1a9" || exit 1
sleep 15; ssh gb10-direct "cd $PA && rm -rf $(for t in $T; do printf 'legged_fixrep_flow_%s legged_fixrep_flow_%s.out ' $t $t; done)"
for t in $T; do mkdir -p artifacts/runs/legged_fixrep_rep_$t; rsync -a gb10-direct:$PA/legged_fixrep_rep_$t/ artifacts/runs/legged_fixrep_rep_$t/ --exclude 'rep_last.pt' --exclude 'snap_*' || exit 1
  [ -f artifacts/runs/legged_fixrep_rep_$t/representation.pt ] || exit 1; done
PYTHONPATH=src ~/work/relational-robot-policy/.venv/bin/python -m rrp.cli ops run --cpu 3 --mem 8G --gpu --gpu-mem 5G --label legged_fixrep_flows_go2h --max-seconds 10800 --detach -- bash scripts/legged_fixrep_train.sh $T
