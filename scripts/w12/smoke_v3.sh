#!/usr/bin/env bash
# W12 / D-126 tiny peer smoke (lowest priority, 1 CPU): teacher v3 vs v2 on a few seeds, and one episode of each
# coordination-task stub. Output: artifacts/runs/w12_smoke/<name>.jsonl (one JSON line per episode).
set -uo pipefail
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
O=artifacts/runs/w12_smoke; mkdir -p $O
export RRP_GRASP_CONTACT=${RRP_GRASP_CONTACT:-v2.1}
for task in support_insert handover; do
  for pair in parm5_pg2__parm5_pg2 panda_pg2__ur5e_pg2; do
    for s in 0 1 2; do
      for v in v3 v2; do
        PYTHONPATH=src $PY scripts/w12/smoke_episode.py $task $pair $s $v >> $O/${task}_${v}.jsonl || echo "{\"error\": \"$task $pair $s $v\"}" >> $O/${task}_${v}.jsonl
      done
    done
  done
done
for task in pivot_against_surface carry_tray_level; do
  PYTHONPATH=src $PY scripts/w12/smoke_episode.py $task parm5_pg2__parm5_pg2 0 stub 900 >> $O/${task}_stub.jsonl || echo "{\"error\": \"$task\"}" >> $O/${task}_stub.jsonl
done
