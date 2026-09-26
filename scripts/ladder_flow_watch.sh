#!/usr/bin/env bash
# Learning curve for a system-i flow trained on the HOST; R2 evals on PEER CPU leases: every EVERY steps, snapshot policy_last.pt and run
# R2 (generated route) on panda_pg2 + parm6_tf3, 30 matched dev seeds, prev-action zero, with the given system-0 bundle.
# Usage (host worktree root): ladder_flow_watch.sh <flow_run_dir> <rep.pt> <tag> [EVERY=4000]
set -uo pipefail
D=$1; REP=$2; TAG=$3; EVERY=${4:-4000}
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
export PYTHONPATH=src
next=$EVERY
ev() {  # $1 snapshot, $2 label. Eval runs on the PEER (host broker is usually full): copy the snapshot there first.
  PD=/dev/shm/rrp-brandonin/repo/$(dirname $1)
  ssh gb10-direct "mkdir -p $PD" && rsync -a $1 gb10-direct:$PD/ || return
  RRP_PEER_REPO=${RRP_PEER_REPO:-/dev/shm/rrp-brandonin/wt/ladder} scripts/peer_run.sh --cpu 3 --mem 8G \
    --label ladder_r2_$2 --max-seconds 7200 -- bash -c "export CUDA_VISIBLE_DEVICES=
    for r in panda_pg2 parm6_tf3; do
      /dev/shm/rrp-brandonin/venv/bin/python scripts/ladder.py --route generated --flow $1 --rep $REP --prev-action zero --robot \$r --n 30 --tag zero_$2 \
        --out artifacts/runs/ladder_v1/\$r
    done" > $D/eval_$2.out 2>&1 &
}
while true; do
  if [ -f $D/policy.pt ]; then
    s=$(python3 -c "import json;print(json.load(open('$D/policy.json'))['step'])")
    cp $D/policy.pt $D/snap_final_s$s.pt; ev $D/snap_final_s$s.pt ${TAG}_s$s; break
  fi
  s=$(python3 -c "import json;print(json.load(open('$D/policy_last.json'))['step'])" 2>/dev/null || echo 0)
  if [ "$s" -ge "$next" ]; then
    cp $D/policy_last.pt $D/snap_s$s.pt; ev $D/snap_s$s.pt ${TAG}_s$s
    next=$(( (s / EVERY + 1) * EVERY ))
  fi
  sleep 60
done
wait
echo watch-done
