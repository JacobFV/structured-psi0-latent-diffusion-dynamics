#!/usr/bin/env bash
# Host-side supervisor: runs scripts/baselines_host_queue.sh under a host broker lease in <=6 h segments until the
# codec slot's done marker exists; retries admission every 10 min; after each segment, syncs the direct seed1703
# source model to the peer store and clears its REMOTE_TRAINING marker there.
set -uo pipefail
cd "$(dirname "$0")/.."
PYH=${PYH:-$HOME/work/relational-robot-policy/.venv/bin/python}
ROOT=artifacts/runs/latent_slice1
D=$ROOT/baseline_direct_action/seed1703/source
P=/dev/shm/rrp-brandonin/wt/baselines/$D
sync_direct() {
  if [ -s $D/policy.pt ] && ssh gb10-direct "test -e $P/REMOTE_TRAINING"; then
    rsync -a $D/policy.pt $D/policy.json $D/result.json $D/config.json $D/train_log.jsonl gb10-direct:$P/ \
      && ssh gb10-direct "rm -f $P/REMOTE_TRAINING" && echo "$(date -Is) synced $D to peer"
  fi
}
for seg in $(seq 1 200); do
  sync_direct
  [ -e "$ROOT/.slot_done_baseline_action_only_codec_1701_1702_1703" ] && { echo "$(date -Is) done"; exit 0; }
  out=$(PYTHONPATH=src $PYH -m rrp.cli ops run --gpu --gpu-mem 20G --cpu 4 --mem 16G --label baselines_host_slot \
        --max-seconds 21000 -- env PY=$PYH bash scripts/baselines_host_queue.sh 2>&1)
  echo "$out" | grep -E '^\[slot\]|lease_id|FAILED|Error' | tail -30
  if echo "$out" | grep -qE 'CapacityError|AdmissionStopped'; then echo "$(date -Is) not admitted; retry in 10 min"; sleep 600; else sleep 30; fi
done
