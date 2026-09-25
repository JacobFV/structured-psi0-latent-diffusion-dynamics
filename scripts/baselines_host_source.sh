#!/usr/bin/env bash
# Host slot: train one baseline SOURCE model (train-only) in <=6 h resumable segments, then rsync it into the peer
# campaign store (the peer slot defers that seed while source/REMOTE_TRAINING exists).
# Admission is retried every 10 min; if the host never admits the job within 2 h, the marker is removed so the peer
# slot trains it instead. Usage: scripts/baselines_host_source.sh <method> <seed>
cd "$(dirname "$0")/.."
PYH=${PYH:-$HOME/work/relational-robot-policy/.venv/bin/python}
m=$1; s=$2; d=artifacts/runs/latent_slice1/$m/seed$s
P=/dev/shm/rrp-brandonin/wt/baselines/$d/source
ssh gb10-direct "mkdir -p $P && test -s $P/policy.pt || echo 'source trained on host (baselines track)' > $P/REMOTE_TRAINING"
refused=0; started=0
for seg in $(seq 1 40); do
  [ -s $d/source/policy.pt ] && break
  out=$(PYTHONPATH=src $PYH -m rrp.cli ops run --cpu 3 --mem 12G --gpu --gpu-mem 8G --label "baselines_host_src_${m#baseline_}_$s" \
        --max-seconds 21000 -- $PYH -m rrp.cli campaign baseline-cell --method "$m" --seed "$s" --target source --train-only 2>&1 \
        | grep -v Warn | tail -3)
  echo "$(date -Is) seg $seg: $out"
  if echo "$out" | grep -q CapacityError; then
    refused=$((refused+1))
    if [ $started = 0 ] && [ $refused -ge 12 ]; then ssh gb10-direct "rm -f $P/REMOTE_TRAINING"; echo "gave up; peer trains it"; exit 0; fi
    sleep 600
  else started=1; sleep 30; fi
done
if [ -s $d/source/policy.pt ]; then
  rsync -a $d/source/{policy.pt,policy.json,result.json,config.json,train_log.jsonl} gb10-direct:$P/ \
    && ssh gb10-direct "rm -f $P/REMOTE_TRAINING" && echo "$(date -Is) synced $d"
fi
