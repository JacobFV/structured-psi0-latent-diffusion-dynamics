#!/usr/bin/env bash
# pull replication eval outputs (small JSONL/JSON) from the peer store. usage: legged_fixrep_pull.sh BODY
set -euo pipefail
B=$1; P=gb10-direct:/dev/shm/rrp-brandonin/repo/artifacts/runs
rsync -a --include='*fixrep*' --exclude='*' $P/legged_ladder/$B/ artifacts/runs/legged_ladder/$B/
rsync -a --include='*fixrep*/***' --exclude='*' $P/legged_edits/$B/ artifacts/runs/legged_edits/$B/
for d in $(ssh gb10-direct "cd /dev/shm/rrp-brandonin/repo/artifacts/runs && ls -d legged_fixrep_{rep,flow}_*_${B}_s? 2>/dev/null"); do
  mkdir -p artifacts/runs/$d; rsync -a --include='*.json' --exclude='*' $P/$d/ artifacts/runs/$d/; done
