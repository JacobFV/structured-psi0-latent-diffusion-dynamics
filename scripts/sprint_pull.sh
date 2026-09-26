#!/usr/bin/env bash
# Pull small result files (json/jsonl) of the given peer run dirs into this worktree's artifacts/runs.
set -euo pipefail
for d in "$@"; do
  mkdir -p "artifacts/runs/$d"
  rsync -a --include='*/' --include='*.json' --include='*.jsonl' --exclude='*' "gb10-direct:/dev/shm/rrp-brandonin/repo/artifacts/runs/$d/" "artifacts/runs/$d/"
done
