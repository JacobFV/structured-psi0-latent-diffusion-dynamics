#!/usr/bin/env bash
# Push project code to the peer's RAM-backed workspace (never node-local state), bounded bandwidth.
# Usage: scripts/peer_sync.sh push | pull-artifacts <remote-subdir> <local-subdir>
set -euo pipefail
PEER=${ROBOT_PEER:-gb10-direct}
P=${RRP_PEER_ROOT:-/dev/shm/rrp-brandonin}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
case "${1:-push}" in
  push)
    rsync -a --bwlimit=100000 --delete \
      --exclude .venv --exclude .cache --exclude .git --exclude node_modules \
      --exclude 'ops/broker/' --exclude 'ops/logs/' --exclude 'ops/watchdog/' --exclude 'ops/resource-ledger*.jsonl' \
      --exclude 'configs/resources.local.json' --exclude 'artifacts/' --exclude 'research/registry.jsonl' \
      --exclude 'ui/node_modules/' --exclude 'ui/dist/' --exclude '__pycache__/' \
      "$ROOT/" "$PEER:$P/repo/" ;;
  pull)
    # pull owned peer artifacts into artifacts/peer/<subdir>; never delete newer local files
    src=${2:?remote subdir under repo}; dst=${3:-$2}
    mkdir -p "$ROOT/$dst"
    rsync -a --update --bwlimit=100000 "$PEER:$P/repo/$src/" "$ROOT/$dst/" ;;
  *) echo "unknown"; exit 2 ;;
esac
