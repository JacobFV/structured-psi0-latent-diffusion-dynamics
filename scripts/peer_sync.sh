#!/usr/bin/env bash
# Push project code to the peer's RAM-backed workspace (never node-local state), bounded bandwidth.
# Usage: scripts/peer_sync.sh push | pull-artifacts <remote-subdir> <local-subdir>
set -euo pipefail
PEER=${ROBOT_PEER:-gb10-direct}
P=${RRP_PEER_ROOT:-/dev/shm/rrp-brandonin}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# Each worktree/agent may use its own peer code dir (RRP_PEER_REPO); artifacts, assets and ops state stay shared
# with the main peer repo so there is one broker and one artifact store.
R=${RRP_PEER_REPO:-$P/repo}
case "${1:-push}" in
  push)
    ssh "$PEER" "mkdir -p $R"
    rsync -a --bwlimit=100000 --delete \
      --exclude .venv --exclude /.cache --exclude .git --exclude node_modules \
      --exclude 'ops/broker/' --exclude 'ops/logs/' --exclude 'ops/watchdog/' --exclude 'ops/resource-ledger*.jsonl' \
      --exclude 'configs/resources.local.json' --exclude '/artifacts' --exclude 'research/registry.jsonl' \
      --exclude 'ui/node_modules/' --exclude 'ui/dist/' --exclude '__pycache__/' \
      "$ROOT/" "$PEER:$R/"
    if [ "$R" != "$P/repo" ]; then
      ssh "$PEER" "cd $R && for d in artifacts .cache; do [ -e \$d ] || ln -s $P/repo/\$d \$d; done
                   mkdir -p ops configs && cp -n $P/repo/configs/resources.local.json configs/ 2>/dev/null; true"
    fi ;;
  pull)
    # pull owned peer artifacts into artifacts/peer/<subdir>; never delete newer local files
    src=${2:?remote subdir under repo}; dst=${3:-$2}
    mkdir -p "$ROOT/$dst"
    rsync -a --update --bwlimit=100000 "$PEER:$R/$src/" "$ROOT/$dst/" ;;
  *) echo "unknown"; exit 2 ;;
esac
