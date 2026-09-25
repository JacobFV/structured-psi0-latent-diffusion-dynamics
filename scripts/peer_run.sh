#!/usr/bin/env bash
# Run a leased job on the peer from this checkout's peer code dir, using the ONE shared peer broker.
# Usage: [RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/<name>] scripts/peer_run.sh <ops-run args> -- PY <args...>
#   e.g. scripts/peer_run.sh --gpu --gpu-mem 12G --cpu 4 --mem 24G --label X --max-seconds 7200 -- PY -m rrp.cli ...
# A bare argument "PY" is replaced by the peer venv python. Relative paths resolve inside the peer code dir, whose
# artifacts/ and .cache/ are symlinks to the shared store. Add --detach to return immediately.
set -euo pipefail
PEER=${ROBOT_PEER:-gb10-direct}
P=${RRP_PEER_ROOT:-/dev/shm/rrp-brandonin}
R=${RRP_PEER_REPO:-$P/repo}
args=()
for x in "$@"; do [ "$x" = PY ] && args+=("$P/venv/bin/python") || args+=("$x"); done
ssh "$PEER" "cd $R && export PATH=$P/bin:\$PATH PYTHONPATH=src RRP_NODE=peer RRP_REPO=$R RRP_OPS_ROOT=$P/repo MUJOCO_GL=egl && python3 -m rrp.cli ops run $(printf '%q ' "${args[@]}")"
