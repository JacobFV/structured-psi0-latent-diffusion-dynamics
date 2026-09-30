#!/usr/bin/env bash
# Push project code to the peer's RAM-backed workspace (never node-local state), bounded bandwidth.
# Usage: ops/bin/peer_sync.sh push | pull <remote-subdir> [<local-subdir>] | revision
# push also writes $R/.rrp_revision = {"git_sha", "dirty", "tree", "src_tree", ...} (the synced copy has no .git; W3 provenance reads it).
set -euo pipefail
PEER=${ROBOT_PEER:-gb10-direct}
P=${RRP_PEER_ROOT:-/dev/shm/rrp-brandonin}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
# Each worktree/agent may use its own peer code dir (RRP_PEER_REPO); artifacts, assets and ops state stay shared
# with the main peer repo so there is one broker and one artifact store.
R=${RRP_PEER_REPO:-$P/repo}
# Guard (D-096): three agents overwrote the shared code dir by pushing without RRP_PEER_REPO. A push now requires an
# explicit RRP_PEER_REPO; pushing into the shared $P/repo additionally requires RRP_ALLOW_SHARED_REPO=1; and a push is
# refused while any peer process runs from the target dir (its code would change under running jobs).
guard_push() {
  if [ -z "${RRP_PEER_REPO:-}" ]; then
    echo "peer_sync: refusing push: export RRP_PEER_REPO=$P/wt/<track> first" >&2; exit 2; fi
  if [ "$R" = "$P/repo" ] && [ "${RRP_ALLOW_SHARED_REPO:-0}" != 1 ]; then
    echo "peer_sync: refusing push into the shared $P/repo (set RRP_ALLOW_SHARED_REPO=1 if you really mean it)" >&2; exit 2; fi
  local busy
  # jobs reference their code dir through their working directory (cwd), not the command line
  busy=$(ssh "$PEER" "for p in \$(pgrep -u \$USER -f 'python|rrp'); do c=\$(readlink /proc/\$p/cwd 2>/dev/null) || continue;
      case \"\$c/\" in '$R'/*) tr '\\0' ' ' < /proc/\$p/cmdline | cut -c1-160; echo;; esac; done | grep -v ops.watchdog | head -3" || true)
  if [ -n "$busy" ] && [ "${RRP_SYNC_FORCE:-0}" != 1 ]; then
    echo "peer_sync: refusing push: processes are running from $R:" >&2; echo "$busy" >&2; exit 3; fi
}
# git sha + dirty flag + tree hashes of the pushed tree, as rrp.core.provenance.code_provenance computes them:
# dirty = any tracked change OR untracked (non-ignored) file; src_tree = git tree of the WORKING src/ (built in a
# throw-away index), the key run-dag uses to decide whether a completed node ran the same code.
revision_json() {
  local sha dirty tree src idx
  sha=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null) || sha=""
  if [ -z "$sha" ]; then echo '{"git_sha": null, "dirty": null, "tree": null, "src_tree": null}'; return; fi
  if [ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]; then dirty=true; else dirty=false; fi
  tree=$(git -C "$ROOT" rev-parse 'HEAD^{tree}' 2>/dev/null)
  idx=$(mktemp -u)
  src=$(GIT_INDEX_FILE=$idx git -C "$ROOT" read-tree HEAD 2>/dev/null && GIT_INDEX_FILE=$idx git -C "$ROOT" add -A -- src \
        && GIT_INDEX_FILE=$idx git -C "$ROOT" write-tree --prefix=src/) || src=""
  rm -f "$idx"
  printf '{"git_sha": "%s", "dirty": %s, "tree": "%s", "src_tree": "%s", "branch": "%s", "synced_at": "%s", "source": "%s"}\n' \
    "$sha" "$dirty" "$tree" "$src" "$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null)" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname):$ROOT"
}
case "${1:-push}" in
  push)
    guard_push
    ssh "$PEER" "mkdir -p $R"
    rsync -a --bwlimit=100000 --delete \
      --exclude .venv --exclude /.cache --exclude .git --exclude node_modules \
      --exclude 'ops/broker/' --exclude 'ops/logs/' --exclude 'ops/watchdog/' --exclude 'ops/resource-ledger*.jsonl' \
      --exclude 'ops/resources.local.json' --exclude '/artifacts' --exclude 'research/registry.jsonl' \
      --exclude '__pycache__/' --exclude '/.rrp_revision' \
      "$ROOT/" "$PEER:$R/"
    revision_json | ssh "$PEER" "cat > $R/.rrp_revision"
    if [ "$R" != "$P/repo" ]; then
      ssh "$PEER" "cd $R && for d in artifacts .cache; do [ -e \$d ] || ln -s $P/repo/\$d \$d; done
                   mkdir -p ops && cp -n $P/repo/ops/resources.local.json ops/ 2>/dev/null; true"
    fi
    # artifacts/ itself is never synced (peer store), but the run-id alias index is code-like and small
    if [ -f "$ROOT/artifacts/run_index.json" ]; then rsync -a "$ROOT/artifacts/run_index.json" "$PEER:$R/artifacts/" 2>/dev/null || true; fi ;;
  pull)
    # pull owned peer artifacts into artifacts/peer/<subdir>; never delete newer local files
    src=${2:?remote subdir under repo}; dst=${3:-$2}
    mkdir -p "$ROOT/$dst"
    rsync -a --update --bwlimit=100000 "$PEER:$R/$src/" "$ROOT/$dst/" ;;
  revision)
    revision_json ;;          # print what push writes (local only; no ssh)
  *) echo "unknown"; exit 2 ;;
esac
