#!/usr/bin/env bash
# Demo sprint: resolve a rebase stopped on INDEX.md (union) or generated docs/demo files (rebuild later), then continue.
set -uo pipefail
cd "$(dirname "$0")/../.."
for i in 1 2 3 4 5; do
  u=$(git diff --name-only --diff-filter=U)
  [ -z "$u" ] && break
  for f in $u; do
    case $f in
      artifacts/video/INDEX.md) sed -i '/^<<<<<<< \|^=======$\|^>>>>>>> /d' $f; git add $f ;;
      docs/demo/*) git checkout --theirs -- $f 2>/dev/null || git checkout --ours -- $f; git add $f ;;
      *) echo "UNRESOLVED $f"; exit 1 ;;
    esac
  done
  GIT_EDITOR=true git rebase --continue >/dev/null 2>&1 || true
done
git status --short | head -3
