#!/usr/bin/env bash
# Demo sprint: rebase on main, pull raw, rebuild, commit + merge to main if anything changed. Prints the commit.
set -euo pipefail
cd "$(dirname "$0")/../.."
git fetch -q origin && git rebase -q origin/main
scripts/demo/pull_raw.sh >/dev/null
python3 scripts/demo/build_page.py
git add -A docs/demo
if git diff --cached --quiet -- docs/demo/raw docs/demo/video; then echo "no new raw/video"; git reset -q; exit 0; fi
git commit -qm "demo: refresh (${1:-new raw results})

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git fetch -q origin && git rebase -q origin/main && git push -q origin HEAD:main && git push -q -f origin HEAD:track/demo
git log --oneline -1
