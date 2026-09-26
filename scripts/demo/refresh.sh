#!/usr/bin/env bash
# Demo sprint: pull raw, rebuild, commit demo paths, rebase on main and merge. Prints the commit.
set -euo pipefail
cd "$(dirname "$0")/../.."
scripts/demo/pull_raw.sh >/dev/null
python3 scripts/demo/build_page.py
git add -A docs/demo scripts/demo research/tracks/demo.md research/reports/evidence_matrix.md artifacts/video
if git diff --cached --quiet; then echo "nothing new staged"; else
git commit -qm "demo: refresh (${1:-new raw results})

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
fi
git fetch -q origin && { git rebase -q origin/main || scripts/demo/resolve.sh; } && python3 scripts/demo/build_page.py >/dev/null && { git add -A docs/demo; git diff --cached --quiet || git commit -qm "demo: rebuild after rebase"; } && git push -q origin HEAD:main && git push -q -f origin HEAD:track/demo
git log --oneline -1
