#!/usr/bin/env bash
# Demo sprint: wait up to $1 s; exit with a line when a lead-relevant event appears:
# (a) a non-teacher ladder summary with success > 0 that is new, (b) binding v4 / system-0 gate results, or any
# new ladder_v1 / ladder_localize / acceptance_sprint file (reported as MINOR).
S=/dev/shm/rrp-brandonin/repo/artifacts/runs
snap() { ssh gb10-direct "ls $S/ladder_v1/*/*.summary.json $S/ladder_localize/*/*.json $S/acceptance_sprint_*/*summary*.json 2>/dev/null; ls -d $S/binding_paired_*_v4/*.json 2>/dev/null" | sort; }
a=$(snap); end=$(( $(date +%s) + ${1:-1800} ))
cd ~/work/relational-robot-policy; git fetch -q origin; base=$(git rev-parse origin/main)
while [ $(date +%s) -lt $end ]; do
  sleep 150
  b=$(snap); new=$(comm -13 <(echo "$a") <(echo "$b"))
  git fetch -q origin 2>/dev/null; ch=$(git log --format='%h %s' $base..origin/main -- research/decisions.md research/tracks artifacts/runs | grep -v "demo:" | head)
  if [ -n "$new" ] || [ -n "$ch" ]; then
    echo "$(date +%H:%M) NEW FILES:"; echo "$new"; echo "COMMITS:"; echo "$ch"
    for f in $(echo "$new" | grep ladder_v1); do ssh gb10-direct "python3 -c \"import json;d=json.load(open('$f'));print('$f'.split('ladder_v1/')[1], d.get('success'), d.get('n'))\""; done
    exit 0
  fi
done
echo "$(date +%H:%M) quiet"
