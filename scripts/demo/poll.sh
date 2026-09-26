#!/usr/bin/env bash
# Demo sprint: wait (up to $1 s) for new sprint results; print what changed and exit.
cd ~/work/relational-robot-policy
git fetch -q origin
base=$(git rev-parse origin/main)
peer0=$(ssh gb10-direct 'ls /dev/shm/rrp-brandonin/repo/artifacts/runs/ladder_v1/*/*.summary.json /dev/shm/rrp-brandonin/repo/artifacts/runs/ladder_localize/*/*.json 2>/dev/null | sort | md5sum')
end=$(( $(date +%s) + ${1:-1800} ))
while [ $(date +%s) -lt $end ]; do
  sleep 120
  git fetch -q origin 2>/dev/null
  ch=$(git log --format='%h %s' $base..origin/main -- research/tracks artifacts/runs research/decisions.md | head -20)
  peer=$(ssh gb10-direct 'ls /dev/shm/rrp-brandonin/repo/artifacts/runs/ladder_v1/*/*.summary.json /dev/shm/rrp-brandonin/repo/artifacts/runs/ladder_localize/*/*.json 2>/dev/null | sort | md5sum')
  if [ -n "$ch" ] || [ "$peer" != "$peer0" ]; then
    echo "CHANGES $(date +%H:%M)"; echo "$ch"
    ssh gb10-direct 'ls -t /dev/shm/rrp-brandonin/repo/artifacts/runs/ladder_v1/*/*.summary.json | head -6'
    exit 0
  fi
done
echo "NO CHANGES $(date +%H:%M)"
