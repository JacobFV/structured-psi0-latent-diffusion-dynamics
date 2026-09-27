#!/usr/bin/env bash
# Wait (bounded: $WAIT_MIN minutes) until the peer's 1-min load is below $MAXLOAD (default 15), then run scripts/peer_run.sh "$@".
# Exit 4 if the peer never became free. RRP_PEER_REPO must be set (a dir with no running jobs of other tracks).
set -u
: "${RRP_PEER_REPO:?}"
MAXLOAD=${MAXLOAD:-15}
for i in $(seq 1 "${WAIT_MIN:-120}"); do
  l=$(timeout 20 ssh gb10-direct "cut -d' ' -f1 /proc/loadavg" 2>/dev/null || echo 99)
  if [ "${l%.*}" -lt "$MAXLOAD" ]; then echo "peer load $l"; exec scripts/peer_run.sh "$@"; fi
  sleep 60
done
echo "peer never below $MAXLOAD"; exit 4
