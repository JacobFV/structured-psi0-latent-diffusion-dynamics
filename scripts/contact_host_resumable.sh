#!/usr/bin/env bash
# Run a tracker_training job on the host and resume it (--resume) when the memory watchdog sheds it. Bounded (D-061):
# at most $ATTEMPTS launches; exits 0 when train_log reaches iter ITERS-1, 3 otherwise.
# usage: ATTEMPTS=8 scripts/contact_host_resumable.sh <label> <out dir> <iters> <cpu> <mem> -- <tracker_training args without --out/--iters/--resume>
set -u
LABEL=$1; OUT=$2; ITERS=$3; CPU=$4; MEM=$5; shift 6
PY=$HOME/work/relational-robot-policy/.venv/bin/python
last=$((ITERS - 1))
for a in $(seq 1 "${ATTEMPTS:-8}"); do
  if [ -f "$OUT/train_log.jsonl" ] && grep -q "\"iter\": $last," "$OUT/train_log.jsonl"; then echo "done"; exit 0; fi
  out=$(TRIES=60 scripts/contact_launch_retry.sh --cpu "$CPU" --mem "$MEM" --label "$LABEL" --max-seconds 21600 -- \
        "$PY" -m rrp.training.tracker_training "$@" --iters "$ITERS" --resume --out "$OUT" | tail -1)
  unit=$(echo "$out" | sed -n 's/.*"unit": "\([^"]*\)".*/\1/p')
  [ -z "$unit" ] && { echo "never admitted"; exit 3; }
  echo "attempt $a: $unit"
  while systemctl --user is-active -q "$unit"; do sleep 30; done
done
grep -q "\"iter\": $last," "$OUT/train_log.jsonl" && { echo done; exit 0; }
exit 3
