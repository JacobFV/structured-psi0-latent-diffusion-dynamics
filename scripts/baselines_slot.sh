#!/usr/bin/env bash
# Serial-per-slot runner for the latent_slice1 BASELINE cells (one leased GPU job per slot).
# Usage (inside a lease, from the peer code dir):  scripts/baselines_slot.sh <method> <seed> [<seed> ...]
# For each seed: source competence cell (trains the source model if missing), then every target x budget cell.
# Cells are idempotent (skipped when their outputs exist), so rerunning the same command resumes.
# A seed whose source model is being trained on another machine (marker source/REMOTE_TRAINING present and no
# policy.pt yet) is deferred to the end of the queue and waited for (polling), never trained twice.
set -uo pipefail
PY=${PY:-python}
method=$1; shift
ROOT=${ROOT:-artifacts/runs/latent_slice1}
TARGETS=${TARGETS:-"xarm7_pg2 xarm7_tf3 panda_tf3"}
BUDGETS=${BUDGETS:-"0 5 20 100"}
run_seed() {
  local s=$1
  echo "[slot] $(date -Is) $method seed$s source"
  $PY -m rrp.cli campaign baseline-cell --method "$method" --seed "$s" --target source --root "$ROOT" \
    || { echo "[slot] FAILED $method seed$s source"; fails=$((fails+1)); return 1; }
  for t in $TARGETS; do for b in $BUDGETS; do
    echo "[slot] $(date -Is) $method seed$s $t b$b"
    $PY -m rrp.cli campaign baseline-cell --method "$method" --seed "$s" --target "$t" --budget "$b" --root "$ROOT" \
      || { echo "[slot] FAILED $method seed$s $t b$b"; fails=$((fails+1)); }
  done; done
}
deferred=()
fails=0
for s in "$@"; do
  d="$ROOT/$method/seed$s/source"
  if [ -e "$d/REMOTE_TRAINING" ] && [ ! -s "$d/policy.pt" ]; then deferred+=("$s"); continue; fi
  run_seed "$s"
done
for s in "${deferred[@]}"; do
  d="$ROOT/$method/seed$s/source"
  while [ -e "$d/REMOTE_TRAINING" ] && [ ! -s "$d/policy.pt" ]; do
    echo "[slot] $(date -Is) waiting for remotely trained $d/policy.pt"; sleep 300
  done
  run_seed "$s"
done
echo "[slot] $(date -Is) done $method $*"
[ "$fails" = 0 ] && touch "$ROOT/.slot_done_${method}_$(echo "$@" | tr ' ' _)_b$(echo $BUDGETS | tr ' ' _)"
