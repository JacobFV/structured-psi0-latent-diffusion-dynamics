#!/usr/bin/env bash
# D-045: rebuild the seed-1701 baseline sources with the B-1 fix on the HOST GPU (fresh root: the pre-fix peer
# checkpoints must never be resumed), then run source competence. Resumable 6 h segments (exact resume).
# Usage: [RRP_NODE=peer] [GPUMEM=16G MEM=10G] scripts/baselines_host_b1fix.sh <method>
set -uo pipefail
M=${1:?method}
cd "$(dirname "$0")/.."
if [ "${RRP_NODE:-host}" = peer ]; then       # peer: run from this checkout's peer dir, shared broker
  export PATH=/dev/shm/rrp-brandonin/bin:$PATH PYTHONPATH=src RRP_REPO=$PWD RRP_OPS_ROOT=/dev/shm/rrp-brandonin/repo
  PY=/dev/shm/rrp-brandonin/venv/bin/python; RUNPY=python3
else
  export PYTHONPATH=src RRP_REPO=$PWD RRP_NODE=host; PY=$PWD/.venv/bin/python; RUNPY=$PY
fi
ROOT=artifacts/runs/latent_slice1_b1fix
for seg in $(seq 1 30); do
  $RUNPY -m rrp.cli ops run --gpu --gpu-mem ${GPUMEM:-16G} --cpu 4 --mem ${MEM:-10G} --label b1fix_$M --max-seconds 21600 -- \
    $PY -m rrp.cli campaign baseline-cell --method $M --seed 1701 --target source --root $ROOT && { echo "$M done"; exit 0; }
  echo "$M attempt $seg ended without completion (capacity refusal or 6 h segment); retry in 120 s"; sleep 120
done
exit 1
