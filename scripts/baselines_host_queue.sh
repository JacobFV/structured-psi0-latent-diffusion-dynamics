#!/usr/bin/env bash
# Host GPU slot (one lease) for the latent_slice1 baselines, run in <=6 h resumable segments by
# scripts/baselines_host_supervise.sh. Queue: direct-action seed1703 SOURCE model (train-only; the supervisor rsyncs
# it to the peer, whose direct slot runs its cells), then every action-only-codec cell for seeds 1701-1703.
set -uo pipefail
PY=${PY:-python}
$PY -m rrp.cli campaign baseline-cell --method baseline_direct_action --seed 1703 --target source --train-only \
  || echo "[slot] FAILED direct seed1703 source"
PY=$PY bash scripts/baselines_slot.sh baseline_action_only_codec 1701 1702 1703
