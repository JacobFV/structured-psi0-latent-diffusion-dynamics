#!/usr/bin/env bash
# armdiv G1-G3 coordinator chain (host; orchestration only). ONE peer GPU lease at a time: the run-dag calls are
# SEQUENTIAL and each uses --max-parallel-gpu 1. Every step checks its exit code; a failed step stops the chain
# (bounded; no automatic retries beyond node `retries`). Resume: rerun this script (completed nodes are skipped by the
# ledgers; a completed step's run-dag returns immediately).
#   (smoke nodes use >= 100 steps: OneCycleLR divides by zero on ~20)
#   systemd-run --user --unit rrp-armdiv-chain --setenv=RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/armdiv \
#     --working-directory=$HOME/work/rrp-wt/armdiv bash scripts/armdiv_chain.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src RRP_PEER_REPO=${RRP_PEER_REPO:?}
PY=$HOME/work/relational-robot-policy/.venv/bin/python
P=/dev/shm/rrp-brandonin
log() { echo "$(date '+%F %T') [armdiv-chain] $*"; }
dag() { log "run-dag $*"; $PY -m rrp.cli run-dag "$@" --max-parallel-gpu 1; local rc=$?; log "rc=$rc for $1"; return $rc; }
STEPS=${STEPS:-"pack bcsmoke lsmoke bc1701 lin_sf1 lin_rest bc1702 bckf lin_kf"}
for s in $STEPS; do
  case $s in
    pack)     # the leased pack (scripts/armdiv_pack.sh) writes meta.json last
      for i in $(seq 1 480); do ssh gb10-direct "test -s $P/repo/artifacts/packed/latent_pp_v7div_s1_H16/meta.json" && break; sleep 60; done
      ssh gb10-direct "test -s $P/repo/artifacts/packed/latent_pp_v7div_s1_H16/meta.json" || { log "pack not done"; exit 1; }
      log "pack done: $(ssh gb10-direct "du -sh $P/repo/artifacts/packed/latent_pp_v7div_s1_H16 | cut -f1")" ;;
    bcsmoke)  dag dags/armdiv_bc_v7div_smoke.yaml || exit 1 ;;
    lsmoke)   dag dags/arm_lineage_v7div_smoke.yaml --max-parallel 3 --retry-failed || exit 1 ;;
    bc1701)   dag dags/armdiv_bc_v7div.yaml --point seed=1701 --max-parallel 3 || exit 1 ;;
    lin_sf1)  dag dags/arm_lineage_v7div.yaml --point variant=semfix,seed=1 --max-parallel 3 || exit 1 ;;
    lin_rest) dag dags/arm_lineage_v7div.yaml --max-parallel 3 || exit 1 ;;
    bc1702)   dag dags/armdiv_bc_v7div.yaml --point seed=1702 --max-parallel 3 || exit 1 ;;
    bckf)     dag dags/armdiv_bc_v7div_kinfeat.yaml --max-parallel 3 || exit 1 ;;
    lin_kf)   dag dags/arm_lineage_v7div_kinfeat.yaml --max-parallel 3 || exit 1 ;;
    *) log "unknown step $s"; exit 2 ;;
  esac
done
log "chain done: $STEPS"
