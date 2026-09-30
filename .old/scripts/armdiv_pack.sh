#!/usr/bin/env bash
# armdiv G1: pack the v7div collection straight to peer disk (artifacts/packed -> ~/rrp-peer-data/packed), one lease.
# Usage: RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/armdiv scripts/armdiv_pack.sh [--detach]
set -euo pipefail
cd "$(dirname "$0")/.."
exec ops/bin/peer_run.sh --cpu 2 --mem 36G --disk 25G --label armdiv_pack_v7div --max-seconds 21600 \
  --env RRP_GRASP_CONTACT=v2.1 "$@" -- PY -m rrp.cli data pack --config scripts/armdiv_pack_config.json \
  --out artifacts/packed/latent_pp_v7div_s1_H16
