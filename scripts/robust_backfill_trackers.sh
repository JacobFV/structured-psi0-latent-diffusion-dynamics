#!/usr/bin/env bash
# W6 tracker gate backfill (report only). PEER ONLY (D-115): run through the peer broker, e.g.
#   RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/robust scripts/peer_run.sh --cpu 1 --mem 3G --label robust_gate_trk \
#       --max-seconds 2800 -- bash scripts/robust_backfill_trackers.sh
# (actors under artifacts/runs/robust/gates/actors/<name>/actor.pt in the peer store; finished trackers are skipped)
set -u
cd "$(dirname "$0")/.."
[ "${RRP_NODE:-}" = peer ] || { echo "robust_backfill_trackers: peer only (D-115); launch via scripts/peer_run.sh" >&2; exit 2; }
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
O=artifacts/runs/robust/gates/trackers; A=artifacts/runs/robust/gates/actors; mkdir -p $O
for spec in anymal_c:anymal_c_installed: go2:go2_installed: t1:t1_w8d_installed: t1:t1_w8c: g1:g1_src: g1:g1_r1_installed:legacy_gains_v0; do
  IFS=: read body name lim <<< "$spec"
  [ -f $O/$name/gate_report.json ] && continue
  mkdir -p $O/$name
  env ${lim:+RRP_ACTUATOR_LIMITS=$lim} PYTHONPATH=src $PY -m rrp.evaluation.tracker_validation --body $body --contact v2 --actor $A/$name/actor.pt \
     --seeds 5 --robust --out $O/$name/validation.json --gate-dir $O/$name > $O/$name/log.txt 2>&1
  echo "$name rc=$?"
done
