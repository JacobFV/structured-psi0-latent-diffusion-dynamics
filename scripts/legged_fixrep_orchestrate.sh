#!/usr/bin/env bash
# Wait-and-launch helper (runs on the host, no compute): when a training lease ends, check its outputs and launch the peer eval.
# usage: legged_fixrep_orchestrate.sh {hex|go2host|go2peer}
set -uo pipefail
export RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/legged_vlm
PL=/dev/shm/rrp-brandonin/repo/ops/logs; PA=/dev/shm/rrp-brandonin/repo/artifacts/runs; PPY=/dev/shm/rrp-brandonin/venv/bin/python
waitpeer() { until ssh gb10-direct "grep -q 'rrp.child' $PL/$1_*.log"; do sleep 60; done; ssh gb10-direct "grep 'rrp.child' $PL/$1_*.log"; }
haveflows() { local ok=0; for t in "$@"; do ssh gb10-direct "test -f $PA/legged_fixrep_flow_$t/policy.pt -a -f $PA/legged_fixrep_flow_$t/snap_s4000.pt" || { echo "missing $t"; ok=1; }; done; return $ok; }
ev() { local B=$1; shift; bash scripts/peer_run.sh --cpu 10 --mem 12G --label legged_fixrep_eval_${B}_$1 --max-seconds 14400 --detach -- \
        bash -c "PY=$PPY bash scripts/legged_fixrep_eval.sh $B 10 $*"; }
case $1 in
  hex) T="fixsem_hexapod6_s1 fixsem_hexapod6_s2 nosem_hexapod6_s1 nosem_hexapod6_s2"
       waitpeer 1790440023_da8335; haveflows $T && ev hexapod6 $T;;
  go2peer) T="fixsem_go2_s1 fixsem_go2_s2 nosem_go2_s2"
       waitpeer 1790440106_9ac1a9; haveflows $T && ev go2 $T;;
  go2host) t=nosem_go2_s1; F=artifacts/runs/legged_fixrep_flow_$t; R=artifacts/runs/legged_fixrep_rep_$t
       until [ -f $F/policy.pt ] || ! systemctl --user is-active -q rrp-job-1790440021_55a138.service; do sleep 60; done; sleep 20
       [ -f $F/policy.pt ] && [ -f $R/probe_posthoc.pt ] || { echo "host chain $t incomplete"; exit 1; }
       rsync -a --mkpath $R/{config.json,result.json,representation.pt,probe_posthoc.pt,probe_posthoc.json,probe_cfg.json} gb10-direct:$PA/legged_fixrep_rep_$t/ &&
       rsync -a --mkpath $F/{config.json,result.json,policy.pt,snap_s4000.pt,snap_s8000.pt} gb10-direct:$PA/legged_fixrep_flow_$t/ || exit 1
       ssh gb10-direct "sed -i 's#\"representation\": \".*\"#\"representation\": \"$R/representation.pt\"#' $PA/legged_fixrep_rep_$t/probe_cfg.json" ; ev go2 $t;;
esac
