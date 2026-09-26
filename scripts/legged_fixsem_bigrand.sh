#!/usr/bin/env bash
# LEGGED FIXED-SEM: larger matched-norm random z controls (|dz| 16, 25) so the halt probe edits (|dz| 12-25) have a
# random control of at least their norm. Same R2 snap_s4000 route args, seeds 10000-10019, t_edit 2 s, window 2-5 s as the
# original suites; written to separate dirs (*_bigrand) with their own unedited run. usage: BODY [PAR]
set -uo pipefail
B=$1; PAR=${2:-8}; T=${B}_v2; rc=0
E="none rand_norm:16 rand_norm:25"
for v in sem nosem fixsem; do
  case $v in
    sem) RA="--flow artifacts/runs/legged_flow_sem_$T/snap_s4000.pt";;
    nosem) RA="--flow artifacts/runs/legged_flow_nosem_$T/snap_s4000.pt --posthoc-probe artifacts/runs/legged_rep_nosem_$T/probe_posthoc.pt";;
    fixsem) RA="--flow artifacts/runs/legged_fixsem_flow_sem_${B}_lv4/snap_s4000.pt";;
  esac
  o=$(EDITS="$E" bash scripts/legged_edit_suite.sh $B r2_${v}_snap_s4000_bigrand $PAR "$RA" 2>&1); echo "$o"; grep -q FAIL <<<"$o" && rc=1
done
exit $rc
