#!/usr/bin/env bash
# task-context edits on the deployable route: system i generates the packet from an EDITED public task context.
# usage: BODY T CK
set -uo pipefail
BODY=$1; T=$2; CK=$3
E="none mirror_goal mirror_active mirror_inactive halt"
EDITS="$E" bash scripts/legged_edit_suite.sh $BODY r2ctx_sem_${CK%.pt} 2 "--flow artifacts/runs/legged_flow_sem_${T}/$CK" &
EDITS="$E" bash scripts/legged_edit_suite.sh $BODY r2ctx_nosem_${CK%.pt} 2 "--flow artifacts/runs/legged_flow_nosem_${T}/$CK --posthoc-probe artifacts/runs/legged_rep_nosem_${T}/probe_posthoc.pt" &
wait
