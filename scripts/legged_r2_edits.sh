#!/usr/bin/env bash
# causal packet edits on the deployable route (flow -> system 0), sem (joint probe) and nosem (post-hoc probe). usage: BODY T CK
set -uo pipefail
BODY=$1; T=$2; CK=$3
bash scripts/legged_edit_suite.sh $BODY r2_sem_${CK%.pt} 2 "--flow artifacts/runs/legged_flow_sem_${T}/$CK" &
bash scripts/legged_edit_suite.sh $BODY r2_nosem_${CK%.pt} 2 "--flow artifacts/runs/legged_flow_nosem_${T}/$CK --posthoc-probe artifacts/runs/legged_rep_nosem_${T}/probe_posthoc.pt" &
wait
