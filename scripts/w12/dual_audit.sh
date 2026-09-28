#!/usr/bin/env bash
# W12 phase-B dual teacher audit (<= 50 episodes per task; peer CPU, 2 workers). Run from the peer code dir via
#   scripts/peer_run.sh --cpu 2 --mem 4G --label w12_dualaudit --max-seconds 3600 --detach -- bash scripts/w12/dual_audit.sh
set -uo pipefail
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
O=artifacts/runs/w12_dualaudit
mkdir -p $O
SI=panda_pg2__ur5e_pg2,parm5_pg2__parm5_pg2,parm6_tf3__parm6_pg2,ur5e_pg2__sawyer_pg2
HO=panda_pg2__ur5e_pg2,parm5_pg2__parm5_pg2,parm5_pg2__parm6_tf3,ur5e_pg2__sawyer_pg2
A="-m rrp.evaluation.dual_teacher_quality --workers 2"
rc=0
for task in support_insert handover; do
  P=$SI; [ $task = handover ] && P=$HO
  MS=1200; [ $task = handover ] && MS=800
  # 28 clean episodes under the current physics (grasp_v2.1)
  RRP_GRASP_CONTACT=v2.1 $PY $A --task $task --pairs $P --seeds 0:7 --max-steps $MS --out $O/${task}_v21_clean.jsonl || rc=1
  # 12 paired clean episodes under the legacy physics (grasp_v1) on seeds 0-2: attributes changes to the physics
  RRP_GRASP_CONTACT=v1 $PY $A --task $task --pairs $P --seeds 0:3 --max-steps $MS --out $O/${task}_v1_clean.jsonl || rc=1
  # 10 DART episodes (N(0, 0.04 rad) bursts 5/25, as the v1 DART datasets) under grasp_v2.1
  RRP_GRASP_CONTACT=v2.1 $PY $A --task $task --pairs ${P%%,*},$(echo $P | cut -d, -f3) --seeds 0:5 --noise 0.04 --burst 25,5 \
      --max-steps $MS --out $O/${task}_v21_dart.jsonl || rc=1
done
exit $rc
