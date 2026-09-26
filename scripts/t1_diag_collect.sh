#!/usr/bin/env bash
# t1 diagnosis (T1 DIAGNOSIS in research/tracks/legged_vlm.md): record matched-state buffers with the received packet
# and the privileged shadow-teacher chunk (DIAGNOSTIC labels) at packet ticks. Seeds 22000-22015 (disjoint from dev).
# R2 (generated, original system 0, final flow) for 4 training seeds x {sem,nosem}; R1 (E(BC chunk)) seed 0; BC.
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
PAR=${PAR:-12}
B=artifacts/runs/legged_bc_t1_v1/policy.pt
O=artifacts/runs/t1_diag/buf
cmds=()
for half in 22000-22007 22008-22015; do
  for ts in v2 v2s1 v2s2 v2s3; do for v in sem nosem; do
    cmds+=("collect --diag --route generated --body t1 --bc $B --rep artifacts/runs/legged_rep_${v}_t1_${ts}/representation.pt --flow artifacts/runs/legged_flow_${v}_t1_${ts}/policy.pt --seeds $half --out $O/r2_${v}_${ts}")
  done; done
  for v in sem nosem; do
    cmds+=("collect --diag --route oracle_bc --body t1 --bc $B --rep artifacts/runs/legged_rep_${v}_t1_v2/representation.pt --seeds $half --out $O/r1_${v}_v2")
  done
  cmds+=("collect --diag --route bc --body t1 --bc $B --seeds $half --out $O/bc")
done
printf '%s\n' "${cmds[@]}" | xargs -P "$PAR" -I{} bash -c "OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY -m rrp.learning.legged_dagger {} > /dev/null 2>&1 || { echo FAIL: {}; exit 1; }"
rc=$?
ls $O/*/t1/*.npz | wc -l
exit $rc
