#!/usr/bin/env bash
# armdiag (D-135 follow-up) closed-loop decomposition on a TARGET body with DEV seeds (>= 3,000,000) only.
# NON-SEALED DIAGNOSTIC: never the sealed eval scenes (2,000,000+) and never reported as a sealed result.
# Usage (inside one peer lease): N=30 V=semfix S=1 TGT=xarm7_pg2 ROUTES="teacher orc_src ..." bash scripts/armdiag_closedloop.sh
# Routes: teacher | orc_src | orc_rz<b> (oracle packets E(teacher look-ahead) -> R) |
#         gen_<flow>_<R> with flow in {src, sft<b>} and R in {src, rz<b>} | bc_zs | bc_sft<b>
set -euo pipefail
export RRP_GRASP_CONTACT=v2.1
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
N=${N:-30}; V=${V:-semfix}; S=${S:-1}; TGT=${TGT:-xarm7_pg2}; SEED0=${SEED0:-3000000}
[ "$SEED0" -ge 3000000 ] || { echo "dev seeds only"; exit 2; }
L=artifacts/runs/armv6/arm6-$V; T=artifacts/runs/armtgt/armtgt6-$V-$TGT
FLOW_src=$L/flow_ft-gdag2h_s$S/policy.pt
REP_src=$L/refit-gendag3_noqd_s$S/representation.pt
BCS=$((1700 + S)); TB=artifacts/runs/armtgt/armtgt6-bc$BCS-$TGT
O=${O:-artifacts/runs/armdiag/closedloop/$V-s$S-$TGT}
mkdir -p $O
flow() { case $1 in src) echo $FLOW_src;; sft*) echo $T/target_adapt-flow_sft_b${1#sft}_s$S/policy.pt;;
         X*) echo ${EXTRA_FLOW:?};; esac; }
rep() { case $1 in src) echo $REP_src;; rz*) echo $T/target_adapt-system0_refit_b${1#rz}_s$S/representation.pt;;
        X*) echo ${EXTRA_REP:?};; esac; }
common=(--robot $TGT --n $N --seed-start $SEED0 --replan 8 --nfe 8 --max-steps 300 --prev-action zero --target-dev-diagnostic --out $O)
for r in ${ROUTES:?}; do
  case $r in
    teacher) args=(--route teacher);;
    orc_*) args=(--route oracle --rep "$(rep ${r#orc_})" --no-compare);;
    gen_*) f=${r#gen_}; f=${f%%_*}; rr=${r##*_}; args=(--route generated --flow "$(flow $f)" --rep "$(rep $rr)");;
    bc_zs) args=(--route learned --policy artifacts/runs/armexpert_bcv6/baseline_direct_action/seed$BCS/source/policy.pt --policy-label bcv6_direct${BCS}_final);;
    bc_sft*) args=(--route learned --policy $TB/target_adapt-bc_sft_b${r#bc_sft}_s$BCS/policy.pt --policy-label bcv6_${BCS}_sft_b${r#bc_sft});;
    *) echo "unknown route $r"; exit 2;;
  esac
  echo "=== $r: ${args[*]}"
  $PY scripts/ladder.py "${common[@]}" "${args[@]}" --tag "$r"
done
echo ARMDIAG_CL_OK
