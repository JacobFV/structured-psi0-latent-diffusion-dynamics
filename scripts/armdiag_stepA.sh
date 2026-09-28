#!/usr/bin/env bash
# armdiag Step A (D-135 addendum): train joint-adaptation variants at matched total updates on the target demo pack
# (budget 100, 600 updates) for one lineage/target; evaluated separately on DEV seeds only (armdiag_closedloop.sh).
set -euo pipefail
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
V=${V:-semfix}; S=${S:-1}; TGT=${TGT:-xarm7_pg2}; B=${B:-100}; STEPS=${STEPS:-600}
L=artifacts/runs/armv6/arm6-$V
O=artifacts/runs/armdiag/stepA/$V-s$S-$TGT
for vv in ${VARIANTS:-split_g0 split_g05 joint_g0 joint_g05}; do
  m=${vv%%_*}; case ${vv##*_} in g0) g=0.0;; g05) g=0.5;; g1) g=1.0;; *) echo bad; exit 2;; esac
  $PY -m rrp.training.joint_adapt --flow $L/flow_ft-gdag2h_s$S/policy.pt --rep $L/refit-gendag3_noqd_s$S/representation.pt \
    --pack artifacts/runs/armtgt/armtgt6-data/pack-${TGT}_s0 --budget $B --seed $((1700 + S)) --steps $STEPS \
    --mode $m --gen-frac $g --out $O/$vv
done
echo STEPA_TRAIN_OK
