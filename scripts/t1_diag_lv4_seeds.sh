#!/usr/bin/env bash
# T1 DIAGNOSIS fix replication: bounded semantic NLL for training seeds 1 and 3 (Stage A then flow). GPU, one lease.
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
for s in ${SEEDS:-1 3}; do
  R=artifacts/runs/t1diag_rep_sem_lv4_s$s; F=artifacts/runs/t1diag_flow_sem_lv4_s$s
  [ -f $R/representation.pt ] || PYTHONPATH=src $PY -m rrp.learning.legged_latent_train rep --config configs/t1_diag/rep_sem_lv4_s$s.json --out $R > $R.out 2>&1 || exit 1
  [ -f $F/policy.pt ] || PYTHONPATH=src $PY -m rrp.learning.legged_latent_train flow --config configs/t1_diag/flow_sem_lv4_s$s.json --out $F > $F.out 2>&1 || exit 2
done
