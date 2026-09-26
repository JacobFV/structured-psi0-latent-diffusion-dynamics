#!/usr/bin/env bash
# T1 DIAGNOSIS fix test: Stage A sem with a bounded semantic NLL (probe lv floor -4), then its flow. GPU, one lease.
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
[ -f artifacts/runs/t1diag_rep_sem_lv4/representation.pt ] || PYTHONPATH=src $PY -m rrp.learning.legged_latent_train rep --config configs/t1_diag/rep_sem_lv4.json --out artifacts/runs/t1diag_rep_sem_lv4 > artifacts/runs/t1diag_rep_sem_lv4.out 2>&1 || exit 1
[ -f artifacts/runs/t1diag_flow_sem_lv4/policy.pt ] || PYTHONPATH=src $PY -m rrp.learning.legged_latent_train flow --config configs/t1_diag/flow_sem_lv4.json --out artifacts/runs/t1diag_flow_sem_lv4 > artifacts/runs/t1diag_flow_sem_lv4.out 2>&1 || exit 2
