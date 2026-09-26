#!/usr/bin/env bash
# Stage B flows sem + nosem in parallel (one host GPU lease). usage: legged_flow_pair.sh TAG
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
for v in sem nosem; do
  PYTHONPATH=src $PY -m rrp.learning.legged_latent_train flow --config configs/legged_latent/flow_${v}_$1.json --out artifacts/runs/legged_flow_${v}_$1 > artifacts/runs/legged_flow_${v}_$1.out 2>&1 &
done
wait
