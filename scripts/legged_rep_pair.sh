#!/usr/bin/env bash
# Stage A sem + capacity-matched nosem in parallel (one host GPU lease). usage: legged_rep_pair.sh TAG (e.g. go2_v2)
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
for v in sem nosem; do
  PYTHONPATH=src $PY -m rrp.learning.legged_latent_train rep --config configs/legged_latent/rep_${v}_$1.json --out artifacts/runs/legged_rep_${v}_$1 > artifacts/runs/legged_rep_${v}_$1.out 2>&1 &
done
wait
