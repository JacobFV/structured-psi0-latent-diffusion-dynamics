#!/usr/bin/env bash
# usage: legged_refit_pair.sh BODY TAG   (configs/legged_dagger/rz_{sem,nosem}_BODY_TAG.json)
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
for v in sem nosem; do
  PYTHONPATH=src $PY -m rrp.learning.legged_dagger refit --config configs/legged_dagger/rz_${v}_$1_$2.json --out artifacts/runs/legged_rz_${v}_$1_$2 > artifacts/runs/legged_rz_${v}_$1_$2.out 2>&1 &
done
wait
