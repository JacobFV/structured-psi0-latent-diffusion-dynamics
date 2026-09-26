#!/usr/bin/env bash
# BC positive controls for several bodies in sequence (one host GPU lease). usage: legged_bc_chain.sh body...
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
for b in "$@"; do
  [ -f artifacts/runs/legged_bc_${b}_v1/policy.pt ] || PYTHONPATH=src $PY -m rrp.learning.legged_bc train --config configs/legged_bc/bc_${b}_v1.json --out artifacts/runs/legged_bc_${b}_v1 > artifacts/runs/legged_bc_${b}_v1.out 2>&1
done
