#!/usr/bin/env bash
# T1 DIAGNOSIS fix test (seed 0): generator-aware system-0 refit vs control refit, sem and nosem identically.
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
rc=0
for v in sem nosem; do for k in genz ctl; do
  o=artifacts/runs/t1diag_rz_${v}_${k}
  [ -f $o/result.json ] && continue
  PYTHONPATH=src $PY -m rrp.learning.legged_dagger refit --config configs/t1_diag/rz_${v}_${k}.json --out $o > $o.out 2>&1 || { echo FAIL $o; rc=1; }
done; done
exit $rc
