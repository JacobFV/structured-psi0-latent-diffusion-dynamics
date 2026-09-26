#!/usr/bin/env bash
# T1 DIAGNOSIS offline measurements for the 4 training seeds (GPU)
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
rc=0
for ts in ${TS:-v2 v2s1 v2s2 v2s3}; do
  PYTHONPATH=src $PY -m rrp.learning.legged_t1_diag --ts $ts --out artifacts/runs/t1_diag/diag_$ts.json > artifacts/runs/t1_diag/diag_$ts.out 2>&1 || { echo FAIL $ts; rc=1; }
done
for ts in ${TS:-v2 v2s1 v2s2 v2s3}; do
  PYTHONPATH=src $PY -m rrp.learning.legged_t1_diag --extra --ts $ts --out artifacts/runs/t1_diag/temporal_$ts.json > artifacts/runs/t1_diag/temporal_$ts.out 2>&1 || { echo FAIL temporal $ts; rc=1; }
done
exit $rc
