#!/usr/bin/env bash
# T1 DIAGNOSIS: remaining offline measurements (CPU): diag for seeds 2-3, temporal + readout for all seeds
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
export OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES= PYTHONPATH=src
rc=0
D=artifacts/runs/t1_diag
for ts in v2s2 v2s3; do [ -f $D/diag_$ts.json ] || $PY -m rrp.learning.legged_t1_diag --ts $ts --out $D/diag_$ts.json > $D/diag_$ts.out 2>&1 || { echo FAIL diag $ts; rc=1; }; done
for ts in v2 v2s1 v2s2 v2s3; do
  [ -f $D/readout_$ts.json ] || $PY -m rrp.learning.legged_t1_diag readout $ts $D/readout_$ts.json > $D/readout_$ts.out 2>&1 || { echo FAIL readout $ts; rc=1; }
  [ -f $D/temporal_$ts.json ] || $PY -m rrp.learning.legged_t1_diag --extra --ts $ts --out $D/temporal_$ts.json > $D/temporal_$ts.out 2>&1 || { echo FAIL temporal $ts; rc=1; }
  [ -f $D/early_$ts.json ] || $PY -m rrp.learning.legged_t1_diag --early 0 100 --ts $ts --out $D/early_$ts.json > $D/early_$ts.out 2>&1 || { echo FAIL early $ts; rc=1; }
done
exit $rc
