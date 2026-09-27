#!/usr/bin/env bash
# Teacher-quality verification split into per-body host leases (the host memory-PSI watchdog sheds long jobs; a shed
# body is simply rerun). Bounded: <= 4 passes. Usage: VER=v2 OUT=artifacts/runs/armexpert_diag/v2b scripts/armexpert_verify_host.sh
set -uo pipefail
cd "$(dirname "$0")/.."
VER=${VER:-v2}; OUT=${OUT:?}; PAR=${PAR:-4}
PY=$HOME/work/relational-robot-policy/.venv/bin/python
B="panda_pg2 parm6_tf3 parm5_pg2 parm5_tf3 parm5l_tf3 parm5s_pg2 parm6_pg2 parm7_pg2 parm7_tf3 sawyer_pg2 sawyer_tf3 ur5e_pg2 ur5e_tf3 parm5s_tf3 parm5l_pg2 panda_tf3 xarm7_pg2 xarm7_tf3"
mkdir -p $OUT
admit() {   # wait (<= 40 min) until the host broker admits again (memory-PSI admission stop)
  for _ in $(seq 1 240); do
    PYTHONPATH=src $PY -m rrp.cli ops status 2>/dev/null | grep -q '"admission_stopped": false' && return 0
    sleep 10
  done; return 1
}
for pass in 1 2 3 4 5 6; do
  todo=(); for b in $B; do [ -s $OUT/$b.summary.json ] || todo+=($b); done
  [ ${#todo[@]} -eq 0 ] && { echo "all done after pass $((pass-1))"; exit 0; }
  echo "$(date +%T) pass $pass: ${#todo[@]} bodies"
  i=0
  for b in "${todo[@]}"; do
    rm -f $OUT/$b.jsonl
    admit || { echo "admission stopped for 40 min"; exit 2; }
    PYTHONPATH=src $PY -m rrp.cli ops run --cpu ${WK:-2} --mem 5G --label axver_$b --max-seconds 3000 -- env PYTHONPATH=src \
      OMP_NUM_THREADS=1 $PY -m rrp.evaluation.teacher_quality --bodies $b --seeds 0-199,3000000-3000099 --versions $VER \
      --workers ${WK:-2} --chunk 25 --out $OUT/$b.jsonl > $OUT/$b.log 2>&1 &
    i=$((i+1)); [ $((i % PAR)) -eq 0 ] && wait
  done
  wait
done
for b in $B; do [ -s $OUT/$b.summary.json ] || { echo "INCOMPLETE: $b"; exit 1; }; done
echo "all done"
