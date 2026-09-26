#!/usr/bin/env bash
# T1 DIAGNOSIS closed-loop tests on the matched dev seeds 10000-10029 (training seed 0 = t1_v2). CPU, one lease.
# usage: t1_diag_evals.sh {refit|oracle|flow_w0}
set -uo pipefail
B=artifacts/runs/legged_bc_t1_v1/policy.pt
PAR=${PAR:-5}
case $1 in
  refit) for v in sem nosem; do for k in genz ctl; do
      bash scripts/legged_ladder.sh t1 t1diag_${v}_rz${k} $PAR r2 - - artifacts/runs/t1diag_rz_${v}_${k}/realizer.pt artifacts/runs/legged_flow_${v}_t1_v2/policy.pt &
    done; done;;
  oracle) for v in sem nosem; do
      bash scripts/legged_ladder.sh t1 t1diag_${v}_v2_orig $PAR r1t - artifacts/runs/legged_rep_${v}_t1_v2/representation.pt &
    done;;
  flow_w0) bash scripts/legged_ladder.sh t1 t1diag_sem_flow_w0 $PAR r2 - - - artifacts/runs/t1diag_flow_sem_w0/policy.pt &;;
esac
wait
