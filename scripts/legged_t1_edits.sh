#!/usr/bin/env bash
set -uo pipefail
B=artifacts/runs/legged_bc_t1_v1/policy.pt
export EDITS="none probe_yaw:0.6 probe_yaw:-0.6 probe_halt rand_norm:4 rand_norm:8"
EDITS="$EDITS" bash scripts/legged_edit_suite.sh t1 r1dag_sem 3 "--rep artifacts/runs/legged_rep_sem_t1_v2/representation.pt --realizer artifacts/runs/legged_rz_sem_t1_dag1/realizer.pt --bc $B --oracle-bc" &
EDITS="$EDITS" bash scripts/legged_edit_suite.sh t1 r1dag_nosem 3 "--rep artifacts/runs/legged_rep_nosem_t1_v2/representation.pt --realizer artifacts/runs/legged_rz_nosem_t1_dag1/realizer.pt --bc $B --oracle-bc --posthoc-probe artifacts/runs/legged_rep_nosem_t1_v2/probe_posthoc.pt" &
wait
