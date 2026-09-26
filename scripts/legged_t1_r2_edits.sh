#!/usr/bin/env bash
# t1 deployable-route edits (final flows + DAgger-1 system 0s): z edits and task-context edits, sem vs nosem
set -uo pipefail
S="--flow artifacts/runs/legged_flow_sem_t1_v2/policy.pt --realizer artifacts/runs/legged_rz_sem_t1_dag1/realizer.pt"
N="--flow artifacts/runs/legged_flow_nosem_t1_v2/policy.pt --realizer artifacts/runs/legged_rz_nosem_t1_dag1/realizer.pt --posthoc-probe artifacts/runs/legged_rep_nosem_t1_v2/probe_posthoc.pt"
Z="none probe_yaw:0.6 probe_yaw:-0.6 probe_halt probe_goal_mirror rand_norm:4 rand_norm:8 rand_norm:12"
C="none mirror_goal mirror_active mirror_inactive halt"
EDITS="$Z" bash scripts/legged_edit_suite.sh t1 r2_sem_dag1_policy 2 "$S" &
EDITS="$Z" bash scripts/legged_edit_suite.sh t1 r2_nosem_dag1_policy 2 "$N" &
EDITS="$C" bash scripts/legged_edit_suite.sh t1 r2ctx_sem_dag1_policy 2 "$S" &
EDITS="$C" bash scripts/legged_edit_suite.sh t1 r2ctx_nosem_dag1_policy 2 "$N" &
wait
