#!/usr/bin/env bash
set -uo pipefail
B=artifacts/runs/legged_bc_go2_v1/policy.pt
EDITS="rand_norm:8 rand_norm:12" bash scripts/legged_edit_suite.sh go2 r1_sem 2 "--rep artifacts/runs/legged_rep_sem_go2_v2/representation.pt --bc $B --oracle-bc" &
EDITS="rand_norm:8 rand_norm:12" bash scripts/legged_edit_suite.sh go2 r1_nosem 2 "--rep artifacts/runs/legged_rep_nosem_go2_v2/representation.pt --bc $B --oracle-bc --posthoc-probe artifacts/runs/legged_rep_nosem_go2_v2/probe_posthoc.pt" &
wait
