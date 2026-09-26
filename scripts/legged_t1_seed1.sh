#!/usr/bin/env bash
# second training seed for t1 (sem vs nosem robustness): stage A then flows, host GPU
set -uo pipefail
bash scripts/legged_rep_pair.sh t1_v2s1
bash scripts/legged_flow_pair.sh t1_v2s1
