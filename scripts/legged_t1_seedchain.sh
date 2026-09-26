#!/usr/bin/env bash
# stage A pair then flows pair for a t1 seed tag (e.g. t1_v2s2)
set -uo pipefail
bash scripts/legged_rep_pair.sh $1
bash scripts/legged_flow_pair.sh $1
