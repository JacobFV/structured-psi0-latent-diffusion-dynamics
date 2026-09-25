#!/usr/bin/env bash
# Stage B (system i flows) for legged latent_sem / latent_nosem in parallel inside ONE GPU lease.
# usage: legged_latent_flow.sh SUFFIX(v1) [steps]
set -uo pipefail
PY=/dev/shm/rrp-brandonin/venv/bin/python
R=/dev/shm/rrp-brandonin/repo/artifacts/runs
V=${1:-v1}
for v in sem nosem; do
  [ -f $R/legged_vlm_flow_${v}_$V/policy.pt ] || $PY -m rrp.learning.legged_latent_train flow --config configs/legged_latent/flow_${v}_$V.json --out $R/legged_vlm_flow_${v}_$V ${2:+--steps $2} > $R/legged_vlm_flow_${v}_$V.out 2>&1 &
done
wait
