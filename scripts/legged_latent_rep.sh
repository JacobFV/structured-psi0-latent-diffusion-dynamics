#!/usr/bin/env bash
# Stage A for the legged latent packet: latent_sem and capacity-matched latent_nosem in parallel (one lease),
# then post-hoc measurement probes (+ metadata-only control) on both frozen representations.
set -uo pipefail
PY=/dev/shm/rrp-brandonin/venv/bin/python
R=/dev/shm/rrp-brandonin/repo/artifacts/runs
for v in sem nosem; do
  [ -f $R/legged_vlm_rep_${v}_v1/representation.pt ] || $PY -m rrp.learning.legged_latent_train rep --config configs/legged_latent/rep_${v}_v1.json --out $R/legged_vlm_rep_${v}_v1 > $R/legged_vlm_rep_${v}_v1.out 2>&1 &
done
wait
for v in sem nosem; do
  echo "{\"representation\": \"$R/legged_vlm_rep_${v}_v1/representation.pt\", \"steps\": 6000}" > $R/legged_vlm_rep_${v}_v1/probe_cfg.json
  $PY -m rrp.learning.legged_latent_train probe --config $R/legged_vlm_rep_${v}_v1/probe_cfg.json --out $R/legged_vlm_rep_${v}_v1 > $R/legged_vlm_rep_${v}_v1/probe.out 2>&1 &
done
wait
