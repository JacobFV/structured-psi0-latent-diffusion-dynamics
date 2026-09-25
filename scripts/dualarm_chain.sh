#!/usr/bin/env bash
# dualarm track GPU chain (one GPU lease, serial, resumable: finished steps are skipped).
# Stage A sem/nosem (identical treatment) -> post-hoc probes (sem, nosem, metadata-only) -> Stage B flows sem/nosem.
# Run on the peer from the dualarm peer dir:  scripts/peer_run.sh --gpu ... -- bash scripts/dualarm_chain.sh
set -uo pipefail
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
PK=artifacts/packed/dualarm_latent_v1_H16
run() { echo "[chain] $(date +%T) $*"; "$@" || { echo "[chain] FAILED: $*"; exit 1; }; }
for v in sem nosem; do
  [ -f artifacts/runs/dualarm_latent_${v}_v1/representation.pt ] || \
    run $PY -m rrp.cli latent train-representation --config configs/latent/rep-dualarm_latent_${v}_v1.json
done
for v in sem nosem; do
  [ -f artifacts/runs/dualarm_latent_${v}_v1/probe_posthoc.pt ] || \
    run $PY -m rrp.cli latent fit-probes --representation artifacts/runs/dualarm_latent_${v}_v1/representation.pt \
        --packed-dir $PK --out artifacts/runs/dualarm_latent_${v}_v1/probe_posthoc.pt --steps 6000
done
[ -f artifacts/runs/dualarm_latent_sem_v1/probe_metadata_only.pt ] || \
  run $PY -m rrp.cli latent fit-probes --representation artifacts/runs/dualarm_latent_sem_v1/representation.pt \
      --packed-dir $PK --out artifacts/runs/dualarm_latent_sem_v1/probe_metadata_only.pt --steps 6000 --metadata-only
for v in sem nosem; do
  [ -f artifacts/runs/dualarm_flow_${v}_v1/policy.pt ] || \
    run $PY -m rrp.cli latent train-flow --config configs/latent/flow_dualarm_${v}_v1.json
done
echo "[chain] $(date +%T) done"
