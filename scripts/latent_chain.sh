#!/usr/bin/env bash
# After representation training: measurement probes (+ metadata-only control), then stage-B flows.
set -uo pipefail
cd /dev/shm/rrp-brandonin/repo
export PATH=/dev/shm/rrp-brandonin/bin:$PATH PYTHONPATH=src RRP_NODE=peer RRP_REPO=$PWD
PY=/dev/shm/rrp-brandonin/venv/bin/python
PK=artifacts/packed/latent_pp_v3dart_s1_H16
for n in latent_sem_v1 latent_nosem_v1; do
  until test -f artifacts/runs/$n/representation.pt; do sleep 60; done
done
for n in latent_sem_v1 latent_nosem_v1; do
  python3 -m rrp.cli ops run --gpu --gpu-mem 10G --cpu 3 --mem 16G --label probe_$n --max-seconds 7200 --detach -- \
    $PY -m rrp.cli latent fit-probes --representation artifacts/runs/$n/representation.pt --packed-dir $PK --out artifacts/runs/$n/probe_posthoc.pt
  python3 -m rrp.cli ops run --gpu --gpu-mem 20G --cpu 4 --mem 24G --label flow_$n --max-seconds 21600 --detach -- \
    $PY -m rrp.cli latent train-flow --config configs/latent/flow_$n.json
done
python3 -m rrp.cli ops run --gpu --gpu-mem 10G --cpu 3 --mem 16G --label probe_metaonly --max-seconds 7200 --detach -- \
  $PY -m rrp.cli latent fit-probes --representation artifacts/runs/latent_sem_v1/representation.pt --packed-dir $PK --out artifacts/runs/latent_sem_v1/probe_metadata_only.pt --metadata-only
echo chain-launched
