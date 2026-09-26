#!/usr/bin/env bash
# post-hoc probes (sem + nosem, metadata-only control) then flows sem + nosem, all in parallel on one GPU lease. usage: TAG
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
T=$1
for v in sem nosem; do
  echo "{\"representation\": \"artifacts/runs/legged_rep_${v}_$T/representation.pt\", \"steps\": 4000}" > artifacts/runs/legged_rep_${v}_$T/probe_cfg.json
  [ -f artifacts/runs/legged_rep_${v}_$T/probe_posthoc.json ] || PYTHONPATH=src $PY -m rrp.learning.legged_latent_train probe --config artifacts/runs/legged_rep_${v}_$T/probe_cfg.json --out artifacts/runs/legged_rep_${v}_$T > artifacts/runs/legged_rep_${v}_$T/probe.out 2>&1 &
done
bash scripts/legged_flow_pair.sh $T &
wait
