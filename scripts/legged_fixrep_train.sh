#!/usr/bin/env bash
# LEGGED FIXED-SEM REPLICATION: Stage A (+ post-hoc probe for nosem) then flow, for each TAG in parallel (one GPU lease).
# usage: legged_fixrep_train.sh TAG... (TAG = {fixsem,nosem}_{go2,hexapod6}_s{1,2}); exit 0 only if every chain finished.
set -uo pipefail
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
chain() { t=$1; R=artifacts/runs/legged_fixrep_rep_$t; F=artifacts/runs/legged_fixrep_flow_$t
  [ -f $R/representation.pt ] || PYTHONPATH=src $PY -m rrp.learning.legged_latent_train rep --config configs/legged_fixsem/rep_$t.json --out $R > $R.out 2>&1 || return 1
  if [[ $t == nosem_* ]]; then
    echo "{\"representation\": \"$R/representation.pt\", \"steps\": 4000}" > $R/probe_cfg.json
    [ -f $R/probe_posthoc.pt ] || PYTHONPATH=src $PY -m rrp.learning.legged_latent_train probe --config $R/probe_cfg.json --out $R > $R/probe.out 2>&1 || return 2
  fi
  [ -f $F/policy.pt ] || PYTHONPATH=src $PY -m rrp.learning.legged_latent_train flow --config configs/legged_fixsem/flow_$t.json --out $F > $F.out 2>&1 || return 3
}
pids=(); for t in "$@"; do chain $t & pids+=($!); done
rc=0; for i in "${!pids[@]}"; do wait ${pids[$i]} || { echo "FAIL chain ${!i}"; rc=1; }; done
for t in "$@"; do [ -f artifacts/runs/legged_fixrep_flow_$t/policy.pt ] || { echo "missing flow $t"; rc=1; }; done
exit $rc
