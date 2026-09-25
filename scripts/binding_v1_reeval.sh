#!/usr/bin/env bash
# Re-evaluate the v1 representations under the fixed (v2) counterexample test, with (a) the original factual
# post-hoc probes and (b) post-hoc probes fitted with counterfactual-binding augmentation (--binding-cf 0.5).
set -euo pipefail
PY=${PY:-python}
O=artifacts/runs/binding_v1_reeval
PK=artifacts/packed/latent_pp_v3dart_s1_H16
for n in sem nosem; do
  R=artifacts/runs/latent_${n}_v1/representation.pt
  $PY -m rrp.cli latent counterfactuals --representation $R --probe artifacts/runs/latent_${n}_v1/probe_posthoc.pt \
    --n 60 --per-episode 2 --out $O/${n}_cf_probe_factual.json
  [ -f $O/${n}_probe_bindcf.pt ] || $PY -m rrp.cli latent fit-probes --representation $R --packed-dir $PK \
    --binding-cf 0.5 --out $O/${n}_probe_bindcf.pt
  $PY -m rrp.cli latent counterfactuals --representation $R --probe $O/${n}_probe_bindcf.pt \
    --n 60 --per-episode 2 --out $O/${n}_cf_probe_bindcf.json
done
[ -f $O/sem_probe_metadata_bindcf.pt ] || $PY -m rrp.cli latent fit-probes --representation artifacts/runs/latent_sem_v1/representation.pt \
  --packed-dir $PK --binding-cf 0.5 --metadata-only --out $O/sem_probe_metadata_bindcf.pt
echo reeval-done
