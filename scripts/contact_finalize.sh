#!/usr/bin/env bash
# Finalize one contact_v2 tracker run (host): install actor, validate final + alpha0 in v2 physics, render videos.
# usage: scripts/contact_finalize.sh <body> <run dir>   (run inside `rrp ops run --gpu`; needs PYLIB with imageio/pillow)
set -euo pipefail
B=$1; RUN=$2
PY=${PY:-$HOME/work/relational-robot-policy/.venv/bin/python}
mkdir -p artifacts/trackers/$B/contact_v2
cp "$RUN/actor.pt" artifacts/trackers/$B/contact_v2/actor.pt
python3 -c "import json,sys; o=open(sys.argv[2],\"w\"); [o.write(l) for l in open(sys.argv[1]) if (lambda r: r[\"iter\"] % 10 == 0 or r.get(\"gate\"))(json.loads(l))]" "$RUN/train_log.jsonl" artifacts/trackers/$B/contact_v2/train_log_every10.jsonl
cp "$RUN/meta.json" artifacts/trackers/$B/contact_v2/meta.json
L=$(mktemp); echo "$B v2 v2 v2trk" > $L
[ -f "$RUN/actor_alpha0.pt" ] && echo "$B $RUN/actor_alpha0.pt v2 v2alpha0" >> $L
SEEDS=5 bash scripts/contact_validate.sh $L 2
cp artifacts/runs/contact_v2/val/${B}_v2trk_physv2.json artifacts/trackers/$B/contact_v2/validation_learned.json
export PYTHONPATH=src:${PYLIB:-}
IT=$($PY -c "import torch;print(torch.load('artifacts/trackers/$B/contact_v2/actor.pt',weights_only=False)['meta']['iter'])")
MUJOCO_GL=egl $PY scripts/render_contact_compare.py --body $B --left v1 --right v2 --cmd forward --tag "v1-vs-v2_iter$IT"
MUJOCO_GL=egl $PY scripts/render_contact_compare.py --body $B --left v1 --right v2 --cmd turn --T 6 --tag "v1-vs-v2_iter$IT"
if [ -f "$RUN/actor_alpha0.pt" ]; then
  MUJOCO_GL=egl $PY scripts/render_contact_compare.py --body $B --left "$RUN/actor_alpha0.pt:v2" --left-label "alpha=0 (priors)" \
     --right v2 --right-label "final alpha" --cmd forward --tag "alpha0-vs-final_iter$IT"
fi
sha256sum artifacts/trackers/$B/contact_v2/actor.pt
