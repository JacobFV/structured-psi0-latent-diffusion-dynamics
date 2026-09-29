#!/usr/bin/env bash
# W13 P1: gate + labelled videos for one GPU-trained tracker (peer, inside a CPU lease; MUJOCO_GL=egl).
# usage: scripts/humanoid_tracker_finalize.sh <body> <actor.pt> <out_dir> <tag>
# Videos (artifacts/video, INDEX.md): installed contact_v2 tracker (left, if any) vs this actor (right), forward + turn.
set -uo pipefail
body=$1; actor=$2; out=$3; tag=$4
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
bash scripts/humanoid_tracker_gate.sh "$body" "$actor" "$out"
export PYTHONPATH=src MUJOCO_GL=egl
left=v2; [ -f "artifacts/trackers/$body/contact_v2/actor.pt" ] || left="$actor:v2"
for c in forward turn; do
  $PY scripts/render_contact_compare.py --body "$body" --left "$left" --right "$actor:v2" --cmd $c --T 8 \
      --tag "${c}_installed-vs-${tag}" > "$out/render_$c.log" 2>&1; echo "render $c rc=$?"
done
