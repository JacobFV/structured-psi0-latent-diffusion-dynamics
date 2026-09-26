#!/usr/bin/env bash
# Demo sprint: scripted-teacher clips (privileged) into artifacts/runs/demo_video/ (not the shared INDEX).
# Run on the peer from the demo peer dir: scripts/peer_run.sh --gpu --gpu-mem 4G ... -- bash scripts/demo/render_teacher.sh
set -uo pipefail
PY=${PY:-/dev/shm/rrp-brandonin/venv/bin/python}
O=artifacts/runs/demo_video
mkdir -p $O
$PY scripts/render_episode.py --robot panda_pg2 --seeds 3000002 --source scripted_teacher --max-steps 500 --out $O
$PY scripts/render_episode.py --robot parm6_tf3 --seeds 3000003 --source scripted_teacher --max-steps 500 --out $O
for p in 0 1 2; do
  $PY scripts/render_episode.py --robot panda_pg2 --task pick_place_paired --patient $p --seeds 3000001 \
      --source scripted_teacher --max-steps 500 --out $O
done
