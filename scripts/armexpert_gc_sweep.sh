#!/usr/bin/env bash
# W7 grasp contact: object friction / mass sweep (W6 semantics: x friction of cube AND finger pads; x cube mass),
# v2 teacher, grasp_v1 vs grasp_v2, 2 bodies x 30 dev seeds. One small host lease per (grasp, body, axis).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/work/relational-robot-policy/.venv/bin/python; O=artifacts/runs/armexpert_grasp/sweep; mkdir -p $O
for gc in v1 v2; do for b in parm6_tf3 panda_pg2; do for ax in fric mass; do
  f=$O/${gc}_${b}_$ax.jsonl; [ -s ${f%.jsonl}.summary.json ] && continue
  if [ $ax = fric ]; then A=(--obj-friction 1.0,0.3,0.1,0.05,0.02); else A=(--obj-mass 10,30,100); fi
  scripts/armexpert_hostrun.sh --cpu 2 --mem 4G --label axgc_sw_${gc}_${b}_$ax --max-seconds 3000 -- env PYTHONPATH=src \
    RRP_GRASP_CONTACT=$gc OMP_NUM_THREADS=1 $PY -m rrp.evaluation.teacher_quality --bodies $b --seeds 3000000-3000029 \
    --versions v2 --workers 2 --chunk 15 "${A[@]}" --out $f > ${f%.jsonl}.log 2>&1 || echo "FAILED $f"
done; done; done
echo sweep done
