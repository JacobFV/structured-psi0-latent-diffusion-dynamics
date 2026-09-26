#!/usr/bin/env bash
# Labelled clips of notable legged results (peer, EGL). Writes to artifacts/runs/legged_video_tmp/<batch>; copy + INDEX by hand.
set -uo pipefail
PY=${PY:-python}; V=artifacts/runs/legged_video_tmp/$1; shift; mkdir -p $V
E="$PY -m rrp.evaluation.legged_latent_eval --video-dir $V --video-n 1"
while [ $# -gt 0 ]; do eval "PYTHONPATH=src $E $1 --out $V/rows.jsonl" > /dev/null 2>&1 || echo "FAIL $1"; shift; done
ls $V
