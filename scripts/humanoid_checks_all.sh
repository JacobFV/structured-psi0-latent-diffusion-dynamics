#!/usr/bin/env bash
# W13 GPU env regression checks (peer, one GPU lease): obs parity (t1, apollo), phum variant parity, morph obs parity.
export PYTHONPATH=src:/home/brandonin/work/ext/pylibs/mjwarp
PY=/dev/shm/rrp-brandonin/venv/bin/python
for b in t1 apollo; do $PY scripts/humanoid_warp_obs_parity.py $b 2>&1 | grep -E '^\{|Error' | tail -1; done
$PY scripts/humanoid_variant_parity.py 2>&1 | grep -E '^\{|Error' | tail -1 | cut -c1-200
$PY scripts/humanoid_morph_obs_parity.py 2>&1 | grep -E '^\{|Error' | tail -1
