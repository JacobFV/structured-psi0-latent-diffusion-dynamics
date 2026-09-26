"""LEGGED FIXED-SEM RERUN diagnostic: Stage-A update scale (from train logs) and packet geometry (posterior sigma,
participation ratio, KL) for original sem, fixed sem (probe_lv_min -4) and nosem, on held-out teacher rows.
usage: PYTHONPATH=src python scripts/legged_fixsem_geom.py BODY OUT.json"""
import json, sys
from pathlib import Path

import numpy as np
import torch

from rrp.learning.legged_latent_train import LeggedData, load_rep, _dev
from rrp.learning.legged_t1_diag import latent_geometry

body, out = sys.argv[1], sys.argv[2]
runs = {"sem_orig": f"artifacts/runs/legged_rep_sem_{body}_v2", "sem_fixed": f"artifacts/runs/legged_fixsem_rep_sem_{body}_lv4",
        "nosem": f"artifacts/runs/legged_rep_nosem_{body}_v2"}
dev = _dev()
res, data = {}, None
for k, d in runs.items():
    gn = [json.loads(l)["gn"] for l in open(f"{d}/train_log.jsonl") if '"gn"' in l]
    gn = np.asarray(gn, dtype=float)
    rcfg, E, R, P, rres = load_rep(Path(f"{d}/representation.pt"), dev)
    if data is None:
        data = LeggedData(Path(rcfg["data"]), rcfg["bodies"], dev)
    torch.manual_seed(0)
    with torch.no_grad():
        geo = latent_geometry(dict(E=E), data)
    res[k] = dict(run=d, median_grad_norm=float(np.median(gn)), mean_update_scale=float(np.minimum(1, 1 / gn).mean()),
                  eval=json.load(open(f"{d}/result.json"))["eval"], geometry=geo)
    print(k, json.dumps({kk: res[k][kk] for kk in ("median_grad_norm", "mean_update_scale")}), json.dumps(geo), flush=True)
Path(out).write_text(json.dumps(res, indent=1))
