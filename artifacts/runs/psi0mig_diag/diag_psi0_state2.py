import glob, json, os, sys
import numpy as np, pandas as pd
from pathlib import Path
from rrp.policies.psi0 import load_launch_config
run = Path(os.path.expanduser("~/work/ext/psi_home/cache/checkpoints/psi0/simple-checkpoints/g1wholebodytabletopgrasp-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2603181503"))
mm = load_launch_config(run).data.transform.field
base = os.path.expanduser("~/work/ext/psi_home/data/simple/G1WholebodyTabletopGraspMP-v0/data")
St = np.concatenate([np.stack(pd.read_parquet(f)["states"].to_numpy()) for f in sorted(glob.glob(base + "/*/*.parquet"))])
np.set_printoptions(precision=4, suppress=True)
print("train raw state 28:32 min", St[:, 28:32].min(0), "max", St[:, 28:32].max(0), "n", len(St))
for k in ("state_min", "state_max", "min", "max", "state_stats"):
    v = getattr(mm, k, None)
    if v is not None: print(k, np.asarray(v).reshape(-1)[26:32] if hasattr(v, "__len__") else v)
print([a for a in dir(mm) if "state" in a.lower() or "norm" in a.lower()][:40])
for raw in ([0, 0, 0, 0.75], [0, 0.09, 0, 0.75], [0, -0.001, 0, 0.75], list(St[0, 28:32])):
    x = np.zeros((1, 36), np.float32); x[0, :28] = St[0, :28]; x[0, 28:32] = raw
    print("raw torso", np.round(raw, 4), "-> normalized", np.asarray(mm.normalize_state_func(x))[0, 28:32])
