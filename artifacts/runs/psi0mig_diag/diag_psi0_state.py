"""D-141 (scratch): closed-loop states vs training states in NORMALIZED units (what R and the heads see)."""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from rrp.policies.psi0 import load_launch_config
R = Path(os.path.expanduser("~/work/ext/runs/psi1z"))
run = Path(os.path.expanduser("~/work/ext/psi_home/cache/checkpoints/psi0/simple-checkpoints/g1wholebodytabletopgrasp-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2603181503"))
mm = load_launch_config(run).data.transform.field
small = torch.load(R / "features/G1WholebodyTabletopGraspMP-v0/small.pt", weights_only=False)
S = torch.stack([s[:36] for s in small["state"]]).numpy()           # normalized training states
fr = np.array(small["fr"]); S0 = S[fr == 0]
def norm(raw32):
    x = np.zeros((len(raw32), 36), np.float32); x[:, :32] = raw32
    return np.asarray(mm.normalize_state_func(x), np.float32)
out = {"train_state_norm": {"mean": S.mean(0)[26:32].round(3).tolist(), "p1": np.percentile(S, 1, 0)[26:32].round(3).tolist(),
                            "p99": np.percentile(S, 99, 0)[26:32].round(3).tolist(), "frame0_mean": S0.mean(0)[26:32].round(3).tolist()},
       "dims": "26,27 = right wrist pitch/yaw; 28,29,30 = torso roll/pitch/yaw (last command); 31 = height"}
for arm in ("structured", "direct"):
    d = np.load(R / f"cl/step2_tabletop_{arm}/part0/simple_eval/ep0_r0.npz", allow_pickle=True)
    q = norm(d["q_states"])
    out[arm] = {f"q{k}": q[k][26:32].round(3).tolist() for k in (0, 1, 2, 4, 8)}
    z = (q[:, :32] - S[:, :32].mean(0)) / (S[:, :32].std(0) + 1e-6)
    out[arm]["max_|z|_by_query"] = np.abs(z).max(1)[:10].round(1).tolist()
    out[arm]["argmax_dim_by_query"] = np.abs(z).argmax(1)[:10].tolist()
print(json.dumps(out, indent=1)); Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
