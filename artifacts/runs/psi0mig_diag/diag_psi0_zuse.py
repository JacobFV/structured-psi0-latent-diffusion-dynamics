"""D-141 (scratch): does system 0 use the packet? CPU, stage A + cached states/actions only (no VLM)."""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from rrp.policies.psi0 import nets as N
from rrp.policies.psi0.data import CachedDataset, collate
torch.set_num_threads(2)
R = Path(os.path.expanduser("~/work/ext/runs/psi1z")); T = "G1WholebodyTabletopGraspMP-v0"
summ = json.loads((R / "train/tabletop_structured_v2e/summary.json").read_text())
A = N.load_stage_a(R / "train/tabletop_stageA_v2e/stage_a.pt").eval(); zs = torch.load(R / "train/tabletop_stageA_v2e/z_stats.pt")
def batch(eps, stride):
    ds = CachedDataset(R / f"features/{T}", None, episodes=set(eps), max_j=1, load_hidden=False)
    return collate([ds[i] for i, it in enumerate(ds.items) if it["fr"] % stride == 0])
bv, bt = batch(summ["val_eps"], 4), batch(summ["train_eps"], 8)
m = bv["amask"][:, :24] > 0
def l1(pred, a=bv["actions"]):
    e = (pred[:, :24] - a[:, :24]).abs()
    return {g: round(float(e[..., s:t][m[..., s:t]].mean()), 4) for g, s, t in (("hand", 0, 14), ("arm", 14, 28), ("waist", 28, 31))}
out = {"note": "L1 in NORMALIZED action units, rows 0..23, held-out TabletopGraspMP frames"}
with torch.no_grad():
    mu, _ = A.E(A.morph, bv["state0"], bv["actions"]); ph = torch.zeros(len(mu))
    out["R(E(a))"] = l1(A.R(A.morph, mu, bv["state0"], ph))
    out["R(z_mean)"] = l1(A.R(A.morph, zs["mean"].expand_as(mu).contiguous(), bv["state0"], ph))
    g = torch.Generator().manual_seed(0); perm = torch.randperm(len(mu), generator=g)
    out["R(E(a) of another frame)"] = l1(A.R(A.morph, mu[perm], bv["state0"], ph))
    out["R(N(0,1) z)"] = l1(A.R(A.morph, zs["mean"] + zs["std"] * torch.randn(mu.shape, generator=g), bv["state0"], ph))
    out["R(E(a)), state of another frame"] = l1(A.R(A.morph, mu, bv["state0"][perm], ph))
    out["hold state (repeat state0 rows)"] = l1(bv["state0"][:, None, :36].expand(-1, 30, -1))
# state-only ridge baseline: chunk from state0
X = torch.cat([bt["state0"], torch.ones(len(bt["state0"]), 1)], 1).double(); Y = bt["actions"].reshape(len(X), -1).double()
W = torch.linalg.solve(X.T @ X + 1e-3 * torch.eye(X.shape[1], dtype=X.dtype), X.T @ Y)
Xv = torch.cat([bv["state0"], torch.ones(len(bv["state0"]), 1)], 1).double()
out["ridge(state0) -> chunk"] = l1((Xv @ W).float().reshape(bv["actions"].shape))
out["n_val_frames"] = len(mu); out["n_train_frames"] = len(X)
print(json.dumps(out, indent=1)); Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
