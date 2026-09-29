"""D-141 offline diagnosis (scratch, not committed): held-out TabletopGraspMP frames, cached frozen-VLM features."""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from rrp.policies.psi0 import load_launch_config, psi_runtime
from rrp.policies.psi0 import nets as N
from rrp.policies.psi0.data import CachedDataset, collate
from rrp.policies.psi0.train import to_dev, GROUPS
psi_runtime()
R = Path(os.path.expanduser("~/work/ext/runs/psi1z")); T = "G1WholebodyTabletopGraspMP-v0"
run = Path(os.path.expanduser("~/work/ext/psi_home/cache/checkpoints/psi0/simple-checkpoints/g1wholebodytabletopgrasp-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2603181503"))
lc = load_launch_config(run); mm = lc.data.transform.field; mcfg = lc.model
summ = json.loads((R / "train/tabletop_structured_v2e/summary.json").read_text())
va = set(summ["val_eps"]); print("val eps", sorted(va), flush=True)
ds = CachedDataset(R / f"features/{T}", R / f"replay_labels/{T}", episodes=va, max_j=1)
idx = [i for i, it in enumerate(ds.items) if it["fr"] % 4 == 0]
A = N.load_stage_a(R / "train/tabletop_stageA_v2e/stage_a.pt"); zs = torch.load(R / "train/tabletop_stageA_v2e/z_stats.pt")
H = N.StructuredHead(mcfg, A, zs["mean"], zs["std"]); N.load_tolerant(H, torch.load(R / "train/tabletop_structured_v2e/final.pt", weights_only=False)["model"])
H = H.cuda().eval()
Dh = N.DirectHead(mcfg); Dh.load_state_dict(torch.load(R / "train/tabletop_direct_s0/final.pt", weights_only=False)["model"]); Dh = Dh.cuda().eval()
g = torch.Generator(device="cuda").manual_seed(0)
acc = {k: [] for k in ("gen", "oracle", "direct", "gen_fp32", "zerr_asm", "zerr_asm_fp32", "zgen_norm", "ztgt_norm")}
for b0 in range(0, len(idx), 32):
    b = to_dev(collate([ds[i] for i in idx[b0:b0 + 32]]), "cuda")
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu, _ = H.A.E(H.A.morph, b["state0"], b["actions"])
            zg = H.sample_z(b, nfe=10, generator=g)
            ag = H.realize(zg, b["state0"], torch.zeros(zg.shape[0], device="cuda")).float()
            ao = H.realize(mu, b["state0"], torch.zeros(mu.shape[0], device="cuda")).float()
            ad = Dh.sample(b, nfe=10, generator=g).float()
        b32 = dict(b, hidden=b["hidden"].float())
        zg32 = H.sample_z(b32, nfe=10, generator=g).float()        # fp32 system i (no autocast)
        ag32 = H.realize(zg32, b["state0"], torch.zeros(zg32.shape[0], device="cuda")).float()
    gt = np.asarray(mm.denormalize(b["actions"].float().cpu().numpy()))
    msk = b["amask"].cpu().numpy() > 0
    for k, a in (("gen", ag), ("oracle", ao), ("direct", ad), ("gen_fp32", ag32)):
        p = np.asarray(mm.denormalize(a.cpu().numpy()))
        acc[k].append(np.where(msk, np.abs(p - gt), np.nan)[:, :24].reshape(-1, 36)); acc[k + "_bias"] = acc.get(k + "_bias", []) + [(p - gt)[:, :24].reshape(-1, 36)]
    nz = lambda z: ((z.float() - H.z_mean) / H.z_std)
    acc["zerr_asm"].append((nz(zg) - nz(mu.float())).pow(2).mean((1, 3)).cpu().numpy())
    acc["zerr_asm_fp32"].append((nz(zg32) - nz(mu.float())).pow(2).mean((1, 3)).cpu().numpy())
    acc["zgen_norm"].append(nz(zg).pow(2).mean((1, 3)).cpu().numpy()); acc["ztgt_norm"].append(nz(mu.float()).pow(2).mean((1, 3)).cpu().numpy())
out = {"frames": len(idx), "val_eps": sorted(va)}
for k in ("gen", "oracle", "direct", "gen_fp32"):
    E = np.concatenate(acc[k]); B = np.concatenate(acc[k + "_bias"])
    out[k] = {n: dict(l1=round(float(np.nanmean(E[:, s:t])), 4), bias=round(float(B[:, s:t].mean()), 4)) for n, s, t in GROUPS}
    out[k]["waist_pitch_bias"] = round(float(B[:, 29].mean()), 4)
for k in ("zerr_asm", "zerr_asm_fp32", "zgen_norm", "ztgt_norm"):
    out[k] = dict(zip(("l_hand", "r_hand", "l_arm", "r_arm", "torso", "base"), np.concatenate(acc[k]).mean(0).round(3).tolist()))
print(json.dumps(out, indent=1)); Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
