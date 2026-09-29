"""D-141 (scratch): R's output on LOGGED closed-loop states, torso-pitch state dim at the closed-loop value (+1) vs the
training constant (-1); packet fixed (z = dataset mean E(a), since system 0 barely reads it). Denormalized outputs."""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from rrp.policies.psi0 import load_launch_config
from rrp.policies.psi0 import nets as N
R = Path(os.path.expanduser("~/work/ext/runs/psi1z"))
run = Path(os.path.expanduser("~/work/ext/psi_home/cache/checkpoints/psi0/simple-checkpoints/g1wholebodytabletopgrasp-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2603181503"))
mm = load_launch_config(run).data.transform.field
A = N.load_stage_a(R / "train/tabletop_stageA_v2e/stage_a.pt").eval(); zs = torch.load(R / "train/tabletop_stageA_v2e/z_stats.pt")
out = {}
for arm in ("structured", "direct"):
    rows = []
    for e in range(10):
        d = np.load(R / f"cl/step2_tabletop_{arm}/part0/simple_eval/ep{e}_r0.npz", allow_pickle=True)
        x = np.zeros((1, 36), np.float32); x[0, :32] = d["q_states"][0]
        s_cl = torch.as_tensor(np.asarray(mm.normalize_state_func(x), np.float32))
        s_tr = s_cl.clone(); s_tr[0, 29] = -1.0
        with torch.no_grad():
            a = [np.asarray(mm.denormalize(A.R(A.morph, zs["mean"][None], s, torch.zeros(1)).numpy()))[0, :24] for s in (s_cl, s_tr)]
        rows.append(dict(waist_pitch_cl=float(a[0][:, 29].mean()), waist_pitch_trainconst=float(a[1][:, 29].mean()),
                         r_shoulder_pitch_end_cl=float(a[0][-1, 21]), r_shoulder_pitch_end_trainconst=float(a[1][-1, 21]),
                         logged_policy_waist_pitch=float(d["q_pred"][0][:, 29].mean())))
    out[arm] = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}
print(json.dumps(out, indent=1)); Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
