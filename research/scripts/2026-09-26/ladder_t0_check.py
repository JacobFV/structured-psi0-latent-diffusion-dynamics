"""Stage-A realization error on the training pack at episode start (t=0) vs t=1 vs t>=5, j=0, per robot, arm/gripper,
with the stored inputs (prev-action column as packed) and with it zeroed (deployment input). Also z norms."""
import json, sys
import numpy as np
import torch
from pathlib import Path
from rrp.learning.latent_train import load_representation, LatentData
from rrp.model.semantic_latent import assembly_tokens
from rrp.evaluation.ladder import _gripper_mask
rep, robot = sys.argv[1], sys.argv[2]
dev = "cpu"
lcfg, E, R, P, res = load_representation(Path(rep), dev)
out = {}
for zp in (False, True):
    data = LatentData(Path("artifacts/packed/latent_pp_v3dart_s1_H16"), zero_prev_action=zp)
    rid = data.ds.meta["robot_ids"][robot]
    rob = np.asarray(data.ds.arr["robot_id"]) == rid
    t = data.t
    rng = np.random.default_rng(0)
    for name, cond in (("t0", t == 0), ("t1", t == 1), ("t5+", t >= 5)):
        idx = np.sort(rng.choice(np.nonzero(rob & cond)[0], 96, replace=False))
        with torch.no_grad():
            batch, a, v, lab, r = data.fetch(idx, idx, dev)
            af, am, ai = assembly_tokens(batch)
            mu, _ = E(batch, a, v, af, am, ai)
            pred = R(mu, am, torch.tensor(lcfg.knot_times), torch.zeros(len(idx)), r["node"], r["node_mask"], r["local"])
        m = (r["v1"] & r["node_mask"]).numpy()
        g = np.broadcast_to(_gripper_mask(robot, m.shape[1]), m.shape)
        e = ((pred - r["a1"]) ** 2).numpy()
        out[f"{'zero_prev' if zp else 'stored'}/{name}"] = dict(
            arm=float(e[m & ~g].mean()), grip=float(e[m & g].mean()),
            grip_pred=float(pred.numpy()[m & g].mean()), grip_label=float(r["a1"].numpy()[m & g].mean()),
            arm_label_sq=float((r["a1"].numpy()[m & ~g] ** 2).mean()),
            z_norm=float(mu.flatten(1).norm(dim=1).mean()))
for k, v in out.items():
    print(k, {a: round(b, 4) for a, b in v.items()})
Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
