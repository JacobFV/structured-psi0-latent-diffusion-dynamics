"""Diagnostics for a BC policy: open-loop chunk error vs teacher, and closed-loop TCP-cube traces."""
import json, sys, numpy as np, torch, mujoco
from pathlib import Path
from rrp.policy.runner import LearnedPolicy
from rrp.learning.data import load_episodes, ChunkDataset
from rrp.model.batch import collate_inputs
ck, ds = sys.argv[1], Path(sys.argv[2])
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    from rrp.ops.gpu import apply_cap; apply_cap()
pol = LearnedPolicy.from_checkpoint(ck, device=dev)
m = pol.model
for robot in ("panda_pg2", "parm6_pg2"):
    eps = load_episodes(ds, robots={robot}, limit_per_robot=3)
    d = ChunkDataset(eps, m.cfg.horizon, stride=8)
    errs, zero, hold = [], [], []
    for batch, a, v, lab, eff in d.batches(64, __import__("random").Random(0), shuffle=False, drop_last=False):
        b = batch.to(dev)
        with torch.no_grad():
            c = m.prepare(b); z = m.sample(c, m.cfg.horizon, nfe=8)[..., 0].cpu()
        mm = (v & batch.node_mask[:, None, :]).float()
        errs.append(float(((z - a[..., 0]) ** 2 * mm).sum() / mm.sum()))
        zero.append(float(((a[..., 0]) ** 2 * mm).sum() / mm.sum()))
        hold.append(float(((a[..., 0] - a[:, :1, :, 0]) ** 2 * mm).sum() / mm.sum()))
    print(robot, "open-loop MSE policy", np.mean(errs), "zero-action", np.mean(zero), "hold-first", np.mean(hold), flush=True)
# closed-loop trace
from rrp.morphology.catalog import workbench_robots
from rrp.sim.scenario import build_pick_place
from rrp.sim.native import Session
robot = workbench_robots()["panda_pg2"]()
for sd in (5, 6):
    s = Session(build_pick_place(robot, sd, n_distractors=0), seed=sd)
    r = s.robots[0]
    site = r.tcp_sites[next(a.id for a in r.spec.assemblies if a.kind in ("gripper", "hand"))]
    sid = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_SITE, site)
    trace = []
    for k in range(200):
        if not s.executor.queue:
            s.submit_chunk(pol.chunks([s])[0], execute_prefix=8)
        s.step(None)
        if k % 10 == 0:
            cube = s.data.xpos[s.model.body("cube").id]
            g = s.data.ctrl[r.controller.act_ids["gripper"][0]]
            trace.append((k, round(float(np.linalg.norm(s.data.site_xpos[sid] - cube)), 3), round(float(s.data.site_xpos[sid][2]), 3), round(float(g), 3)))
    print("seed", sd, "(step, tcp-cube dist, tcp z, grip cmd):", trace, flush=True)
