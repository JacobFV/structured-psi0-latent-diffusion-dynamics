"""Per-tick rollout trace for closed-loop failure localization (feeds the ladder track, item 3).

For each seed, runs (a) the SCRIPTED TEACHER (privileged planner, native commands) and (b) the LEARNED latent path
(system i ODE packets -> system 0 -> tracker), optionally after a labelled teacher prefix, and logs every `every`
ticks: TCP-to-cube distance (privileged sim truth, diagnostic only), TCP speed, arm tracking error
|commanded target - measured q| (next tick), commanded-target step size, gripper command, cube height, cube-zone
distance, public event statuses. Output: JSONL rows + a printed per-phase summary.

  PYTHONPATH=src python scripts/diag_rollout_trace.py --checkpoint <policy.pt> --robot panda_pg2 \
      --seeds 3000000,3000001 --out artifacts/runs/<name>/trace.jsonl [--teacher-prefix-steps 40]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--robot", default="panda_pg2")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--teacher-prefix-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from rrp.morphology.catalog import workbench_robots
    from rrp.sim.scenario import BUILDERS
    from rrp.sim.native import Session
    from rrp.control.teachers import PickPlaceTeacher
    from rrp.control.latent_realizer import LatentSystem0
    from rrp.policy.latent_runner import LatentPolicy
    from rrp.learning.checkpoint import load_checkpoint
    from rrp.learning.latent_train import load_representation
    from rrp.evaluation.latent_eval import _tcp
    dev = "cpu"
    pol = LatentPolicy.from_checkpoint(a.checkpoint, device=dev, nfe=8)
    _, _, R, _, _ = load_representation(Path(load_checkpoint(a.checkpoint)["config"]["representation"]), dev)
    robot = workbench_robots()[a.robot]()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "w")
    for sd in [int(x) for x in a.seeds.split(",")]:
        for mode in ("scripted_teacher", "learned_latent"):
            s = Session(BUILDERS["pick_place"](robot, sd, n_distractors=sd % 3), seed=sd)
            rb = s.robots[0]
            narm = len(rb.arm_joints)
            teacher = PickPlaceTeacher(s)
            s0 = LatentSystem0(R, pol.featurizer(s), latent_space_version=pol.lsv, realizer_compat_version=pol.rcv,
                               device=dev)
            cube = s.model.body("cube").id
            zone = s.model.body("target_zone").id
            prev_tcp, prev_cmd = _tcp(s), None
            for k in range(a.max_steps):
                src = "scripted_teacher" if (mode == "scripted_teacher" or k < a.teacher_prefix_steps) else "learned"
                if src == "scripted_teacher":
                    cmd = teacher.act()
                else:
                    if k % 8 == 0 or s0.packet is None:
                        s0.receive(pol.packets([s])[0], now=float(s.data.time), graph_version=s.runtime.graph_version)
                    cmd = s0.tick(s, s.controller_version())
                arm_cmd = np.array(cmd.groups["arm"]) if cmd is not None and "arm" in cmd.groups else None
                s.step(cmd)
                q = s.data.qpos[rb.qadr[:narm]].copy()
                tcp = _tcp(s)
                if k % a.every == 0 or s.runtime.succeeded():
                    row = dict(seed=sd, mode=mode, tick=k, controller=src,
                               tcp_cube_d=float(np.linalg.norm(tcp - s.data.xpos[cube])),
                               tcp_speed=float(np.linalg.norm(tcp - prev_tcp) / s.dt),
                               track_err=None if arm_cmd is None else float(np.abs(arm_cmd[:narm] - q).max()),
                               cmd_step=None if arm_cmd is None or prev_cmd is None else float(np.abs(arm_cmd - prev_cmd).max()),
                               grip_cmd=None if cmd is None else [round(float(x), 3) for x in cmd.groups.get("gripper", [])],
                               cube_z=float(s.data.xpos[cube][2]),
                               cube_zone_d=float(np.linalg.norm(s.data.xpos[cube][:2] - s.data.xpos[zone][:2])),
                               events={e: v.status for e, v in s.runtime.instances.items()},
                               fallback_holds=s0.stats.fallback_holds)
                    fh.write(json.dumps(row) + "\n")
                prev_tcp, prev_cmd = tcp, arm_cmd
                if s.runtime.succeeded():
                    break
            print(sd, mode, "ticks", k + 1, "success", bool(s.privileged_success()),
                  {e: v.status for e, v in s.runtime.instances.items()}, flush=True)
    fh.close()


if __name__ == "__main__":
    main()
