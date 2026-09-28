"""D-118 (1): which robot geoms penetrate the task cube in DART episodes? Replays a subset of v5dart episodes (the same
generator inputs; exact replay checked) and records, per robot geom touching the cube, the max penetration (-dist), with
the teacher phase and the geom's contact parameters. Usage: armexpert_pen_bodies.py <dataset_dir> <gate replay.jsonl> <out.jsonl> [n] [grasp]"""
from __future__ import annotations

import collections
import gzip
import json
import os
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np


def main():
    ds, replay, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 120
    if len(sys.argv) > 5:
        os.environ["RRP_GRASP_CONTACT"] = sys.argv[5]
    else:
        os.environ["RRP_GRASP_CONTACT"] = json.loads((ds / "data_config.json").read_text()).get("grasp_contact", "v1")
    rows = [json.loads(l) for l in replay.read_text().splitlines()]
    dart = [r for r in rows if r["exec_noise"]]
    over = sorted([r for r in dart if r["motion"]["penetration_max_m"] > 0.003], key=lambda r: -r["motion"]["penetration_max_m"])
    rng = np.random.default_rng(0)
    pick = over[: n // 2] + [over[i] for i in rng.choice(len(over), size=min(n - n // 2, len(over)), replace=False)]
    seen, sel = set(), []
    for r in pick:
        if r["episode_id"] not in seen:
            seen.add(r["episode_id"]); sel.append(r)
    ctl = [r for r in dart if r["motion"]["penetration_max_m"] <= 0.003][:20]      # controls (should be small)
    sel += ctl
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.native import Session
    from rrp.envs.scenario import BUILDERS
    from rrp.teachers.arm_smooth import make_arm_teacher
    import rrp.data.collect as C
    cfg = json.loads((ds / "data_config.json").read_text())
    man = {e["episode_id"]: e for e in json.loads((ds / "manifest.json").read_text())["episodes"]}
    cache = {}
    with open(out, "w") as f:
        for r in sel:
            m = man[r["episode_id"]]
            rk = m["robot_key"]
            if rk not in cache:
                cache.clear(); cache[rk] = workbench_robots()[rk]()
            s = Session(BUILDERS["pick_place"](cache[rk], m["seed"], n_distractors=m["n_distractors"]), seed=m["seed"])
            M, D = s.model, s.data
            cube = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "cube")
            rob = {l.name for l in s.robots[0].spec.links}
            per = collections.defaultdict(lambda: dict(pen=0.0, phase=None))
            orig_step = s.step
            holder = {}

            def step(cmd=None, robot=0):
                res = orig_step(cmd, robot)
                t = holder.get("t")
                for i in range(D.ncon):
                    c = D.contact[i]
                    b1, b2 = M.geom_bodyid[c.geom1], M.geom_bodyid[c.geom2]
                    if cube not in (b1, b2):
                        continue
                    og = c.geom2 if b1 == cube else c.geom1
                    ob = M.body(M.geom_bodyid[og]).name
                    if ob not in rob:
                        continue
                    p = -float(c.dist)
                    e = per[M.geom(og).name or f"geom{og}"]
                    if p > e["pen"]:
                        e.update(pen=p, phase=getattr(t, "phase", None), body=ob,
                                 solref=M.geom_solref[og].tolist(), priority=int(M.geom_priority[og]),
                                 friction=M.geom_friction[og].tolist())
                return res
            s.step = step
            orig_mk = C.make_arm_teacher if hasattr(C, "make_arm_teacher") else None

            import rrp.teachers.arm_smooth as A
            real = A.make_arm_teacher

            def mk(sess, v, **kw):
                t = real(sess, v, **kw); holder["t"] = t; return t
            A.make_arm_teacher = mk
            rec = C.collect_teacher_episode(s, episode_id=m["episode_id"], exec_noise=m.get("exec_noise", 0.0),
                                            noise_seed=m["seed"], teacher_version=cfg.get("teacher_version"))
            A.make_arm_teacher = real
            stored = pickle.loads(gzip.decompress((ds / "episodes" / f"{m['episode_id']}.public.pkl.gz").read_bytes()))
            f.write(json.dumps(dict(episode_id=m["episode_id"], robot_key=rk, status=m["status"],
                                    grasp=os.environ["RRP_GRASP_CONTACT"],
                                    exact_replay=stored["actions"] == rec.public["actions"],
                                    pen_max_m=max([e["pen"] for e in per.values()] or [0.0]),
                                    per_geom={k: v for k, v in per.items()})) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
