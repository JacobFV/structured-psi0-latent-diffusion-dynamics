"""Interface check: teacher actions executed ONLY via the learned-policy chunk path."""
import sys, numpy as np
from rrp.morphology.catalog import workbench_robots
from rrp.sim.scenario import build_pick_place
from rrp.sim.native import Session
from rrp.control.teachers import PickPlaceTeacher
from rrp.data.collect import featurizer_for
from rrp.contracts.action import ActionChunk, GroupCommand
H, P = 16, 8
for rk in sys.argv[1].split(","):
    robot = workbench_robots()[rk](); ok = 0; n = 0
    for sd in range(3000000, 3000010):
        s = Session(build_pick_place(robot, sd, n_distractors=sd % 3), seed=sd)
        t = PickPlaceTeacher(s)
        if not t.feasibility()["feasible"]:
            continue
        n += 1
        f = featurizer_for(s)
        for step in range(300):
            if not s.executor.queue:
                obs = s.observe(); pi = f(obs)
                snap = s.snapshot(); ts = t.state()
                cmds = []
                for h in range(H):
                    c = t.act(); cmds.append(c.groups); s.step(c)
                s.restore(snap); t.load(ts)
                a = f.aspace.normalize(cmds, pi.q0)              # exactly the policy target
                seq = f.aspace.denormalize(a, pi.q0)             # exactly the policy decode
                gn = sorted(set(f.aspace.node_group), key=f.aspace.node_group.index)
                cg = [GroupCommand(group=g, values=np.array([row[g] for row in seq], float),
                                   mask=np.ones((H, len(seq[0][g])), bool)) for g in gn]
                ch = ActionChunk(observation_id=obs.observation_id, graph_version=s.runtime.graph_version,
                                 runtime_version=s.runtime.runtime_version, robot_spec_hash=pi.meta["spec_hash"],
                                 controller_version=s.controller_version(), policy_version="teacher_via_chunks",
                                 codec_version=None, start_time=float(s.data.time), dt=s.dt, horizon=H,
                                 command_groups=cg, sampling_seed=None, source="scripted_teacher")
                s.submit_chunk(ch, execute_prefix=P)
                # teacher internal state must advance P steps: replay P acts
                for h in range(P):
                    t.act()
            s.step(None)
            if s.runtime.succeeded():
                break
        ok += int(s.privileged_success())
    print(rk, "teacher-through-chunk success", ok, "/", n, flush=True)
