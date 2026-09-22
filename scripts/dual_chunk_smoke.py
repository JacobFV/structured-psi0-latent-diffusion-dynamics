import numpy as np
from rrp.control.dual_validate import make_session
from rrp.data.features_multi import MultiFeaturizer
from rrp.contracts.action import ActionChunk, GroupCommand
s = make_session("support_insert", "parm5_pg2__parm7_tf3", 0)
f = MultiFeaturizer(s.model, s.scenario.robots)
obs = s.observe(); pi = f(obs)
H = 4
a = np.zeros((H, f.N), np.float32); a[:, 0] = 0.2          # rotate robot-0 base joint by +0.1 rad
rows = f.aspace.denormalize(a, pi.q0)
gn = sorted(set(f.aspace.node_group), key=f.aspace.node_group.index)
cg = [GroupCommand(group=g, values=np.array([r[g] for r in rows], float), mask=np.ones((H, len(rows[0][g])), bool)) for g in gn]
ch = ActionChunk(observation_id=obs.observation_id, graph_version=s.runtime.graph_version, runtime_version=s.runtime.runtime_version,
                 robot_spec_hash=pi.meta["spec_hash"], controller_version=s.multi_controller_version, policy_version="chk",
                 codec_version=None, start_time=float(s.data.time), dt=s.dt, horizon=H, command_groups=cg, sampling_seed=None, source="learned")
q_before = s.data.qpos[s.robots[0].qadr[0]]
s.submit_chunk(ch)
for _ in range(H):
    out = s.step(None)
print("source", out.source, "commands robots", sorted(out.commands), "q0 moved", round(float(s.data.qpos[s.robots[0].qadr[0]] - q_before), 4))
