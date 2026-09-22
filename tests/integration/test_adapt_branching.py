"""Full-state deterministic branching at the public `grasp` boundary + GRPO ratio==1 on the real FlowPolicy."""
import numpy as np
import torch

from rrp.control.teachers import PickPlaceTeacher
from rrp.learning.flow_sde import SDEConfig
from rrp.learning.grpo import GRPOConfig, GRPOLearner
from rrp.learning.rollout import SDEPolicy, EpisodeState, drive
from rrp.model.flow import FlowPolicy, PolicyConfig
from rrp.morphology.catalog import workbench_robots
from rrp.sim.native import Session
from rrp.sim.scenario import BUILDERS


def _session(seed=2000001):
    sc = BUILDERS["pick_place"](workbench_robots()["xarm7_pg2"](), seed, n_distractors=1)
    return Session(sc, seed=seed)


def test_branch_from_grasp_boundary_is_deterministic():
    s = _session()
    teacher = PickPlaceTeacher(s)
    for _ in range(400):
        s.step(teacher.act())
        if s.runtime.status("grasp") == "succeeded":
            break
    assert s.runtime.status("grasp") == "succeeded"
    snap = s.snapshot()
    rows = [teacher.act() for _ in range(1)]  # a native command to replay identically
    outs = []
    for _ in range(2):
        b = Session(s.scenario, seed=s.seed)
        b.restore(snap)
        assert b.runtime.status("grasp") == "succeeded"
        traj = []
        for k in range(30):
            r = b.step(rows[0])
            traj.append(np.concatenate([r.qpos, [len(r.observation.predicate_estimates)]]))
        outs.append((np.stack(traj), b.tracker.state(), b.runtime.runtime_version))
    assert np.array_equal(outs[0][0], outs[1][0]) and outs[0][2] == outs[1][2]


def test_real_policy_ratio_is_one_at_identical_weights():
    torch.manual_seed(0)
    m = FlowPolicy(PolicyConfig(width=32, heads=2, ctx_layers=1, blocks=1, horizon=16))
    torch.nn.init.normal_(m.out.weight, std=0.05)
    lr = GRPOLearner(m, GRPOConfig(epochs=1, minibatch=5), "cpu", version_prefix="t")
    pol = SDEPolicy(m, None, "cpu", SDEConfig(nfe=4), version=lr.version)
    sts = [EpisodeState(_session(2000000 + i), i, 20) for i in range(3)]
    drive(pol, sts)
    recs = [c for st in sts for c in st.chunks]
    samples = [dict(pi=c["pi"], path=c["path"], adv=float(i % 2) * 2 - 1) for i, c in enumerate(recs)]
    before = [p.detach().clone() for p in lr.params]
    ctx_before = [p.detach().clone() for p in m.context.parameters()]
    st = lr.update(samples)
    assert st["first_pass_max_abs_ratio_minus_1"] < 1e-4
    assert any(not torch.equal(a, p) for a, p in zip(before, lr.params))           # trainable moved
    assert all(torch.equal(a, p) for a, p in zip(ctx_before, m.context.parameters()))  # frozen encoder
