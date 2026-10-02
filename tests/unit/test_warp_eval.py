"""warpeval (research/tracks/compute.md, docs/architecture.md 14.8): the batched Warp evaluation backend. CPU-only tests of the
plumbing (grouping, fallbacks with recorded reasons, per-episode legged policy copies, the default path adding nothing to rows) and one
Warp-on-CPU-device parity smoke (slow). The measured parity / throughput numbers are in research/tracks/compute.md."""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")

from rrp.envs.mujoco.session import Integrate  # noqa: E402


def test_integrate_schedule_and_energy_accumulation():
    r = Integrate(3, np.arange(6.0).reshape(3, 2), energy_act=np.array([0]))
    assert list(r.ctrl_at(2)) == [4.0, 5.0]
    c = Integrate(3, np.array([1.0, 2.0]))
    assert list(c.ctrl_at(2)) == [1.0, 2.0]                      # constant ctrl over the round


def _xml(n_geoms=1, z=0.1):
    geoms = "".join(f'<geom type="sphere" size="0.05" pos="{0.2 * k} 0 0"/>' for k in range(n_geoms))
    return f'<mujoco><worldbody><body name="b" pos="0 0 {z}"><freejoint/>{geoms}</body></worldbody></mujoco>'


def test_group_models_batches_only_per_world_differences():
    pytest.importorskip("mujoco_warp")
    from rrp.envs.warp.batch_sim import group_models
    a, b = (mujoco.MjModel.from_xml_string(_xml(1, z)) for z in (0.1, 0.3))
    c = mujoco.MjModel.from_xml_string(_xml(2))
    assert group_models([a, b, c, a]) == [[0, 1, 3], [2]]       # same structure with a different body_pos: one group; another geom count: another


class _StubPolicy:
    from rrp.policies.base import PolicyInfo, Requirements
    info = PolicyInfo("stub", "scripted_teacher", "v", Requirements(frozenset()))

    def reset(self, spec, task, seeds, *, envs=None):
        pass

    def act(self, obs):
        raise AssertionError("never reached: the group is refused or ends first")


def test_unbatchable_group_falls_back_to_cpu_and_records_why():
    """A hook that declares itself batch-unsafe, and an env that is not a MuJoCo Session, run on CPU with the reason in the row."""
    from rrp.harness import rollout as R
    from rrp.harness.eval import tracker_validation as tv
    from rrp.envs.warp.batch_sim import env_reason
    from rrp.bodies.legged import legged_body, standalone_model
    from rrp.envs.mujoco.legged_core import LeggedBinding
    from rrp.envs.mujoco.legged_tracker import CPGTracker

    model, _, meta = standalone_model(legged_body("hexapod6"), contact="v1")
    b = LeggedBinding(model, meta)
    b.min_h = 10.0
    env = tv._BenchEnv(model, b, CPGTracker(b, meta), 1)
    assert "no step generator" in env_reason(env)

    class Kick:
        batch_unsafe = "applies external forces"
    assert "Kick" in R.batch_reason([], _StubPolicy(), [Kick()])
    eps = R.rollout(lambda sd: tv._BenchEnv(model, b, CPGTracker(b, meta), sd), tv._ScriptedCommand([0.1, 0, 0]), tv._tracker_task(),
                    [1, 2], batch=2, max_steps=5, hooks=[tv._Meter(5, [0.1, 0, 0])], eval_backend="warp")
    for e in eps:
        n = e.provenance["eval_backend"]
        assert (n["requested"], n["effective"]) == ("warp", "cpu") and "no step generator" in n["fallback"]
    cpu = R.rollout(lambda sd: tv._BenchEnv(model, b, CPGTracker(b, meta), sd), tv._ScriptedCommand([0.1, 0, 0]), tv._tracker_task(),
                    [1, 2], batch=2, max_steps=5, hooks=[tv._Meter(5, [0.1, 0, 0])])
    assert all("eval_backend" not in e.provenance for e in cpu)              # the default path adds nothing to rows
    assert [(e.outcome, e.steps) for e in cpu] == [(e.outcome, e.steps) for e in eps]


def test_legged_policy_batch_gives_each_episode_its_own_controller_and_stream():
    from rrp.policies import legged as L

    class Ctl:
        dev = torch.device("cpu")
        edit, t_edit = "none", None

        def __init__(self):
            self.gen = torch.Generator().manual_seed(5)
            self.trace, self.packets = [], []

        def bind(self, s, m):
            self.trace, self.packets = [], []
            self.bound = s

    class Pol(L._LeggedPolicy):
        def reset(self, spec, task, seeds, *, envs=None):
            if len(envs) != 1:
                return super().reset(spec, task, seeds, envs=envs)
            self.ctl.bind(envs[0], None)                              # the batch=1 body needs a real session; the stub only binds
            self.env = envs[0]

    p = Pol(Ctl(), None)
    p.reset(None, None, [3, 4, 5], envs=["a", "b", "c"])
    assert [s.ctl.bound for s in p.subs] == ["a", "b", "c"] and len({id(s.ctl) for s in p.subs}) == 3
    draws = [torch.rand(2, generator=s.ctl.gen).tolist() for s in p.subs]
    assert draws[0] != draws[1] != draws[2]
    q = Pol(Ctl(), None)
    q.reset(None, None, [4], envs=["b"])                              # the same seed alone: batch=1 keeps the controller's own stream
    q.reset(None, None, [9, 4], envs=["z", "b"])
    assert torch.rand(2, generator=q.subs[1].ctl.gen).tolist() == draws[1]   # a seed's stream does not depend on its batch neighbours


@pytest.mark.slow
def test_warp_hybrid_matches_cpu_on_the_hexapod_fixture_warp_cpu_device():
    pytest.importorskip("mujoco_warp")
    from rrp.harness.eval.warp_parity import FIXTURES, closed_loop, single_step
    fx = dict(FIXTURES["F2"], seeds=[0, 1])
    s = single_step(fx, "cpu")
    assert s["ok"], s
    c = closed_loop(fx, "cpu")
    assert c["outcomes_identical"] and c["within_tolerance"], c["base_xy_err"]
