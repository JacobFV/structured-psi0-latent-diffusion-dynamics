"""`rrp train pointer edit` realizes the edited packet on `harness.rollout` (no private env.step loop): H ticks of the
packet's system 0, the pointer read before rollout closes the env."""
from rrp.envs.base import make_env
from rrp.harness.train.pointer.data import Demos
from rrp.harness.train.pointer.diagnostics import _EditedPacketPolicy
from rrp.harness.hooks import budget_task
from rrp.harness.rollout import rollout
from rrp.policies.teachers.computerworld import CWTeacher


class _TeacherAsSystem0:
    """Stands in for LearnedSystem0 (the packet realizer): `tick(env)` is the scripted teacher's next command."""

    def __init__(self, env, task):
        self.t = CWTeacher(env, task)
        self.calls = 0

    def tick(self, env):
        self.calls += 1
        return self.t.act()


def test_edited_packet_runs_h_ticks_on_rollout_and_reports_the_end_pointer():
    task = "cw/calc_sum"
    env = make_env("computerworld", task=task, body="cw_pointer", seed=0)
    p0 = (env.pointer.u, env.pointer.v)
    pol = _EditedPacketPolicy(_TeacherAsSystem0(env, task), env)
    ep, = rollout(lambda sd: env, pol, budget_task("pointer_edit", env.spec.env_id), [0], batch=1, max_steps=Demos.H,
                  hooks=[pol])
    assert ep.outcome != "crash" and ep.steps == Demos.H == pol.s0.calls
    assert pol.end_px is not None and pol.end_px != p0
