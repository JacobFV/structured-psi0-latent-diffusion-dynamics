"""D-140 S3 interfaces: Env / EnvSpec / make_env, Policy / negotiate, TaskSpec registry, harness.rollout (tiny episodes)."""
import numpy as np
import pytest

from rrp.envs.base import ENVS, ActionSpace, BodyInfo, Env, EnvSpec, make_env
from rrp.policies.base import Act, PolicyInfo, Requirements, negotiate
from rrp.tasks.spec import TASKS, get_task

LATENT_ARM = PolicyInfo("latent", "learned", "v", Requirements(frozenset({"joint_position", "gripper"}),
                                                               observations=frozenset({"proprio", "object_descriptors", "task_graph"}),
                                                               body_families=frozenset({"arm"})))


def _cw_spec():
    return EnvSpec(env_id="computerworld", backend="computerworld", task="cw/click", control_hz=10.0,
                   bodies=[BodyInfo(robot=0, family="pointer", key="cw_pointer", robot_spec_hash="h")],
                   action_spaces=[ActionSpace(group="pointer", kind="cartesian_position", width=2, rate_hz=10.0),
                                  ActionSpace(group="button", kind="button", width=1, rate_hz=10.0),
                                  ActionSpace(group="key", kind="discrete", width=1, rate_hz=10.0, vocab=["Enter", "a"])],
                   capabilities=["privileged_truth", "snapshot", "images", "object_descriptors", "proprio", "language"],
                   frame={"units": "m", "up": "+y", "m_per_px": 0.001, "depth": "constant"})


def test_arm_env_spec_and_protocol():
    env = make_env("mujoco/arm", task="pick_place", body="parm6_pg2", seed=3)
    assert isinstance(env, Env)
    sp = env.spec
    assert sp.env_id == "mujoco/arm" and sp.task == "pick_place" and sp.batch == 1 and sp.control_hz == 20.0
    assert [b.family for b in sp.bodies] == ["arm"] and sp.bodies[0].robot_spec_hash == env.robots[0].spec.spec_hash
    assert {"joint_position", "gripper"} <= sp.action_kinds() and sp.has("privileged_truth") and sp.has("chunk_executor")
    assert sp.space("arm").width == len(env.robots[0].arm_joints)
    assert negotiate(LATENT_ARM, sp, get_task("pick_place")).ok


def test_dual_env_spec_has_one_space_set_per_robot():
    sp = make_env("mujoco/dual", task="support_insert", body=["parm5l_pg2", "parm6_pg2"], seed=3).spec
    assert sp.env_id == "mujoco/dual" and len(sp.bodies) == 2
    assert {a.robot for a in sp.action_spaces} == {0, 1} and sp.space("gripper", robot=1).kind == "gripper"


def test_negotiation_declines_with_reasons():
    c = negotiate(LATENT_ARM, _cw_spec())
    assert not c.ok
    assert any("needs gripper, joint_position; env offers button, cartesian_position, discrete" in r for r in c.reasons)
    assert any("task_graph" in r for r in c.reasons) and any("body family pointer" in r for r in c.reasons)
    pointer_bc = PolicyInfo("pointer_bc", "bc", "v", Requirements(frozenset({"cartesian_position", "button"}),
                                                                  observations=frozenset({"object_descriptors"})))
    assert negotiate(pointer_bc, _cw_spec()).ok
    legged = EnvSpec(env_id="mujoco/legged", backend="mujoco", task="waypoint_contact", control_hz=10.0,
                     bodies=[BodyInfo(robot=0, family="quadruped", key="go2", robot_spec_hash="h")],
                     action_spaces=[ActionSpace(group="base_velocity", kind="base_velocity", width=3, rate_hz=10.0)],
                     capabilities=["proprio", "task_graph", "object_descriptors"])
    r = negotiate(LATENT_ARM, legged, get_task("waypoint_contact")).reasons
    assert any("body family quadruped" in x for x in r) and any("does not exist in mujoco/legged" in x for x in
                                                                   negotiate(LATENT_ARM, legged, get_task("pick_place")).reasons)
    teacher = PolicyInfo("teacher:pick_place", "scripted_teacher", "t", Requirements(
        frozenset({"joint_position", "gripper"}), tasks=frozenset({"pick_place"}), privileged=True))
    assert any("privileged" in x for x in negotiate(teacher, legged).reasons)
    tracker = PolicyInfo("tracker", "learned", "t", Requirements(frozenset({"joint_position"}),
                                                                 observations=frozenset({"vector"}), batched_env=True))
    assert any("batched" in x for x in negotiate(tracker, legged).reasons)


def test_declared_but_missing_envs_fail_clearly():
    with pytest.raises(ValueError, match="g1_simple"):         # implemented: a wrong body fails before Isaac starts
        make_env("simple", task="x", body="y")
    with pytest.raises(KeyError, match="cw_pointer"):          # implemented: a wrong body fails before the wheel loads
        make_env("computerworld", task="x", body="y")
    with pytest.raises(KeyError):
        make_env("nope", task="x", body="y")


def test_every_task_lives_in_registered_envs():
    for t in TASKS.values():
        assert set(t.envs) <= set(ENVS), t.name


class _Hold:
    info = PolicyInfo("hold", "mock", "v0", Requirements(frozenset({"joint_position"})))

    def reset(self, spec, task, seeds, *, envs=None):
        self.n = len(seeds)
        assert len(envs) == len(seeds)

    def act(self, obs):
        return {i: Act(None) for i in obs}


class _Count:
    def __init__(self):
        self.steps = {}

    def on_step(self, i, env, act, step):
        self.steps[i] = self.steps.get(i, 0) + 1

    def on_end(self, i, env, ep):
        return {"hook_steps": self.steps[i]}


def test_rollout_tiny_hold_episodes():
    from rrp.harness.rollout import rollout
    hook = _Count()
    eps = rollout(lambda sd: make_env("mujoco/arm", task="pick_place", body="parm6_pg2", seed=sd), _Hold(),
                  get_task("pick_place"), [3, 4], batch=2, max_seconds=0.3, hooks=[hook])
    assert [e.seed for e in eps] == [3, 4]
    for e in eps:
        assert e.outcome == "timeout" and e.success_privileged is False and e.source == "mock"
        assert e.steps == 6 and e.metrics["hook_steps"] == 6 and abs(e.time - 0.3) < 1e-9
        assert e.env_id == "mujoco/arm" and e.provenance["policy_version"] == "v0"


def test_warp_env_adapter_with_fake_engine():
    from rrp.envs.base import BatchCommand
    from rrp.envs.warp.tracker_env import WarpEnv

    class Fake:
        N, nA, obs_dim, dt, body = 4, 3, 5, 0.02, "go2"
        b = type("B", (), {"kind": "quadruped"})()

        def reset_all(self):
            return np.zeros((4, 5))

        def observe(self):
            return np.ones((4, 5))

        def step(self, a):
            assert a.shape == (4, 3)
            return np.ones((4, 5)), np.full((4, 2), 7.0), np.zeros(4), np.zeros(4, bool), np.zeros(4, bool)

    env = WarpEnv(Fake())
    assert isinstance(env, Env) and env.spec.batch == 4 and env.spec.has("batched") and env.spec.space("legs").width == 3
    assert env.reset().vec.shape == (4, 5)
    st = env.step(BatchCommand(groups={"legs": np.zeros((4, 3))}, source="learned"))
    assert st.observation.layout == [("tracker_obs", 5)] and abs(st.time - 0.02) < 1e-12 and env.truth()[0, 0] == 7.0


def test_rollout_on_reset_infeasible_and_max_steps():
    from rrp.harness.hooks import Feasibility, SessionRecord
    from rrp.harness.eval.evaluate import summarize
    from rrp.harness.rollout import rollout
    feas = Feasibility(lambda env: env.seed != 4)
    eps = rollout(lambda sd: make_env("mujoco/arm", task="pick_place", body="parm5_pg2", seed=sd), _Hold(),
                  get_task("pick_place"), [3, 4], batch=2, max_steps=3, hooks=[feas, SessionRecord()])
    assert [(e.outcome, e.steps) for e in eps] == [("timeout", 3), ("infeasible", 0)]
    assert eps[1].failure_reason == "teacher_infeasible" and "events" in eps[1].metrics
    s = summarize(eps)
    assert s["attempted"] == 1 and s["infeasible"] == 1 and s["control_steps"] == 3 and s["successes"] == 0


def test_matrix_reports_every_cell(tmp_path):
    from rrp.harness.eval.evaluate import matrix
    rows = matrix(["teacher:pick_place", "teacher:waypoint_contact", "nope"], [("mujoco/arm", "parm5_pg2")],
                  ["pick_place", "waypoint_contact"], out=tmp_path / "m.jsonl")
    cell = {(r["policy"], r["task"]): r for r in rows}
    assert len(rows) == 6 and len((tmp_path / "m.jsonl").read_text().splitlines()) == 6
    assert cell[("teacher:pick_place", "pick_place")]["status"] == "accepted"
    assert cell[("teacher:waypoint_contact", "pick_place")]["reasons"][0].startswith("needs base_velocity")
    assert cell[("teacher:pick_place", "waypoint_contact")]["reasons"] == \
        ["task 'waypoint_contact' does not exist in mujoco/arm"]
    assert cell[("nope", "pick_place")]["reasons"][0].startswith("policy unavailable: KeyError")


def test_rrp_eval_cli(tmp_path):
    import json
    from rrp.cli.main import main
    out = tmp_path / "ev.jsonl"
    main(["eval", "--policy", "teacher:pick_place", "--env", "mujoco/arm", "--task", "pick_place", "--body",
          "parm5_pg2", "--seeds", "3,50", "--max-steps", "4", "--out", str(out)])
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert [(r["seed"], r["outcome"], r["source"]) for r in rows] == [(3, "timeout", "scripted_teacher"),
                                                                      (50, "infeasible", "scripted_teacher")]
    summ = json.loads(out.with_suffix(".summary.json").read_text())
    assert summ["attempted"] == 1 and summ["infeasible"] == 1


def test_legged_judge_rules():
    from types import SimpleNamespace as NS
    from rrp.tasks.spec import legged_judge
    j = legged_judge()

    def env(boundary=True, fell=False, pub=False, priv=False, ev=None):
        rt = NS(succeeded=lambda: pub, instances={k: NS(status=v) for k, v in (ev or {}).items()})
        return NS(boundary=boundary, fell=fell, runtime=rt, privileged_success=lambda: priv)
    assert not j(env(boundary=False, fell=True), 99.0, 60.0).done          # only on 10 Hz boundary ticks
    assert (j(env(fell=True), 1.0, 60.0).outcome, j(env(fell=True), 1.0, 60.0).failure_reason) == ("fell", "fell")
    assert j(env(pub=True, priv=True), 1.0, 60.0).outcome == "success"
    assert j(env(pub=True), 1.0, 60.0).failure_reason == "privileged_failure"
    assert not j(env(), 1.0, 60.0).done
    walk = lambda a, b: {"walk_to_a": a, "walk_to_b": b, "halt": "pending"}
    for ev, why in ((walk("active", "pending"), "drift_a"), (walk("succeeded", "active"), "drift_b"),
                    (walk("succeeded", "succeeded"), "halt"), (None, "timeout")):
        r = j(env(ev=ev), 60.0, 60.0)
        assert (r.outcome, r.failure_reason) == ("timeout", why)
    assert j(env(priv=True), 60.0, 60.0).outcome == "success"               # privileged success at the budget
